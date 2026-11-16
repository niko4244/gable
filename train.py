"""Bounded Unsloth/TRL Gemma 4 LoRA post-training entrypoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

from rcl import checked_attempts, contains_reserved_marker, derive_training_rows, jsonl

PIN_REVISION = "5fdfef5f3aa9dd3c680a22078f2aa97b4a31c919"
PIN_PATH = Path(__file__).with_name("source_pin.json")
PIN_MANIFEST_SHA256 = "12f7045e3735a0949063621d174bc3cfb0a3c526ba89a3dc47558c4b893b9fe8"
DPO_GEMMA4_UNIFIED_TEXT_ONLY_RUNTIME_VERIFIED = False
TRAINING_ROLE_MARKERS = (
    "<|", "|>", "<bos>", "<eos>", "<pad>",
    "<start_of_turn>", "<end_of_turn>",
)


def exact_sft_assistant_prefix(rendered: str) -> str:
    """Extract assistant header from the fully rendered native conversation.

    User-only Gemma generation prompts may add a thought channel that is absent
    when rendering a completed user and assistant conversation for SFT.
    """
    marker = "<|turn>model\n"
    if rendered.count(marker) != 1:
        raise RuntimeError("Expected exactly one native Gemma 4 assistant header")
    return rendered[:rendered.index(marker) + len(marker)].removeprefix("<bos>")


def sft_dataset_prefix_ids(tokenizer, rendered: str, input_ids: list[int]) -> list[int]:
    """Mirror TRL's text-field tokenizer including its automatic BOS policy.

    Do not compare add_special_tokens=False prefixes against a dataset encoded
    with add_special_tokens=True. If version-specific TRL tokenization changes,
    refuse the pilot rather than silently shifting source/response loss masks.
    """
    text = rendered.removeprefix("<bos>")
    observed = tokenizer(text, add_special_tokens=True)["input_ids"]
    if list(observed) != list(input_ids):
        raise RuntimeError("SFT dataset tokenizer encoding differs from native template proof")
    prefix = exact_sft_assistant_prefix(rendered)
    return list(tokenizer(prefix, add_special_tokens=True)["input_ids"])


def prove_sft_prompt_mask(input_ids: list[int], labels: list[int],
                          prompt_prefix_ids: list[int]) -> None:
    """Require an exact user-turn token prefix with zero trainable prompt labels."""
    if (
        not prompt_prefix_ids
        or len(prompt_prefix_ids) >= len(input_ids)
        or list(input_ids[:len(prompt_prefix_ids)]) != list(prompt_prefix_ids)
        or len(labels) != len(input_ids)
        or any(int(label) != -100 for label in labels[:len(prompt_prefix_ids)])
        or any(int(label) not in (-100, int(input_ids[index]))
               for index, label in enumerate(labels))
    ):
        raise RuntimeError("SFT user/source tokens or label IDs fail exact loss-mask invariants")


def prove_sft_completion_coverage(decoded_targets: str, completion: str) -> None:
    """Require that all original assistant source text, not just line one, survives."""
    if not completion.strip() or completion.strip() not in decoded_targets:
        raise RuntimeError("Complete assistant code is absent from unmasked SFT targets")


def prove_dpo_length_config(config, required_context: int) -> None:
    """Deny TRL's historical silent 512-token prompt truncation default."""
    if (
        getattr(config, "max_length", None) != required_context
        or getattr(config, "max_prompt_length", None) != required_context
    ):
        raise RuntimeError("DPO prompt may be truncated by TRL max_prompt_length")


def replay_expected_training_rows(kind: str, tasks: Path, attempts: Path
                                  ) -> tuple[list[dict], list[dict]]:
    """Independently reconstruct candidate labels for the selected task family."""
    if kind == "repository":
        from repo_rcl import checked_repo_attempts, derive_repo_training_rows, load_repo_tasks
        source = checked_repo_attempts(tasks, attempts, reverify=True)
        task_map = {row["id"]: row for row in load_repo_tasks(tasks)}
        sft, dpo, _ = derive_repo_training_rows(source, task_map)
    elif kind == "function":
        source = checked_attempts(tasks, attempts, reverify=True)
        sft, dpo, _ = derive_training_rows(source)
    else:
        raise ValueError(f"Unsupported training task family: {kind}")
    return sft, dpo


def prove_nonsynthetic_training_source(manifest: dict, attempts: Path) -> None:
    """Never admit public/synthetic smoke examples as model-generated training data.

    This validates recorded provenance, not cryptographic attestation of the
    generation process. Real training also requires independent reward replay.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Training manifest must be a JSON object")
    digest = manifest.get("model_digest")
    if (
        manifest.get("collection_origin") != "ollama_local"
        or manifest.get("synthetic_fixture_only") is not False
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or digest == "0" * 64
    ):
        raise ValueError("Synthetic or unverified model collection cannot be used for real training")
    rows = jsonl(attempts)
    if not rows:
        raise ValueError("Training source attempts are empty")
    for row in rows:
        if row.get("collection_origin") != "ollama_local" or row.get("model_digest") != digest:
            raise ValueError("Training source attempts contradict model collection manifest")
        if (not isinstance(row.get("model_prompt_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["model_prompt_sha256"])):
            raise ValueError("Training attempts are missing frozen rendered prompt evidence")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_pin() -> dict:
    if sha256(PIN_PATH) != PIN_MANIFEST_SHA256:
        raise ValueError("Frozen upstream source manifest SHA256 mismatch")
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    if pin.get("revision") != PIN_REVISION or pin.get("repo") != "zaakirio/gemma-4-12b-it-uncensored":
        raise ValueError("Abliterated Gemma source revision does not match the frozen project pin")
    expected_files = {
        "chat_template.jinja", "config.json", "model.safetensors.index.json",
        "processor_config.json", "tokenizer.json", "tokenizer_config.json",
        *(f"model-{number:05d}-of-00005.safetensors" for number in range(1, 6)),
    }
    if not isinstance(pin.get("files"), dict) or set(pin["files"]) != expected_files:
        raise ValueError("Pinned Gemma model files are incomplete")
    return pin


def verify_source_identity(base: Path, *, full_hash: bool) -> None:
    """Match pinned upstream release bytes, including all five BF16 shards."""
    for filename, meta in source_pin()["files"].items():
        target = base / filename
        if not target.is_file() or target.stat().st_size != meta["size"]:
            raise ValueError(f"Missing or size-mismatched pinned source file: {filename}")
        if full_hash:
            if "sha256" in meta:
                if sha256(target) != meta["sha256"]:
                    raise ValueError(f"Source SHA256 mismatch: {filename}")
            elif "git_blob_sha1" in meta:
                sha = hashlib.sha1()
                sha.update(f"blob {target.stat().st_size}\0".encode("ascii"))
                with target.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        sha.update(block)
                if sha.hexdigest() != meta["git_blob_sha1"]:
                    raise ValueError(f"Git blob identity mismatch: {filename}")


def validate_safetensors(path: Path) -> None:
    """Reject empty, truncated and fabricated shards before a 12B weight allocation."""
    size = path.stat().st_size
    if size < 1024**3:
        raise ValueError(f"Implausibly small Gemma 12B weight shard: {path.name}")
    with path.open("rb") as stream:
        header_size_bytes = stream.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"Truncated safetensors header: {path.name}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if not 16 <= header_size <= min(size - 8, 100 * 1024 * 1024):
            raise ValueError(f"Invalid safetensors header size: {path.name}")
        tensors = json.loads(stream.read(header_size))
    if not isinstance(tensors, dict) or len(tensors) < 1:
        raise ValueError(f"Missing safetensors tensor metadata: {path.name}")
    names = [k for k in tensors if k != "__metadata__"]
    if not names:
        raise ValueError(f"No tensors in safetensors header: {path.name}")
    max_payload = size - header_size - 8
    for name in names:
        meta = tensors[name]
        if not isinstance(meta, dict) or not isinstance(meta.get("dtype"), str):
            raise ValueError(f"Invalid tensor metadata in {path.name}")
        offsets = meta.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= max_payload):
            raise ValueError(f"Invalid safetensors offsets in {path.name}")


def reference_adapter_parameters(model) -> list:
    """Identify reference LoRA parameters by PEFT's named adapter path."""
    params = [param for name, param in model.named_parameters()
              if ".reference." in name and ("lora_" in name or "modules_to_save" in name)]
    if not params:
        raise RuntimeError("DPO frozen reference adapter parameters were not found")
    return params


def prove_reference_excluded_from_optimizer(model, trainer) -> list:
    """Fail closed if the reference adapter can receive a gradient optimizer update."""
    refs = reference_adapter_parameters(model)
    for parameter in refs:
        parameter.requires_grad_(False)
    trainer.create_optimizer()
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        raise RuntimeError("DPO optimizer could not be inspected")
    reference_ids = {id(parameter) for parameter in refs}
    for group in optimizer.param_groups:
        if any(id(parameter) in reference_ids for parameter in group["params"]):
            raise RuntimeError("Frozen DPO reference adapter entered the optimizer")
    return refs


def preflight(args: argparse.Namespace) -> dict:
    base = args.base.resolve()
    output = args.output.resolve()
    dataset = args.data.resolve()
    task_kind = getattr(args, "task_kind", "function")
    reasons: list[str] = []
    if args.phase == "dpo" and not DPO_GEMMA4_UNIFIED_TEXT_ONLY_RUNTIME_VERIFIED:
        reasons.append(
            "DPO blocked: stock TRL routes gemma4_unified through vision process_row "
            "requiring images/pixel_values; a text-only Gemma4Unified DPO path has "
            "not been independently validated"
        )
    if task_kind not in ("function", "repository"):
        reasons.append("Unknown training task family")
    if args.min_vram_mib < 13000 or args.min_disk_gib < 20:
        reasons.append("Training resource floors cannot be reduced below 13000 MiB VRAM and 20 GiB disk")
    config: dict = {}
    if not base.is_dir() or not (base / "config.json").is_file():
        reasons.append("No local Hugging Face model/config.json; GGUF cannot be used for LoRA training")
        model_type = None
    else:
        try:
            config = json.loads((base / "config.json").read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("config root must be an object")
            model_type = config.get("model_type")
        except (ValueError, OSError):
            config = {}
            model_type = None
        if model_type != "gemma4_unified":
            reasons.append(f"Expected Gemma 4 Unified 12B model_type=gemma4_unified, got {model_type!r}")
        index_path = base / "model.safetensors.index.json"
        if not index_path.is_file():
            reasons.append("Full Gemma 4 12B safetensors shard index is missing")
        else:
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map")
                if not isinstance(weight_map, dict):
                    raise ValueError("Shard weight map is not a dictionary")
                names = set(weight_map.values())
                if len(names) != 5 or any(
                    not isinstance(name, str)
                    or Path(name).name != name
                    or not name.endswith(".safetensors")
                    or not (base / name).is_file()
                    for name in names
                ):
                    reasons.append("HF model weight shards are incomplete or unsafe")
                else:
                    for name in sorted(names):
                        validate_safetensors(base / name)
            except (ValueError, KeyError, TypeError, OSError):
                reasons.append("HF model shard index is invalid")
        try:
            verify_source_identity(base, full_hash=False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reasons.append(f"Frozen abliterated checkpoint identity mismatch: {exc}")
        for name in ("tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja"):
            if not (base / name).is_file():
                reasons.append(f"Missing native Gemma 4 tokenizer/processor file: {name}")
        text_config = config.get("text_config", {})
        if not isinstance(text_config, dict):
            text_config = {}
        if (
            config.get("architectures") != ["Gemma4UnifiedForConditionalGeneration"]
            or text_config.get("model_type") != "gemma4_unified_text"
            or text_config.get("num_hidden_layers") != 48
            or text_config.get("hidden_size") != 3840
            or text_config.get("vocab_size") != 262144
        ):
            reasons.append("Model config is not the verified Gemma 4 Unified 12B architecture")
    if args.phase == "dpo" and (args.sft_adapter is None or not (args.sft_adapter / "adapter_config.json").is_file()):
        reasons.append("DPO requires a completed, locally saved SFT LoRA adapter")
    elif args.phase == "dpo":
        try:
            adapter = args.sft_adapter
            receipt = json.loads((adapter / "rcl_train_receipt.json").read_text(encoding="utf-8"))
            adapter_weights = adapter / "adapter_model.safetensors"
            if (
                receipt.get("status") != "TRAINED_LOCAL_UNEVALUATED"
                or receipt.get("phase") != "sft"
                or receipt.get("base_config_sha256") != sha256(base / "config.json")
                or receipt.get("base_template_sha256") != sha256(base / "chat_template.jinja")
                or receipt.get("adapter_config_sha256") != sha256(adapter / "adapter_config.json")
                or receipt.get("source_revision") != PIN_REVISION
                or receipt.get("task_kind", "function") != task_kind
                or receipt.get("source_tasks_sha256") != sha256(args.tasks)
                or not adapter_weights.is_file()
                or receipt.get("adapter_weights_sha256") != sha256(adapter_weights)
            ):
                reasons.append("DPO SFT adapter does not match the original checkpoint and its training receipt")
        except (OSError, ValueError, TypeError):
            reasons.append("DPO SFT adapter missing or invalid provenance/weight receipt")
    task_path = getattr(args, "tasks", None)
    attempts_path = getattr(args, "attempts", None)
    if task_path is None or attempts_path is None:
        reasons.append("Source task and attempt JSONL paths must be supplied for independent reward replay")
    if not dataset.is_file():
        reasons.append(f"Dataset missing: {dataset}")
        count = 0
    else:
        try:
            rows = jsonl(dataset)
            names = ("prompt", "completion") if args.phase == "sft" else ("prompt", "chosen", "rejected")
            for row in rows:
                if any(not isinstance(row.get(name), str) or not row[name].strip() for name in names):
                    raise ValueError(f"Malformed {args.phase.upper()} data")
                if any(contains_reserved_marker(row[name]) for name in names):
                    raise ValueError("Untrusted model role delimiter in training text")
            if len(rows) < 1:
                raise ValueError("No verified training examples")
            count = len(rows)
        except ValueError as exc:
            reasons.append(str(exc))
            count = 0
        manifest_path = dataset.parent / "manifest.json"
        if not manifest_path.is_file():
            reasons.append("Training data has no RCL reward provenance manifest")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_key = "sft_sha256" if args.phase == "sft" else "dpo_sha256"
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("schema") != (
                        "gemma-coding-repo-rcl-v1" if task_kind == "repository"
                        else "gemma-coding-rcl-v1"
                    )
                    or manifest.get(expected_key) != sha256(dataset)
                    or (task_path is not None and manifest.get("source_tasks_sha256") != sha256(task_path))
                    or (attempts_path is not None and manifest.get("source_attempts_sha256") != sha256(attempts_path))
                ):
                    reasons.append("Training dataset digest does not match RCL export provenance")
                if attempts_path is None:
                    reasons.append("Training collection identity cannot be verified without attempts")
                else:
                    try:
                        prove_nonsynthetic_training_source(manifest, attempts_path)
                    except (ValueError, OSError) as exc:
                        reasons.append(f"Training collection provenance blocked: {exc}")
            except (OSError, ValueError):
                reasons.append("Training dataset provenance manifest is malformed")
    if output.exists():
        reasons.append(f"Output already exists: {output}")
    else:
        parent = next((p for p in [output.parent, *output.parents] if p.is_dir()), None)
        free_disk = shutil.disk_usage(parent).free if parent else 0
        if free_disk < args.min_disk_gib * (1024**3):
            reasons.append(f"Output drive has {free_disk / 1024**3:.1f} GiB free, need {args.min_disk_gib} GiB")
    free_vram = 0
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True, timeout=10)
        free_vram = max(int(value.strip()) for value in query.stdout.splitlines() if value.strip())
        if free_vram < args.min_vram_mib:
            reasons.append(f"GPU has {free_vram} MiB free, need {args.min_vram_mib} MiB")
    except (OSError, ValueError, subprocess.SubprocessError):
        reasons.append("Cannot verify NVIDIA GPU free VRAM")
    if not reasons:
        try:
            verify_source_identity(base, full_hash=True)
            sft, dpo = replay_expected_training_rows(task_kind, task_path, attempts_path)
            expected = sft if args.phase == "sft" else dpo
            if jsonl(dataset) != expected:
                reasons.append("Training examples do not match independently replayed rewards")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            reasons.append(f"Reward replay blocked: {exc}")
    return {"status": "READY_FOR_PILOT" if not reasons else "BLOCKED",
            "phase": args.phase, "task_kind": task_kind,
            "base": str(base), "model_type": model_type,
            "examples": count, "free_gpu_mib": free_vram,
            "min_gpu_mib": args.min_vram_mib, "min_disk_gib": args.min_disk_gib,
            "reasons": reasons}


def train(args: argparse.Namespace, audit: dict) -> None:
    if args.phase == "dpo" and not DPO_GEMMA4_UNIFIED_TEXT_ONLY_RUNTIME_VERIFIED:
        raise RuntimeError(
            "Gemma4Unified text-only DPO compatibility has not passed a real runtime qualification"
        )
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    import torch
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if args.phase == "dpo":
        required = {"model_adapter_name", "ref_adapter_name", "max_prompt_length"}
        if not required.issubset(inspect.signature(DPOConfig).parameters):
            raise RuntimeError(
                "Installed TRL DPOConfig lacks named policy/reference adapters; "
                "use a tested compatible TRL release"
            )
    model, tokenizer = FastModel.from_pretrained(
        model_name=str(args.base.resolve()), max_seq_length=args.max_seq_length,
        load_in_4bit=True, full_finetuning=False,
    )
    native_template = (args.base / "chat_template.jinja").read_text(encoding="utf-8")
    if tokenizer.chat_template != native_template:
        raise RuntimeError("The loaded tokenizer template differs from the exact Gemma source; cannot train safely")
    if args.phase == "sft":
        model = FastModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=8, lora_alpha=8, lora_dropout=0, bias="none", random_state=3407,
        )
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, str(args.sft_adapter.resolve()), is_trainable=True, adapter_name="train"
        )
        model.load_adapter(str(args.sft_adapter.resolve()), adapter_name="reference", is_trainable=False)
        if "reference" not in model.peft_config or "train" not in model.peft_config:
            raise RuntimeError("A distinct frozen SFT reference policy is required for DPO")
    rows = jsonl(args.data)
    output = str(args.output.resolve())
    if args.phase == "sft":
        def format_row(row: dict) -> dict:
            conversation = [{"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["completion"]}]
            rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            return {"text": rendered.removeprefix("<bos>")}
        dataset = Dataset.from_list([format_row(r) for r in rows])
        config = SFTConfig(
            output_dir=output, max_length=args.max_seq_length, dataset_text_field="text",
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            max_steps=args.max_steps, learning_rate=2e-5, bf16=True,
            optim="adamw_8bit", gradient_checkpointing=True,
            warmup_steps=2, logging_steps=1, save_strategy="no",
            report_to="none", seed=3407, packing=False,
        )
        trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=config)
        trainer = train_on_responses_only(
            trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n",
        )
        # Insist some target tokens remain after response masking.
        for original, row in zip(rows, trainer.train_dataset, strict=True):
            labels = row.get("labels", [])
            full_chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": original["prompt"]},
                 {"role": "assistant", "content": original["completion"]}],
                tokenize=False, add_generation_prompt=False,
            )
            prefix_ids = sft_dataset_prefix_ids(
                tokenizer, full_chat, row.get("input_ids", []),
            )
            prove_sft_prompt_mask(row.get("input_ids", []), labels, prefix_ids)
            if not labels or not any(token != -100 for token in labels):
                raise RuntimeError("An SFT sample has no target assistant tokens after masking/truncation")
            target = [int(token) for token in labels if token != -100]
            if len(target) < 10 or len(row.get("input_ids", [])) >= args.max_seq_length:
                raise RuntimeError("Assistant training signal too short or truncated")
            decoded = tokenizer.decode(target, skip_special_tokens=False)
            prove_sft_completion_coverage(decoded, original["completion"])
    else:
        for row in rows:
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
            if len(prompt_ids) >= args.max_seq_length:
                raise RuntimeError("DPO user/source prompt is too long for full context")
            for candidate in ("chosen", "rejected"):
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]},
                     {"role": "assistant", "content": row[candidate]}],
                    tokenize=False, add_generation_prompt=False,
                )
                ids = tokenizer.encode(rendered, add_special_tokens=False)
                if len(ids) >= args.max_seq_length:
                    raise RuntimeError(f"DPO {candidate} answer is truncated at max_seq_length")
        dataset = Dataset.from_list([
            {"prompt": [{"role": "user", "content": r["prompt"]}],
             "chosen": [{"role": "assistant", "content": r["chosen"]}],
             "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows
        ])
        config = DPOConfig(
            output_dir=output, max_length=args.max_seq_length,
            max_prompt_length=args.max_seq_length,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            max_steps=args.max_steps, learning_rate=5e-6, bf16=True,
            optim="adamw_8bit", gradient_checkpointing=True,
            beta=0.1, report_to="none", logging_steps=1,
            save_strategy="no", seed=3407,
            model_adapter_name="train", ref_adapter_name="reference",
        )
        prove_dpo_length_config(config, args.max_seq_length)
        trainer = DPOTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=config)
        frozen_refs = prove_reference_excluded_from_optimizer(model, trainer)
    trainer.train()
    if args.phase == "dpo":
        if any(parameter.requires_grad for parameter in frozen_refs):
            raise RuntimeError("DPO reference adapter was unfrozen during training")
        model.set_adapter("train")
        model.save_pretrained(output, selected_adapters=["train"])
        named_adapter = args.output / "train"
        if not (named_adapter / "adapter_config.json").is_file():
            raise RuntimeError("The trained DPO adapter was not serialized in the expected directory")
        for source in named_adapter.iterdir():
            destination = args.output / source.name
            if destination.exists():
                raise RuntimeError(f"Refusing to overwrite existing training artifact {destination}")
            source.rename(destination)
        named_adapter.rmdir()
    else:
        trainer.save_model(output)
    tokenizer.save_pretrained(output)
    if not (args.output / "adapter_config.json").is_file():
        raise RuntimeError("Final LoRA adapter serialization did not yield adapter_config.json")
    adapter_weights = args.output / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise RuntimeError("Final LoRA adapter_model.safetensors was not saved")
    receipt = {
        **audit, "status": "TRAINED_LOCAL_UNEVALUATED",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256(args.data),
        "source_tasks_sha256": sha256(args.tasks),
        "source_attempts_sha256": sha256(args.attempts),
        "base_config_sha256": sha256(args.base / "config.json"),
        "base_template_sha256": sha256(args.base / "chat_template.jinja"),
        "adapter_config_sha256": sha256(args.output / "adapter_config.json"),
        "adapter_weights_sha256": sha256(adapter_weights),
        "source_revision": PIN_REVISION,
        "task_kind": getattr(args, "task_kind", "function"),
        "torch_version": torch.__version__,
        "max_steps": args.max_steps, "max_seq_length": args.max_seq_length,
        "note": "No coding-quality claim until held-out original-vs-adapter evaluation",
    }
    (args.output / "rcl_train_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["sft", "dpo"], required=True)
    parser.add_argument("--task-kind", choices=["function", "repository"], default="function",
                        help="Select independently replayed code-function or repository-patch rewards")
    parser.add_argument("--base", type=Path, required=True, help="Exact local HF safetensors abliterated checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="RCL verified sft.jsonl or dpo.jsonl")
    parser.add_argument("--tasks", type=Path, required=True, help="Immutable source task JSONL for reward replay")
    parser.add_argument("--attempts", type=Path, required=True, help="Original generated attempts for reward replay")
    parser.add_argument("--output", type=Path, required=True, help="New LoRA adapter directory")
    parser.add_argument("--sft-adapter", type=Path, help="For DPO, the completed SFT adapter")
    parser.add_argument("--min-vram-mib", type=int, default=13000)
    parser.add_argument("--min-disk-gib", type=int, default=20)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="Actually allocate GPU/train after passing preflight")
    args = parser.parse_args(argv)
    if args.max_seq_length < 128 or args.max_seq_length > 4096 or args.max_steps < 1 or args.max_steps > 10000:
        parser.error("max-seq-length must be 128..4096 and max-steps 1..10000")
    try:
        report = preflight(args)
        print(json.dumps(report, indent=2))
        if report["status"] != "READY_FOR_PILOT":
            return 2
        if args.execute:
            train(args, report)
        else:
            print("DRY RUN: preflight passed; supply --execute for the bounded training pilot")
        return 0
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.SubprocessError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
