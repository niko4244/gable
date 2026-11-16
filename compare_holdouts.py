"""Paired independent coding holdout comparison with score replay.

This module compares two frozen candidate attempt files over exactly the same
evaluation tasks, attempt indices, decoding budgets, and seed schedule. It
never invents pass rates from incomplete or timed-out evaluations.

Statistical inferences apply to the inspected tasks only; a paired result on
small local fixtures cannot establish frontier coding-model parity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import struct
import sys

import rcl


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_NAME = re.compile(r"[a-zA-Z0-9_.-]+\Z")
HF_SOURCE_PIN = Path(__file__).with_name("source_pin.json")
HF_SOURCE_PIN_SHA256 = "12f7045e3735a0949063621d174bc3cfb0a3c526ba89a3dc47558c4b893b9fe8"


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Missing or malformed {name}: require lower-case SHA256")
    return value


def _json_native(value: object) -> bool:
    """Refuse ambiguous Python-only values in recorded JSON configuration."""
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_native(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_native(item) for key, item in value.items())
    return False


def _validate_adapter_safetensors(path: Path) -> None:
    """Check the bounded metadata framing without loading tensor payloads."""
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Truncated HF adapter safetensors header")
        header_length = struct.unpack("<Q", length_bytes)[0]
        payload_length = path.stat().st_size - 8 - header_length
        if not 2 <= header_length <= 16 * 1024 * 1024 or payload_length < 1:
            raise ValueError("Invalid HF adapter safetensors framing")
        try:
            header = json.loads(stream.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed HF adapter safetensors metadata") from exc
    if type(header) is not dict:
        raise ValueError("Malformed HF adapter safetensors tensor table")
    entries = [value for key, value in header.items() if key != "__metadata__"]
    if not entries:
        raise ValueError("HF adapter safetensors contains no tensors")
    spans = []
    for item in entries:
        if (type(item) is not dict or type(item.get("dtype")) is not str
                or type(item.get("shape")) is not list
                or any(type(dim) is not int or dim < 0 for dim in item["shape"])):
            raise ValueError("Malformed HF adapter safetensors tensor descriptor")
        offsets = item.get("data_offsets")
        if (type(offsets) is not list or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload_length):
            raise ValueError("Malformed HF adapter safetensors offsets")
        spans.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("Overlapping or incomplete HF adapter tensor payload")
        cursor = end
    if cursor != payload_length:
        raise ValueError("Unaccounted HF adapter tensor payload")


def _snapshot_files(snapshot: dict, *, adapter: bool, cache: dict) -> str:
    """Hash all declared local HF files, and prove the weight set is complete."""
    required_keys = {"directory", "files"} if adapter else {
        "directory", "repo", "revision", "files",
    }
    if type(snapshot) is not dict or set(snapshot) != required_keys:
        raise ValueError("Incomplete HF checkpoint/adapter identity evidence")
    if (type(snapshot["directory"]) is not str or not snapshot["directory"]
            or not Path(snapshot["directory"]).is_absolute()):
        raise ValueError("HF identity requires an absolute local snapshot directory")
    directory = Path(snapshot["directory"])
    if not directory.is_dir():
        raise ValueError("HF identity snapshot directory is absent")
    if not adapter and (
        type(snapshot["repo"]) is not str
        or re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", snapshot["repo"]) is None
        or type(snapshot["revision"]) is not str
        or _REVISION.fullmatch(snapshot["revision"]) is None
    ):
        raise ValueError("HF base requires exact repository and immutable revision")
    if not adapter:
        if sha256(HF_SOURCE_PIN) != HF_SOURCE_PIN_SHA256:
            raise ValueError("Trusted HF source pin has changed")
        pin = json.loads(HF_SOURCE_PIN.read_text(encoding="utf-8"))
        if (snapshot["repo"], snapshot["revision"]) != (pin["repo"], pin["revision"]):
            raise ValueError("HF base is not the independently pinned source revision")
    files = snapshot["files"]
    if type(files) is not dict or not files:
        raise ValueError("HF file digest manifest must be nonempty")
    for name, meta in files.items():
        if (type(name) is not str or name in (".", "..")
                or _SAFE_NAME.fullmatch(name) is None
                or type(meta) is not dict or set(meta) != {"size", "sha256"}
                or type(meta["size"]) is not int or meta["size"] < 1):
            raise ValueError("Malformed HF file identity manifest")
        _digest(meta["sha256"], f"HF file {name} digest")
    names = set(files)
    if not adapter:
        pinned_files = pin.get("files")
        if type(pinned_files) is not dict or names != set(pinned_files):
            raise ValueError("Incomplete or expanded independently pinned HF base manifest")
        for name, source in pinned_files.items():
            if (files[name]["size"] != source.get("size") or
                    ("sha256" in source and files[name]["sha256"] != source["sha256"])):
                raise ValueError(f"HF base differs from trusted source pin: {name}")
    config_file = "adapter_config.json" if adapter else "config.json"
    if adapter:
        required = {"adapter_config.json"}
        weight_files = {name for name in names if name.endswith(".safetensors")}
        if not weight_files or not all(name.startswith("adapter_model") for name in weight_files):
            raise ValueError("Incomplete HF adapter weights")
        index_name = "adapter_model.safetensors.index.json"
    else:
        required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
        weight_files = {name for name in names if name.endswith(".safetensors")}
        if not weight_files or not all(name.startswith("model") for name in weight_files):
            raise ValueError("Incomplete HF base weights")
        index_name = "model.safetensors.index.json"
    if not required <= names:
        raise ValueError("HF snapshot lacks required config/tokenizer/adapter files")
    if len(weight_files) > 1 and index_name not in names:
        raise ValueError("Sharded HF weights require the complete index")
    if index_name in names:
        required.add(index_name)
    # Account for every locally present weight and prompt/processor file, even
    # when the claimed manifest attempts to omit a file that changes behavior.
    influential = {
        "chat_template.jinja", "processor_config.json", "preprocessor_config.json",
        "generation_config.json", "special_tokens_map.json", "tokenizer.model",
        "added_tokens.json", "adapter_config.json", "model.safetensors.index.json",
        "adapter_model.safetensors.index.json",
    }
    actual = {p.name for p in directory.iterdir() if p.is_file() and (
        p.name.endswith(".safetensors") or p.name in influential)}
    if not actual <= names:
        raise ValueError("HF manifest omits local weights or generation-affecting files")
    evidence_key = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    if evidence_key not in cache:
        for name in sorted(names):
            target = directory / name
            if not target.is_file():
                raise ValueError(f"Missing HF identity file: {name}")
            stat_before = target.stat()
            if stat_before.st_size != files[name]["size"]:
                raise ValueError(f"HF identity size mismatch: {name}")
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            stat_after = target.stat()
            if (digest.hexdigest() != files[name]["sha256"]
                    or (stat_before.st_size, stat_before.st_mtime_ns)
                    != (stat_after.st_size, stat_after.st_mtime_ns)):
                raise ValueError(f"HF identity digest or file-stability mismatch: {name}")
        config = json.loads((directory / config_file).read_text(encoding="utf-8"))
        if type(config) is not dict:
            raise ValueError("Invalid HF configuration")
        if adapter:
            if type(config.get("peft_type")) is not str or not config["peft_type"]:
                raise ValueError("HF adapter configuration lacks PEFT type")
        elif config.get("model_type") == "gemma4_unified" and not {
            "chat_template.jinja", "processor_config.json",
        } <= names:
            raise ValueError("Gemma 4 Unified HF snapshot lacks processor or chat template")
        if index_name in names:
            index = json.loads((directory / index_name).read_text(encoding="utf-8"))
            mapping = index.get("weight_map") if type(index) is dict else None
            if (type(mapping) is not dict or not mapping or
                    any(type(k) is not str or type(v) is not str for k, v in mapping.items())
                    or set(mapping.values()) != weight_files):
                raise ValueError("HF shard index does not account for every weight shard")
        elif weight_files != {"adapter_model.safetensors" if adapter else "model.safetensors"}:
            raise ValueError("Unindexed HF snapshot must use a single named safetensors file")
        if adapter:
            for name in sorted(weight_files):
                _validate_adapter_safetensors(directory / name)
        if not adapter:
            for name, source in pinned_files.items():
                if "git_blob_sha1" in source:
                    digest = hashlib.sha1()
                    digest.update(f"blob {files[name]['size']}\0".encode("ascii"))
                    with (directory / name).open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != source["git_blob_sha1"]:
                        raise ValueError(f"HF base Git blob differs from trusted pin: {name}")
        cache[evidence_key] = hashlib.sha256(json.dumps({
            k: v for k, v in snapshot.items() if k != "directory"
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return cache[evidence_key]


def require_model_identity(row: dict, *, cache: dict) -> tuple[str, ...]:
    """Require content identity, not an editable model tag or self-verified flag."""
    if "model_digest" in row and "model_identity" in row:
        raise ValueError("Ambiguous competing model identity evidence")
    if "model_digest" in row:
        return ("ollama", _digest(row["model_digest"], "Ollama model digest"))
    identity = row.get("model_identity")
    if type(identity) is not dict or set(identity) != {"kind", "base", "adapter"}:
        raise ValueError("Missing Ollama digest or complete HF model identity evidence")
    if identity["kind"] != "hf":
        raise ValueError("Unsupported or ambiguous HF identity kind")
    base = identity["base"]
    base_digest = _snapshot_files(base, adapter=False, cache=cache)
    adapter = identity["adapter"]
    if adapter is None:
        return ("hf", base_digest, "no-adapter")
    if type(adapter) is not dict or set(adapter) != {
        "directory", "files", "base_repo", "base_revision",
    } or (adapter["base_repo"], adapter["base_revision"]) != (
        base["repo"], base["revision"]
    ):
        raise ValueError("HF adapter is not bound to exact base revision")
    adapter_digest = _snapshot_files({
        "directory": adapter["directory"], "files": adapter["files"],
    }, adapter=True, cache=cache)
    adapter_config = json.loads((Path(adapter["directory"]) / "adapter_config.json")
                                .read_text(encoding="utf-8"))
    referenced_base = adapter_config.get("base_model_name_or_path")
    if referenced_base is not None and referenced_base not in (
        base["repo"], base["directory"], str(Path(base["directory"]).resolve()),
    ):
        raise ValueError("HF adapter configuration targets a different base model")
    referenced_revision = adapter_config.get("revision")
    if referenced_revision is not None and referenced_revision != base["revision"]:
        raise ValueError("HF adapter configuration targets a different base revision")
    weights = {name: meta for name, meta in adapter["files"].items()
               if name.endswith(".safetensors")}
    weights_digest = hashlib.sha256(json.dumps(
        sorted(meta["sha256"] for meta in weights.values()), separators=(",", ":"),
    ).encode()).hexdigest()
    return ("hf", base_digest, adapter_digest, weights_digest)


def require_prompt_metadata(row: dict, task: dict, *, repository: bool) -> tuple:
    prompt = task.get("prompt")
    if type(prompt) is not str:
        raise ValueError("Missing frozen evaluation prompt")
    expected = rcl.digest(rcl.canonical_prompt(prompt))
    if _digest(row.get("prompt_sha256"), "task prompt digest") != expected:
        raise ValueError("Evaluation prompt digest changed")
    if repository:
        import repo_rcl
        rendered = repo_rcl.repo_prompt(task)
    else:
        rendered = rcl.training_prompt(prompt)
    if _digest(row.get("model_prompt_sha256"), "model-rendered prompt digest") != rcl.digest(rendered):
        raise ValueError("Rendered model prompt or prompt configuration changed")
    if "prompt_config" in row:
        prompt_cfg = row["prompt_config"]
        if type(prompt_cfg) is not dict or not prompt_cfg or not _json_native(prompt_cfg):
            raise ValueError("Invalid optional prompt configuration")
        config_text = json.dumps(prompt_cfg, sort_keys=True, allow_nan=False)
    else:
        config_text = None
    return (expected, rcl.digest(rendered), config_text)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(wins: int, total: int) -> list[float]:
    if total < 1 or not (0 <= wins <= total):
        raise ValueError("Invalid Wilson interval counts")
    z = 1.959963984540054
    fraction = wins / total
    divisor = 1 + (z * z / total)
    center = (fraction + z * z / (2 * total)) / divisor
    margin = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total))
    return [max(0.0, center - margin), min(1.0, center + margin)]


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    """Two-sided conditional exact binomial significance on discordant cases."""
    if improved < 0 or regressed < 0:
        raise ValueError("Discordant counts must be nonnegative")
    n = improved + regressed
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(min(improved, regressed) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def bootstrap_delta_ci(pairs: list[tuple[int, int]], *, samples: int = 10_000,
                       seed: int = 3407) -> list[float]:
    """Paired percentile bootstrap across tasks (not across candidate samples)."""
    if not pairs or not 100 <= samples <= 100_000:
        raise ValueError("Need task pairs and 100..100000 bootstrap replicates")
    n = len(pairs)
    rng = random.Random(seed)
    deltas = sorted(sum(pairs[rng.randrange(n)][1] - pairs[rng.randrange(n)][0]
                        for _ in range(n)) / n for _ in range(samples))
    lower = deltas[int((samples - 1) * 0.025)]
    upper = deltas[int((samples - 1) * 0.975)]
    return [lower, upper]


def require_generation_metadata(record: dict) -> tuple:
    cfg = record.get("generation_config")
    if not isinstance(cfg, dict) or set(cfg) != {
        "temperature", "top_p", "num_predict", "seed",
    }:
        raise ValueError("Missing or unexpected generation configuration; collection must record budgets")
    temperature, top_p, max_tokens, seed = (
        cfg["temperature"], cfg["top_p"], cfg["num_predict"], cfg["seed"]
    )
    if (type(temperature) not in (int, float) or not 0 <= temperature <= 2
            or type(top_p) not in (int, float) or not 0 < top_p <= 1
            or type(max_tokens) is not int or max_tokens < 1
            or type(seed) is not int or seed < 0):
        raise ValueError("Invalid deterministic generation budget or seed")
    return (float(temperature), float(top_p), max_tokens, seed)


def measured_generation_timings(rows: list[dict]) -> dict | None:
    """Report collection wall time when every attempt records it; never impute."""
    if not rows or any("generation_seconds" not in row for row in rows):
        return None
    values = [row["generation_seconds"] for row in rows]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid observed generation timing")
    return {
        "median_generation_seconds": round(statistics.median(values), 4),
        "total_generation_seconds": round(sum(values), 4),
        "attempts_observed": len(values),
        "caution": "Wall times include local queue/loading effects, not normalized hardware cost.",
    }


def checked_heldout_rows(task_path: Path, attempt_path: Path, task_kind: str,
                         *, image: str = "python:3.11-slim") -> tuple[list[dict], list[dict]]:
    """Replay ALL rewards and refuse data from a non-evaluation split."""
    if task_kind == "function":
        tasks = rcl.load_tasks(task_path)
        rows = rcl.checked_attempts(task_path, attempt_path, reverify=True, image=image)
    elif task_kind == "repository":
        import repo_rcl
        tasks = repo_rcl.load_repo_tasks(task_path)
        rows = repo_rcl.checked_repo_attempts(task_path, attempt_path,
                                              reverify=True, image=image)
    else:
        raise ValueError("Unknown task kind")
    if any(row["split"] != "eval" for row in rows):
        raise ValueError("Holdout attempt files may contain evaluation-split records only")
    return tasks, rows


def _summarize_rows(tasks: list[dict], rows: list[dict], *, samples: int,
                    split: str = "eval", exact_count: bool = True,
                    identity_cache: dict | None = None) -> dict:
    tasks_by_id = {task["id"]: task for task in tasks if task["split"] == split}
    if not tasks_by_id:
        raise ValueError("Evaluation task set is empty")
    found = defaultdict(list)
    labels = set()
    model_identities = set()
    identity_cache = {} if identity_cache is None else identity_cache
    for row in rows:
        if row["task_id"] not in tasks_by_id:
            raise ValueError("Attempt references a non-evaluation task")
        found[row["task_id"]].append(row)
        label = row.get("model")
        if type(label) is not str or not label.strip():
            raise ValueError("Missing or malformed declared model label")
        labels.add(label)
        model_identities.add(require_model_identity(row, cache=identity_cache))
    if set(found) != set(tasks_by_id):
        missing = sorted(set(tasks_by_id) - set(found))
        raise ValueError(f"Incomplete held-out coverage; missing tasks: {missing}")
    if len(labels) != 1:
        raise ValueError("Attempt file must contain exactly one declared model identity")
    if len(model_identities) != 1:
        raise ValueError("Model digest or HF identity changed within one evaluation arm")
    if not isinstance(samples, int) or not 1 <= samples <= 16:
        raise ValueError("Sample count must be 1..16")
    decoded = {}
    common_configs: list[tuple] | None = None
    for task_id, attempts in found.items():
        if len(attempts) < samples or (exact_count and len(attempts) != samples):
            raise ValueError(f"Incorrect sample coverage for {task_id}")
        attempts = sorted(attempts, key=lambda row: row["attempt"])
        if [row["attempt"] for row in attempts] != list(range(len(attempts))):
            raise ValueError(f"Missing, duplicate or noncontiguous attempt indices for {task_id}")
        configs = [require_generation_metadata(row) for row in attempts]
        prompts = [require_prompt_metadata(
            row, tasks_by_id[task_id], repository="repo_family" in tasks_by_id[task_id],
        ) for row in attempts]
        if any(value != prompts[0] for value in prompts[1:]):
            raise ValueError(f"Changed prompt configuration within task {task_id}")
        if common_configs is None:
            common_configs = configs
        elif configs != common_configs:
            raise ValueError(f"Inconsistent cross-task generation budget/seed for {task_id}")
        decoded[task_id] = {
            "rewards": tuple(row["reward"] for row in attempts),
            "configs": configs,
            "prompt": prompts[0],
            "domain": tasks_by_id[task_id].get("domain", "unclassified"),
        }
    return {"label": next(iter(labels)), "identity": next(iter(model_identities)),
            "tasks": decoded}


def summarize_single_run(tasks: list[dict], rows: list[dict], *,
                         split: str = "eval", k: int = 1) -> dict:
    """Complete single-model empirical score; rejects mixed/incomplete runs."""
    selected = [row for row in rows if row["split"] == split]
    run = _summarize_rows(tasks, selected, samples=k, split=split, exact_count=False)
    per_task = run["tasks"]
    n = len(per_task)
    first_pass = sum(item["rewards"][0] for item in per_task.values())
    any_k = sum(any(item["rewards"][:k]) for item in per_task.values())
    return {
        "split": split, "model": run["label"], "model_identity": run["identity"],
        "tasks": n,
        "samples_per_task_recorded": len(next(iter(per_task.values()))["rewards"]),
        "pass_at_1": first_pass / n, "wilson95_pass1": wilson_interval(first_pass, n),
        "observed_any_pass_first_k": any_k / n, "k": k,
        "generation_timing": measured_generation_timings(selected),
        "caution": "Local candidate sampling, not an independent frontier benchmark.",
    }


def compare_records(tasks: list[dict], baseline: list[dict], candidate: list[dict],
                    *, samples: int = 1, bootstrap_samples: int = 10_000) -> dict:
    """Compare already replayed reward records; callers must reverify first."""
    identity_cache: dict = {}
    original = _summarize_rows(tasks, baseline, samples=samples,
                               identity_cache=identity_cache)
    changed = _summarize_rows(tasks, candidate, samples=samples,
                              identity_cache=identity_cache)
    if original["label"] == changed["label"]:
        raise ValueError("Model labels must distinguish baseline from candidate")
    if original["identity"] == changed["identity"]:
        raise ValueError("Baseline and candidate have identical immutable model identity")
    if (original["identity"][0] == changed["identity"][0] == "hf"
            and original["identity"][1] == changed["identity"][1]
            and len(original["identity"]) == len(changed["identity"]) == 4
            and original["identity"][3] == changed["identity"][3]):
        raise ValueError("Baseline and candidate have identical HF adapter weight digests")
    paired = []
    by_domain: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for task_id in sorted(original["tasks"]):
        before = original["tasks"][task_id]
        after = changed["tasks"][task_id]
        if before["configs"] != after["configs"]:
            raise ValueError(f"Unequal sampling budget/seed for {task_id}")
        if before["prompt"] != after["prompt"]:
            raise ValueError(f"Unequal rendered prompt or prompt configuration for {task_id}")
        pair = (before["rewards"][0], after["rewards"][0])
        paired.append(pair)
        if before["domain"] != after["domain"]:
            raise ValueError(f"Evaluation task domain changed for {task_id}")
        if not isinstance(before["domain"], str) or not before["domain"].strip():
            raise ValueError(f"Missing or malformed task domain for {task_id}")
        by_domain[before["domain"]].append(pair)
    n = len(paired)
    pass_original = sum(x for x, _ in paired)
    pass_changed = sum(y for _, y in paired)
    improved = sum(x == 0 and y == 1 for x, y in paired)
    regressed = sum(x == 1 and y == 0 for x, y in paired)
    both_pass = sum(x == 1 and y == 1 for x, y in paired)
    both_fail = n - improved - regressed - both_pass
    firstk_before = sum(any(original["tasks"][key]["rewards"][:samples])
                        for key in original["tasks"]) / n
    firstk_after = sum(any(changed["tasks"][key]["rewards"][:samples])
                       for key in changed["tasks"]) / n
    return {
        "schema": "gemma-coding-paired-eval-v2",
        "task_kind": "repository" if any("repo_family" in task for task in tasks) else "function",
        "split": "eval",
        "baseline": {"declared_model": original["label"],
                     "model_identity_evidence": original["identity"],
                     "pass_at_1": pass_original / n,
                     "wilson95": wilson_interval(pass_original, n),
                     "observed_any_pass_first_k": firstk_before,
                     "generation_timing": measured_generation_timings(baseline)},
        "candidate": {"declared_model": changed["label"],
                      "model_identity_evidence": changed["identity"],
                      "pass_at_1": pass_changed / n,
                      "wilson95": wilson_interval(pass_changed, n),
                      "observed_any_pass_first_k": firstk_after,
                      "generation_timing": measured_generation_timings(candidate)},
        "matched_tasks": n,
        "samples_per_task": samples,
        "paired_delta_pass1": (pass_changed - pass_original) / n,
        "paired_delta_bootstrap95": bootstrap_delta_ci(
            paired, samples=bootstrap_samples),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(improved, regressed),
        "discordant": {"improved": improved, "regressed": regressed,
                       "both_pass": both_pass, "both_fail": both_fail},
        "domains": {
            name: {"tasks": len(domain_pairs),
                   "baseline_pass1": sum(x for x, _ in domain_pairs) / len(domain_pairs),
                   "candidate_pass1": sum(y for _, y in domain_pairs) / len(domain_pairs),
                   "improved": sum(x == 0 and y == 1 for x, y in domain_pairs),
                   "regressed": sum(x == 1 and y == 0 for x, y in domain_pairs)}
            for name, domain_pairs in sorted(by_domain.items())
        },
        "cautions": [
            "Ollama digests are collected claims; re-attest loaded model on a trusted endpoint for external audit.",
            "HF files are hashed against local evidence; this cannot prove which weights inference loaded.",
            "Paired bootstrap assumes evaluation tasks are representative and independent.",
            "Exact McNemar is a paired correctness test, not a model-equivalence test.",
            "Observed first-k success is not unbiased pass@k on unequal or adaptive candidate sampling.",
            "No Opus or coding-improvement claim without sufficiently powered independent holdouts.",
        ],
    }


def compare_files(args: argparse.Namespace) -> dict:
    tasks, baseline = checked_heldout_rows(
        args.tasks, args.baseline_attempts, args.task_kind, image=args.image)
    second_tasks, candidate = checked_heldout_rows(
        args.tasks, args.candidate_attempts, args.task_kind, image=args.image)
    if tasks != second_tasks:
        raise ValueError("Evaluation task source changed during comparison")
    report = compare_records(tasks, baseline, candidate, samples=args.samples,
                             bootstrap_samples=args.bootstrap_samples)
    report["provenance"] = {
        "source_tasks_sha256": sha256(args.tasks),
        "baseline_attempts_sha256": sha256(args.baseline_attempts),
        "candidate_attempts_sha256": sha256(args.candidate_attempts),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-kind", choices=["function", "repository"], required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--baseline-attempts", type=Path, required=True)
    parser.add_argument("--candidate-attempts", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--output", type=Path, help="Create-only JSON report path")
    args = parser.parse_args(argv)
    try:
        report = compare_files(args)
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        print(rendered)
        return 0
    except (ValueError, OSError, RuntimeError, FileExistsError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
