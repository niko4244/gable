# Gable — Gemma Coding Reinforcement Learning

**Public source mirror.** Gable develops independently checked function-level
coding tasks, multi-file repository-repair rewards, guarded SFT/DPO preparation,
and paired held-out evaluation for an abliterated Gemma 4 12B checkpoint.
See [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md) for the publication boundary: hidden
evaluation cases, reference fixes, private test corpora, generated model attempts
and model weights are deliberately not published. The original local project
retains the private evidence. The public CPU contracts run without those files.

**Current model status:** no completed real Gemma fine-tuning, proven coding
improvement, or independently demonstrated frontier-model parity. The sections
below document the original V0/V2 research design and historical measurements;
do not interpret them as a live performance report.

**Goal:** make the user's installed, Heretic-abliterated Gemma 4 12B a much better coding model through measured post-training. The aspirational benchmark is Opus 5.5 coding quality; no equivalence is claimed. RCL here means **Reinforcement Coding Loop**: generate multiple candidate solutions, run executable checks, save verified trajectories, perform SFT and preference optimization, and measure on unseen problems.

## Starting checkpoint (observed November 15, 2026)

- Installed Ollama identity: `hf.co/zaakirio/gemma-4-12b-it-uncensored-GGUF:Q6_K`.
- Local model observation: an installed 9,786,020,480-byte Ollama GGUF blob with a separate projector; the GGUF is an inference artifact, not an HF trainable checkpoint.
- Corresponding trainable *abliterated* BF16 source: [zaakirio/gemma-4-12b-it-uncensored](https://huggingface.co/zaakirio/gemma-4-12b-it-uncensored), Hugging Face safetensors, `gemma4_unified`. Do not silently substitute an unmodified Google model or the unrelated `gemma4-bench` alias.
- No corresponding BF16 weights were found in inspected local caches. The Q6_K GGUF is for inference and cannot directly receive regular Hugging Face LoRA weight updates.
- Host RTX 5060 Ti 16,311 MiB total. Free GPU memory fluctuated from ~1.7 GiB to ~10.3 GiB during the session, still below the default 13,000 MiB preflight floor. Every local drive had less than 7 GiB free; the five original BF16 weight shards alone require roughly 24 GB. Real training is currently blocked; no downloads or process suspensions were undertaken.
- Model license is Gemma, not Apache 2.0. Six elementary original project-authored tasks have CC0 labels and exist solely to test pipeline mechanics.

## Curriculum and evaluation

| Stage | Skill target | Verifiable reward | Promotion |
| --- | --- | --- | --- |
| C0 baseline | Existing unmodified/abliterated coding performance | Frozen unseen tests, pass@1, token/time budgets | Baselines recorded before training |
| C1 fundamentals | Functions, edge cases, parsing, typing | Host-checked hidden JSON input/output cases, test isolation | Better unseen pass@1 |
| C2 algorithms | Dynamic programming, graphs, complexity, data structures | Deterministic tests and performance constraints | Holds against algorithm holdouts |
| C3 repository repair | Multi-file bugs, safe patches, regression fixes | Hidden fail-to-pass and pass-to-pass checks | No existing tests broken |
| C4 tool use | Inspect/edit/run tests and revise | Real tool outcomes, final tests, step limits | Higher same-harness solve rate |
| C5 advanced agentic | Larger refactors, system design, long tasks | Independent private evaluation | Significant held-out transfer |

Freeze evaluation suites and harnesses *before* looking at adapter results. Keep test implementation out of generated-model prompts. Exclude identical and near-duplicate problems, related repository families, and any public benchmark cases from the training pool. The loader rejects reused test suites across splits even when a function is renamed, plus exact and high-token-overlap prompt duplicates. These rules cannot prove semantic independence where both the prompt and cases are rewritten; human curation and separate independent holdouts remain essential. Evaluate baseline and tuned models with identical generation settings, token budgets, tools and task order. Compare pass@1 and uncertainty by domain, syntax failures, pass-to-pass regressions, latency and token cost. EvalPlus and LiveCodeBench are useful public checks; fresh private problems and a repository-level SWE harness are needed to assess agentic parity. A near-perfect score on the six toy examples is not evidence of advanced coding skill.

**Learning sequence:** collect 4 or more local candidates per task → constrained Docker verifier → SFT examples from verified passes → same-prompt DPO success/failure pairs → optional GRPO if a future dedicated-GPU fit test succeeds. Unsloth documents Gemma 4 SFT and RL, but published low-VRAM RL requirements refer to the E2B model; Gemma 4 12B GRPO on 16GB remains unverified.

For an actual coding-improvement campaign, add hundreds to thousands of independently sourced and license-reviewed training tasks. Keep evaluation on unseen tasks from different repositories and release dates, with a separate adjudicator for suspected paraphrase overlap. Promoting a LoRA adapter should require positive, statistically defensible change on the frozen evaluation split, no critical pass-to-pass regressions, and better completion accuracy under the same tool/time/token budget. An Opus 5.5 quality claim additionally requires a direct, dated comparison using the same independent tasks, execution environment, allowed tools, budget and scoring protocol. Neither a pilot training loss decrease nor this smoke test can establish such parity.

## C3 V1: verified repository repairs (research-only until independent validation)

A separate repository-repair track now supports small Python source trees, explicit editable-file allowlists, host-held fail-to-pass and protected regression cases, baseline-defect proof, source/test-set SHA binding, and separately replayed SFT/DPO exports. Candidates return JSON of whole-file replacements, not a loose prose patch. The sandbox uses an independently forked parent reporter; candidate modules never load in that parent. Repository source and patches are additionally restricted to a narrow Python AST subset, non-shadowing standard library modules, and collision-free Windows filenames. This is a constrained research fixture, not a trustworthy evaluator for arbitrary hostile repository code.

The new task dataset contains one project-authored training fixture and one disjoint evaluation fixture. It is too small to establish skill. Check C3_REPOSITORY_RCL_V1.md for usage, exact data schema, model-collection commands, successful synthetic Podman verification, security caveats and resume gates. The trainer selects this lane with --task-kind repository and keeps the old function path as the default. No genuine Gemma repository candidate has been evaluated or trained.

A one-candidate attempt to measure the installed abliterated Gemma Q6 on the repository evaluation fixture timed out at the Ollama API, produced no attempt file, and must not be scored as a coding failure or success. The temporary inference model was explicitly unloaded afterward to restore GPU memory. Verified evidence is limited to 73 passing CPU tests, native Podman synthetic positives/negatives, paired evaluator statistics on constructed fixtures, and independent red-team rejection of historical forged rewards. The additional adversarial_integration_smoke.py checks real Podman rejection of tuple/list API confusion and dictionary integer/string key mutation. DPO configuration prevents the historical TRL 512-token prompt truncation default, but DPO remains intentionally **BLOCKED** pending runtime qualification: stock TRL dispatches Gemma4Unified through its image-processing path, while Unsloth provides a patched image-free alternative that is not yet tested against this exact community model and processor. Native tokenizer and trained-policy integration remain untested.

## V2: matched held-out coding evaluation

The new compare_holdouts.py requires complete disjoint evaluation runs from two different declared model identities. It independently replays every result, checks exact task coverage, single-model arms, contiguous samples, equal seed schedules and decoding budgets, and refuses incomplete or unverified data. It reports pass@1 with Wilson bounds, observed first-k success, paired gains/regressions, an exact McNemar comparison, deterministic paired bootstrap 95% difference interval, domain-level changes and optional observed generation latency. Model labels and metadata are declarations, not cryptographic proofs of checkpoint identity; an independent benchmark with a proper task count, tool budget and blind holdouts remains necessary.

    python compare_holdouts.py --task-kind function --tasks seed_tasks.jsonl --baseline-attempts baseline_eval.jsonl --candidate-attempts tuned_eval.jsonl --samples 1 --output paired_eval.json

Use --task-kind repository with repo_seed_tasks.jsonl and full repository attempt files for the C3 lane. The command refuses create-time overwrites. Both function and repository collectors record temperature, top_p, token budget, generation seed and separate generation/verifier durations for a reproducible comparison. Type-aware sandbox reports now reject a top-level tuple when the task requires a list, even when ordinary JSON values coincide. See STATUS_20261115_V2.md for current test evidence and all remaining training prerequisites.

## Local CPU preparation

With Python 3.11+ (the data pipeline uses only the standard library):

    python rcl.py validate --tasks seed_tasks.jsonl
    python -m unittest -v test_rcl

For candidate generation and verification, a native, Python-accessible Docker or Podman CLI with Linux-container support must be installed with an already present image `python:3.11-slim`. This machine has the native executable `C:\Program Files\RedHat\Podman\podman.exe` and the necessary local image, which the pipeline now detects automatically; its Bash-only `docker` shim does not need to be invoked. The pipeline does not pull images during verification. Use the installed Ollama model when inference capacity is available:

    python rcl.py collect --tasks seed_tasks.jsonl --split train --model hf.co/zaakirio/gemma-4-12b-it-uncensored-GGUF:Q6_K --samples 4 --output train_attempts.jsonl
    python rcl.py export --tasks seed_tasks.jsonl --attempts train_attempts.jsonl --out-dir export_01
    python rcl.py collect --tasks seed_tasks.jsonl --split eval --model hf.co/zaakirio/gemma-4-12b-it-uncensored-GGUF:Q6_K --samples 1 --output eval_attempts.jsonl
    python rcl.py metrics --tasks seed_tasks.jsonl --attempts eval_attempts.jsonl --split eval -k 1

Candidate Python runs only in isolated Docker/Podman with no network, read-only root, no capabilities, CPU/memory/process/time limits and a read-only mounted exercise directory. Each case sends a single function name and input arguments into the sandbox; expected outputs stay on the host and are checked there. Before exporting training examples or reporting pass rates, the program replays every asserted reward through the sandbox; a self-reported success never qualifies. Timeouts, unavailable containers and excessive output are classified as `verified=false` and excluded from learning, instead of being used as negative DPO preferences. Host stdout/stderr capture has fixed byte limits. If the container runtime/image is missing, verification blocks. Syntax validity alone is not a success reward. This is a useful local isolation boundary, not an absolute security guarantee.

Run the observed complete CPU/container smoke test to verify both a correct and an incorrect synthetic answer, including the reverified preference export:

    python integration_smoke.py

On the observed host this finished `PASS`: five cases per candidate, correct solution passed, incorrect solution failed, one SFT example and one DPO pair. This was synthetic fixture validation, with no Gemma model inference or training.

## GPU QLoRA pilot (resource-gated)

Use a separate Linux/WSL environment with a tested, pinned set of `unsloth`, `transformers`, `trl`, `peft`, `torch`, `datasets`, `bitsandbytes` and CUDA. The entrypoint imports heavy dependencies only *after* preflight, and never downloads weights or installs packages. The path `/mnt/x/path/to/abliterated-HF-weights` must hold the actual matching BF16 safetensors checkpoint and tokenizer:

    python train.py --phase sft --base /mnt/x/path/to/abliterated-HF-weights --tasks seed_tasks.jsonl --attempts train_attempts.jsonl --data export_01/sft.jsonl --output adapters/sft_01
    python train.py --phase sft --base /mnt/x/path/to/abliterated-HF-weights --tasks seed_tasks.jsonl --attempts train_attempts.jsonl --data export_01/sft.jsonl --output adapters/sft_01 --execute
    python train.py --phase dpo --base /mnt/x/path/to/abliterated-HF-weights --tasks seed_tasks.jsonl --attempts train_attempts.jsonl --sft-adapter adapters/sft_01 --data export_01/dpo.jsonl --output adapters/dpo_01 --execute

First intended settings: 4-bit Unsloth FastModel, rank 8/alpha 8 LoRA, text/attention/MLP layers, 2,048 tokens, microbatch 1, gradient accumulation 4, BF16, 8-bit AdamW, eight trial steps. Default free-space gates are 13,000 MiB GPU memory and 20 GiB on output drive. Check peak GPU memory before increasing sequence length or batch size. The complete BF16 Gemma 4 Unified five-shard snapshot and processor files are required, and untrusted/partial weight files are refused. Source revision `5fdfef5f3aa9dd3c680a22078f2aa97b4a31c919`, every shard SHA-256, tokenizer SHA-256, and Git-blob identities for the remaining config/processor/template files are pinned in `source_pin.json`. Full byte-for-byte source checks must pass before readiness. The preflight also checks exported data, source task hashes, attempt hashes and repeats reward verification. SFT masks user portions and verifies that assistant code remains in the targets. DPO uses separate training and frozen SFT reference adapters: before training, it checks the installed TRL's adapter configuration API, explicitly freezes reference parameters and proves they are absent from the optimizer; it also checks the reference remains frozen afterward. No Unsloth/TRL/PEFT version combination has yet been validated with the actual 12B weights. Do not use the live DPO stage until the precise library versions, reference behavior and memory requirements are verified. The CLI deliberately returns BLOCKED in the current host state.

## Files

- `rcl.py` — load/validate tasks and provenance, isolate test execution, generate via local Ollama, export SFT/DPO datasets and compute evaluation statistics.
- `oracle_runner.py` — sandboxed single-call Python executor. It sees inputs only, while expected answers remain on the host.
- `train.py` — pretrained Gemma 4 Unified checkpoint identity and resource preflight; opt-in Unsloth SFT/DPO pilot.
- `source_pin.json` — frozen upstream HF source revision, all eleven required file identities, and model shard hashes.
- `seed_tasks.jsonl` — six synthetic function-and-JSON-case exercises: three train, one validation, two eval. They are pipeline smoke tests only.
- `test_rcl.py` — CPU contract tests around score integrity, task isolation, model identity gate, and verifier restrictions.
- `integration_smoke.py` / `smoke_solution.py` — actual local Podman positive/negative and SFT/DPO export smoke with a synthetic solution.

Each collection and export uses a new path, refusing to overwrite previous evidence. Do not overwrite original GGUF or BF16 weights. Preserve original tokenizer, processor, config and template behavior. Do not merge or quantize the resulting adapter until independently held-out quality and regression checks pass.

Independent red-team QA revealed and drove fixes for an initial C3 in-process JSON reporter spoof, cross-split source/test duplication, Gemma role markers hidden in repository files, and NTFS casefold path aliasing. Follow-up security checks should still be treated as an explicit admission gate; neither static AST filtering nor a forked process guarantees integrity against arbitrary malicious Python programs. Only curated/reviewed task corpora should ever reach training, and no release claim is permitted without separate held-out evaluation.

## References

- [Gemma 4 official model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Abliterated source model and license](https://huggingface.co/zaakirio/gemma-4-12b-it-uncensored)
- [Unsloth Gemma 4 training guide](https://unsloth.ai/docs/models/gemma-4/train)
- [TRL SFT](https://huggingface.co/docs/trl/sft_trainer), [TRL DPO](https://huggingface.co/docs/trl/dpo_trainer), [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)
- [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym), [EvalPlus](https://evalplus.github.io/)
