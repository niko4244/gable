# Gemma Coding RCL C3 — Repository Repair V1 (2026-11-15)

## Delivered and scope

This is a **working CPU/container-verified repository-repair training-data track**, not a trained coding adapter or a claim of model improvement. It extends the V0 single-function loop without changing the pinned Gemma checkpoint. It accepts small Python repositories with declared rights, a bug report, baseline files, an allowlist of editable source files, and **private host-held fix and regression test cases**. The model receives the relevant source tree and allowed filenames; it never receives expected outputs, case classifications, or the private test set.

Candidates return a single JSON object mapping allowed relative .py paths to **complete replacement source contents**. The project stages those files in an isolated, **read-only** Podman/Docker container. Each hidden input runs in an independent Python process; the trusted host compares the observed result to its private expected output or expected exception. Malformed patch syntax is a verified failure; missing containers, timeouts, malformed sandbox protocol, and output overflows are **inconclusive**, not DPO negative examples. Repository baselines must prove at least one fail-to-pass defect and must already pass all protected regressions. Export, metrics and trainer preflight independently replay asserted rewards.

## Commands

From the local project root (private task fixtures are not included in public Git):

    python repo_rcl.py validate --tasks repo_seed_tasks.jsonl
    python repo_rcl.py verify --tasks repo_seed_tasks.jsonl --task-id repo_range_fix_001 --baseline
    python -m unittest -q test_rcl test_repo_rcl
    python repo_integration_smoke.py

The native Podman image python:3.11-slim must already be installed. The synthetic integration test covers the original failing baseline, unchanged protected functions, a correct file replacement, an incorrect patch, reward replay, SFT export and DPO selection. These fixtures **prove plumbing**, not model coding capability.

When local inference resources permit, collect actual Gemma candidates and export verified records:

    python repo_rcl.py collect --tasks repo_seed_tasks.jsonl --split train --samples 4 --model hf.co/zaakirio/gemma-4-12b-it-uncensored-GGUF:Q6_K --output repo_attempts_01.jsonl
    python repo_rcl.py export --tasks repo_seed_tasks.jsonl --attempts repo_attempts_01.jsonl --out-dir repo_export_01

For real coding improvement, replace the synthetic tasks with legally sourced, independently curated genuine repository repairs. Each task must have a distinct repo_family across train/evaluation splits, at least two fix and two regression cases, and editable source paths explicitly allowlisted. The loader rejects same-family train/eval overlap but cannot prove semantic independence or audit falsely declared licenses.

The separate trainer accepts the new family-specific replay after the base checkpoint and resource prerequisites are satisfied. Example dry-run preflight (**do not append --execute yet**):

Important: Stock TRL 0.24.0 DPOTrainer sends Gemma4Unified through a vision-processing path, whose unpatched implementation expects image data. Unsloth 2026.6.1 contains a patched text-only-capable vision process_row implementation, but this project has NOT tested it with the pinned community Gemma4Unified 12B processor and adapters. DPO is explicitly disabled until that concrete runtime and a no-truncation dataset proof are verified. The commands describe intended data lineage, not currently executable DPO training.

    python train.py --task-kind repository --phase sft --base /mnt/x/GEMMA-HF-SOURCE --tasks repo_seed_tasks.jsonl --attempts repo_attempts_01.jsonl --data repo_export_01/sft.jsonl --output adapters/repo_sft_01

The trainer validates the repository manifest schema, source tasks and attempts hashes, all sandbox rewards, base checkpoint, VRAM and disk. The original V0 function path remains default --task-kind function. For DPO, use --task-kind repository, the matching SFT adapter, and the repo-exported dpo.jsonl; do not mix task families or rename schemas.

## Provenance and tests

Security scope: The pilot intentionally rejects many valid general-purpose Python programs, including most standard-library imports, global or attribute mutation, and interpreter-introspection features. This narrows the attack surface but also limits what repository fixes can be represented. An independent red-team initially demonstrated that an in-process reporter could be monkeypatched to forge positive rewards, that disguised copies of training repositories could contaminate holdouts, and that NTFS case aliases could alter a protected path. The implementation was changed to fork an independent reporter parent, reject source/test fingerprint overlap across splits, and reject casefold-equivalent/Windows-reserved files. These protections are defense-in-depth, NOT a proof that arbitrary untrusted code cannot fool the verifier.

- Repository generation prompts include full source files and editable paths but **exclude host-held hidden input cases**.
- Every attempt is tied to its task, original prompt hash, full serialized model prompt hash, and **complete task-file SHA-256 including hidden cases**.
- An edited or renamed source file outside the allowlist, duplicate JSON key, path traversal, overwritten output, invalid response, unverified reward, forged success, or changed task/test snapshot is rejected.
- The trained SFT/DPO dataset contains only a train-split verified positive, with optional same-task negative. Evaluation-split samples are not exported as training.
- The repo runner is sandboxed without network, capabilities or writable root, under fixed CPU/memory/PID/output/time limits. All original source files and verifier comparison logic stay outside the candidate writable tree.
- The synthetic real Podman smoke and CPU unit tests confirm the listed mechanics; they do **not** establish correctness of arbitrary user-supplied Python code or resistance to a malicious container-runtime breakout.

## Remaining gates

1. The matching 5-shard BF16 checkpoint for Hugging Face source zaakirio/gemma-4-12b-it-uncensored, revision 5fdfef5f3aa9dd3c680a22078f2aa97b4a31c919, is not installed. The available Q6_K GGUF cannot directly receive the current HF LoRA update.
2. Insufficient checked local disk space and GPU available memory; actual 12B QLoRA/DPO resource fit has **not** been demonstrated.
3. Pin and validate an actual compatible CUDA/Unsloth/Transformers/PEFT/TRL version set with Gemma4Unified 12B. Recent TRL DPOConfig APIs differ in named adapter support; trainer currently refuses unsupported versions rather than silently optimizing a wrong reference.
4. The community Gemma 4 12B checkpoint is not one of the canonical Google sizes in Unsloth's general Gemma 4 family guide; community-specific architecture/processor behavior must be proved with a small gated pilot.
5. Need hundreds or thousands of rights-reviewed real coding tasks, disjoint frozen holdouts, actual before/after model measurements, and independent direct comparison under equal token/tool/time budgets.

The current constrained Python file-replacement harness is **not** full SWE-bench or a multi-step tool-using coding agent. No Opus 5.5 parity or measurable coding improvement is claimed.
