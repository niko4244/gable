# Gable public repository policy

Gable is the public source-code mirror of the local Gemma coding post-training
project. It publishes the code for sandboxed candidate verification, small
repository repair, Hard Band research curriculum mechanics,
dataset/optimizer admission, and paired evaluation.

## What is deliberately private

The hidden cases and expected outputs, held-out tasks, private case definitions,
reference solutions, generated attempts, model binaries, adapters, receipts
containing local paths, and credential/configuration files stay outside the
public Git history. Publishing them would contaminate the independent evaluation
and undermine the reward verifier's test secrecy. The original local workspace
retains them; they are not deleted or modified by publication.

For Hard Band, `hardband/corpus.py`, `hardband/_trusted.py`, and
`test_hardband.py` stay local. They include toy reference repairs,
deterministic hidden-case construction, or private corpus-specific tests. The
public framework is importable without these files; building a toy corpus
requires a separately supplied, locally licensed corpus implementation.

This means the public repository is a **source mirror**, not a complete,
ready-to-execute copy of the private research environment. The public CI
exercises source contracts without revealing private task corpora.

## Verified boundaries

- Function and repository rewards depend on independently re-running task
  cases through a local, isolated container.
- The Python subset restriction and Podman/Docker isolation are defense in
  depth, not a formal sandbox for arbitrary malicious Python.
- Synthetic smoke outputs and unit-test success do not constitute model gains.
- The exact pinned Gemma training weights, tokenizer and qualified runtime must
  pass source and resource admission before any LoRA/DPO training.
- No fine-tuned Gemma model, independently verified coding gain, or Claude Opus
  parity is asserted by publication.

## Publishing updates

Use the explicit staged-file allowlist in `publish_gable.ps1`. It prevents
accidentally committing local task JSONL, secrets, protected test code and
model artifacts. Review all diffs before pushing; an allowlist does not
substitute for careful security or rights review.

No license has been assigned to the project in this initial publication. A
public repository is not automatically an open-source software license.
