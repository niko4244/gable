# Hard Band: frontier-hard coding curriculum for Gable RCL

This repository publishes the curriculum framework, not a qualified private
benchmark. The upstream Claude archive also included a three-project authored
toy corpus (`hardband/corpus.py`) and a test file containing its reference
solutions and hidden-case generators. Those files are **demonstration material**:
if published, none of their cases can serve as secret holdouts, frontier
qualification evidence, or claims of generalization. Gable's public release
excludes them, along with the local-only `_trusted.py` helper.

Importing `hardband` and using its prompt, calibration, replay, and export
interfaces requires no private corpus. The `build` subcommand requires
separately supplied, reviewed `hardband/corpus.py` and `hardband/_trusted.py`
and fails clearly when they are missing. Treat a local-only reference-case
generator as trusted author code; candidate model code must execute in Gable's
restricted container runner.

Hardness is measured, not assumed. A task enters the research export only if
the frontier model's measured **per-attempt success rate** is greater than zero
and at most 0.5 across at least eight distinct valid samples, and the student
passes at least one of four or more samples. This is not a pass@8 statistic.
The declared model digests must be stable and distinct; replay also checks
deterministic prompt hashes, decoding settings, and sample seed schedules.
Digest provenance must be independently verified before using these results.

Families implemented: single_defect, coupled_defects (k=2,3; fixing any one
defect still fails hidden cases). Plausible-wrong-fix traps: visible tests pass
on the broken repo; defects are logic bugs (no crashes).
Planned: underspecified behavior, seeded concurrency, complexity budgets,
differential-fuzz reimplementation, log-only debugging (C4).

Pipeline:
  python -m hardband build --out-dir data_01 --seed 7 --per-project 9
  python -m hardband prompts --tasks data_01/tasks_train.jsonl --output p.jsonl
  (collect frontier.jsonl and student.jsonl attempts bound to the prompts)
  python -m hardband replay --tasks data_01/tasks_train.jsonl \
      --private data_01/private_train.jsonl \
      --attempts frontier.jsonl student.jsonl --output replay.jsonl
  python -m hardband calibrate --tasks data_01/tasks_train.jsonl \
      --private data_01/private_train.jsonl \
      --attempts frontier.jsonl student.jsonl \
      --frontier FRONTIER_ID --student STUDENT_ID --output buckets.json
  python -m hardband export --tasks data_01/tasks_train.jsonl \
      --private data_01/private_train.jsonl --attempts student.jsonl \
      --frontier-attempts frontier.jsonl \
      --frontier FRONTIER_ID --student STUDENT_ID --out-dir export_01

Each attempt must contain task_id, model, model_digest (64 lower-case hex
characters representing an independently verified immutable identity),
sample (non-negative integer), response,
prompt_sha256, and generation_config with temperature, top_p, num_predict, and
seed. The replay recomputes rewards; user-supplied verified/reward labels are
not trusted. Files named private_*.jsonl and model responses must remain local.

Buckets: learnable H1-H4 (by frontier per-attempt success rate),
frontier_unsolved (frontier had no successes, kept out of training),
too_easy, not_yet (student had no successes), and insufficient
(fewer than eight verified frontier samples or four student samples).
Reward: 1.0 only if all hidden fail-to-pass, hidden pass-to-pass and visible
checks pass; shaping <= 0.1 and gated on zero regressions, never used for DPO
pair selection. Unparseable or infrastructure failures cannot be negatives.
Export refuses any eval-split input and declares training_admitted false.

Limits: The original toy demonstration contained only three repositories.
The supplied corpus has exposed reference answers, so it cannot provide
independent evaluation. The reference-generation helper is local and
not a security boundary; only reviewed author code may enter that path.
Splits are by repository. Qualifying a real training campaign requires
license-reviewed repositories, hundreds of independently sourced tasks,
sealed evaluation holdouts, verified model identities, and real candidate
executions in the container runner.
