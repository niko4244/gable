"""Hard-band curriculum CLI; build corpus remains local and private."""
import argparse
import json
from pathlib import Path

import rcl
from .calibrate import calibrate
from .prompt import build_prompt, export, replay


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hardband")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--out-dir", type=Path, required=True)
    build_parser.add_argument("--seed", type=int, default=0)
    build_parser.add_argument("--per-project", type=int, default=6)
    build_parser.add_argument("--eval-projects", nargs="+", default=["ledger"])
    for name in ("prompts", "replay", "calibrate", "export"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--tasks", type=Path, required=True)
        if name != "export":
            cmd.add_argument("--output", type=Path, required=True)
        else:
            cmd.add_argument("--out-dir", type=Path, required=True)
        if name != "prompts":
            cmd.add_argument("--private", type=Path, required=True)
            cmd.add_argument("--attempts", type=Path, nargs="+", required=True)
        if name in ("calibrate", "export"):
            cmd.add_argument("--frontier", required=True)
            cmd.add_argument("--student", required=True)
        if name == "export":
            cmd.add_argument("--frontier-attempts", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "build":
        from .build import build
        print(json.dumps(build(args.out_dir, args.seed, args.per_project,
                               eval_projects=tuple(args.eval_projects))))
        return
    tasks = rcl.jsonl(args.tasks)
    if args.cmd == "prompts":
        rcl.write_jsonl(args.output, [{"task_id": t["task_id"], "prompt": build_prompt(t),
                                    "prompt_sha256": rcl.digest(build_prompt(t))} for t in tasks])
        return
    private = rcl.jsonl(args.private)
    attempts = [row for path in args.attempts for row in rcl.jsonl(path)]
    if args.cmd == "export":
        frontier_attempts = [row for path in args.frontier_attempts for row in rcl.jsonl(path)]
        sft, dpo = export(tasks, private, attempts, frontier_attempts, args.frontier, args.student)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        rcl.write_jsonl(args.out_dir / "sft.jsonl", sft)
        rcl.write_jsonl(args.out_dir / "dpo.jsonl", dpo)
        manifest = {"schema": "gable-hardband-research-v1", "training_admitted": False,
                    "source_tasks_sha256": rcl.file_digest(args.tasks),
                    "source_private_sha256": rcl.file_digest(args.private),
                    "attempt_file_sha256": [rcl.file_digest(p) for p in args.attempts],
                    "frontier_attempt_file_sha256": [rcl.file_digest(p) for p in args.frontier_attempts],
                    "sft_sha256": rcl.file_digest(args.out_dir / "sft.jsonl"),
                    "dpo_sha256": rcl.file_digest(args.out_dir / "dpo.jsonl"),
                    "sft_rows": len(sft), "dpo_rows": len(dpo),
                    "limitation": "Research export; not a qualified RCL trainer input."}
        with (args.out_dir / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(json.dumps(manifest))
    else:
        results = replay(tasks, private, attempts)
        if args.cmd == "calibrate":
            output = calibrate(results, args.frontier, args.student, [t["task_id"] for t in tasks])
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(output, handle, indent=2)
        else:
            rcl.write_jsonl(args.output, results)


if __name__ == "__main__":
    main()

