"""Container-only execution; uses Gable's restricted source and isolated reporter."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import rcl
import repo_rcl


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def run_cases(files, cases, timeout=12):
    """Expected values remain on the host. Infrastructure failures raise."""
    if (not isinstance(files, dict) or not files or len(files) > 64
            or repo_rcl._json_size(files) > repo_rcl.MAX_REPO_BYTES
            or any(not repo_rcl._valid_file(p) or not isinstance(s, str)
                   for p, s in files.items())
            or len({p.casefold() for p in files}) != len(files)
            or any(p.split('/')[0].split('.')[0] in sys.stdlib_module_names for p in files)):
        raise ValueError("Unsafe repository source tree")
    for path, source in files.items():
        repo_rcl.require_restricted_source(source, path, files)
    runtime = rcl.container_runtime()
    if runtime is None:
        raise RuntimeError("Native container runtime required; no subprocess fallback")
    image = "python:3.11-slim"
    subprocess.run([runtime, "image", "inspect", image], check=True,
                   capture_output=True, timeout=10)
    results = []
    with tempfile.TemporaryDirectory(prefix="gable-hardband-") as folder:
        root = Path(folder)
        (root / "repo_case_runner.py").write_bytes(
            Path(repo_rcl.__file__).with_name("repo_case_runner.py").read_bytes())
        for path, source in files.items():
            target = root / "repository" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        for index, case in enumerate(cases):
            if not repo_rcl.ENTRY_RE.fullmatch(case["fn"]):
                raise ValueError("Invalid entry point")
            name = "gable-hardband-" + rcl.digest(str(root))[:12] + f"-{index}"
            command = [runtime, "run", "-i", "--rm", "--pull=never", "--name", name,
                       "--network=none", "--read-only", "--cap-drop=ALL",
                       "--security-opt=no-new-privileges", "--pids-limit=64",
                       "--memory=384m", "--cpus=1", "--user=65534:65534",
                       "--tmpfs=/tmp:rw,nosuid,noexec,size=16m",
                       "--mount", f"type=bind,source={root},target=/app,readonly",
                       "--workdir=/app", image, "python", "-I", "-B",
                       "/app/repo_case_runner.py"]
            try:
                proc = rcl.run_bounded(command, json.dumps({"entry": case["fn"],
                                       "args": case["args"]}, allow_nan=False), timeout=timeout)
            except (OSError, subprocess.TimeoutExpired, rcl.OutputLimitExceeded) as exc:
                subprocess.run([runtime, "rm", "-f", name], capture_output=True, timeout=10)
                raise RuntimeError("Inconclusive container execution") from exc
            if proc.returncode:
                raise RuntimeError("Container reporter failed")
            result = json.loads(proc.stdout, object_pairs_hook=rcl.unique_json_pairs,
                                parse_constant=rcl.reject_json_constant)
            if result.get("kind") == "value":
                results.append({"ok": True, "value": result["value"],
                                "value_type": result["value_type"]})
            elif result.get("kind") == "exception":
                results.append({"ok": False, "error": result["type"]})
            else:
                raise RuntimeError("Inconclusive container protocol")
    return results


def judge(files, cases, timeout=12):
    return ["error" if not r["ok"] else
            "pass" if r["value_type"] == type(c["expected"]).__name__
            and same(c["expected"], r["value"]) else "fail"
            for c, r in zip(cases, run_cases(files, cases, timeout), strict=True)]

