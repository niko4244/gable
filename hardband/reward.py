"""Execution validity and task success are distinct; no host candidate execution."""
import subprocess

import repo_rcl
from .sandbox import judge


def score(public, private, candidate_files):
    zero = {"verified": False, "passed": False, "f2p_frac": 0.0, "reward": 0.0}
    if not isinstance(candidate_files, dict) or not candidate_files:
        return {**zero, "reason": "no_files"}
    if any(p not in public["editable"] or not isinstance(s, str)
           for p, s in candidate_files.items()):
        return {**zero, "reason": "edit_outside_allowlist"}
    files = {**public["files"], **candidate_files}
    try:
        for path, source in files.items():
            repo_rcl.require_restricted_source(source, path, files)
    except (SyntaxError, ValueError):
        return {**zero, "reason": "restricted_source"}
    f2p = private["hidden_f2p"]
    keep = public["visible_tests"] + private["hidden_p2p"]
    if not f2p or not keep:
        raise ValueError("Both repair and regression cases are required")
    try:
        states = judge(files, f2p + keep)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError):
        return {**zero, "reason": "harness"}
    fraction = states[:len(f2p)].count("pass") / len(f2p)
    keep_ok = all(s == "pass" for s in states[len(f2p):])
    passed = keep_ok and fraction == 1.0
    return {"verified": True, "passed": passed, "f2p_frac": fraction,
            "reward": 1.0 if passed else 0.1 * fraction if keep_ok else 0.0,
            "reason": "passed" if passed else "hidden_fail" if keep_ok else "regression"}
