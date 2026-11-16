"""Repository-repair coding loop with host-held tests and reproducible provenance.

A repo task contains a small, licensed Python source tree, an explicit editable
allowlist, and private, independently checked fix/regression cases. No task
tests/expected outputs are sent to the generator or copied into the sandbox.
This is a CPU pilot harness, not a substitute for independent SWE benchmarks.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import tempfile
import time
from urllib import request

import rcl


MAX_REPO_BYTES = 120_000
MAX_REPO_PATCH_BYTES = 40_000
REPO_SCHEMA = "gemma-coding-repo-rcl-v1"
REPO_INSTRUCTION = (
    "You are repairing a Python repository. Respond with ONLY a JSON object "
    "whose keys are allowed relative Python filenames and whose values are "
    "their complete replacement source text. Do not write tests, explanations, "
    "markdown fences, diffs, or unrelated files.\n\n"
)
FILE_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.py\Z")
ENTRY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){1,8}\Z")
LICENSES = {"CC0-1.0", "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}
RESERVED = rcl.RESERVED_CODE_MARKERS
SAFE_BUILTIN_CALLS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float",
    "int", "len", "list", "max", "min", "range", "round",
    "set", "sorted", "str", "sum", "tuple", "zip",
    "ValueError", "TypeError", "KeyError", "IndexError",
}
SAFE_METHOD_CALLS = {
    "lower", "upper", "strip", "lstrip", "rstrip", "split", "join",
    "replace", "startswith", "endswith", "isdigit", "isalpha",
    "count", "append", "pop", "get", "items", "keys", "values",
}
SAFE_AST_NODES = (
    ast.Module, ast.Expr, ast.ImportFrom, ast.alias, ast.FunctionDef,
    ast.arguments, ast.arg, ast.Return, ast.If, ast.Raise, ast.Pass,
    ast.Assign, ast.AugAssign, ast.For, ast.While, ast.Break,
    ast.Continue, ast.Constant, ast.Name, ast.Load, ast.Store,
    ast.Call, ast.keyword, ast.Attribute, ast.Subscript, ast.Slice,
    ast.Tuple, ast.List, ast.Dict, ast.Set, ast.Compare, ast.BoolOp,
    ast.BinOp, ast.UnaryOp, ast.IfExp,
    ast.And, ast.Or, ast.Not, ast.UAdd, ast.USub,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
)


def _local_assignment_names(tree: ast.FunctionDef) -> set[str]:
    """Only ordinary lexical locals; never infer globals from another file."""
    result = {
        arg.arg for arg in (
            [*tree.args.posonlyargs, *tree.args.args, *tree.args.kwonlyargs]
            + ([tree.args.vararg] if tree.args.vararg else [])
            + ([tree.args.kwarg] if tree.args.kwarg else [])
        )
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.For)):
            targets = (
                node.targets if isinstance(node, ast.Assign)
                else [node.target]
            )
            for target in targets:
                result.update(child.id for child in ast.walk(target)
                              if isinstance(child, ast.Name) and
                              isinstance(child.ctx, ast.Store))
    return result


def require_restricted_source(source: str, filename: str, files: dict[str, str]) -> None:
    """Allow only a bounded, pure-Python source subset (not arbitrary repair).

    Positive syntax and name allowlists are used instead of incomplete
    dangerous-name blacklists. This still is not a formal execution sandbox.
    """
    if any(marker in source for marker in RESERVED):
        raise ValueError(f"Reserved chat-template token in repository source: {filename}")
    tree = ast.parse(source, filename=filename)
    local_modules = {
        name[:-3].replace("/", ".").removesuffix(".__init__"): name
        for name in files
    }
    # A decoy def in a different file must never whitelist a dangerous
    # implicit Python builtin in this module (getattr, setattr, __import__).
    declared_functions = {
        child.name for child in tree.body if isinstance(child, ast.FunctionDef)
    }
    unsafe_builtins = set(dir(builtins)) - SAFE_BUILTIN_CALLS
    allowed_imported_names: set[str] = set()
    scopes: dict[ast.FunctionDef, set[str]] = {}
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(top, ast.Expr) and isinstance(top.value, ast.Constant) and (
                isinstance(top.value.value, str)):
            continue  # module docstring
        raise ValueError(f"Executable module-global statement forbidden: {filename}")
    for top in tree.body:
        if isinstance(top, ast.FunctionDef):
            if (top.args.defaults or any(v is not None for v in top.args.kw_defaults)
                    or top.returns is not None
                    or any(arg.annotation is not None for arg in (
                        [*top.args.posonlyargs, *top.args.args, *top.args.kwonlyargs]
                        + ([top.args.vararg] if top.args.vararg else [])
                        + ([top.args.kwarg] if top.args.kwarg else [])
                    ))):
                raise ValueError(f"Function default/annotation evaluation forbidden: {filename}")
            if top.name in dir(builtins):
                raise ValueError(f"Builtin-shadowing function forbidden: {filename}")
            scopes[top] = _local_assignment_names(top)
            for nested in ast.walk(top):
                if nested is not top and isinstance(nested, (ast.FunctionDef, ast.ImportFrom)):
                    raise ValueError(f"Nested definition/import forbidden: {filename}")
    for node in ast.walk(tree):
        if not isinstance(node, SAFE_AST_NODES):
            raise ValueError(f"Unsupported or interpreter-control Python syntax: {filename}")
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module not in local_modules:
                raise ValueError(f"Untrusted import-from in restricted repository: {filename}")
            root = node.module.split(".", 1)[0]
            if root in sys.stdlib_module_names:
                raise ValueError(f"Stdlib import shadowing forbidden: {filename}")
            imported = ast.parse(files[local_modules[node.module]])
            exported = {
                child.name for child in imported.body
                if isinstance(child, ast.FunctionDef)
            }
            for alias in node.names:
                if (alias.name not in exported or "__" in alias.name
                        or (alias.asname and "__" in alias.asname)
                        or (alias.asname or alias.name) in dir(builtins)):
                    raise ValueError(f"Unreviewed or private import alias: {filename}")
                allowed_imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.FunctionDef):
            if node.decorator_list or "__" in node.name:
                raise ValueError(f"Unreviewed function metadata/decorator: {filename}")
        elif isinstance(node, ast.Name):
            if "__" in node.id or node.id in {"globals", "locals", "eval", "exec"}:
                raise ValueError(f"Interpreter-control builtin is forbidden: {filename}")
            if isinstance(node.ctx, ast.Load) and node.id in unsafe_builtins:
                raise ValueError(f"Unsafe builtin reference forbidden: {filename}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(term in node.value for term in ("__builtins__", "__dict__", "__class__")):
                raise ValueError(f"Interpreter introspection literal forbidden: {filename}")
        elif isinstance(node, ast.Attribute):
            if (node.attr not in SAFE_METHOD_CALLS or
                    not isinstance(node.ctx, ast.Load)):
                raise ValueError(f"Attribute access/mutation not allowlisted: {filename}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in (
                        SAFE_BUILTIN_CALLS | declared_functions | allowed_imported_names):
                    raise ValueError(f"Nonallowlisted dynamic function call: {filename}")
            elif not (isinstance(node.func, ast.Attribute) and
                      node.func.attr in SAFE_METHOD_CALLS):
                raise ValueError(f"Dynamic call target forbidden: {filename}")
        elif isinstance(node, ast.Assign):
            if any(not isinstance(target, (ast.Name, ast.Tuple, ast.List))
                   for target in node.targets):
                raise ValueError(f"Complex assignment target forbidden: {filename}")
        elif isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                raise ValueError(f"Complex augmented assignment forbidden: {filename}")

    # No unbound name may resolve through the Python builtins, globals,
    # import side effects, or a decoy definition in another source file.
    for fn, local_names in scopes.items():
        permitted = (
            local_names | declared_functions | allowed_imported_names
            | SAFE_BUILTIN_CALLS
        )
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in permitted:
                    raise ValueError(f"Unbound/cross-file global reference forbidden: {filename}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if (node.func.id in local_names and
                        node.func.id not in declared_functions and
                        node.func.id not in allowed_imported_names and
                        node.func.id not in SAFE_BUILTIN_CALLS):
                    raise ValueError(f"Call through runtime local/argument forbidden: {filename}")


def repo_prompt(task: dict) -> str:
    """Expose task source code, never private verification cases."""
    source = []
    for path, code in sorted(task["files"].items()):
        source.append(f"FILE: {path}\n{code}")
    return (REPO_INSTRUCTION + "ISSUE:\n" + task["prompt"] + "\n\n"
            + "EDITABLE FILES:\n" + "\n".join(task["editable_files"])
            + "\n\nREPOSITORY SNAPSHOT:\n" + "\n\n".join(source))


def _valid_file(path: str) -> bool:
    return (
        isinstance(path, str) and FILE_RE.fullmatch(path) is not None
        and all(not PureWindowsPath(part).is_reserved() for part in path.split("/"))
    )


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_size(value: object) -> int:
    return len(json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8"))


def load_repo_tasks(path: Path) -> list[dict]:
    tasks = rcl.jsonl(path)
    if not tasks:
        raise ValueError("Repository task file is empty")
    ids: set[str] = set()
    families: dict[str, str] = {}
    old_prompts: list[tuple[str, set[str]]] = []
    cross_split_sources: list[tuple[str, str, str, set[str], set[str], set[str]]] = []
    for task in tasks:
        for field in ("id", "prompt", "source", "license", "repo_family", "split"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise ValueError(f"Missing repository task field {field}")
        task_id = task["id"]
        if task_id in ids:
            raise ValueError(f"Duplicate repository task ID: {task_id}")
        if task["split"] not in rcl.ALLOWED_SPLITS:
            raise ValueError(f"Invalid split: {task_id}")
        if task.get("language") != "python":
            raise ValueError(f"Only Python repositories supported: {task_id}")
        if task["license"] not in LICENSES:
            raise ValueError(f"Unreviewed or unsupported rights label: {task_id}")
        if any(m in task["prompt"] for m in RESERVED):
            raise ValueError(f"Reserved chat marker in prompt: {task_id}")
        family = task["repo_family"].casefold()
        if family in families and families[family] != task["split"]:
            raise ValueError(f"Repository family reused across train/evaluation: {family}")
        families[family] = task["split"]
        words = rcl.prompt_tokens(task["prompt"])
        for previous, earlier_words in old_prompts:
            intersection = len(words & earlier_words)
            union = max(1, len(words | earlier_words))
            if intersection / union >= 0.75:
                raise ValueError(f"Near-duplicate repository problem: {previous} / {task_id}")
        old_prompts.append((task_id, words))
        files = task.get("files")
        allowed = task.get("editable_files")
        if (not isinstance(files, dict) or not 1 <= len(files) <= 64 or
                not all(_valid_file(k) and isinstance(v, str) for k, v in files.items()) or
                _json_size(files) > MAX_REPO_BYTES):
            raise ValueError(f"Unsafe or oversized repository file tree: {task_id}")
        if any(filename.split("/", 1)[0].split(".", 1)[0] in sys.stdlib_module_names
               for filename in files):
            raise ValueError(f"Repository shadows a Python standard-library module: {task_id}")
        if len({filename.casefold() for filename in files}) != len(files):
            raise ValueError(f"Windows-casefold collision in repository source: {task_id}")
        for filename, source in files.items():
            require_restricted_source(source, filename, files)
        if (not isinstance(allowed, list) or not 1 <= len(allowed) <= 16 or
                any(not _valid_file(f) or f not in files for f in allowed) or
                len(allowed) != len(set(allowed))):
            raise ValueError(f"Invalid editable-files allowlist: {task_id}")
        if len(repo_prompt(task).encode("utf-8")) > 16_000:
            raise ValueError(f"Repository prompt exceeds bounded 16KB context: {task_id}")
        cases = task.get("cases")
        if not isinstance(cases, list) or not 4 <= len(cases) <= 64:
            raise ValueError(f"Need 4..64 repository hidden cases: {task_id}")
        if _json_size(cases) > rcl.MAX_TEST_BYTES:
            raise ValueError(f"Oversized repository hidden cases: {task_id}")
        purposes = []
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError(f"Malformed repository case: {task_id}")
            entry = case.get("entry")
            if not isinstance(entry, str) or not ENTRY_RE.fullmatch(entry):
                raise ValueError(f"Malformed repository entry point: {task_id}")
            module = entry.rsplit(".", 1)[0]
            if module.replace(".", "/") + ".py" not in files and (
                    module.replace(".", "/") + "/__init__.py" not in files):
                raise ValueError(f"Entry point module absent: {entry}")
            if not isinstance(case.get("args", []), list) or not isinstance(case.get("kwargs", {}), dict):
                raise ValueError(f"Malformed repository case arguments: {task_id}")
            if ("expected" in case) == ("raises" in case):
                raise ValueError(f"Must specify expected OR raises: {task_id}")
            if "raises" in case and case["raises"] not in ("ValueError", "TypeError", "IndexError", "KeyError"):
                raise ValueError(f"Unsupported expected exception: {task_id}")
            if case.get("purpose") not in ("fix", "regression"):
                raise ValueError(f"Specify fix/regression purpose: {task_id}")
            purposes.append(case["purpose"])
        if purposes.count("fix") < 2 or purposes.count("regression") < 2:
            raise ValueError(f"Need at least two fix and two regression cases: {task_id}")
        source_digest = hashlib.sha256(json.dumps(
            files, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        source_asts = {
            ast.dump(ast.parse(contents), include_attributes=False)
            for contents in files.values() if contents.strip()
        }
        source_tokens = rcl.prompt_tokens("\n".join(
            contents for contents in files.values() if contents.strip()))
        case_signatures = {
            json.dumps({name: value for name, value in case.items()
                        if name not in ("entry", "purpose")},
                       sort_keys=True, ensure_ascii=False, allow_nan=False)
            for case in cases
        }
        for earlier_id, earlier_split, earlier_digest, earlier_asts, earlier_words, earlier_cases in cross_split_sources:
            if earlier_split == task["split"]:
                continue
            ast_overlap = len(source_asts & earlier_asts)
            case_overlap = len(case_signatures & earlier_cases)
            word_union = len(source_tokens | earlier_words)
            if (
                source_digest == earlier_digest
                or (min(len(source_asts), len(earlier_asts)) >= 1 and
                    ast_overlap / min(len(source_asts), len(earlier_asts)) >= 0.8)
                or (min(len(source_tokens), len(earlier_words)) >= 8 and
                    len(source_tokens & earlier_words) / max(word_union, 1) >= 0.9)
                or (min(len(case_signatures), len(earlier_cases)) >= 2 and
                    case_overlap / min(len(case_signatures), len(earlier_cases)) >= 0.8)
            ):
                raise ValueError(
                    f"Repository source/test contamination across splits: {earlier_id} / {task_id}"
                )
        cross_split_sources.append((task_id, task["split"], source_digest,
                                    source_asts, source_tokens, case_signatures))
        ids.add(task_id)
    return tasks


def parse_patch(raw: str, task: dict) -> dict[str, str]:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_REPO_PATCH_BYTES:
        raise ValueError("Patch response is oversized or not text")
    try:
        changes = json.loads(raw, object_pairs_hook=_strict_pairs,
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Patch response must be a plain JSON object") from exc
    if not isinstance(changes, dict) or not changes:
        raise ValueError("Patch must modify at least one allowed file")
    editable = set(task["editable_files"])
    for name, source in changes.items():
        if name not in editable or not _valid_file(name):
            raise ValueError(f"Patch changes a file outside the editable allowlist: {name}")
        if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_REPO_PATCH_BYTES:
            raise ValueError("Patch contains invalid replacement source")
        require_restricted_source(source, name, {**task["files"], **changes})
    return changes


def _run_project(task: dict, changes: dict[str, str], *, image: str, seconds: int) -> dict:
    runtime = rcl.container_runtime()
    if not runtime:
        return {"passed": False, "verified": False, "reason": "Container CLI unavailable"}
    try:
        inspected = subprocess.run([runtime, "image", "inspect", image],
                                   capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {"passed": False, "verified": False, "reason": "Image inspection unavailable"}
    if inspected.returncode:
        return {"passed": False, "verified": False, "reason": "Required local container image missing"}
    runner = Path(__file__).with_name("repo_case_runner.py").read_bytes()
    with tempfile.TemporaryDirectory(prefix="gemma-rcl-repo-") as folder:
        root = Path(folder)
        (root / "repo_case_runner.py").write_bytes(runner)
        tree = root / "repository"
        for name, contents in {**task["files"], **changes}.items():
            destination = tree / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(contents, encoding="utf-8")
        container = "gemma-rcl-repo-" + rcl.digest(str(root))[:12]
        failures = []
        passed_cases = []
        started = time.monotonic()
        for index, case in enumerate(task["cases"]):
            remaining = seconds - (time.monotonic() - started)
            if remaining <= 0:
                return {"verified": False, "passed": False,
                        "reason": "Global repository verifier timeout", "test_count": index}
            cmd = [
                runtime, "run", "-i", "--rm", "--pull=never", "--name", container,
                "--network=none", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--pids-limit=64",
                "--memory=384m", "--cpus=1", "--user=65534:65534",
                "--tmpfs=/tmp:rw,nosuid,noexec,size=16m",
                "--mount", f"type=bind,source={root},target=/app,readonly",
                "--workdir=/app", image, "python", "-I", "-B", "/app/repo_case_runner.py",
            ]
            inputs = {"entry": case["entry"], "args": case.get("args", []),
                      "kwargs": case.get("kwargs", {})}
            try:
                executed = rcl.run_bounded(cmd, json.dumps(inputs, allow_nan=False),
                                           timeout=min(remaining, 12))
            except (rcl.OutputLimitExceeded, subprocess.TimeoutExpired, OSError) as exc:
                subprocess.run([runtime, "rm", "-f", container], capture_output=True, timeout=10)
                return {"verified": False, "passed": False,
                        "reason": f"Repository sandbox unavailable at case {index}: {type(exc).__name__}"}
            try:
                result = json.loads(executed.stdout)
                if not isinstance(result, dict) or result.get("kind") not in ("value", "exception", "unsupported"):
                    raise ValueError("Bad sandbox protocol")
            except (ValueError, TypeError):
                return {"verified": False, "passed": False,
                        "reason": f"Repository sandbox protocol failure at case {index}"}
            if executed.returncode:
                return {"verified": False, "passed": False,
                        "reason": f"Repository sandbox exited abnormally at case {index}"}
            if "raises" in case:
                passed = result.get("kind") == "exception" and result.get("type") == case["raises"]
            else:
                passed = (result.get("kind") == "value"
                          and result.get("value_type") == type(case["expected"]).__name__
                          and json.dumps(result.get("value"), allow_nan=False, sort_keys=True)
                          == json.dumps(case["expected"], allow_nan=False, sort_keys=True))
            if passed and case.get("unchanged_args"):
                passed = (
                    json.dumps(result.get("args_after"), sort_keys=True, allow_nan=False)
                    == json.dumps(case.get("args", []), sort_keys=True, allow_nan=False)
                    and json.dumps(result.get("kwargs_after"), sort_keys=True, allow_nan=False)
                    == json.dumps(case.get("kwargs", {}), sort_keys=True, allow_nan=False)
                )
            passed_cases.append(passed)
            if not passed:
                failures.append({"case": index, "purpose": case["purpose"]})
        return {"verified": True, "passed": all(passed_cases),
                "test_count": len(passed_cases), "per_case_pass": passed_cases,
                "reason": "passed" if not failures else "repository hidden cases failed",
                "failures": failures[:8]}


def repo_verifier(task: dict, completion: str, *, baseline: bool = False,
                  image: str = "python:3.11-slim", seconds: int = 90) -> dict:
    if baseline:
        changes = {}
    else:
        try:
            changes = parse_patch(completion, task)
        except (ValueError, SyntaxError) as exc:
            # Invalid syntax is a model failure; no untrusted payload is executed.
            return {
                "verified": True, "passed": False, "reason": f"Invalid patch: {exc}",
                "test_count": 0, "per_case_pass": [],
                "failures": [{"case": "parse", "reason": "invalid_patch"}],
            }
    return _run_project(task, changes, image=image, seconds=seconds)


def prove_baseline(task: dict, *, image: str = "python:3.11-slim") -> dict:
    result = repo_verifier(task, "", baseline=True, image=image)
    if not result.get("verified"):
        raise RuntimeError(f"Repository baseline cannot be verified for {task['id']}: {result['reason']}")
    cases = task["cases"]
    observed = result["per_case_pass"]
    if any(not passed for passed, case in zip(observed, cases, strict=True)
           if case["purpose"] == "regression"):
        raise ValueError(f"Baseline breaks protected regression tests: {task['id']}")
    if not any(not passed for passed, case in zip(observed, cases, strict=True)
               if case["purpose"] == "fix"):
        raise ValueError(f"Baseline has no demonstrated fail-to-pass defect: {task['id']}")
    return result


def ollama_repo_completion(endpoint: str, model: str, prompt: str, seed: int,
                           *, temperature: float = 0.8, top_p: float = 0.95,
                           max_tokens: int = 4096, timeout: int = 240) -> str:
    if not re.fullmatch(r"http://(?:127\.0\.0\.1|localhost):\d{2,5}/api/chat", endpoint):
        raise ValueError("Endpoint must be loopback Ollama /api/chat")
    data = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": prompt}],
                       "options": {"temperature": temperature, "top_p": top_p,
                                   "seed": seed, "num_predict": max_tokens}}).encode()
    req = request.Request(endpoint, data=data, headers={"Content-Type": "application/json"},
                          method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        answer = json.load(response)
    if (not isinstance(answer, dict) or answer.get("model") != model
            or answer.get("done") is not True):
        raise ValueError("Ollama repository response has wrong model identity or is incomplete")
    value = answer.get("message", {}).get("content")
    if not isinstance(value, str):
        raise ValueError("Missing model completion")
    return value


def _check_repo_provenance(row: dict, task: dict) -> tuple[str, str, tuple, int]:
    """Validate a complete, typed and internally consistent candidate receipt."""
    if row.get("source_task_sha256") != rcl.task_digest(task):
        raise ValueError("Repository individual task/hidden-case digest mismatch")
    digest = row.get("model_digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Missing or invalid repository model SHA256 digest")
    origin = row.get("collection_origin")
    if origin not in ("ollama_local", "synthetic_fixture"):
        raise ValueError("Missing or invalid repository collection origin")
    if origin == "ollama_local" and digest == "0" * 64:
        raise ValueError("Real repository collection cannot claim synthetic model identity")
    if origin == "synthetic_fixture" and digest != "0" * 64:
        raise ValueError("Synthetic repository fixture requires explicit zero model identity")
    cfg = row.get("generation_config")
    if not isinstance(cfg, dict) or set(cfg) != {
            "temperature", "top_p", "num_predict", "seed"}:
        raise ValueError("Incomplete repository generation configuration")
    temperature, top_p = cfg["temperature"], cfg["top_p"]
    budget, seed = cfg["num_predict"], cfg["seed"]
    if (type(temperature) not in (int, float)
            or not math.isfinite(temperature) or not 0 <= temperature <= 2
            or type(top_p) not in (int, float)
            or not math.isfinite(top_p) or not 0 < top_p <= 1
            or type(budget) is not int or not 64 <= budget <= 4096
            or type(seed) is not int or seed < 0):
        raise ValueError("Invalid repository generation settings")
    timeout = row.get("generation_timeout_s")
    if type(timeout) is not int or not 15 <= timeout <= 240:
        raise ValueError("Invalid repository generation timeout")
    for field in ("generation_seconds", "verifier_seconds"):
        value = row.get(field)
        if (type(value) not in (float, int) or
                not math.isfinite(value) or not 0 < value <= 3600):
            raise ValueError(f"Missing or invalid repository timing: {field}")
    timestamp = row.get("observed_utc")
    try:
        parsed = datetime.fromisoformat(timestamp)
        if parsed.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("not UTC")
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError("Missing or invalid repository UTC timestamp") from exc
    verdict = row["verdict"]
    count, per_case, failures = (
        verdict.get("test_count"), verdict.get("per_case_pass"),
        verdict.get("failures"),
    )
    reason = verdict.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError("Repository verdict requires a failure/success reason")
    if reason.startswith("Invalid patch:"):
        if (verdict["passed"] is not False or type(count) is not int or count != 0
                or per_case != [] or failures != [
                    {"case": "parse", "reason": "invalid_patch"}]):
            raise ValueError("Invalid patch lacks authenticated parser-failure evidence")
    elif (
        type(count) is not int or count != len(task["cases"])
        or not isinstance(per_case, list) or len(per_case) != count
        or any(type(value) is not bool for value in per_case)
        or not isinstance(failures, list)
        or len(failures) != min(8, per_case.count(False))
        or verdict["passed"] is not all(per_case)
        or (reason == "passed") is not verdict["passed"]
    ):
        raise ValueError("Repository verdict lacks complete per-case replay evidence")
    if not reason.startswith("Invalid patch:"):
        expected_failures = [
            {"case": index, "purpose": case["purpose"]}
            for index, (case, success) in enumerate(zip(
                task["cases"], per_case, strict=True)) if not success
        ]
        if failures != expected_failures[:8]:
            # Failures may be capped to eight in the verifier, but no other
            # case index or purpose can be asserted by untrusted evidence.
            raise ValueError("Repository verdict case failures disagree with outcomes")
    return digest, origin, (float(temperature), float(top_p), budget, timeout), seed


def checked_repo_attempts(task_path: Path, attempts_path: Path, *,
                          reverify: bool = False, image: str = "python:3.11-slim") -> list[dict]:
    known = {task["id"]: task for task in load_repo_tasks(task_path)}
    frozen_tasks_sha256 = hashlib.sha256(task_path.read_bytes()).hexdigest()
    seen: set[tuple[str, str, int]] = set()
    records = rcl.jsonl(attempts_path)
    if not records:
        raise ValueError("No repository attempts")
    checked_baselines: set[str] = set()
    digest_set: set[str] = set()
    origin_set: set[str] = set()
    settings_set: set[tuple] = set()
    seeds_by_attempt: dict[int, int] = {}
    for row in records:
        task = known.get(row.get("task_id"))
        if task is None:
            raise ValueError("Unknown repository task")
        for key in ("split", "license", "source", "prompt", "repo_family"):
            if row.get(key) != task[key]:
                raise ValueError(f"Repository provenance mismatch: {key}")
        if row.get("kind") != "repository":
            raise ValueError("Candidate is not repository coding task")
        if row.get("source_tasks_sha256") != frozen_tasks_sha256:
            raise ValueError("Repository task snapshot/test set changed after candidate generation")
        if row.get("prompt_sha256") != rcl.digest(rcl.canonical_prompt(task["prompt"])):
            raise ValueError("Repository prompt hash changed")
        if row.get("model_prompt_sha256") != rcl.digest(repo_prompt(task)):
            raise ValueError("Repository snapshot/instruction prompt hash changed")
        if not isinstance(row.get("completion"), str) or (
                row.get("completion_sha256") != rcl.digest(row["completion"])):
            raise ValueError("Repository completion hash mismatch")
        verdict = row.get("verdict")
        if (not isinstance(verdict, dict) or verdict.get("verified") is not True or
                type(verdict.get("passed")) is not bool or type(row.get("reward")) is not int or
                row["reward"] != int(verdict["passed"])):
            raise ValueError("Invalid or unverified repository reward")
        if not isinstance(row.get("model"), str) or not row["model"] or (
                type(row.get("attempt")) is not int or row["attempt"] < 0):
            raise ValueError("Invalid repository attempt identity")
        digest, origin, settings, seed = _check_repo_provenance(row, task)
        digest_set.add(digest)
        origin_set.add(origin)
        settings_set.add(settings)
        previous_seed = seeds_by_attempt.setdefault(row["attempt"], seed)
        if previous_seed != seed:
            raise ValueError("Repository seed schedule changes between tasks")
        identity = (task["id"], row["model"], row["attempt"])
        if identity in seen:
            raise ValueError("Duplicate repository attempt")
        seen.add(identity)
        if reverify:
            if task["id"] not in checked_baselines:
                prove_baseline(task, image=image)
                checked_baselines.add(task["id"])
            observed = repo_verifier(task, row["completion"], image=image)
            if observed.get("verified") is not True:
                raise ValueError(f"Repository reward replay unavailable: {observed.get('reason')}")
            if observed != verdict:
                raise ValueError(f"Repository complete reward replay disagrees: {task['id']}")
    if len(digest_set) != 1 or len(origin_set) != 1 or len(settings_set) != 1:
        raise ValueError("Mixed repository model digests, origins or generation budgets")
    rcl.validate_sampling_arm(records)
    return records


def derive_repo_training_rows(rows: list[dict], task_map: dict[str, dict]
                              ) -> tuple[list[dict], list[dict], int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            groups[row["task_id"]].append(row)
    sft: list[dict] = []
    dpo: list[dict] = []
    for _, candidates in sorted(groups.items()):
        positives = sorted((r for r in candidates if r["reward"] == 1),
                           key=lambda x: x["attempt"])
        negatives = sorted((r for r in candidates if r["reward"] == 0),
                           key=lambda x: x["attempt"])
        if not positives:
            continue
        chosen = positives[0]
        full_prompt = repo_prompt(task_map[chosen["task_id"]])
        sft.append({"prompt": full_prompt, "completion": chosen["completion"]})
        if negatives:
            dpo.append({"prompt": full_prompt,
                        "chosen": chosen["completion"], "rejected": negatives[0]["completion"]})
    return sft, dpo, len(groups)


def export_repo(tasks: Path, attempts: Path, destination: Path, *,
                image: str = "python:3.11-slim") -> dict:
    rows = checked_repo_attempts(tasks, attempts, reverify=True, image=image)
    task_map = {t["id"]: t for t in load_repo_tasks(tasks)}
    sft, dpo, train_count = derive_repo_training_rows(rows, task_map)
    if not sft:
        raise ValueError("No successful, independently verified repository fixes")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite repository export {destination}")
    destination.mkdir(parents=True)
    rcl.write_jsonl(destination / "sft.jsonl", sft)
    rcl.write_jsonl(destination / "dpo.jsonl", dpo)
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema": REPO_SCHEMA, "train_tasks_with_pass": len(sft),
                "train_tasks_with_preferences": len(dpo), "train_attempts": train_count,
                "sft_sha256": sha(destination / "sft.jsonl"),
                "dpo_sha256": sha(destination / "dpo.jsonl"),
                "source_attempts_sha256": sha(attempts),
                "source_tasks_sha256": sha(tasks),
                "collection_origin": rows[0]["collection_origin"],
                "model_digest": rows[0]["model_digest"],
                "synthetic_fixture_only": rows[0]["collection_origin"] == "synthetic_fixture"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


def collect_repo(args: argparse.Namespace) -> None:
    tasks = [t for t in load_repo_tasks(args.tasks) if t["split"] == args.split]
    source_tasks_sha256 = hashlib.sha256(args.tasks.read_bytes()).hexdigest()
    if not tasks:
        raise ValueError(f"No repository tasks in split {args.split}")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite repository attempts")
    model_digest = rcl.ollama_model_digest(args.endpoint, args.model)
    temperature = getattr(args, "temperature", 0.8)
    top_p = getattr(args, "top_p", 0.95)
    max_tokens = getattr(args, "max_tokens", 4096)
    timeout = getattr(args, "generation_timeout", 240)
    rows = []
    for task in tasks:
        prove_baseline(task, image=args.image)
        for attempt in range(args.samples):
            generation_started = time.monotonic()
            completion = ollama_repo_completion(
                args.endpoint, args.model, repo_prompt(task), args.seed + attempt,
                temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, timeout=timeout,
            )
            if rcl.ollama_model_digest(args.endpoint, args.model) != model_digest:
                raise RuntimeError("Local Ollama model digest changed during repository collection")
            generation_seconds = time.monotonic() - generation_started
            verification_started = time.monotonic()
            verdict = repo_verifier(task, completion, image=args.image)
            if not verdict["verified"]:
                raise RuntimeError(f"Verifier inconclusive; abort collection: {verdict['reason']}")
            rows.append({"kind": "repository", "task_id": task["id"], "split": task["split"],
                         "source": task["source"], "license": task["license"],
                         "source_tasks_sha256": source_tasks_sha256,
                         "source_task_sha256": rcl.task_digest(task),
                         "repo_family": task["repo_family"], "prompt": task["prompt"],
                         "prompt_sha256": rcl.digest(rcl.canonical_prompt(task["prompt"])),
                         "model_prompt_sha256": rcl.digest(repo_prompt(task)),
                         "model": args.model, "model_digest": model_digest, "attempt": attempt,
                         "collection_origin": "ollama_local",
                         "generation_config": {"temperature": temperature, "top_p": top_p,
                                               "seed": args.seed + attempt,
                                               "num_predict": max_tokens},
                         "generation_timeout_s": timeout,
                         "generation_seconds": max(0.0001, round(generation_seconds, 4)),
                         "verifier_seconds": max(0.0001, round(time.monotonic() - verification_started, 4)),
                         "completion": completion, "completion_sha256": rcl.digest(completion),
                         "reward": int(verdict["passed"]), "verdict": verdict,
                         "observed_utc": datetime.now(timezone.utc).isoformat()})
    rcl.write_jsonl(args.output, rows)
    print(f"Collected {len(rows)} repository attempts; no training performed")


def metrics_repo(args: argparse.Namespace) -> dict:
    rows = checked_repo_attempts(args.tasks, args.attempts, reverify=True, image=args.image)
    from compare_holdouts import summarize_single_run
    try:
        return summarize_single_run(
            load_repo_tasks(args.tasks), rows, split=args.split, k=args.k,
        )
    except ValueError as exc:
        if "Incorrect sample coverage" in str(exc):
            raise ValueError(f"Requested -k={args.k} exceeds repository samples per task") from exc
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--tasks", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--tasks", type=Path, required=True)
    verify.add_argument("--task-id", required=True)
    verify.add_argument("--patch", type=Path)
    verify.add_argument("--baseline", action="store_true")
    collect = commands.add_parser("collect")
    collect.add_argument("--tasks", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--split", choices=sorted(rcl.ALLOWED_SPLITS), default="train")
    collect.add_argument("--samples", type=int, default=4)
    collect.add_argument("--seed", type=int, default=3407)
    collect.add_argument("--temperature", type=float, default=0.8)
    collect.add_argument("--top-p", type=float, default=0.95)
    collect.add_argument("--max-tokens", type=int, default=4096)
    collect.add_argument("--generation-timeout", type=int, default=240)
    collect.add_argument("--model", required=True)
    collect.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    export = commands.add_parser("export")
    export.add_argument("--tasks", type=Path, required=True)
    export.add_argument("--attempts", type=Path, required=True)
    export.add_argument("--out-dir", type=Path, required=True)
    metrics = commands.add_parser("metrics")
    metrics.add_argument("--tasks", type=Path, required=True)
    metrics.add_argument("--attempts", type=Path, required=True)
    metrics.add_argument("--split", choices=sorted(rcl.ALLOWED_SPLITS), default="eval")
    metrics.add_argument("-k", type=int, default=4)
    for cmd in (verify, collect, export, metrics):
        cmd.add_argument("--image", default="python:3.11-slim")
    args = parser.parse_args(argv)
    try:
        if args.action == "validate":
            print(f"VALID: {len(load_repo_tasks(args.tasks))} repository tasks")
        elif args.action == "verify":
            task = next((t for t in load_repo_tasks(args.tasks)
                         if t["id"] == args.task_id), None)
            if task is None:
                raise ValueError("Unknown task ID")
            if args.baseline:
                result = prove_baseline(task, image=args.image)
            else:
                if args.patch is None:
                    raise ValueError("--patch required outside baseline")
                prove_baseline(task, image=args.image)
                result = repo_verifier(task, args.patch.read_text(encoding="utf-8"), image=args.image)
            print(json.dumps(result, indent=2))
            if not result["verified"] or (not args.baseline and not result["passed"]):
                return 2
        elif args.action == "collect":
            if args.samples not in range(1, 17):
                raise ValueError("--samples must be 1..16")
            if not (0 <= args.temperature <= 2 and 0 < args.top_p <= 1
                    and 64 <= args.max_tokens <= 4096
                    and 15 <= args.generation_timeout <= 240):
                raise ValueError("Collection bounds invalid (temperature, top-p, tokens 64..4096, timeout 15..240)")
            collect_repo(args)
        elif args.action == "export":
            print(json.dumps(export_repo(args.tasks, args.attempts, args.out_dir,
                                         image=args.image), indent=2))
        else:
            print(json.dumps(metrics_repo(args), indent=2))
        return 0
    except (ValueError, RuntimeError, FileExistsError, OSError, subprocess.SubprocessError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
