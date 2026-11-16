"""Local Reinforcement Coding Loop: rollout, execute, select, and measure."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from urllib import request

MAX_CODE_BYTES = 40_000
MAX_TEST_BYTES = 20_000
MAX_OUTPUT_BYTES = 8_000
ALLOWED_SPLITS = {"train", "validation", "eval"}
RESERVED_CODE_MARKERS = (
    "<|", "|>", "<bos>", "<eos>", "<pad>",
    "<start_of_turn>", "<end_of_turn>",
)


def contains_reserved_marker(value: str) -> bool:
    """Block Gemma role/tool delimiters, including <turn|> and <channel|>."""
    return any(marker in value for marker in RESERVED_CODE_MARKERS)
CODING_INSTRUCTION = (
    "Return only a complete Python module. Define the requested function. "
    "Do not include prose, markdown, tests, or external packages.\n\n"
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_digest(task: dict) -> str:
    """Bind attempts to the entire private task, including every hidden case."""
    return digest(json.dumps(task, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False))


def unique_json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON key in source data: {name}")
        result[name] = value
    return result


def reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant forbidden: {value}")


def canonical_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def training_prompt(value: str) -> str:
    return CODING_INSTRUCTION + value


def name_from_prompt(value: str) -> str | None:
    match = re.search(r"\b(?:def|function)\s+([A-Za-z_]\w*)\s*\(", value, re.I)
    if match is None:
        match = re.search(r"\b(?:implement|write)\s+(?:def\s+)?([A-Za-z_]\w*)\s*\(", value, re.I)
    return match.group(1).lower() if match else None


def prompt_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_]{2,}", canonical_prompt(value)))


def case_signature(case: dict) -> str:
    """Detect reused tests even when a task's function name is paraphrased."""
    canonical = {name: value for name, value in case.items() if name != "fn"}
    return json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing JSONL file: {path}")
    records = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw, object_pairs_hook=unique_json_pairs,
                             parse_constant=reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        records.append(obj)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def load_tasks(path: Path) -> list[dict]:
    records = jsonl(path)
    if not records:
        raise ValueError("Task file is empty")
    ids = set()
    prompt_hash_split = {}
    function_names = {}
    seen_tokens = []
    seen_cases = []
    for item in records:
        required = ("id", "prompt", "split", "source", "license")
        if any(not isinstance(item.get(field), str) or not item[field].strip() for field in required):
            raise ValueError(f"Task must have nonempty string fields: {required}")
        if contains_reserved_marker(item["prompt"]):
            raise ValueError(f"Task prompt contains a reserved Gemma chat-template token: {item['id']}")
        if item["id"] in ids:
            raise ValueError(f"Repeated task id: {item['id']}")
        if item["split"] not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split for {item['id']}")
        if item.get("language", "python") != "python":
            raise ValueError(f"Only Python supported: {item['id']}")
        cases = item.get("cases")
        if not isinstance(cases, list) or not 2 <= len(cases) <= 64:
            raise ValueError(f"Task needs 2..64 hidden executable cases: {item['id']}")
        if len(json.dumps(cases).encode("utf-8")) > MAX_TEST_BYTES:
            raise ValueError(f"Cases too large: {item['id']}")
        for case in cases:
            if not isinstance(case, dict) or not re.fullmatch(r"[A-Za-z_]\w*", str(case.get("fn", ""))):
                raise ValueError(f"Invalid function name in cases: {item['id']}")
            if not isinstance(case.get("args", []), list) or not isinstance(case.get("kwargs", {}), dict):
                raise ValueError(f"Cases require JSON args/kwargs: {item['id']}")
            if ("expected" in case) == ("raises" in case):
                raise ValueError(f"Each case must contain exactly one expected or raises: {item['id']}")
            if "raises" in case and case["raises"] not in ("ValueError", "TypeError", "KeyError", "IndexError"):
                raise ValueError(f"Unsupported exception case: {item['id']}")
        normalized_cases = {case_signature(case) for case in cases}
        for previous_id, previous_split, earlier_cases in seen_cases:
            shared = len(normalized_cases & earlier_cases)
            minimum = min(len(normalized_cases), len(earlier_cases))
            if previous_split != item["split"] and minimum >= 2 and shared / minimum >= 0.85:
                raise ValueError(
                    f"Reused test cases across data splits: {previous_id} / {item['id']}"
                )
        seen_cases.append((item["id"], item["split"], normalized_cases))
        if item["license"].lower() in {"unknown", "unlicensed", "none"}:
            raise ValueError(f"Rights unverified: {item['id']}")
        ph = digest(canonical_prompt(item["prompt"]))
        previous = prompt_hash_split.get(ph)
        if previous is not None:
            raise ValueError(f"Duplicate/contaminated task prompt across splits: {item['id']}")
        function_name = name_from_prompt(item["prompt"])
        if function_name and function_name in function_names:
            raise ValueError(f"Duplicate/contaminated function {function_name}: {item['id']}")
        tokens = prompt_tokens(item["prompt"])
        for earlier, old_tokens in seen_tokens:
            similarity = len(tokens & old_tokens) / max(len(tokens | old_tokens), 1)
            if similarity >= 0.75:
                raise ValueError(f"Near-duplicate task prompt: {earlier} / {item['id']}")
        seen_tokens.append((item["id"], tokens))
        if function_name:
            function_names[function_name] = item["id"]
        prompt_hash_split[ph] = item["split"]
        ids.add(item["id"])
    return records


def extract_code(raw: str) -> str:
    raw = raw.strip()
    if contains_reserved_marker(raw):
        raise ValueError("Generated code contains a reserved Gemma control marker")
    fenced = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", raw, flags=re.S | re.I)
    if fenced:
        raw = fenced.group(1).strip()
    if not raw or "```" in raw or len(raw.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError("Response is empty, mixes prose/fences, or exceeds maximum code size")
    ast.parse(raw)
    return raw + "\n"


def container_runtime() -> str | None:
    """Select a native container CLI. Windows Bash shims are not Windows executables."""
    commands = ("docker.exe", "podman.exe") if os.name == "nt" else ("docker", "podman")
    for executable in commands:
        found = shutil.which(executable)
        if found:
            return found
    return None


class OutputLimitExceeded(RuntimeError):
    pass


def run_bounded(command: list[str], payload: str, *, timeout: float) -> subprocess.CompletedProcess:
    """Bound each subprocess pipe independently to MAX_OUTPUT_BYTES.

    Reader and writer threads prevent a container from stalling by filling
    stdout/stderr while the host is supplying input. The subprocess is killed
    when either stream exceeds the limit, and its output is never retained
    without a fixed bound.
    """
    captured: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def drain(stream, kind: str) -> None:
        try:
            while True:
                data = stream.read(4096)
                if not data:
                    break
                remaining = MAX_OUTPUT_BYTES - len(captured[kind])
                captured[kind].extend(data[:max(0, remaining)])
                if len(data) > remaining:
                    overflow.set()
                    process.kill()
                    break
        except (OSError, ValueError):
            pass
        finally:
            stream.close()

    def send() -> None:
        try:
            process.stdin.write(payload.encode("utf-8"))
            process.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            process.stdin.close()

    threads = [
        threading.Thread(target=drain, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, "stderr"), daemon=True),
        threading.Thread(target=send, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        raise
    finally:
        for thread in threads:
            thread.join(timeout=5)
    if overflow.is_set():
        raise OutputLimitExceeded(f"Container output exceeds {MAX_OUTPUT_BYTES} bytes per stream")
    return subprocess.CompletedProcess(
        command, process.returncode,
        stdout=captured["stdout"].decode("utf-8", errors="replace"),
        stderr=captured["stderr"].decode("utf-8", errors="replace"),
    )


def docker_verifier(code: str, cases: list[dict], *, image: str = "python:3.11-slim", seconds: int = 90) -> dict:
    """Compare on the HOST; each isolated Docker call sees input but no expected output."""
    runtime = container_runtime()
    if runtime is None:
        return {"passed": False, "verified": False, "reason": "Native Docker/Podman CLI unavailable"}
    inspection = subprocess.run([runtime, "image", "inspect", image], capture_output=True, text=True, timeout=10)
    if inspection.returncode:
        return {"passed": False, "verified": False, "reason": f"Docker image {image} is not local; --pull is disabled"}
    try:
        code = extract_code(code)
    except (ValueError, SyntaxError) as exc:
        return {"passed": False, "verified": True, "reason": f"Invalid Python: {exc}"}
    try:
        # Unrestricted Python can corrupt its own result protocol and forge
        # a reward; V0 only evaluates the same restricted pure-Python subset
        # as C3 until an independently attested oracle exists.
        from repo_rcl import require_restricted_source
        require_restricted_source(code, "solution.py", {"solution.py": code})
    except (ValueError, SyntaxError) as exc:
        return {"passed": False, "verified": True,
                "reason": f"Unreviewed Python execution rejected: {exc}"}
    if not cases:
        return {"passed": False, "verified": False, "reason": "No hidden cases"}
    with tempfile.TemporaryDirectory(prefix="gemma-rcl-") as temp:
        root = Path(temp)
        (root / "solution.py").write_text(code, encoding="utf-8")
        (root / "oracle_runner.py").write_bytes(Path(__file__).with_name("oracle_runner.py").read_bytes())
        container = "gemma-rcl-" + digest(str(root))[:12]
        failures = []
        start = time.monotonic()
        for number, case in enumerate(cases):
            remaining = seconds - (time.monotonic() - start)
            if remaining <= 0:
                return {"passed": False, "verified": False, "reason": "Global verifier timeout",
                        "test_count": number}
            cmd = [
                runtime, "run", "-i", "--rm", "--pull=never", "--name", container,
                "--network=none", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--pids-limit=64",
                "--memory=384m", "--cpus=1", "--user=65534:65534",
                "--tmpfs=/tmp:rw,nosuid,noexec,size=16m",
                "--mount", f"type=bind,source={root},target=/app,readonly",
                "--workdir=/app", image, "python", "-I", "-B", "/app/oracle_runner.py",
            ]
            inputs = {"fn": case["fn"], "args": case.get("args", []), "kwargs": case.get("kwargs", {})}
            try:
                run = run_bounded(cmd, json.dumps(inputs), timeout=min(remaining, 12))
            except subprocess.TimeoutExpired:
                subprocess.run([runtime, "rm", "-f", container], capture_output=True, timeout=10)
                return {"passed": False, "verified": False,
                        "reason": f"Sandbox execution timeout at case {number}", "test_count": number}
            except OutputLimitExceeded:
                subprocess.run([runtime, "rm", "-f", container], capture_output=True, timeout=10)
                return {"passed": False, "verified": False,
                        "reason": f"Sandbox output exceeded capture limit at case {number}",
                        "test_count": number}
            try:
                result = json.loads(run.stdout)
                if not isinstance(result, dict):
                    raise ValueError("Expected JSON result object")
            except ValueError:
                return {"passed": False, "verified": False, "reason": "Sandbox output protocol unavailable",
                        "output": (run.stdout + run.stderr)[-MAX_OUTPUT_BYTES:]}
            if run.returncode != 0 or result.get("kind") not in ("exception", "value", "unsupported"):
                return {"passed": False, "verified": False, "reason": "Sandbox execution failed"}
            if "raises" in case:
                passed_case = result.get("kind") == "exception" and result.get("type") == case["raises"]
            else:
                passed_case = result.get("kind") == "value" and (
                    result.get("value_type") == type(case["expected"]).__name__
                    and
                    json.dumps(result.get("value"), sort_keys=True, allow_nan=False)
                    == json.dumps(case["expected"], sort_keys=True, allow_nan=False)
                )
            if passed_case and case.get("unchanged_args"):
                passed_case = (
                    json.dumps(result.get("args_after"), sort_keys=True, allow_nan=False)
                    == json.dumps(case.get("args", []), sort_keys=True, allow_nan=False)
                    and json.dumps(result.get("kwargs_after"), sort_keys=True, allow_nan=False)
                    == json.dumps(case.get("kwargs", {}), sort_keys=True, allow_nan=False)
                )
            if not passed_case:
                failures.append({"case": number, "reason": "Incorrect output, exception, or input mutation"})
        return {"passed": not failures, "verified": True, "test_count": len(cases),
                "reason": "passed" if not failures else "hidden cases failed", "failures": failures[:8]}


def ollama_model_digest(endpoint: str, model: str) -> str:
    """Resolve the actual locally installed Ollama model identity at collection time."""
    if not re.fullmatch(r"http://(?:127\.0\.0\.1|localhost):\d{2,5}/api/chat", endpoint):
        raise ValueError("Endpoint must be a loopback Ollama /api/chat URL")
    tags_url = endpoint.removesuffix("/api/chat") + "/api/tags"
    with request.urlopen(tags_url, timeout=8) as response:
        if response.status != 200:
            raise ValueError("Local Ollama model metadata unavailable")
        payload = response.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("Ollama model metadata exceeds bounded size")
    models = json.loads(payload).get("models")
    if not isinstance(models, list):
        raise ValueError("Malformed Ollama model inventory")
    matches = [m.get("digest") for m in models
               if isinstance(m, dict) and m.get("name") == model]
    if len(matches) != 1 or not isinstance(matches[0], str) or (
            re.fullmatch(r"[a-f0-9]{64}", matches[0]) is None):
        raise ValueError(f"Model {model} is not uniquely identified by local Ollama digest")
    return matches[0]


def local_ollama_completion(endpoint: str, model: str, prompt: str, seed: int,
                            *, temperature: float = 0.8, top_p: float = 0.95,
                            max_tokens: int = 2048, timeout: int = 240) -> str:
    if not re.fullmatch(r"http://(?:127\.0\.0\.1|localhost):\d{2,5}/api/chat", endpoint):
        raise ValueError("Endpoint must be a loopback Ollama /api/chat URL")
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": training_prompt(prompt)}],
                       "options": {"temperature": temperature, "top_p": top_p,
                                   "seed": seed, "num_predict": max_tokens}}).encode()
    req = request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        answer = json.load(response)
    if not isinstance(answer, dict) or answer.get("model") != model or answer.get("done") is not True:
        raise ValueError("Ollama response lacks matching model identity or completion status")
    result = answer.get("message", {}).get("content")
    if not isinstance(result, str):
        raise ValueError("Ollama returned no message content")
    return result


def collect(args: argparse.Namespace) -> None:
    tasks = [t for t in load_tasks(args.tasks) if t["split"] == args.split]
    if not tasks:
        raise ValueError(f"No tasks found in split {args.split}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if container_runtime() is None:
        raise RuntimeError("Native Docker or Podman with a preloaded Python image is required")
    model_digest = ollama_model_digest(args.endpoint, args.model)
    source_tasks_sha256 = file_digest(args.tasks)
    temperature = getattr(args, "temperature", 0.8)
    top_p = getattr(args, "top_p", 0.95)
    max_tokens = getattr(args, "max_tokens", 2048)
    timeout = getattr(args, "generation_timeout", 240)
    records = []
    for task in tasks:
        for attempt in range(args.samples):
            started = time.monotonic()
            raw = local_ollama_completion(
                args.endpoint, args.model, task["prompt"], args.seed + attempt,
                temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, timeout=timeout,
            )
            if ollama_model_digest(args.endpoint, args.model) != model_digest:
                raise RuntimeError("Local Ollama model digest changed while collecting attempts")
            generation_seconds = time.monotonic() - started
            verification_started = time.monotonic()
            try:
                completion = extract_code(raw)
                verdict = docker_verifier(completion, task["cases"], image=args.image)
            except (ValueError, SyntaxError) as exc:
                completion = raw[:MAX_CODE_BYTES]
                verdict = {"passed": False, "verified": True, "reason": f"Invalid Python: {exc}"}
            if not verdict["verified"]:
                raise RuntimeError(f"Verifier unavailable for {task['id']}: {verdict['reason']}")
            records.append({
                "task_id": task["id"], "split": task["split"], "source": task["source"],
                "source_tasks_sha256": source_tasks_sha256,
                "source_task_sha256": task_digest(task),
                "license": task["license"], "prompt": task["prompt"],
                "prompt_sha256": digest(canonical_prompt(task["prompt"])),
                "model_prompt_sha256": digest(training_prompt(task["prompt"])),
                "model": args.model, "model_digest": model_digest, "attempt": attempt,
                "collection_origin": "ollama_local",
                "generation_config": {"temperature": temperature, "top_p": top_p,
                                      "seed": args.seed + attempt, "num_predict": max_tokens},
                "generation_timeout_s": timeout,
                "generation_seconds": round(generation_seconds, 4),
                "verifier_seconds": round(time.monotonic() - verification_started, 4),
                "completion": completion, "completion_sha256": digest(completion),
                "reward": 1 if verdict["passed"] else 0, "verdict": verdict,
                "observed_utc": datetime.now(timezone.utc).isoformat(),
            })
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} verified attempts to {args.output}")


def validate_sampling_arm(records: list[dict]) -> str:
    """Require a single model and complete, equal per-task attempt coverage.

    Mixed models, missing attempt indices or unequal sample counts silently
    bias pass@1 / observed solve-within-k and training pair selection.
    """
    model_ids = {row["model"] for row in records}
    if len(model_ids) != 1 or any(not model for model in model_ids):
        raise ValueError("Attempts mix model identities; separate evaluation arms")
    groups: dict[tuple[str, str], set[int]] = {}
    for row in records:
        key = (row["split"], row["task_id"])
        attempts = groups.setdefault(key, set())
        if row["attempt"] in attempts:
            raise ValueError(f"Duplicate sampling attempt: {key}")
        attempts.add(row["attempt"])
    counts: dict[str, set[int]] = {}
    for (split, task_id), indices in groups.items():
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"Missing/noncontiguous sampling attempts: {task_id}")
        counts.setdefault(split, set()).add(len(indices))
    if any(len(values) != 1 for values in counts.values()):
        raise ValueError("Unequal sampling attempt coverage within one split")
    return next(iter(model_ids))


def checked_attempts(task_path: Path, attempts_path: Path, *,
                     reverify: bool = False, image: str = "python:3.11-slim") -> list[dict]:
    known = {t["id"]: t for t in load_tasks(task_path)}
    source_tasks_sha256 = file_digest(task_path)
    seen = set()
    accepted = []
    observed_model_digests = set()
    observed_origins = set()
    sampling_settings = set()
    attempt_seeds: dict[int, int] = {}
    for row in jsonl(attempts_path):
        task = known.get(row.get("task_id"))
        if task is None:
            raise ValueError(f"Unknown task in attempts: {row.get('task_id')}")
        if row.get("split") != task["split"] or row.get("source") != task["source"] or row.get("license") != task["license"]:
            raise ValueError(f"Split/source/license mismatch for {task['id']}")
        if (row.get("source_tasks_sha256") != source_tasks_sha256
                or row.get("source_task_sha256") != task_digest(task)):
            raise ValueError(f"Full private task/source digest mismatch for {task['id']}")
        if row.get("prompt") != task["prompt"] or row.get("prompt_sha256") != digest(canonical_prompt(task["prompt"])):
            raise ValueError(f"Prompt changed for {task['id']}")
        if row.get("model_prompt_sha256") != digest(training_prompt(task["prompt"])):
            raise ValueError(f"Rendered model prompt changed for {task['id']}")
        if not isinstance(row.get("completion"), str) or not isinstance(row.get("verdict"), dict):
            raise ValueError(f"Invalid completion/verdict structure for {task['id']}")
        if row.get("completion_sha256") != digest(row["completion"]):
            raise ValueError(f"Completion digest mismatch for {task['id']}")
        if type(row.get("reward")) is not int or row["reward"] not in (0, 1) or row["reward"] != int(row["verdict"].get("passed") is True):
            raise ValueError(f"Invalid reward for {task['id']}")
        if row["verdict"].get("verified") is not True or type(row["verdict"].get("passed")) is not bool:
            raise ValueError(f"Unverified reward for {task['id']}")
        if type(row.get("attempt")) is not int or row["attempt"] < 0 or not isinstance(row.get("model"), str):
            raise ValueError(f"Invalid attempt identity for {task['id']}")
        model_digest = row.get("model_digest")
        if not isinstance(model_digest, str) or re.fullmatch(r"[a-f0-9]{64}", model_digest) is None:
            raise ValueError(f"Missing/invalid immutable model digest for {task['id']}")
        origin = row.get("collection_origin")
        if origin not in ("ollama_local", "synthetic_fixture"):
            raise ValueError(f"Unknown or missing candidate origin for {task['id']}")
        if origin == "synthetic_fixture" and model_digest != "0" * 64:
            raise ValueError(f"Synthetic fixture must have an explicit zero model digest for {task['id']}")
        if origin == "ollama_local" and model_digest == "0" * 64:
            raise ValueError(f"Real generation cannot claim synthetic model digest for {task['id']}")
        cfg = row.get("generation_config")
        if not isinstance(cfg, dict) or set(cfg) != {"temperature", "top_p", "num_predict", "seed"}:
            raise ValueError(f"Missing or malformed generation configuration for {task['id']}")
        temp, top_p, budget, seed = (cfg[name] for name in
                                     ("temperature", "top_p", "num_predict", "seed"))
        if (type(temp) not in (int, float) or not math.isfinite(temp) or not 0 <= temp <= 2
                or type(top_p) not in (int, float) or not math.isfinite(top_p) or not 0 < top_p <= 1
                or type(budget) is not int or not 64 <= budget <= 4096
                or type(seed) is not int or seed < 0):
            raise ValueError(f"Invalid generation settings for {task['id']}")
        if row["attempt"] in attempt_seeds and attempt_seeds[row["attempt"]] != seed:
            raise ValueError(f"Seed schedule changed between tasks at attempt {row['attempt']}")
        attempt_seeds[row["attempt"]] = seed
        if origin == "ollama_local":
            timeout_s = row.get("generation_timeout_s")
            if type(timeout_s) is not int or not 15 <= timeout_s <= 240:
                raise ValueError(f"Invalid generation timeout for {task['id']}")
        for timing in ("generation_seconds", "verifier_seconds"):
            elapsed = row.get(timing)
            if (type(elapsed) not in (int, float) or
                    not math.isfinite(elapsed) or elapsed < 0 or elapsed > 3600):
                raise ValueError(f"Invalid {timing} evidence for {task['id']}")
        timestamp = row.get("observed_utc")
        try:
            observed_utc = datetime.fromisoformat(timestamp)
            if observed_utc.utcoffset() != timezone.utc.utcoffset(None):
                raise ValueError("Timestamp is not UTC")
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid observation UTC timestamp for {task['id']}") from exc
        observed_model_digests.add(model_digest)
        observed_origins.add(origin)
        sampling_settings.add((float(temp), float(top_p), budget))
        key = (task["id"], row["attempt"], row["model"])
        if key in seen:
            raise ValueError(f"Duplicate candidate identity for {task['id']}")
        if reverify:
            observed = docker_verifier(row["completion"], task["cases"], image=image)
            if observed.get("verified") is not True:
                raise ValueError(f"Unable to replay verifier for {task['id']}: {observed.get('reason')}")
            if observed != row["verdict"]:
                raise ValueError(f"Complete reward replay disagrees with claimed verdict for {task['id']}")
        seen.add(key)
        accepted.append(row)
    if not accepted:
        raise ValueError("No attempts")
    if len(observed_model_digests) != 1 or len(observed_origins) != 1 or len(sampling_settings) != 1:
        raise ValueError("Mixed model digests, origins or generation settings in one sampling arm")
    validate_sampling_arm(accepted)
    return accepted


def derive_training_rows(records: list[dict]) -> tuple[list[dict], list[dict], int]:
    by_task = {}
    for row in records:
        if row["split"] == "train":
            by_task.setdefault(row["task_id"], []).append(row)
    sft, dpo = [], []
    for task_id, candidates in sorted(by_task.items()):
        positive = sorted((c for c in candidates if c["reward"] == 1), key=lambda c: c["attempt"])
        negative = sorted((c for c in candidates if c["reward"] == 0), key=lambda c: c["attempt"])
        if not positive:
            continue
        sft.append({"prompt": training_prompt(positive[0]["prompt"]),
                    "completion": positive[0]["completion"]})
        if negative:
            dpo.append({"prompt": training_prompt(positive[0]["prompt"]), "chosen": positive[0]["completion"],
                        "rejected": negative[0]["completion"]})
    return sft, dpo, len(by_task)


def export(args: argparse.Namespace) -> None:
    records = checked_attempts(args.tasks, args.attempts, reverify=True,
                               image=getattr(args, "image", "python:3.11-slim"))
    sft, dpo, train_tasks = derive_training_rows(records)
    if not sft:
        raise ValueError("No training tasks have verified successful completions")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    write_jsonl(args.out_dir / "sft.jsonl", sft)
    write_jsonl(args.out_dir / "dpo.jsonl", dpo)
    manifest = {
        "schema": "gemma-coding-rcl-v1", "train_tasks_with_pass": len(sft),
        "train_tasks_with_preferences": len(dpo), "train_attempts": train_tasks,
        "sft_sha256": hashlib.sha256((args.out_dir / "sft.jsonl").read_bytes()).hexdigest(),
        "dpo_sha256": hashlib.sha256((args.out_dir / "dpo.jsonl").read_bytes()).hexdigest(),
        "source_attempts_sha256": hashlib.sha256(args.attempts.read_bytes()).hexdigest(),
        "source_tasks_sha256": file_digest(args.tasks),
        "collection_origin": records[0]["collection_origin"],
        "model_digest": records[0]["model_digest"],
        "synthetic_fixture_only": records[0]["collection_origin"] == "synthetic_fixture",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def metrics(args: argparse.Namespace) -> None:
    records = checked_attempts(args.tasks, args.attempts, reverify=True,
                               image=getattr(args, "image", "python:3.11-slim"))
    from compare_holdouts import summarize_single_run
    try:
        report = summarize_single_run(
            load_tasks(args.tasks), records, split=args.split, k=args.k,
        )
    except ValueError as exc:
        if "Incorrect sample coverage" in str(exc):
            raise ValueError(f"Requested -k={args.k} exceeds observed samples per task") from exc
        raise
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    validation = subs.add_parser("validate", help="Validate task data and split isolation")
    validation.add_argument("--tasks", type=Path, required=True)
    verification = subs.add_parser("verify", help="Run one source file through hidden case verifier")
    verification.add_argument("--tasks", type=Path, required=True)
    verification.add_argument("--task-id", required=True)
    verification.add_argument("--code", type=Path, required=True)
    verification.add_argument("--image", default="python:3.11-slim")
    generation = subs.add_parser("collect", help="Collect and Docker-verify Ollama candidates")
    generation.add_argument("--tasks", type=Path, required=True)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--model", required=True)
    generation.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    generation.add_argument("--image", default="python:3.11-slim")
    generation.add_argument("--split", choices=sorted(ALLOWED_SPLITS), default="train")
    generation.add_argument("--samples", type=int, default=4)
    generation.add_argument("--seed", type=int, default=3407)
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--top-p", type=float, default=0.95)
    generation.add_argument("--max-tokens", type=int, default=2048)
    generation.add_argument("--generation-timeout", type=int, default=240)
    selection = subs.add_parser("export", help="Export verified SFT and preference examples")
    selection.add_argument("--tasks", type=Path, required=True)
    selection.add_argument("--attempts", type=Path, required=True)
    selection.add_argument("--out-dir", type=Path, required=True)
    selection.add_argument("--image", default="python:3.11-slim")
    summary = subs.add_parser("metrics", help="Compute empirical local pass rates")
    summary.add_argument("--tasks", type=Path, required=True)
    summary.add_argument("--attempts", type=Path, required=True)
    summary.add_argument("--split", choices=sorted(ALLOWED_SPLITS), default="eval")
    summary.add_argument("--image", default="python:3.11-slim")
    summary.add_argument("-k", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.action == "validate":
            print(f"VALID: {len(load_tasks(args.tasks))} tasks, disjoint normalized prompts")
        elif args.action == "verify":
            task = next((t for t in load_tasks(args.tasks) if t["id"] == args.task_id), None)
            if task is None:
                raise ValueError(f"No task ID found: {args.task_id}")
            outcome = docker_verifier(args.code.read_text(encoding="utf-8"), task["cases"], image=args.image)
            print(json.dumps(outcome, indent=2))
            if not outcome["verified"] or not outcome["passed"]:
                return 2
        elif args.action == "collect":
            if args.samples < 1 or args.samples > 16:
                raise ValueError("--samples must be 1..16")
            if not (0 <= args.temperature <= 2 and 0 < args.top_p <= 1 and
                    64 <= args.max_tokens <= 4096 and
                    15 <= args.generation_timeout <= 240):
                raise ValueError("Collection bounds invalid (temperature, top-p, tokens 64..4096, timeout 15..240)")
            collect(args)
        elif args.action == "export":
            export(args)
        else:
            metrics(args)
        return 0
    except (ValueError, FileExistsError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
