"""Bound task replay, measured admission, and replayed train-only exports."""
import ast
import json
import re

import rcl
from .calibrate import calibrate
from .reward import score


def build_prompt(public):
    parts = ["Fix the defects in this Python repository. Follow its docstrings.",
             "Visible checks pass but do not cover every behavior."]
    for path, source in public["files"].items():
        parts.append(f"### {path}\n" + source)
    parts.append("Visible checks: " + json.dumps(public["visible_tests"], sort_keys=True))
    parts.append("Editable files: " + ", ".join(public["editable"]))
    parts.append('Reply only with JSON: {"files": {"path": "complete replacement source"}}')
    rendered = "\n\n".join(parts)
    if rcl.contains_reserved_marker(rendered):
        raise ValueError("Reserved model role marker in task")
    return rendered


def parse_response(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > 40_000:
        return None
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    try:
        obj = json.loads(text, object_pairs_hook=rcl.unique_json_pairs,
                         parse_constant=rcl.reject_json_constant)
    except ValueError:
        return None
    if not isinstance(obj, dict) or set(obj) != {"files"}:
        return None
    files = obj["files"]
    return files if isinstance(files, dict) and files else None


def task_maps(public_rows, private_rows):
    pub = {row["task_id"]: row for row in public_rows}
    prv = {row["task_id"]: row for row in private_rows}
    if len(pub) != len(public_rows) or len(prv) != len(private_rows) or pub.keys() != prv.keys():
        raise ValueError("Duplicate, missing or unmatched private tasks")
    families, split_owners = {}, {}
    for task, public in pub.items():
        private = prv[task]
        split = public["split"]
        if (split not in {"train", "eval"} or private["project"] != public["project"]
                or families.setdefault(public["project"], split) != split):
            raise ValueError("Repository family crosses splits")
        source_sha = rcl.digest(json.dumps(public["files"], sort_keys=True))
        cases_sha = rcl.digest(json.dumps(private["hidden_f2p"] + private["hidden_p2p"], sort_keys=True))
        if source_sha != private["source_sha"] or cases_sha != private["case_sha"]:
            raise ValueError("Frozen source or case hash mismatch")
        marks = [("source", ast.dump(ast.parse(source), include_attributes=False))
                 for source in public["files"].values() if source.strip()]
        marks += [("case", json.dumps(case, sort_keys=True)) for case in
                  public["visible_tests"] + private["hidden_f2p"] + private["hidden_p2p"]]
        for mark in marks:
            if split_owners.setdefault(mark, split) != split:
                raise ValueError("Source or test case reused across splits")
    return pub, prv


def replay(public_rows, private_rows, attempts):
    pub, prv = task_maps(public_rows, private_rows)
    out, seen = [], set()
    settings, seeds, identities = set(), {}, {}
    for attempt in attempts:
        task = attempt.get("task_id")
        model = attempt.get("model")
        sample = attempt.get("sample")
        digest = attempt.get("model_digest", "")
        config = attempt.get("generation_config", {})
        if (task not in pub or not isinstance(model, str) or not model.strip()
                or type(sample) is not int or sample < 0
                or (task, model, sample) in seen
                or not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)
                or digest == "0" * 64 or not isinstance(config, dict)
                or set(config) != {"temperature", "top_p", "num_predict", "seed"}
                or type(config["seed"]) is not int
                or type(config["num_predict"]) is not int or not 1 <= config["num_predict"] <= 32768
                or type(config["temperature"]) not in (int, float)
                or not 0 <= config["temperature"] <= 2
                or type(config["top_p"]) not in (int, float) or not 0 < config["top_p"] <= 1
                or attempt.get("prompt_sha256") != rcl.digest(build_prompt(pub[task]))):
            raise ValueError("Invalid, duplicate or unbound attempt metadata")
        seen.add((task, model, sample))
        if identities.setdefault(model, digest) != digest:
            raise ValueError("Model digest changed within an arm")
        settings.add((config["temperature"], config["top_p"], config["num_predict"]))
        if len(settings) > 1 or seeds.setdefault(sample, config["seed"]) != config["seed"]:
            raise ValueError("Unmatched decoding settings or seed schedule")
        response = attempt.get("response")
        files = parse_response(response)
        result = score(pub[task], prv[task], files) if files else {
            "verified": False, "passed": False, "f2p_frac": 0.0,
            "reward": 0.0, "reason": "unparseable"}
        out.append({"task_id": task, "model": model, "model_digest": digest,
                    "sample": sample, "response": response,
                    "task_sha256": rcl.task_digest({"public": pub[task], "private": prv[task]}),
                    **result})
    if len(set(identities.values())) != len(identities):
        raise ValueError("Distinct model labels cannot share one declared model digest")
    return out


def export(public_rows, private_rows, attempts, frontier_attempts, frontier, student):
    """Ignore saved rewards; replay both arms and calibrate before admitting data."""
    if not public_rows or any(t.get("split") != "train" for t in public_rows):
        raise ValueError("Research export accepts train-split tasks only")
    if any(a.get("model") != student for a in attempts):
        raise ValueError("Training attempts must all be from the declared student")
    if any(a.get("model") != frontier for a in frontier_attempts):
        raise ValueError("Frontier attempts do not match the declared frontier")
    results = replay(public_rows, private_rows, attempts + frontier_attempts)
    buckets = calibrate(results, frontier, student, [t["task_id"] for t in public_rows])
    admitted = {task for ids in buckets["learnable"].values() for task in ids}
    sft, dpo = [], []
    for public in public_rows:
        task = public["task_id"]
        if public["split"] != "train" or task not in admitted:
            continue
        rows = [r for r in results if r["task_id"] == task and r["model"] == student]
        good = list(dict.fromkeys(r["response"] for r in rows if r["verified"] and r["passed"]))
        bad = list(dict.fromkeys(r["response"] for r in rows if r["verified"] and not r["passed"]
                                and r["reason"] in {"hidden_fail", "regression"}))
        prompt = build_prompt(public)
        sft.extend({"prompt": prompt, "completion": response} for response in good[:2])
        if good and bad:
            dpo.append({"prompt": prompt, "chosen": good[0], "rejected": bad[0]})
    return sft, dpo

