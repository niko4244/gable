"""Build coupled-defect repair tasks by composing AST mutations.

A composed task is kept only if:
  * reference passes every case (expected values come from the reference);
  * no failing case raises (plausible logic bugs, not crashes);
  * >=1 hidden fail-to-pass case and >=3 passing cases (visible/p2p pool);
  * k>=2: fixing any single defect still leaves a failing case
    (no one-line fix suffices).
"""
import hashlib
import itertools
import json
import os
import random

from .mutate import apply_sites, enumerate_sites

# The authored reference repositories and their expected-result generator are
# private evaluation inputs. They are intentionally absent from public Gable.
# Keep the framework importable without either file.
try:
    from .corpus import PROJECTS
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}.corpus":
        raise
    PROJECTS = None
try:
    from ._trusted import judge, run_cases
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}._trusted":
        raise
    judge = run_cases = None


def _require_private_builder():
    if PROJECTS is None or judge is None or run_cases is None:
        raise RuntimeError(
            "Hard Band task generation requires a locally supplied, reviewed "
            "private hardband.corpus and hardband._trusted. The public package "
            "does not distribute reference solutions or hidden test generators."
        )


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def repo_with(ref_files, muts):
    """Reference files with exactly the given mutations applied."""
    files = dict(ref_files)
    by_file = {}
    for m in muts:
        by_file.setdefault(m["file"], []).append(m["site"])
    for f, sites in by_file.items():
        files[f] = apply_sites(ref_files[f], sites)
    return files


def _status(ref_files, muts, cases):
    files = repo_with(ref_files, muts)
    for src in files.values():
        try:
            compile(src, "<m>", "exec")
        except SyntaxError:
            return None
    return judge(files, cases)


def _clean(st):
    return (st is not None and "error" not in st and "fail" in st
            and st.count("pass") >= 3)


def build_project(name, rng, per_project, ks, attempts=40):
    _require_private_builder()
    proj = PROJECTS[name]
    ref = dict(proj["files"])
    raw = proj["cases"](rng)
    res = run_cases(ref, raw)
    if not all(r["ok"] for r in res):
        raise RuntimeError(f"reference crashes on generated cases: {name}")
    cases = [{**c, "expected": r["value"]} for c, r in zip(raw, res)]

    singles = []
    for f in proj["editable"]:
        for s in enumerate_sites(ref[f]):
            m = {"file": f, "site": s}
            if _clean(_status(ref, [m], cases)):
                singles.append(m)
    tasks = []
    seen = set()
    want = max(1, per_project // len(ks))
    for k in ks:
        got, tries = 0, 0
        combos = list(itertools.combinations(range(len(singles)), k)) \
            if len(singles) <= 40 or k == 1 else None
        while got < want and tries < attempts:
            tries += 1
            if combos is not None:
                if not combos:
                    break
                idxs = combos.pop(rng.randrange(len(combos)))
            else:
                idxs = tuple(sorted(rng.sample(range(len(singles)), k)))
            muts = [singles[i] for i in idxs]
            keys = {(m["file"], m["site"]["idx"]) for m in muts}
            key = tuple(sorted(keys))
            if len(keys) < k or key in seen:
                continue
            st = _status(ref, muts, cases)
            if not _clean(st):
                continue
            if k >= 2:
                ok = True
                for i in range(k):
                    sub = _status(ref, muts[:i] + muts[i + 1:], cases)
                    if sub is None or "fail" not in sub:
                        ok = False
                        break
                if not ok:
                    continue
            seen.add(key)
            order = list(range(len(cases)))
            rng.shuffle(order)
            passing = [i for i in order if st[i] == "pass"]
            visible = passing[:3]
            p2p = passing[3:]
            f2p = [i for i in order if st[i] == "fail"]
            files = repo_with(ref, muts)
            tid = f"{name}-k{k}-{_sha(key)[:8]}"
            tasks.append({
                "public": {
                    "task_id": tid, "project": name, "files": files,
                    "editable": list(proj["editable"]),
                    "visible_tests": [cases[i] for i in visible]},
                "private": {
                    "task_id": tid, "project": name,
                    "family": "single_defect" if k == 1 else "coupled_defects",
                    "n_defects": k, "mutations": muts,
                    "reference_files": ref,
                    "hidden_f2p": [cases[i] for i in f2p],
                    "hidden_p2p": [cases[i] for i in p2p],
                    "source_sha": _sha(files),
                    "case_sha": _sha([cases[i] for i in f2p + p2p])}})
            got += 1
    return tasks


def check_split_disjoint(private_by_split):
    """No hidden case or task source may appear in two splits."""
    owner = {}
    for split, rows in private_by_split.items():
        for p in rows:
            marks = [("src", p["source_sha"])] + [
                ("case", _sha([c["fn"], c["args"]]))
                for c in p["hidden_f2p"] + p["hidden_p2p"]]
            for mk in marks:
                if owner.setdefault(mk, split) != split:
                    raise ValueError(f"split leak on {mk[0]} in {p['task_id']}")


def build(out_dir, seed=0, per_project=6, ks=(1, 2, 3),
          eval_projects=("ledger",)):
    _require_private_builder()
    if per_project < 1 or not ks or any(k not in (1, 2, 3) for k in ks):
        raise ValueError("Invalid task count or defect count")
    if not set(eval_projects) < set(PROJECTS) or not eval_projects:
        raise ValueError("Need nonempty disjoint train and evaluation projects")
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for split in ("train", "eval"):
        for kind in ("tasks", "private"):
            p = os.path.join(out_dir, f"{kind}_{split}.jsonl")
            if os.path.exists(p):
                raise FileExistsError(p)
            paths[(kind, split)] = p
    rows = {"train": [], "eval": []}
    for i, name in enumerate(sorted(PROJECTS)):
        split = "eval" if name in eval_projects else "train"
        rng = random.Random(f"{seed}:{name}")
        rows[split] += build_project(name, rng, per_project, ks)
    check_split_disjoint({s: [t["private"] for t in r] for s, r in rows.items()})
    for split, ts in rows.items():
        for kind in ("public", "private"):
            name = "tasks" if kind == "public" else "private"
            with open(paths[(name, split)], "x", encoding="utf-8") as fh:
                for t in ts:
                    row = dict(t[kind])
                    if kind == "public":
                        row["split"] = split
                    fh.write(json.dumps(row) + "\n")
    return {s: len(r) for s, r in rows.items()}
