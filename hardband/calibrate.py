"""Difficulty is observed per-attempt success rate, not pass@k."""
import math


def rates(results, model):
    grouped = {}
    seen = set()
    for row in results:
        if row["model"] != model:
            continue
        key = (row["task_id"], row["sample"])
        if (type(row["sample"]) is not int or row["sample"] < 0 or key in seen
                or type(row["verified"]) is not bool or type(row["passed"]) is not bool):
            raise ValueError("Duplicate or invalid calibration sample")
        seen.add(key)
        grouped.setdefault(row["task_id"], []).append(row)
    out = {}
    for task, rows in grouped.items():
        if ({r["sample"] for r in rows} != set(range(len(rows)))
                or not all(r["verified"] for r in rows)):
            continue
        out[task] = (sum(r["passed"] for r in rows) / len(rows), len(rows))
    return out


def tier(rate):
    if not math.isfinite(rate) or not 0 < rate <= 0.5:
        raise ValueError("Hard-band rate must be in (0, 0.5]")
    return "H1" if rate > 0.375 else "H2" if rate > 0.25 else "H3" if rate > 0.125 else "H4"


def calibrate(results, frontier, student, task_ids, min_frontier=8, min_student=4):
    if (not frontier or frontier == student or min_frontier < 8 or min_student < 4
            or len(task_ids) != len(set(task_ids))):
        raise ValueError("Distinct models, unique tasks and at least 8/4 samples required")
    if any(r["task_id"] not in task_ids for r in results):
        raise ValueError("Unknown task in calibration results")
    fr, st = rates(results, frontier), rates(results, student)
    out = {"learnable": {key: [] for key in ("H1", "H2", "H3", "H4")},
           "frontier_unsolved": [], "too_easy": [], "not_yet": [], "insufficient": []}
    for task in task_ids:
        if task not in fr or task not in st or fr[task][1] < min_frontier or st[task][1] < min_student:
            out["insufficient"].append(task)
        elif fr[task][0] == 0:
            out["frontier_unsolved"].append(task)
        elif fr[task][0] > 0.5:
            out["too_easy"].append(task)
        elif st[task][0] == 0:
            out["not_yet"].append(task)
        else:
            out["learnable"][tier(fr[task][0])].append(task)
    return out
