"""Public-source Hard Band contract tests; no private corpus or holdouts.

These tests intentionally use hand-authored, non-secret toy fixtures and mocks.
They must remain executable from the public repository with corpus.py absent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import rcl
from hardband.calibrate import calibrate, rates
from hardband.prompt import build_prompt, export, parse_response, replay
from hardband.reward import score


ROOT = Path(__file__).resolve().parent


def public_task(split: str = "train") -> dict:
    return {
        "task_id": "contract-1",
        "split": split,
        "project": "contract",
        "editable": ["calculator.py"],
        "files": {"calculator.py": "def double(value):\n    return value * 2\n"},
        "visible_tests": [
            {"fn": "calculator.double", "args": [2], "expected": 4},
        ],
    }


def private_shape(task: dict | None = None) -> dict:
    """Deliberately no reference solutions or private expected-answer corpus."""
    if task is None:
        task = public_task()
    data = {
        "task_id": "contract-1",
        "project": "contract",
        "hidden_f2p": [{"fn": "calculator.double", "args": [4], "expected": 8}],
        "hidden_p2p": [{"fn": "calculator.double", "args": [3], "expected": 6}],
    }
    data["source_sha"] = rcl.digest(json.dumps(task["files"], sort_keys=True))
    data["case_sha"] = rcl.digest(json.dumps(
        data["hidden_f2p"] + data["hidden_p2p"], sort_keys=True,
    ))
    return data


def observed(model: str, task: str, successes: int, count: int) -> list[dict]:
    return [
        {
            "model": model,
            "task_id": task,
            "sample": index,
            "verified": True,
            "passed": index < successes,
        }
        for index in range(count)
    ]


class PublicDistribution(unittest.TestCase):
    def test_package_and_cli_import_without_private_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "hardband"
            package.mkdir()
            for filename in (
                "__init__.py", "__main__.py", "build.py", "mutate.py",
                "calibrate.py", "prompt.py", "reward.py", "sandbox.py",
            ):
                shutil.copyfile(ROOT / "hardband" / filename, package / filename)
            for filename in ("rcl.py", "repo_rcl.py"):
                shutil.copyfile(ROOT / filename, root / filename)
            self.assertFalse((package / "corpus.py").exists())
            code = (
                "import hardband; import hardband.prompt; import hardband.reward; "
                "import hardband.calibrate; import hardband.build"
            )
            environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            imported = subprocess.run(
                [sys.executable, "-c", code], cwd=root,
                capture_output=True, text=True, timeout=20, env=environment,
                check=False,
            )
            self.assertEqual(imported.returncode, 0, imported.stderr)
            for args in (("--help",), ("calibrate", "--help"),
                         ("replay", "--help"), ("export", "--help")):
                cli = subprocess.run(
                    [sys.executable, "-m", "hardband", *args], cwd=root,
                    capture_output=True, text=True, timeout=20,
                    env=environment, check=False,
                )
                self.assertEqual(
                    cli.returncode, 0,
                    f"Corpus-free CLI {args} failed: {cli.stderr}",
                )


class PublicPromptAndMapping(unittest.TestCase):
    def test_parse_valid_json_and_reject_unparseable(self):
        valid = '{"files":{"calculator.py":"def double(x): return x * 2"}}'
        self.assertEqual(parse_response(valid), {
            "calculator.py": "def double(x): return x * 2",
        })
        fence = chr(96) * 3
        self.assertEqual(parse_response(fence + "json\n" + valid + "\n" + fence), {
            "calculator.py": "def double(x): return x * 2",
        })
        self.assertIsNone(parse_response("no candidate source"))
        self.assertIsNone(parse_response('{"files":{}}'))

    def test_duplicate_file_keys_are_rejected(self):
        response = ('{"files":{"calculator.py":"correct",'
                    '"calculator.py":"silently overwritten"}}')
        self.assertIsNone(parse_response(response))

    def test_replay_requires_exact_task_mapping(self):
        task = public_task()
        secret = private_shape(task)
        attempt = {
            "task_id": task["task_id"], "sample": 0, "model": "fixture",
            "model_digest": "a" * 64,
            "generation_config": {
                "temperature": 0.4, "top_p": 0.9,
                "num_predict": 128, "seed": 7,
            },
            "prompt_sha256": rcl.digest(build_prompt(task)),
            "response": '{"files":{"calculator.py":"def double(x): return x"}}',
        }
        with mock.patch(
            "hardband.prompt.score",
            return_value={
                "verified": True, "passed": False, "reason": "hidden_fail",
                "reward": 0.0, "f2p_frac": 0.0,
            },
        ):
            result = replay([task], [secret], [attempt])
            self.assertEqual(len(result), 1)
            self.assertFalse(result[0]["passed"])
            with self.assertRaises(ValueError):
                replay([task, task], [secret], [attempt])
            with self.assertRaises(ValueError):
                replay([task], [secret, secret], [attempt])
            with self.assertRaises(ValueError):
                replay([task], [secret], [{**attempt, "task_id": "missing"}])

    def test_export_requires_passed_success_and_training_split(self):
        task = public_task()
        secret = private_shape(task)
        failed = {
            "task_id": task["task_id"], "sample": 0, "model": "student",
            "response": "incorrect", "verified": True, "passed": False,
            "reason": "hidden_fail", "reward": 0.0,
        }
        success = {
            **failed, "sample": 1, "response": "correct",
            "passed": True, "reason": "passed", "reward": 1.0,
        }

        def mocked_export(rows):
            with (
                mock.patch("hardband.prompt.replay", return_value=rows),
                mock.patch("hardband.prompt.calibrate", return_value={
                    "learnable": {"H3": [task["task_id"]]},
                }),
            ):
                return export(
                    [task], [secret], [{"model": "student"}],
                    [{"model": "frontier"}], "frontier", "student",
                )

        only_failed_sft, only_failed_dpo = mocked_export([failed])
        self.assertEqual(only_failed_sft, [])
        self.assertEqual(only_failed_dpo, [])
        sft, dpo = mocked_export([failed, success])
        self.assertEqual(len(sft), 1)
        self.assertEqual(sft[0]["completion"], "correct")
        self.assertEqual(len(dpo), 1)
        self.assertEqual(dpo[0]["chosen"], "correct")
        self.assertEqual(dpo[0]["rejected"], "incorrect")
        with self.assertRaises(ValueError):
            export([public_task("eval")], [secret], [{"model": "student"}],
                   [{"model": "frontier"}], "frontier", "student")


class PublicRewardAndCalibration(unittest.TestCase):
    def test_failed_repair_is_verified_but_not_a_positive(self):
        public = public_task()
        private = private_shape()
        candidate = {
            "calculator.py": "def double(value):\n    return value * 2\n",
        }
        with (
            mock.patch("hardband.reward.repo_rcl.require_restricted_source"),
            mock.patch("hardband.reward.judge", return_value=["fail", "pass", "pass"]),
        ):
            verdict = score(public, private, candidate)
        self.assertIs(verdict["verified"], True)
        self.assertIs(verdict["passed"], False)
        self.assertEqual(verdict["reason"], "hidden_fail")
        self.assertEqual(verdict["reward"], 0.0)

    def test_hard_band_tiers_use_passed_not_verification(self):
        results = (
            observed("frontier", "hard", 2, 8) +
            observed("student", "hard", 1, 4) +
            observed("frontier", "unsolved", 0, 8) +
            observed("student", "unsolved", 0, 4) +
            observed("frontier", "easy", 8, 8) +
            observed("student", "easy", 2, 4)
        )
        classified = calibrate(
            results, "frontier", "student", ["hard", "unsolved", "easy"]
        )
        self.assertEqual(classified["learnable"]["H3"], ["hard"])
        self.assertEqual(classified["frontier_unsolved"], ["unsolved"])
        self.assertEqual(classified["too_easy"], ["easy"])
        with self.assertRaises(ValueError):
            rates(results[:8] + [results[0]], "frontier")
        incomplete = [
            {**row, "verified": False} if row["sample"] == 3 else row
            for row in results
        ]
        self.assertNotIn("hard", rates(incomplete, "frontier"))


if __name__ == "__main__":
    unittest.main()
