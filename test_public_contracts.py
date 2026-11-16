"""Public, fixture-free CPU checks; no protected task data or model artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import rcl
import repo_rcl


class PublicContracts(unittest.TestCase):
    def test_safe_single_function(self):
        code = "def clip(value, lower, upper):\n    return min(max(lower, value), upper)\n"
        repo_rcl.require_restricted_source(code, "solution.py", {"solution.py": code})

    def test_default_argument_runtime_capture_rejected(self):
        code = "def clip(value, getter=getattr):\n    return value\n"
        with self.assertRaises(ValueError):
            repo_rcl.require_restricted_source(code, "solution.py", {"solution.py": code})

    def test_cross_file_builtins_decoy_rejected(self):
        files = {
            "pkg/api.py": "def getattr():\n    pass\n",
            "pkg/helpers.py": "def fix(value):\n    return getattr(value, 'secret')\n",
        }
        with self.assertRaises(ValueError):
            repo_rcl.require_restricted_source(files["pkg/helpers.py"], "pkg/helpers.py", files)

    def test_gemma_message_control_tokens_rejected(self):
        for value in ("<|turn>", "<turn|>", "<|channel>", "<start_of_turn>"):
            self.assertTrue(rcl.contains_reserved_marker(value))

    def test_private_task_hash_uses_all_cases(self):
        public_fixture = {
            "id": "documented-sample", "prompt": "illustrative only", "cases": [
                {"fn": "f", "args": [2], "expected": 3},
                {"fn": "f", "args": [3], "expected": 4},
            ],
        }
        modified = json.loads(json.dumps(public_fixture))
        modified["cases"][1]["expected"] = 5
        self.assertNotEqual(rcl.task_digest(public_fixture), rcl.task_digest(modified))

    def test_json_duplicate_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = Path(directory) / "example.jsonl"
            tasks.write_text('{"id":"x","id":"y"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                rcl.jsonl(tasks)

    def test_unsafe_import_rejected(self):
        code = "import json\ndef fix(value):\n    return value\n"
        with self.assertRaises(ValueError):
            repo_rcl.require_restricted_source(code, "solution.py", {"solution.py": code})


if __name__ == "__main__":
    unittest.main()
