"""Host-private executable oracle protocol with a fork-isolated reporter parent.

Only function names/inputs reach the sandbox; expected outputs stay on the host.
The parent process never imports the untrusted generated solution and cannot
be monkeypatched by the child's Python module imports. Restricted execution
and private independent tests remain necessary; no arbitrary Python grader
can guarantee immunity to deliberately hostile candidate behavior.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys


MAX_RESULT_BYTES = 8192


def require_json_string_keys(value: object) -> None:
    if value is None or type(value) in (str, int, float, bool):
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("Non-string dictionary key in oracle response")
        for nested in value.values():
            require_json_string_keys(nested)
    elif type(value) is list:
        for nested in value:
            require_json_string_keys(nested)
    else:
        raise ValueError("Non-JSON-native Python output type cannot be safely scored")


def unique_json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in oracle response")
        result[key] = value
    return result


def run() -> int:
    try:
        inputs = json.load(sys.stdin)
        if not isinstance(inputs, dict):
            raise ValueError("Invalid call shape")
        fn = inputs["fn"]
        args = inputs.get("args", [])
        kwargs = inputs.get("kwargs", {})
        child_encode = json.JSONEncoder(allow_nan=False).encode
        parent_encode = json.JSONEncoder(allow_nan=False).encode
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                with open(os.devnull, "w", encoding="utf-8") as sink:
                    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                        try:
                            sys.path.insert(0, "/app")
                            module = importlib.import_module("solution")
                            value = getattr(module, fn)(*args, **kwargs)
                            answer = {"kind": "value", "value": value,
                                      "value_type": type(value).__name__,
                                      "args_after": args, "kwargs_after": kwargs}
                        except BaseException as exc:
                            answer = {"kind": "exception", "type": type(exc).__name__}
                try:
                    require_json_string_keys(answer)
                except ValueError:
                    answer = {"kind": "unsupported", "type": "non_json_native"}
                raw = child_encode(answer).encode("utf-8")
                if len(raw) > MAX_RESULT_BYTES:
                    raise ValueError("Unbounded candidate output")
                while raw:
                    raw = raw[os.write(write_fd, raw):]
                os.close(write_fd)
                os._exit(0)
            except BaseException:
                os._exit(2)
        os.close(write_fd)
        chunks = bytearray()
        overflow = False
        while True:
            part = os.read(read_fd, 4096)
            if not part:
                break
            if len(chunks) + len(part) > MAX_RESULT_BYTES:
                overflow = True
                break
            chunks.extend(part)
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        if overflow or not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            raise ValueError("Child execution or protocol failed")
        answer = json.loads(chunks.decode("utf-8"), object_pairs_hook=unique_json_pairs)
        if not isinstance(answer, dict) or answer.get("kind") not in ("value", "exception", "unsupported"):
            raise ValueError("Invalid child output")
        print(parent_encode(answer))
        return 0
    except (ValueError, OSError, TypeError, KeyError):
        print('{"kind":"protocol_error"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
