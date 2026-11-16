"""Trusted parent reporter with untrusted repository code isolated in a forked child.

The parent process NEVER imports the candidate repository. A child may mutate
its own Python stdlib/serializer state, but that cannot mutate the parent.
The child encodes its actual call result through an encoder bound before import.
The host retains expected values; the sandbox only sees inputs.

Not an absolute adversarial Python execution proof: restricted AST and external
holdouts remain mandatory, and the child can still attempt protocol cheating.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys


MAX_RESULT_BYTES = 8192


def require_json_string_keys(value: object) -> None:
    """Only permit JSON-native Python types; never erase tuple/key identity."""
    if value is None or type(value) in (str, int, float, bool):
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("Non-string dictionary key would corrupt JSON identity")
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
            raise ValueError("Duplicate JSON key in sandbox protocol")
        result[key] = value
    return result


def run() -> int:
    try:
        call = json.load(sys.stdin)
        if not isinstance(call, dict):
            raise ValueError("Not a function call")
        entry = call["entry"]
        module_name, attribute = entry.rsplit(".", 1)
        args = call.get("args", [])
        kwargs = call.get("kwargs", {})
        # Bind encoder before untrusted import; keep a separate parent encoder.
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
                            sys.path.insert(0, "/app/repository")
                            module = importlib.import_module(module_name)
                            function = getattr(module, attribute)
                            value = function(*args, **kwargs)
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
                    raise ValueError("Candidate output exceeds bounded wire protocol")
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
            piece = os.read(read_fd, 4096)
            if not piece:
                break
            if len(chunks) + len(piece) > MAX_RESULT_BYTES:
                overflow = True
                break
            chunks.extend(piece)
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        if overflow or not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            raise ValueError("Child execution or protocol failed")
        result = json.loads(chunks.decode("utf-8"), object_pairs_hook=unique_json_pairs)
        if not isinstance(result, dict) or result.get("kind") not in ("value", "exception", "unsupported"):
            raise ValueError("Invalid child result")
        print(parent_encode(result))
        return 0
    except (KeyError, TypeError, OSError, ValueError):
        print('{"kind":"protocol_error"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
