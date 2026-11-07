"""Deterministic acceptance checks, run by the critic before any LLM judgement.

Why both? An LLM critic is good at "is this summary faithful?" and bad at "does
this file exist?". Cheap mechanical checks go first; if they fail, we skip the
LLM call entirely and send the precise failure back as revision feedback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .tools.base import ToolError, resolve_in_sandbox

__all__ = ["run_checks", "CheckResult"]

CheckResult = tuple[bool, str]


def _one(check: dict[str, Any], output: str, sandbox: Path) -> CheckResult:
    kind = check["type"]
    if kind == "contains":
        v = str(check.get("value", ""))
        return v in output, f"output must contain {v!r}"
    if kind == "not_contains":
        v = str(check.get("value", ""))
        return v not in output, f"output must not contain {v!r}"
    if kind == "regex":
        pattern = str(check.get("pattern", ""))
        return re.search(pattern, output, re.MULTILINE) is not None, f"output must match /{pattern}/"
    if kind == "min_length":
        n = int(check.get("value", 1))
        return len(output.strip()) >= n, f"output must be at least {n} characters (got {len(output.strip())})"
    if kind == "json_keys":
        keys = list(check.get("keys", []))
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return False, "output must be a JSON object"
        if not isinstance(data, dict):
            return False, "output must be a JSON object"
        missing = [k for k in keys if k not in data]
        return not missing, f"JSON output is missing keys {missing}" if missing else "ok"
    if kind == "file_exists":
        path = str(check.get("path", ""))
        try:
            target = resolve_in_sandbox(sandbox, path)
        except ToolError as exc:
            return False, str(exc)
        ok = target.is_file() and target.stat().st_size > 0
        return ok, f"file {path} must exist and be non-empty"
    return False, f"unknown check type {kind!r}"


def run_checks(checks: tuple[dict[str, Any], ...] | list[dict[str, Any]], output: str,
               sandbox: Path) -> list[tuple[dict[str, Any], bool, str]]:
    return [(c, *_one(c, output, sandbox)) for c in checks]
