"""Crash-safe checkpoints.

Written after every task that *finishes* (pass or permanent failure), never
mid-task: a resumed run redoes at most the tasks that were in flight. Saved
atomically (write temp file, fsync, rename) so a crash during the save leaves the
previous checkpoint intact instead of a truncated JSON file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["save_checkpoint", "load_checkpoint", "CHECKPOINT_VERSION"]

CHECKPOINT_VERSION = 1


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump({"version": CHECKPOINT_VERSION, **state}, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version {data.get('version')!r}")
    return data
