"""Shared blackboard: the one place agents publish results for each other.

Workers never talk to each other directly. A task's output is published under
``result:<task_id>`` and downstream workers receive it through their prompt. That
keeps data flow explicit (it follows DAG edges) and makes the whole state a
plain dict we can checkpoint.

Writes are guarded by a ``threading.Lock`` rather than an ``asyncio.Lock``
because tools run on worker threads via ``asyncio.to_thread``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

__all__ = ["Blackboard", "Entry"]


@dataclass(frozen=True)
class Entry:
    value: Any
    author: str
    version: int


class Blackboard:
    def __init__(self) -> None:
        self._data: dict[str, Entry] = {}
        self._version = 0
        self._lock = threading.Lock()

    def put(self, key: str, value: Any, author: str) -> int:
        with self._lock:
            self._version += 1
            self._data[key] = Entry(value, author, self._version)
            return self._version

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._data.get(key)
        return default if entry is None else entry.value

    def entry(self, key: str) -> Entry | None:
        return self._data.get(key)

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._data if k.startswith(prefix))

    @property
    def version(self) -> int:
        return self._version

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"version": self._version,
                    "entries": {k: {"value": e.value, "author": e.author, "version": e.version}
                                for k, e in self._data.items()}}

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "Blackboard":
        bb = cls()
        bb._version = int(data.get("version", 0))
        for key, e in data.get("entries", {}).items():
            bb._data[key] = Entry(e["value"], e["author"], int(e["version"]))
        return bb
