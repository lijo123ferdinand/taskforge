"""Structured event trace, written as JSON Lines as the run happens.

JSONL rather than one JSON document because the run may crash: every line that
was flushed is still a valid record, and a resumed run simply appends. ``t`` is
seconds since the tracer started (monotonic clock, immune to wall-clock jumps);
``wall`` is an ISO timestamp for humans correlating with other logs.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO

__all__ = ["Tracer", "load_trace"]


class Tracer:
    def __init__(self, path: Path | str | None = None, run_id: str = "") -> None:
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self._t0 = time.monotonic()
        self._offset = 0.0
        self._lock = threading.Lock()
        self._fh: IO[str] | None = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # On resume, continue the timeline after the last recorded event so the
            # renderer shows the crash gap honestly instead of overlapping bars.
            if self.path.exists():
                previous = load_trace(self.path)
                if previous:
                    self._offset = float(previous[-1].get("t", 0.0)) + 0.001
            self._fh = self.path.open("a", encoding="utf-8")

    def now(self) -> float:
        return round(self._offset + time.monotonic() - self._t0, 6)

    def emit(self, type_: str, **fields: Any) -> dict[str, Any]:
        event = {"t": self.now(), "wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                 "run": self.run_id, "type": type_, **fields}
        with self._lock:
            self.events.append(event)
            if self._fh:
                self._fh.write(json.dumps(event, default=str) + "\n")
                self._fh.flush()
        return event

    def of_type(self, type_: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == type_]

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None


def load_trace(path: Path | str) -> list[dict[str, Any]]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A crash mid-write can leave one torn final line; skip it, keep the rest.
            continue
    return events
