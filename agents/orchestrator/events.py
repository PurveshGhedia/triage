"""Append-only run event log: the single source for the dashboard's live feed,
for replaying a finished run, and for the audit trail (SYS-02 / SEC-03).

One JSON object per line, each with a monotonically increasing `seq`, a UTC
timestamp, the run id and a `type`. Safe to call from many threads.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


class EventLog:
    def __init__(
        self,
        run_id: str,
        path: str | Path | None = None,
        listener: Callable[[dict], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self._path = Path(path) if path else None
        self._listener = listener
        self._lock = threading.Lock()
        self._seq = 0
        self.events: list[dict] = []
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, type: str, **payload) -> dict:
        with self._lock:
            self._seq += 1
            event = {
                "seq": self._seq,
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "type": type,
                **payload,
            }
            self.events.append(event)
            if self._path:
                with self._path.open("a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
            # Called inside the lock so concurrent tiles print whole lines, in order.
            if self._listener:
                self._listener(event)
        return event


def read_events(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]
