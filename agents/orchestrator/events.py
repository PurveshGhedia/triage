"""Append-only run event log: the single source for the dashboard's live feed,
for replaying a finished run, and for the audit trail (SYS-02 / SEC-03).

One JSON object per line, each with a monotonically increasing `seq`, a UTC
timestamp, the run id and a `type`. Safe to call from many threads.

With TRIAGE_JSON_LOGS=1 (or json_logs=True) every event also writes whitelisted structured
lines to stdout for Cloud Logging; see agents/orchestrator/audit_log.py. A logging failure
never affects a run.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from agents.orchestrator import audit_log


class EventLog:
    def __init__(
        self,
        run_id: str,
        path: str | Path | None = None,
        listener: Callable[[dict], None] | None = None,
        json_logs: bool | None = None,
    ) -> None:
        self.run_id = run_id
        self._path = Path(path) if path else None
        self._listener = listener
        self._json_logs = audit_log.json_logs_enabled() if json_logs is None else json_logs
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
            if self._json_logs:
                try:
                    audit_log.write_audit_lines(event)
                except Exception:  # noqa: BLE001 - logging must never break a run
                    pass
        return event


def read_events(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]
