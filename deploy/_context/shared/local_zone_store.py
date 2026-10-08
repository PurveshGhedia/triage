"""Local JSON-Lines append-only store for zone rankings — a stand-in for
ZoneRankingStore (BigQuery) when BigQuery dataset creation is blocked on an
account (e.g. by a domain-restricted-sharing org policy that blocks IAM
policy writes even for the project owner), or for fully offline testing.

Append-only by construction: every write opens the file in append mode and
adds one JSON line per ranking. Nothing already on disk is ever rewritten or
deleted — the same guarantee PLN-05 requires, just backed by a plain file
instead of a managed table. Swapping this for the real ZoneRankingStore later
is a one-line change in scripts/run_planner_batch.py; nothing about
TriagePlanner or ZoneRanking changes either way.

Versioning is per batch: `next_version(batch_id)` removes the need to remember
`--version N`, and `read_latest(batch_id)` is what a dashboard shows (SYS-03:
only the most recent finalized ranking, never a stale intermediate one).
"""
from __future__ import annotations

from pathlib import Path

from agents.perception.schemas import DEFAULT_BATCH_ID
from agents.planner.schemas import ZoneRanking


class LocalZoneRankingStore:
    def __init__(self, path: str | Path = "data/zone_rankings.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, ranking: ZoneRanking) -> None:
        self.append_all([ranking])

    def append_all(self, rankings: list[ZoneRanking]) -> None:
        with self._path.open("a") as f:
            for ranking in rankings:
                f.write(ranking.model_dump_json() + "\n")

    def read_all(self) -> list[ZoneRanking]:
        """Reads every version of every zone ever written — useful for
        directly confirming append-only behavior (PLN-05): re-run the
        planner with a new --version and this should return BOTH the old
        and new rows for the same zone_id, never just the latest.
        """
        if not self._path.exists():
            return []
        with self._path.open() as f:
            return [ZoneRanking.model_validate_json(line) for line in f if line.strip()]

    def latest_version(self, batch_id: str = DEFAULT_BATCH_ID) -> int:
        """Highest version written for this batch; 0 if none."""
        versions = [r.version for r in self.read_all() if r.batch_id == batch_id]
        return max(versions, default=0)

    def next_version(self, batch_id: str = DEFAULT_BATCH_ID) -> int:
        return self.latest_version(batch_id) + 1

    def read_latest(self, batch_id: str = DEFAULT_BATCH_ID) -> list[ZoneRanking]:
        """Only the most recent version's rows for this batch (SYS-03)."""
        rows = [r for r in self.read_all() if r.batch_id == batch_id]
        if not rows:
            return []
        latest = max(r.version for r in rows)
        return sorted(
            (r for r in rows if r.version == latest),
            key=lambda r: r.priority_score,
            reverse=True,
        )
