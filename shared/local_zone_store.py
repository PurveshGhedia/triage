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
"""
from __future__ import annotations

from pathlib import Path

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
