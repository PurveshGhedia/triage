"""Bounded wait for a batch to finish reporting (SYS-04).

Agent 2 must not stall forever because one tile's job died before writing
anything. `wait_for_batch` polls until `expected` tiles have reported or the
timeout passes, then hands back whatever is there — flagged as incomplete — so
planning always proceeds. With timeout 0 (the default) it never sleeps: it
loads once and reports any shortfall.

Clock and sleep are injectable so tests run instantly.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agents.perception.schemas import TileAssessment


@dataclass
class BatchReadiness:
    tiles: list[TileAssessment]
    expected: int | None

    @property
    def missing(self) -> int:
        if self.expected is None:
            return 0
        return max(self.expected - len(self.tiles), 0)

    @property
    def incomplete(self) -> bool:
        return self.missing > 0


def wait_for_batch(
    load_tiles: Callable[[], list[TileAssessment]],
    expected: int | None = None,
    timeout_s: float = 0.0,
    poll_interval_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> BatchReadiness:
    tiles = load_tiles()
    if expected is None:
        return BatchReadiness(tiles=tiles, expected=None)

    deadline = clock() + timeout_s
    while len(tiles) < expected:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sleep(min(poll_interval_s, remaining))
        tiles = load_tiles()

    return BatchReadiness(tiles=tiles, expected=expected)
