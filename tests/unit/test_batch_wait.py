"""SYS-04: Agent 2 waits a bounded time for a batch, then plans with what it has."""
from __future__ import annotations

from agents.perception.schemas import TileAssessment, TileStatus
from shared.batch_wait import wait_for_batch


def tiles(n):
    return [TileAssessment(tile_id=f"t{i}", status=TileStatus.ASSESSED) for i in range(n)]


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_no_expected_count_means_no_waiting():
    time = FakeTime()
    result = wait_for_batch(lambda: tiles(3), expected=None, sleep=time.sleep, clock=time.clock)
    assert len(result.tiles) == 3 and not result.incomplete and time.sleeps == []


def test_complete_batch_returns_immediately():
    time = FakeTime()
    result = wait_for_batch(
        lambda: tiles(5), expected=5, timeout_s=60, sleep=time.sleep, clock=time.clock
    )
    assert not result.incomplete and time.sleeps == []


def test_default_zero_timeout_reports_shortfall_without_sleeping():
    time = FakeTime()
    result = wait_for_batch(lambda: tiles(3), expected=5, sleep=time.sleep, clock=time.clock)
    assert result.incomplete and result.missing == 2 and time.sleeps == []


def test_late_tile_arrives_within_timeout_and_polling_stops():
    time = FakeTime()
    loads = iter([tiles(3), tiles(4), tiles(5), tiles(5)])
    result = wait_for_batch(
        lambda: next(loads), expected=5, timeout_s=60, poll_interval_s=5,
        sleep=time.sleep, clock=time.clock,
    )
    assert not result.incomplete
    assert time.sleeps == [5, 5]


def test_tile_that_never_arrives_times_out_and_planning_proceeds():
    time = FakeTime()
    result = wait_for_batch(
        lambda: tiles(4), expected=5, timeout_s=12, poll_interval_s=5,
        sleep=time.sleep, clock=time.clock,
    )
    assert result.incomplete and result.missing == 1 and len(result.tiles) == 4
    assert time.now == 12  # waited the full timeout, not a second longer
    assert time.sleeps == [5, 5, 2]
