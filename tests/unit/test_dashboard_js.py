"""Runs the dashboard's front-end logic tests (tests/js/*.test.js) under node, so they are
part of the normal `pytest` run. Skipped when node is not installed."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_dashboard_state_logic_passes_under_node():
    result = subprocess.run(
        ["node", "--test", str(ROOT / "tests/js/state.test.js")],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-500:]
    assert "# fail 0" in result.stdout
