"""Tracker writes one JSON record per call with `step` and `t` fields."""

from __future__ import annotations

import json
from pathlib import Path

from nanotse.utils.tracker import Tracker


def test_tracker_appends_jsonl_records(tmp_path: Path) -> None:
    t = Tracker(tmp_path / "metrics.jsonl")
    t.log(0, loss=1.5, si_snr_db=0.1)
    t.log(10, loss=1.2, si_snr_db=2.4)

    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["step"] == 0
    assert first["loss"] == 1.5
    assert first["si_snr_db"] == 0.1
    assert "t" in first

    second = json.loads(lines[1])
    assert second["step"] == 10


def test_tracker_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nest" / "metrics.jsonl"
    t = Tracker(nested)
    t.log(1, x=0.0)
    assert nested.exists()
