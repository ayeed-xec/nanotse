"""Append-only JSONL metric tracker.

Every smoke + cloud run writes its numbers through this. Comparing
``runs/<old>/metrics.jsonl`` to ``runs/<new>/metrics.jsonl`` is the
contract for the "no silent regressions" rule in PLAN.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class Tracker:
    """Writes one JSON record per ``log()`` call to a JSONL file.

    Pass ``append=True`` to preserve any prior contents (used on ``--resume``
    so the existing trace is kept). Default truncates for fresh runs.
    """

    def __init__(self, path: Path, append: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()
        if not append:
            self.path.write_text("")

    def log(self, step: int, **metrics: float) -> None:
        record: dict[str, float | int] = {
            "step": step,
            "t": round(time.time() - self.start, 3),
            **metrics,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
