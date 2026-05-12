"""Smoke test for the latency benchmark module."""

from __future__ import annotations

import torch
from torch import nn

from nanotse.eval.latency_bench import _resolve_device, _sync, bench


def test_bench_returns_expected_keys() -> None:
    model = nn.Sequential(nn.Linear(640, 640))
    stats = bench(model, torch.device("cpu"), chunk_samples=640, n_warmup=1, n_iter=3)
    for k in ("p50", "p95", "p99", "mean", "min"):
        assert k in stats
        assert stats[k] >= 0.0
    assert stats["min"] <= stats["p50"] <= stats["p95"] <= stats["p99"]


def test_resolve_device_explicit_cpu() -> None:
    assert _resolve_device("cpu").type == "cpu"


def test_sync_cpu_is_noop() -> None:
    _sync(torch.device("cpu"))


def test_main_cli_smoke(capsys: object) -> None:
    """Run the CLI with tiny iter counts; must complete and print p95."""
    from nanotse.eval.latency_bench import main

    rc = main(["--device", "cpu", "--n-warmup", "1", "--n-iter", "3"])
    assert rc in (0, 1)
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "p95" in out
