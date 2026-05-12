"""Streaming latency benchmark for NanoTSE models and baselines.

Measures p50/p95/p99 forward-pass time on one 40 ms chunk at 16 kHz. The
sub-60 ms target from PLAN.md / ARCHITECTURE.md is checked against the p95.

The same script runs across devices -- ``--device auto`` picks CUDA, else
MPS, else CPU. Run it per deployment target:

    python -m nanotse.eval.latency_bench                  # auto-pick
    python -m nanotse.eval.latency_bench --device cpu     # commodity CPU bar
    python -m nanotse.eval.latency_bench --device mps     # M-series Mac
    python -m nanotse.eval.latency_bench --device cuda    # 3060 / A100

As new modules land (AudioFrontend, ChunkAttnBackbone, NamedSlotMemory, ...)
add them to the ``CONFIGS`` list below so the latency regression budget
catches creep early.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch
from torch import nn

from nanotse.models.baselines.tdse import TDSEBaseline
from nanotse.models.nanotse import NanoTSE


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench(
    model: nn.Module,
    device: torch.device,
    chunk_samples: int,
    n_warmup: int,
    n_iter: int,
) -> dict[str, float]:
    """Run a tight forward-pass loop, return p50/p95/p99/mean/min in ms."""
    model = model.to(device).eval()
    x = torch.randn(1, chunk_samples, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
            _sync(device)

        times_ms: list[float] = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = model(x)
            _sync(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "p50": statistics.median(times_ms),
        "p95": statistics.quantiles(times_ms, n=20)[18],
        "p99": statistics.quantiles(times_ms, n=100)[98],
        "mean": statistics.fmean(times_ms),
        "min": min(times_ms),
    }


# Models to bench. Add new rows as modules land.
CONFIGS: list[tuple[str, Callable[[], nn.Module]]] = [
    ("TDSEBaseline default     ", lambda: TDSEBaseline()),
    (
        "TDSEBaseline small       ",
        lambda: TDSEBaseline(n_feat=64, bottleneck=32, n_blocks=3),
    ),
    ("NanoTSE audio-only (W2.4)", lambda: NanoTSE()),
    (
        "NanoTSE audio-only small ",
        lambda: NanoTSE(d_model=128, n_heads=2, n_layers=1, cache_len=100),
    ),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", help="auto / cpu / mps / cuda")
    p.add_argument("--chunk-ms", type=float, default=40.0, help="chunk size in ms")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--budget-ms", type=float, default=60.0, help="p95 latency budget in ms")
    args = p.parse_args(argv)

    device = _resolve_device(args.device)
    chunk_samples = int(args.chunk_ms * args.sample_rate / 1000.0)

    print(f"device: {device}")
    print(f"chunk:  {args.chunk_ms} ms  ({chunk_samples} samples @ {args.sample_rate} Hz)")
    print(f"target: p95 < {args.budget_ms} ms")
    print(f"iters:  warmup={args.n_warmup}, measure={args.n_iter}")
    print()

    any_over = False
    for name, factory in CONFIGS:
        model = factory()
        n_params = sum(par.numel() for par in model.parameters())
        s = bench(model, device, chunk_samples, args.n_warmup, args.n_iter)
        rtf = s["p95"] / args.chunk_ms
        verdict = "OK  " if s["p95"] < args.budget_ms else "OVER"
        if verdict == "OVER":
            any_over = True
        print(f"{name}  ({n_params:,} params)")
        print(
            f"  p50={s['p50']:6.2f} ms  "
            f"p95={s['p95']:6.2f} ms  "
            f"p99={s['p99']:6.2f} ms  "
            f"mean={s['mean']:6.2f} ms  "
            f"min={s['min']:6.2f} ms"
        )
        print(f"  RTF (p95 / chunk) = {rtf:5.3f}x   [{verdict}]")
        print()

    return 1 if any_over else 0


if __name__ == "__main__":
    raise SystemExit(main())
