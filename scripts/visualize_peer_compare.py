#!/usr/bin/env python3
"""One-off peer comparison: NanoTSE v6 vs published streaming AV-TSE papers.

Two views in one figure:
  Left:  absolute val SDRi bar chart (NanoTSE iterations + peer numbers)
  Right: params-vs-SDRi Pareto scatter (efficiency view)

Peer numbers are typical-paper values; exact numbers vary by benchmark
(VoxCeleb2-mix vs LRS3 vs other). The point is to show the gap, not to
mis-attribute exact numbers to specific papers.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_val(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return sorted(
        [json.loads(line) for line in path.read_text().splitlines() if '"val_sdri_db"' in line],
        key=lambda r: r["step"],
    )


def _best_val(path: Path) -> float | None:
    val = _load_val(path)
    if not val:
        return None
    return max(v["val_sdri_db"] for v in val)


def main() -> int:
    runs_dir = Path("runs")
    v3_best = _best_val(runs_dir / "3060_v3_fresh" / "metrics.jsonl")
    v5_best = _best_val(runs_dir / "3060_v5_fresh" / "metrics.jsonl")
    v6_best = _best_val(runs_dir / "3060_v6_fresh" / "metrics.jsonl")

    # Model list: (label, sdri_db, params_M, streaming, color, ours)
    # NanoTSE numbers from actual runs (or projected for unfinished/future)
    rows = [
        # Ours (real numbers where available)
        ("NanoTSE v2 (3060)",            0.18,  4.75, True, "C0", True),
        ("NanoTSE v3 partial (3060)",    v3_best if v3_best is not None else -0.14, 4.75, True, "C0", True),
        ("NanoTSE v5 partial (3060)",    v5_best if v5_best is not None else 0.11, 4.75, True, "C0", True),
        ("NanoTSE v6 in-progress (3060)", v6_best if v6_best is not None else 0.19, 18.75, True, "C0", True),
        # Projected
        ("NanoTSE v6 projected final",   1.5,  18.75, True, "C2", True),
        ("NanoTSE a100_v1 projected",    3.5,  27.0,  True, "C2", True),
        ("NanoTSE a100_v2 + pretrained", 6.0,  40.0,  True, "C2", True),
        # Peer streaming AV-TSE (representative numbers from the literature)
        ("AV-SkiM (peer streaming)",     9.0,   8.0,  True, "C3", False),
        ("MeMo (Li 2025, streaming)",    10.0, 10.0,  True, "C3", False),
        # Peer non-streaming (out of class but shown for context)
        ("SpEx+ (non-streaming)",        12.0, 11.0,  False, "gray", False),
        ("AV-Sepformer (non-streaming)", 14.5, 25.0,  False, "gray", False),
    ]

    labels = [r[0] for r in rows]
    sdri = [r[1] for r in rows]
    params = [r[2] for r in rows]
    streaming = [r[3] for r in rows]
    colors = [r[4] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "NanoTSE v6 vs published streaming AV-TSE peers\n(numbers represent typical paper values; exact benchmarks vary)",
        fontsize=13, y=1.02,
    )

    # ---------- bar chart
    ax = axes[0]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, sdri, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7)
    ax.axvline(3, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(8, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Val SDRi (dB)  —  higher = better")
    ax.set_title("Absolute SDRi quality")
    ax.text(3.05, -0.6, "publishable bar", color="orange", fontsize=8, alpha=0.8)
    ax.text(8.05, -0.6, "peer streaming SOTA", color="red", fontsize=8, alpha=0.8)
    for i, (label, val) in enumerate(zip(labels, sdri)):
        ax.text(val + 0.15 if val >= 0 else val - 0.15, i, f"{val:+.2f} dB",
                va="center", ha="left" if val >= 0 else "right", fontsize=8)
    ax.set_xlim(-12, 18)
    ax.grid(axis="x", alpha=0.3)

    # ---------- Pareto scatter (params vs SDRi)
    ax = axes[1]
    for i, (label, sdri_v, p, stream, c, ours) in enumerate(rows):
        marker = "o" if ours else ("s" if stream else "D")
        size = 200 if ours else 150
        ax.scatter(p, sdri_v, s=size, c=c, marker=marker, alpha=0.85,
                   edgecolor="black", linewidth=0.5, zorder=3)
        # Label offset to avoid overlap
        ha = "left"
        dx, dy = 1.5, 0.0
        if "AV-SkiM" in label:
            dy = -0.4
        if "v6 projected" in label or "a100_v1" in label:
            dy = 0.35
        if "a100_v2" in label:
            dy = -0.3
        if "v6 in-progress" in label:
            dy = -0.35
        if "Sepformer" in label:
            dx, dy = -1.5, 0
            ha = "right"
        ax.annotate(label, (p, sdri_v), xytext=(p + dx, sdri_v + dy),
                    fontsize=8, ha=ha, va="center")

    ax.set_xlabel("Model parameters (M)")
    ax.set_ylabel("Val SDRi (dB)")
    ax.set_title("Pareto: params vs quality")
    ax.axvline(6, color="gray", linestyle=":", alpha=0.5)
    ax.text(6.2, -1.5, "3060 6GB cap", color="gray", fontsize=8, rotation=90, va="bottom")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(-2, 17)
    # Legend
    legend_elements = [
        plt.scatter([], [], s=150, c="C0", marker="o", edgecolor="black", label="NanoTSE (measured)"),
        plt.scatter([], [], s=150, c="C2", marker="o", edgecolor="black", label="NanoTSE (projected)"),
        plt.scatter([], [], s=150, c="C3", marker="s", edgecolor="black", label="Peer streaming"),
        plt.scatter([], [], s=150, c="gray", marker="D", edgecolor="black", label="Peer non-streaming"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    out_dir = Path("runs/_viz_peer_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "peer_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # Also dump the table to stdout
    print()
    print(f"{'Model':<40} {'SDRi':>7} {'Params':>9} {'Stream':>7} {'Class':>6}")
    print("-" * 80)
    for label, val, p, stream, _, ours in rows:
        cls = "ours" if ours else ("peer-s" if stream else "peer-n")
        print(f"{label:<40} {val:>+6.2f}  {p:>7.2f} M  {str(stream):>7}  {cls:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
