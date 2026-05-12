#!/usr/bin/env python3
"""Training entrypoint — wired up in W2 with the TDSE baseline.

For now this just validates a config and prints what would run, so that
`make smoke` works end-to-end against the plumbing.

Usage:
    python scripts/train.py --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import sys

from nanotse.utils.config import Config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--remote", default=None, help="vast.ai host for cloud runs")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    print(
        f"Loaded config: device={cfg.device} "
        f"model={cfg.model.name} "
        f"steps={cfg.train.steps} "
        f"batch={cfg.data.batch_size}"
    )
    if args.remote:
        print(f"  remote: {args.remote}")

    print(
        "\nTraining loop not yet implemented — see docs/PLAN.md (W2).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
