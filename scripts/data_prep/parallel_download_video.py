#!/usr/bin/env python3
"""Parallel video-tar download with HTTP Range resume.

Sequential single-stream download from HF was bound at ~13 MiB/s; the
chained-stream resume kept it correct on socket drops but not fast.
This script runs N=4 worker threads that each download a different
``orig_part_*`` to a local file in ``<root>/raw_video/`` using
``Range: bytes=<saved_size>-`` for resume on interrupt.

After all parts are local, run ``extract_from_local_tars.py`` to read
them as a concatenated stream and produce ``<root>/faces/*.npz`` (the
existing inline-extraction code; mp4 buffers never persisted to disk).

Disk peak: ~280 GB raw + ~80 GB face cache + ~28 GB audio = ~390 GB,
under the 400 GB cap. Raw tars get deleted once extraction finishes.

Usage:
    python scripts/data_prep/parallel_download_video.py --out data/v2 --workers 4
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fetch_voxceleb2_mix_smoke import HF_BASE, _hf_headers

ALL_VIDEO_PARTS = [f"orig_part_a{c}" for c in "abcdefg"]
_CHUNK = 1 << 16  # 64 KiB
_READ_TIMEOUT = 60.0
_MAX_RETRIES_PER_URL = 16


def _content_length(url: str, headers: dict[str, str]) -> int:
    req = urllib.request.Request(url, headers=headers, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length", 0))


def _download_one(url: str, dest: Path, headers: dict[str, str]) -> tuple[str, int, int]:
    """Download ``url`` to ``dest``, resumable on interrupt. Returns ``(url, bytes_written, total)``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _content_length(url, headers)
    already = dest.stat().st_size if dest.exists() else 0
    if already >= total > 0:
        print(f"  {dest.name}: already complete ({total / 1e9:.2f} GB), skipping", flush=True)
        return url, 0, total

    retries = 0
    while already < total:
        hdr = dict(headers)
        if already > 0:
            hdr["Range"] = f"bytes={already}-"
            print(
                f"  {dest.name}: resuming @ {already / 1e9:.2f} / {total / 1e9:.2f} GB",
                flush=True,
            )
        else:
            print(f"  {dest.name}: starting ({total / 1e9:.2f} GB)", flush=True)
        try:
            req = urllib.request.Request(url, headers=hdr)
            resp = urllib.request.urlopen(req, timeout=_READ_TIMEOUT)
            with contextlib.suppress(AttributeError, OSError):
                resp.fp.raw._sock.settimeout(_READ_TIMEOUT)  # type: ignore[attr-defined,union-attr]
            with open(dest, "ab") as f:
                last_report = already
                while True:
                    try:
                        chunk = resp.read(_CHUNK)
                    except (
                        TimeoutError,
                        OSError,
                        urllib.error.URLError,
                        http.client.IncompleteRead,
                        http.client.RemoteDisconnected,
                    ) as e:
                        print(
                            f"  {dest.name}: read error @ {already / 1e9:.2f} GB: {type(e).__name__}",
                            flush=True,
                        )
                        break
                    if not chunk:
                        break
                    f.write(chunk)
                    already += len(chunk)
                    if already - last_report > 1_000_000_000:
                        last_report = already
                        print(
                            f"  {dest.name}: {already / 1e9:.1f} / {total / 1e9:.1f} GB",
                            flush=True,
                        )
            resp.close()
        except (OSError, urllib.error.URLError) as e:
            print(f"  {dest.name}: connection error: {type(e).__name__}: {e}", flush=True)

        if already < total:
            retries += 1
            if retries > _MAX_RETRIES_PER_URL:
                print(
                    f"  {dest.name}: GIVING UP after {retries} retries at {already / 1e9:.2f} GB",
                    file=sys.stderr,
                    flush=True,
                )
                break

    print(f"  {dest.name}: done {already / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
    return url, already, total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/v2"))
    p.add_argument(
        "--workers", type=int, default=4, help="parallel downloads (default 4; HF may rate-limit)"
    )
    p.add_argument(
        "--parts",
        default="all",
        help="comma-separated part names (e.g. 'orig_part_aa,orig_part_ab') or 'all'",
    )
    args = p.parse_args(argv)

    parts = ALL_VIDEO_PARTS if args.parts == "all" else args.parts.split(",")
    urls = [f"{HF_BASE}/{p}" for p in parts]
    raw_dir = args.out / "raw_video"
    raw_dir.mkdir(parents=True, exist_ok=True)

    headers = _hf_headers()
    print(f"parallel downloading {len(urls)} part(s) with {args.workers} worker(s) -> {raw_dir}")
    if "Authorization" in headers:
        print("  (HF token attached)")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_one, url, raw_dir / url.rsplit("/", 1)[-1], headers): url
            for url in urls
        }
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                _, got, total = fut.result()
                if got < total and total > 0:
                    print(f"  WARN: incomplete download for {url} ({got}/{total} bytes)")
            except Exception as e:
                print(f"  FAIL: {url}: {e}", file=sys.stderr)

    print(f"\ndone. raw tars under {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
