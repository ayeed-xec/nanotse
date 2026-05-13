#!/usr/bin/env bash
# Pack repo + data for Colab upload.
#
# Outputs two artefacts in ./colab_payload/:
#   repo.tar.gz   -- code only (~500 MB), can also git clone in the notebook
#   data_v2.tar   -- audio + faces + manifest (~55-60 GB uncompressed,
#                   WAV/NPZ don't compress much)
#
# Then upload both to Google Drive folder `nanotse/` -- the notebook expects
# them at /content/drive/MyDrive/nanotse/{repo.tar.gz,data_v2.tar}.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/colab_payload"
mkdir -p "$OUT"

echo "==> Packing repo (code only, no data/runs/.venv)..."
tar -C "$REPO_ROOT" \
    --exclude='./data' \
    --exclude='./runs' \
    --exclude='./.venv' \
    --exclude='./.git' \
    --exclude='./__pycache__' \
    --exclude='./.pytest_cache' \
    --exclude='./.mypy_cache' \
    --exclude='./.ruff_cache' \
    --exclude='./colab_payload' \
    --exclude='*.pyc' \
    -czf "$OUT/repo.tar.gz" .
ls -lh "$OUT/repo.tar.gz"

echo "==> Packing data/v2/ (audio + faces + manifest)..."
echo "    This produces a ~55-60 GB tar; WAV/NPZ are already poorly compressible."
tar -C "$REPO_ROOT/data" \
    -cf "$OUT/data_v2.tar" v2/
ls -lh "$OUT/data_v2.tar"

echo
echo "==> Upload these two files to Google Drive:"
echo "    folder: nanotse/"
echo "    files:  repo.tar.gz, data_v2.tar"
echo
echo "Suggested upload method:"
echo "  - rclone copy ./colab_payload/ <drive_remote>:nanotse/  (fastest)"
echo "  - or drag-and-drop in the Drive web UI"
echo
echo "Then open notebooks/colab_train.ipynb in Colab and follow the cells."
