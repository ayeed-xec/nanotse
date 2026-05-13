#!/usr/bin/env bash
# Fresh from-scratch pipeline: full audio + inline face extraction + training.
# Designed for 400 GB disk cap.
#
# Stages (each idempotent / resumable):
#   A. Full audio fetch         (audio_clean_part_aa..ag, ~280 GB on disk)
#   B. Video stream + ROI inline (orig_part_aa..ag; mp4 NEVER persisted;
#                                 cap 5 clips/speaker; ~80 GB face cache)
#   D. From-scratch training    (configs/3060_v2.yaml, fresh weights,
#                                AV + enrollment, 50k steps × batch 8)
#   E. Diagnose                 (best.pt on the new val split)
#
# Disk budget summary:
#   ~280 GB (audio)  +  ~80 GB (face npz)  +  ~5 GB (ckpts/working)  +  buffer
#   = ~370 GB peak  ≤ 400 GB cap

set -euo pipefail
trap 'log "ABORT: stage failed -- pipeline exited at line $LINENO"' ERR

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO/runs/auto"
mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/full_pipeline.log"

log() { printf '[%(%Y-%m-%dT%H:%M:%SZ)T] %s\n' -1 "$*" | tee -a "$ORCH_LOG"; }

DATA="$REPO/data/v2"
mkdir -p "$DATA"

_train_count() {
    [ -f "$1" ] || { echo 0; return; }
    .venv/bin/python -c "import json,sys; print(len(json.load(open(sys.argv[1])).get('train',[])))" "$1" 2>/dev/null || echo 0
}

_disk_usage_gb() {
    df -BG /home/ayeed/nanotse/data | tail -1 | awk '{gsub(/G/,"",$3); print $3}'
}

log "starting full_pipeline (current disk used = $(_disk_usage_gb) GB)"

# ── Stage A: full audio fetch ────────────────────────────────────────────
if [ "$(_train_count "$DATA/manifest.json")" -lt 50000 ]; then
    log "Stage A: full audio fetch -> $DATA  (~280 GB, hours)"
    .venv/bin/python -u "$REPO/scripts/data_prep/fetch_voxceleb2_mix_smoke.py" \
        --all-parts --unlimited --out "$DATA" \
        > "$LOG_DIR/full_A_audio.log" 2>&1
    log "Stage A done: $(grep 'wrote' "$LOG_DIR/full_A_audio.log" | tail -1)  (disk now $(_disk_usage_gb) GB)"
else
    log "Stage A skipped: manifest already has $(_train_count "$DATA/manifest.json") train clips  (disk $(_disk_usage_gb) GB)"
fi

# ── Stage B: parallel download all video parts + local extraction ──────
# Sequential single-stream was 13 MiB/s; 4 parallel range downloads with
# auth should be ~3-4x faster. Raw tars land in $DATA/raw_video/ (temp,
# deleted after extraction).
if [ "$(find "$DATA/faces" -name '*.npz' 2>/dev/null | wc -l)" -lt 5000 ]; then
    log "Stage B.1: parallel video download -> $DATA/raw_video/"
    .venv/bin/python -u "$REPO/scripts/data_prep/parallel_download_video.py" \
        --out "$DATA" --workers 4 \
        > "$LOG_DIR/full_B1_download.log" 2>&1
    log "Stage B.1 done  (disk $(_disk_usage_gb) GB)"

    log "Stage B.2: mouth-ROI extraction from local tars -> $DATA/faces/"
    .venv/bin/python -u "$REPO/scripts/data_prep/extract_from_local_tars.py" \
        --root "$DATA" --max-clips-per-speaker 15 --face-size 112 --fps 25 \
        > "$LOG_DIR/full_B2_extract.log" 2>&1
    log "Stage B.2 done: $(grep 'done:' "$LOG_DIR/full_B2_extract.log" | tail -1)  (disk $(_disk_usage_gb) GB)"

    log "Stage B.3: deleting raw tars to reclaim disk"
    rm -rf "$DATA/raw_video"
    log "Stage B.3 done  (disk $(_disk_usage_gb) GB)"
else
    log "Stage B skipped: $(find "$DATA/faces" -name '*.npz' | wc -l) face npz already extracted  (disk $(_disk_usage_gb) GB)"
fi

# ── Stage D: from-scratch training ───────────────────────────────────────
RUN_NAME="3060_v2_fresh"
log "Stage D: from-scratch training -> runs/$RUN_NAME"

# Point configs/3060_full.yaml at data/v2 for this run by writing a derived config.
TMP_CFG="$REPO/configs/3060_v2.yaml"
.venv/bin/python - "$TMP_CFG" <<'PY'
import sys
from pathlib import Path
import yaml
repo = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
src = yaml.safe_load(Path("configs/3060_full.yaml").read_text())
src["data"]["root"] = "data/v2"
Path(sys.argv[1]).write_text(yaml.safe_dump(src, sort_keys=False))
print(f"wrote {sys.argv[1]}")
PY

.venv/bin/python -u "$REPO/scripts/train.py" \
    --config "$TMP_CFG" \
    --run-name "$RUN_NAME" \
    > "$LOG_DIR/full_D_train.log" 2>&1
log "Stage D done -- ckpts under runs/$RUN_NAME/  (disk $(_disk_usage_gb) GB)"

# ── Stage E: diagnose ──────────────────────────────────────────────────────
log "Stage E: diagnose best.pt"
.venv/bin/python "$REPO/scripts/diagnose.py" \
    --ckpt "$REPO/runs/$RUN_NAME/best.pt" \
    --num-clips 20 \
    --data-root "$DATA" \
    > "$LOG_DIR/full_E_diagnose.log" 2>&1 || log "  diagnose returned non-zero"

log "full pipeline complete  (final disk $(_disk_usage_gb) GB)"
