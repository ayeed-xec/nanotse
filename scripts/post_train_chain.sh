#!/usr/bin/env bash
# Auto-fire chain that runs after the current 3060 training finishes.
#
# Stages:
#   0. Wait for the running train.py (PID arg) to exit
#   1. Diagnose its best.pt to lock the audio-only baseline number
#   2. Re-fetch full audio_clean_* with resume support -> data/full_v2/
#   3. Fetch orig_part_* video for the new speaker set
#   4. Extract mouth-ROI from each mp4 -> data/full_v2/faces/
#   5. Kick off configs/3060_full.yaml retrain (AV + real lips + new losses)
#
# Each stage writes its own log under runs/auto/ and is skipped if the
# expected output already exists.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO/runs/auto"
mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/orchestrator.log"

log() { printf '[%(%Y-%m-%dT%H:%M:%SZ)T] %s\n' -1 "$*" | tee -a "$ORCH_LOG"; }

TRAIN_PID="${1:-125023}"
log "watching train PID $TRAIN_PID"
while [ -d "/proc/$TRAIN_PID" ]; do
    sleep 30
done
log "train PID $TRAIN_PID exited"

CURRENT_RUN_DIR="$REPO/runs/3060_full_v1"
if [ -f "$CURRENT_RUN_DIR/best.pt" ]; then
    log "stage 1: diagnose best.pt"
    .venv/bin/python "$REPO/scripts/diagnose.py" \
        --ckpt "$CURRENT_RUN_DIR/best.pt" \
        --num-clips 10 \
        --data-root "$REPO/data/full" \
        > "$LOG_DIR/stage1_diagnose_v1.log" 2>&1 || log "  diagnose returned non-zero (ok if no data)"
else
    log "stage 1 skipped: no best.pt at $CURRENT_RUN_DIR"
fi

DATA_V2="$REPO/data/full_v2"
if [ ! -f "$DATA_V2/manifest.json" ] || [ "$(jq '.train | length' "$DATA_V2/manifest.json" 2>/dev/null || echo 0)" -lt 50000 ]; then
    log "stage 2: full audio fetch -> $DATA_V2"
    .venv/bin/python -u "$REPO/scripts/data_prep/fetch_voxceleb2_mix_smoke.py" \
        --all-parts --unlimited --out "$DATA_V2" \
        > "$LOG_DIR/stage2_audio_fetch.log" 2>&1
    log "stage 2 done: $(grep 'wrote' "$LOG_DIR/stage2_audio_fetch.log" | tail -1)"
else
    log "stage 2 skipped: manifest.json already has enough clips"
fi

if [ ! -d "$DATA_V2/video" ] || [ "$(find "$DATA_V2/video" -name '*.mp4' 2>/dev/null | wc -l)" -lt 100 ]; then
    log "stage 3: video fetch -> $DATA_V2/video/"
    .venv/bin/python -u "$REPO/scripts/data_prep/fetch_voxceleb2_video.py" \
        --out "$DATA_V2" --all-parts \
        > "$LOG_DIR/stage3_video_fetch.log" 2>&1
    log "stage 3 done: $(grep 'done:' "$LOG_DIR/stage3_video_fetch.log" | tail -1)"
else
    log "stage 3 skipped: $DATA_V2/video already populated"
fi

if [ ! -d "$DATA_V2/faces" ] || [ "$(find "$DATA_V2/faces" -name '*.npz' 2>/dev/null | wc -l)" -lt 100 ]; then
    log "stage 4: mouth-ROI extraction -> $DATA_V2/faces/"
    .venv/bin/python -u "$REPO/scripts/data_prep/extract_mouth_roi.py" \
        --root "$DATA_V2" --workers 2 \
        > "$LOG_DIR/stage4_mouth_roi.log" 2>&1
    log "stage 4 done: $(grep 'done:' "$LOG_DIR/stage4_mouth_roi.log" | tail -1)"
else
    log "stage 4 skipped: $DATA_V2/faces already populated"
fi

log "stage 5: AV retrain with real lips + Tier-2/3 losses"
.venv/bin/python -u "$REPO/scripts/train.py" \
    --config "$REPO/configs/3060_full.yaml" \
    --run-name 3060_full_v2 \
    > "$LOG_DIR/stage5_retrain.log" 2>&1
log "stage 5 done -- final ckpts under runs/3060_full_v2/"

log "stage 6: diagnose new best.pt"
.venv/bin/python "$REPO/scripts/diagnose.py" \
    --ckpt "$REPO/runs/3060_full_v2/best.pt" \
    --num-clips 10 \
    --data-root "$DATA_V2" \
    > "$LOG_DIR/stage6_diagnose_v2.log" 2>&1 || log "  diagnose returned non-zero"

log "chain complete"
