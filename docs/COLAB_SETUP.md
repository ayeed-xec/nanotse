# Colab A100 setup guide

Train NanoTSE on Colab A100 with data on Google Drive (1 TB) and only the
repo + checkpoints on the 60 GB Colab disk.

## TL;DR

1. **Locally:** `bash scripts/prepare_for_colab.sh` -> writes
   `colab_payload/repo.tar.gz` (~500 MB) and `colab_payload/data_v2.tar` (~60 GB).
2. **Upload to Drive:** put both files in `MyDrive/nanotse/`.
3. **Extract data tar TO Drive** (not Colab disk) -- one-off step in a Colab
   cell (`tar -xf .../data_v2.tar -C /content/drive/MyDrive/nanotse/data/`).
   Takes ~30-60 min over Drive I/O but only once.
4. **Open `notebooks/colab_train.ipynb` in Colab** and run cells in order.

## Why this layout

* Data lives on Drive (1 TB plenty). Training reads .wav / .npz over Drive I/O.
  Slower than local SSD but fine with `num_workers=8` + `prefetch_factor=4`.
* Colab disk (60 GB) holds: repo (~500 MB), pip cache, ~3 GB live checkpoints.
* Run dir is symlinked to Drive so disconnects don't lose progress. `--resume`
  picks up from `latest.pt` automatically.

## Storage budget

| Item | Where | Size |
|---|---|---|
| `data/v2/audio/` (70k wav) | Drive | ~28 GB |
| `data/v2/faces/` (face npz) | Drive | ~35 GB |
| `data/v2/manifest.json` | Drive | ~7 MB |
| `repo.tar.gz` | Drive (one-time) | ~500 MB |
| Extracted repo | Colab disk | ~500 MB |
| Checkpoints + logs | Drive (symlinked) | ~3-5 GB |
| Python + pip cache | Colab disk | ~3 GB |
| Working room | Colab disk | ~50 GB free |

## Step-by-step

### 1. Pack locally

On your 3060 box, in the repo root:

```bash
bash scripts/prepare_for_colab.sh
```

Produces:

```
colab_payload/
  repo.tar.gz          # ~500 MB
  data_v2.tar          # ~55-60 GB
```

### 2. Upload to Drive

Easiest: drag-and-drop the two files in the Drive web UI under `MyDrive/nanotse/`.
Faster: use [rclone](https://rclone.org/drive/):

```bash
rclone copy ./colab_payload/ <drive-remote>:nanotse/ --progress
```

### 3. Extract data tar to Drive (one-time, in Colab)

The notebook's Cell 2 includes an option to extract `repo.tar.gz` -- but `data_v2.tar`
is too big for Colab disk, so extract it directly to Drive:

```python
# Run once in a Colab cell
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/nanotse/data', exist_ok=True)
!tar -xf /content/drive/MyDrive/nanotse/data_v2.tar -C /content/drive/MyDrive/nanotse/data/
!ls /content/drive/MyDrive/nanotse/data/v2/
```

Takes ~30-60 min over Drive I/O. After this, the tar can be deleted from Drive
to save space.

### 4. Train

Open `notebooks/colab_train.ipynb` in Colab and run cells in order. The notebook:

* Mounts Drive, verifies data is there
* Extracts repo to `/content/nanotse/`
* Symlinks `runs/colab_a100_v1/` to Drive (so checkpoints persist)
* Fires `scripts/train.py --config configs/colab_a100_v1.yaml`
* Detects existing `latest.pt` on resume and auto-adds `--resume`

### 5. Resume after disconnect

Re-open the notebook, re-run cells 1-4. Cell 4 detects `runs/colab_a100_v1/latest.pt`
in the Drive-symlinked dir and resumes with EMA state preserved.

## Expected throughput / wall time

* Effective batch: 32 (batch=16 x accum=2)
* Total steps: 30000 in the default `configs/colab_a100_v1.yaml`
* A100 40GB at bf16: ~1.5-2x faster than 3060 fp32 per sample
* Drive read overhead: ~1.5-2x slowdown vs local SSD
* Net: probably **0.5-1 effective sps**, finishing in **8-16 h** for 30k steps

Chain runs with `--resume` for longer training (up to 100k+ effective steps).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "manifest.json not found" | Drive isn't mounted or data wasn't extracted to the expected path. Re-run Cell 1. |
| Throughput < 0.1 sps | Drive read I/O bottleneck. Increase `num_workers` to 12, `prefetch_factor` to 6. |
| Session disconnects after ~12h | Colab Pro limit. Re-open notebook, training resumes from `latest.pt` automatically. |
| OOM at batch=16 | Drop to `batch_size=8`, `accum_steps=4` (same effective batch). |
| GPU is V100/T4 instead of A100 | Colab GPU availability varies. Force `device: cuda` in config; the recipe works on V100 (slower) or T4 (memory-tight, may need batch=4). |
