#!/bin/bash
# bf16 smoke: 1 training step, batch 8, L40S — verify no OOM and finite loss.
#SBATCH --job-name=cleandift-smoke
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=l40s-48:1
#SBATCH -t 0:20:00
#SBATCH -o cleandift_smoke_%j.out
#SBATCH -e cleandift_smoke_%j.err

set -euo pipefail
trap 'echo "[end] rc=$? at $(date +%T)"' EXIT

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/ocean/projects/cis260205p/jke2/.hf_cache/hub
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"

REFSEG_WS=/ocean/projects/cis260205p/jke2/refseg-workspace
DATASET_DIRS="[/ocean/projects/cis260205p/jke2/refseg-workspace/segmentation/datasets/expanded-refseg,/ocean/projects/cis260205p/jke2/refseg-workspace/segmentation/datasets/mixed-persp,/ocean/projects/cis260205p/jke2/refseg-workspace/UFM-train/datasets/shelf-optflow]"

cd "$REFSEG_WS/cleandift"
uv sync --locked

uv run cleandift-train --config-name sd21_isaac_obsmask_multi \
  "data.dataset_dirs=${DATASET_DIRS}" \
  checkpoint_dir=/tmp/cleandift-smoke-$$ \
  data.batch_size=4 \
  max_steps=1 \
  val_freq=null
