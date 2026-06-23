#!/bin/bash
# CleanDIFT fine-tune on isaac ObsMask umbrellas (3750 frames).
# One L40S-48 GPU (torch 2.11+cu128 has no kernels for V100 CC 7.0).
#SBATCH --job-name=cleandift-obsmask
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=l40s-48:1
#SBATCH -t 0:45:00
#SBATCH -o cleandift_obsmask_%j.out
#SBATCH -e cleandift_obsmask_%j.err

set -euo pipefail
trap 'echo "[end] rc=$? at $(date +%T)"' EXIT

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/ocean/projects/cis260205p/jke2/.hf_cache/hub
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"

REFSEG_WS=/ocean/projects/cis260205p/jke2/refseg-workspace
OCEAN_ROOT=/ocean/projects/cis260205p/jke2
CKPT_DIR="$OCEAN_ROOT/cleandift-checkpoints/sd21-obsmask-multi"

DATASET_DIRS="[/ocean/projects/cis260205p/jke2/refseg-workspace/segmentation/datasets/expanded-refseg,/ocean/projects/cis260205p/jke2/refseg-workspace/segmentation/datasets/mixed-persp,/ocean/projects/cis260205p/jke2/refseg-workspace/UFM-train/datasets/shelf-optflow]"

echo "[start] job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "flags : [$-]   (no 'i' = non-interactive)"
echo "uv    : $(command -v uv || echo 'NOT FOUND')"
uv --version
echo "cache : $(uv cache dir)"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_TOKEN set: $([ -n "${HF_TOKEN:-}" ] && echo yes || echo no)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

mkdir -p "$CKPT_DIR" "$HF_HUB_CACHE"

cd "$REFSEG_WS/cleandift"
uv sync --locked

uv run python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see GPU'; print('torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))"

uv run cleandift-train --config-name sd21_isaac_obsmask_multi \
  "data.dataset_dirs=${DATASET_DIRS}" \
  "checkpoint_dir=${CKPT_DIR}" \
  data.batch_size=8
