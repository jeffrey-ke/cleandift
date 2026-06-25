#!/bin/bash
# Correspondence viz with official CompVis/cleandift pretrained weights.
#SBATCH --job-name=cleandift-viz-pre
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=l40s-48:1
#SBATCH -t 0:20:00
#SBATCH -o cleandift_viz_pretrained_%j.out
#SBATCH -e cleandift_viz_pretrained_%j.err

set -euo pipefail
trap 'echo "[end] rc=$? at $(date +%T)"' EXIT

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/ocean/projects/cis260205p/jke2/.hf_cache/hub
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"

OUT=/ocean/projects/cis260205p/jke2/cleandift-checkpoints/sd21-obsmask-multi/eval_pretrained

cd /ocean/projects/cis260205p/jke2/refseg-workspace/cleandift
uv sync --locked

uv run python scripts/eval_obsmask.py \
  --pretrained \
  --skip-val \
  --output-dir "$OUT"

echo "[done] $OUT/correspondence"
ls -lh "$OUT/correspondence"
