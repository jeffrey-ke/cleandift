#!/bin/bash
# Full-dataset validation + correspondence viz for step_400 ObsMask fine-tune.
#SBATCH --job-name=cleandift-eval
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=h100-80:1
#SBATCH -t 0:45:00
#SBATCH -o cleandift_eval_%j.out
#SBATCH -e cleandift_eval_%j.err

set -euo pipefail
trap 'echo "[end] rc=$? at $(date +%T)"' EXIT

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/ocean/projects/cis260205p/jke2/.hf_cache/hub
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"

REFSEG_WS=/ocean/projects/cis260205p/jke2/refseg-workspace
CKPT=/ocean/projects/cis260205p/jke2/cleandift-checkpoints/sd21-obsmask-multi/step_400.pth
OUT=/ocean/projects/cis260205p/jke2/cleandift-checkpoints/sd21-obsmask-multi/eval_step400

echo "[start] job ${SLURM_JOB_ID} on $(hostname) at $(date)"
cd "$REFSEG_WS/cleandift"
uv sync --locked

uv run python scripts/eval_obsmask.py \
  --checkpoint "$CKPT" \
  --output-dir "$OUT"

echo "[done] results in $OUT"
ls -lh "$OUT" "$OUT/correspondence" 2>/dev/null || true
