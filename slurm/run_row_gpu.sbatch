#!/bin/bash
#SBATCH --job-name=pal-sweep-gpu
#SBATCH --account=<ACCOUNT>
#SBATCH --time=06:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/pal-sweep-gpu-%A_%a.out
#SBATCH --error=logs/pal-sweep-gpu-%A_%a.err
# Runs one row of a sweep's jobs file (row = SLURM_ARRAY_TASK_ID).
# slurm/launch_sweep.sh sets --gpus / --gres per VRAM tier on the command line.

set -euo pipefail

JOBS_FILE="${1:?usage: sbatch ... run_row_gpu.sbatch <jobs_file>}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"

echo "[slurm] host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  jobs=$JOBS_FILE"
echo "[slurm] gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[slurm] entity=$WANDB_ENTITY  project=$WANDB_PROJECT"

mkdir -p logs

pal sweep run-row \
    --jobs-file "$JOBS_FILE" \
    --row "$SLURM_ARRAY_TASK_ID" \
    --device cuda \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    --viz-train-every "${VIZ_TRAIN_EVERY:-0}" \
    ${VIZ_FINAL:+--viz-final} \
    ${VIZ_N:+--viz-n "$VIZ_N"}
