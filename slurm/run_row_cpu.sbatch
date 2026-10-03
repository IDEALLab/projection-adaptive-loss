#!/bin/bash
#SBATCH --job-name=pal-sweep-cpu
#SBATCH --account=<ACCOUNT>
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/pal-sweep-cpu-%A_%a.out
#SBATCH --error=logs/pal-sweep-cpu-%A_%a.err
# Runs one row of a sweep's jobs_cpu.jsonl (row = SLURM_ARRAY_TASK_ID).

set -euo pipefail

JOBS_FILE="${1:?usage: sbatch ... run_row_cpu.sbatch <jobs_file>}"

# Activate the pal venv. Edit path if the venv lives elsewhere.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"

echo "[slurm] host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  jobs=$JOBS_FILE"
echo "[slurm] entity=$WANDB_ENTITY  project=$WANDB_PROJECT"

mkdir -p logs

pal sweep run-row \
    --jobs-file "$JOBS_FILE" \
    --row "$SLURM_ARRAY_TASK_ID" \
    --device cpu \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    --viz-train-every "${VIZ_TRAIN_EVERY:-0}" \
    ${VIZ_FINAL:+--viz-final} \
    ${VIZ_N:+--viz-n "$VIZ_N"}
