#!/bin/bash
#SBATCH --job-name=pal-eng-e4
#SBATCH --account=<ACCOUNT>
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-11
#SBATCH --output=<SCRATCH>/logs/pal-eng-e4-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e4-%A_%a.err
# e4/chip_layout trained methods (4 methods x 3 seeds = 12 array tasks), IPOPT is in e4_ipopt_x86.sbatch.
# alm_bolton reuses alm training: run `pal eval --method-override alm_bolton` on each alm run.

set -euo pipefail

# geometry/ is a top-level folder, so the repo root must be on PYTHONPATH.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi

mkdir -p $SCRATCH/logs

export WANDB_DIR="${WANDB_DIR:-$SCRATCH/pal/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH/wandb_cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$SCRATCH/wandb_artifacts}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

pal run \
    --config pal/configs/runs/e4.yaml \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    --runs-root "${RUNS_ROOT:-$SCRATCH/pal_engineering/e4}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT:-pal}" \
    ${WANDB_ENTITY:+${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"}} \
    --wandb-group "e4__${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
