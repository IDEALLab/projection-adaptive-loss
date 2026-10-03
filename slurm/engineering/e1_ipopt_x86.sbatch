#!/bin/bash
#SBATCH --job-name=pal-eng-e1-ipopt
#SBATCH --account=<ACCOUNT>
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-59
#SBATCH --output=<SCRATCH>/logs/pal-eng-e1-ipopt-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e1-ipopt-%A_%a.err
# IPOPT on e1/bwb: one multistart restart per array task (3 seeds x 20 restarts).
# Shards are merged by scripts/aggregate_engineering.py.

set -euo pipefail

# cyipopt needs libipopt.so on LD_LIBRARY_PATH at runtime.
# Load the site modules providing gcc and ipopt here.

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi

mkdir -p $SCRATCH/logs

SEEDS=3
RESTART_SHARDS=20
SEED_IDX=$((SLURM_ARRAY_TASK_ID / RESTART_SHARDS))
RESTART_R=$((SLURM_ARRAY_TASK_ID % RESTART_SHARDS))

export WANDB_DIR="${WANDB_DIR:-$SCRATCH/pal/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH/wandb_cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$SCRATCH/wandb_artifacts}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

pal run \
    --config pal/configs/runs/e1_ipopt.yaml \
    --shard "${SEED_IDX}/${SEEDS}" \
    --restart-shard "${RESTART_R}/${RESTART_SHARDS}" \
    --runs-root "${RUNS_ROOT:-$SCRATCH/pal_engineering/e1_ipopt}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT:-pal}" \
    ${WANDB_ENTITY:+${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"}} \
    --wandb-group "e1_ipopt__${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
