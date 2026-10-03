#!/bin/bash
#SBATCH --job-name=pal-eng-e4-ipopt
#SBATCH --account=<ACCOUNT>
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19
#SBATCH --output=<SCRATCH>/logs/pal-eng-e4-ipopt-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e4-ipopt-%A_%a.err
# IPOPT on e4/chip_layout: one multistart restart per array task (1 seed x 20 restarts).
# One seed suffices (fixed instance), the aggregator broadcasts it to every trained-method seed.

set -euo pipefail

# geometry/ is a top-level folder, so the repo root must be on PYTHONPATH.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# cyipopt needs libipopt.so on LD_LIBRARY_PATH at runtime.
# Load the site modules providing gcc and ipopt here.

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi

mkdir -p $SCRATCH/logs

SEEDS=1
RESTART_SHARDS=20
SEED_IDX=$((SLURM_ARRAY_TASK_ID / RESTART_SHARDS))
RESTART_R=$((SLURM_ARRAY_TASK_ID % RESTART_SHARDS))

export WANDB_DIR="${WANDB_DIR:-$SCRATCH/pal/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH/wandb_cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$SCRATCH/wandb_artifacts}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

pal run \
    --config pal/configs/runs/e4_ipopt.yaml \
    --shard "${SEED_IDX}/${SEEDS}" \
    --restart-shard "${RESTART_R}/${RESTART_SHARDS}" \
    --runs-root "${RUNS_ROOT:-$SCRATCH/pal_engineering/e4_ipopt}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT:-pal}" \
    ${WANDB_ENTITY:+${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"}} \
    --wandb-group "e4_ipopt__${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
