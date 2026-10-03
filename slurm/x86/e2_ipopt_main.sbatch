#!/bin/bash
#SBATCH --job-name=pal-e2-ipopt-x86
#SBATCH --account=<ACCOUNT>
#SBATCH --gpus=<GPU_TYPE>:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-63
#SBATCH --output=<SCRATCH>/logs/pal-e2-ipopt-x86-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-e2-ipopt-x86-%A_%a.err
# IPOPT baseline for e2/urban_wind on an x86 medium_gpu node (12-24 GB).
# One array task = one eval query with multi_start restarts, --array=0-63 covers the 64 eval queries.
# Traces go to per-shard subdirs, fold them with scripts/analyze_ipopt_trace.py.

set -euo pipefail

# cyipopt needs libipopt.so on LD_LIBRARY_PATH at runtime.
# Load the site modules providing gcc and ipopt here.

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi

mkdir -p $SCRATCH/logs

SEED="${SEED:-0}"
MAX_ITER="${MAX_ITER:-500}"
MULTI_START="${MULTI_START:-2}"
ARM="${ARM:-main}"
POINTS_DIR="${POINTS_DIR:?POINTS_DIR must point at the generated eval-points shards}"
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/pal_engineering/e2_ipopt_x86/${ARM}}"
TRACE_ROOT="${TRACE_ROOT:-${RUNS_ROOT}/trace}"

TASK="$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")"
QUERY_JSON="${POINTS_DIR}/seed${SEED}_shard${TASK}.json"
if [[ ! -f "${QUERY_JSON}" ]]; then
    echo "ERROR: shard file not found: ${QUERY_JSON}" >&2
    echo "       run scripts/gen_e2_ipopt_points.py to build POINTS_DIR first." >&2
    exit 2
fi

export PAL_IPOPT_JAC_MODE=loop              # jacrev crashes the surrogate
export E2_CLEARANCE_PER_PAIR=1            # K=49 per-pair constraints
export PAL_IPOPT_EVAL_POINTS="${QUERY_JSON}"          # solver solves this query
export PAL_IPOPT_TRACE="${TRACE_ROOT}/shard${TASK}"   # per-shard, no collision
mkdir -p "${PAL_IPOPT_TRACE}"
# Optional best-effort / init knobs, echoed for the record:
echo "[arm=${ARM}] INIT_MODE=${PAL_IPOPT_INIT_MODE:-box} SCALING=${PAL_IPOPT_SCALING:-<off>} " \
     "ACCEPTABLE=${PAL_IPOPT_ACCEPTABLE:-<off>} F64=${PAL_IPOPT_F64:-<off>} " \
     "MAX_ITER=${MAX_ITER} MULTI_START=${MULTI_START} SEED=${SEED}"

export WANDB_DIR="${WANDB_DIR:-$SCRATCH/pal/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH/wandb_cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$SCRATCH/wandb_artifacts}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

# e2 surrogate weights (HF).
if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
fi
python scripts/download_artifacts.py --benchmark e2/urban_wind

pal run \
    --config pal/configs/runs/e2_ipopt_x86.yaml \
    --eval-points "${QUERY_JSON}" \
    --seeds "${SEED}" \
    --max-iter "${MAX_ITER}" \
    --multi-start "${MULTI_START}" \
    --runs-root "${RUNS_ROOT}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT:-pal}" \
    ${WANDB_ENTITY:+${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"}} \
    --wandb-group "e2_ipopt_x86_${ARM}__${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
