#!/bin/bash -l
#SBATCH --job-name=pal-eval-e2
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --mem=440G
#SBATCH --time=08:00:00
#SBATCH --output=<SCRATCH>/logs/pal-eval-e2-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eval-e2-%j.err

# e2 final eval of one trained run on a single GH200: 64 random queries, overwriting the 1-point in-process eval.
# Required env: RUN_ID. Optional: RUNS_ROOT, N_EVAL, PREDICT_BATCH_SIZE, METHOD_OVERRIDE.

set -euo pipefail

if [[ -z "${RUN_ID:-}" ]]; then
    echo "ERROR: RUN_ID env var must be set (run dir name under RUNS_ROOT)." >&2
    exit 2
fi

if [[ -n "${PAL_DIR:-}" ]]; then
    PAL_REPO="$(cd "${PAL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cluster_runtime.sh" ]]; then
    PAL_REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    PAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
source "${PAL_REPO}/scripts/cluster_runtime.sh"

EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gpu-pal-training}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e2}"
N_EVAL="${N_EVAL:-64}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-8}"
PAL_DEVICE="${PAL_DEVICE:-cuda}"
METHOD_OVERRIDE="${METHOD_OVERRIDE:-}"

cluster_resolve_hf_auth || true
cluster_resolve_wandb_auth || true

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail
    cd /workspace/pal
    source /workspace/pal/scripts/cluster_runtime.sh
    export RUNS_ROOT="'"${RUNS_ROOT}"'"
    export E2_CLEARANCE_PER_PAIR=1
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    cluster_prepare_runtime

    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
    fi
    python scripts/download_artifacts.py --benchmark e2/urban_wind

    METHOD_OVERRIDE_ARGS=()
    if [[ -n "'"${METHOD_OVERRIDE}"'" ]]; then
        METHOD_OVERRIDE_ARGS=(--method-override "'"${METHOD_OVERRIDE}"'")
    fi

    echo "=== PAL E2 Eval (single-GPU) ==="
    echo "Run id:          '"${RUN_ID}"'"
    echo "Runs root:       '"${RUNS_ROOT}"'"
    echo "N eval:          '"${N_EVAL}"'"
    echo "Predict chunk:   '"${PREDICT_BATCH_SIZE}"'"
    echo "Method override: '"${METHOD_OVERRIDE:-<none>}"'"
    echo "================================="

    python -m pal.runner.cli eval \
        --run-id "'"${RUN_ID}"'" \
        --runs-root "'"${RUNS_ROOT}"'" \
        --device "'"${PAL_DEVICE}"'" \
        --n-eval "'"${N_EVAL}"'" \
        --predict-batch-size "'"${PREDICT_BATCH_SIZE}"'" \
        "${METHOD_OVERRIDE_ARGS[@]}"
'
