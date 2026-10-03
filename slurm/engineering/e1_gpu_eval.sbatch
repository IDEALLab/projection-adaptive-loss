#!/bin/bash -l
#SBATCH --job-name=pal-eval-e1
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --mem=440G
#SBATCH --time=01:00:00
#SBATCH --output=<SCRATCH>/logs/pal-eval-e1-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eval-e1-%j.err

# e1 final eval of one trained run on a single GH200 (decoupled from training to avoid OOM).
# Required env: RUN_ID. Optional: RUNS_ROOT, N_EVAL, PREDICT_BATCH_SIZE, METHOD_OVERRIDE, E1_* envelope.

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
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e1}"
N_EVAL="${N_EVAL:-64}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-1}"
E1_N_STATIONS="${E1_N_STATIONS:-8}"
BWB_FAST_SDF="${BWB_FAST_SDF:-1}"
E1_ALT_LO="${E1_ALT_LO:-2000.0}"
E1_ALT_HI="${E1_ALT_HI:-2000.0}"
E1_V_LO="${E1_V_LO:-60.0}"
E1_V_HI="${E1_V_HI:-60.0}"
PAL_DEVICE="${PAL_DEVICE:-cuda}"
METHOD_OVERRIDE="${METHOD_OVERRIDE:-}"

cluster_resolve_hf_auth || true
cluster_resolve_wandb_auth || true

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail
    cd /workspace/pal
    source /workspace/pal/scripts/cluster_runtime.sh
    export RUNS_ROOT="'"${RUNS_ROOT}"'"
    export E1_N_STATIONS="'"${E1_N_STATIONS}"'"
    export BWB_FAST_SDF="'"${BWB_FAST_SDF}"'"
    export E1_ALT_LO="'"${E1_ALT_LO}"'"
    export E1_ALT_HI="'"${E1_ALT_HI}"'"
    export E1_V_LO="'"${E1_V_LO}"'"
    export E1_V_HI="'"${E1_V_HI}"'"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    cluster_prepare_runtime

    METHOD_OVERRIDE_ARGS=()
    if [[ -n "'"${METHOD_OVERRIDE}"'" ]]; then
        METHOD_OVERRIDE_ARGS=(--method-override "'"${METHOD_OVERRIDE}"'")
    fi

    echo "=== PAL E1 Eval (single-GPU) ==="
    echo "Run id:          '"${RUN_ID}"'"
    echo "Runs root:       '"${RUNS_ROOT}"'"
    echo "N eval:          '"${N_EVAL}"'"
    echo "Predict chunk:   '"${PREDICT_BATCH_SIZE}"'"
    echo "Alt envelope:    [${E1_ALT_LO}, ${E1_ALT_HI}] m"
    echo "V envelope:      [${E1_V_LO}, ${E1_V_HI}] m/s"
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
