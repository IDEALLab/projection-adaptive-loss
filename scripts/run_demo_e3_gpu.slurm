#!/bin/bash -l
#SBATCH --job-name=pal-demo-e3
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=<SCRATCH>/logs/pal-demo-e3-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-demo-e3-%j.err

set -euo pipefail

if [[ -n "${PAL_DIR:-}" ]]; then
    PAL_REPO="$(cd "${PAL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cluster_runtime.sh" ]]; then
    PAL_REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    PAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "${PAL_REPO}/scripts/cluster_runtime.sh"

EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gpu-pal-training}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/e3_demo}"
EPOCHS="${EPOCHS:-200}"
SEEDS="${SEEDS:-0}"
METHODS="${METHODS:-pal}"
BENCHMARK="${BENCHMARK:-e3/acopf_ieee30}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TOKEN_FILE="${WANDB_TOKEN_FILE:-}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"

cluster_resolve_hf_auth || true
cluster_resolve_wandb_auth || true

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail

    cd /workspace/pal
    source /workspace/pal/scripts/cluster_runtime.sh
    export RUNS_ROOT="'"${RUNS_ROOT}"'"
    export ENABLE_WANDB="'"${ENABLE_WANDB}"'"
    export WANDB_PROJECT="'"${WANDB_PROJECT}"'"
    export WANDB_ENTITY="'"${WANDB_ENTITY}"'"
    cluster_prepare_runtime
    wandb_args=()
    if cluster_should_enable_wandb; then
        wandb_args+=(--wandb --wandb-project "${WANDB_PROJECT}")
        if [[ -n "${WANDB_ENTITY:-}" ]]; then
            wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
        fi
    fi
    wandb_status="token-missing"
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        wandb_status="token-present"
    fi

    echo "=== PAL GPU Demo Run ==="
    echo "Repo:       $(pwd)"
    echo "Runs root:  '"${RUNS_ROOT}"'"
    echo "PAL_DATA_DIR=${PAL_DATA_DIR:-}"
    echo "HF_HOME=${HF_HOME:-}"
    echo "WANDB_DIR=${WANDB_DIR:-}"
    echo "WandB:      ${ENABLE_WANDB} / ${wandb_status}"
    echo "========================"

    pal run \
        --method "'"${METHODS}"'" \
        --benchmarks "'"${BENCHMARK}"'" \
        --seeds "'"${SEEDS}"'" \
        --epochs "'"${EPOCHS}"'" \
        --device cuda \
        --runs-root "'"${RUNS_ROOT}"'" \
        "${wandb_args[@]}"
'
