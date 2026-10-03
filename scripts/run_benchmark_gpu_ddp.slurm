#!/bin/bash -l
#SBATCH --job-name=pal-benchmark-ddp
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=02:00:00
#SBATCH --output=<SCRATCH>/logs/pal-benchmark-ddp-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-benchmark-ddp-%j.err

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
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/manual_ddp}"
EPOCHS="${EPOCHS:-200}"
SEEDS="${SEEDS:-0}"
METHODS="${METHODS:-pal}"
BENCHMARK="${BENCHMARK:?set BENCHMARK to a registered benchmark id}"
PAL_DEVICE="${PAL_DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
MASTER_PORT="${MASTER_PORT:-29500}"
N_EVAL="${N_EVAL:-64}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-0}"
PAL_EXTRA_ARGS="${PAL_EXTRA_ARGS:-}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TOKEN_FILE="${WANDB_TOKEN_FILE:-}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"

export NNODES MASTER_PORT MASTER_ADDR
cluster_resolve_hf_auth || true
cluster_resolve_wandb_auth || true

srun -u \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --gpus-per-task="${NPROC_PER_NODE}" \
    --gpu-bind=none \
    --environment="${EDF_ENVIRONMENT}" \
    /bin/bash -lc '
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
    pal_extra_args=()
    if [[ -n "'"${PAL_EXTRA_ARGS}"'" ]]; then
        read -r -a pal_extra_args <<< "'"${PAL_EXTRA_ARGS}"'"
    fi
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    echo "=== PAL Benchmark DDP Run ==="
    echo "Repo:       $(pwd)"
    echo "Benchmark:  '"${BENCHMARK}"'"
    echo "Device:     '"${PAL_DEVICE}"'"
    echo "Batch size: '"${BATCH_SIZE}"' (global)"
    echo "Eval size:  '"${N_EVAL}"'"
    echo "Predict chunk: '"${PREDICT_BATCH_SIZE}"'"
    echo "World size: '"${NPROC_PER_NODE}"'"
    echo "Nodes:      ${NNODES}"
    echo "Master:     ${MASTER_ADDR}:${MASTER_PORT}"
    echo "Runs root:  '"${RUNS_ROOT}"'"
    echo "Extra args: '"${PAL_EXTRA_ARGS}"'"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "PAL_DATA_DIR=${PAL_DATA_DIR:-}"
    echo "HF_HOME=${HF_HOME:-}"
    echo "XDG_CACHE_HOME=${XDG_CACHE_HOME:-}"
    echo "PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-}"
    echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-}"
    echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-}"
    echo "WANDB_DIR=${WANDB_DIR:-}"
    echo "WandB:      ${ENABLE_WANDB} / ${wandb_status}"
    echo "============================="

    torchrun \
        --nnodes "${NNODES}" \
        --nproc-per-node "'"${NPROC_PER_NODE}"'" \
        --node-rank "${SLURM_NODEID}" \
        --rdzv-backend c10d \
        --rdzv-id "${SLURM_JOB_ID}" \
        --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
        -m pal.runner.cli run \
        --method "'"${METHODS}"'" \
        --benchmarks "'"${BENCHMARK}"'" \
        --seeds "'"${SEEDS}"'" \
        --epochs "'"${EPOCHS}"'" \
        --batch-size "'"${BATCH_SIZE}"'" \
        --n-eval "'"${N_EVAL}"'" \
        --predict-batch-size "'"${PREDICT_BATCH_SIZE}"'" \
        --device "'"${PAL_DEVICE}"'" \
        --runs-root "'"${RUNS_ROOT}"'" \
        "${wandb_args[@]}" \
        "${pal_extra_args[@]}"
'
