#!/bin/bash -l
#SBATCH --job-name=pal-eng-e1
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=<SCRATCH>/logs/pal-eng-e1-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e1-%A_%a.err
# e1/bwb training on 4x GH200 (DDP), one (method, seed) per array task: slots 0-9 pal_loggap, 10-19 alm.
# EXTRA_RUNNER_ARGS, RUNS_ROOT and WANDB_GROUP_PREFIX are env-overridable. Keep GRID_SIZE in sync with the YAML.
# Training only: evaluate each run (and alm_bolton overrides) with e1_gpu_eval.sbatch.

set -euo pipefail

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
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
MASTER_PORT="${MASTER_PORT:-29500}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-e1}"
EXTRA_RUNNER_ARGS="${EXTRA_RUNNER_ARGS:-}"

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

    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    # e1 eval-time isocontour allocator fragmentation needs expandable_segments.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    torchrun \
        --nnodes "${NNODES}" \
        --nproc-per-node "'"${NPROC_PER_NODE}"'" \
        --node-rank "${SLURM_NODEID}" \
        --rdzv-backend c10d \
        --rdzv-id "${SLURM_JOB_ID}" \
        --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
        -m pal.runner.cli run \
        --config pal/configs/runs/e1_gpu.yaml \
        --shard "${SLURM_ARRAY_TASK_ID}/${GRID_SIZE:-20}" \
        --runs-root "'"${RUNS_ROOT}"'" \
        --wandb-group "'"${WANDB_GROUP_PREFIX}"'__${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        --no-final-eval \
        "${wandb_args[@]}" \
        '"${EXTRA_RUNNER_ARGS}"'
'
