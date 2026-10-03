#!/bin/bash -l
#SBATCH --job-name=pal-e2-eq
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=<SCRATCH>/logs/pal-e2-eq-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-e2-eq-%A_%a.err

# Equality-constraint variant (2000 epochs, coverage tol=1e-3) of e2.sbatch, trained with 4-GPU DDP.
# Checkpoints every 25 epochs with --auto-resume, so chained resubmissions (scripts/submit_eq_rerun.sh) continue a run.
# Array layout as in e2.sbatch: idx 0-9 pal_loggap, 10-19 alm. Set PAL_GRID_TOTAL=20 for partial arrays.

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
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e2_eq_2000ep}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
MASTER_PORT="${MASTER_PORT:-29500}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
# Grid size for the --shard guard (override for partial arrays).
PAL_GRID_TOTAL="${PAL_GRID_TOTAL:-${SLURM_ARRAY_TASK_COUNT}}"

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
    export PYTHONPATH="/workspace/pal${PYTHONPATH:+:${PYTHONPATH}}"
    source /workspace/pal/scripts/cluster_runtime.sh
    export RUNS_ROOT="'"${RUNS_ROOT}"'"
    export WANDB_PROJECT="'"${WANDB_PROJECT}"'"
    export WANDB_ENTITY="'"${WANDB_ENTITY}"'"
    export E2_CLEARANCE_PER_PAIR=1
    cluster_prepare_runtime

    wandb_args=()
    if cluster_should_enable_wandb; then
        wandb_args+=(--wandb --wandb-project "${WANDB_PROJECT}")
        if [[ -n "${WANDB_ENTITY:-}" ]]; then
            wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
        fi
    fi

    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
    fi
    python scripts/download_artifacts.py --benchmark e2/urban_wind

    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    torchrun \
        --nnodes "${NNODES}" \
        --nproc-per-node "'"${NPROC_PER_NODE}"'" \
        --node-rank "${SLURM_NODEID}" \
        --rdzv-backend c10d \
        --rdzv-id "${SLURM_JOB_ID}" \
        --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
        -m pal.runner.cli run \
        --config pal/configs/runs/e2_eq.yaml \
        --shard "${SLURM_ARRAY_TASK_ID}/'"${PAL_GRID_TOTAL}"'" \
        --runs-root "'"${RUNS_ROOT}"'" \
        --wandb-group "e2_eq__${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        --checkpoint-every 25 \
        --auto-resume \
        "${wandb_args[@]}"
'
