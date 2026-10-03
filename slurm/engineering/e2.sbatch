#!/bin/bash -l
#SBATCH --job-name=pal-e2
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=<SCRATCH>/logs/pal-e2-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-e2-%A_%a.err

# e2/urban_wind training, pal_loggap and alm x 10 seeds: idx 0-9 pal_loggap, 10-19 alm.
# For a partial array set PAL_GRID_TOTAL=20, e.g. PAL_GRID_TOTAL=20 sbatch --array=3-9,13-19 ...

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
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e2_500ep}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
# Grid size for the --shard guard (override for partial arrays).
PAL_GRID_TOTAL="${PAL_GRID_TOTAL:-${SLURM_ARRAY_TASK_COUNT}}"

cluster_resolve_hf_auth || true
cluster_resolve_wandb_auth || true

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
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

    pal run \
        --config pal/configs/runs/e2.yaml \
        --shard "${SLURM_ARRAY_TASK_ID}/'"${PAL_GRID_TOTAL}"'" \
        --runs-root "'"${RUNS_ROOT}"'" \
        --wandb-group "e2__${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        "${wandb_args[@]}"
'
