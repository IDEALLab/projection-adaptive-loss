#!/bin/bash -l
#SBATCH --job-name=pal-enforce-v4-warmup
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=<SCRATCH>/logs/pal-enforce-v4-warmup-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-enforce-v4-warmup-%A_%a.err
# ENFORCE v4 warm-up arm on 4x GH200 as a single `pal run` process (enforce_v4 is not DDP-capable).
# Usage: BENCH=e1|e2 sbatch slurm/engineering/enforce_v4_warmup_gpu.sbatch
# Env overrides: SEEDS, EPOCH_START_HARD, MICRO_BATCH, EPOCHS, IFT_BACKWARD, SKIP_NONFINITE. Training only.

set -euo pipefail

BENCH="${BENCH:-e1}"
case "${BENCH}" in
    e1)
        RUN_CONFIG="pal/configs/runs/e1_gpu.yaml"
        BENCHMARK_ID="e1/bwb"
        DEFAULT_EPOCH_START_HARD=100
        DEFAULT_MICRO_BATCH=8
        DOWNLOAD_BENCHMARK=""
        ;;
    e2)
        RUN_CONFIG="pal/configs/runs/e2.yaml"
        BENCHMARK_ID="e2/urban_wind"
        DEFAULT_EPOCH_START_HARD=250
        DEFAULT_MICRO_BATCH=1
        DOWNLOAD_BENCHMARK="e2/urban_wind"
        ;;
    *)
        echo "ERROR: BENCH must be 'e1' or 'e2' (got '${BENCH}')" >&2
        exit 1
        ;;
esac

EPOCH_START_HARD="${EPOCH_START_HARD:-${DEFAULT_EPOCH_START_HARD}}"
MICRO_BATCH="${MICRO_BATCH:-${DEFAULT_MICRO_BATCH}}"
# EPOCHS is passed after --config so it wins over the run-config.
EPOCHS="${EPOCHS:-}"
if [[ -n "${EPOCHS}" ]]; then
    EPOCHS_ARG="--epochs ${EPOCHS}"
else
    EPOCHS_ARG=""
fi

# IFT_BACKWARD=1: run Newton detached with an IFT backward (constant memory in depth).
IFT_BACKWARD="${IFT_BACKWARD:-}"
if [[ "${IFT_BACKWARD}" == "1" ]]; then
    IFT_BACKWARD_ARG="--enforce-v4-ift-backward"
else
    IFT_BACKWARD_ARG=""
fi

# SKIP_NONFINITE=1 skips non-finite steps instead of raising.
SKIP_NONFINITE="${SKIP_NONFINITE:-}"
if [[ "${SKIP_NONFINITE}" == "1" ]]; then
    SKIP_NONFINITE_ARG="--enforce-v4-skip-nonfinite-step"
else
    SKIP_NONFINITE_ARG=""
fi

if [[ -n "${SEEDS:-}" ]]; then
    read -r -a _SEED_LIST <<< "${SEEDS//,/ }"
    SEED="${_SEED_LIST[${SLURM_ARRAY_TASK_ID}]}"
else
    SEED="${SLURM_ARRAY_TASK_ID}"
fi

echo "=== enforce_v4 warm-up: BENCH=${BENCH} (${BENCHMARK_ID}) seed=${SEED} epoch_start_hard=${EPOCH_START_HARD} micro_batch=${MICRO_BATCH} epochs=${EPOCHS:-<yaml>} ift_backward=${IFT_BACKWARD:-<off>} skip_nonfinite=${SKIP_NONFINITE:-<off>} ==="

if [[ -n "${PAL_DIR:-}" ]]; then
    PAL_REPO="$(cd "${PAL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cluster_runtime.sh" ]]; then
    PAL_REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    PAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
source "${PAL_REPO}/scripts/cluster_runtime.sh"

EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gpu-pal-training}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_enforce_v4_warmup_${BENCH}}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

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
    cluster_prepare_runtime

    if [[ "'"${BENCH}"'" == "e2" ]]; then
        export E2_CLEARANCE_PER_PAIR=1
    fi

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
    if [[ -n "'"${DOWNLOAD_BENCHMARK}"'" ]]; then
        python scripts/download_artifacts.py --benchmark "'"${DOWNLOAD_BENCHMARK}"'"
    fi

    pal run \
        --config '"${RUN_CONFIG}"' \
        '"${EPOCHS_ARG}"' \
        --method enforce_v4 \
        --seeds "'"${SEED}"'" \
        --enforce-v4-epoch-start-hard "'"${EPOCH_START_HARD}"'" \
        --enforce-v4-micro-batch "'"${MICRO_BATCH}"'" \
        '"${IFT_BACKWARD_ARG}"' \
        '"${SKIP_NONFINITE_ARG}"' \
        --device cuda \
        --runs-root "'"${RUNS_ROOT}"'" \
        --no-final-eval \
        --wandb-group "enforce_v4_warmup_'"${BENCH}"'__${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        "${wandb_args[@]}"
'
