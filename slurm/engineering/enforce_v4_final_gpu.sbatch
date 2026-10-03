#!/bin/bash -l
#SBATCH --job-name=pal-enforce-v4-final
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=<SCRATCH>/logs/pal-enforce-v4-final-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-enforce-v4-final-%A_%a.err
# ENFORCE v4 on E1 (e1/bwb) and E2 (e2/urban_wind), 10 seeds each, as a single `pal run` per node.
# Fixed protocol: soft warm-up for half the epochs, micro-batch 1, IFT backward, skip non-finite steps.
# Usage: [ARM=nowarmup] BENCH=e1|e2 sbatch slurm/engineering/enforce_v4_final_gpu.sbatch. Training only.

set -euo pipefail

BENCH="${BENCH:-e1}"
case "${BENCH}" in
    e1)
        RUN_CONFIG="pal/configs/runs/e1_gpu.yaml"
        BENCHMARK_ID="e1/bwb"
        EPOCH_START_HARD=100
        DOWNLOAD_BENCHMARK=""
        ;;
    e2)
        RUN_CONFIG="pal/configs/runs/e2.yaml"
        BENCHMARK_ID="e2/urban_wind"
        EPOCH_START_HARD=250
        DOWNLOAD_BENCHMARK="e2/urban_wind"
        ;;
    *)
        echo "ERROR: BENCH must be 'e1' or 'e2' (got '${BENCH}')" >&2
        exit 1
        ;;
esac

SEED="${SLURM_ARRAY_TASK_ID}"

# ARM=warmup projects from EPOCH_START_HARD, ARM=nowarmup from epoch 0.
ARM="${ARM:-warmup}"
case "${ARM}" in
    warmup) ;;
    nowarmup) EPOCH_START_HARD=0 ;;
    *)
        echo "ERROR: ARM must be 'warmup' or 'nowarmup' (got '${ARM}')" >&2
        exit 1
        ;;
esac

echo "=== enforce_v4 final: BENCH=${BENCH} (${BENCHMARK_ID}) arm=${ARM} seed=${SEED} epoch_start_hard=${EPOCH_START_HARD} micro_batch=1 epochs=<yaml> ift_backward=on skip_nonfinite=on ==="

if [[ -n "${PAL_DIR:-}" ]]; then
    PAL_REPO="$(cd "${PAL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cluster_runtime.sh" ]]; then
    PAL_REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    PAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
source "${PAL_REPO}/scripts/cluster_runtime.sh"

EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gpu-pal-training}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_enforce_v4_final_${ARM}_${BENCH}}"
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
        --method enforce_v4 \
        --seeds "'"${SEED}"'" \
        --enforce-v4-epoch-start-hard "'"${EPOCH_START_HARD}"'" \
        --enforce-v4-micro-batch 1 \
        --enforce-v4-ift-backward \
        --enforce-v4-skip-nonfinite-step \
        --device cuda \
        --runs-root "'"${RUNS_ROOT}"'" \
        --no-final-eval \
        --wandb-group "enforce_v4_final_'"${ARM}"'_'"${BENCH}"'__${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        "${wandb_args[@]}"
'
