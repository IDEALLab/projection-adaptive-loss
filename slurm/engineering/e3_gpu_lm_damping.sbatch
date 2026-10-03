#!/bin/bash -l
#SBATCH --job-name=pal-eng-e3-lm-damping
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --output=<SCRATCH>/logs/pal-eng-e3-lm-damping-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e3-lm-damping-%j.err
# e3/acopf_ieee57 LM damping experiment: pal_loggap + dc3, seeds 0-2, 1000 epochs,
# with proj_lambda_min=1e-3 and proj_delta=1e-1 against eval-time overshoot.

set -euo pipefail

if [[ -n "${PAL_DIR:-}" ]]; then
    PAL_REPO="$(cd "${PAL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cluster_runtime.sh" ]]; then
    PAL_REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    PAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
source "${PAL_REPO}/scripts/cluster_runtime.sh"

EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gpu-pal-ipopt}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e3_lm_damping}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

cluster_resolve_hf_auth || true
cluster_resolve_wandb_auth || true

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail

    cd /workspace/pal
    export PYTHONPATH="/workspace/pal${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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
    GROUP="e3_lm_damping__${SLURM_JOB_ID:-local}"

    # 6 procs total (2 methods x 3 seeds) on a 288-core node -> ~48 threads each.
    export OMP_NUM_THREADS=48
    export MKL_NUM_THREADS=48
    export OPENBLAS_NUM_THREADS=48
    echo "[diag] torch.get_num_threads() = $(python -c "import torch; print(torch.get_num_threads())" 2>/dev/null || echo unknown)"

    BENCH=e3/acopf_ieee57
    LOG_DIR="${RUNS_ROOT}/_launch_logs"
    mkdir -p "${LOG_DIR}"

    pids=()
    for METHOD in pal_loggap dc3; do
        for SEED in 0 1 2; do
            ( pal run \
                --benchmarks "${BENCH}" \
                --method "${METHOD}" \
                --seeds "${SEED}" \
                --protocol paper-faithful \
                --epochs 1000 \
                --device cpu \
                --measure-repair-mem \
                --runs-root "${RUNS_ROOT}" \
                --wandb-group "${GROUP}" \
                --set proj_lambda_min=1e-3 \
                --set proj_delta=1e-1 \
                "${wandb_args[@]}" \
                > "${LOG_DIR}/${METHOD}_s${SEED}.log" 2>&1 ) &
            pids+=($!)
            echo "[launch] ${METHOD} seed=${SEED} pid=${pids[-1]} log=${LOG_DIR}/${METHOD}_s${SEED}.log"
        done
    done

    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "[launch] pid=${pid} FAILED"
            fail=$((fail+1))
        fi
    done

    if (( fail > 0 )); then
        echo "[launch] ${fail} job(s) failed; see ${LOG_DIR}/*.log"
        exit 1
    fi
    echo "[launch] all ${#pids[@]} jobs completed OK"
'
