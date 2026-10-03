#!/bin/bash -l
#SBATCH --job-name=pal-eng-e3-gpu-alm-fill
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=08:00:00
#SBATCH --output=<SCRATCH>/logs/pal-eng-e3-gpu-alm-fill-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e3-gpu-alm-fill-%j.err
# e3/acopf_ieee57: trains alm x 10 seeds (paper-faithful) in a fresh RUNS_ROOT,
# then runs alm_bolton override evals on each alm run.

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
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e3_gpu_alm_fill}"
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
    GROUP="e3_gpu_alm_fill__${SLURM_JOB_ID:-local}"

    # 10 procs on 288 cores -> 28 threads each (280 used, 8-core headroom).
    export OMP_NUM_THREADS=28
    export MKL_NUM_THREADS=28
    export OPENBLAS_NUM_THREADS=28
    echo "[diag] torch.get_num_threads() = $(python -c "import torch; print(torch.get_num_threads())" 2>/dev/null || echo unknown)"

    BENCH=e3/acopf_ieee57
    LOG_DIR="${RUNS_ROOT}/_launch_logs"
    mkdir -p "${LOG_DIR}"

    pids=()
    for SEED in 0 1 2 3 4 5 6 7 8 9; do
        ( pal run \
            --benchmarks "${BENCH}" \
            --method alm \
            --seeds "${SEED}" \
            --protocol paper-faithful \
            --epochs 1000 \
            --device cpu \
            --measure-repair-mem \
            --runs-root "${RUNS_ROOT}" \
            --wandb-group "${GROUP}" \
            "${wandb_args[@]}" \
            > "${LOG_DIR}/alm_s${SEED}.log" 2>&1 ) &
        pids+=($!)
        echo "[launch] phase 1: alm seed=${SEED} pid=${pids[-1]} log=${LOG_DIR}/alm_s${SEED}.log"
    done

    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "[launch] phase 1: pid=${pid} FAILED"
            fail=$((fail+1))
        fi
    done

    if (( fail > 0 )); then
        echo "[launch] phase 1: ${fail} job(s) failed; see ${LOG_DIR}/*.log"
        exit 1
    fi
    echo "[launch] phase 1: all ${#pids[@]} jobs completed OK"

    echo "[launch] phase 2: alm_bolton override evals"
    oeval_pids=()
    shopt -s nullglob
    for ALM_DIR in "${RUNS_ROOT}"/*_alm_e3-acopf_ieee57_seed*/; do
        ALM_NAME="$(basename "${ALM_DIR%/}")"
        case "${ALM_NAME}" in
            *__as_*|*alm_bolton*) continue;;
        esac
        ( pal eval \
            --run-id "${ALM_NAME}" \
            --runs-root "${RUNS_ROOT}" \
            --device cpu \
            --method-override alm_bolton \
            > "${LOG_DIR}/${ALM_NAME}__as_alm_bolton.log" 2>&1 ) &
        oeval_pids+=($!)
        echo "[launch] phase 2: override-eval ${ALM_NAME} -> __as_alm_bolton pid=${oeval_pids[-1]}"
    done
    shopt -u nullglob

    ofail=0
    for pid in "${oeval_pids[@]}"; do
        if ! wait "${pid}"; then
            echo "[launch] phase 2: pid=${pid} FAILED"
            ofail=$((ofail+1))
        fi
    done

    if (( ofail > 0 )); then
        echo "[launch] phase 2: ${ofail} override-eval(s) failed"
        exit 1
    fi
    echo "[launch] phase 2: all ${#oeval_pids[@]} override evals completed OK"
'
