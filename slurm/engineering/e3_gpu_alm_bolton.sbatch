#!/bin/bash -l
#SBATCH --job-name=pal-eng-e3-alm-bolton
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=01:00:00
#SBATCH --output=<SCRATCH>/logs/pal-eng-e3-alm-bolton-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e3-alm-bolton-%j.err
# alm_bolton override evals over existing e3 alm runs in eng_e3_gpu/ (no retraining).

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
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e3_gpu}"

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail

    cd /workspace/pal
    export PYTHONPATH="/workspace/pal${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    source /workspace/pal/scripts/cluster_runtime.sh
    export RUNS_ROOT="'"${RUNS_ROOT}"'"
    cluster_prepare_runtime

    export OMP_NUM_THREADS=64
    export MKL_NUM_THREADS=64
    export OPENBLAS_NUM_THREADS=64

    LOG_DIR="${RUNS_ROOT}/_launch_logs"
    mkdir -p "${LOG_DIR}"

    echo "[launch] alm_bolton override evals over existing alm runs"
    pids=()
    shopt -s nullglob
    for ALM_DIR in "${RUNS_ROOT}"/*_alm_*/; do
        ALM_NAME="$(basename "${ALM_DIR%/}")"
        case "${ALM_NAME}" in
            *_alm_bolton_*|*__as_*) continue;;
        esac
        ( pal eval \
            --run-id "${ALM_NAME}" \
            --runs-root "${RUNS_ROOT}" \
            --device cpu \
            --method-override alm_bolton \
            > "${LOG_DIR}/${ALM_NAME}__as_alm_bolton.log" 2>&1 ) &
        pids+=($!)
        echo "[launch] override-eval ${ALM_NAME} -> __as_alm_bolton pid=${pids[-1]}"
    done
    shopt -u nullglob

    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "[launch] pid=${pid} FAILED"
            fail=$((fail+1))
        fi
    done

    if (( fail > 0 )); then
        echo "[launch] ${fail} override-eval(s) failed"
        exit 1
    fi
    echo "[launch] all ${#pids[@]} override evals completed OK"
'
