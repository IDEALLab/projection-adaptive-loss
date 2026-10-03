#!/bin/bash -l
#SBATCH --job-name=pal-eng-e4-gpu
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --output=<SCRATCH>/logs/pal-eng-e4-gpu-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e4-gpu-%j.err
# e4/chip_layout on one GH200 node: 4 trained methods x 2 seeds and 20 IPOPT restart shards (seed 0)
# in parallel, then alm_bolton override evals on each alm run.

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
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e4_gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

cluster_resolve_hf_auth || true
cluster_resolve_wandb_auth || true

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail

    cd /workspace/pal
    export PYTHONPATH="/workspace/pal${PYTHONPATH:+:${PYTHONPATH}}"
    # Prefer HPCX libucs over the system copy pulled in by libipopt, else torch import fails.
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
    GROUP="e4_gpu__${SLURM_JOB_ID:-local}"

    # Up to 28 concurrent procs on a 288-core node, ~10 threads each.
    export OMP_NUM_THREADS=10
    export MKL_NUM_THREADS=10
    export OPENBLAS_NUM_THREADS=10
    echo "[diag] torch.get_num_threads() = $(python -c "import torch; print(torch.get_num_threads())" 2>/dev/null || echo unknown)"

    BENCH=e4/chip_layout
    EVAL_POINTS=pal/configs/eval_points/e4.json
    LOG_DIR="${RUNS_ROOT}/_launch_logs"
    mkdir -p "${LOG_DIR}"

    # fsnet uses 100 outer epochs (x 20 L-BFGS steps, Table C.1 FS=100), the others 2000.
    pids=()
    for METHOD in pal_loggap alm fsnet snarenet; do
        if [[ "${METHOD}" == "fsnet" ]]; then
            METHOD_EPOCHS=100
        else
            METHOD_EPOCHS=2000
        fi
        for SEED in 0 1; do
            ( pal run \
                --benchmarks "${BENCH}" \
                --method "${METHOD}" \
                --seeds "${SEED}" \
                --epochs "${METHOD_EPOCHS}" \
                --device cpu \
                --eval-points "${EVAL_POINTS}" \
                --measure-repair-mem \
                --runs-root "${RUNS_ROOT}" \
                --wandb-group "${GROUP}" \
                "${wandb_args[@]}" \
                > "${LOG_DIR}/${METHOD}_s${SEED}.log" 2>&1 ) &
            pids+=($!)
            echo "[launch] ${METHOD} seed=${SEED} pid=${pids[-1]} log=${LOG_DIR}/${METHOD}_s${SEED}.log"
        done
    done

    # IPOPT: 20 restart shards, one multistart each.
    for R in $(seq 0 19); do
        ( pal run \
            --benchmarks "${BENCH}" \
            --method ipopt \
            --seeds 0 \
            --device cpu \
            --multi-start 20 \
            --restart-shard "${R}/20" \
            --eval-points "${EVAL_POINTS}" \
            --runs-root "${RUNS_ROOT}" \
            --wandb-group "${GROUP}" \
            "${wandb_args[@]}" \
            > "${LOG_DIR}/ipopt_r${R}.log" 2>&1 ) &
        pids+=($!)
        echo "[launch] ipopt restart=${R}/20 pid=${pids[-1]} log=${LOG_DIR}/ipopt_r${R}.log"
    done

    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "[launch] pid=${pid} FAILED"
            fail=$((fail+1))
        fi
    done

    if (( fail > 0 )); then
        echo "[launch] phase 1: ${fail} job(s) failed; see ${LOG_DIR}/*.log"
        exit 1
    fi
    echo "[launch] phase 1: all ${#pids[@]} jobs completed OK"

    # alm_bolton override evals, skipping __as_* siblings and alm_bolton dirs.
    echo "[launch] phase 2: alm_bolton override evals"
    oeval_pids=()
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
            --eval-points "${EVAL_POINTS}" \
            --method-override alm_bolton \
            > "${LOG_DIR}/${ALM_NAME}__as_alm_bolton.log" 2>&1 ) &
        oeval_pids+=($!)
        echo "[launch] override-eval ${ALM_NAME} -> __as_alm_bolton pid=${oeval_pids[-1]}"
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
