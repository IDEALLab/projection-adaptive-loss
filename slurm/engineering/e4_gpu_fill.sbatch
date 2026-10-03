#!/bin/bash -l
#SBATCH --job-name=pal-eng-e4-fill
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=04:00:00
#SBATCH --output=<SCRATCH>/logs/pal-eng-e4-fill-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e4-fill-%j.err
# Tops up e4/chip_layout seeds in an existing RUNS_ROOT, then runs alm_bolton
# override evals, aggregation and table rendering over old and new runs.

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
    GROUP="e4_gpu_fill__${SLURM_JOB_ID:-local}"

    # 24 procs x 10 threads on a 288-core node.
    export OMP_NUM_THREADS=10
    export MKL_NUM_THREADS=10
    export OPENBLAS_NUM_THREADS=10
    echo "[diag] torch.get_num_threads() = $(python -c "import torch; print(torch.get_num_threads())" 2>/dev/null || echo unknown)"

    BENCH=e4/chip_layout
    EVAL_POINTS=pal/configs/eval_points/e4.json
    LOG_DIR="${RUNS_ROOT}/_fill_logs"
    mkdir -p "${LOG_DIR}"

    pids=()
    declare -A METHOD_SEEDS=(
        [pal_loggap]="2 3 4 5 6 7 8 9"
        [alm]="2 3 4 5 6 7 8 9"
        [snarenet]="2 3 4"
        [fsnet]="0 1 2 3 4"
    )
    for METHOD in pal_loggap alm snarenet fsnet; do
        if [[ "${METHOD}" == "fsnet" ]]; then
            METHOD_EPOCHS=100   # Table C.1 setting (FS=100)
        else
            METHOD_EPOCHS=2000
        fi
        for SEED in ${METHOD_SEEDS[$METHOD]}; do
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
            echo "[fill] ${METHOD} seed=${SEED} epochs=${METHOD_EPOCHS} pid=${pids[-1]} log=${LOG_DIR}/${METHOD}_s${SEED}.log"
        done
    done

    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "[fill] phase 1 pid=${pid} FAILED"
            fail=$((fail+1))
        fi
    done
    if (( fail > 0 )); then
        echo "[fill] phase 1: ${fail} job(s) failed; see ${LOG_DIR}/*.log"
        exit 1
    fi
    echo "[fill] phase 1: all ${#pids[@]} jobs completed OK"

    # alm_bolton override evals for alm runs without an __as_alm_bolton sibling.
    echo "[fill] phase 2: alm_bolton override evals (new alm seeds only)"
    oeval_pids=()
    shopt -s nullglob
    for ALM_DIR in "${RUNS_ROOT}"/*_alm_*/; do
        ALM_NAME="$(basename "${ALM_DIR%/}")"
        case "${ALM_NAME}" in
            *_alm_bolton_*|*__as_*) continue;;
        esac
        if [[ -d "${RUNS_ROOT}/${ALM_NAME}__as_alm_bolton" ]]; then
            echo "[fill] skip override-eval ${ALM_NAME} (sibling exists)"
            continue
        fi
        ( pal eval \
            --run-id "${ALM_NAME}" \
            --runs-root "${RUNS_ROOT}" \
            --device cpu \
            --eval-points "${EVAL_POINTS}" \
            --method-override alm_bolton \
            > "${LOG_DIR}/${ALM_NAME}__as_alm_bolton.log" 2>&1 ) &
        oeval_pids+=($!)
        echo "[fill] override-eval ${ALM_NAME} pid=${oeval_pids[-1]}"
    done
    shopt -u nullglob

    ofail=0
    for pid in "${oeval_pids[@]}"; do
        if ! wait "${pid}"; then
            echo "[fill] phase 2: pid=${pid} FAILED"
            ofail=$((ofail+1))
        fi
    done
    if (( ofail > 0 )); then
        echo "[fill] phase 2: ${ofail} override-eval(s) failed"
        exit 1
    fi
    echo "[fill] phase 2: ${#oeval_pids[@]} override evals OK"

    echo "[fill] aggregate: build e4 gap parquet"
    OUT_DIR="results/$(date +%Y-%m-%d)/engineering"
    mkdir -p "${OUT_DIR}"
    python scripts/aggregate_engineering.py \
        --bench "${BENCH}" \
        --runs-root "${RUNS_ROOT}" \
        --out "${OUT_DIR}/e4_chip_layout_gap.parquet"

    echo "[fill] render: build .md + .tex tables"
    python scripts/render_engineering.py \
        --bench "${BENCH}" \
        --gap-parquet "${OUT_DIR}/e4_chip_layout_gap.parquet" \
        --out-dir "${OUT_DIR}"

    echo "[fill] done. outputs in ${OUT_DIR}/"
    ls -la "${OUT_DIR}/" | head -20
'
