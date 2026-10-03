#!/bin/bash -l
#SBATCH --job-name=pal-eng-e4-finish
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=00:30:00
#SBATCH --output=<SCRATCH>/logs/pal-eng-e4-finish-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e4-finish-%j.err
# Finishes a e4_gpu run: alm_bolton override evals, aggregation and table rendering.

set -euo pipefail

PAL_REPO="${PAL_DIR:-$SCRATCH/pal}"
source "${PAL_REPO}/scripts/cluster_runtime.sh"

EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gpu-pal-ipopt}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e4_gpu}"

cluster_resolve_hf_auth || true

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail
    cd /workspace/pal
    export PYTHONPATH="/workspace/pal${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    source /workspace/pal/scripts/cluster_runtime.sh
    export RUNS_ROOT="'"${RUNS_ROOT}"'"
    cluster_prepare_runtime

    BENCH=e4/chip_layout
    EVAL_POINTS=pal/configs/eval_points/e4.json
    LOG_DIR="${RUNS_ROOT}/_finish_logs"
    mkdir -p "${LOG_DIR}"

    echo "[finish] phase 2: alm_bolton override evals"
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
            --eval-points "${EVAL_POINTS}" \
            --method-override alm_bolton \
            > "${LOG_DIR}/${ALM_NAME}__as_alm_bolton.log" 2>&1 ) &
        pids+=($!)
        echo "[finish] override-eval ${ALM_NAME} pid=${pids[-1]}"
    done
    shopt -u nullglob

    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "[finish] phase 2 pid=${pid} FAILED"
            fail=$((fail+1))
        fi
    done
    if (( fail > 0 )); then
        echo "[finish] phase 2: ${fail} override-eval(s) failed; aborting"
        exit 1
    fi
    echo "[finish] phase 2: ${#pids[@]} override evals OK"

    # --runs-root must point at the dir holding the trained runs.
    echo "[finish] aggregate: build e4 gap parquet"
    OUT_DIR="results/$(date +%Y-%m-%d)/engineering"
    mkdir -p "${OUT_DIR}"
    python scripts/aggregate_engineering.py \
        --bench "${BENCH}" \
        --runs-root "${RUNS_ROOT}" \
        --out "${OUT_DIR}/e4_chip_layout_gap.parquet"

    echo "[finish] render: build .md + .tex tables"
    python scripts/render_engineering.py \
        --bench "${BENCH}" \
        --gap-parquet "${OUT_DIR}/e4_chip_layout_gap.parquet" \
        --out-dir "${OUT_DIR}"

    echo "[finish] done. outputs in ${OUT_DIR}/"
    ls -la "${OUT_DIR}/" | head -20
'
