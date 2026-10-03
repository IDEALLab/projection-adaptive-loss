#!/bin/bash -l
#SBATCH --job-name=pal-eng-e1-ipopt-gpu
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --output=<SCRATCH>/logs/pal-eng-e1-ipopt-gpu-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e1-ipopt-gpu-%j.err
# IPOPT on e1/bwb in the GH200 IPOPT image: seed 0, 20 restart shards in parallel on CPU.
# Shards are merged into one row per (method, bench, seed) by shard_merge.py.

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
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e1_ipopt_gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
E1_N_STATIONS="${E1_N_STATIONS:-8}"
BWB_FAST_SDF="${BWB_FAST_SDF:-1}"

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
    export E1_N_STATIONS="'"${E1_N_STATIONS}"'"
    export BWB_FAST_SDF="'"${BWB_FAST_SDF}"'"
    cluster_prepare_runtime

    wandb_args=()
    if cluster_should_enable_wandb; then
        wandb_args+=(--wandb --wandb-project "${WANDB_PROJECT}")
        if [[ -n "${WANDB_ENTITY:-}" ]]; then
            wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
        fi
    fi
    GROUP="e1_ipopt_gpu__${SLURM_JOB_ID:-local}"

    # 20 parallel shards x ~14 threads each on a 288-core node.
    export OMP_NUM_THREADS=14
    export MKL_NUM_THREADS=14
    export OPENBLAS_NUM_THREADS=14
    echo "[diag] torch.get_num_threads() = $(python -c "import torch; print(torch.get_num_threads())" 2>/dev/null || echo unknown)"

    BENCH=e1/bwb
    EVAL_POINTS=pal/configs/eval_points/e1.json
    LOG_DIR="${RUNS_ROOT}/_launch_logs"
    mkdir -p "${LOG_DIR}"

    pids=()
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
        echo "[launch] ${fail} restart shard(s) failed; see ${LOG_DIR}/*.log"
        exit 1
    fi
    echo "[launch] all ${#pids[@]} restart shards completed OK"
'
