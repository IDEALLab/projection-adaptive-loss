#!/bin/bash -l
#SBATCH --job-name=pal-ablation
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --output=<SCRATCH>/logs/pal-ablation-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-ablation-%j.err
# Ablation sweep on one GH200 node: plan, then run all rows through run_shard.py in one ProcessPool.
# Default ablation: pal/configs/ablations/s1_s6.yaml.
# Tunables (env): ABLATION, RUNS_ROOT, JOBS_FILE, WORKERS, THREADS, FORCE, RETRY_FAILED, BENCHMARKS.

set -euo pipefail

if [[ -n "${PAL_DIR:-}" ]]; then
    PAL_REPO="$(cd "${PAL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cluster_runtime.sh" ]]; then
    PAL_REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    PAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "${PAL_REPO}/scripts/cluster_runtime.sh"

EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gpu-pal-training}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/ablation_gpu}"
ABLATION="${ABLATION:-pal/configs/ablations/s1_s6.yaml}"
JOBS_FILE="${JOBS_FILE:-${RUNS_ROOT}/jobs.jsonl}"
WORKERS="${WORKERS:-30}"
THREADS="${THREADS:-9}"
FORCE="${FORCE:-0}"
RETRY_FAILED="${RETRY_FAILED:-0}"
BENCHMARKS="${BENCHMARKS:-}"
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

    ABLATION="'"${ABLATION}"'"
    JOBS_FILE="'"${JOBS_FILE}"'"
    WORKERS="'"${WORKERS}"'"
    THREADS="'"${THREADS}"'"
    FORCE="'"${FORCE}"'"
    RETRY_FAILED="'"${RETRY_FAILED}"'"
    BENCHMARKS="'"${BENCHMARKS}"'"

    LOG_DIR="${RUNS_ROOT}/_launch_logs"
    mkdir -p "${LOG_DIR}" "$(dirname "${JOBS_FILE}")"

    plan_args=(--ablation "${ABLATION}" --runs-root "${RUNS_ROOT}" --plan-out "${JOBS_FILE}")
    if [[ "${FORCE}" == "1" ]]; then plan_args+=(--force); fi
    if [[ "${RETRY_FAILED}" == "1" ]]; then plan_args+=(--retry-failed); fi
    if [[ -n "${BENCHMARKS}" ]]; then plan_args+=(--benchmarks "${BENCHMARKS}"); fi

    echo "[plan] python scripts/run_ablation.py ${plan_args[*]}"
    python scripts/run_ablation.py "${plan_args[@]}"
    n_rows=$(wc -l < "${JOBS_FILE}" | tr -d " ")
    if [[ "${n_rows}" -eq 0 ]]; then
        echo "[plan] nothing to do (0 pending rows)."
        exit 0
    fi
    echo "[plan] ${n_rows} pending rows"

    echo "[exec] workers=${WORKERS} threads/worker=${THREADS}"
    python scripts/run_shard.py \
        --jobs "${JOBS_FILE}" \
        --shard 0/1 \
        --workers "${WORKERS}" \
        --threads "${THREADS}" \
        --runs-root "${RUNS_ROOT}" \
        --log-dir "${LOG_DIR}"

    echo "[ablation] done. runs at: ${RUNS_ROOT}"
    echo "[ablation] render with:"
    echo "  python scripts/aggregate.py --runs-root ${RUNS_ROOT} --out ${RUNS_ROOT}/runs.parquet"
    echo "  python scripts/render_ablation.py --ablation ${ABLATION} --runs-parquet ${RUNS_ROOT}/runs.parquet --format latex --out paper/tables/ablation_s1_s6.tex"
'
