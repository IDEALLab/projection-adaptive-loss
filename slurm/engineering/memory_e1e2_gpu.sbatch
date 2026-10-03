#!/bin/bash -l
#SBATCH --job-name=pal-eng-memory-e1e2
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=02:00:00
#SBATCH --output=<SCRATCH>/logs/pal-eng-memory-e1e2-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-memory-e1e2-%j.err
# GPU memory measurement for e1/bwb + e2/urban_wind at BS=1, one single-GPU subprocess per
# (method, bench, seed), 4 in parallel. fsnet OOM and snarenet failures at BS=1 are expected.

set -euo pipefail

if [[ -n "${PAL_DIR:-}" ]]; then
    PAL_REPO="$(cd "${PAL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cluster_runtime.sh" ]]; then
    PAL_REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    PAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
source "${PAL_REPO}/scripts/cluster_runtime.sh"

EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gpu-pal-training}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_memory_e1e2}"
CONFIG="${CONFIG:-pal/configs/runs/memory_e1e2_bs1.yaml}"
N_GPUS="${N_GPUS:-4}"

cluster_resolve_hf_auth || true
cluster_resolve_wandb_auth || true

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail

    cd /workspace/pal
    export PYTHONPATH="/workspace/pal${PYTHONPATH:+:${PYTHONPATH}}"
    source /workspace/pal/scripts/cluster_runtime.sh
    export RUNS_ROOT="'"${RUNS_ROOT}"'"
    cluster_prepare_runtime

    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
    fi
    python scripts/download_artifacts.py --benchmark e2/urban_wind || true

    python scripts/run_memory_experiment.py \
        --config "'"${CONFIG}"'" \
        --runs-root "'"${RUNS_ROOT}"'" \
        --n-gpus "'"${N_GPUS}"'"

    echo
    echo "=== aggregate ==="
    python scripts/aggregate_memory_results.py "'"${RUNS_ROOT}"'/_results.csv"
'
