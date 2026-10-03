#!/bin/bash -l
#SBATCH --job-name=pal-ipopt
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=<SCRATCH>/logs/pal-ipopt-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-ipopt-%j.err

set -euo pipefail

if [[ -n "${PAL_DIR:-}" ]]; then
    PAL_REPO="$(cd "${PAL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cluster_runtime.sh" ]]; then
    PAL_REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    PAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "${PAL_REPO}/scripts/cluster_runtime.sh"

EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gpu-pal-ipopt}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/ipopt}"
SEEDS="${SEEDS:-0}"
BENCHMARK="${BENCHMARK:?set BENCHMARK to a registered benchmark id}"
PAL_DEVICE="${PAL_DEVICE:-cuda}"
PAL_EXTRA_ARGS="${PAL_EXTRA_ARGS:-}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"
# Defaults match nlp_adapter._JAC_MODE / solver._ITER_LOG_EVERY.
PAL_IPOPT_JAC_MODE="${PAL_IPOPT_JAC_MODE:-jacrev}"
PAL_IPOPT_ITER_LOG_EVERY="${PAL_IPOPT_ITER_LOG_EVERY:-10}"

cluster_resolve_hf_auth || true

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail

    cd /workspace/pal
    source /workspace/pal/scripts/cluster_runtime.sh
    export RUNS_ROOT="'"${RUNS_ROOT}"'"
    export PAL_IPOPT_JAC_MODE="'"${PAL_IPOPT_JAC_MODE}"'"
    export PAL_IPOPT_ITER_LOG_EVERY="'"${PAL_IPOPT_ITER_LOG_EVERY}"'"
    cluster_prepare_runtime

    pal_extra_args=()
    if [[ -n "'"${PAL_EXTRA_ARGS}"'" ]]; then
        read -r -a pal_extra_args <<< "'"${PAL_EXTRA_ARGS}"'"
    fi

    echo "=== PAL IPOPT Baseline ==="
    echo "Repo:       $(pwd)"
    echo "Benchmark:  '"${BENCHMARK}"'"
    echo "Seeds:      '"${SEEDS}"'"
    echo "Device:     '"${PAL_DEVICE}"'"
    echo "Runs root:  '"${RUNS_ROOT}"'"
    echo "HF token:   ${HF_TOKEN:+present}"
    echo "Extra args: '"${PAL_EXTRA_ARGS}"'"
    echo "=========================="

    python scripts/probe_cyipopt.py

    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
    fi

    case "'"${BENCHMARK}"'" in
        e1/*|e2/*)
            python scripts/download_artifacts.py --benchmark "'"${BENCHMARK}"'"
            ;;
        *)
            echo "[ipopt] no artifact preflight for '"${BENCHMARK}"'"
            ;;
    esac

    pal run \
        --method ipopt \
        --benchmarks "'"${BENCHMARK}"'" \
        --seeds "'"${SEEDS}"'" \
        --device "'"${PAL_DEVICE}"'" \
        --runs-root "'"${RUNS_ROOT}"'" \
        "${pal_extra_args[@]}"
'
