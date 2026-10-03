#!/bin/bash -l
#SBATCH --job-name=pal-eng-e3-loggap-initstd
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=00:25:00
#SBATCH --output=<SCRATCH>/logs/pal-eng-e3-loggap-initstd-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-eng-e3-loggap-initstd-%j.err
# Diagnostic: pal_loggap on e3/acopf_ieee57, seed 0, paper-faithful, with a smaller output init std
# so the tanh head is not saturated at init. Only the early training dynamics matter.

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
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/eng_e3_loggap_initstd}"

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

    # sqrt(4/200) / 4: raw outputs ~N(0, 0.5) instead of ~N(0, 2).
    export E3_OUTPUT_INIT_STD=0.035
    # Logs per-component gradients at y (three extra autograd.grad calls per minibatch).
    export E3_LOG_GRAD_DIAG=1

    LOG_DIR="${RUNS_ROOT}/_launch_logs"
    mkdir -p "${LOG_DIR}"

    echo "[launch] pal_loggap x e3/acopf_ieee57, paper-faithful, seed 0, E3_OUTPUT_INIT_STD=${E3_OUTPUT_INIT_STD}"
    pal run \
        --benchmarks e3/acopf_ieee57 \
        --method pal_loggap \
        --seeds 0 \
        --protocol paper-faithful \
        --device cpu \
        --measure-repair-mem \
        --runs-root "${RUNS_ROOT}" \
        > "${LOG_DIR}/pal_loggap_s0_initstd.log" 2>&1 || echo "[launch] training exited (walltime or error), partial log retained"
    echo "[launch] done; metrics in ${RUNS_ROOT}/<run_id>/metrics.jsonl"
    ls -la "${RUNS_ROOT}/"
'
