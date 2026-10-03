#!/bin/bash
# SLURM array launcher for the repair-step ablation (SQP / interior-point vs. LM
# projector): 3 methods x 6 synthetic benches x 10 seeds = 180 tasks.

#SBATCH --job-name=pal_repair_ablation
#SBATCH --account=<ACCOUNT>
#SBATCH --array=0-179
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --output=<SCRATCH>/pal_repair_ablation/logs/task_%A_%a.out
#SBATCH --error=<SCRATCH>/pal_repair_ablation/logs/task_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Cluster-specific settings: edit before submitting.
REPO_ROOT="${REPO_ROOT:-$HOME/pal-sqp-innerpoint}"

# Load the site modules providing gcc and python_cuda here.

# Venv needs qpsolvers, osqp>=1.0 and clarabel (pal_sqp).
source "${PAL_VENV:?PAL_VENV must be set to the venv activate script on the x86 cluster, e.g. PAL_VENV=\$VENV/bin/activate}"

# Shared runs/ dir for all tasks; pass the same path to aggregate.py --campaign-root.
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/pal_repair_ablation/runs}"

EPOCHS=2000

METHODS=(pal_loggap pal_sqp pal_ip)
BENCHES=(s1_sphere_track s2_active_set_switch s3_illcond_tube s4_qv_coupling s5_overdetermined s6_redundant_ineq)
N_SEEDS=10

mkdir -p "$RUNS_ROOT"
cd "$REPO_ROOT"

# idx = method_idx*60 + bench_idx*10 + seed
idx="${SLURM_ARRAY_TASK_ID:?this script must be run as a SLURM array task}"
n_bench=${#BENCHES[@]}
per_method=$(( n_bench * N_SEEDS ))
method_idx=$(( idx / per_method ))
rem=$(( idx % per_method ))
bench_idx=$(( rem / N_SEEDS ))
seed=$(( rem % N_SEEDS ))

METHOD="${METHODS[$method_idx]}"
BENCH="${BENCHES[$bench_idx]}"

extra_args=()
if [[ "$METHOD" == "pal_ip" ]]; then
    extra_args+=(--set proj_ip_mu0=0.01)
fi

echo "[task $idx] method=$METHOD bench=$BENCH seed=$seed epochs=$EPOCHS node=$(hostname) start=$(date -Iseconds)"

# Bound BLAS/OMP thread pools to the --cpus-per-task budget.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python -m pal.runner.cli run \
    --method "$METHOD" \
    --benchmarks "$BENCH" \
    --seeds "$seed" \
    --epochs "$EPOCHS" \
    --device cpu \
    --runs-root "$RUNS_ROOT" \
    "${extra_args[@]}"

echo "[task $idx] done=$(date -Iseconds)"
