#!/bin/bash
#SBATCH --job-name=pal-e3-bl
#SBATCH --account=<ACCOUNT>
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=<SCRATCH>/logs/pal-e3-bl-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-e3-bl-%A_%a.err
# e3/acopf_ieee57 baselines under --protocol paper-faithful, idx = cfg_i * N_SEEDS + seed_i over
# CONFIGS (enforce_v4_nogate sets ada_np_auto_activation=False). Submit with --array=0-39.
# Env: PAL_VENV, PAL_REPO (required), RUNS_ROOT, N_SEEDS, SEED_OFFSET, EPOCHS, EXTRA_ARGS.

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi
: "${PAL_REPO:?PAL_REPO must be set to the checkout to run from (PYTHONPATH pin)}"
export PYTHONPATH="$PAL_REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$PAL_REPO"

mkdir -p $SCRATCH/logs
export WANDB_MODE="${WANDB_MODE:-disabled}"

CONFIGS=(fsnet snarenet enforce_v4 enforce_v4_nogate)
N_SEEDS="${N_SEEDS:-10}"
SEED_OFFSET="${SEED_OFFSET:-0}"
idx="${SLURM_ARRAY_TASK_ID:-0}"
cfg="${CONFIGS[$((idx / N_SEEDS))]}"
seed=$((SEED_OFFSET + idx % N_SEEDS))

method="$cfg"
set_args=()
if [[ "$cfg" == "enforce_v4_nogate" ]]; then
    method="enforce_v4"
    set_args=(--set ada_np_auto_activation=False)
fi
if [[ "$cfg" == "snarenet" ]]; then
    # vmap Jacobian (same Newton math, faster on CPU). newton_maxiter=5 bounds cost: the
    # LM repair does not converge at 100 either, so the returned point is unchanged.
    set_args=(--set jacobian_mode=vmap --set newton_maxiter=5)
fi
epoch_args=()
if [[ -n "${EPOCHS:-}" ]]; then
    epoch_args=(--epochs "$EPOCHS")
fi

RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/pal_engineering/e3_baselines}/$cfg"
mkdir -p "$RUNS_ROOT"
echo "[e3-bl] cfg=$cfg method=$method seed=$seed runs_root=$RUNS_ROOT sha=$(git rev-parse --short HEAD)"

python -c "import pal, sys; print('[e3-bl] pal from', pal.__file__)"

pal run \
    --benchmarks e3/acopf_ieee57 \
    --method "$method" \
    --seeds "$seed" \
    --protocol paper-faithful \
    --device cpu \
    --measure-repair-mem \
    --runs-root "$RUNS_ROOT" \
    "${set_args[@]}" "${epoch_args[@]}" ${EXTRA_ARGS:-}
