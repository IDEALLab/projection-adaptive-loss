#!/bin/bash
#SBATCH --job-name=pal-e3-bl-eval
#SBATCH --account=<ACCOUNT>
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=<SCRATCH>/logs/pal-e3-bl-eval-%A_%a.out
#SBATCH --error=<SCRATCH>/logs/pal-e3-bl-eval-%A_%a.err
# Per-(config, seed) table rows for e3 baseline runs (index scheme of e3_baselines_x86.sbatch). Merge with:
#   python scripts/e3_paper_table.py <RUNS_ROOT>/<cfg> --from-json '<RUNS_ROOT>/<cfg>/_table/seed*.json' --out table.md
set -euo pipefail
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set}"
fi
: "${PAL_REPO:?PAL_REPO must be set}"
export PYTHONPATH="$PAL_REPO" PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$PAL_REPO"
CONFIGS=(fsnet snarenet enforce_v4 enforce_v4_nogate)
N_SEEDS="${N_SEEDS:-10}"; SEED_OFFSET="${SEED_OFFSET:-0}"
idx="${SLURM_ARRAY_TASK_ID:-0}"
cfg="${CONFIGS[$((idx / N_SEEDS))]}"; seed=$((SEED_OFFSET + idx % N_SEEDS))
ROOT="${RUNS_ROOT:-$SCRATCH/pal_engineering/e3_baselines}/$cfg"
mkdir -p "$ROOT/_table"
echo "[e3-bl-eval] cfg=$cfg seed=$seed root=$ROOT sha=$(git rev-parse --short HEAD)"
python scripts/e3_paper_table.py "$ROOT" --seeds "$seed" --include-failed \
    --json-out "$ROOT/_table/seed${seed}.json" ${EXTRA_ARGS:-}
