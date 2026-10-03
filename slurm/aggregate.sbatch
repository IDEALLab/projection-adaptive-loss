#!/bin/bash
#SBATCH --job-name=pal-sweep-aggregate
#SBATCH --account=<ACCOUNT>
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/pal-sweep-aggregate-%j.out
#SBATCH --error=logs/pal-sweep-aggregate-%j.err
# Runs after both row arrays finish (via `--dependency=afterany:<cpu>:<gpu>`).
# Scans the sweep's runs/ and writes results.md.

set -euo pipefail

SWEEP_DIR="${1:?usage: sbatch ... aggregate.sbatch <sweep_dir>}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi

mkdir -p logs

echo "[aggregate] sweep=$SWEEP_DIR"
pal sweep aggregate --sweep-dir "$SWEEP_DIR"
echo "[aggregate] done. results.md written."
