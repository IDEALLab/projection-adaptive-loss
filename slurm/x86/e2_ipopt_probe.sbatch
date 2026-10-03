#!/bin/bash
#SBATCH --job-name=pal-e2-ipopt-probe
#SBATCH --account=<ACCOUNT>
#SBATCH --gpus=<GPU_TYPE>:1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=<SCRATCH>/logs/pal-e2-ipopt-probe-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-e2-ipopt-probe-%j.err
# VRAM and latency probe for the e2 surrogate at batch 1 with K=49 per-pair constraints.
# Prints peak_gpu_mem_GB, per_jac_row_s and est_solve_wall. No IPOPT run, no W&B.

set -euo pipefail

# Load the site modules providing gcc and ipopt here.

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi

mkdir -p $SCRATCH/logs

export PAL_IPOPT_JAC_MODE=loop
export E2_CLEARANCE_PER_PAIR=1  # K=49 per-pair constraints

if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
fi
python scripts/download_artifacts.py --benchmark e2/urban_wind

echo "=== probe @ max_iter=500 (main) ==="
python scripts/probe_e2_ipopt_vram.py --device cuda --forwards 5 --jacs 3 --max-iter 500
echo "=== projected wall @ max_iter=2000 (control) ==="
python scripts/probe_e2_ipopt_vram.py --device cuda --forwards 3 --jacs 2 --max-iter 2000
