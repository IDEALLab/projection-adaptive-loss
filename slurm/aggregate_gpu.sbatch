#!/bin/bash -l
#SBATCH --job-name=pal-aggregate
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=<SCRATCH>/logs/pal-aggregate-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-aggregate-%j.err
# Runs scripts/aggregate.py on RUNS_ROOT and optionally a render script.
# Usage: RUNS_ROOT=... RENDER=paper_heatmap sbatch slurm/aggregate_gpu.sbatch
# RENDER: paper_heatmap | ablation_s1_s6 | none (aggregate only).

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
RUNS_ROOT="${RUNS_ROOT:?RUNS_ROOT must be set}"
RENDER="${RENDER:-none}"

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail
    cd /workspace/pal
    export PYTHONPATH="/workspace/pal${PYTHONPATH:+:${PYTHONPATH}}"
    RUNS_ROOT="'"${RUNS_ROOT}"'"
    RENDER="'"${RENDER}"'"

    python scripts/aggregate.py --runs-root "${RUNS_ROOT}" --out "${RUNS_ROOT}/runs.parquet"

    case "${RENDER}" in
      paper_heatmap)
        python scripts/render_paper_heatmap.py \
            --runs-parquet "${RUNS_ROOT}/runs.parquet" \
            --out "${RUNS_ROOT}/paper_cost_table.tex"
        echo "[render] wrote ${RUNS_ROOT}/paper_cost_table.tex"
        ;;
      ablation_s1_s6)
        python scripts/render_ablation.py \
            --ablation pal/configs/ablations/s1_s6.yaml \
            --runs-parquet "${RUNS_ROOT}/runs.parquet" \
            --format latex \
            --out "${RUNS_ROOT}/ablation_s1_s6.tex"
        ;;
      none) ;;
      *) echo "[render] unknown RENDER=${RENDER}; skipping"; ;;
    esac
'
