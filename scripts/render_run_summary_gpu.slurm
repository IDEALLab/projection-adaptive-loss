#!/bin/bash -l
#SBATCH --job-name=pal-render
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=<SCRATCH>/logs/pal-render-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-render-%j.err

set -euo pipefail

EDF_ENVIRONMENT="${EDF_ENVIRONMENT:-gpu-pal-training}"
RUN_GROUP="${RUN_GROUP:-smoke}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/${RUN_GROUP}}"
RUNS_PARQUET="${RUNS_PARQUET:-${RUNS_ROOT}/runs.parquet}"
RESULTS_MD="${RESULTS_MD:-${RUNS_ROOT}/results.md}"
PAPER_RESULTS_DIR="${PAPER_RESULTS_DIR:-${RUNS_ROOT}/paper_results}"
F_STAR_PATH="${F_STAR_PATH:-}"

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail

    cd /workspace/pal
    mkdir -p "'"${RUNS_ROOT}"'" "'"${PAPER_RESULTS_DIR}"'"

    echo "=== PAL Render Summary ==="
    echo "Repo:        $(pwd)"
    echo "Runs root:   '"${RUNS_ROOT}"'"
    echo "Runs parquet:'"${RUNS_PARQUET}"'"
    echo "Results md:  '"${RESULTS_MD}"'"
    echo "Paper dir:   '"${PAPER_RESULTS_DIR}"'"
    echo "f*:          '"${F_STAR_PATH:-<default>}"'"
    echo "=========================="

    python scripts/aggregate.py \
        --runs-root "'"${RUNS_ROOT}"'" \
        --out "'"${RUNS_PARQUET}"'"

    python scripts/render_results.py \
        --runs-parquet "'"${RUNS_PARQUET}"'" \
        --out "'"${RESULTS_MD}"'"

    render_args=(
        --runs-parquet "'"${RUNS_PARQUET}"'"
        --out-dir "'"${PAPER_RESULTS_DIR}"'"
    )
    if [[ -n "'"${F_STAR_PATH}"'" ]]; then
        render_args+=(--f-star "'"${F_STAR_PATH}"'")
    fi

    python scripts/render_paper_tables.py "${render_args[@]}"
'
