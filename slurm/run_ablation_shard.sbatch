#!/bin/bash
#SBATCH --account=<ACCOUNT>
# Runs one shard of an ablation sweep, submitted by scripts/launch_ablation.sh.
# Required env: SHARD_I, SHARD_N, WORKERS, THREADS, JOBS_FILE, RUNS_ROOT, LOG_DIR.
set -euo pipefail

: "${SHARD_I:?SHARD_I not set}"
: "${SHARD_N:?SHARD_N not set}"
: "${WORKERS:?WORKERS not set}"
: "${THREADS:?THREADS not set}"
: "${JOBS_FILE:?JOBS_FILE not set}"
: "${RUNS_ROOT:?RUNS_ROOT not set}"
: "${LOG_DIR:?LOG_DIR not set}"

# Load the site modules providing gcc and python_cuda here.
: "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
source "$PAL_VENV"
export PYTHONUNBUFFERED=1
# Import pal from the submitting checkout, not the editable install target.
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$HOME/pal}:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR:-$HOME/pal}"

echo "[slurm] host=$(hostname) jobid=${SLURM_JOB_ID:-?} shard=${SHARD_I}/${SHARD_N}"
echo "[slurm] workers=${WORKERS} threads/worker=${THREADS}"
echo "[slurm] jobs=${JOBS_FILE}"
echo "[slurm] runs_root=${RUNS_ROOT}"
echo "[slurm] log_dir=${LOG_DIR}"

python scripts/run_shard.py \
    --jobs "$JOBS_FILE" \
    --shard "${SHARD_I}/${SHARD_N}" \
    --workers "$WORKERS" \
    --threads "$THREADS" \
    --runs-root "$RUNS_ROOT" \
    --log-dir "$LOG_DIR"
