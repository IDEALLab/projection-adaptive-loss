#!/bin/bash
# Plan an ablation sweep with run_ablation.py and submit one sbatch job per shard (run_shard.py).
# --max-parallel-benchmarks moves the named benches into a separate 1-row-per-task slab.
# Usage: scripts/launch_ablation.sh --ablation <yaml> [flags]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

ABLATION=""
SHARDS=40
WORKERS_PER_SHARD=8
THREADS_PER_WORKER=2
CORES=16
MEM_PER_CPU="2G"
TIME_LIMIT="04:00:00"
GPU_SPEC=""
RUNS_ROOT="$SCRATCH/pal_ablation/runs"
LOG_DIR="$SCRATCH/pal_ablation/logs"
JOBS_FILE="$SCRATCH/pal_ablation/jobs.jsonl"
DRY_RUN=0
FORCE=0
RETRY_FAILED=0
BENCH_FILTER=""
MAX_PAR_BENCHES_OVERRIDE="__unset__"   # sentinel: if unset, fall back to YAML
MAX_PAR_TIME="00:30:00"

usage() {
    cat <<EOF
usage: launch_ablation.sh --ablation <yaml> [options]

  --ablation PATH                ablation YAML (required)
  --shards N                     upper bound on PACKED shards (default 40; capped to pending rows)
  --workers-per-shard W          PACKED ProcessPool width (default 8)
  --threads-per-worker T         PACKED OMP/MKL/OPENBLAS threads per subprocess (default 2)
  --cores C                      PACKED sbatch -c value; must equal W*T (default 16)
  --mem-per-cpu M                sbatch --mem-per-cpu (default 2G)
  --time H                       PACKED sbatch --time (default 04:00:00; fits a short partition)
  --gpu-spec S                   extra sbatch arg for GPU (e.g. "--gpus=<GPU_TYPE>:1"); default empty
  --runs-root DIR                pal run output root (default ${RUNS_ROOT})
  --log-dir DIR                  shard logs + per-row stdout/err (default ${LOG_DIR})
  --jobs-file PATH               planned JSONL (default ${JOBS_FILE}; tiered mode appends .packed/.maxpar)
  --benchmarks LIST              comma-separated subset of YAML's benchmarks (default: all)
  --max-parallel-benchmarks LIST comma-separated benches to run 1-row-per-task in a separate MAXPAR slab
                                 (default: YAML's top-level 'max_parallel_benchmarks:' field, if any)
  --no-max-parallel-benchmarks   force-disable MAXPAR slab even if the YAML declares one
  --max-parallel-time H          MAXPAR sbatch --time (default ${MAX_PAR_TIME})
  --dry-run                      print sbatch commands; do not submit
  --force                        re-plan all rows; workers never skip
  --retry-failed                 re-run rows whose prior attempt failed
  -h | --help                    this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --ablation)                 ABLATION="$2"; shift 2;;
        --shards)                   SHARDS="$2"; shift 2;;
        --workers-per-shard)        WORKERS_PER_SHARD="$2"; shift 2;;
        --threads-per-worker)       THREADS_PER_WORKER="$2"; shift 2;;
        --cores)                    CORES="$2"; shift 2;;
        --mem-per-cpu)              MEM_PER_CPU="$2"; shift 2;;
        --time)                     TIME_LIMIT="$2"; shift 2;;
        --gpu-spec)                 GPU_SPEC="$2"; shift 2;;
        --runs-root)                RUNS_ROOT="$2"; shift 2;;
        --log-dir)                  LOG_DIR="$2"; shift 2;;
        --jobs-file)                JOBS_FILE="$2"; shift 2;;
        --benchmarks)               BENCH_FILTER="$2"; shift 2;;
        --max-parallel-benchmarks)  MAX_PAR_BENCHES_OVERRIDE="$2"; shift 2;;
        --no-max-parallel-benchmarks) MAX_PAR_BENCHES_OVERRIDE=""; shift;;
        --max-parallel-time)        MAX_PAR_TIME="$2"; shift 2;;
        --dry-run)                  DRY_RUN=1; shift;;
        --force)                    FORCE=1; shift;;
        --retry-failed)             RETRY_FAILED=1; shift;;
        -h|--help)                  usage; exit 0;;
        *) echo "unknown flag: $1" >&2; usage >&2; exit 2;;
    esac
done

if [ -z "$ABLATION" ]; then
    echo "--ablation is required" >&2; usage >&2; exit 2
fi
if [ ! -f "$ABLATION" ]; then
    echo "ablation YAML not found: $ABLATION" >&2; exit 2
fi

if [ "$(( WORKERS_PER_SHARD * THREADS_PER_WORKER ))" -ne "$CORES" ]; then
    echo "--cores ($CORES) must equal --workers-per-shard ($WORKERS_PER_SHARD) * --threads-per-worker ($THREADS_PER_WORKER)" >&2
    exit 2
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi

mkdir -p "$RUNS_ROOT" "$LOG_DIR" "$(dirname "$JOBS_FILE")"

# An explicit --max-parallel-benchmarks wins over the YAML's max_parallel_benchmarks field.
if [ "$MAX_PAR_BENCHES_OVERRIDE" = "__unset__" ]; then
    MAX_PAR_BENCHES=$(python -c "
import sys, yaml
ab = yaml.safe_load(open('$ABLATION'))
print(','.join(ab.get('max_parallel_benchmarks') or []))
")
else
    MAX_PAR_BENCHES="$MAX_PAR_BENCHES_OVERRIDE"
fi

SBATCH_TEMPLATE="$SCRIPT_DIR/../slurm/run_ablation_shard.sbatch"

# plan_and_submit <benches_csv> <jobs_file> <shards> <workers> <threads> <cores> <time> <label>
plan_and_submit() {
    local benches="$1" jobs_file="$2" shards="$3"
    local workers="$4" threads="$5" cores="$6" time_limit="$7" label="$8"

    local plan_args=(--ablation "$ABLATION" --runs-root "$RUNS_ROOT" --plan-out "$jobs_file")
    [ "$FORCE"        -eq 1 ] && plan_args+=(--force)
    [ "$RETRY_FAILED" -eq 1 ] && plan_args+=(--retry-failed)
    [ -n "$benches" ]         && plan_args+=(--benchmarks "$benches")

    echo "[launch:$label] planning: python scripts/run_ablation.py ${plan_args[*]}"
    python scripts/run_ablation.py "${plan_args[@]}"

    local n_pending; n_pending=$(wc -l < "$jobs_file" | tr -d ' ')
    if [ "$n_pending" -eq 0 ]; then
        echo "[launch:$label] nothing to do (0 pending rows)"; return 0
    fi

    local effective_shards=$shards
    if [ "$effective_shards" -gt "$n_pending" ]; then
        effective_shards=$n_pending
    fi
    echo "[launch:$label] pending=$n_pending shards=$effective_shards (requested=$shards)"
    echo "[launch:$label] workers=$workers threads=$threads cores=$cores mem/cpu=$MEM_PER_CPU time=$time_limit"
    [ -n "$GPU_SPEC" ] && echo "[launch:$label] gpu-spec: $GPU_SPEC"

    local submitted=()
    local i jid export_vars
    for (( i=0; i<effective_shards; i++ )); do
        export_vars="ALL,SHARD_I=${i},SHARD_N=${effective_shards},WORKERS=${workers},THREADS=${threads},JOBS_FILE=${jobs_file},RUNS_ROOT=${RUNS_ROOT},LOG_DIR=${LOG_DIR}"
        local sbatch_args=(
            -c "$cores"
            --mem-per-cpu="$MEM_PER_CPU"
            --time="$time_limit"
            --job-name="pal_abl_${label}${i}"
            -o "${LOG_DIR}/pal_abl_${label}${i}_%j.out"
            -e "${LOG_DIR}/pal_abl_${label}${i}_%j.err"
            --export="$export_vars"
        )
        if [ -n "$GPU_SPEC" ]; then
            # shellcheck disable=SC2206
            local gpu_tokens=($GPU_SPEC)
            sbatch_args+=("${gpu_tokens[@]}")
        fi
        sbatch_args+=("$SBATCH_TEMPLATE")

        if [ "$DRY_RUN" -eq 1 ]; then
            printf '[dry-run:%s] sbatch' "$label"; printf ' %q' "${sbatch_args[@]}"; printf '\n'
        else
            jid=$(sbatch --parsable "${sbatch_args[@]}")
            submitted+=("$jid")
        fi
    done
    if [ "$DRY_RUN" -ne 1 ] && [ ${#submitted[@]} -gt 0 ]; then
        echo "[launch:$label] submitted ${#submitted[@]} jids: ${submitted[0]}..${submitted[-1]}"
    fi
}

if [ -n "$MAX_PAR_BENCHES" ]; then
    if [ -n "$BENCH_FILTER" ]; then
        FULL_LIST="$BENCH_FILTER"
    else
        FULL_LIST=$(python -c "import yaml,sys; print(','.join(yaml.safe_load(open('$ABLATION'))['benchmarks']))")
    fi
    PACKED_LIST=$(python -c "
maxp = set('${MAX_PAR_BENCHES}'.split(','))
print(','.join(b for b in '${FULL_LIST}'.split(',') if b not in maxp))
")
    JOBS_PACKED="${JOBS_FILE%.jsonl}.packed.jsonl"
    JOBS_MAXPAR="${JOBS_FILE%.jsonl}.maxpar.jsonl"

    if [ -n "$PACKED_LIST" ]; then
        plan_and_submit "$PACKED_LIST" "$JOBS_PACKED" "$SHARDS" \
            "$WORKERS_PER_SHARD" "$THREADS_PER_WORKER" "$CORES" "$TIME_LIMIT" "packed"
    else
        echo "[launch] all benches are in --max-parallel-benchmarks; skipping packed slab"
    fi

    # MAXPAR: 1 core, 1 worker, 1 thread, one shard per pending row.
    plan_and_submit "$MAX_PAR_BENCHES" "$JOBS_MAXPAR" "100000" \
        "1" "1" "1" "$MAX_PAR_TIME" "maxpar"
else
    plan_and_submit "$BENCH_FILTER" "$JOBS_FILE" "$SHARDS" \
        "$WORKERS_PER_SHARD" "$THREADS_PER_WORKER" "$CORES" "$TIME_LIMIT" "shard"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[launch] dry-run: nothing submitted."
    exit 0
fi

echo "[launch] tail logs: tail -F ${LOG_DIR}/pal_abl_*.out"
echo "[launch] status per row: cat ${LOG_DIR}/shard*.log | jq ."
