#!/bin/bash
# Chain-submit the 2000-epoch equality-constraint reruns for e1/e2, NUM_HOPS hops per bench.
# Hops chain with afterany so a walltime timeout still resumes from the checkpoint (--auto-resume).
# Usage: scripts/submit_eq_rerun.sh [e1|e2|all]   (env: NUM_HOPS=3, DRY_RUN=1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

NUM_HOPS="${NUM_HOPS:-3}"
DRY_RUN="${DRY_RUN:-0}"
WHICH="${1:-all}"

E1_SBATCH="${REPO_ROOT}/slurm/engineering/e1_eq_gpu.sbatch"
E2_SBATCH="${REPO_ROOT}/slurm/engineering/e2_eq.sbatch"

case "$WHICH" in
    e1) BENCHES=(e1) ;;
    e2) BENCHES=(e2) ;;
    all)  BENCHES=(e1 e2) ;;
    *) echo "usage: $(basename "$0") [e1|e2|all]" >&2; exit 2 ;;
esac

if ! [[ "$NUM_HOPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_HOPS must be a positive integer (got '$NUM_HOPS')" >&2
    exit 2
fi

# submit_one <sbatch_path> [dependency_jobid]
submit_one() {
    local sbatch_path="$1" dep="${2:-}"
    local args=(--parsable)
    [[ -n "$dep" ]] && args+=(--dependency="afterany:${dep}")
    args+=("$sbatch_path")

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] sbatch ${args[*]}" >&2
        echo "DRYRUN"
        return 0
    fi

    local out jobid
    out="$(sbatch "${args[@]}")"
    # --parsable prints "<jobid>" or "<jobid>;<cluster>"; take the leading digits.
    jobid="${out%%;*}"
    jobid="$(printf '%s' "$jobid" | tr -cd '0-9')"
    if [[ -z "$jobid" ]]; then
        echo "ERROR: could not parse job id from sbatch output: '$out'" >&2
        exit 1
    fi
    echo "$jobid"
}

declare -a SUMMARY=()

for bench in "${BENCHES[@]}"; do
    case "$bench" in
        e1) sbatch_path="$E1_SBATCH" ;;
        e2) sbatch_path="$E2_SBATCH" ;;
    esac
    if [[ ! -f "$sbatch_path" ]]; then
        echo "ERROR: sbatch not found: $sbatch_path" >&2
        exit 1
    fi

    echo ">>> $bench: submitting $NUM_HOPS chained hop(s) of $(basename "$sbatch_path")" >&2
    prev=""
    chain=()
    for (( hop=1; hop<=NUM_HOPS; hop++ )); do
        jobid="$(submit_one "$sbatch_path" "$prev")"
        if [[ -n "$prev" ]]; then
            echo "    hop $hop: job $jobid (afterany:$prev)" >&2
        else
            echo "    hop $hop: job $jobid (head)" >&2
        fi
        chain+=("$jobid")
        prev="$jobid"
    done
    SUMMARY+=("$bench: ${chain[*]}")
done

echo
echo "=== submitted job chains ==="
for line in "${SUMMARY[@]}"; do
    echo "  $line"
done
