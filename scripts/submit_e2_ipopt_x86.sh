#!/bin/bash
# Submit the e2 IPOPT x86 campaign: probe (VRAM/latency), main, control (max_iter 2000), besteffort.
# Non-probe modes freeze per-shard eval points via gen_e2_ipopt_points.py, then submit one array task per shard.
# Usage: scripts/submit_e2_ipopt_x86.sh <probe|main|control|besteffort>   (env: SEED, N_EVAL, PACK, DRY_RUN)
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
    echo "usage: $0 <probe|main|control|besteffort>" >&2
    exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH}/pal_engineering/e2_ipopt_x86}"
SEED="${SEED:-0}"
PACK="${PACK:-1}"
DRY_RUN="${DRY_RUN:-0}"

run() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN:'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

submit_array() {
    local arm="$1" n_shards="$2"; shift 2
    local last=$(( n_shards - 1 ))
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN: sbatch --array=0-%s' "${last}"
        printf ' %q' "$@"
        printf ' %s/slurm/x86/e2_ipopt_main.sbatch\n' "${REPO}"
        return
    fi
    local out
    out="$(sbatch --array=0-"${last}" "$@" "${REPO}/slurm/x86/e2_ipopt_main.sbatch")"
    echo "${out}"
    echo "${out}" | grep -oE '[0-9]+' | tail -1
}

gen_points() {
    local points_dir="$1" n_eval="$2"
    run python "${REPO}/scripts/gen_e2_ipopt_points.py" \
        --seed "${SEED}" --n-eval "${n_eval}" --pack "${PACK}" \
        --mode replicate --out "${points_dir}"
}

case "${MODE}" in
    probe)
        run sbatch "${REPO}/slurm/x86/e2_ipopt_probe.sbatch"
        ;;

    main)
        N_EVAL="${N_EVAL:-64}"; MULTI_START="${MULTI_START:-2}"
        POINTS_DIR="${POINTS_DIR:-${SCRATCH_ROOT}/points_main_seed${SEED}}"
        RUNS_ROOT="${RUNS_ROOT:-${SCRATCH_ROOT}/main}"
        gen_points "${POINTS_DIR}" "${N_EVAL}"
        n_shards=$(( (N_EVAL + PACK - 1) / PACK ))
        submit_array main "${n_shards}" \
            --export=ALL,ARM=main,SEED="${SEED}",MAX_ITER=500,MULTI_START="${MULTI_START}",POINTS_DIR="${POINTS_DIR}",RUNS_ROOT="${RUNS_ROOT}"
        ;;

    control)
        N_EVAL="${N_EVAL:-4}"; MULTI_START="${MULTI_START:-2}"
        POINTS_DIR="${POINTS_DIR:-${SCRATCH_ROOT}/points_control_seed${SEED}}"
        RUNS_ROOT="${RUNS_ROOT:-${SCRATCH_ROOT}/control}"
        gen_points "${POINTS_DIR}" "${N_EVAL}"
        n_shards=$(( (N_EVAL + PACK - 1) / PACK ))
        submit_array control "${n_shards}" \
            --export=ALL,ARM=control,SEED="${SEED}",MAX_ITER=2000,MULTI_START="${MULTI_START}",POINTS_DIR="${POINTS_DIR}",RUNS_ROOT="${RUNS_ROOT}"
        ;;

    besteffort)
        N_EVAL="${N_EVAL:-8}"; MULTI_START="${MULTI_START:-2}"
        POINTS_DIR="${POINTS_DIR:-${SCRATCH_ROOT}/points_besteffort_seed${SEED}}"
        RUNS_ROOT="${RUNS_ROOT:-${SCRATCH_ROOT}/besteffort}"
        # Env overrides win. Set PAL_IPOPT_INIT_MODE=warmstart for the layout warm start.
        : "${PAL_IPOPT_SCALING:=gradient-based}"
        : "${PAL_IPOPT_ACCEPTABLE:=1}"
        : "${PAL_IPOPT_F64:=1}"
        : "${PAL_IPOPT_INIT_MODE:=site}"
        gen_points "${POINTS_DIR}" "${N_EVAL}"
        n_shards=$(( (N_EVAL + PACK - 1) / PACK ))
        submit_array besteffort "${n_shards}" \
            --export=ALL,ARM=besteffort,SEED="${SEED}",MAX_ITER=500,MULTI_START="${MULTI_START}",POINTS_DIR="${POINTS_DIR}",RUNS_ROOT="${RUNS_ROOT}",PAL_IPOPT_SCALING="${PAL_IPOPT_SCALING}",PAL_IPOPT_ACCEPTABLE="${PAL_IPOPT_ACCEPTABLE}",PAL_IPOPT_F64="${PAL_IPOPT_F64}",PAL_IPOPT_INIT_MODE="${PAL_IPOPT_INIT_MODE}"
        ;;

    *)
        echo "unknown mode: ${MODE} (want probe|main|control|besteffort)" >&2
        exit 2
        ;;
esac
