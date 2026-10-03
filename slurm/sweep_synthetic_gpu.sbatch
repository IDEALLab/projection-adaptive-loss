#!/bin/bash -l
#SBATCH --job-name=pal-synth-sweep
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus=4
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --output=<SCRATCH>/logs/pal-synth-sweep-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-synth-sweep-%j.err
# Synthetic-benchmark sweep on one GH200 node: 7 methods x 6 benches x 10 seeds = 420 runs.
# Phase A/B run the CPU methods, phase C runs snarenet on the 4 GPUs with soft_epochs=$SNARENET_SOFT_EPOCHS.
# Tunables (env): EPOCHS, SEEDS, SNARENET_SOFT_EPOCHS, PHASE_{A,B,C}_WORKERS, PHASE_{A,B}_THREADS, RUNS_ROOT, SKIP_PHASE_{A,B,C}.

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
RUNS_ROOT="${RUNS_ROOT:-/workspace/pal-runs/synth_sweep_gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

EPOCHS="${EPOCHS:-2000}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
SNARENET_SOFT_EPOCHS="${SNARENET_SOFT_EPOCHS:-500}"

PHASE_A_WORKERS="${PHASE_A_WORKERS:-24}"
PHASE_A_THREADS="${PHASE_A_THREADS:-12}"
PHASE_B_WORKERS="${PHASE_B_WORKERS:-8}"
PHASE_B_THREADS="${PHASE_B_THREADS:-32}"
PHASE_C_WORKERS="${PHASE_C_WORKERS:-4}"

SKIP_PHASE_A="${SKIP_PHASE_A:-0}"
SKIP_PHASE_B="${SKIP_PHASE_B:-0}"
SKIP_PHASE_C="${SKIP_PHASE_C:-0}"

cluster_resolve_hf_auth || true
cluster_resolve_wandb_auth || true

srun -u --environment="${EDF_ENVIRONMENT}" /bin/bash -lc '
    set -euo pipefail

    cd /workspace/pal
    export PYTHONPATH="/workspace/pal${PYTHONPATH:+:${PYTHONPATH}}"
    source /workspace/pal/scripts/cluster_runtime.sh
    export RUNS_ROOT="'"${RUNS_ROOT}"'"
    export WANDB_PROJECT="'"${WANDB_PROJECT}"'"
    export WANDB_ENTITY="'"${WANDB_ENTITY}"'"
    cluster_prepare_runtime

    EPOCHS="'"${EPOCHS}"'"
    SEEDS="'"${SEEDS}"'"
    SNARENET_SOFT_EPOCHS="'"${SNARENET_SOFT_EPOCHS}"'"
    PHASE_A_WORKERS="'"${PHASE_A_WORKERS}"'"
    PHASE_A_THREADS="'"${PHASE_A_THREADS}"'"
    PHASE_B_WORKERS="'"${PHASE_B_WORKERS}"'"
    PHASE_B_THREADS="'"${PHASE_B_THREADS}"'"
    PHASE_C_WORKERS="'"${PHASE_C_WORKERS}"'"
    SKIP_PHASE_A="'"${SKIP_PHASE_A}"'"
    SKIP_PHASE_B="'"${SKIP_PHASE_B}"'"
    SKIP_PHASE_C="'"${SKIP_PHASE_C}"'"

    wandb_args=()
    if cluster_should_enable_wandb; then
        wandb_args+=(--wandb --wandb-project "${WANDB_PROJECT}")
        if [[ -n "${WANDB_ENTITY:-}" ]]; then
            wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
        fi
    fi
    GROUP="synth_sweep__${SLURM_JOB_ID:-local}"

    LOG_DIR="${RUNS_ROOT}/_launch_logs"
    mkdir -p "${LOG_DIR}"

    BENCHES=(s1_sphere_track s2_active_set_switch s3_illcond_tube s4_qv_coupling s5_overdetermined s6_redundant_ineq)
    IFS="," read -r -a SEED_ARR <<< "${SEEDS}"

    run_one() {
        local method="$1" bench="$2" seed="$3" device="$4"; shift 4
        local logf="${LOG_DIR}/${method}_${bench}_s${seed}.log"
        pal run \
            --method "${method}" \
            --benchmarks "${bench}" \
            --seeds "${seed}" \
            --epochs "${EPOCHS}" \
            --device "${device}" \
            --runs-root "${RUNS_ROOT}" \
            --wandb-group "${GROUP}" \
            "$@" \
            "${wandb_args[@]}" \
            > "${logf}" 2>&1
    }
    export -f run_one
    export RUNS_ROOT EPOCHS GROUP LOG_DIR
    export wandb_args_str="${wandb_args[*]}"

    # Drain a queue of "method|bench|seed|device|extra1,extra2" lines through N parallel slots.
    drain_queue() {
        local n="$1"; shift
        local -a slots=()
        local line
        while IFS= read -r line; do
            local method bench seed device extras
            IFS="|" read -r method bench seed device extras <<< "${line}"
            local -a extra_args=()
            if [[ -n "${extras}" ]]; then
                IFS="," read -r -a extra_args <<< "${extras}"
            fi
            ( run_one "${method}" "${bench}" "${seed}" "${device}" "${extra_args[@]}" ) &
            slots+=($!)
            while (( ${#slots[@]} >= n )); do
                local new=()
                for pid in "${slots[@]}"; do
                    if kill -0 "${pid}" 2>/dev/null; then
                        new+=("${pid}")
                    else
                        wait "${pid}" || echo "[warn] pid=${pid} exited non-zero"
                    fi
                done
                slots=("${new[@]}")
                sleep 0.5
            done
        done
        local fail=0
        for pid in "${slots[@]}"; do
            if ! wait "${pid}"; then fail=$((fail+1)); fi
        done
        return ${fail}
    }

    if [[ "${SKIP_PHASE_A}" != "1" ]]; then
        echo "[phase A] cheap methods on CPU; workers=${PHASE_A_WORKERS} threads=${PHASE_A_THREADS}"
        export OMP_NUM_THREADS=${PHASE_A_THREADS}
        export MKL_NUM_THREADS=${PHASE_A_THREADS}
        export OPENBLAS_NUM_THREADS=${PHASE_A_THREADS}
        {
            for METHOD in pal_loggap alm alm_bolton dc3 enforce_orig; do
                for BENCH in "${BENCHES[@]}"; do
                    for SEED in "${SEED_ARR[@]}"; do
                        echo "${METHOD}|${BENCH}|${SEED}|cpu|"
                    done
                done
            done
        } | drain_queue "${PHASE_A_WORKERS}"
        echo "[phase A] done"
    else
        echo "[phase A] skipped"
    fi

    if [[ "${SKIP_PHASE_B}" != "1" ]]; then
        echo "[phase B] fsnet on CPU; workers=${PHASE_B_WORKERS} threads=${PHASE_B_THREADS}"
        export OMP_NUM_THREADS=${PHASE_B_THREADS}
        export MKL_NUM_THREADS=${PHASE_B_THREADS}
        export OPENBLAS_NUM_THREADS=${PHASE_B_THREADS}
        {
            for BENCH in "${BENCHES[@]}"; do
                for SEED in "${SEED_ARR[@]}"; do
                    echo "fsnet|${BENCH}|${SEED}|cpu|"
                done
            done
        } | drain_queue "${PHASE_B_WORKERS}"
        echo "[phase B] done"
    else
        echo "[phase B] skipped"
    fi

    if [[ "${SKIP_PHASE_C}" != "1" ]]; then
        echo "[phase C] snarenet on GPU; workers=${PHASE_C_WORKERS} soft_epochs=${SNARENET_SOFT_EPOCHS}"
        export OMP_NUM_THREADS=8
        export MKL_NUM_THREADS=8
        export OPENBLAS_NUM_THREADS=8
        # Each job pins one GPU via CUDA_VISIBLE_DEVICES.
        run_one_gpu_pinned() {
            local method="$1" bench="$2" seed="$3" gpu="$4"; shift 4
            local logf="${LOG_DIR}/${method}_${bench}_s${seed}_gpu${gpu}.log"
            CUDA_VISIBLE_DEVICES="${gpu}" pal run \
                --method "${method}" \
                --benchmarks "${bench}" \
                --seeds "${seed}" \
                --epochs "${EPOCHS}" \
                --device cuda \
                --runs-root "${RUNS_ROOT}" \
                --wandb-group "${GROUP}" \
                --set "soft_epochs=${SNARENET_SOFT_EPOCHS}" \
                "${wandb_args[@]}" \
                > "${logf}" 2>&1
        }
        idx=0
        pids=()
        for BENCH in "${BENCHES[@]}"; do
            for SEED in "${SEED_ARR[@]}"; do
                gpu=$(( idx % 4 ))
                ( run_one_gpu_pinned snarenet "${BENCH}" "${SEED}" "${gpu}" ) &
                pids+=($!)
                idx=$((idx+1))
                while (( ${#pids[@]} >= PHASE_C_WORKERS )); do
                    new=()
                    for pid in "${pids[@]}"; do
                        if kill -0 "${pid}" 2>/dev/null; then
                            new+=("${pid}")
                        else
                            wait "${pid}" || echo "[warn] snarenet pid=${pid} exited non-zero"
                        fi
                    done
                    pids=("${new[@]}")
                    sleep 1
                done
            done
        done
        cfail=0
        for pid in "${pids[@]}"; do
            if ! wait "${pid}"; then cfail=$((cfail+1)); fi
        done
        if (( cfail > 0 )); then
            echo "[phase C] ${cfail} job(s) failed"
        fi
        echo "[phase C] done"
    else
        echo "[phase C] skipped"
    fi

    echo "[sweep] all phases done. runs at: ${RUNS_ROOT}"
'
