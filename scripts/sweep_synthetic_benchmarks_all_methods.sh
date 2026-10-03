#!/usr/bin/env bash
# Synthetic-benchmark sweep: three sbatch arrays (fast CPU, slow fsnet on CPU, slow snarenet on GPU).
# Toggle slabs with INCLUDE_FAST / INCLUDE_SLOW_FSNET_CPU / INCLUDE_SLOW_SNARENET_GPU (default 1).
# Other env vars: EPOCHS, SEEDS, RUNS_ROOT.

set -euo pipefail

INCLUDE_FAST="${INCLUDE_FAST:-1}"
INCLUDE_SLOW_FSNET_CPU="${INCLUDE_SLOW_FSNET_CPU:-1}"
INCLUDE_SLOW_SNARENET_GPU="${INCLUDE_SLOW_SNARENET_GPU:-1}"

EPOCHS="${EPOCHS:-2000}"
SEEDS="${SEEDS:-0,1,2,3,4}"
# SnareNet soft-epoch warmup: ~25% of training, as in the SnareNet reference setup.
SNARENET_SOFT_EPOCHS="${SNARENET_SOFT_EPOCHS:-500}"
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/paper_cost_table}"
LOG_DIR="$SCRATCH/logs"

mkdir -p "$RUNS_ROOT" "$LOG_DIR"

submitted=()

if [[ "$INCLUDE_FAST" == "1" ]]; then
    # SEEDS is baked into the heredoc because sbatch --export splits comma-containing values.
    jobid=$(sbatch --parsable --array=0-53 --account=<ACCOUNT> \
        -c 16 --mem-per-cpu=2G --time=01:30:00 \
        --job-name=pal_paper_fast \
        --export=ALL,EPOCHS="$EPOCHS",RUNS_ROOT="$RUNS_ROOT",SNARENET_SOFT_EPOCHS="$SNARENET_SOFT_EPOCHS" \
        -o "$LOG_DIR/paper_fast_%A_%a.out" \
        -e "$LOG_DIR/paper_fast_%A_%a.err" <<EOF
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
# Load the site modules providing gcc and python_cuda here.
source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"

SEEDS="$SEEDS"
IFS=',' read -r -a SEED_ARR <<< "\$SEEDS"

FAST_METHODS=(alm_bolton enforce_orig dc3 pal_loggap)
ALL_BENCHES=(s1_sphere_track s2_active_set_switch s3_illcond_tube s4_qv_coupling s5_overdetermined s6_redundant_ineq)
PERSEED_METHODS=(fsnet snarenet)
LOWK_BENCHES=(s1_sphere_track s2_active_set_switch s3_illcond_tube)

declare -a TM TB TS  # method, bench, seed (TS empty for bundled tasks)
for m in "\${FAST_METHODS[@]}"; do
    for b in "\${ALL_BENCHES[@]}"; do
        TM+=("\$m"); TB+=("\$b"); TS+=("")
    done
done
for m in "\${PERSEED_METHODS[@]}"; do
    for b in "\${LOWK_BENCHES[@]}"; do
        for s in "\${SEED_ARR[@]}"; do
            TM+=("\$m"); TB+=("\$b"); TS+=("\$s")
        done
    done
done

i=\${SLURM_ARRAY_TASK_ID}
m=\${TM[\$i]}; b=\${TB[\$i]}; s=\${TS[\$i]}
if [[ -z "\$s" ]]; then
    seed_arg="\$SEEDS"
else
    seed_arg="\$s"
fi
echo "[fast \$i] method=\$m bench=\$b seeds=\$seed_arg node=\$(hostname) start=\$(date -Iseconds)"
cd "${PAL_REPO:-$HOME/pal}"
extra_args=()
if [[ "\$m" == "snarenet" && "\${SNARENET_SOFT_EPOCHS:-0}" -gt 0 ]]; then
    extra_args+=(--set "soft_epochs=\${SNARENET_SOFT_EPOCHS}")
fi
pal run --method "\$m" --benchmarks "\$b" --seeds "\$seed_arg" --epochs "\$EPOCHS" --device cpu --runs-root "\$RUNS_ROOT" "\${extra_args[@]}"
echo "[fast \$i] done=\$(date -Iseconds)"
EOF
    )
    echo "[launch] FAST slab          -> array $jobid (54 tasks, CPU 1h30min)"
    submitted+=("$jobid")
fi

if [[ "$INCLUDE_SLOW_FSNET_CPU" == "1" ]]; then
    jobid=$(sbatch --parsable --array=0-14 --account=<ACCOUNT> \
        -c 4 --mem-per-cpu=2G --time=03:00:00 \
        --job-name=pal_paper_fsnet_slow \
        --export=ALL,EPOCHS="$EPOCHS",RUNS_ROOT="$RUNS_ROOT" \
        -o "$LOG_DIR/paper_fsnet_slow_%A_%a.out" \
        -e "$LOG_DIR/paper_fsnet_slow_%A_%a.err" <<'EOF'
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
# Load the site modules providing gcc and python_cuda here.
source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"

HIGHK_BENCHES=(s4_qv_coupling s5_overdetermined s6_redundant_ineq)
i=${SLURM_ARRAY_TASK_ID}
b=${HIGHK_BENCHES[$((i / 5))]}
s=$((i % 5))
echo "[fsnet_slow $i] method=fsnet bench=$b seed=$s node=$(hostname) start=$(date -Iseconds)"
cd "${PAL_REPO:-$HOME/pal}"
pal run --method fsnet --benchmarks "$b" --seeds "$s" --epochs "$EPOCHS" --device cpu --runs-root "$RUNS_ROOT"
echo "[fsnet_slow $i] done=$(date -Iseconds)"
EOF
    )
    echo "[launch] SLOW_FSNET_CPU slab -> array $jobid (15 tasks, CPU 3h, 1 seed/task)"
    submitted+=("$jobid")
fi

if [[ "$INCLUDE_SLOW_SNARENET_GPU" == "1" ]]; then
    jobid=$(sbatch --parsable --array=0-14 --account=<ACCOUNT> \
        --gpus="<GPU_TYPE>":1 -c 8 --mem-per-cpu=4G --time=01:00:00 \
        --job-name=pal_paper_snarenet_gpu \
        --export=ALL,EPOCHS="$EPOCHS",RUNS_ROOT="$RUNS_ROOT",SNARENET_SOFT_EPOCHS="$SNARENET_SOFT_EPOCHS" \
        -o "$LOG_DIR/paper_snarenet_gpu_%A_%a.out" \
        -e "$LOG_DIR/paper_snarenet_gpu_%A_%a.err" <<'EOF'
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
# Load the site modules providing gcc and python_cuda here.
source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"

HIGHK_BENCHES=(s4_qv_coupling s5_overdetermined s6_redundant_ineq)
i=${SLURM_ARRAY_TASK_ID}
b=${HIGHK_BENCHES[$((i / 5))]}
s=$((i % 5))
echo "[snarenet_gpu $i] method=snarenet bench=$b seed=$s device=cuda node=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "${PAL_REPO:-$HOME/pal}"
extra_args=()
if [[ "${SNARENET_SOFT_EPOCHS:-0}" -gt 0 ]]; then
    extra_args+=(--set "soft_epochs=${SNARENET_SOFT_EPOCHS}")
fi
pal run --method snarenet --benchmarks "$b" --seeds "$s" --epochs "$EPOCHS" --device cuda --runs-root "$RUNS_ROOT" "${extra_args[@]}"
echo "[snarenet_gpu $i] done=$(date -Iseconds)"
EOF
    )
    echo "[launch] SLOW_SNARENET_GPU slab -> array $jobid (15 tasks, GPU 30min, 1 seed/task)"
    submitted+=("$jobid")
fi

if [[ ${#submitted[@]} -eq 0 ]]; then
    echo "[launch] no slabs enabled (all INCLUDE_* set to 0)" >&2
    exit 1
fi
echo "[launch] submitted ${#submitted[@]} arrays: ${submitted[*]}"
echo "[launch] watch with: squeue -u \$USER -j $(IFS=,; echo "${submitted[*]}")"
