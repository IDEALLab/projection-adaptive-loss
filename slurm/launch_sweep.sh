#!/bin/bash
# Runs `pal sweep plan`, then submits the CPU array, per-VRAM-tier GPU arrays and the aggregator.
# Usage: slurm/launch_sweep.sh [--methods M,...] [--seeds S,...] [--skip-benches A,B] [--cpu-pool N] [--gpu-pool N] [--name NAME] [--wandb] [--retry-failed]
# W&B entity/project come from $WANDB_ENTITY / $WANDB_PROJECT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PAL_VENV:?PAL_VENV must be set; example: PAL_VENV=\$VENV/bin/activate}"
fi

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-pal}"

PLAN_OUTPUT=$(pal sweep plan \
    --cpu-sbatch "$SCRIPT_DIR/run_row_cpu.sbatch" \
    --gpu-sbatch "$SCRIPT_DIR/run_row_gpu.sbatch" \
    --aggregate-sbatch "$SCRIPT_DIR/aggregate.sbatch" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    "$@")

echo "$PLAN_OUTPUT"

SWEEP_DIR=$(echo "$PLAN_OUTPUT" | awk '/^\[plan\] sweep dir:/ {print $4}')
if [[ -z "$SWEEP_DIR" ]]; then
    echo "[launch] ERROR: could not parse sweep dir from plan output." >&2
    exit 1
fi

CPU_POOL=$(python3 -c "import json; print(json.load(open('$SWEEP_DIR/manifest.json'))['cpu_pool'])")
GPU_POOL=$(python3 -c "import json; print(json.load(open('$SWEEP_DIR/manifest.json'))['gpu_pool'])")
CPU_CHEAP_N=$(wc -l < "$SWEEP_DIR/jobs_cpu_cheap.jsonl")
CPU_STD_N=$(wc -l < "$SWEEP_DIR/jobs_cpu_std.jsonl")
GPU_S_N=$(wc -l < "$SWEEP_DIR/jobs_gpu_s.jsonl")
GPU_M_N=$(wc -l < "$SWEEP_DIR/jobs_gpu_m.jsonl")
GPU_L_N=$(wc -l < "$SWEEP_DIR/jobs_gpu_l.jsonl")

DEPS_ARG=""

CPU_CHEAP_JID=
if (( CPU_CHEAP_N > 0 )); then
    CPU_CHEAP_JID=$(sbatch --parsable \
        --array="0-$((CPU_CHEAP_N - 1))%${CPU_POOL}" \
        --partition="<PARTITION>" --time=04:00:00 \
        --export=ALL,WANDB_ENTITY="$WANDB_ENTITY",WANDB_PROJECT="$WANDB_PROJECT" \
        "$SCRIPT_DIR/run_row_cpu.sbatch" "$SWEEP_DIR/jobs_cpu_cheap.jsonl")
    echo "[launch] CPU_CHEAP_JID=$CPU_CHEAP_JID  ($CPU_CHEAP_N rows, short partition, pool=$CPU_POOL)"
    DEPS_ARG="${DEPS_ARG}${DEPS_ARG:+:}$CPU_CHEAP_JID"
fi

CPU_STD_JID=
if (( CPU_STD_N > 0 )); then
    CPU_STD_JID=$(sbatch --parsable \
        --array="0-$((CPU_STD_N - 1))%${CPU_POOL}" \
        --export=ALL,WANDB_ENTITY="$WANDB_ENTITY",WANDB_PROJECT="$WANDB_PROJECT" \
        "$SCRIPT_DIR/run_row_cpu.sbatch" "$SWEEP_DIR/jobs_cpu_std.jsonl")
    echo "[launch] CPU_STD_JID=$CPU_STD_JID  ($CPU_STD_N rows, default partition, pool=$CPU_POOL)"
    DEPS_ARG="${DEPS_ARG}${DEPS_ARG:+:}$CPU_STD_JID"
fi

GPU_S_JID=
if (( GPU_S_N > 0 )); then
    GPU_S_JID=$(sbatch --parsable \
        --array="0-$((GPU_S_N - 1))%${GPU_POOL}" \
        --gpus=1 \
        --export=ALL,WANDB_ENTITY="$WANDB_ENTITY",WANDB_PROJECT="$WANDB_PROJECT" \
        "$SCRIPT_DIR/run_row_gpu.sbatch" "$SWEEP_DIR/jobs_gpu_s.jsonl")
    echo "[launch] GPU_S_JID=$GPU_S_JID  ($GPU_S_N rows, tier=S, --gpus=1, pool=$GPU_POOL)"
    DEPS_ARG="${DEPS_ARG}${DEPS_ARG:+:}$GPU_S_JID"
fi

GPU_M_JID=
if (( GPU_M_N > 0 )); then
    GPU_M_JID=$(sbatch --parsable \
        --array="0-$((GPU_M_N - 1))%${GPU_POOL}" \
        --gpus=1 --gres="<GPU_MEM>" \
        --export=ALL,WANDB_ENTITY="$WANDB_ENTITY",WANDB_PROJECT="$WANDB_PROJECT" \
        "$SCRIPT_DIR/run_row_gpu.sbatch" "$SWEEP_DIR/jobs_gpu_m.jsonl")
    echo "[launch] GPU_M_JID=$GPU_M_JID  ($GPU_M_N rows, tier=M, --gpus=1 --gres=<GPU_MEM>, pool=$GPU_POOL)"
    DEPS_ARG="${DEPS_ARG}${DEPS_ARG:+:}$GPU_M_JID"
fi

if (( GPU_L_N > 0 )); then
    echo "[launch] SKIPPING $GPU_L_N L-tier rows (>=40 GB VRAM)."
    echo "[launch]   File: $SWEEP_DIR/jobs_gpu_l.jsonl"
    echo "[launch]   Run these on a larger cluster (GH200 / A100-80G)."
fi

if [[ -n "$DEPS_ARG" ]]; then
    AGG_JID=$(sbatch --parsable \
        --dependency="afterany:$DEPS_ARG" \
        "$SCRIPT_DIR/aggregate.sbatch" "$SWEEP_DIR")
    echo "[launch] AGG_JID=$AGG_JID  (dependency=afterany:$DEPS_ARG)"
else
    echo "[launch] no rows submitted, skipping aggregator"
fi

echo
echo "[launch] sweep dir: $SWEEP_DIR"
echo "[launch] Monitor: squeue -u \$USER    |    tail -f $SWEEP_DIR/runs/*/jsonl"
echo "[launch] Results will land at: $SWEEP_DIR/results.md"
