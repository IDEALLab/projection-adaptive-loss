#!/usr/bin/env bash

# Cluster auth/runtime helpers: source on the submit host to resolve tokens, and in the container for cache paths.

_cluster_trimmed_file_contents() {
    local candidate="${1:-}"
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
        tr -d '[:space:]' < "${candidate}"
        return 0
    fi
    return 1
}

cluster_resolve_hf_auth() {
    local resolved=""
    local candidate=""

    if [[ -n "${HF_TOKEN:-}" ]]; then
        resolved="${HF_TOKEN}"
    elif [[ -n "${HF_HUB_TOKEN:-}" ]]; then
        resolved="${HF_HUB_TOKEN}"
    elif [[ -n "${HF_TOKEN_FILE:-}" ]]; then
        resolved="$(_cluster_trimmed_file_contents "${HF_TOKEN_FILE}")" || true
    else
        for candidate in \
            "${HOME}/.config/pal/hf_token.txt" \
            "${HOME}/.cache/huggingface/token" \
            "${HOME}/.huggingface/token"
        do
            if resolved="$(_cluster_trimmed_file_contents "${candidate}")"; then
                break
            fi
        done
    fi

    if [[ -n "${resolved}" ]]; then
        export HF_TOKEN="${resolved}"
        export HF_HUB_TOKEN="${resolved}"
        return 0
    fi
    return 1
}

cluster_resolve_wandb_auth() {
    local resolved=""
    local candidate=""

    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        resolved="${WANDB_API_KEY}"
    elif [[ -n "${WANDB_TOKEN_FILE:-}" ]]; then
        resolved="$(_cluster_trimmed_file_contents "${WANDB_TOKEN_FILE}")" || true
    else
        for candidate in \
            "${HOME}/.config/pal/wandb_token.txt"
        do
            if resolved="$(_cluster_trimmed_file_contents "${candidate}")"; then
                break
            fi
        done
    fi

    if [[ -n "${resolved}" ]]; then
        export WANDB_API_KEY="${resolved}"
        return 0
    fi
    return 1
}

cluster_prepare_runtime() {
    if [[ -z "${RUNS_ROOT:-}" ]]; then
        echo "[cluster-runtime] RUNS_ROOT must be set before prepare_runtime" >&2
        return 1
    fi

    local alloc_conf="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/pal-data/cache}"
    export TORCH_HOME="${TORCH_HOME:-/workspace/pal-data/torch}"
    export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${XDG_CACHE_HOME}/torch/kernels}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
    export WANDB_DIR="${RUNS_ROOT}/wandb"
    export WANDB_CACHE_DIR="${XDG_CACHE_HOME}/wandb"
    export PYTORCH_ALLOC_CONF="${alloc_conf}"
    unset PYTORCH_CUDA_ALLOC_CONF

    mkdir -p \
        "${RUNS_ROOT}" \
        /workspace/pal-data/hf \
        "${XDG_CACHE_HOME}" \
        "${TORCH_HOME}" \
        "${PYTORCH_KERNEL_CACHE_PATH}" \
        "${TORCHINDUCTOR_CACHE_DIR}" \
        "${TRITON_CACHE_DIR}" \
        "${WANDB_DIR}" \
        "${WANDB_CACHE_DIR}"
}

cluster_should_enable_wandb() {
    if [[ "${ENABLE_WANDB:-0}" != "1" ]]; then
        return 1
    fi

    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "[wandb] ENABLE_WANDB=1 but no WANDB_API_KEY resolved; continuing with JSONL-only logging" >&2
        return 1
    fi

    return 0
}
