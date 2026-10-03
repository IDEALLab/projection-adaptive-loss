#!/bin/bash -l
#SBATCH --job-name=pal-gpu-ipopt-build
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --mem=440G
#SBATCH --time=03:00:00
#SBATCH --output=<SCRATCH>/logs/pal-gpu-ipopt-build-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-gpu-ipopt-build-%j.err

set -euo pipefail

if [[ -n "${PAL_DIR:-}" ]]; then
    PAL_REPO="$(cd "${PAL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PAL_REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PAL_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

SCRIPT_DIR="${PAL_REPO}/scripts"

export IMAGE_NAME="${IMAGE_NAME:-gpu-pal-ipopt-2511}"
export DOCKERFILE="${DOCKERFILE:-${PAL_REPO}/docker/training_gpu_ipopt.Dockerfile}"
export PAL_DIR="${PAL_REPO}"

exec /bin/bash "${SCRIPT_DIR}/build_gpu_container.slurm"
