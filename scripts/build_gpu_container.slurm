#!/bin/bash -l
#SBATCH --job-name=pal-gpu-build
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --mem=440G
#SBATCH --time=03:00:00
#SBATCH --output=<SCRATCH>/logs/pal-gpu-build-%j.out
#SBATCH --error=<SCRATCH>/logs/pal-gpu-build-%j.err

set -euo pipefail

if [[ -n "${PAL_DIR:-}" ]]; then
    PAL_REPO="$(cd "${PAL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PAL_REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PAL_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

ROOT_DIR="${ROOT_DIR:-$(cd "${PAL_REPO}/.." && pwd)}"

PAL_DIR="${PAL_DIR:-${PAL_REPO}}"

IMAGE_NAME="${IMAGE_NAME:-gpu-pal-training-2511}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="${DOCKERFILE:-${PAL_DIR}/docker/training_gpu.Dockerfile}"
CONTAINER_DIR="${CONTAINER_DIR:-${SCRATCH}/containers}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/logs}"
TMP_ROOT="${TMP_ROOT:-${SCRATCH}/tmp}"
PODMAN_TMPDIR="${PODMAN_TMPDIR:-/tmp/${USER}/pal-podman-${SLURM_JOB_ID:-$$}}"
PODMAN_STATE_ROOT="${PODMAN_STATE_ROOT:-/tmp/${USER}/pal-podman-state-${SLURM_JOB_ID:-$$}}"
ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-${TMP_ROOT}/enroot}"
ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-/tmp/${USER}/enroot-runtime-${SLURM_JOB_ID:-$$}}"

mkdir -p "${CONTAINER_DIR}" "${LOG_DIR}" "${TMP_ROOT}" "${PODMAN_TMPDIR}" \
    "${PODMAN_STATE_ROOT}/storage" "${PODMAN_STATE_ROOT}/runroot" \
    "${ENROOT_TEMP_PATH}" "${ENROOT_RUNTIME_PATH}"
export ENROOT_TEMP_PATH
export ENROOT_RUNTIME_PATH

# NFS home lacks lsetxattr (needed by overlay-fs), so point Podman storage at node-local tmpfs.
CONTAINERS_STORAGE_CONF="${PODMAN_STATE_ROOT}/storage.conf"
cat > "${CONTAINERS_STORAGE_CONF}" <<EOF_STORAGE_CONF
[storage]
driver = "overlay"
graphroot = "${PODMAN_STATE_ROOT}/storage"
runroot = "${PODMAN_STATE_ROOT}/runroot"
EOF_STORAGE_CONF
export CONTAINERS_STORAGE_CONF

cleanup() {
    # Overlay files are owned by namespaced UIDs, so plain rm hits EPERM without podman unshare.
    rm -rf "${PODMAN_TMPDIR}" "${ENROOT_RUNTIME_PATH}" 2>/dev/null || true
    podman unshare rm -rf "${PODMAN_STATE_ROOT}" 2>/dev/null || true
}
trap cleanup EXIT

if [[ ! -d "${PAL_DIR}" ]]; then
    echo "ERROR: PAL repo not found at ${PAL_DIR}"
    exit 1
fi

cd "${ROOT_DIR}"

echo "=== PAL GH200 Container Build ==="
echo "SLURM submit dir: ${SLURM_SUBMIT_DIR:-<unset>}"
echo "Root dir:       ${ROOT_DIR}"
echo "PAL repo:       ${PAL_DIR}"
echo "Image:          ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Dockerfile:     ${DOCKERFILE}"
echo "Build context:  ${ROOT_DIR}"
echo "Container dir:  ${CONTAINER_DIR}"
echo "Podman tmpdir:  ${PODMAN_TMPDIR}"
echo "Podman state:   ${PODMAN_STATE_ROOT}"
echo "Storage conf:   ${CONTAINERS_STORAGE_CONF}"
echo "Enroot temp:    ${ENROOT_TEMP_PATH}"
echo "Enroot runtime: ${ENROOT_RUNTIME_PATH}"
echo "================================="
echo "=== Preflight: podman info ==="
podman info
echo "=== Preflight: filesystem capacity ==="
df -h /tmp /dev/shm "${SCRATCH}"
echo "======================================"

PAL_SRC="$(basename "${PAL_DIR}")"

TMPDIR="${PODMAN_TMPDIR}" podman build \
  --platform linux/arm64 \
  --build-arg "PAL_SRC=${PAL_SRC}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  .

rm -f "${CONTAINER_DIR}/${IMAGE_NAME}.tar" "${CONTAINER_DIR}/${IMAGE_NAME}.sqsh"

TMPDIR="${PODMAN_TMPDIR}" podman save \
  -o "${CONTAINER_DIR}/${IMAGE_NAME}.tar" \
  "localhost/${IMAGE_NAME}:${IMAGE_TAG}"

enroot import -x mount \
  -o "${CONTAINER_DIR}/${IMAGE_NAME}.sqsh" \
  "podman://localhost/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Built image:     localhost/${IMAGE_NAME}:${IMAGE_TAG}"
echo "Saved tarball:   ${CONTAINER_DIR}/${IMAGE_NAME}.tar"
echo "Saved squashfs:  ${CONTAINER_DIR}/${IMAGE_NAME}.sqsh"
