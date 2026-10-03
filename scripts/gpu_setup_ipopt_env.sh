#!/usr/bin/env bash
# One-time setup of the conda-forge cyipopt env on scratch (no aarch64 PyPI wheel exists).
# The env is bind-mounted into the container and side-loaded via PYTHONPATH/LD_LIBRARY_PATH.
# Idempotent. Usage: bash scripts/gpu_setup_ipopt_env.sh

set -euo pipefail

ENV_PREFIX="${SCRATCH}/pal-data/ipopt-env"
MICROMAMBA_BIN="${HOME}/.local/bin/micromamba"
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ipopt_environment.yml"

if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
    echo "[setup] installing micromamba to ${HOME}/.local/bin"
    mkdir -p "${HOME}/.local/bin"
    (
        cd "${HOME}/.local"
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest \
            | tar -xj bin/micromamba
    )
fi
echo "[setup] micromamba: $("${MICROMAMBA_BIN}" --version)"

if [[ -d "${ENV_PREFIX}" && -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "[setup] env exists at ${ENV_PREFIX}; skipping create"
else
    echo "[setup] creating env at ${ENV_PREFIX} from ${ENV_FILE}"
    mkdir -p "$(dirname "${ENV_PREFIX}")"
    "${MICROMAMBA_BIN}" create -y -p "${ENV_PREFIX}" -f "${ENV_FILE}"
fi

# Stage the probe inside the bind-mounted env so the container can run it too.
PROBE_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/probe_cyipopt.py"
PROBE_STAGED="${ENV_PREFIX}/probe_cyipopt.py"
cp "${PROBE_SRC}" "${PROBE_STAGED}"

echo "[setup] running cyipopt smoke test (login node)"
"${ENV_PREFIX}/bin/python" "${PROBE_STAGED}"

cat <<EOF

[setup] login-node smoke test passed.

Recommended: container-side smoke test (catches ABI mismatches the login
node won't). Run once before your first sbatch:

  srun --pty -A <ACCOUNT> --environment=gpu-pal-training /bin/bash -lc '
    export PYTHONPATH=/workspace/pal-data/ipopt-env/lib/python3.12/site-packages:\${PYTHONPATH:-}
    export LD_LIBRARY_PATH=/workspace/pal-data/ipopt-env/lib:\${LD_LIBRARY_PATH:-}
    python /workspace/pal-data/ipopt-env/probe_cyipopt.py
  '

If that prints x*~3.0, you're done. If it errors on missing GLIBCXX_*
symbols, uncomment the LD_PRELOAD line in run_ipopt_gpu.slurm to
use the conda env's libstdc++.
EOF
