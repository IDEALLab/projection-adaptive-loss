"""IPOPT classical baseline (cyipopt).

Requires `cyipopt` to be importable. Supported paths are:
- a CPU env with `conda install -c conda-forge cyipopt`
- the dedicated GH200 IPOPT container path documented in README
"""

import os
import sys

# Homebrew-built cyipopt on macOS links its own libomp via Homebrew gcc;
# torch ships a separate libomp. Loading both raises `OMP: Error #15` and
# aborts the process unless KMP_DUPLICATE_LIB_OK is set before the second
# one loads. Conda-forge cyipopt doesn't have this problem, and neither
# does Linux. Keep the workaround darwin-scoped and idempotent.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__all__ = ["IPOPTConfig", "IPOPTSolver"]


def __getattr__(name):
    # Lazy so that the cyipopt-free submodules (trace, shard_merge) import
    # without cyipopt installed.
    if name in __all__:
        from pal.baselines.ipopt import solver

        return getattr(solver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
