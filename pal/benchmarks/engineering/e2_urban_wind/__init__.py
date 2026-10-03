"""E2 Urban Wind, sys.path wiring for the vendored WinDiNet slice.

Registration happens in `pal.benchmarks.registry._bootstrap`, not here, so
that the pal registry stays the sole source of truth for benchmark ids.
"""

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).parent
_VENDOR = _PKG_DIR / "_vendor"

for _p in (_VENDOR, _PKG_DIR):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from .benchmark import E2UrbanWind  # noqa: E402

__all__ = ["E2UrbanWind"]
