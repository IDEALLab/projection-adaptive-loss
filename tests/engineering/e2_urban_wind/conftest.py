"""Put the vendored windinet and inverse packages on sys.path during collection."""

import sys
from pathlib import Path

_BENCH_PKG = Path(__file__).resolve().parents[2].parent / "pal" / "benchmarks" / "engineering" / "e2_urban_wind"
_VENDOR = _BENCH_PKG / "_vendor"

for _p in (_VENDOR, _BENCH_PKG):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
