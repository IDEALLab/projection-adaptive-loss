"""Canonical sha256 fingerprint for `Query` (zeta + conditions).

- Cast tensors to `<f8` (little-endian float64) before hashing, removes any
  fp32-vs-fp64 drift.
- Header per tensor: `"|dtype_str|shape_str|"` then raw bytes.
- Empty `conditions` (shape `[N, 0]`) is treated as `None` and feeds
  `b"|NONE|"`, so unconditional benchmarks hash deterministically.
"""

from __future__ import annotations

import hashlib

import torch

from pal.benchmarks.base import Query


def _feed(h: hashlib._Hash, t: torch.Tensor) -> None:
    arr = t.detach().cpu().numpy().astype("<f8")
    h.update(str(arr.dtype.str).encode())
    h.update(b"|")
    h.update(str(arr.shape).encode())
    h.update(b"|")
    h.update(arr.tobytes())


def query_sha256(query: Query) -> str:
    """Return `"sha256:<hex>"` for `(zeta, conditions)`.

    Unconditional queries (`conditions.shape[1] == 0`) hash as if
    `conditions is None`, so fingerprints of `Query(zeta, empty(N,0))` are
    deterministic across conditional/unconditional call sites.
    """
    h = hashlib.sha256()
    _feed(h, query.zeta)
    if query.conditions is None or query.conditions.shape[-1] == 0:
        h.update(b"|NONE|")
    else:
        _feed(h, query.conditions)
    return "sha256:" + h.hexdigest()
