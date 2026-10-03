"""Load vendored SnareNet `BaseModel` + `SnareNet` + `SnareNetRepairLayer`.

Files are loaded via `importlib` with transient `sys.modules` entries (scrubbed on
exit), and upstream's module-level `torch.set_default_dtype(float64)` is undone.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from types import ModuleType

import torch

_HERE = os.path.dirname(__file__)
_UPSTREAM_MODELS = os.path.join(_HERE, "upstream", "models")


@contextmanager
def _preserve_default_dtype():
    prev = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _load_file_as(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_vendored():
    """Return `(BaseModel, SnareNet, SnareNetRepairLayer)`; idempotent."""
    models_pkg = ModuleType("models")
    models_pkg.__path__ = [_UPSTREAM_MODELS]
    sys.modules["models"] = models_pkg

    try:
        with _preserve_default_dtype():
            bm = _load_file_as(
                "models.base_model",
                os.path.join(_UPSTREAM_MODELS, "base_model.py"),
            )
            sn = _load_file_as(
                "models.snarenet",
                os.path.join(_UPSTREAM_MODELS, "snarenet.py"),
            )
        BaseModel = bm.BaseModel
        SnareNet = sn.SnareNet
        SnareNetRepairLayer = sn.SnareNetRepairLayer
    finally:
        for key in [
            k for k in list(sys.modules)
            if k == "models" or k.startswith("models.")
        ]:
            del sys.modules[key]

    return BaseModel, SnareNet, SnareNetRepairLayer
