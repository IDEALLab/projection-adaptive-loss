"""Load DC3's algorithmic core from ``upstream/method.py``.

method.py @ 35437af imports several modules at top level:
  - ``waitGPU`` (soft-imported via try/except; ImportError is swallowed).
  - ``setproctitle.setproctitle`` (used only inside ``main()``).
  - ``utils.my_hash``, ``utils.str_to_bool`` (used only inside ``main()``).
  - ``default_args`` (used only inside ``main()``).

The missing modules and the two ``utils`` attrs are stubbed, the file is
loaded via ``importlib.util.spec_from_file_location``, and the stubs are
removed after exec. The re-exported ``grad_steps``, ``grad_steps_all`` and
``total_loss`` close over no state from those stubs.

method.py:10 ``torch.set_default_dtype(torch.float64)`` is undone by
``_preserve_default_dtype``. ``NNSolver`` is not re-exported.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import torch

_HERE = Path(__file__).resolve().parent
_UPSTREAM_METHOD_PY = _HERE / "upstream" / "method.py"


@contextmanager
def _preserve_default_dtype():
    prev = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _make_utils_stub() -> ModuleType:
    mod = ModuleType("utils")
    mod.my_hash = lambda s: "dummy"
    mod.str_to_bool = lambda s: True
    return mod


def _make_default_args_stub() -> ModuleType:
    mod = ModuleType("default_args")
    mod.method_default_args = lambda probType: {}
    return mod


def _make_setproctitle_stub() -> ModuleType:
    mod = ModuleType("setproctitle")
    mod.setproctitle = lambda *a, **k: None
    return mod


_STUB_FACTORIES = {
    "utils": _make_utils_stub,
    "default_args": _make_default_args_stub,
    "setproctitle": _make_setproctitle_stub,
}


def load_vendored():
    """Return ``(grad_steps, grad_steps_all, total_loss)`` from method.py.

    Idempotent: stubs are restored to their pre-call state, the module
    cache key for the loaded method.py is removed, and the caller's torch
    default dtype is preserved across the call.
    """
    prior_modules: dict[str, ModuleType | None] = {
        name: sys.modules.get(name) for name in _STUB_FACTORIES
    }
    for name, factory in _STUB_FACTORIES.items():
        if name not in sys.modules:
            sys.modules[name] = factory()

    method_mod_key = "pal.baselines.dc3.upstream.method"
    prior_method_mod = sys.modules.pop(method_mod_key, None)

    try:
        with _preserve_default_dtype():
            spec = importlib.util.spec_from_file_location(
                method_mod_key, str(_UPSTREAM_METHOD_PY)
            )
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"could not build spec for {_UPSTREAM_METHOD_PY}"
                )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[method_mod_key] = mod
            spec.loader.exec_module(mod)
        grad_steps = mod.grad_steps
        grad_steps_all = mod.grad_steps_all
        total_loss = mod.total_loss
    finally:
        sys.modules.pop(method_mod_key, None)
        if prior_method_mod is not None:
            sys.modules[method_mod_key] = prior_method_mod
        for name, prior in prior_modules.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior

    return grad_steps, grad_steps_all, total_loss
