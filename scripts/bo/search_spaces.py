#!/usr/bin/env python3
"""BO search spaces and the Ax-param -> ``--set`` override mapping.

The logical ``lr`` key is mapped by the cell executor; all other keys are native config fields.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

# ALM searches u = log10(1 - alpha), u in [-3, -1], i.e. alpha in [0.9, 0.999].


def alpha_from_u(u: float) -> float:
    """alpha = 1 - 10**u. u=-3 -> 0.999, u=-1 -> 0.9."""
    return 1.0 - (10.0 ** u)


def u_from_alpha(alpha: float) -> float:
    """u = log10(1 - alpha). Inverse of :func:`alpha_from_u`."""
    return math.log10(1.0 - alpha)


# snarenet: soft_epochs + decay_epochs must fit the epoch budget (clip decay, else re-ask).

SNARENET_EPOCH_BUDGET = 2000
_SNARENET_DECAY_CHOICES = (100, 250, 500, 1000)


@dataclass(frozen=True)
class GuardResult:
    action: str  # "ok" | "clipped" | "reject"
    params: dict[str, Any]
    note: str


def apply_snarenet_guard(
    params: Mapping[str, Any],
    budget: int = SNARENET_EPOCH_BUDGET,
    decay_choices: tuple[int, ...] = _SNARENET_DECAY_CHOICES,
) -> GuardResult:
    """Enforce ``soft_epochs + decay_epochs <= budget`` for snarenet.

    Returns a GuardResult whose ``params`` is the (possibly clipped) config to
    evaluate. Pure; the controller logs every non-"ok" action to the ledger.
    """
    out = dict(params)
    soft = int(out.get("soft_epochs", 0))
    decay = int(out.get("decay_epochs", 0))
    if soft + decay <= budget:
        return GuardResult("ok", out, "")
    room = budget - soft
    fitting = [c for c in decay_choices if c <= room]
    if not fitting:
        return GuardResult(
            "reject",
            out,
            f"soft_epochs={soft} leaves no decay_epochs choice <= {room}",
        )
    clipped = max(fitting)
    out["decay_epochs"] = clipped
    return GuardResult(
        "clipped",
        out,
        f"decay_epochs {decay} -> {clipped} (soft_epochs={soft}, budget={budget})",
    )


_LR_PARAM = {
    "name": "lr",
    "type": "range",
    "bounds": [1e-5, 2.5e-4],
    "value_type": "float",
    "log_scale": True,
}


@dataclass(frozen=True)
class MethodSpace:
    method: str
    ax_parameters: list[dict[str, Any]]
    to_overrides: Callable[[Mapping[str, Any]], dict[str, str]]
    sobol: int
    bo: int
    knobs: tuple[str, ...]  # documentation / arity

    @property
    def total(self) -> int:
        return self.sobol + self.bo


def _fmt(v: Any) -> str:
    """Render a native override value as a string for ``--set``.

    Ints stay integral (so ``--set soft_epochs=250`` casts cleanly to int);
    floats use repr to preserve precision.
    """
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


def _ov_pal_loggap(p: Mapping[str, Any]) -> dict[str, str]:
    return {
        "lr": _fmt(p["lr"]),
        "rate": _fmt(p["rate"]),
        "max_decades": _fmt(p["max_decades"]),
    }


def _ov_dc3(p: Mapping[str, Any]) -> dict[str, str]:
    return {
        "lr": _fmt(p["lr"]),
        "soft_weight": _fmt(int(p["soft_weight"])),
        "corr_train_steps": _fmt(int(p["corr_train_steps"])),
        "corr_lr": _fmt(p["corr_lr"]),
    }


def _ov_alm(p: Mapping[str, Any]) -> dict[str, str]:
    return {
        "lr": _fmt(p["lr"]),
        "gamma": _fmt(p["gamma"]),
        "alpha": _fmt(alpha_from_u(float(p["alpha_u"]))),
        "eps": _fmt(p["eps"]),
    }


def _ov_fsnet(p: Mapping[str, Any]) -> dict[str, str]:
    return {
        "lr": _fmt(p["lr"]),
        "dist_weight": _fmt(float(p["dist_weight"])),
        "weight_decay": _fmt(p["weight_decay"]),
    }


def _ov_enforce_orig(p: Mapping[str, Any]) -> dict[str, str]:
    return {
        "lr": _fmt(p["lr"]),
        "adanp_delta": _fmt(p["adanp_delta"]),
        "adanp_warmup_frac": _fmt(p["adanp_warmup_frac"]),
        "adanp_lambda_d": _fmt(p["adanp_lambda_d"]),
    }


def _ov_enforce_v4(p: Mapping[str, Any]) -> dict[str, str]:
    # Other keys are EnforceV4Config fields. FB eps and inference_tolerance stay at defaults.
    return {
        "lr": _fmt(p["lr"]),
        "eps_chol": _fmt(p["eps_chol"]),
        "training_tolerance": _fmt(p["training_tolerance"]),
        "max_it": _fmt(int(p["max_it"])),
        "epoch_start_hard_constrained": _fmt(int(p["epoch_start_hard_constrained"])),
        "ada_np_auto_activation": _fmt(bool(p["ada_np_auto_activation"])),
        "weighting_option": _fmt(int(p["weighting_option"])),
        "weight_loss_displacement": _fmt(p["weight_loss_displacement"]),
    }


def _ov_snarenet(p: Mapping[str, Any]) -> dict[str, str]:
    # apply_snarenet_guard must already have been applied to `p` (the controller does so).
    return {
        "lr": _fmt(p["lr"]),
        "lambd": _fmt(p["lambd"]),
        "soft_weight": _fmt(p["soft_weight"]),
        "soft_epochs": _fmt(int(p["soft_epochs"])),
        "decay_epochs": _fmt(int(p["decay_epochs"])),
    }


# alm_bolton: no lr, only the inference-time projector knobs are searched.
def _ov_alm_bolton(p: Mapping[str, Any]) -> dict[str, str]:
    return {
        "proj_delta": _fmt(p["proj_delta"]),
        "proj_max_iters": _fmt(int(p["proj_max_iters"])),
    }


def _choice(name: str, values: list[Any], value_type: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "choice",
        "values": values,
        "value_type": value_type,
        "is_ordered": True,
        "sort_values": True,
    }


def _range(name: str, lo: float, hi: float, log: bool, vtype: str = "float") -> dict[str, Any]:
    return {
        "name": name,
        "type": "range",
        "bounds": [lo, hi],
        "value_type": vtype,
        "log_scale": log,
    }


METHOD_SPACES: dict[str, MethodSpace] = {
    "pal_loggap": MethodSpace(
        "pal_loggap",
        [dict(_LR_PARAM), _range("rate", 1e-3, 1e-1, True), _range("max_decades", 0.2, 5.0, False)],
        _ov_pal_loggap,
        sobol=8, bo=24, knobs=("lr", "rate", "max_decades"),
    ),
    "dc3": MethodSpace(
        "dc3",
        [
            dict(_LR_PARAM),
            _choice("soft_weight", [1, 10, 100], "int"),
            _choice("corr_train_steps", [1, 2, 5, 10, 100], "int"),
            _range("corr_lr", 1e-7, 1e-4, True),
        ],
        _ov_dc3,
        sobol=8, bo=24, knobs=("lr", "soft_weight", "corr_train_steps", "corr_lr"),
    ),
    "alm": MethodSpace(
        "alm",
        [
            dict(_LR_PARAM),
            _range("gamma", 1e-4, 1e-1, True),
            _range("alpha_u", -3.0, -1.0, False),  # u = log10(1 - alpha)
            _range("eps", 1e-10, 1e-4, True),
        ],
        _ov_alm,
        sobol=8, bo=24, knobs=("lr", "gamma", "alpha_u", "eps"),
    ),
    "fsnet": MethodSpace(
        "fsnet",
        [
            dict(_LR_PARAM),
            _choice("dist_weight", [0.0, 0.5, 2.5, 5.0, 10.0, 20.0, 50.0], "float"),
            _range("weight_decay", 1e-6, 1e-2, True),
        ],
        _ov_fsnet,
        sobol=8, bo=24, knobs=("lr", "dist_weight", "weight_decay"),
    ),
    "enforce_orig": MethodSpace(
        "enforce_orig",
        [
            dict(_LR_PARAM),
            _range("adanp_delta", 1e-6, 1e-1, True),
            _range("adanp_warmup_frac", 0.0, 0.75, False),
            _range("adanp_lambda_d", 1e-2, 10.0, True),
        ],
        _ov_enforce_orig,
        sobol=8, bo=24, knobs=("lr", "adanp_delta", "adanp_warmup_frac", "adanp_lambda_d"),
    ),
    "enforce_v4": MethodSpace(
        "enforce_v4",
        [
            dict(_LR_PARAM),
            _range("eps_chol", 1e-10, 1e-4, True),
            _range("training_tolerance", 1e-6, 1e-2, True),
            _range("max_it", 10, 100, False, vtype="int"),
            # Pinned to the author default 0 (projection active from the first epoch).
            {"name": "epoch_start_hard_constrained", "type": "fixed", "value": 0,
             "value_type": "int"},
            # ada_np_auto_activation {on,off} -> bool field. on=True, off=False.
            {
                "name": "ada_np_auto_activation",
                "type": "choice",
                "values": [False, True],
                "value_type": "bool",
                "is_ordered": True,
                "sort_values": True,
            },
            _choice("weighting_option", [1, 3, 5], "int"),
            _range("weight_loss_displacement", 0.05, 5.0, True),
        ],
        _ov_enforce_v4,
        sobol=8, bo=24,
        knobs=("lr", "eps_chol", "training_tolerance", "max_it",
               "ada_np_auto_activation",
               "weighting_option", "weight_loss_displacement"),
    ),
    "snarenet": MethodSpace(
        "snarenet",
        [
            dict(_LR_PARAM),
            _range("lambd", 1e-3, 10.0, True),
            _range("soft_weight", 1.0, 100.0, True),
            _choice("soft_epochs", [0, 100, 250, 500, 1000], "int"),
            _choice("decay_epochs", [100, 250, 500, 1000], "int"),
        ],
        _ov_snarenet,
        sobol=12, bo=28,
        knobs=("lr", "lambd", "soft_weight", "soft_epochs", "decay_epochs"),
    ),
    "alm_bolton": MethodSpace(
        "alm_bolton",
        [
            _range("proj_delta", 1e-5, 1e-1, True),
            _choice("proj_max_iters", [1, 2, 5, 10, 20], "int"),
        ],
        _ov_alm_bolton,
        sobol=8, bo=12, knobs=("proj_delta", "proj_max_iters"),
    ),
}


def default_budget(method: str) -> tuple[int, int]:
    """Predeclared (sobol, bo) trial counts for a method."""
    sp = METHOD_SPACES[method]
    return sp.sobol, sp.bo


def dc3_effective_soft_weight(requested: float, bench_id: str) -> tuple[float, bool | None]:
    """DC3 auto-x10s soft_weight when completion is unavailable on a bench
    (dc3/solver.py::_dc3_args_dict:176). Returns (effective, use_compl).

    Best-effort: resolves the bench's DC3 partial-vars spec. On any import/
    lookup failure returns (requested, None) so ledger logging never crashes.
    """
    try:
        from pal.baselines.dc3.solver import resolve_partial_vars  # type: ignore
        from pal.benchmarks import get_benchmark  # type: ignore

        use_compl = resolve_partial_vars(get_benchmark(bench_id)) is not None
    except Exception:
        return requested, None
    return requested * (1.0 if use_compl else 10.0), use_compl
