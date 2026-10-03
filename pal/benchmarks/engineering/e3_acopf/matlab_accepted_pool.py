"""MATLAB-accepted load pool for e3 ACOPF.

Port of the DC3 data generation: perturb nominal loads, solve ACOPF with thermal limits
disabled, accept converged samples. Thermal and angle limits are not enforced on the pool.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.resources
import io

import numpy as np
import torch
from torch import Tensor

CASES: tuple[str, ...] = ("ieee30", "ieee57", "ieee118")

POOL_FORMAT_VERSION = 2

MATLAB_PARAMS: dict[str, float] = {
    "MaxChangeLoad": 0.10,   # max fractional load change
    "CorrCoeff": 0.75,        # load-to-load correlation coefficient
    "MIN_PF": 0.80,           # min power factor
    "MAX_PF": 1.00,           # max power factor
    "sampler_seed": 42,       # perturbation draw seed
    "split_seed": 1337,       # train/eval partition seed
    # Loose vm bounds used only during generation (PIPS diverges on ieee57 otherwise).
    "gen_vm_lower": 0.5,
    "gen_vm_upper": 1.5,
}


class PoolCaseMismatch(RuntimeError):
    """Pool case fingerprint does not match the live benchmark grid."""


class PoolNotPackaged(FileNotFoundError):
    """Pool .pt file is missing from the installed package data."""


def _case_fingerprint(data: dict[str, Tensor], case_name: str) -> dict:
    """Structural fingerprint of the case data, checked by `load_pool`."""
    return {
        "case_name": case_name,
        "baseMVA": 100.0,
        "n_load": int(data["L"].item()),
        "n_bus": int(data["N"].item()),
        "n_gen": int(data["G"].item()),
        "pd_nominal_sha256": hashlib.sha256(
            data["pd_nominal"].detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "qd_nominal_sha256": hashlib.sha256(
            data["qd_nominal"].detach().cpu().numpy().tobytes()
        ).hexdigest(),
    }


def _sample_loads(
    pd_nom: np.ndarray,          # [n_load]
    qd_nom: np.ndarray,          # [n_load]
    n: int,
    rng: np.random.Generator,
    *,
    max_change: float,
    corr_coeff: float,
    min_pf: float,
    max_pf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n perturbed load vectors, DC3 MATLAB-style.

    Per sample:
      global_eps ~ U[-max_change, +max_change]                            (scalar)
      local_eps  ~ U[-max_change, +max_change]                          (per load)
      mult = 1 + corr*global_eps + sqrt(1 - corr**2)*local_eps
      pd   = max(pd_nom * mult, 0)
      PF   ~ U[min_pf, max_pf]                                        (per sample)
      qd   = sign(qd_nom) * pd * tan(acos(PF))
    """
    n_load = pd_nom.shape[0]
    global_eps = rng.uniform(-max_change, max_change, size=(n, 1))
    local_eps = rng.uniform(-max_change, max_change, size=(n, n_load))
    mult = 1.0 + corr_coeff * global_eps + np.sqrt(1.0 - corr_coeff**2) * local_eps
    pd = np.maximum(pd_nom[None, :] * mult, 0.0)
    pf = rng.uniform(min_pf, max_pf, size=(n, 1))
    q_over_p = np.tan(np.arccos(pf))
    sign_q = np.where(qd_nom >= 0, 1.0, -1.0)[None, :]
    qd = sign_q * pd * q_over_p
    return pd.astype(np.float32), qd.astype(np.float32)


def _relax_solve_bounds(net, gen_vm_lower: float, gen_vm_upper: float) -> dict:
    """Relax thermal + voltage bounds for solve; return a restore-state dict.

    - Thermal: zero out line + trafo limits (MATLAB branch(:,6:8)=0 analogue).
    - Voltage: widen bus vm bounds to [gen_vm_lower, gen_vm_upper].
    """
    saved = {
        "line_mlp": net.line["max_loading_percent"].copy() if "max_loading_percent" in net.line else None,
        "line_max_i_ka": net.line["max_i_ka"].copy() if "max_i_ka" in net.line else None,
        "trafo_mlp": net.trafo["max_loading_percent"].copy() if "max_loading_percent" in net.trafo else None,
        "bus_max_vm_pu": net.bus["max_vm_pu"].copy() if "max_vm_pu" in net.bus else None,
        "bus_min_vm_pu": net.bus["min_vm_pu"].copy() if "min_vm_pu" in net.bus else None,
    }
    if "max_loading_percent" in net.line:
        net.line["max_loading_percent"] = 1.0e9
    if "max_i_ka" in net.line:
        net.line["max_i_ka"] = 1.0e6
    if "max_loading_percent" in net.trafo:
        net.trafo["max_loading_percent"] = 1.0e9
    if "max_vm_pu" in net.bus:
        net.bus["max_vm_pu"] = gen_vm_upper
    if "min_vm_pu" in net.bus:
        net.bus["min_vm_pu"] = gen_vm_lower
    return saved


def _restore_solve_bounds(net, saved: dict) -> None:
    if saved["line_mlp"] is not None:
        net.line["max_loading_percent"] = saved["line_mlp"]
    if saved["line_max_i_ka"] is not None:
        net.line["max_i_ka"] = saved["line_max_i_ka"]
    if saved["trafo_mlp"] is not None:
        net.trafo["max_loading_percent"] = saved["trafo_mlp"]
    if saved["bus_max_vm_pu"] is not None:
        net.bus["max_vm_pu"] = saved["bus_max_vm_pu"]
    if saved["bus_min_vm_pu"] is not None:
        net.bus["min_vm_pu"] = saved["bus_min_vm_pu"]


def _extract_solution(
    net, n_ext_grid: int, baseMVA: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull (pg, qg, vm, va) from a solved pandapower net in PAL gen order (ext_grid first)."""
    pg_ext = net.res_ext_grid["p_mw"].to_numpy() / baseMVA
    qg_ext = net.res_ext_grid["q_mvar"].to_numpy() / baseMVA
    pg_gen = net.res_gen["p_mw"].to_numpy() / baseMVA
    qg_gen = net.res_gen["q_mvar"].to_numpy() / baseMVA
    pg = np.concatenate([pg_ext, pg_gen], axis=0).astype(np.float32)
    qg = np.concatenate([qg_ext, qg_gen], axis=0).astype(np.float32)
    vm = net.res_bus["vm_pu"].to_numpy().astype(np.float32)
    va = np.deg2rad(net.res_bus["va_degree"].to_numpy()).astype(np.float32)
    return pg, qg, vm, va


def generate_matlab_accepted_pool(
    case_name: str,
    *,
    target_accepted: int = 1200,
    train_size: int = 1000,
    eval_size: int = 200,
    use_numba: bool = False,
    max_attempts: int | None = None,
    batch_log_every: int = 200,
    verbose: bool = True,
) -> dict:
    """Run the DC3-style accept/reject loop and return a pool dict.

    Args:
        case_name: one of `CASES`.
        target_accepted: stop after this many accepted samples.
        train_size, eval_size: partition sizes; must sum to <= target_accepted.
        use_numba: pass `numba=True` to pandapower's OPF (~5x faster but adds
            a numba dependency). Defaults to False for portability.
        max_attempts: safety cap; defaults to `target_accepted * 20`.
        batch_log_every: progress-line cadence.
        verbose: print progress lines.

    Returns:
        A pool dict with pd/qd/pg/qg/vm/va tensors and a `metadata` dict.
    """
    if case_name not in CASES:
        raise ValueError(f"unknown case {case_name!r}; expected one of {CASES}")
    if train_size + eval_size > target_accepted:
        raise ValueError(
            f"train_size + eval_size ({train_size + eval_size}) "
            f"exceeds target_accepted ({target_accepted})"
        )
    if max_attempts is None:
        max_attempts = target_accepted * 20

    import pandapower as pp
    import pandapower.networks as pn

    from .grid_adapter import pandapower_to_ml4opf

    loaders = {"ieee30": pn.case_ieee30, "ieee57": pn.case57, "ieee118": pn.case118}
    net = loaders[case_name]()
    pp.runpp(net, numba=False)
    baseMVA = float(net._ppc["baseMVA"])
    n_ext_grid = len(net.ext_grid)

    data = pandapower_to_ml4opf(case_name)
    fp = _case_fingerprint(data, case_name)

    pd_nom_arr = (data["pd_nominal"].detach().cpu().numpy()
                  if isinstance(data["pd_nominal"], Tensor)
                  else np.asarray(data["pd_nominal"]))
    qd_nom_arr = (data["qd_nominal"].detach().cpu().numpy()
                  if isinstance(data["qd_nominal"], Tensor)
                  else np.asarray(data["qd_nominal"]))

    load_p_nom_mw = net.load["p_mw"].to_numpy().copy()
    load_q_nom_mvar = net.load["q_mvar"].to_numpy().copy()

    rng = np.random.default_rng(int(MATLAB_PARAMS["sampler_seed"]))
    saved = _relax_solve_bounds(
        net,
        gen_vm_lower=float(MATLAB_PARAMS["gen_vm_lower"]),
        gen_vm_upper=float(MATLAB_PARAMS["gen_vm_upper"]),
    )

    pd_accepted: list[np.ndarray] = []
    qd_accepted: list[np.ndarray] = []
    pg_accepted: list[np.ndarray] = []
    qg_accepted: list[np.ndarray] = []
    vm_accepted: list[np.ndarray] = []
    va_accepted: list[np.ndarray] = []

    n_attempted = 0
    batch = max(64, target_accepted // 10)
    try:
        while len(pd_accepted) < target_accepted:
            if n_attempted >= max_attempts:
                break
            pd_batch, qd_batch = _sample_loads(
                pd_nom_arr, qd_nom_arr, batch, rng,
                max_change=float(MATLAB_PARAMS["MaxChangeLoad"]),
                corr_coeff=float(MATLAB_PARAMS["CorrCoeff"]),
                min_pf=float(MATLAB_PARAMS["MIN_PF"]),
                max_pf=float(MATLAB_PARAMS["MAX_PF"]),
            )
            for k in range(batch):
                n_attempted += 1
                net.load["p_mw"] = pd_batch[k] * baseMVA
                net.load["q_mvar"] = qd_batch[k] * baseMVA
                # Silence pandapower's per-solve informational prints.
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        pp.runopp(net, numba=use_numba, suppress_warnings=True)
                except Exception:
                    continue
                if not net.OPF_converged:
                    continue
                pg, qg, vm, va = _extract_solution(net, n_ext_grid, baseMVA)
                if not (np.isfinite(pg).all() and np.isfinite(qg).all()
                        and np.isfinite(vm).all() and np.isfinite(va).all()):
                    continue
                pd_accepted.append(pd_batch[k])
                qd_accepted.append(qd_batch[k])
                pg_accepted.append(pg)
                qg_accepted.append(qg)
                vm_accepted.append(vm)
                va_accepted.append(va)
                if len(pd_accepted) >= target_accepted:
                    break
                if verbose and len(pd_accepted) % batch_log_every == 0:
                    rate = len(pd_accepted) / max(n_attempted, 1)
                    print(
                        f"  [{case_name}] accepted {len(pd_accepted)}/{target_accepted} "
                        f"(attempts={n_attempted}, accept_rate={rate:.1%})"
                    )
    finally:
        net.load["p_mw"] = load_p_nom_mw
        net.load["q_mvar"] = load_q_nom_mvar
        _restore_solve_bounds(net, saved)

    n_accepted = len(pd_accepted)
    if n_accepted < train_size + eval_size:
        raise RuntimeError(
            f"generation hit max_attempts={max_attempts} with only "
            f"{n_accepted} accepted (<{train_size + eval_size} required). "
            "Increase max_attempts or loosen MaxChangeLoad."
        )

    pd_t = torch.from_numpy(np.stack(pd_accepted, axis=0))
    qd_t = torch.from_numpy(np.stack(qd_accepted, axis=0))
    pg_t = torch.from_numpy(np.stack(pg_accepted, axis=0))
    qg_t = torch.from_numpy(np.stack(qg_accepted, axis=0))
    vm_t = torch.from_numpy(np.stack(vm_accepted, axis=0))
    va_t = torch.from_numpy(np.stack(va_accepted, axis=0))

    split_rng = np.random.default_rng(int(MATLAB_PARAMS["split_seed"]))
    perm = split_rng.permutation(n_accepted)
    train_idx = torch.from_numpy(perm[:train_size]).long()
    eval_idx = torch.from_numpy(perm[train_size:train_size + eval_size]).long()

    # Violation rates of the benchmark constraints on the OPF solutions themselves.
    diag = _compute_violation_diagnostic(
        pd_t, qd_t, pg_t, qg_t, vm_t, va_t, data,
    )

    metadata: dict = {
        "case_name": case_name,
        "case_fingerprint": fp,
        "matlab_params": dict(MATLAB_PARAMS),
        "n_attempted": n_attempted,
        "n_accepted": n_accepted,
        "split_sizes": {"train": int(train_size), "eval": int(eval_size)},
        "split_indices": {"train": train_idx, "eval": eval_idx},
        "thermal_limits_during_solve": False,
        "solve_vm_bounds": (
            float(MATLAB_PARAMS["gen_vm_lower"]),
            float(MATLAB_PARAMS["gen_vm_upper"]),
        ),
        "pandapower_version": pp.__version__,
        "violation_diagnostic": diag,
        "format_version": POOL_FORMAT_VERSION,
    }

    return {
        "pd": pd_t, "qd": qd_t,
        "pg": pg_t, "qg": qg_t,
        "vm": vm_t, "va": va_t,
        "metadata": metadata,
    }


def _compute_violation_diagnostic(
    pd: Tensor, qd: Tensor,
    pg: Tensor, qg: Tensor, vm: Tensor, va: Tensor,
    data: dict[str, Tensor],
) -> dict:
    """Violation rates of the PAL constraint set on the accepted pool."""
    from . import _safe_ops

    with torch.no_grad():
        viols = _safe_ops.calc_violations(pd, qd, pg, qg, vm, va, data=data)

    groups = [
        "p_balance", "q_balance",
        "vm_lower", "vm_upper",
        "pg_lower", "pg_upper",
        "qg_lower", "qg_upper",
        "thrm_1", "thrm_2",
        "dva_lower", "dva_upper",
    ]
    out: dict[str, dict] = {}
    for g in groups:
        v = viols[g]  # [N, K]
        if g in ("p_balance", "q_balance"):
            # Equality: violated when |residual| exceeds the benchmark tol.
            tol = 1e-3
            per_sample = v.abs().amax(dim=-1)
            viol_mask = per_sample > tol
        else:
            # Inequality: violated when the residual exceeds the 1e-2 margin.
            margin = 1e-2
            per_sample = v.amax(dim=-1).clamp(min=0.0)
            viol_mask = v.amax(dim=-1) > margin
        out[g] = {
            "any_violated_ratio": float(viol_mask.float().mean().item()),
            "max_residual": float(per_sample.max().item()),
            "mean_residual": float(per_sample.mean().item()),
        }
    return out


def describe_pool(pool: dict) -> str:
    """Format a human-readable summary of a pool dict."""
    md = pool["metadata"]
    fp = md["case_fingerprint"]
    diag = md["violation_diagnostic"]
    lines: list[str] = []
    lines.append(
        f"Case {md['case_name']} MATLAB-accepted pool "
        f"({md['n_accepted']} samples, sampler_seed={md['matlab_params']['sampler_seed']})"
    )
    lines.append(
        f"  MATPOWER convergence rate: "
        f"{md['n_accepted'] / max(md['n_attempted'], 1):.1%} "
        f"({md['n_attempted']} attempted)"
    )
    lines.append(
        f"  Splits: train={md['split_sizes']['train']}, "
        f"eval={md['split_sizes']['eval']} (split_seed={md['matlab_params']['split_seed']})"
    )
    lines.append(
        f"  Grid: n_bus={fp['n_bus']}, n_gen={fp['n_gen']}, n_load={fp['n_load']}"
    )
    lines.append("  PAL constraint violation rates (thermal/dva not enforced during solve):")
    for g in ("p_balance", "q_balance", "thrm_1", "thrm_2",
              "dva_lower", "dva_upper", "vm_lower", "vm_upper"):
        d = diag[g]
        lines.append(
            f"    {g:10s}  any={d['any_violated_ratio']:6.1%}  "
            f"max|res|={d['max_residual']:.3e}"
        )
    return "\n".join(lines)


def _pool_filename(case_name: str) -> str:
    return f"matlab_accepted_pool_{case_name}.pt"


def save_pool(pool: dict, path) -> None:
    """Persist a pool dict to disk via torch.save."""
    torch.save(pool, path)


def load_pool(
    case_name: str,
    *,
    live_data: dict[str, Tensor] | None = None,
    path=None,
) -> dict:
    """Load a pool for `case_name` and verify it matches the live grid data.

    Raises:
        PoolNotPackaged: pool file missing from package data.
        PoolCaseMismatch: fingerprint differs from `live_data`.
    """
    if case_name not in CASES:
        raise ValueError(f"unknown case {case_name!r}; expected one of {CASES}")

    if path is None:
        pkg_files = importlib.resources.files(
            "pal.benchmarks.engineering.e3_acopf.data"
        )
        res = pkg_files / _pool_filename(case_name)
        if not res.is_file():
            raise PoolNotPackaged(
                f"pool file {_pool_filename(case_name)} missing from package "
                "data; regenerate via `python scripts/gen_e3_pools.py --case "
                f"{case_name}` and reinstall."
            )
        with importlib.resources.as_file(res) as p:
            pool = torch.load(p, map_location="cpu", weights_only=False)
    else:
        pool = torch.load(path, map_location="cpu", weights_only=False)

    md = pool["metadata"]
    if md.get("format_version") != POOL_FORMAT_VERSION:
        raise PoolCaseMismatch(
            f"pool format_version={md.get('format_version')} != "
            f"{POOL_FORMAT_VERSION}; regenerate the pool."
        )
    if live_data is not None:
        live_fp = _case_fingerprint(live_data, case_name)
        stored_fp = md["case_fingerprint"]
        if live_fp != stored_fp:
            raise PoolCaseMismatch(
                f"case fingerprint mismatch for {case_name}.\n"
                f"  stored: {stored_fp}\n"
                f"  live:   {live_fp}\n"
                "Regenerate the pool via "
                f"`python scripts/gen_e3_pools.py --case {case_name}`."
            )
    return pool
