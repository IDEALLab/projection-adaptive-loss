"""`pal run` CLI entry point.

Runs the cartesian product `--method M1,M2 --benchmarks B1,B2 --seeds 0,1`, one
run directory per triple. Classical solvers (`slsqp`, `ipopt`) have no model and
solve the eval set directly.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from pal.baselines import (
    ALMBoltOnConfig,
    ALMBoltOnSolver,
    ALMConfig,
    ALMSolver,
    DC3Config,
    DC3Solver,
    EnforceOrigConfig,
    EnforceOrigSolver,
    EnforceV4Config,
    EnforceV4Solver,
    FSNetConfig,
    FSNetSolver,
    SnareNetConfig,
    SnareNetSolver,
)
from pal.baselines.hparams import load_hparams
from pal.baselines.slsqp import SLSQPConfig, SLSQPSolver
from pal.benchmarks import get as get_benchmark
from pal.benchmarks import list_filtered as list_benchmarks_filtered
from pal.eval import EvalResult, query_sha256, run_final_eval, run_inference
from pal.method import (
    PALIpConfig,
    PALIpSolver,
    PALLogGapConfig,
    PALLogGapSolver,
    PALSqpConfig,
    PALSqpSolver,
)
from pal.runner.probe import BenchProbe
from pal.runner.shard import select_shard
from pal.solvers.base import TrainResult
from pal.tracking import CompositeLogger, JSONLLogger, create_run_dir
from pal.tracking.step_logger import save_local_artifacts

_SUPPORTED_METHODS = {"pal_loggap", "pal_sqp", "pal_ip", "alm", "alm_bolton", "dc3", "enforce_orig", "enforce_v4", "fsnet", "snarenet", "slsqp", "ipopt"}
_CLASSICAL_METHODS = {"slsqp", "ipopt"}

# `None` means --epochs was not passed: methods fall back to their own default.
_CLI_DEFAULT_EPOCHS = None
_DEFAULT_EPOCHS = 2000
_CLI_DEFAULT_BATCH_SIZE = 32
_CLI_DEFAULT_LR = 1e-4
_CLI_DEFAULT_GRAD_CLIP = 1.0


class _NullLogger:
    """No-op logger for non-zero DDP ranks."""

    def log_config(self, **cfg: Any) -> None:
        return None

    def log_step(self, step: int, **scalars: float) -> None:
        return None

    def log_projection_trajectory(
        self, step: int, phase: str, trajectory: list[Any]
    ) -> None:
        return None

    def log_artifact(self, step: int, name: str, payload: Any) -> None:
        return None

    def log_final(self, **final: Any) -> None:
        return None

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        return None


def _dist_is_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    return dist.get_rank() if _dist_is_active() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if _dist_is_active() else 1


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _write_eval_rows_parquet(
    run_dir: Path,
    eval_rows: list[dict[str, Any]] | None,
    *,
    method: str,
    bench_id: str,
    seed: int,
    tolerance: float,
    eval_queries_fingerprint: str | None,
    source: str,
) -> Path | None:
    if not eval_rows:
        return None
    import polars as pl

    rows = []
    for row in eval_rows:
        rows.append(
            {
                "run_id": run_dir.name,
                "run_path": str(run_dir),
                "method": method,
                "benchmark_id": bench_id,
                "seed": int(seed),
                "source": source,
                "tolerance": float(tolerance),
                "eval_queries_fingerprint": eval_queries_fingerprint,
                **row,
            }
        )
    out = run_dir / "eval_rows.parquet"
    pl.from_dicts(rows).sort("query_idx").write_parquet(out)
    return out


def _maybe_init_distributed(device: str) -> None:
    if _dist_is_active():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    backend = "nccl" if device.startswith("cuda") else "gloo"
    dist.init_process_group(backend=backend)
    if device.startswith("cuda"):
        torch.cuda.set_device(_local_rank())


def _broadcast_string(value: str | None) -> str:
    if not _dist_is_active():
        return value or ""
    payload = [value if _dist_is_main() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0] or ""


def _safe_barrier(label: str) -> None:
    """`dist.barrier()` that doesn't propagate as DistBackendError.

    Catching here lets the run finish cleanly when another rank crashed and the
    NCCL watchdog kills the stuck collective.
    """
    if not _dist_is_active():
        return
    try:
        dist.barrier()
    except Exception as e:  # noqa: BLE001
        print(f"[pal run] {label} barrier failed: {e}", file=sys.stderr)


def _build_solver(method: str, cfg, bench_id: str | None = None):
    if method == "pal_loggap":
        return PALLogGapSolver(cfg)
    if method == "pal_sqp":
        return PALSqpSolver(cfg)
    if method == "pal_ip":
        return PALIpSolver(cfg)
    if method == "alm":
        return ALMSolver(cfg)
    if method == "alm_bolton":
        return ALMBoltOnSolver(cfg)
    if method == "dc3":
        return DC3Solver(cfg, bench_id=bench_id)
    if method == "enforce_orig":
        return EnforceOrigSolver(cfg)
    if method == "enforce_v4":
        return EnforceV4Solver(cfg)
    if method == "fsnet":
        return FSNetSolver(cfg)
    if method == "snarenet":
        return SnareNetSolver(cfg)
    if method == "slsqp":
        return SLSQPSolver(cfg)
    if method == "ipopt":
        try:
            from pal.baselines.ipopt import IPOPTSolver
        except ImportError as e:
            raise RuntimeError(
                "IPOPT baseline requires cyipopt. Install in a CPU env via "
                "`conda install -c conda-forge cyipopt`. Not supported in the "
                "GH200 container, run this baseline off-cluster. "
                "See pal/baselines/ipopt/README.md. "
                f"({type(e).__name__}: {e})"
            ) from e
        return IPOPTSolver(cfg)
    raise ValueError(
        f"unknown method '{method}'. supported: {sorted(_SUPPORTED_METHODS)}."
    )


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_branch(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_diff(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        return ""


def _resolve_batch_size(args: argparse.Namespace, bench_id: str | None) -> tuple[int, bool]:
    """Resolve training batch size + whether to override YAML defaults.

    Returns `(batch, should_override_yaml)`; the YAML value is overridden when
    `--batch-size` is explicit or the bench declares `spec.train_batch_size`.
    """
    explicit_cli = args.batch_size != _CLI_DEFAULT_BATCH_SIZE
    if explicit_cli:
        return int(args.batch_size), True
    if bench_id is None:
        return int(args.batch_size), False
    spec = get_benchmark(bench_id).spec
    if spec.train_batch_size is not None:
        return int(spec.train_batch_size), True
    return int(args.batch_size), False


def _apply_set_overrides(cfg, overrides: list[str]):
    """Apply `--set KEY=VALUE` pairs to a dataclass config in place.

    Values are cast via the field type declared on the dataclass. Unknown keys
    raise `ValueError`.
    """
    if not overrides:
        return cfg
    fields_by_name = {f.name: f for f in dataclasses.fields(cfg)}
    for pair in overrides:
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got: {pair!r}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if key not in fields_by_name:
            valid = ", ".join(sorted(fields_by_name))
            raise ValueError(
                f"--set: '{key}' is not a field of {type(cfg).__name__}. "
                f"Valid keys: {valid}"
            )
        field_type = fields_by_name[key].type
        # Field types may be strings under `from __future__ import annotations`.
        if isinstance(field_type, str):
            type_name = field_type
        else:
            type_name = getattr(field_type, "__name__", str(field_type))
        if type_name == "bool":
            lowered = raw.lower()
            if lowered in ("true", "1", "yes", "y"):
                value = True
            elif lowered in ("false", "0", "no", "n"):
                value = False
            else:
                raise ValueError(f"--set {key}={raw!r}: expected bool")
        elif type_name == "int":
            value = int(raw)
        elif type_name == "float":
            value = float(raw)
        elif type_name == "str":
            value = raw
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        setattr(cfg, key, value)
    return cfg


def _resolve_eval_cfg_overrides(
    cfg,
    set_overrides: list[str],
    forced: dict[str, Any],
) -> dict[str, Any]:
    """Apply eval-time cfg mutations and report what actually landed.

    `forced` is applied first so an explicit `--set` always wins over it.
    Returns `{field: value}` read back off `cfg`.
    """
    touched: list[str] = []
    for key, value in forced.items():
        setattr(cfg, key, value)
        touched.append(key)
    _apply_set_overrides(cfg, set_overrides)
    for pair in set_overrides:
        touched.append(pair.split("=", 1)[0].strip())
    return {key: getattr(cfg, key) for key in dict.fromkeys(touched)}


def _is_paper_faithful(args: argparse.Namespace, bench_id: str | None) -> bool:
    """`--protocol paper-faithful` applies to e3 only."""
    return (
        getattr(args, "protocol", "synthetic") == "paper-faithful"
        and bench_id is not None
        and bench_id.startswith("e3/")
    )


def _build_eval_queries(
    bench, n_eval: int | None, seed: int, *,
    paper_faithful: bool,
    eval_points: str | None = None,
):
    """Construct the fingerprinted eval set handed to every method.

    With `paper_faithful`, use the pool's "eval" split with zero zeta. With
    `eval_points`, load the frozen-points JSON and use those queries verbatim.
    """
    from pal.benchmarks.base import Query

    if eval_points is not None:
        from pal.eval.frozen_points import load_frozen_points
        fp = load_frozen_points(eval_points)
        if len(fp) == 0:
            raise ValueError(
                f"--eval-points {eval_points!r} contains no points; "
                "pass a frozen eval-points file that lists at least one point."
            )
        return fp.to_query(bench.spec)
    if paper_faithful:
        n = n_eval if n_eval is not None else int(bench.spec.n_eval_default)
        q = bench.sample_queries(n=n, split="eval", seed=seed)
        return Query(zeta=torch.zeros_like(q.zeta), conditions=q.conditions)
    return bench.eval_queries(seed, n=n_eval)


def _resolve_precision(args: argparse.Namespace, spec) -> str:
    """Effective precision for one run: `--precision` if given, else the spec's.

    Args:
        args: Parsed `run` namespace (`--precision` may be absent/None).
        spec: The benchmark's `BenchmarkSpec`.

    Returns:
        One of "fp32" | "bf16" | "fp16" | "fp64".
    """
    return getattr(args, "precision", None) or getattr(spec, "precision", "fp32")


@contextlib.contextmanager
def _use_dtype(dtype: torch.dtype):
    """Set torch's default dtype for the enclosed block, restore afterwards.

    Args:
        dtype: Default dtype to install.

    Yields:
        None.
    """
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev_dtype)


@contextlib.contextmanager
def _default_dtype(precision: str):
    """Set torch's default dtype to float64 for the enclosed block on fp64.

    Args:
        precision: Effective precision string.

    Yields:
        None.
    """
    if precision != "fp64":
        yield
        return
    with _use_dtype(torch.float64):
        yield


def _cast_queries(queries, precision: str):
    """Cast an eval `Query` to float64 when running fp64.

    Benchmarks build some query tensors as float32, so `set_default_dtype`
    alone does not make them fp64.

    Args:
        queries: The fingerprinted eval `Query`.
        precision: Effective precision string.

    Returns:
        The queries, cast to float64 for fp64, unchanged otherwise.
    """
    if precision != "fp64":
        return queries
    from pal.benchmarks.base import Query

    return Query(
        zeta=queries.zeta.to(torch.float64),
        conditions=queries.conditions.to(torch.float64),
    )


# Paper-faithful (e3) overrides for the vendored baselines: 200x2 backbone,
# 2000 epochs at batch 200, and zero zeta. snarenet rtol=1e-6 is the fp32
# translation of the upstream fp64 rtol=1e-8.
_PAPER_FAITHFUL_BASELINE_OVERRIDES: dict[str, dict[str, Any]] = {
    "fsnet": dict(
        hidden_dim=200,
        num_layers=2,
        zeta_zero=True,
        epochs=2000,
        batch_size=200,
        steps_per_epoch=1,
        lr=5e-4,
    ),
    "snarenet": dict(
        hidden_size=200,
        num_hidden_layers=2,
        zeta_zero=True,
        epochs=2000,
        batch_size=200,
        rtol=1e-6,
    ),
    "enforce_v4": dict(
        hidden=200,
        n_layers=2,
        zeta_zero=True,
        epochs=2000,
        batch_size=200,
    ),
    # Predict-time knob, applied by `pal eval` when an e3 paper-faithful run is
    # evaluated as alm_bolton. It sets the LM projector damping floor to the
    # pal_loggap value. alm_bolton trains with the shared defaults below.
    "alm_bolton": dict(
        proj_lambda_min=1e-4,
    ),
}

# Methods whose entry above is applied only at eval time.
_PAPER_FAITHFUL_EVAL_ONLY = frozenset({"alm_bolton"})


def _paper_faithful_eval_overrides(method: str, cfg) -> dict[str, Any]:
    """Eval-time knobs for `pal eval` of an e3 paper-faithful run."""
    if method not in _PAPER_FAITHFUL_EVAL_ONLY:
        return {}
    return {
        key: value
        for key, value in _PAPER_FAITHFUL_BASELINE_OVERRIDES[method].items()
        if hasattr(cfg, key)
    }


def _paper_faithful_overrides(method: str | None = None) -> dict[str, Any]:
    """Knobs injected into method configs under `--protocol paper-faithful` on e3."""
    if method == "dc3":
        return dict(
            train_dataset_size=1000,
            jacobian_mode="loop",
        )
    if method in _PAPER_FAITHFUL_BASELINE_OVERRIDES and method not in _PAPER_FAITHFUL_EVAL_ONLY:
        return dict(_PAPER_FAITHFUL_BASELINE_OVERRIDES[method])
    overrides = dict(
        hidden=200,
        n_layers=2,
        train_dataset_size=1000,
        zeta_zero=True,
        epochs=2000,
        batch_size=200,
        lr=1e-3,
        grad_clip=0.0,
    )
    if method == "pal_loggap":
        overrides["proj_lambda_min"] = 1e-4
    return overrides


def _parse_restart_shard(spec: str | None) -> tuple[int, int] | None:
    """Parse `--restart-shard r/R` into (r, R). None on unset; raise on malformed.

    Format: two non-negative ints separated by `/`, with `0 <= r < R`.
    """
    if spec is None or spec == "":
        return None
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError(f"--restart-shard expects 'r/R', got {spec!r}")
    try:
        r, R = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"--restart-shard expects integers, got {spec!r}") from exc
    if R <= 0 or r < 0 or r >= R:
        raise ValueError(f"--restart-shard requires 0 <= r < R, got r={r} R={R}")
    return r, R


def _build_cfg(method: str, args: argparse.Namespace, seed: int, bench_id: str | None = None):
    if method in _CLASSICAL_METHODS:
        if method == "slsqp":
            cfg = SLSQPConfig(seed=seed, device=args.device)
            if args.multi_start is not None:
                cfg.multi_start = args.multi_start
            if args.max_iter is not None:
                cfg.maxiter = args.max_iter
            if args.tol is not None:
                cfg.ftol = args.tol
            if getattr(args, "restart_shard", None):
                raise ValueError(
                    "--restart-shard is IPOPT-only; SLSQP runs all restarts in one process"
                )
            return cfg
        from pal.baselines.ipopt import IPOPTConfig
        cfg = IPOPTConfig(seed=seed, device=args.device)
        if args.multi_start is not None:
            cfg.multi_start = args.multi_start
        if args.max_iter is not None:
            cfg.max_iter = args.max_iter
        if args.tol is not None:
            cfg.tol = args.tol
        cfg.restart_shard = _parse_restart_shard(getattr(args, "restart_shard", None))
        return cfg

    if getattr(args, "restart_shard", None):
        raise ValueError(
            f"--restart-shard is IPOPT-only (method={method!r} doesn't multistart)"
        )

    effective_batch, override_batch = _resolve_batch_size(args, bench_id)

    explicit_epochs = args.epochs is not None
    epochs_common = args.epochs if explicit_epochs else _DEFAULT_EPOCHS

    common = dict(
        seed=seed,
        epochs=epochs_common,
        batch_size=effective_batch,
        lr=args.lr,
        device=args.device,
    )
    if method in ("pal_loggap", "pal_sqp", "pal_ip"):
        lg_kwargs = dict(
            **common,
            eval_every=args.eval_every,
            eval_samples=args.eval_samples,
            projection_log_every=args.projection_log_every,
        )
        # Per-bench solver hparams apply before CLI overrides; pal_sqp / pal_ip
        # fall back to pal_loggap's so the repair-step arms share hparams.
        if bench_id is not None:
            declared = get_benchmark(bench_id).spec.solver_hparams
            spec_hp = declared.get(method, declared.get("pal_loggap", {}))
            lg_kwargs.update(spec_hp)
        if explicit_epochs:
            lg_kwargs["epochs"] = args.epochs
        if args.proj_method is not None:
            lg_kwargs["proj_method"] = args.proj_method
        if args.max_decades is not None:
            lg_kwargs["max_decades"] = args.max_decades
        if args.loggap_rate is not None:
            lg_kwargs["rate"] = args.loggap_rate
        if args.measure_repair_mem:
            lg_kwargs["measure_repair_mem"] = True
            lg_kwargs["measure_repair_mem_every"] = args.measure_repair_mem_every
        if _is_paper_faithful(args, bench_id):
            lg_kwargs.update(_paper_faithful_overrides("pal_loggap"))
        if method == "pal_sqp":
            return PALSqpConfig(**lg_kwargs)
        if method == "pal_ip":
            return PALIpConfig(**lg_kwargs)
        return PALLogGapConfig(**lg_kwargs)

    def _maybe_inject_repair_mem(cfg):
        if args.measure_repair_mem:
            cfg.measure_repair_mem = True
            cfg.measure_repair_mem_every = args.measure_repair_mem_every
        return cfg

    if method == "alm":
        # e1/* uses the hand-tuned alm_e1.yaml; everything else alm.yaml.
        tier = "alm_e1" if (bench_id and bench_id.startswith("e1/")) else "alm"
        hp = load_hparams(tier)
        if explicit_epochs:
            hp["epochs"] = args.epochs
        if override_batch:
            hp["batch_size"] = effective_batch
        if args.lr != _CLI_DEFAULT_LR:
            hp["lr"] = args.lr
        if args.alm_gamma is not None:
            hp["gamma"] = args.alm_gamma
        if args.alm_alpha is not None:
            hp["alpha"] = args.alm_alpha
        if _is_paper_faithful(args, bench_id):
            hp.update(_paper_faithful_overrides("alm"))
        return _maybe_inject_repair_mem(ALMConfig(**hp, seed=seed, device=args.device))
    if method == "alm_bolton":
        hp = load_hparams("alm_bolton")
        if explicit_epochs:
            hp["epochs"] = args.epochs
        if override_batch:
            hp["batch_size"] = effective_batch
        if args.lr != _CLI_DEFAULT_LR:
            hp["lr"] = args.lr
        if _is_paper_faithful(args, bench_id):
            hp.update(_paper_faithful_overrides("alm_bolton"))
        return _maybe_inject_repair_mem(ALMBoltOnConfig(**hp, seed=seed, device=args.device))
    if method == "enforce_orig":
        return _maybe_inject_repair_mem(EnforceOrigConfig(**common))
    if method == "enforce_v4":
        v4_kwargs = dict(common)
        if _is_paper_faithful(args, bench_id):
            v4_kwargs.update(_paper_faithful_overrides("enforce_v4"))
        epoch_start_hard = getattr(args, "enforce_v4_epoch_start_hard", None)
        if epoch_start_hard is not None:
            v4_kwargs["epoch_start_hard_constrained"] = epoch_start_hard
        micro_batch = getattr(args, "enforce_v4_micro_batch", None)
        if micro_batch is not None:
            v4_kwargs["micro_batch"] = micro_batch
        if getattr(args, "enforce_v4_ift_backward", False):
            v4_kwargs["ift_backward"] = True
        if getattr(args, "enforce_v4_skip_nonfinite_step", False):
            v4_kwargs["skip_nonfinite_step"] = True
        return _maybe_inject_repair_mem(EnforceV4Config(**v4_kwargs))
    if method == "snarenet":
        # SnareNet keeps its authors' config; only --epochs / --batch-size apply.
        cfg = SnareNetConfig(seed=seed, device=args.device)
        if explicit_epochs:
            cfg.epochs = args.epochs
        if override_batch:
            cfg.batch_size = effective_batch
        if _is_paper_faithful(args, bench_id):
            for key, value in _paper_faithful_overrides("snarenet").items():
                setattr(cfg, key, value)
        return _maybe_inject_repair_mem(cfg)
    if method == "fsnet":
        hp = load_hparams("fsnet")
        if explicit_epochs:
            hp["epochs"] = args.epochs
        if override_batch:
            hp["batch_size"] = effective_batch
        if args.lr != _CLI_DEFAULT_LR:
            hp["lr"] = args.lr
        if _is_paper_faithful(args, bench_id):
            hp.update(_paper_faithful_overrides("fsnet"))
        return _maybe_inject_repair_mem(FSNetConfig(**hp, seed=seed, device=args.device))
    if method == "dc3":
        # acopf tier for e3/*, nonconvex otherwise.
        tier = "dc3_acopf" if (bench_id and bench_id.startswith("e3/")) else "dc3"
        hp = load_hparams(tier)
        if explicit_epochs:
            hp["epochs"] = args.epochs
        if override_batch:
            hp["batch_size"] = effective_batch
        if args.lr != _CLI_DEFAULT_LR:
            hp["lr"] = args.lr
        if _is_paper_faithful(args, bench_id):
            hp.update(_paper_faithful_overrides("dc3"))
        return _maybe_inject_repair_mem(DC3Config(**hp, seed=seed, device=args.device))
    raise ValueError(f"no config builder for method '{method}'")


def _cfg_to_dict(cfg) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def _resolve_device(device: str) -> str:
    requested = device
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return f"cuda:{_local_rank()}"
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pal", description="Projection-Adaptive Loss runner")
    sub = p.add_subparsers(dest="verb", required=True)

    run = sub.add_parser("run", help="train a solver + run end-of-run eval")
    run.add_argument(
        "--config", default=None,
        help="path to a run-config YAML (pal/configs/runs/*.yaml). "
             "YAML flags are prepended to argv; explicit CLI flags still override.",
    )
    run.add_argument("--method", required=False, default=None, help="comma-separated; any of pal_loggap|pal_sqp|pal_ip|alm|alm_bolton|dc3|enforce_orig|enforce_v4|fsnet|snarenet|slsqp|ipopt (required unless supplied by --config)")
    run.add_argument(
        "--benchmarks", default=None,
        help="comma-separated benchmark ids; required unless --cost is passed",
    )
    run.add_argument(
        "--cost", default=None,
        help="comma-separated cost tiers (cheap,mid,expensive); expands to all "
             "matching benchmarks. unioned with --benchmarks if both passed.",
    )
    run.add_argument(
        "--target-device", default=None,
        help="comma-separated device tiers (cpu,gpu) for --cost expansion. "
             "filters which benchmarks are picked, not the runtime --device.",
    )
    run.add_argument("--seeds", default="0", help="comma-separated integer seeds")

    run.add_argument("--epochs", type=int, default=_CLI_DEFAULT_EPOCHS)
    run.add_argument("--batch-size", type=int, default=_CLI_DEFAULT_BATCH_SIZE)
    run.add_argument("--lr", type=float, default=_CLI_DEFAULT_LR)
    run.add_argument("--grad-clip", type=float, default=_CLI_DEFAULT_GRAD_CLIP,
                     help="max grad norm (clip_grad_norm_); PAL default 20.0")
    run.add_argument("--device", default="cpu", help="'auto' | 'cpu' | 'cuda' | 'mps'")
    run.add_argument(
        "--precision", default=None, choices=["fp32", "bf16", "fp16", "fp64"],
        help="override `spec.precision` for this run. unset -> the benchmark "
             "spec's own value. fp64 additionally wraps the whole solve+eval "
             "path of each (method, bench, seed) cell in "
             "torch.set_default_dtype(torch.float64) and casts the eval "
             "queries to float64. The effective value lands in config.json.",
    )

    run.add_argument("--multi-start", type=int, default=None,
                     help="classical: restarts per query (zeta-seeded); "
                          "Deb-dominance picks the winner")
    run.add_argument("--max-iter", type=int, default=None,
                     help="classical: SLSQP `maxiter` or IPOPT `max_iter`")
    run.add_argument("--tol", type=float, default=None,
                     help="classical: SLSQP `ftol` or IPOPT `tol`")
    run.add_argument(
        "--restart-shard", type=str, default=None, metavar="r/R",
        help="IPOPT only: run only the r-th of R slices of multistart restarts "
             "for SLURM fan-out. e.g. `--multi-start 20 --restart-shard 0/4` "
             "runs restarts 0,4,8,12,16 in this job. Run-dir gets suffix "
             "`_r{r}_of{R}`. Distinct from `--shard I/N` which slices the "
             "methodxbenchxseed grid. Use the merge-shards helper after all "
             "shards complete to recover the global per-query winner.",
    )

    run.add_argument("--eval-every", type=int, default=100,
                     help="inference eval during training (RNG-consuming)")
    run.add_argument(
        "--eval-samples", type=int, default=64,
        help="periodic inference eval batch size; distinct from --n-eval",
    )
    run.add_argument(
        "--n-eval", type=int, default=None,
        help="number of final-eval queries (default: spec.n_eval_default, 64 for rosenbrock_eq)",
    )
    run.add_argument(
        "--protocol", choices=("synthetic", "paper-faithful"), default="synthetic",
        help="Training/eval protocol mode. 'synthetic' (default): canonical "
             "bench.eval_queries + per-epoch fresh sampling for ALM/PAL. "
             "'paper-faithful' (e3 only): runner replaces eval queries with "
             "bench's pool 'eval' split + zero zeta; for ALM/PAL_loggap, also "
             "injects 200x2 backbone, fixed pool of 1000, zero-zeta training. "
             "Mirrors DC3's paper-canonical setup so PAL/ALM play DC3's game.",
    )
    run.add_argument(
        "--predict-batch-size", type=int, default=None,
        help="PAL only: chunk final-eval prediction/projection per rank to reduce GPU memory",
    )
    run.add_argument(
        "--final-eval-main-only", action="store_true",
        help="when running under DDP, skip distributed final eval and run it on rank 0 only",
    )
    run.add_argument("--projection-log-every", type=int, default=0,
                     help="log a full projection trajectory every N training epochs (0=disabled)")
    run.add_argument(
        "--proj-method", choices=("eigh", "lm_k", "sqp", "ip"), default=None,
        help="PAL only: inner projection solve (default eigh, sqp for pal_sqp, ip for pal_ip).",
    )
    run.add_argument(
        "--max-decades", type=float, default=None,
        help="pal_loggap only: override PALLogGapConfig.max_decades.",
    )
    run.add_argument(
        "--loggap-rate", type=float, default=None,
        help="pal_loggap only: override PALLogGapConfig.rate.",
    )
    run.add_argument(
        "--measure-repair-mem", action="store_true",
        help="record CUDA peak memory around the repair step (off by default).",
    )
    run.add_argument(
        "--measure-repair-mem-every", type=int, default=100,
        help="epoch cadence for --measure-repair-mem (default 100).",
    )
    run.add_argument(
        "--eval-points", default=None,
        help="frozen eval-points JSON that overrides bench.eval_queries.",
    )
    run.add_argument(
        "--alm-gamma", type=float, default=None,
        help="alm only: override ALMConfig.gamma (default 1e-2).",
    )
    run.add_argument(
        "--alm-alpha", type=float, default=None,
        help="alm only: override ALMConfig.alpha (default 0.99).",
    )
    run.add_argument(
        "--enforce-v4-epoch-start-hard", type=int, default=None,
        help="enforce_v4 only: soft warm-up epochs before hard projection (default 0).",
    )
    run.add_argument(
        "--enforce-v4-micro-batch", type=int, default=None,
        help="enforce_v4 only: micro-batch size with gradient accumulation (default full batch).",
    )
    run.add_argument(
        "--enforce-v4-ift-backward", action="store_true",
        help="enforce_v4 only: use the upstream implicit-function backward.",
    )
    run.add_argument(
        "--enforce-v4-skip-nonfinite-step", action="store_true",
        help="enforce_v4 only: skip non-finite steps instead of raising.",
    )
    run.add_argument(
        "--viz-train-every", type=int, default=0,
        help="render bench.visualize_train every N epochs (0 = off).",
    )
    run.add_argument(
        "--viz-final", action="store_true",
        help="render bench.visualize_final once after training.",
    )
    run.add_argument(
        "--viz-n", type=int, default=1,
        help="number of eval queries passed to the visualizers (default 1).",
    )

    run.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="write <run_dir>/checkpoint_latest.pt every N completed epochs "
             "(and always after the final epoch). 0 (default) = disabled. "
             "Only pal_loggap and alm implement checkpointing; ignored for "
             "other methods. Rank 0 writes atomically so a SLURM wall-time "
             "kill mid-write leaves the previous checkpoint intact.",
    )
    run.add_argument(
        "--auto-resume", action="store_true",
        help="deterministic run-dir naming (<bench>__<method>__seed<seed>, no "
             "timestamp/uuid, exist_ok) + resume from checkpoint_latest.pt if "
             "one is found inside. metrics.jsonl is appended, not truncated. "
             "For SLURM jobs that must survive the wall-time boundary; pair "
             "with --checkpoint-every.",
    )
    run.add_argument("--runs-root", default=None, help="override 'runs/' root")
    run.add_argument("--no-final-eval", action="store_true", help="skip end-of-run eval")
    run.add_argument("--wandb", action="store_true")
    run.add_argument("--wandb-project", default="pal")
    run.add_argument("--wandb-entity", default=None)
    run.add_argument(
        "--wandb-group", default=None,
        help="W&B group name for this run. Default: benchmark_id. Useful to "
             "tag all shards from one SLURM array job under a single group.",
    )
    run.add_argument(
        "--shard", default=None, metavar="I/N",
        help="run only the I-th tuple of an N-sized (method x bench x seed) "
             "grid. Used by SLURM array jobs: pass "
             "`--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT` so each "
             "array task picks one tuple. The full grid is "
             "len(methods)*len(benches)*len(seeds); N must equal that.",
    )
    run.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", dest="set_overrides",
        help="generic hparam override applied after typed flags. Repeatable. "
             "Keys must be dataclass fields of the method's config. Values are "
             "cast via the field type (int/float/bool/str). Used by ablation "
             "sweeps to set fine-grained knobs like proj_delta, "
             "proj_lambda_min, rate, detach_j, ablation_name, "
             "scenario_label.",
    )

    ev = sub.add_parser(
        "eval",
        help="re-run inference on an already-trained run (no retraining)",
    )
    ev.add_argument(
        "--run-id", required=True,
        help="run dir name under --runs-root (full or unique prefix)",
    )
    ev.add_argument("--runs-root", default=None, help="override 'runs/' root")
    ev.add_argument("--device", default="cpu", help="'auto' | 'cpu' | 'cuda' | 'mps'")
    ev.add_argument(
        "--n-eval", type=int, default=None,
        help="override n eval queries (default: same n the run was trained with)",
    )
    ev.add_argument(
        "--predict-batch-size", type=int, default=None,
        help="override predict-time chunk size (queries per projector.project "
             "call). default: whatever the run trained with (None -> len(q), "
             "i.e. all queries at once). Use small values (1-8) on heavy "
             "engineering benches whose per-query autograd graph is large.",
    )
    ev.add_argument(
        "--eval-points", default=None,
        help="path to a frozen eval-points JSON; same semantics as on `run`.",
    )
    ev.add_argument(
        "--inference-trajectory-max", type=int, default=200,
        help="cap per-query trajectories logged (rest get aggregate-only)",
    )
    ev.add_argument(
        "--inference-trajectory-downsample", type=int, default=1,
        help="keep every Kth projection iter in each trajectory",
    )
    ev.add_argument(
        "--method-override", default=None,
        help="evaluate the trained checkpoint as a different method (e.g. "
             "--method-override alm_bolton against an alm run, since the two "
             "share training byte-for-byte and only differ in predict-time "
             "projection). Outputs land in a sibling run dir named "
             "'<run_id>__as_<override>/' to keep the original eval intact.",
    )
    ev.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", dest="set_overrides",
        help="generic hparam override applied to the rehydrated config, same "
             "syntax/semantics as on `run` (repeatable; unknown key is a hard "
             "error). Requires --method-override: the resolved values are "
             "written into the sibling run dir's config.json, and mutating the "
             "source run's recorded hparams in place is not allowed.",
    )
    ev.add_argument(
        "--viz-final", action="store_true",
        help="render bench.visualize_final(x) from the rehydrated model. "
             "Local-only. Outputs saved under runs/<id>/final/<name>.{png,pdf}.",
    )
    ev.add_argument(
        "--viz-n", type=int, default=1,
        help="size of the fixed grid (first N zetas of eval_queries) "
             "passed to visualize_final. Default 1.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "sweep":
        from pal.runner.sweep import main as sweep_main
        return sweep_main(raw[1:])
    # YAML flags are prepended so explicit CLI flags win (last occurrence wins).
    if "--config" in raw:
        from pal.runner.run_config import extract_config_from_argv
        cleaned, yaml_argv = extract_config_from_argv(raw)
        if cleaned and cleaned[0] in {"run", "eval"}:
            raw = [cleaned[0]] + yaml_argv + cleaned[1:]
        else:
            raw = yaml_argv + cleaned
    args = _parse_args(raw)
    if args.verb == "eval":
        return _main_eval(args)
    if args.verb != "run":
        raise SystemExit(f"unknown verb '{args.verb}'")

    args.device = _resolve_device(args.device)
    _maybe_init_distributed(args.device)

    if not args.method:
        raise SystemExit("--method is required (pass via CLI or `--config <yaml>`)")
    methods = [m.strip() for m in args.method.split(",") if m.strip()]
    benches = _resolve_benchmarks(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    runs_root = Path(args.runs_root) if args.runs_root else None

    if _dist_is_active():
        _DDP_SUPPORTED = {"pal_loggap", "pal_sqp", "pal_ip", "alm", "alm_bolton"}
        unsupported = [m for m in methods if m not in _DDP_SUPPORTED]
        if unsupported:
            raise SystemExit(
                "[pal run] distributed execution supports "
                f"{sorted(_DDP_SUPPORTED)}. "
                f"Unsupported methods in this launch: {unsupported}"
            )

    grid: list[tuple[str, str, int]] = [
        (m, b, s) for m in methods for b in benches for s in seeds
    ]
    if args.shard is not None:
        grid = select_shard(grid, args.shard)

    ok, fail = 0, 0
    for method, bench_id, seed in grid:
        if method not in _SUPPORTED_METHODS:
            print(
                f"[pal run] method '{method}' not supported; "
                f"skipping (available: {sorted(_SUPPORTED_METHODS)})",
                file=sys.stderr,
            )
            fail += 1
            continue
        try:
            _run_one(method, bench_id, seed, args, runs_root)
            ok += 1
        except Exception:
            traceback.print_exc()
            fail += 1
    if _dist_is_main():
        print(f"[pal run] done. ok={ok} fail={fail}")
    # Always tear down, even if a rank crashed mid-collective.
    _safe_barrier("teardown")
    if _dist_is_active():
        try:
            dist.destroy_process_group()
        except Exception as e:  # noqa: BLE001
            print(f"[pal run] destroy_process_group failed: {e}", file=sys.stderr)
    return 0 if fail == 0 else 1


def _main_eval(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root) if args.runs_root else Path("runs")
    args.device = _resolve_device(args.device)
    try:
        _eval_one(args.run_id, args, runs_root)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def _run_one(
    method: str,
    bench_id: str,
    seed: int,
    args: argparse.Namespace,
    runs_root: Path | None,
) -> None:
    """Run one (method, bench, seed) cell under its effective precision.

    Keeps the whole solve+eval path inside one float64 default-dtype block
    for fp64.

    Args:
        method: Solver key.
        bench_id: Benchmark id.
        seed: Run seed.
        args: Parsed `run` namespace.
        runs_root: Root dir for the run dir, or None for the default.

    Returns:
        None.
    """
    raw_bench = get_benchmark(bench_id, device=args.device)
    precision = _resolve_precision(args, raw_bench.spec)
    if precision != raw_bench.spec.precision:
        # Baselines read spec.precision to pick their dtype; override it per run.
        object.__setattr__(raw_bench.spec, "precision", precision)
    native_dtype = torch.get_default_dtype()
    with _default_dtype(precision):
        _run_one_impl(
            method, bench_id, seed, args, runs_root, raw_bench, precision, native_dtype,
        )


def _run_one_impl(
    method: str,
    bench_id: str,
    seed: int,
    args: argparse.Namespace,
    runs_root: Path | None,
    raw_bench,
    precision: str,
    native_dtype: torch.dtype,
) -> None:
    bench = BenchProbe(raw_bench)
    cfg = _build_cfg(method, args, seed, bench_id=bench_id)
    cfg = _apply_set_overrides(cfg, getattr(args, "set_overrides", []) or [])
    solver = _build_solver(method, cfg, bench_id=bench_id)

    restart_shard = _parse_restart_shard(getattr(args, "restart_shard", None))
    shard_name_suffix = f"__r{restart_shard[0]}_of{restart_shard[1]}" if restart_shard else ""

    auto_resume = bool(getattr(args, "auto_resume", False))
    checkpoint_every = int(getattr(args, "checkpoint_every", 0))
    supports_checkpoint = method in ("pal_loggap", "alm")

    run_dir_str = None
    if _dist_is_main():
        run_dir_str = str(
            create_run_dir(
                method=method,
                bench_id=bench_id,
                seed=seed,
                runs_root=runs_root,
                restart_shard=restart_shard,
                deterministic=auto_resume,
            )
        )
    run_dir = Path(_broadcast_string(run_dir_str))
    is_main = _dist_is_main()
    sinks = [JSONLLogger(run_dir)] if is_main else []
    if args.wandb and is_main:
        # W&B init failures must not kill the run; JSONL always runs.
        try:
            from pal.tracking.wandb_logger import WandBLogger
            wandb_group = getattr(args, "wandb_group", None) or bench_id
            sinks.append(
                WandBLogger(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    group=wandb_group,
                    name=f"{method}__{bench_id}__seed{seed}{shard_name_suffix}",
                    tags=[method, bench_id],
                    config=_cfg_to_dict(cfg),
                )
            )
        except Exception as exc:
            print(
                f"  [wandb] init failed, continuing with JSONL-only: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    logger = CompositeLogger(sinks) if is_main else _NullLogger()

    paper_faithful = _is_paper_faithful(args, bench_id)
    # Sample at the native dtype so fp64 runs get the same eval points, then cast.
    with _use_dtype(native_dtype):
        queries = _build_eval_queries(
            bench, args.n_eval, seed,
            paper_faithful=paper_faithful,
            eval_points=getattr(args, "eval_points", None),
        )
    fingerprint = query_sha256(queries)
    queries = _cast_queries(queries, precision)

    pal_root = Path(__file__).resolve().parents[2]
    config_blob = {
        "schema_version": 1,
        "pal_git_sha": _git_sha(pal_root),
        "pal_git_branch": _git_branch(pal_root),
        "pal_git_diff": _git_diff(pal_root),
        "torch_version": torch.__version__,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": cfg.device,
        "precision": precision,
        "wall_start": datetime.now(UTC).isoformat(),
        "seed": seed,
        "method": method,
        "benchmark_id": bench_id,
        "benchmark_spec": bench.spec.to_json_dict(),
        "protocol": getattr(args, "protocol", "synthetic"),
        "eval_queries_fingerprint": fingerprint,
        "eval_queries_zeta_shape": list(queries.zeta.shape),
        "eval_queries_conditions_shape": list(queries.conditions.shape),
        "hparams": _cfg_to_dict(cfg),
        "n_eval_effective": len(queries),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "shard_spec": getattr(args, "shard", None),
    }
    if is_main:
        logger.log_config(**config_blob)
        print(
            f"[pal run] {method} / {bench_id} / seed={seed} -> {run_dir.name} "
            f"(device={cfg.device}, n_eval={len(queries)}, fingerprint={fingerprint[:17]}...)"
        )

    viz_callback = _make_viz_train_callback(
        bench=bench, queries=queries, args=args, logger=logger, device=cfg.device,
    )
    if not is_main:
        viz_callback = None

    # All ranks load the same checkpoint before training so DDP starts identical.
    resume_state = None
    if auto_resume and supports_checkpoint:
        ckpt_path = run_dir / "checkpoint_latest.pt"
        if ckpt_path.exists():
            resume_state = torch.load(
                ckpt_path, map_location=cfg.device, weights_only=False
            )
            if is_main:
                print(
                    f"  [resume] {ckpt_path.name} @ epoch "
                    f"{resume_state.get('epoch')} -> continuing to {cfg.epochs}"
                )

    train_kwargs: dict[str, Any] = {"on_epoch_end": viz_callback}
    if supports_checkpoint:
        train_kwargs["checkpoint_every"] = checkpoint_every
        train_kwargs["checkpoint_dir"] = run_dir
        train_kwargs["resume_state"] = resume_state

    try:
        bench.set_phase("train")
        train_result = solver.train(
            bench, seed=seed, logger=logger, **train_kwargs,
        )
        if is_main and train_result.model_state is not None:
            torch.save(train_result.model_state, run_dir / "model.pt")
        if args.no_final_eval:
            if is_main:
                logger.log_final(**bench.flat_snapshot())
                logger.finish(status="ok")
                print("  [skip] --no-final-eval")
            _safe_barrier("no-final-eval-sync")
            return
        bench.set_phase("final_eval")
        # Free training-time cache before eval to avoid fragmentation OOMs.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        main_only_final_eval = args.final_eval_main_only and _dist_is_active()
        if main_only_final_eval and not is_main:
            _safe_barrier("main-only-eval-wait")
            return
        eval_result: EvalResult = run_final_eval(
            bench=bench,
            solver=solver,
            train_result=train_result,
            queries=queries,
            logger=logger,
            distributed=not main_only_final_eval,
        )
        if is_main and args.viz_final and train_result.final_x_on_eval is not None:
            _render_viz_final(
                bench=bench,
                x_on_eval=train_result.final_x_on_eval,
                queries=queries,
                run_dir=run_dir,
                viz_n=args.viz_n,
                logger=logger,
            )
        final_payload = {
            "obj_mean_raw": eval_result.obj_mean_raw,
            "obj_mean_post": eval_result.obj_mean_post,
            "viol_max_raw": eval_result.viol_max_raw,
            "viol_max_post": eval_result.viol_max_post,
            "feasibility_raw": eval_result.feasibility_raw,
            "feasibility_post": eval_result.feasibility_post,
            "n_queries": eval_result.n_queries,
            "n_restarts": eval_result.n_restarts,
            "train_wall_time_s": eval_result.train_wall_time_s,
            "predict_wall_time_s": eval_result.predict_wall_time_s,
            "tolerance": eval_result.tolerance,
            "per_constraint_viol_max_raw": eval_result.raw.per_constraint_viol_max,
            "per_constraint_viol_max_post": eval_result.post.per_constraint_viol_max,
            "inf_iters_median": eval_result.inf_iters_median,
            "inf_iters_p90": eval_result.inf_iters_p90,
            "inf_iters_max": eval_result.inf_iters_max,
            "inf_iters_n_converged": eval_result.inf_iters_n_converged,
            "inf_iters_max_allowed": eval_result.inf_iters_max_allowed,
        }
        final_payload.update(bench.flat_snapshot())
        # Per-query restart records from multistart baselines.
        per_query_extras = train_result.extras.get("per_query")
        if per_query_extras is not None:
            final_payload["per_query"] = per_query_extras
        if is_main:
            _write_eval_rows_parquet(
                run_dir,
                eval_result.eval_rows,
                method=method,
                bench_id=bench_id,
                seed=seed,
                tolerance=eval_result.tolerance,
                eval_queries_fingerprint=fingerprint,
                source="run_final_eval",
            )
        if is_main:
            logger.log_final(**final_payload)
            print(
                f"  [ok] obj(raw)={eval_result.obj_mean_raw:+.4e} "
                f"obj(post)={eval_result.obj_mean_post:+.4e} "
                f"feas(post)={eval_result.feasibility_post:.2f}"
            )
            logger.finish(status="ok")
        _safe_barrier("post-eval-sync")
    except Exception as err:
        # Record a failed final.json so sweeps skip this row on retry.
        try:
            fail_payload = {"status": "failed", "error": str(err)}
            try:
                fail_payload.update(bench.flat_snapshot())
            except Exception:
                pass
            logger.log_final(**fail_payload)
        except Exception as log_err:
            print(
                f"  [warn] also failed to write failed-run final.json: "
                f"{type(log_err).__name__}: {log_err}",
                file=sys.stderr,
            )
        logger.finish(status="failed", error=str(err))
        raise


def _resolve_benchmarks(args: argparse.Namespace) -> list[str]:
    """Combine --benchmarks (explicit) with --cost / --target-device expansion.

    Both filters union, passing `--benchmarks rosenbrock_eq --cost expensive` runs rosenbrock_eq
    *and* every expensive bench. At least one source must be specified.
    """
    explicit: list[str] = []
    if args.benchmarks:
        explicit = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    expanded: list[str] = []
    if args.cost or args.target_device:
        costs = (
            [c.strip() for c in args.cost.split(",") if c.strip()]
            if args.cost else None
        )
        devices = (
            [d.strip() for d in args.target_device.split(",") if d.strip()]
            if args.target_device else None
        )
        expanded = list_benchmarks_filtered(costs=costs, devices=devices)
        if not expanded:
            raise SystemExit(
                f"[pal run] --cost={args.cost} --target-device={args.target_device} "
                f"matched no benchmarks."
            )
    if not explicit and not expanded:
        raise SystemExit(
            "[pal run] no benchmarks selected. pass --benchmarks <ids> or "
            "--cost <tier> (or both)."
        )
    merged = list(explicit)
    for b in expanded:
        if b not in merged:
            merged.append(b)
    return merged


def _resolve_run_dir(run_id: str, runs_root: Path) -> Path:
    """Resolve a (possibly partial) run id to a unique run dir under `runs_root`.

    Accepts a full directory name or any unique prefix. Raises if no match or
    multiple matches.
    """
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")
    direct = runs_root / run_id
    if direct.is_dir():
        return direct
    matches = sorted(d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith(run_id))
    if not matches:
        raise FileNotFoundError(
            f"no run dir under {runs_root} matches '{run_id}'"
        )
    if len(matches) > 1:
        sample = ", ".join(m.name for m in matches[:3])
        raise ValueError(
            f"ambiguous run id '{run_id}' under {runs_root} "
            f"({len(matches)} matches; first: {sample}). pass a longer prefix."
        )
    return matches[0]


def _build_cfg_from_hparams(method: str, hparams: dict[str, Any]):
    """Reconstruct a typed config from `config.json[hparams]`.

    Drops keys the dataclass doesn't recognize (in case the schema evolved
    between train-time and eval-time).
    """
    classes: dict[str, Any] = {
        "pal_loggap": PALLogGapConfig,
        "pal_sqp": PALSqpConfig,
        "pal_ip": PALIpConfig,
        "alm": ALMConfig,
        "alm_bolton": ALMBoltOnConfig,
        "dc3": DC3Config,
        "enforce_orig": EnforceOrigConfig,
        "enforce_v4": EnforceV4Config,
        "fsnet": FSNetConfig,
        "snarenet": SnareNetConfig,
        "slsqp": SLSQPConfig,
    }
    if method == "ipopt":
        from pal.baselines.ipopt import IPOPTConfig
        classes["ipopt"] = IPOPTConfig
    cls = classes[method]
    known = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in hparams.items() if k in known})


def _eval_one(
    run_id: str,
    args: argparse.Namespace,
    runs_root: Path,
) -> None:
    src_run_dir = _resolve_run_dir(run_id, runs_root)
    config_path = src_run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json under {src_run_dir}")
    config = json.loads(config_path.read_text())

    trained_method = config["method"]
    bench_id = config["benchmark_id"]
    seed = int(config["seed"])
    if trained_method not in _SUPPORTED_METHODS:
        raise ValueError(f"unknown method '{trained_method}' recorded in {config_path}")

    set_overrides = list(getattr(args, "set_overrides", None) or [])
    override_method = getattr(args, "method_override", None)
    if override_method is not None and override_method == trained_method:
        override_method = None  # no-op
    if set_overrides and override_method is None:
        # Writing --set values into the source run dir would misreport training.
        raise ValueError(
            "`pal eval --set` requires --method-override. Without it the eval "
            f"writes back into the source run dir ({src_run_dir.name}), whose "
            "config.json records the training hparams and must not be "
            "rewritten with eval-time overrides. Pass --method-override to get "
            "a sibling run dir that carries its own provenance."
        )
    sibling_config_path: Path | None = None
    if override_method is not None:
        if override_method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"unknown --method-override '{override_method}'. "
                f"supported: {sorted(_SUPPORTED_METHODS)}"
            )
        if override_method in _CLASSICAL_METHODS or trained_method in _CLASSICAL_METHODS:
            raise ValueError(
                f"--method-override is for learned methods only; got "
                f"trained={trained_method} override={override_method}. "
                f"Classical methods (ipopt, slsqp) re-solve at predict and "
                f"share no checkpoint."
            )
        sibling_id = f"{src_run_dir.name}__as_{override_method}"
        run_dir = src_run_dir.parent / sibling_id
        run_dir.mkdir(exist_ok=True)
        if not (src_run_dir / "final.json").exists():
            raise RuntimeError(
                f"refusing --method-override against {src_run_dir.name}: "
                f"missing final.json (training likely crashed before completion). "
                f"Override eval would record train_wall_time_s=0.0 and corrupt "
                f"the cost table."
            )
        sibling_model = run_dir / "model.pt"
        if not sibling_model.exists():
            try:
                os.link(src_run_dir / "model.pt", sibling_model)
            except (OSError, NotImplementedError):
                import shutil as _shutil
                _shutil.copy2(src_run_dir / "model.pt", sibling_model)
        new_config = dict(config)
        new_config["method"] = override_method
        new_config["override_source_run_id"] = src_run_dir.name
        new_config["override_source_method"] = trained_method
        sibling_config_path = run_dir / "config.json"
        sibling_config_path.write_text(json.dumps(new_config, indent=2))
        method = override_method
    else:
        run_dir = src_run_dir
        method = trained_method

    # Re-evaluating one IPOPT restart shard would overwrite merged metrics.
    hparams = config.get("hparams") or {}
    if hparams.get("restart_shard") is not None:
        raise ValueError(
            f"refusing to eval shard run {run_dir.name}: "
            f"hparams.restart_shard={hparams['restart_shard']!r}. "
            "Re-eval would only touch this shard's slice and overwrite the "
            "merged metrics that aggregate_engineering.py produces from the full "
            "shard pool. To re-evaluate the global pool, re-run training across "
            "all sibling shards and re-aggregate."
        )

    cfg = _build_cfg_from_hparams(method, hparams)
    cfg.device = args.device
    forced: dict[str, Any] = {}
    if (
        bench_id.startswith("e3/")
        and config.get("protocol", "synthetic") == "paper-faithful"
    ):
        forced = _paper_faithful_eval_overrides(method, cfg)
    applied_overrides = _resolve_eval_cfg_overrides(cfg, set_overrides, forced)
    if sibling_config_path is not None:
        provenance = json.loads(sibling_config_path.read_text())
        provenance["hparams"] = {**hparams, **applied_overrides}
        provenance["eval_set_overrides"] = list(set_overrides)
        sibling_config_path.write_text(json.dumps(provenance, indent=2))
    if getattr(args, "predict_batch_size", None) is not None and hasattr(cfg, "predict_batch_size"):
        cfg.predict_batch_size = args.predict_batch_size
    solver = _build_solver(method, cfg, bench_id=bench_id)

    if method in _CLASSICAL_METHODS:
        train_result = TrainResult(
            solver_name=method,
            train_wall_time_s=0.0,
            n_restarts=1,
        )
    else:
        model_path = run_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"missing model.pt under {run_dir}. `pal eval` needs the trained "
                f"weights; re-run training to write them."
            )
        model_state = torch.load(model_path, map_location=args.device, weights_only=True)
        train_wall_src = src_run_dir / "final.json"
        train_result = TrainResult(
            solver_name=method,
            train_wall_time_s=float(
                (json.loads(train_wall_src.read_text()).get("train_wall_time_s") or 0.0)
                if train_wall_src.exists() else 0.0
            ),
            n_restarts=1,
            model_state=model_state,
        )

    raw_bench = get_benchmark(bench_id, device=args.device)
    bench = BenchProbe(raw_bench)
    paper_faithful = (
        config.get("protocol", "synthetic") == "paper-faithful"
        and bench_id.startswith("e3/")
    )
    queries = _build_eval_queries(
        bench, args.n_eval, seed,
        paper_faithful=paper_faithful,
        eval_points=getattr(args, "eval_points", None),
    )
    fingerprint = query_sha256(queries)
    recorded_fp = config.get("eval_queries_fingerprint")
    if args.n_eval is None and recorded_fp and recorded_fp != fingerprint:
        raise RuntimeError(
            f"eval_queries fingerprint mismatch for {run_dir.name}.\n"
            f"  config.json: {recorded_fp}\n"
            f"  current:     {fingerprint}\n"
            f"  bench.eval_queries() has drifted since the original run. "
            f"either revert the change or pass --n-eval to acknowledge."
        )

    sinks = [JSONLLogger(run_dir)]
    logger = CompositeLogger(sinks)

    print(
        f"[pal eval] {method} / {bench_id} / seed={seed} -> {run_dir.name} "
        f"(device={cfg.device}, n_eval={len(queries)}, n_traj_max={args.inference_trajectory_max})"
    )

    bench.set_phase("predict")
    eval_result: EvalResult = run_inference(
        bench=bench,
        solver=solver,
        train_result=train_result,
        queries=queries,
        logger=logger,
        inference_trajectory_max=args.inference_trajectory_max,
        inference_trajectory_downsample=args.inference_trajectory_downsample,
    )
    predict_payload = {
        "obj_mean_raw": eval_result.obj_mean_raw,
        "obj_mean_post": eval_result.obj_mean_post,
        "viol_max_raw": eval_result.viol_max_raw,
        "viol_max_post": eval_result.viol_max_post,
        "feasibility_raw": eval_result.feasibility_raw,
        "feasibility_post": eval_result.feasibility_post,
        "n_queries": eval_result.n_queries,
        "n_restarts": eval_result.n_restarts,
        "train_wall_time_s": eval_result.train_wall_time_s,
        "predict_wall_time_s": eval_result.predict_wall_time_s,
        "tolerance": eval_result.tolerance,
        "per_constraint_viol_max_raw": eval_result.raw.per_constraint_viol_max,
        "per_constraint_viol_max_post": eval_result.post.per_constraint_viol_max,
        "inf_iters_median": eval_result.inf_iters_median,
        "inf_iters_p90": eval_result.inf_iters_p90,
        "inf_iters_max": eval_result.inf_iters_max,
        "inf_iters_n_converged": eval_result.inf_iters_n_converged,
        "inf_iters_max_allowed": eval_result.inf_iters_max_allowed,
    }
    predict_payload.update(bench.flat_snapshot())
    _write_eval_rows_parquet(
        run_dir,
        eval_result.eval_rows,
        method=method,
        bench_id=bench_id,
        seed=seed,
        tolerance=eval_result.tolerance,
        eval_queries_fingerprint=fingerprint,
        source="pal_eval",
    )
    logger.log_final(**predict_payload)
    print(
        f"  [ok] obj(raw)={eval_result.obj_mean_raw:+.4e} "
        f"obj(post)={eval_result.obj_mean_post:+.4e} "
        f"feas(post)={eval_result.feasibility_post:.2f}"
    )
    if args.viz_final:
        outputs = solver.predict(bench, queries, train_result)
        _render_viz_final(
            bench=bench,
            x_on_eval=outputs.post.detach().cpu(),
            queries=queries,
            run_dir=run_dir,
            viz_n=args.viz_n,
            logger=logger,
        )
    logger.finish(status="ok")


def _make_viz_train_callback(
    bench: Any,
    queries: Any,
    args: argparse.Namespace,
    logger: Any,
    device: str,
):
    """Build the `on_epoch_end(epoch, model)` closure that solvers invoke.

    No-op when `--viz-train-every 0`. Otherwise runs a forward pass on the
    first `--viz-n` eval queries every N epochs, calls
    `bench.visualize_train`, and fans the result to the logger (JSONL writes
    PNG+PDF, W&B logs the Figure).
    """
    every = int(getattr(args, "viz_train_every", 0))
    if every <= 0:
        return None
    viz_n = max(1, int(args.viz_n))
    zeta = queries.zeta[:viz_n].to(device)
    conds_full = queries.conditions[:viz_n].to(device) if bench.spec.condition_dim > 0 else None

    def _callback(epoch: int, model: Any) -> None:
        if epoch % every != 0:
            return
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                x = model(zeta, conds_full if conds_full is not None else torch.empty(viz_n, 0, device=device))
            cond_arg = conds_full[0] if conds_full is not None else None
            try:
                fig = bench.visualize_train(x.detach().cpu(), cond_arg)
            except Exception as exc:
                print(
                    f"  [viz_train] epoch {epoch} skipped: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return
            if fig is not None:
                logger.log_artifact(epoch, "viz_train", fig)
        finally:
            if was_training:
                model.train()

    return _callback


def _render_viz_final(
    bench: Any,
    x_on_eval: Any,
    queries: Any,
    run_dir: Path,
    viz_n: int,
    logger: Any | None = None,
) -> None:
    """Call `visualize_final` on the first `viz_n` eval designs, save locally,
    and optionally fan each figure/image to `logger` (e.g. W&B)."""
    n = max(1, int(viz_n))
    x = x_on_eval[:n]
    conds = queries.conditions[0] if bench.spec.condition_dim > 0 else None
    conds_batch = queries.conditions[:n] if bench.spec.condition_dim > 0 else None
    # Prime the bench's cached forward so visualize_final sees this x.
    try:
        bench.forward(x, conds_batch)
    except Exception as exc:
        print(
            f"  [viz_final] prime forward failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    try:
        payload = bench.visualize_final(x, conds)
    except Exception as exc:
        print(
            f"  [viz_final] skipped: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return
    if payload is None:
        print(f"  [viz_final] {bench.spec.id}: returned None; nothing to save")
        return
    written = save_local_artifacts(run_dir, payload, subdir="final")
    if written:
        print(f"  [viz_final] wrote {len(written)} file(s) to {run_dir / 'final'}/")
    if logger is not None:
        _fan_viz_final_to_logger(logger, payload)


def _fan_viz_final_to_logger(logger: Any, payload: Any) -> None:
    """Fan hero/final visualizations to the logger's artifact channel.

    Uses `step=0` as a sentinel for end-of-run.
    """
    items: dict[str, Any]
    if isinstance(payload, dict):
        items = payload
    else:
        items = {"final": payload}
    for name, value in items.items():
        if value is None:
            continue
        try:
            logger.log_artifact(0, f"viz_final/{name}", value)
        except Exception as exc:
            print(
                f"  [viz_final] logger fan-out skipped for '{name}': "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
