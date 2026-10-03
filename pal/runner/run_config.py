"""Run-config YAML loader.

Expands a run-config YAML (`methods`, `benchmark(s)`, `seeds`, scalar and
boolean CLI keys, and an `env:` mapping) into a `pal run` argv list that the
CLI prepends, so explicit CLI flags still win.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Scalar keys -> CLI flag.
_SCALAR_FLAGS: dict[str, str] = {
    "epochs": "--epochs",
    "batch_size": "--batch-size",
    "lr": "--lr",
    "grad_clip": "--grad-clip",
    "device": "--device",
    "precision": "--precision",
    "multi_start": "--multi-start",
    "max_iter": "--max-iter",
    "tol": "--tol",
    "eval_every": "--eval-every",
    "eval_samples": "--eval-samples",
    "n_eval": "--n-eval",
    "protocol": "--protocol",
    "predict_batch_size": "--predict-batch-size",
    "eval_points": "--eval-points",
    "measure_repair_mem_every": "--measure-repair-mem-every",
    "runs_root": "--runs-root",
    "wandb_project": "--wandb-project",
    "wandb_entity": "--wandb-entity",
    "run_id": "--run-id",
    "alm_alpha": "--alm-alpha",
    "alm_gamma": "--alm-gamma",
    "loggap_rate": "--loggap-rate",
    "max_decades": "--max-decades",
    "proj_method": "--proj-method",
    "viz_train_every": "--viz-train-every",
    "viz_n": "--viz-n",
    "projection_log_every": "--projection-log-every",
    "checkpoint_every": "--checkpoint-every",
}

# Boolean keys -> store_true flag, added only when truthy.
_BOOL_FLAGS: dict[str, str] = {
    "measure_repair_mem": "--measure-repair-mem",
    "wandb": "--wandb",
    "final_eval_main_only": "--final-eval-main-only",
    "viz_final": "--viz-final",
    "no_final_eval": "--no-final-eval",
    "auto_resume": "--auto-resume",
}


def _coerce_list(val: Any) -> list[str]:
    if isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()]
    if isinstance(val, (list, tuple)):
        return [str(v) for v in val]
    return [str(val)]


def expand_config_to_argv(path: Path | str) -> list[str]:
    """Load YAML at `path`, return the equivalent `pal run` flag list.

    Side effect: any keys under `env:` are written to os.environ immediately,
    so module-level env reads (e.g. e1 cruise envelope) see them.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"run config not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"run config {path} must be a YAML mapping at top level")

    argv: list[str] = []

    if "methods" in cfg:
        argv += ["--method", ",".join(_coerce_list(cfg["methods"]))]
    if "benchmark" in cfg and "benchmarks" in cfg:
        raise ValueError(
            f"run config {path}: set either `benchmark` (single) or "
            f"`benchmarks` (list/comma), not both"
        )
    if "benchmark" in cfg:
        argv += ["--benchmarks", str(cfg["benchmark"])]
    elif "benchmarks" in cfg:
        argv += ["--benchmarks", ",".join(_coerce_list(cfg["benchmarks"]))]
    if "seeds" in cfg:
        argv += ["--seeds", ",".join(_coerce_list(cfg["seeds"]))]

    for key, flag in _SCALAR_FLAGS.items():
        if key in cfg and cfg[key] is not None:
            argv += [flag, str(cfg[key])]

    for key, flag in _BOOL_FLAGS.items():
        if cfg.get(key):
            argv.append(flag)

    env = cfg.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError(f"run config {path}: `env:` must be a mapping")
    for k, v in env.items():
        os.environ[str(k)] = str(v)

    # Fail on unknown top-level keys so typos are not silently ignored.
    known = (
        {"methods", "benchmark", "benchmarks", "seeds", "env"}
        | set(_SCALAR_FLAGS)
        | set(_BOOL_FLAGS)
    )
    extras = set(cfg) - known
    if extras:
        raise ValueError(
            f"run config {path}: unknown keys {sorted(extras)}; "
            f"known keys = {sorted(known)}"
        )

    return argv


def extract_config_from_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Pop `--config <path>` from argv. Returns (cleaned_argv, yaml_argv).

    `yaml_argv` is empty if no --config flag was present.
    """
    if "--config" not in argv:
        return argv, []
    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        raise SystemExit("--config requires a path argument")
    path = argv[idx + 1]
    cleaned = argv[:idx] + argv[idx + 2:]
    return cleaned, expand_config_to_argv(path)
