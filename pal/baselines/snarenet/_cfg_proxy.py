"""Hydra-shaped cfg proxy for vendored SnareNet code (attribute and dict access)."""

from __future__ import annotations

from typing import Any


class _DictCfg(dict):
    """dict with attribute access; nested dicts auto-wrap on read."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return _DictCfg(v) if isinstance(v, dict) else v


def _make_cfg_proxy(config: Any, prob_type: str) -> _DictCfg:
    """Build the Hydra-shaped tree upstream expects from a `SnareNetConfig`."""
    model = {
        "hidden_size": config.hidden_size,
        # Read by the vendored MLP constructor (upstream hardcodes 2).
        "num_hidden_layers": config.num_hidden_layers,
        "dropout": config.dropout,
        "use_batch_norm": config.batchnorm_dropout,
        "newton_maxiter": config.newton_maxiter,
        "rtol": config.rtol,
        "lambd": config.lambd,
        "adaptive_relaxation": config.adaptive_relaxation,
        "decay_epochs": config.decay_epochs,
        "decay_schedule": config.decay_schedule,
        "trust_region": config.trust_region,
        "is_cg": config.is_cg,
        "cg_maxiter": config.cg_maxiter,
    }
    dataset = {"prob_type": prob_type}
    return _DictCfg({
        "model": model,
        "dataset": dataset,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "soft_epochs": config.soft_epochs,
        "soft_weight": config.soft_weight,
        "seed": config.seed,
    })
