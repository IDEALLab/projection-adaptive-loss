"""Load authors'-default hparams for a named baseline method."""

from __future__ import annotations

from pathlib import Path

import yaml

_HPARAMS_DIR = Path(__file__).parent


def load_hparams(method: str) -> dict:
    """Return the authors'-default hparam dict for `method`.

    Splat into the method's Config dataclass:
        FSNetConfig(**load_hparams("fsnet"), seed=0, device="cpu")
    """
    path = _HPARAMS_DIR / f"{method}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No hparams file for method {method!r}: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not parse to a dict; got {type(data).__name__}")
    return data


def list_methods() -> list[str]:
    """Return the names of methods that have a hparams YAML file on disk."""
    return sorted(p.stem for p in _HPARAMS_DIR.glob("*.yaml"))
