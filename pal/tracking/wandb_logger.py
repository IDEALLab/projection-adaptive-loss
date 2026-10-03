"""Weights & Biases Logger sink.

Imports `wandb` lazily so the default JSONL-only path works without it.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Literal


class WandBLogger:
    """Fan-out events into a W&B run.

    `log_artifact` routes figures to `wandb.Image`, tensors to
    `wandb.Histogram`, and everything else to JSON.
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        group: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=name,
            tags=tags or [],
            config=config or {},
        )

    def log_config(self, **cfg: Any) -> None:
        self._run.config.update(_to_wandb(cfg), allow_val_change=True)

    def log_step(self, step: int, **scalars: float) -> None:
        self._wandb.log(dict(scalars), step=int(step))

    def log_projection_trajectory(
        self, step: int, phase: str, trajectory: list[Any]
    ) -> None:
        if not trajectory:
            return
        iters = [getattr(s, "iter", i) for i, s in enumerate(trajectory)]
        obj = [float(getattr(s, "obj", 0.0)) for s in trajectory]
        constraints = [list(getattr(s, "constraints", [])) for s in trajectory]
        n_constraints = len(constraints[0]) if constraints else 0
        ys = [obj] + [[c[k] for c in constraints] for k in range(n_constraints)]
        keys = ["obj"] + [f"c{k}" for k in range(n_constraints)]
        plot = self._wandb.plot.line_series(
            xs=iters,
            ys=ys,
            keys=keys,
            title=f"projection_trajectory/{phase}/step_{int(step)}",
            xname="inner_iter",
        )
        self._wandb.log({f"projection_trajectory/{phase}": plot}, step=int(step))

    def log_artifact(self, step: int, name: str, payload: Any) -> None:
        fig_cls = _optional_matplotlib_figure()
        if fig_cls is not None and isinstance(payload, fig_cls):
            self._wandb.log({name: self._wandb.Image(payload)}, step=int(step))
            return

        if _is_image_ndarray(payload):
            # [H, W, 3|4] uint8, e.g. PyVista screenshots.
            self._wandb.log({name: self._wandb.Image(payload)}, step=int(step))
            return

        tensor_cls = _optional_torch_tensor()
        if tensor_cls is not None and isinstance(payload, tensor_cls):
            flat = payload.detach().reshape(-1).cpu().float().numpy()
            self._wandb.log(
                {name: self._wandb.Histogram(flat)}, step=int(step)
            )
            return

        self._wandb.log({name: _to_wandb(payload)}, step=int(step))

    def log_final(self, **final: Any) -> None:
        self._wandb.summary.update(_to_wandb(final))

    def finish(
        self,
        status: Literal["ok", "failed"] = "ok",
        error: str | None = None,
    ) -> None:
        self._wandb.summary["status"] = status
        if error is not None:
            self._wandb.summary["error"] = error
        exit_code = 0 if status == "ok" else 1
        self._wandb.finish(exit_code=exit_code)


def _to_wandb(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_wandb(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_wandb(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_wandb(v) for v in obj]
    return obj


def _optional_matplotlib_figure():
    try:
        from matplotlib.figure import Figure

        return Figure
    except ImportError:
        return None


def _optional_torch_tensor():
    try:
        import torch

        return torch.Tensor
    except ImportError:
        return None


def _is_image_ndarray(obj: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    if not isinstance(obj, np.ndarray):
        return False
    return obj.ndim == 3 and obj.shape[-1] in (3, 4)
