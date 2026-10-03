"""JSONL file-based Logger sink."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


def save_local_artifacts(
    run_dir: Path,
    payload: Any,
    subdir: str = "final",
) -> list[Path]:
    """Write hero/inference plots under `run_dir/<subdir>/` with no logger sink.

    Figures are saved as `<name>.{png,pdf}` and `[H, W, 3|4]` uint8 ndarrays
    as `<name>.png`; a dict payload writes one file per key.
    Returns the list of written paths.
    """
    out_dir = Path(run_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    items: dict[str, Any]
    if isinstance(payload, dict):
        items = payload
    else:
        items = {"final": payload}

    fig_cls = _optional_matplotlib_figure()
    for name, value in items.items():
        if value is None:
            continue
        if fig_cls is not None and isinstance(value, fig_cls):
            png = out_dir / f"{name}.png"
            pdf = out_dir / f"{name}.pdf"
            value.savefig(png, bbox_inches="tight")
            value.savefig(pdf, bbox_inches="tight")
            written.extend([png, pdf])
            continue
        if _is_image_ndarray(value):
            png = out_dir / f"{name}.png"
            _save_ndarray_as_png(value, png)
            written.append(png)
            continue
        raise TypeError(
            f"save_local_artifacts: unsupported payload for key '{name}' "
            f"({type(value).__name__}); expected matplotlib Figure or "
            f"[H,W,3|4] uint8 ndarray"
        )
    return written


def _is_image_ndarray(obj: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    if not isinstance(obj, np.ndarray):
        return False
    return obj.ndim == 3 and obj.shape[-1] in (3, 4)


def _save_ndarray_as_png(arr: Any, path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "save_local_artifacts: PIL/Pillow required to save ndarray "
            "screenshots (e.g. PyVista renders). install Pillow."
        ) from exc
    import numpy as np

    img = arr
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion to a JSON-serializable structure."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


class JSONLLogger:
    """Append-only JSONL sink for scalars + structured trajectories.

    Layout under `run_dir`:
      - `metrics.jsonl`, one line per `log_step`
      - `projection_trajectories.jsonl`, `log_projection_trajectory` with `phase != "eval"`
      - `inference_trajectories.jsonl`, `log_projection_trajectory` with `phase == "eval"`
      - `artifacts/<name>/step_NNNN.*`, payloads routed through `log_artifact`
      - `config.json`, written on `log_config`
      - `final.json`, written on `log_final`
      - `status.json`, written on `finish`
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._traj_path = self.run_dir / "projection_trajectories.jsonl"
        self._inference_traj_path = self.run_dir / "inference_trajectories.jsonl"
        self._artifacts_dir = self.run_dir / "artifacts"
        self._start = datetime.now(UTC)

    def log_config(self, **cfg: Any) -> None:
        with (self.run_dir / "config.json").open("w") as f:
            json.dump(_to_jsonable(cfg), f, indent=2, sort_keys=True)

    def log_step(self, step: int, **scalars: float) -> None:
        for k, v in scalars.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(
                    f"log_step expects scalar (int/float); got {type(v).__name__} "
                    f"for key '{k}'. Route non-scalar payloads through log_artifact."
                )
        record = {"step": int(step), **{k: float(v) for k, v in scalars.items()}}
        with self._metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def log_projection_trajectory(
        self, step: int, phase: str, trajectory: list[Any]
    ) -> None:
        record = {
            "step": int(step),
            "phase": phase,
            "trajectory": _to_jsonable(trajectory),
        }
        target = (
            self._inference_traj_path if phase == "eval" else self._traj_path
        )
        with target.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def log_artifact(self, step: int, name: str, payload: Any) -> None:
        """Serialize non-scalar payload to disk.

        Routing by type:
          - `matplotlib.figure.Figure` -> both `step_NNNN.png` (W&B-friendly
             raster preview) and `step_NNNN.pdf` (vector, paper-quality)
          - `[H, W, 3|4]` uint8 ndarray -> `step_NNNN.png` (PyVista screenshots)
          - `torch.Tensor`             -> `step_NNNN.pt`
          - dict / list / scalar       -> `step_NNNN.json`
        """
        dest = self._artifacts_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        stem = f"step_{int(step):06d}"

        fig_cls = _optional_matplotlib_figure()
        if fig_cls is not None and isinstance(payload, fig_cls):
            payload.savefig(dest / f"{stem}.png", bbox_inches="tight")
            payload.savefig(dest / f"{stem}.pdf", bbox_inches="tight")
            return

        if _is_image_ndarray(payload):
            _save_ndarray_as_png(payload, dest / f"{stem}.png")
            return

        tensor_cls = _optional_torch_tensor()
        if tensor_cls is not None and isinstance(payload, tensor_cls):
            import torch

            torch.save(payload, dest / f"{stem}.pt")
            return

        with (dest / f"{stem}.json").open("w") as f:
            json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)

    def log_final(self, **final: Any) -> None:
        with (self.run_dir / "final.json").open("w") as f:
            json.dump(_to_jsonable(final), f, indent=2, sort_keys=True)

    def finish(
        self,
        status: Literal["ok", "failed"] = "ok",
        error: str | None = None,
    ) -> None:
        duration = (datetime.now(UTC) - self._start).total_seconds()
        record = {
            "status": status,
            "error": error,
            "duration_s": duration,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        with (self.run_dir / "status.json").open("w") as f:
            json.dump(record, f, indent=2, sort_keys=True)


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
