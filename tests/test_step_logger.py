"""JSONLLogger unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pal.tracking.step_logger import JSONLLogger, save_local_artifacts


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_log_config_writes_config_json(tmp_path: Path) -> None:
    log = JSONLLogger(tmp_path)
    log.log_config(method="pal_loggap", bench="rosenbrock_eq", seed=0)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config == {"bench": "rosenbrock_eq", "method": "pal_loggap", "seed": 0}


def test_log_step_appends_scalars_as_jsonl(tmp_path: Path) -> None:
    log = JSONLLogger(tmp_path)
    log.log_step(0, loss=1.25, viol=0.01)
    log.log_step(1, loss=0.75, viol=0.005)

    rows = _read_jsonl(tmp_path / "metrics.jsonl")
    assert rows == [
        {"step": 0, "loss": 1.25, "viol": 0.01},
        {"step": 1, "loss": 0.75, "viol": 0.005},
    ]


def test_log_step_rejects_non_scalar_values(tmp_path: Path) -> None:
    log = JSONLLogger(tmp_path)
    with pytest.raises(TypeError, match="scalar"):
        log.log_step(0, x=[1, 2, 3])
    with pytest.raises(TypeError, match="scalar"):
        log.log_step(0, ok=True)


def test_log_projection_trajectory_writes_structured_entry(tmp_path: Path) -> None:
    from dataclasses import dataclass

    @dataclass
    class _Step:
        iter: int
        obj: float
        constraints: list[float]

    traj = [
        _Step(iter=0, obj=1.0, constraints=[0.5, -0.2]),
        _Step(iter=1, obj=0.5, constraints=[0.1, -0.1]),
    ]
    log = JSONLLogger(tmp_path)
    log.log_projection_trajectory(100, phase="train", trajectory=traj)

    rows = _read_jsonl(tmp_path / "projection_trajectories.jsonl")
    assert len(rows) == 1
    assert rows[0]["step"] == 100
    assert rows[0]["phase"] == "train"
    assert rows[0]["trajectory"] == [
        {"iter": 0, "obj": 1.0, "constraints": [0.5, -0.2]},
        {"iter": 1, "obj": 0.5, "constraints": [0.1, -0.1]},
    ]


def test_log_artifact_routes_scalar_payload_as_json(tmp_path: Path) -> None:
    log = JSONLLogger(tmp_path)
    log.log_artifact(step=42, name="summary", payload={"a": 1, "b": [2, 3]})

    path = tmp_path / "artifacts" / "summary" / "step_000042.json"
    assert path.exists()
    assert json.loads(path.read_text()) == {"a": 1, "b": [2, 3]}


def test_log_artifact_routes_torch_tensor_as_pt(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    log = JSONLLogger(tmp_path)
    payload = torch.tensor([1.0, 2.0, 3.0])
    log.log_artifact(step=7, name="x_final", payload=payload)

    path = tmp_path / "artifacts" / "x_final" / "step_000007.pt"
    assert path.exists()
    reloaded = torch.load(path, weights_only=False)
    assert torch.equal(reloaded, payload)


def test_log_artifact_routes_matplotlib_figure_as_png_and_pdf(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1, 2], [1, 0, 1])

    log = JSONLLogger(tmp_path)
    log.log_artifact(step=12, name="viz", payload=fig)

    png_path = tmp_path / "artifacts" / "viz" / "step_000012.png"
    pdf_path = tmp_path / "artifacts" / "viz" / "step_000012.pdf"
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 100  # non-empty PNG
    assert pdf_path.read_bytes().startswith(b"%PDF")  # valid PDF magic


def test_log_projection_trajectory_routes_eval_phase_to_inference_file(
    tmp_path: Path,
) -> None:
    from dataclasses import dataclass

    @dataclass
    class _Step:
        iter: int
        obj: float
        constraints: list[float]

    log = JSONLLogger(tmp_path)
    log.log_projection_trajectory(0, phase="eval", trajectory=[_Step(0, 1.0, [0.0])])
    log.log_projection_trajectory(100, phase="train", trajectory=[_Step(0, 2.0, [0.5])])

    eval_rows = _read_jsonl(tmp_path / "inference_trajectories.jsonl")
    train_rows = _read_jsonl(tmp_path / "projection_trajectories.jsonl")
    assert [r["phase"] for r in eval_rows] == ["eval"]
    assert [r["phase"] for r in train_rows] == ["train"]


def test_log_final_and_finish_writes_status(tmp_path: Path) -> None:
    log = JSONLLogger(tmp_path)
    log.log_final(obj_mean=0.5, viol_max=0.01, feasibility=1.0)
    log.finish(status="ok")

    final = json.loads((tmp_path / "final.json").read_text())
    assert final == {"feasibility": 1.0, "obj_mean": 0.5, "viol_max": 0.01}

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "ok"
    assert status["error"] is None
    assert status["duration_s"] >= 0.0
    assert "finished_at" in status


def test_finish_records_failure(tmp_path: Path) -> None:
    log = JSONLLogger(tmp_path)
    log.finish(status="failed", error="KeyError: query not in cache")

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["error"] == "KeyError: query not in cache"


def test_save_local_artifacts_writes_figure_dict_as_png_and_pdf(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig_a = Figure()
    fig_a.subplots().plot([0, 1], [1, 0])
    fig_b = Figure()
    fig_b.subplots().plot([0, 1], [0, 1])

    written = save_local_artifacts(
        tmp_path, {"map_2d": fig_a, "overview": fig_b}, subdir="final"
    )

    assert (tmp_path / "final" / "map_2d.png").exists()
    assert (tmp_path / "final" / "map_2d.pdf").exists()
    assert (tmp_path / "final" / "overview.png").exists()
    assert (tmp_path / "final" / "overview.pdf").exists()
    assert len(written) == 4


def test_save_local_artifacts_writes_ndarray_screenshot_as_png(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL")
    import numpy as np

    arr = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)
    written = save_local_artifacts(tmp_path, {"3d_view": arr})

    png = tmp_path / "final" / "3d_view.png"
    assert png.exists()
    assert png.stat().st_size > 100
    assert written == [png]


def test_save_local_artifacts_accepts_single_figure_with_default_name(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure()
    fig.subplots().plot([0, 1], [0, 1])
    save_local_artifacts(tmp_path, fig)

    assert (tmp_path / "final" / "final.png").exists()
    assert (tmp_path / "final" / "final.pdf").exists()


def test_save_local_artifacts_rejects_unsupported_payload(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="unsupported payload"):
        save_local_artifacts(tmp_path, {"bogus": "not a figure"})
