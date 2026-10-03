"""Build a wingbox CADProgram with the `bwb` shape from the analytical Bezier OML or neural SDF.

The analytical BWB is x-flipped to match the neural convention (LE at +x).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
import yaml

from ..analytical_bwb.geometry import (
    build_yaml_config,
    compute_control_nets,
    compute_stations,
)

HERE = Path(__file__).resolve().parent
WINGBOX_YAML = HERE / "wingbox.yaml"


# BWB_FAST_SDF=1 lowers the Bezier closest-UV search quality (~3.7x fewer evals).
if os.environ.get("BWB_FAST_SDF"):
    from geometry.stdlib import surface as _tc_surface
    _tc_closest_uv_full = _tc_surface._closest_uv
    def _closest_uv_fast(query, net, n_coarse=8, n_newton=2):
        return _tc_closest_uv_full(query, net, n_coarse=n_coarse, n_newton=n_newton)
    _tc_surface._closest_uv = _closest_uv_fast


BWB_PARAM_NAMES = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]
STRUCT_PARAM_NAMES = [
    "skin_t", "front_spar_x", "front_spar_w", "front_spar_dev",
    "rear_spar_x", "rear_spar_w",
    "bat_x", "bat_y", "bat_z", "bat_z_center",
    "rib_start_y", "rib_end_y", "rib_w", "center_rib_w",
]


def _analytical_bwb_shapes(params: dict, resolution: int) -> dict:
    """Build the analytical BWB config (unit-scale ratios, C1=1.0) and return its `shapes`."""
    kwargs = dict(
        B1=params["B1"], B2=params["B2"], B3=params["B3"],
        C1=1.0,  # unit-scale ratio convention
        C2=params["C2"], C3=params["C3"], C4=params["C4"],
        S1=params["S1"], S2=params["S2"], S3=params["S3"],
        L=1.0,  # output in unit scale (meters @ L=1m)
    )
    control_nets = compute_control_nets(**kwargs)
    stations = compute_stations(**kwargs)
    cfg = build_yaml_config(control_nets, stations, resolution=resolution)
    return cfg["shapes"]


def build_config(
    params: dict,
    resolution: int = 128,
    bwb_source: str = "analytical",
) -> dict:
    """Build a wingbox YAML config dict with a BWB baked in.

    Args:
        params: Dict with the 9 BWB shape params (B1..S3). L is not used
            here (analytical geometry is built at unit scale). Structural
            params (skin_t, front_spar_x, ...) are passed as bindings and
            can be overridden later via `with_bindings(structural=...)`.
        resolution: Mesh grid resolution (used by CADProgram).
        bwb_source: "analytical" (default) injects the Bezier patch tree;
            "neural" injects a single neural3d node pointing at ./bwb_sdf.

    Returns:
        A dict that CADProgram can consume (same shape as a parsed YAML).
    """
    with open(WINGBOX_YAML) as f:
        cfg = yaml.safe_load(f)

    cfg["shapes"].pop("bwb", None)

    if bwb_source == "analytical":
        bwb_shapes = _analytical_bwb_shapes(params, resolution)
        bwb_shapes["bwb_pre_flip"] = bwb_shapes.pop("bwb")
        cfg["shapes"].update(bwb_shapes)
        cfg["shapes"]["bwb"] = {
            "type": "mirror", "shape": "bwb_pre_flip", "plane": "yz", "offset": 0,
        }
    elif bwb_source == "neural":
        # Neural SDF: one neural3d node, path relative to the temp YAML in HERE.
        cfg["shapes"]["bwb"] = {
            "type": "neural3d",
            "path": "bwb_sdf",
            **{k: f"${k}" for k in BWB_PARAM_NAMES},
        }
    else:
        raise ValueError(
            f"bwb_source must be 'neural' or 'analytical', got {bwb_source!r}"
        )

    cfg["bounds"]["resolution"] = resolution
    return cfg


def build_program(
    params: dict,
    resolution: int = 128,
    device: str = "cpu",
    bwb_source: str = "analytical",
):
    """Build a wingbox CADProgram, structural tree bound to `params`, BWB baked in."""
    from geometry.program import CADProgram  # lazy

    cfg = build_config(
        params,
        resolution=resolution,
        bwb_source=bwb_source,
    )

    # Round-trip through a temp YAML so $var expressions in bindings are evaluated.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", dir=HERE, delete=False,
    ) as f:
        yaml.safe_dump(cfg, f)
        tmp_path = Path(f.name)
    try:
        prog = CADProgram.load_from_yaml(str(tmp_path), device=device)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Bind the bwb_conditions (still used by $S3 etc.) and the structural params.
    bwb_tensor = torch.tensor(
        [[params[k] for k in BWB_PARAM_NAMES]],
        dtype=torch.float32, device=device,
    )
    struct_tensor = torch.tensor(
        [[params[k] for k in STRUCT_PARAM_NAMES]],
        dtype=torch.float32, device=device,
    )
    prog = prog.with_bindings(bwb_conditions=bwb_tensor, structural=struct_tensor)
    return prog
