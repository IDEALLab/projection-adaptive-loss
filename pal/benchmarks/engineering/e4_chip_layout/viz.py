"""e4 chip layout visualizations: a matplotlib diagnostic and a PyVista 3D render."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

W_MIN = 0.5

BLOCK_HEIGHT = 1.8
SUBSTRATE_DEPTH = 2.5
WIRE_Z = BLOCK_HEIGHT + 0.4
PIN_Z = BLOCK_HEIGHT / 2.0
MARGIN_FRAC = 0.06

BG_COLOR = "#ffffff"
SUBSTRATE_COLOR = "#1e1e2e"
BORDER_COLOR = "#3a3a6a"
WIRE_B2B_COLOR = "#4477dd"
WIRE_P2B_COLOR = "#33aaaa"
PIN_COLOR = "#aabbee"

TYPE_COLORS = {
    "rect": "#3d7ec9",
    "L": "#3daa6b",
    "T": "#9b5ec9",
}


def _extract_layout(bench: Any, x_row: Tensor) -> dict[str, Any]:
    """Derive per-block geometry + connectivity from pal bench state + one design.

    Args:
        bench: E4ChipLayout instance.
        x_row: Decision vector of shape [3*N] (1D) or [1, 3*N].
    """
    if x_row.ndim == 2:
        x_row = x_row[0]
    N = bench.n_blocks
    positions = x_row[: 2 * N]
    widths_raw = x_row[2 * N :]
    cx = positions[0::2].detach().cpu()
    cy = positions[1::2].detach().cpu()
    w = F.softplus(widths_raw, beta=1.0).detach().cpu() + W_MIN
    h = bench.area_target.detach().cpu() / w

    b2b = None
    if bench.b2b_i.numel() > 0:
        b2b = torch.stack(
            [bench.b2b_i.float().cpu(), bench.b2b_j.float().cpu(), bench.b2b_w.cpu()],
            dim=1,
        )
    p2b = None
    if bench.p2b_pin.numel() > 0:
        p2b = torch.stack(
            [
                bench.p2b_pin.float().cpu(),
                bench.p2b_blk.float().cpu(),
                bench.p2b_w.cpu(),
            ],
            dim=1,
        )
    return {
        "cx": cx.numpy(),
        "cy": cy.numpy(),
        "w": w.numpy(),
        "h": h.numpy(),
        "block_ids": np.arange(N),
        "block_types": ["rect"] * N,
        "b2b": b2b,
        "p2b": p2b,
        "pins_pos": bench.pins_pos.detach().cpu().numpy(),
        "canvas": float(bench.canvas_size),
        "n_blocks": N,
    }


def _pairwise_overlap(cx, cy, w, h):
    """Return per-pair overlap volumes (thickness 0.1) and pair index arrays.

    Matches the AABB overlap the benchmark objective uses.
    """
    N = len(cx)
    ia, ib = np.triu_indices(N, k=1)
    dx = np.abs(cx[ia] - cx[ib])
    dy = np.abs(cy[ia] - cy[ib])
    ox = np.maximum(0.0, (w[ia] + w[ib]) / 2 - dx)
    oy = np.maximum(0.0, (h[ia] + h[ib]) / 2 - dy)
    vol = ox * oy * 0.1
    return ia, ib, vol


def render_diagnostic_2d(bench: Any, x_row: Tensor):
    """Matplotlib optimization-state diagnostic. Returns `matplotlib.figure.Figure`."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    g = _extract_layout(bench, x_row)
    cx, cy, w, h = g["cx"], g["cy"], g["w"], g["h"]
    N = g["n_blocks"]
    canvas = g["canvas"]
    block_types = g["block_types"]

    ia, ib, vol = _pairwise_overlap(cx, cy, w, h)
    violating = set(ia[vol > 0]) | set(ib[vol > 0])

    x_row_t = x_row if x_row.ndim == 2 else x_row.unsqueeze(0)
    with torch.no_grad():
        obj, cons = bench.forward(x_row_t.to(bench.area_target.device))
    obj_val = float(obj.item())
    cons_val = float(cons[0].value.item())
    feasible = cons_val <= float(bench.spec.tolerance)

    from torch.nn.functional import softplus as _sp

    widths_raw = x_row_t[0, 2 * N :]
    w_t = _sp(widths_raw, beta=1.0) + W_MIN
    h_t = bench.area_target.cpu() / w_t.cpu()
    cx_t = x_row_t[0, : 2 * N][0::2].cpu()
    cy_t = x_row_t[0, : 2 * N][1::2].cpu()
    bb_area = float(
        ((cx_t + w_t.cpu() / 2).max() - (cx_t - w_t.cpu() / 2).min())
        * ((cy_t + h_t / 2).max() - (cy_t - h_t / 2).min())
    )
    hpwl = obj_val - bench.lambda_area * bb_area

    fig = plt.figure(figsize=(12, 7), dpi=110)
    gs = GridSpec(2, 2, width_ratios=[3, 2], height_ratios=[3, 2], figure=fig, hspace=0.3, wspace=0.25)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_bars = fig.add_subplot(gs[1, 1])

    ax_main.add_patch(
        mpatches.Rectangle(
            (0, 0), canvas, canvas,
            linewidth=1.2, edgecolor="#3a3a6a", facecolor="#f6f6fa", zorder=0,
        )
    )

    for i in range(N):
        color = TYPE_COLORS.get(block_types[i], "#3d7ec9")
        edge = "#cc2233" if i in violating else "#000010"
        lw = 2.0 if i in violating else 0.8
        rect = mpatches.FancyBboxPatch(
            (cx[i] - w[i] / 2, cy[i] - h[i] / 2), w[i], h[i],
            boxstyle="round,pad=0.0",
            linewidth=lw, edgecolor=edge,
            facecolor=color, alpha=0.55, zorder=2,
        )
        ax_main.add_patch(rect)
        ax_main.text(
            cx[i], cy[i], str(i), ha="center", va="center",
            fontsize=7, fontweight="bold", color="white", zorder=4,
        )

    if g["b2b"] is not None:
        b2b_np = g["b2b"].numpy()
        for e in range(b2b_np.shape[0]):
            i, j = int(b2b_np[e, 0]), int(b2b_np[e, 1])
            if i >= N or j >= N:
                continue
            ax_main.plot(
                [cx[i], cx[j]], [cy[i], cy[j]],
                color=WIRE_B2B_COLOR, linewidth=0.4 + 0.8 * float(b2b_np[e, 2]),
                alpha=0.35, zorder=1,
            )

    pins_pos = g["pins_pos"]
    if g["p2b"] is not None and pins_pos.shape[0] > 0:
        p2b_np = g["p2b"].numpy()
        for e in range(p2b_np.shape[0]):
            pi, bi = int(p2b_np[e, 0]), int(p2b_np[e, 1])
            if pi >= pins_pos.shape[0] or bi >= N:
                continue
            ax_main.plot(
                [pins_pos[pi, 0], cx[bi]], [pins_pos[pi, 1], cy[bi]],
                color=WIRE_P2B_COLOR, linewidth=0.5, alpha=0.35, zorder=1,
            )
    if pins_pos.shape[0] > 0:
        ax_main.scatter(
            pins_pos[:, 0], pins_pos[:, 1],
            c="#cc2233", s=26, marker="x", linewidths=1.2, zorder=5,
        )

    x_lo = float((cx - w / 2).min())
    x_hi = float((cx + w / 2).max())
    y_lo = float((cy - h / 2).min())
    y_hi = float((cy + h / 2).max())
    ax_main.add_patch(
        mpatches.Rectangle(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            linewidth=1.3, edgecolor="black", facecolor="none",
            linestyle="--", alpha=0.45, zorder=3,
        )
    )

    margin = canvas * MARGIN_FRAC
    ax_main.set_xlim(-margin, canvas + margin)
    ax_main.set_ylim(-margin, canvas + margin)
    ax_main.set_aspect("equal")
    ax_main.set_title(f"e4 chip_layout ({N} blocks)")
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("y")

    ax_text.axis("off")
    feasible_str = "FEASIBLE" if feasible else "INFEASIBLE"
    feasible_color = "#2a8a3a" if feasible else "#cc2233"
    lines = [
        f"HPWL            {hpwl:12.3f}",
        f"bbox area       {bb_area:12.3f}",
        f"lambda * bbox   {bench.lambda_area * bb_area:12.3f}",
        f"objective       {obj_val:12.3f}",
        "",
        f"overlap sum     {cons_val:12.4g}",
        f"tolerance       {bench.spec.tolerance:12.4g}",
    ]
    ax_text.text(
        0.0, 1.0, "\n".join(lines),
        family="monospace", fontsize=10, va="top", ha="left",
        transform=ax_text.transAxes,
    )
    ax_text.text(
        0.0, 0.02, feasible_str,
        family="monospace", fontsize=13, fontweight="bold",
        color=feasible_color, va="bottom", ha="left",
        transform=ax_text.transAxes,
    )

    if vol.max() > 0:
        order = np.argsort(-vol)[:8]
        labels = [f"{ia[k]}-{ib[k]}" for k in order if vol[k] > 0]
        values = [vol[k] for k in order if vol[k] > 0]
        y = np.arange(len(values))
        ax_bars.barh(y, values, color="#cc2233", alpha=0.75)
        ax_bars.set_yticks(y)
        ax_bars.set_yticklabels(labels, fontsize=8)
        ax_bars.invert_yaxis()
        ax_bars.set_xlabel("overlap volume")
        ax_bars.set_title("top overlapping pairs", fontsize=10)
    else:
        ax_bars.axis("off")
        ax_bars.text(
            0.5, 0.5, "all pairs clear",
            ha="center", va="center", fontsize=12,
            color="#2a8a3a", transform=ax_bars.transAxes,
        )

    return fig


def _lines_polydata(segments):
    import pyvista as pv

    if not segments:
        return pv.PolyData()
    pts = np.vstack([[p0, p1] for p0, p1 in segments])
    n = len(segments)
    conn = np.empty(n * 3, dtype=int)
    for k in range(n):
        conn[k * 3], conn[k * 3 + 1], conn[k * 3 + 2] = 2, k * 2, k * 2 + 1
    pd = pv.PolyData(pts)
    pd.lines = conn
    return pd


def _render_hero_3d_impl(bench: Any, x_row: Tensor) -> np.ndarray:
    import pyvista as pv

    g = _extract_layout(bench, x_row)
    cx, cy, w, h = g["cx"], g["cy"], g["w"], g["h"]
    N = g["n_blocks"]
    canvas = g["canvas"]
    block_types = g["block_types"]
    pins_pos = g["pins_pos"]

    pl = pv.Plotter(window_size=(1600, 1000), off_screen=True)
    pl.set_background(BG_COLOR)

    margin = canvas * MARGIN_FRAC
    pl.add_mesh(
        pv.Box(bounds=(-margin, canvas + margin, -margin, canvas + margin, -SUBSTRATE_DEPTH, 0.0)),
        color=SUBSTRATE_COLOR,
        smooth_shading=True,
        specular=0.05,
    )

    border_pts = np.array([
        [0.0, 0.0, 0.01], [canvas, 0.0, 0.01],
        [canvas, canvas, 0.01], [0.0, canvas, 0.01], [0.0, 0.0, 0.01],
    ])
    pl.add_mesh(pv.Spline(border_pts, 5), color=BORDER_COLOR, line_width=1.2, opacity=0.8)

    for i in range(N):
        color = TYPE_COLORS.get(block_types[i], "#3d7ec9")
        pl.add_mesh(
            pv.Box(bounds=(
                cx[i] - w[i] / 2, cx[i] + w[i] / 2,
                cy[i] - h[i] / 2, cy[i] + h[i] / 2,
                0.0, BLOCK_HEIGHT,
            )),
            color=color,
            smooth_shading=False,
            specular=0.6,
            specular_power=30,
            show_edges=True,
            edge_color="#000010",
            line_width=0.5,
        )

    if g["b2b"] is not None:
        b2b_np = g["b2b"].numpy()
        segs = [
            (np.array([cx[int(b2b_np[e, 0])], cy[int(b2b_np[e, 0])], WIRE_Z]),
             np.array([cx[int(b2b_np[e, 1])], cy[int(b2b_np[e, 1])], WIRE_Z]))
            for e in range(b2b_np.shape[0])
            if int(b2b_np[e, 0]) < N and int(b2b_np[e, 1]) < N
        ]
        if segs:
            pl.add_mesh(_lines_polydata(segs), color=WIRE_B2B_COLOR,
                        line_width=1.0, opacity=0.45, render_lines_as_tubes=True)

    if g["p2b"] is not None and pins_pos.shape[0] > 0:
        p2b_np = g["p2b"].numpy()
        segs_p = [
            (np.array([float(pins_pos[int(p2b_np[e, 0]), 0]),
                       float(pins_pos[int(p2b_np[e, 0]), 1]), WIRE_Z]),
             np.array([cx[int(p2b_np[e, 1])], cy[int(p2b_np[e, 1])], WIRE_Z]))
            for e in range(p2b_np.shape[0])
            if int(p2b_np[e, 0]) < pins_pos.shape[0] and int(p2b_np[e, 1]) < N
        ]
        if segs_p:
            pl.add_mesh(_lines_polydata(segs_p), color=WIRE_P2B_COLOR,
                        line_width=0.8, opacity=0.35, render_lines_as_tubes=True)

    pin_r = canvas * 0.006
    for pi in range(pins_pos.shape[0]):
        pl.add_mesh(
            pv.Sphere(radius=pin_r, center=(float(pins_pos[pi, 0]), float(pins_pos[pi, 1]), PIN_Z)),
            color=PIN_COLOR, specular=0.9, specular_power=60,
        )

    pl.enable_lightkit()
    pl.add_light(pv.Light(
        position=(canvas * 0.2, canvas * 1.8, canvas * 2.0),
        focal_point=(canvas / 2, canvas / 2, 0),
        intensity=0.6,
    ))

    cx_c, cy_c = canvas / 2, canvas / 2
    pl.camera_position = [
        (cx_c - canvas * 1.3, cy_c - canvas * 1.6, canvas * 1.2),
        (cx_c, cy_c, 0.0),
        (0, 0, 1),
    ]

    arr = pl.screenshot(return_img=True)
    pl.close()
    return np.asarray(arr)


def render_hero_3d(bench: Any, x_row: Tensor):
    """PyVista render as a uint8 ndarray, or the matplotlib diagnostic without GL."""
    try:
        return _render_hero_3d_impl(bench, x_row)
    except Exception as exc:
        import sys
        print(
            f"  [e4 viz_final] PyVista render failed ({type(exc).__name__}: {exc}); "
            f"falling back to matplotlib diagnostic",
            file=sys.stderr,
        )
        return render_diagnostic_2d(bench, x_row)
