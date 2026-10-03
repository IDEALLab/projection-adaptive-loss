"""E3: AC optimal power flow on IEEE bus systems (ml4opf constraints, pandapower grids).

Conditions (pd, qd) are drawn from a pre-generated MATLAB-accepted pool.
"""

from __future__ import annotations

import math
import os
from typing import Literal

import torch
from ml4opf.formulations.ac.violation import ACViolation
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint

from . import _safe_ops
from .grid_adapter import pandapower_to_ml4opf
from .matlab_accepted_pool import load_pool

_CASE_FAMILY = {
    "ieee30": ("e3/acopf_ieee30", "acopf_ieee30"),
    "ieee57": ("e3/acopf_ieee57", "acopf_ieee57"),
    "ieee118": ("e3/acopf_ieee118", "acopf_ieee118"),
}

_INEQ_GROUPS = [
    "vm_lower", "vm_upper",
    "pg_lower", "pg_upper",
    "qg_lower", "qg_upper",
    "thrm_1", "thrm_2",
    "dva_lower", "dva_upper",
]


def _build_spec(case: str, data: dict[str, Tensor]) -> BenchmarkSpec:
    bench_id, variant = _CASE_FAMILY[case]
    n_bus = int(data["N"].item())
    n_gen = int(data["G"].item())
    n_load = int(data["L"].item())
    n_branch = int(data["E"].item())

    n_eq = 2 * n_bus
    n_ineq = 2 * n_bus + 2 * n_gen + 2 * n_gen + 2 * n_branch + 2 * n_branch

    eq_names = [f"p_balance_{j}" for j in range(n_bus)] + [
        f"q_balance_{j}" for j in range(n_bus)
    ]
    ineq_counts = [n_bus, n_bus, n_gen, n_gen, n_gen, n_gen,
                   n_branch, n_branch, n_branch, n_branch]
    ineq_names: list[str] = []
    for g, k in zip(_INEQ_GROUPS, ineq_counts, strict=False):
        ineq_names += [f"{g}_{j}" for j in range(k)]

    pgmin, pgmax = data["pgmin"], data["pgmax"]
    qgmin, qgmax = data["qgmin"], data["qgmax"]
    vmin, vmax = data["vmin"], data["vmax"]

    # Cap MATPOWER's +/-1e7 "no limit" slack pg sentinels at a multiple of total load.
    _CAP_SLACK_PG = True
    if _CAP_SLACK_PG:
        NOLIMIT_THRESHOLD = 1e6
        sent_hi = pgmax >= NOLIMIT_THRESHOLD
        sent_lo = pgmin <= -NOLIMIT_THRESHOLD
        if bool(sent_hi.any()) or bool(sent_lo.any()):
            total_pload = float(data["pd_nominal"].abs().sum().item())
            slack_cap = max(15.0 * total_pload, 100.0)
            cap_t = torch.tensor(slack_cap, dtype=pgmax.dtype)
            if bool(sent_hi.any()):
                pgmax = torch.where(sent_hi, cap_t, pgmax)
            if bool(sent_lo.any()):
                pgmin = torch.where(sent_lo, -cap_t, pgmin)
    lo_list = (
        pgmin.tolist() + qgmin.tolist() + vmin.tolist() + [-math.pi] * n_bus
    )
    hi_list = (
        pgmax.tolist() + qgmax.tolist() + vmax.tolist() + [math.pi] * n_bus
    )
    lo = torch.tensor(lo_list, dtype=torch.float32)
    hi = torch.tensor(hi_list, dtype=torch.float32)

    model_hparams: dict = {}
    _init_std_env = os.environ.get("E3_OUTPUT_INIT_STD")
    if _init_std_env is not None:
        model_hparams["output_init_std"] = float(_init_std_env)

    return BenchmarkSpec(
        id=bench_id,
        family="e3",
        variant=variant,
        dim=2 * n_gen + 2 * n_bus,
        n_eq=n_eq,
        n_ineq=n_ineq,
        constraint_names=eq_names + ineq_names,
        constraint_types=["eq"] * n_eq + ["ineq"] * n_ineq,
        output_bounds=(lo, hi),
        condition_dim=2 * n_load,
        zeta_dim=8,
        tolerance=1e-3,
        tau=1e-3,
        model_hparams=model_hparams,
        cost="mid",
        recommended_device="cpu",
        precision="fp32",
        train_batch_size=200,
        n_eval_default=64,
        notes=f"ACOPF {case}; ml4opf violation + pandapower grid data",
    )


class E3ACOPF:
    """pal Benchmark wrapper for ACOPF on an IEEE bus system."""

    def __init__(
        self,
        case: Literal["ieee30", "ieee57", "ieee118"] = "ieee30",
    ) -> None:
        if case not in _CASE_FAMILY:
            raise ValueError(
                f"unknown case {case!r}; expected one of {sorted(_CASE_FAMILY)}"
            )
        self.case = case

        self._data = pandapower_to_ml4opf(case)
    # ACViolation is kept for parity checks; the hot path uses _safe_ops (vmap-compatible).
        self._violation = ACViolation(self._data)

        self._n_bus = int(self._data["N"].item())
        self._n_gen = int(self._data["G"].item())
        self._n_load = int(self._data["L"].item())
        self._n_branch = int(self._data["E"].item())

        self.spec = _build_spec(case, self._data)
        self._viz_layout: dict[int, tuple[float, float]] | None = None
        self._data_by_device: dict[torch.device, dict[str, Tensor]] = {}

    # Raises PoolNotPackaged / PoolCaseMismatch if the pool is missing or stale.
        self._pool = load_pool(case, live_data=self._data)
        pool_md = self._pool["metadata"]
        self._pool_split_indices: dict[str, Tensor] = {
            "train": pool_md["split_indices"]["train"],
            "eval": pool_md["split_indices"]["eval"],
        }

    def _data_on(self, device: torch.device) -> dict[str, Tensor]:
        """Return a device-resident copy of ``self._data`` (cached per device)."""
        if device not in self._data_by_device:
            self._data_by_device[device] = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in self._data.items()
            }
        return self._data_by_device[device]

    # Bus / gen partition helpers for DC3's variable-partition completion.
    # y-vector layout: ``[pg (n_gen), qg (n_gen), vm (n_bus), va (n_bus)]``.
    # Gen ordering: ext_grid generators occupy the first ``n_ext_grid`` slots.

    @property
    def objective_scale(self) -> float:
        """Per-bench cost-rescaling factor ``genbase.mean() ** 2`` shared by all methods."""
        return float(self._data["genbase"].float().mean().item() ** 2)

    @property
    def n_ext_grid(self) -> int:
        """Number of ext-grid (slack) generators."""
        return int(self._data["is_ext_grid_bus"].sum().item())

    @property
    def slack_bus_idx(self) -> list[int]:
        """Bus indices of ext-grid (slack) buses, sorted ascending."""
        mask = self._data["is_ext_grid_bus"].cpu().numpy()
        return sorted(int(i) for i in range(self._n_bus) if bool(mask[i]))

    @property
    def pv_bus_idx(self) -> list[int]:
        """Bus indices of PV (non-slack generator-attached) buses, sorted."""
        slack_set = set(self.slack_bus_idx)
        bus_gens = self._data["bus_gens"].cpu().numpy()  # [n_bus, max_gens]
        # Pad value is n_gen, anything < n_gen is a real generator slot.
        n_gen = self._n_gen
        pv: list[int] = []
        for i in range(self._n_bus):
            if i in slack_set:
                continue
            if any(int(g) < n_gen for g in bus_gens[i]):
                pv.append(i)
        return sorted(pv)

    @property
    def spv_bus_idx(self) -> list[int]:
        """Bus indices of slack + PV (all generator-attached) buses, sorted."""
        return sorted(set(self.slack_bus_idx) | set(self.pv_bus_idx))

    @property
    def slack_gen_idx(self) -> list[int]:
        """Gen-array indices of slack/ext-grid generators."""
        return list(range(self.n_ext_grid))

    @property
    def pv_gen_idx(self) -> list[int]:
        """Gen-array indices of non-slack (PV) generators."""
        return list(range(self.n_ext_grid, self._n_gen))

    @property
    def pg_start_yidx(self) -> int:
        return 0

    @property
    def qg_start_yidx(self) -> int:
        return self._n_gen

    @property
    def vm_start_yidx(self) -> int:
        return 2 * self._n_gen

    @property
    def va_start_yidx(self) -> int:
        return 2 * self._n_gen + self._n_bus

    def _unpack_output(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        g, b = self._n_gen, self._n_bus
        pg = x[:, :g]
        qg = x[:, g:2 * g]
        vm = x[:, 2 * g:2 * g + b]
        va = x[:, 2 * g + b:2 * g + 2 * b]
        return pg, qg, vm, va

    def _unpack_conditions(self, conditions: Tensor) -> tuple[Tensor, Tensor]:
        n_load = self._n_load
        return conditions[:, :n_load], conditions[:, n_load:]

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        if conditions is None:
            raise ValueError("e3 is conditional, conditions must be provided")
        pg, qg, vm, va = self._unpack_output(x)
        pd, qd = self._unpack_conditions(conditions)

        data = self._data_on(x.device)
        obj = _safe_ops.objective(pg, data["c0"], data["c1"], data["c2"])
        violations = _safe_ops.calc_violations(pd, qd, pg, qg, vm, va, data=data)

        B = x.shape[0]
        dev = x.device
        cons: list[Constraint] = []

        for name in ("p_balance", "q_balance"):
            raw = violations[name]  # [B, n_bus]
            for j in range(raw.shape[1]):
                cons.append(
                    Constraint(
                        value=raw[:, j], type="eq",
                        tol=torch.full((B,), 1e-3, device=dev),
                        margin=torch.zeros(B, device=dev),
                        name=f"{name}_{j}",
                    )
                )

        for name in _INEQ_GROUPS:
            raw = violations[name]  # [B, K]
            for j in range(raw.shape[1]):
                cons.append(
                    Constraint(
                        value=raw[:, j], type="ineq",
                        tol=torch.zeros(B, device=dev),
                        margin=torch.full((B,), 1e-2, device=dev),
                        name=f"{name}_{j}",
                    )
                )

        return obj, cons

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        if conditions is None:
            raise ValueError("e3 is conditional, conditions must be provided")
        pg, qg, vm, va = self._unpack_output(x)
        pd, qd = self._unpack_conditions(conditions)
        data = self._data_on(x.device)
        violations = _safe_ops.calc_violations(pd, qd, pg, qg, vm, va, data=data)
        parts = [violations["p_balance"], violations["q_balance"]]
        for name in _INEQ_GROUPS:
            parts.append(violations[name])
        return torch.cat(parts, dim=-1)

    def constraint_list(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> list[Constraint]:
        return self.forward(x, conditions)[1]

    def _sample_condition_batch(
        self, n: int, split: Literal["train", "eval"], g: torch.Generator,
    ) -> Tensor:
        """Sample n load conditions from the MATLAB-accepted pool (with replacement)."""
        idx_pool = self._pool_split_indices[split]
        picks = idx_pool[torch.randint(len(idx_pool), (n,), generator=g)]
        pd = self._pool["pd"][picks]
        qd = self._pool["qd"][picks]
        return torch.cat([pd, qd], dim=-1)

    def sample_queries(
        self, n: int, split: Literal["train", "eval"], seed: int,
    ) -> Query:
        g_cpu = torch.Generator("cpu").manual_seed(int(seed))
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g_cpu)
        conditions = self._sample_condition_batch(n, split, g_cpu)
        return Query(zeta=zeta, conditions=conditions)

    def _canonical_condition(self) -> Tensor:
        """Pandapower's published IEEE base case load vector (pd_nominal, qd_nominal)."""
        return torch.cat([self._data["pd_nominal"], self._data["qd_nominal"]])

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        # Evaluate at the canonical IEEE base case; only zeta varies across the batch.
        n = int(self.spec.n_eval_default if n is None else n)
        g_cpu = torch.Generator("cpu").manual_seed(int(seed) * 1000 + 7919)
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g_cpu)
        conditions = self._canonical_condition().unsqueeze(0).expand(n, -1).contiguous()
        return Query(zeta=zeta, conditions=conditions)

    def check_env(self) -> None:
        """Verify ml4opf + pandapower adapter are wired correctly."""
        lo, hi = self.spec.output_bounds
        x = 0.5 * (lo + hi).unsqueeze(0)
        pd_nom = self._data["pd_nominal"].unsqueeze(0)
        qd_nom = self._data["qd_nominal"].unsqueeze(0)
        conds = torch.cat([pd_nom, qd_nom], dim=-1)
        obj, _ = self.forward(x, conds)
        if not torch.isfinite(obj).all():
            raise RuntimeError("e3 check_env: non-finite objective from midpoint input")

    def visualize_train(self, x: Tensor, conditions: Tensor | None = None):
        """Graph-layout view of the bus network (nodes colored by vm, edges by |S_from|)."""
        try:
            import math

            import matplotlib.pyplot as plt
            import networkx as nx
            from matplotlib.collections import LineCollection
        except ImportError:
            return None

        x_single = x.detach().cpu().reshape(1, -1)
        pg, qg, vm, va = self._unpack_output(x_single)
        vm_np = vm[0].cpu().numpy()
        va_np = va[0].cpu().numpy()

        data = self._data
        bus_fr = data["bus_fr"].cpu().numpy()
        bus_to = data["bus_to"].cpu().numpy()
        n_bus = self._n_bus
        n_branch = self._n_branch

        # Apparent power from the 'from' end of each branch:
        #   S_fr = V_i * conj( yff * V_i + yft * V_j )
        v_complex = vm_np * (
            (va_np * 0 + 0j) + 1.0  # float -> complex dtype
        )  # placeholder; overwritten below
        import numpy as np

        v_complex = vm_np * np.exp(1j * va_np)
        yff = data["gff"].cpu().numpy() + 1j * data["bff"].cpu().numpy()
        yft = data["gft"].cpu().numpy() + 1j * data["bft"].cpu().numpy()
        vi = v_complex[bus_fr]
        vj = v_complex[bus_to]
        s_fr = vi * np.conj(yff * vi + yft * vj)
        s_mag = np.abs(s_fr)
        smax_np = data["smax"].cpu().numpy()
        flow_rel = np.clip(s_mag / np.maximum(smax_np, 1e-6), 0.0, 1.5)

        # Prefer pandapower's IEEE geodata, else a force-directed layout.
        if self._viz_layout is None:
            bus_xy = data.get("bus_xy")
            xy_np = bus_xy.cpu().numpy() if bus_xy is not None else None
            if xy_np is not None and np.isfinite(xy_np).all():
                self._viz_layout = {i: tuple(xy_np[i]) for i in range(n_bus)}
            else:
                g = nx.Graph()
                g.add_nodes_from(range(n_bus))
                g.add_edges_from(zip(bus_fr.tolist(), bus_to.tolist(), strict=False))
                if n_bus <= 40:
                    self._viz_layout = nx.kamada_kawai_layout(g)
                else:
                    self._viz_layout = nx.spring_layout(
                        g, seed=0, iterations=200, k=1.0 / math.sqrt(n_bus),
                    )

        pos = self._viz_layout

        fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=120)

        # Edges as a LineCollection, width ~ flow, color fades when overloaded.
        edge_segments = [(pos[int(bus_fr[k])], pos[int(bus_to[k])])
                         for k in range(n_branch)]
        edge_widths = 0.6 + 2.2 * flow_rel
        edge_colors = [
            "#c0392b" if flow_rel[k] > 1.0 else "#4A6070"
            for k in range(n_branch)
        ]
        ax.add_collection(LineCollection(
            edge_segments, linewidths=edge_widths, colors=edge_colors,
            zorder=1, alpha=0.8,
        ))

        xs = [pos[i][0] for i in range(n_bus)]
        ys = [pos[i][1] for i in range(n_bus)]

        # Fixed colormap bounds from the spec so frames stay comparable.
        vmin = float(data["vmin"].min().item())
        vmax_ = float(data["vmax"].max().item())
        sc = ax.scatter(
            xs, ys, c=vm_np, cmap="coolwarm", vmin=vmin, vmax=vmax_,
            s=70, edgecolors="#222", linewidths=0.8, zorder=3,
        )

        # Slack buses get a yellow square marker.
        ext_grid_mask = (
            data["is_ext_grid_bus"].cpu().numpy()
            if "is_ext_grid_bus" in data
            else np.zeros(n_bus, dtype=bool)
        )
        gen_buses_all = sorted({int(b) for b in data["bus_gens"].cpu().numpy().flatten()
                                if int(b) < n_bus})
        gen_buses = [i for i in gen_buses_all if not ext_grid_mask[i]]
        ext_grid_buses = [i for i in range(n_bus) if ext_grid_mask[i]]

        if gen_buses:
            ax.scatter(
                [pos[i][0] for i in gen_buses],
                [pos[i][1] for i in gen_buses],
                s=170, facecolors="none", edgecolors="#27ae60", linewidths=1.6,
                zorder=2, label="generator",
            )
        if ext_grid_buses:
            ax.scatter(
                [pos[i][0] for i in ext_grid_buses],
                [pos[i][1] for i in ext_grid_buses],
                s=140, marker="s", facecolors="none", edgecolors="#f1c40f",
                linewidths=1.8, zorder=2, label="slack / ext grid",
            )

        # Transformer glyph at branch midpoints where the tap ratio deviates from unity.
        if "is_trafo" in data:
            trafo_mask = data["is_trafo"].cpu().numpy()
            trafo_mid_x = []
            trafo_mid_y = []
            for k in range(n_branch):
                if not trafo_mask[k]:
                    continue
                p1 = pos[int(bus_fr[k])]
                p2 = pos[int(bus_to[k])]
                trafo_mid_x.append(0.5 * (p1[0] + p2[0]))
                trafo_mid_y.append(0.5 * (p1[1] + p2[1]))
            if trafo_mid_x:
                ax.scatter(
                    trafo_mid_x, trafo_mid_y, s=28, marker="s",
                    facecolors="white", edgecolors="#444", linewidths=0.8,
                    zorder=3, label="transformer",
                )

        load_buses = sorted({int(b) for b in data["bus_loads"].cpu().numpy().flatten()
                             if int(b) < n_bus})
        if load_buses:
            ax.scatter(
                [pos[i][0] for i in load_buses],
                [pos[i][1] for i in load_buses],
                s=45, marker="v", facecolors="#f39c12", edgecolors="#8B5A00",
                linewidths=0.6, zorder=4, label="load",
            )

        with torch.no_grad():
            obj_val = float(self._violation.objective(pg).item())
            conds_for_viol = (
                conditions.detach().cpu().reshape(1, -1)
                if conditions is not None
                else torch.cat([data["pd_nominal"], data["qd_nominal"]]).unsqueeze(0)
            )
            pd_v, qd_v = self._unpack_conditions(conds_for_viol)
            viols = self._violation.calc_violations(
                pd_v, qd_v, pg, qg, vm, va=va,
                reduction="none", clamp=False,
            )
        eq_viol = max(
            float(viols["p_balance"].abs().max().item()),
            float(viols["q_balance"].abs().max().item()),
        )
        ineq_viol = max(
            float(viols[name].clamp(min=0).max().item())
            for name in _INEQ_GROUPS
        )

        annot = (
            f"case:    {self.case}\n"
            f"obj:     {obj_val:+.3e}\n"
            f"|v| eq:  {eq_viol:.3e}\n"
            f"max ineq:{ineq_viol:.3e}"
        )
        ax.text(
            0.02, 0.98, annot, transform=ax.transAxes, fontsize=8,
            va="top", ha="left", family="monospace",
            bbox=dict(facecolor="white", edgecolor="#ccc", alpha=0.9, pad=4),
        )

        cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("voltage magnitude (pu)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"e3 ACOPF: {self.case} grid")
        ax.legend(
            loc="lower right", fontsize=7, framealpha=0.9,
            markerscale=0.7, labelspacing=1.1, borderpad=0.6,
            handletextpad=0.8,
        )
        fig.tight_layout()
        return fig

    def visualize_final(self, x: Tensor, conditions: Tensor | None = None):
        """Same grid view as `visualize_train`, saved as `final/grid.{png,pdf}`."""
        fig = self.visualize_train(x, conditions)
        if fig is None:
            return None
        return {"grid": fig}
