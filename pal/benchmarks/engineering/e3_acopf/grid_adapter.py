"""Convert pandapower IEEE cases to ml4opf-compatible data dicts (no Julia/PGLearn needed)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import Tensor


def _pad_ragged(lists: list[list[int]], pad_value: int) -> Tensor:
    """Pad ragged list-of-lists to a dense int tensor."""
    max_len = max(len(row) for row in lists) if lists else 0
    max_len = max(max_len, 1)
    out = np.full((len(lists), max_len), pad_value, dtype=np.int64)
    for i, row in enumerate(lists):
        for j, v in enumerate(row):
            out[i, j] = v
    return torch.as_tensor(out)


def pandapower_to_ml4opf(case_name: str) -> dict[str, Tensor]:
    """Load a pandapower IEEE case and return an ml4opf-compatible data dict.

    All power quantities are converted to per-unit (baseMVA = 100).
    Admittance parameters computed from the MATPOWER pi-model.

    Args:
        case_name: 'ieee30', 'ieee57', or 'ieee118'.

    Returns:
        Dict of tensors compatible with ml4opf ACViolation.__init__.
    """
    import pandapower as pp
    import pandapower.networks as pn

    loaders = {
        "ieee30": pn.case_ieee30,
        "ieee57": pn.case57,
        "ieee118": pn.case118,
    }
    if case_name not in loaders:
        raise ValueError(f"Unknown case '{case_name}'. Available: {sorted(loaders)}")

    net = loaders[case_name]()
    pp.runpp(net, numba=False)
    ppc = net._ppc
    baseMVA = ppc["baseMVA"]

    bus = ppc["bus"]
    gen = ppc["gen"]
    branch = ppc["branch"]

    n_bus = bus.shape[0]
    n_gen = gen.shape[0]
    n_branch = branch.shape[0]

    load_buses = net.load.bus.values.astype(int)
    n_load = len(load_buses)

    # pandapower ships VMIN=0 / VMAX=2 for these cases: fall back to MATPOWER's [0.94, 1.06].
    vmin_raw = bus[:, 12].astype(np.float32)  # VMIN col
    vmax_raw = bus[:, 11].astype(np.float32)  # VMAX col
    vmin_raw = np.where(vmin_raw <= 0.5, 0.94, vmin_raw)
    vmax_raw = np.where(vmax_raw >= 1.5, 1.06, vmax_raw)
    vmin = torch.as_tensor(vmin_raw, dtype=torch.float32)
    vmax = torch.as_tensor(vmax_raw, dtype=torch.float32)
    gs = torch.as_tensor(bus[:, 4] / baseMVA, dtype=torch.float32)  # GS col
    bs = torch.as_tensor(bus[:, 5] / baseMVA, dtype=torch.float32)  # BS col
    # Base-case PF solution: Newton warm start for DC3 (a flat start diverges on ieee57).
    vm_init = torch.as_tensor(bus[:, 7], dtype=torch.float32)  # VM col (pu)
    va_init = torch.as_tensor(np.deg2rad(bus[:, 8]), dtype=torch.float32)  # VA col -> rad

    gen_buses = gen[:, 0].astype(int)
    pgmax_np = (gen[:, 8] / baseMVA).astype("float32")
    pgmin_np = (gen[:, 9] / baseMVA).astype("float32")
    qgmax_np = (gen[:, 3] / baseMVA).astype("float32")
    qgmin_np = (gen[:, 4] / baseMVA).astype("float32")

    # pandapower puts qmin=qmax=0 on ext_grid rows in ppc; the OPF limits live on net.ext_grid.
    n_ext_grid = len(net.ext_grid)
    for eg_pos, (_, row) in enumerate(net.ext_grid.iterrows()):
        qmin_mvar = float(row["min_q_mvar"]) if "min_q_mvar" in row and row["min_q_mvar"] is not None else 0.0
        qmax_mvar = float(row["max_q_mvar"]) if "max_q_mvar" in row and row["max_q_mvar"] is not None else 0.0
        qgmin_np[eg_pos] = qmin_mvar / baseMVA
        qgmax_np[eg_pos] = qmax_mvar / baseMVA

    pgmax = torch.as_tensor(pgmax_np)
    pgmin = torch.as_tensor(pgmin_np)
    qgmax = torch.as_tensor(qgmax_np)
    qgmin = torch.as_tensor(qgmin_np)

    c0 = torch.zeros(n_gen, dtype=torch.float32)
    c1 = torch.zeros(n_gen, dtype=torch.float32)
    c2 = torch.zeros(n_gen, dtype=torch.float32)

    gen_idx = 0  # track position in PYPOWER gen array
    for _, row in net.ext_grid.iterrows():
        cost_rows = net.poly_cost[
            (net.poly_cost["element"] == row.name)
            & (net.poly_cost["et"] == "ext_grid")
        ]
        if len(cost_rows) > 0:
            cr = cost_rows.iloc[0]
            c0[gen_idx] = cr["cp0_eur"]
            # per-unit pg: cost(pg_pu) = c0 + c1*baseMVA*pg_pu + c2*baseMVA^2*pg_pu^2
            c1[gen_idx] = cr["cp1_eur_per_mw"] * baseMVA
            c2[gen_idx] = cr["cp2_eur_per_mw2"] * baseMVA**2
        gen_idx += 1

    for _, row in net.gen.iterrows():
        cost_rows = net.poly_cost[
            (net.poly_cost["element"] == row.name)
            & (net.poly_cost["et"] == "gen")
        ]
        if len(cost_rows) > 0:
            cr = cost_rows.iloc[0]
            c0[gen_idx] = cr["cp0_eur"]
            c1[gen_idx] = cr["cp1_eur_per_mw"] * baseMVA
            c2[gen_idx] = cr["cp2_eur_per_mw2"] * baseMVA**2
        gen_idx += 1

    r = branch[:, 2]
    x = branch[:, 3]
    b_ch = branch[:, 4]
    tap_raw = branch[:, 8].copy()
    shift_deg = branch[:, 9]

    # tap != 0 and (tap != 1 or phase shift != 0) marks a MATPOWER transformer.
    is_trafo = (tap_raw != 0.0) & (
        (tap_raw != 1.0) | (shift_deg != 0.0)
    )

    tap = tap_raw.copy()
    tap[tap == 0.0] = 1.0
    shift_rad = np.deg2rad(shift_deg)

    z = r + 1j * x
    ys = 1.0 / z

    tap_c = tap * np.exp(1j * shift_rad)

    yff = (ys + 1j * b_ch / 2) / (np.abs(tap_c) ** 2)
    yft = -ys / np.conj(tap_c)
    ytf = -ys / tap_c
    ytt = ys + 1j * b_ch / 2

    gff = torch.as_tensor(yff.real, dtype=torch.float32)
    bff = torch.as_tensor(yff.imag, dtype=torch.float32)
    gft = torch.as_tensor(yft.real, dtype=torch.float32)
    bft = torch.as_tensor(yft.imag, dtype=torch.float32)
    gtf = torch.as_tensor(ytf.real, dtype=torch.float32)
    btf = torch.as_tensor(ytf.imag, dtype=torch.float32)
    gtt = torch.as_tensor(ytt.real, dtype=torch.float32)
    btt = torch.as_tensor(ytt.imag, dtype=torch.float32)

    # pandapower's ppc puts the tap on the "to" bus; flip transformer rows to MATPOWER's convention.
    fbus = branch[:, 0].astype(int).copy()
    tbus = branch[:, 1].astype(int).copy()
    for k in range(n_branch):
        if is_trafo[k]:
            fbus[k], tbus[k] = tbus[k], fbus[k]
    bus_fr = torch.as_tensor(fbus, dtype=torch.long)
    bus_to = torch.as_tensor(tbus, dtype=torch.long)

    # thermal limits (per-unit, 0 = unconstrained -> large value)
    rate_a = branch[:, 5]
    smax_np = rate_a / baseMVA
    smax_np[smax_np <= 0] = 9999.0
    smax_np[smax_np > 9998.0] = 9999.0
    smax = torch.as_tensor(smax_np, dtype=torch.float32)

    angmin_deg = branch[:, 11]
    angmax_deg = branch[:, 12]
    # -360/360 means unconstrained
    angmin_deg[angmin_deg <= -360] = -360
    angmax_deg[angmax_deg >= 360] = 360
    dvamin = torch.as_tensor(np.deg2rad(angmin_deg), dtype=torch.float32)
    dvamax = torch.as_tensor(np.deg2rad(angmax_deg), dtype=torch.float32)

    arcs_fr: dict[int, list[int]] = defaultdict(list)
    arcs_to: dict[int, list[int]] = defaultdict(list)
    for k in range(n_branch):
        arcs_fr[fbus[k]].append(k)
        arcs_to[tbus[k]].append(k)

    bus_arcs_fr_lists = [arcs_fr.get(i, []) for i in range(n_bus)]
    bus_arcs_to_lists = [arcs_to.get(i, []) for i in range(n_bus)]

    gens_at_bus: dict[int, list[int]] = defaultdict(list)
    for g in range(n_gen):
        gens_at_bus[gen_buses[g]].append(g)
    bus_gens_lists = [gens_at_bus.get(i, []) for i in range(n_bus)]

    loads_at_bus: dict[int, list[int]] = defaultdict(list)
    for l_idx, lb in enumerate(load_buses):
        loads_at_bus[int(lb)].append(l_idx)
    bus_loads_lists = [loads_at_bus.get(i, []) for i in range(n_bus)]

    bus_arcs_fr_t = _pad_ragged(bus_arcs_fr_lists, pad_value=n_branch)
    bus_arcs_to_t = _pad_ragged(bus_arcs_to_lists, pad_value=n_branch)
    bus_gens_t = _pad_ragged(bus_gens_lists, pad_value=n_gen)
    bus_loads_t = _pad_ragged(bus_loads_lists, pad_value=n_load)

    n_ext_grid = len(net.ext_grid)
    ext_grid_bus_set = set(int(b) for b in gen_buses[:n_ext_grid])
    is_ext_grid_bus = torch.tensor(
        [i in ext_grid_bus_set for i in range(n_bus)], dtype=torch.bool
    )

    pd_nominal = torch.as_tensor(
        net.load.p_mw.values / baseMVA, dtype=torch.float32
    )
    qd_nominal = torch.as_tensor(
        net.load.q_mvar.values / baseMVA, dtype=torch.float32
    )

    import json

    coords = np.zeros((n_bus, 2), dtype=np.float32)
    geo_series = net.bus["geo"].tolist() if "geo" in net.bus.columns else []
    for i, g in enumerate(geo_series):
        if g:
            try:
                coords[i] = json.loads(g)["coordinates"]
            except (TypeError, ValueError, KeyError):
                coords[i] = (np.nan, np.nan)
        else:
            coords[i] = (np.nan, np.nan)
    bus_xy = torch.as_tensor(coords, dtype=torch.float32)

    # Machine base (MBASE) per generator; NaN entries fall back to baseMVA.
    gen_mbase = gen[:, 6].astype("float32")
    gen_mbase[np.isnan(gen_mbase)] = float(baseMVA)
    gen_mbase[gen_mbase <= 0.0] = float(baseMVA)
    genbase = torch.as_tensor(gen_mbase)

    data = {
        "N": torch.tensor(n_bus),
        "G": torch.tensor(n_gen),
        "L": torch.tensor(n_load),
        "E": torch.tensor(n_branch),
        "baseMVA": torch.tensor(float(baseMVA)),
        "genbase": genbase,
        "vmin": vmin,
        "vmax": vmax,
        "gs": gs,
        "bs": bs,
        # base-case PF warm start
        "vm_init": vm_init,
        "va_init": va_init,
        "pgmin": pgmin,
        "pgmax": pgmax,
        "qgmin": qgmin,
        "qgmax": qgmax,
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "gff": gff,
        "gft": gft,
        "gtf": gtf,
        "gtt": gtt,
        "bff": bff,
        "bft": bft,
        "btf": btf,
        "btt": btt,
        "bus_fr": bus_fr,
        "bus_to": bus_to,
        "smax": smax,
        "dvamin": dvamin,
        "dvamax": dvamax,
        "bus_arcs_fr": bus_arcs_fr_t,
        "bus_arcs_to": bus_arcs_to_t,
        "bus_gens": bus_gens_t,
        "bus_loads": bus_loads_t,
        "pd_nominal": pd_nominal,
        "qd_nominal": qd_nominal,
        "bus_xy": bus_xy,
        "is_ext_grid_bus": is_ext_grid_bus,
        "is_trafo": torch.as_tensor(is_trafo, dtype=torch.bool),
    }
    return data
