"""Benchmark id -> factory lookup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pal.benchmarks.base import Benchmark

_KNOWN_COSTS = {"cheap", "mid", "expensive"}
_KNOWN_DEVICES = {"cpu", "gpu"}


@dataclass(frozen=True)
class _Entry:
    factory: Callable[..., Benchmark]
    cost: str
    device: str


_REGISTRY: dict[str, _Entry] = {}


def register(
    bench_id: str,
    factory: Callable[..., Benchmark],
    *,
    cost: str,
    device: str,
) -> None:
    if bench_id in _REGISTRY:
        raise ValueError(f"benchmark '{bench_id}' already registered")
    if cost not in _KNOWN_COSTS:
        raise ValueError(f"unknown cost tier '{cost}'; expected one of {sorted(_KNOWN_COSTS)}")
    if device not in _KNOWN_DEVICES:
        raise ValueError(f"unknown device '{device}'; expected one of {sorted(_KNOWN_DEVICES)}")
    _REGISTRY[bench_id] = _Entry(factory=factory, cost=cost, device=device)


def get(bench_id: str, *, device: str | None = None) -> Benchmark:
    if bench_id not in _REGISTRY:
        raise KeyError(
            f"unknown benchmark '{bench_id}'. available: "
            f"{', '.join(sorted(_REGISTRY.keys()))}"
        )
    entry = _REGISTRY[bench_id]
    if device is None:
        return entry.factory()
    try:
        return entry.factory(device=device)
    except TypeError:
        return entry.factory()


def list_all() -> list[str]:
    return sorted(_REGISTRY.keys())


def list_filtered(
    costs: list[str] | None = None,
    devices: list[str] | None = None,
) -> list[str]:
    """Bench ids matching all given filters. `None` means 'no filter'."""
    out = []
    for bench_id, entry in _REGISTRY.items():
        if costs and entry.cost not in costs:
            continue
        if devices and entry.device not in devices:
            continue
        out.append(bench_id)
    return sorted(out)


def _bootstrap() -> None:
    from pal.benchmarks.synthetic.rosenbrock_eq import RosenbrockEq
    from pal.benchmarks.synthetic.two_basins import TwoBasins
    from pal.benchmarks.synthetic.equality_dominated import EqualityDominated
    from pal.benchmarks.synthetic.s1_sphere_track import S1SphereTrack
    from pal.benchmarks.synthetic.s2_active_set_switch import S2ActiveSetSwitch
    from pal.benchmarks.synthetic.s3_illcond_tube import S3IllcondTube
    from pal.benchmarks.synthetic.s4_qv_coupling import S4QvCoupling
    from pal.benchmarks.synthetic.s5_overdetermined import S5Overdetermined
    from pal.benchmarks.synthetic.s6_redundant_ineq import S6RedundantIneq
    from pal.benchmarks.synthetic.curvature_hinge import (
        VARIANTS as CURVATURE_HINGE_VARIANTS,
    )
    from pal.benchmarks.synthetic.curvature_hinge import CurvatureHinge
    from pal.benchmarks.synthetic.curvature_sine import (
        VARIANTS as CURVATURE_SINE_VARIANTS,
    )
    from pal.benchmarks.synthetic.curvature_sine import CurvatureSine
    from pal.benchmarks.synthetic.curvature_warp import (
        VARIANTS as CURVATURE_WARP_VARIANTS,
    )
    from pal.benchmarks.synthetic.curvature_warp import CurvatureWarp

    def _curvature_hinge_factory(variant: str) -> Callable[[], Benchmark]:
        def factory() -> Benchmark:
            return CurvatureHinge(variant=variant)
        return factory

    def _curvature_sine_factory(variant: str) -> Callable[[], Benchmark]:
        def factory() -> Benchmark:
            return CurvatureSine(variant=variant)
        return factory

    def _curvature_warp_factory(variant: str) -> Callable[[], Benchmark]:
        def factory() -> Benchmark:
            return CurvatureWarp(variant=variant)
        return factory

    def _e1_factory(device: str | None = None) -> Benchmark:
        import os

        from pal.benchmarks.engineering.e1_bwb import E1BWB
        from pal.benchmarks.engineering.e1_bwb.spec import N_STATIONS_DEFAULT

        n = int(os.environ.get("E1_N_STATIONS", N_STATIONS_DEFAULT))
        return E1BWB(device=device, n_stations=n)

    def _e2_factory(device: str | None = None) -> Benchmark:
        from pal.benchmarks.engineering.e2_urban_wind import E2UrbanWind
        return E2UrbanWind(device=device) if device is not None else E2UrbanWind()

    def _e3_ieee30_factory() -> Benchmark:
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF
        return E3ACOPF(case="ieee30")

    def _e3_ieee57_factory() -> Benchmark:
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF
        return E3ACOPF(case="ieee57")

    def _e3_ieee118_factory() -> Benchmark:
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF
        return E3ACOPF(case="ieee118")

    def _e4_factory() -> Benchmark:
        from pal.benchmarks.engineering.e4_chip_layout import E4ChipLayout
        return E4ChipLayout()

    register("rosenbrock_eq", RosenbrockEq, cost="cheap", device="cpu")
    register("two_basins", TwoBasins, cost="cheap", device="cpu")
    register("equality_dominated", EqualityDominated, cost="cheap", device="cpu")
    register("s1_sphere_track", S1SphereTrack, cost="cheap", device="cpu")
    register("s2_active_set_switch", S2ActiveSetSwitch, cost="cheap", device="cpu")
    register("s3_illcond_tube", S3IllcondTube, cost="cheap", device="cpu")
    register("s4_qv_coupling", S4QvCoupling, cost="cheap", device="cpu")
    register("s5_overdetermined", S5Overdetermined, cost="cheap", device="cpu")
    register("s6_redundant_ineq", S6RedundantIneq, cost="mid", device="cpu")
    for _curvature_hinge_variant in CURVATURE_HINGE_VARIANTS:
        register(
            f"curvature_hinge_{_curvature_hinge_variant}",
            _curvature_hinge_factory(_curvature_hinge_variant),
            cost="cheap",
            device="cpu",
        )
    for _curvature_sine_variant in CURVATURE_SINE_VARIANTS:
        register(
            f"curvature_sine_{_curvature_sine_variant}",
            _curvature_sine_factory(_curvature_sine_variant),
            cost="cheap",
            device="cpu",
        )
    # curvature_warp adds the high-curvature extension k11 ... k13 (kappa = 1e4, 1e5, 1e6).
    for _curvature_warp_variant in CURVATURE_WARP_VARIANTS:
        register(
            f"curvature_warp_{_curvature_warp_variant}",
            _curvature_warp_factory(_curvature_warp_variant),
            cost="cheap",
            device="cpu",
        )
    register("e1/bwb", _e1_factory, cost="expensive", device="gpu")
    register("e2/urban_wind", _e2_factory, cost="expensive", device="gpu")
    register("e3/acopf_ieee30", _e3_ieee30_factory, cost="mid", device="gpu")
    register("e3/acopf_ieee57", _e3_ieee57_factory, cost="mid", device="gpu")
    register("e3/acopf_ieee118", _e3_ieee118_factory, cost="mid", device="gpu")
    register("e4/chip_layout", _e4_factory, cost="cheap", device="cpu")


_bootstrap()
