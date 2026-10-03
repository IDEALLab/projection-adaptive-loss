"""Table-1 metrics and the BO objective, computed from per-run records.

S = L1 - EPS1*L2 - EPS2*L3 with L1 the mean feasibility, L2 the clipped
normalized objective gap and L3 the violation score; diverged seeds score
worst-case. Selection uses the lexicographic tuple (L1 desc, L2 asc, L3 asc).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pal.eval.table1_constants import (
    BENCH_OBJ_CONSTANTS,
    EPS1,
    EPS2,
    OBJ_FAIL_THRESHOLD,
    STRUCTURAL_EXCLUSIONS,
    TABLE1_BENCHES,
    TABLE1_METHODS,
    VIOL_TAU_C,
    VIOL_WINDOW_DECADES,
)

_RUN_COLUMNS = (
    "method",
    "benchmark_id",
    "seed",
    "status",
    "obj_mean_post",
    "feasibility_post",
    "viol_max_post",
    "n_queries",
    "tolerance",
    "wall_start",
)


@dataclass(frozen=True)
class RunRow:
    """One normalized ``(method, bench, seed)`` run record."""

    method: str
    bench: str
    seed: int | None
    status: str | None
    obj_post: float | None
    feas_post: float | None
    viol_post: float | None
    n_queries: int | None
    tolerance: float | None
    wall_start: str | float | None


@dataclass(frozen=True)
class CellMetrics:
    """Fully aggregated metrics for one ``(method, bench)`` cell.

    ``*_disp`` are the seed mean / sample-std for display; ``*_bo`` / ``l2_b`` /
    ``l3_b`` include worst-case imputation for diverged seeds.
    """

    method: str
    bench: str
    n_attempted: int
    n_ok: int
    n_diverged: int
    is_structural: bool
    is_fully_diverged: bool
    c_b: float
    feas_disp_mean: float | None
    feas_disp_std: float | None
    obj_disp_mean: float | None
    obj_disp_std: float | None
    viol_disp_mean: float | None
    viol_disp_std: float | None
    n_queries: int | None
    feas_bo: float
    obj_bo: float
    l2_b: float
    l3_b: float


@dataclass(frozen=True)
class BOObjective:
    """Aggregate objective for one method / trial over its applicable benches."""

    l1: float
    l2: float
    l3: float
    scalar: float
    n_applicable_benches: int
    n_seeds: int
    n_queries: int
    n_diverged_cells: int
    n_diverged_seeds: int
    diverged: bool

    @property
    def lex_key(self) -> tuple[float, float, float]:
        """Sort key for lexicographic argmax: L1 desc, L2 asc, L3 asc."""
        return (-self.l1, self.l2, self.l3)


def _mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    """Seed mean and sample std (n-1). None if no finite values; std 0 if n<2."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None, None
    n = len(finite)
    mean = sum(finite) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in finite) / (n - 1)
    return mean, math.sqrt(var)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _viol_score(viol: float | None, tau_c: float, window_decades: float) -> float:
    """Clamped log10 violation score in [0, 1].

    Violations at or below tau_c score 0 (feasible); at or above
    tau_c * 10**window_decades score 1; linear in log10 between.
    """
    if viol is None or not math.isfinite(viol) or viol <= tau_c or tau_c <= 0.0:
        return 0.0
    excess = math.log10(viol) - math.log10(tau_c)
    return _clip01(excess / window_decades)


def run_rows_from_records(records: Iterable[Mapping[str, object]]) -> list[RunRow]:
    """Normalize raw run dicts (parquet rows) into ``RunRow`` records.

    Missing columns become ``None``.
    """
    rows: list[RunRow] = []
    for r in records:
        rows.append(
            RunRow(
                method=r.get("method"),  # type: ignore[arg-type]
                bench=r.get("benchmark_id"),  # type: ignore[arg-type]
                seed=_as_int(r.get("seed")),
                status=r.get("status"),  # type: ignore[arg-type]
                obj_post=_as_float(r.get("obj_mean_post")),
                feas_post=_as_float(r.get("feasibility_post")),
                viol_post=_as_float(r.get("viol_max_post")),
                n_queries=_as_int(r.get("n_queries")),
                tolerance=_as_float(r.get("tolerance")),
                wall_start=_as_wall_start(r.get("wall_start")),
            )
        )
    return rows


def load_run_rows(
    parquet_path: str | Path,
    methods: Sequence[str] = TABLE1_METHODS,
    benches: Sequence[str] = TABLE1_BENCHES,
) -> list[RunRow]:
    """Read a runs parquet and return normalized rows for the given cells."""
    import polars as pl

    df = pl.read_parquet(parquet_path)
    df = df.filter(
        pl.col("method").is_in(list(methods))
        & pl.col("benchmark_id").is_in(list(benches))
    )
    have = set(df.columns)
    return run_rows_from_records(
        df.select([c for c in _RUN_COLUMNS if c in have]).to_dicts()
    )


def _as_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(x: object) -> int | None:
    if x is None:
        return None
    try:
        return int(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_wall_start(x: object) -> str | float | None:
    """Keep wall_start as a sortable value (ISO string or epoch float)."""
    if x is None:
        return None
    if isinstance(x, str):
        return x
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _is_later(a: str | float | None, b: str | float | None) -> bool:
    """True if run timestamp ``a`` is strictly later than ``b`` (None is oldest)."""
    if a is None:
        return False
    if b is None:
        return True
    try:
        return a > b  # type: ignore[operator]
    except TypeError:
        return False


def _is_ok(row: RunRow, obj_fail_threshold: float) -> bool:
    """Replicate the renderer's ok filter: status ok and obj null-or-<=threshold."""
    if row.status != "ok":
        return False
    if row.obj_post is None:
        return True
    return math.isfinite(row.obj_post) and row.obj_post <= obj_fail_threshold


def aggregate_cell(
    rows: Sequence[RunRow],
    method: str,
    bench: str,
    *,
    bench_obj_constants: Mapping[str, float] = BENCH_OBJ_CONSTANTS,
    structural_exclusions: frozenset[tuple[str, str]] = STRUCTURAL_EXCLUSIONS,
    obj_fail_threshold: float = OBJ_FAIL_THRESHOLD,
    tau_c: float = VIOL_TAU_C,
    window_decades: float = VIOL_WINDOW_DECADES,
) -> CellMetrics:
    """Aggregate all runs for one ``(method, bench)`` into a ``CellMetrics``."""
    c_b = bench_obj_constants[bench]
    is_structural = (method, bench) in structural_exclusions

    cell_rows = [r for r in rows if r.method == method and r.bench == bench]
    attempted = {r.seed for r in cell_rows}

    # Latest ok row per seed.
    clean: dict[int | None, RunRow] = {}
    for r in cell_rows:
        if not _is_ok(r, obj_fail_threshold):
            continue
        prev = clean.get(r.seed)
        if prev is None or _is_later(r.wall_start, prev.wall_start):
            clean[r.seed] = r

    n_attempted = len(attempted)
    n_ok = len(clean)
    n_diverged = n_attempted - n_ok
    is_fully_diverged = n_attempted > 0 and n_ok == 0 and not is_structural

    feas_vals: list[float] = []
    obj_vals: list[float] = []
    l3_vals: list[float] = []
    n_queries: int | None = None
    for seed in attempted:
        r = clean.get(seed)
        if r is not None:
            if r.feas_post is not None:
                feas_vals.append(r.feas_post)
            if r.obj_post is not None:
                obj_vals.append(r.obj_post)
            l3_vals.append(_viol_score(r.viol_post, tau_c, window_decades))
            if r.n_queries is not None:
                n_queries = r.n_queries if n_queries is None else max(n_queries, r.n_queries)
        else:
            feas_vals.append(0.0)
            obj_vals.append(c_b)
            l3_vals.append(1.0)

    feas_disp_mean, feas_disp_std = _mean_std(feas_vals)
    obj_disp_mean, obj_disp_std = _mean_std(obj_vals)
    viol_disp_mean, viol_disp_std = _mean_std(
        [r.viol_post for r in clean.values()]
    )

    feas_bo = feas_disp_mean if feas_disp_mean is not None else 0.0
    obj_bo = obj_disp_mean if obj_disp_mean is not None else c_b
    l2_b = _clip01(obj_bo / c_b) if c_b > 0.0 else 0.0
    l3_b = sum(l3_vals) / len(l3_vals) if l3_vals else 1.0

    return CellMetrics(
        method=method,
        bench=bench,
        n_attempted=n_attempted,
        n_ok=n_ok,
        n_diverged=n_diverged,
        is_structural=is_structural,
        is_fully_diverged=is_fully_diverged,
        c_b=c_b,
        feas_disp_mean=feas_disp_mean,
        feas_disp_std=feas_disp_std,
        obj_disp_mean=obj_disp_mean,
        obj_disp_std=obj_disp_std,
        viol_disp_mean=viol_disp_mean,
        viol_disp_std=viol_disp_std,
        n_queries=n_queries,
        feas_bo=feas_bo,
        obj_bo=obj_bo,
        l2_b=l2_b,
        l3_b=l3_b,
    )


def aggregate_cells(
    rows: Sequence[RunRow],
    methods: Sequence[str] = TABLE1_METHODS,
    benches: Sequence[str] = TABLE1_BENCHES,
    **kwargs: object,
) -> dict[tuple[str, str], CellMetrics]:
    """Aggregate every ``(method, bench)`` in the grid into ``CellMetrics``."""
    return {
        (m, b): aggregate_cell(rows, m, b, **kwargs)  # type: ignore[arg-type]
        for m in methods
        for b in benches
    }


def applicable_benches(
    method: str,
    benches: Sequence[str] = TABLE1_BENCHES,
    structural_exclusions: frozenset[tuple[str, str]] = STRUCTURAL_EXCLUSIONS,
) -> list[str]:
    """Benches contributing to a method's aggregate (structural cells dropped)."""
    return [b for b in benches if (method, b) not in structural_exclusions]


def method_feas_mean(
    cells: Mapping[tuple[str, str], CellMetrics],
    method: str,
    benches: Sequence[str] = TABLE1_BENCHES,
) -> float:
    """Per-method Mean-row feasibility (worst-case imputed, structural dropped)."""
    bs = applicable_benches(method, benches)
    return sum(cells[(method, b)].feas_bo for b in bs) / len(bs)


def method_obj_mean(
    cells: Mapping[tuple[str, str], CellMetrics],
    method: str,
    benches: Sequence[str] = TABLE1_BENCHES,
) -> float:
    """Per-method Mean-row objective (worst-case imputed, structural dropped)."""
    bs = applicable_benches(method, benches)
    return sum(cells[(method, b)].obj_bo for b in bs) / len(bs)


def compute_bo_objective(
    cells: Mapping[tuple[str, str], CellMetrics],
    method: str,
    benches: Sequence[str] = TABLE1_BENCHES,
    *,
    eps1: float = EPS1,
    eps2: float = EPS2,
) -> BOObjective:
    """Compute (L1, L2, L3, S) and the lexicographic tuple for one method.

    Asserts ``eps1 + eps2 < 1/(q*s*b)`` for the (q, s, b) of the data.
    """
    bs = applicable_benches(method, benches)
    b = len(bs)
    if b == 0:
        raise ValueError(f"method {method!r} has no applicable benches")

    cell_list = [cells[(method, bench)] for bench in bs]
    s = max((c.n_attempted for c in cell_list), default=0)
    q_vals = [c.n_queries for c in cell_list if c.n_queries is not None]
    q = max(q_vals) if q_vals else 0

    if q > 0 and s > 0 and b > 0:
        bound = 1.0 / (q * s * b)
        assert eps1 + eps2 < bound, (
            f"tie-breaker weights eps1+eps2={eps1 + eps2:g} must be < "
            f"1/(q*s*b)=1/({q}*{s}*{b})={bound:g}"
        )

    l1 = sum(c.feas_bo for c in cell_list) / b
    l2 = sum(c.l2_b for c in cell_list) / b
    l3 = sum(c.l3_b for c in cell_list) / b
    scalar = l1 - eps1 * l2 - eps2 * l3

    n_diverged_cells = sum(1 for c in cell_list if c.is_fully_diverged)
    n_diverged_seeds = sum(c.n_diverged for c in cell_list)

    return BOObjective(
        l1=l1,
        l2=l2,
        l3=l3,
        scalar=scalar,
        n_applicable_benches=b,
        n_seeds=s,
        n_queries=q,
        n_diverged_cells=n_diverged_cells,
        n_diverged_seeds=n_diverged_seeds,
        diverged=n_diverged_seeds > 0,
    )


def select_best_trial(trials: Sequence[BOObjective]) -> int:
    """Index of the lexicographic winner (L1 desc, L2 asc, L3 asc)."""
    if not trials:
        raise ValueError("no trials to select from")
    best = 0
    for i in range(1, len(trials)):
        if trials[i].lex_key < trials[best].lex_key:
            best = i
    return best
