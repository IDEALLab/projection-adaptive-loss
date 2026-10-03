#!/usr/bin/env python3
"""Post-hoc analysis of a PAL_IPOPT_TRACE directory.

Writes summary.csv, per_query.csv and convergence.csv, and prints a headline summary.

Usage:
    python scripts/analyze_ipopt_trace.py --trace <dir> [--out <dir>] \
        [--paper-tol 1e-4]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_solve(jsonl: Path) -> dict[str, Any]:
    """Parse one solve's JSONL into {iters:[...], derivs:[...], final:dict|None}."""
    iters: list[dict] = []
    derivs: list[dict] = []
    final: dict | None = None
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row.get("kind")
        if kind == "iter":
            iters.append(row)
        elif kind == "deriv":
            derivs.append(row)
        elif kind == "final":
            final = row
    return {"path": jsonl, "iters": iters, "derivs": derivs, "final": final}


def _shard_name(jsonl: Path, root: Path) -> str:
    """Shard identity = the trace subdir under root (or '.' if flat)."""
    try:
        rel = jsonl.parent.relative_to(root)
    except ValueError:
        rel = jsonl.parent
    s = str(rel)
    return s if s not in ("", ".") else "_root"


def _deb_winner(records: list[dict]) -> dict | None:
    """Deb constraint-dominance over the reported final iterate of each restart."""
    if not records:
        return None
    inf = float("inf")

    def viol(r: dict) -> float:
        v = r.get("max_viol_final")
        return inf if v is None else v

    def obj(r: dict) -> float:
        o = r.get("obj_final")
        return inf if o is None else o

    feas = [r for r in records if r.get("feasible_at_paper_tol")]
    if feas:
        return min(feas, key=obj)
    return min(records, key=lambda r: (viol(r), obj(r)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--paper-tol", type=float, default=1e-4)
    args = ap.parse_args()
    out = args.out or args.trace
    out.mkdir(parents=True, exist_ok=True)

    jsonls = sorted(args.trace.rglob("*.jsonl"))
    if not jsonls:
        raise SystemExit(f"no *.jsonl trace files under {args.trace}")

    solves = [_load_solve(p) for p in jsonls]

    summary_rows: list[dict] = []
    conv_path = out / "convergence.csv"
    with conv_path.open("w", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["query", "restart", "iter", "restoration", "obj",
                     "inf_pr", "inf_du", "mu", "d_norm", "regularization_size",
                     "alpha_pr", "alpha_du", "ls_trials", "t"])
        for s in solves:
            fin = s["final"] or {}
            shard = _shard_name(s["path"], args.trace)
            qi = fin.get("query_idx")
            ri = fin.get("restart_idx")
            if qi is None or ri is None:
                # Incomplete solve: recover ids from the filename.
                stem = s["path"].stem  # qXXXX_rYYY
                try:
                    qi = int(stem.split("_")[0][1:])
                    ri = int(stem.split("_")[1][1:])
                except (IndexError, ValueError):
                    qi, ri = -1, -1
            s["shard"] = shard
            s["qi"] = qi
            s["ri"] = ri
            for it in s["iters"]:
                cw.writerow([qi, ri, it.get("iter"), it.get("restoration"),
                             it.get("obj"), it.get("inf_pr"), it.get("inf_du"),
                             it.get("mu"), it.get("d_norm"),
                             it.get("regularization_size"), it.get("alpha_pr"),
                             it.get("alpha_du"), it.get("ls_trials"), it.get("t")])
            row = {
                "shard": shard,
                "query": qi,
                "restart": ri,
                "complete": s["final"] is not None,
                "status": fin.get("status"),
                "status_msg": fin.get("status_msg"),
                "n_iter": fin.get("n_iter", len(s["iters"])),
                "wall_s": fin.get("wall_s"),
                "obj_final": fin.get("obj_final"),
                "max_viol_final": fin.get("max_viol_final"),
                "best_max_viol": fin.get("best_max_viol"),
                "best_obj": fin.get("best_obj"),
                "feasible_at_1e-4": fin.get("feasible_at_1e-4"),
                "feasible_at_paper_tol": fin.get("feasible_at_paper_tol"),
            }
            summary_rows.append(row)

    _write_csv(out / "summary.csv", summary_rows)

    # Each query shard reuses local query_idx 0, so key on (shard, query_idx).
    by_query: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for s in solves:
        if s["final"] is not None:
            by_query[(s["shard"], s["qi"])].append(s["final"])

    per_query_rows: list[dict] = []
    for (shard, qi) in sorted(by_query):
        recs = by_query[(shard, qi)]
        w = _deb_winner(recs)
        best_viol_across = min(
            (r["best_max_viol"] for r in recs if r.get("best_max_viol") is not None),
            default=None,
        )
        per_query_rows.append({
            "shard": shard,
            "query": qi,
            "n_restarts": len(recs),
            "winner_restart": w.get("restart_idx") if w else None,
            "winner_obj": w.get("obj_final") if w else None,
            "winner_max_viol": w.get("max_viol_final") if w else None,
            "feasible_at_1e-4": bool(w and w.get("feasible_at_1e-4")),
            "feasible_at_paper_tol": bool(w and w.get("feasible_at_paper_tol")),
            "best_max_viol_any_restart": best_viol_across,
        })
    _write_csv(out / "per_query.csv", per_query_rows)

    n_solves = len(summary_rows)
    n_complete = sum(1 for r in summary_rows if r["complete"])
    status_hist = Counter(r["status"] for r in summary_rows if r["complete"])
    walls = [r["wall_s"] for r in summary_rows if r.get("wall_s") is not None]
    nq = len(per_query_rows)
    feas_1e4 = sum(1 for r in per_query_rows if r["feasible_at_1e-4"])
    feas_paper = sum(1 for r in per_query_rows if r["feasible_at_paper_tol"])

    print(f"trace dir:            {args.trace}")
    print(f"solves (files):       {n_solves}  (complete finals: {n_complete})")
    print(f"queries:              {nq}")
    if nq:
        print(f"per-query feasible@1e-4:      {feas_1e4}/{nq} "
              f"({100.0 * feas_1e4 / nq:.1f}%)")
        print(f"per-query feasible@paper_tol: {feas_paper}/{nq} "
              f"({100.0 * feas_paper / nq:.1f}%)")
    if walls:
        walls_sorted = sorted(walls)
        print(f"wall/solve (s):       mean={sum(walls) / len(walls):.1f} "
              f"median={walls_sorted[len(walls) // 2]:.1f} "
              f"min={walls_sorted[0]:.1f} max={walls_sorted[-1]:.1f}")
        print(f"total solve wall (h): {sum(walls) / 3600:.2f}")
    print(f"status histogram:     {dict(status_hist)}")
    print(f"\nwrote: {out / 'summary.csv'}, {out / 'per_query.csv'}, {conv_path}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def _fmt(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return v


if __name__ == "__main__":
    main()
