"""kappa-half per arm from campaign_results.json.

kappa-half = first kappa in grid order where mean feasibility over completed seeds
drops below 0.5. A cell with 0 completed seeds is flagged [all-diverged] and triggers it.
"""
import json
import sys

KAPPA = {"k0": 0, "k4": 1, "k6": 10, "k8": 100, "k10": 1e3, "k11": 1e4,
         "k12": 1e5, "k13": 1e6}
ORDER = ["k0", "k4", "k6", "k8", "k10", "k11", "k12", "k13"]


def main(path: str) -> None:
    data = json.load(open(path))
    by_arm: dict[str, dict[str, dict]] = {}
    for c in data["cells"]:
        by_arm.setdefault(c["arm"], {})[c["variant"]] = c
    for arm in sorted(by_arm):
        row = by_arm[arm]
        hit = None
        for v in ORDER:
            c = row.get(v)
            if c is None:
                hit = (v, "cell absent from results")
                break
            feas = c["metrics"]["feas"]
            n, mean = feas["n"], feas["mean"]
            if n == 0 or mean is None:
                hit = (v, f"all {c['n_expected']} seeds diverged [all-diverged]")
                break
            if mean < 0.5:
                vals = [round(x, 3) for x in feas["values"].values() if x is not None]
                hit = (v, f"mean={mean:.3f} std={feas['std']:.3f} "
                          f"n={n}/{c['n_expected']} per-seed={vals}")
                break
        if hit is None:
            print(f"{arm:24s} kappa-half > 1e6 (never <50% on grid)")
        else:
            v, why = hit
            print(f"{arm:24s} kappa-half = {KAPPA[v]:g} ({v})  [{why}]")


if __name__ == "__main__":
    main(sys.argv[1])
