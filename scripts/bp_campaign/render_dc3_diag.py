"""Combine frozen dc3 (main campaign json) with the diagnostic-arm jsons into
one feasibility / gap / seeds table."""
import json, sys
KAPPA = ["k0","k4","k6","k8","k10","k11","k12","k13"]
def load(path):
    d = json.load(open(path)); return {(c["arm"], c["variant"]): c for c in d["cells"]}
cells = {}
for p in sys.argv[1:]:
    cells.update(load(p))
arms = [a for a in ["dc3", "dc3[newton]", "dc3[newton,yf=1]", "dc3[newton,lm=0.1]"] if any(a == k[0] for k in cells)]
def fmt(c, m, f):
    if c is None or c["n_completed"] == 0: return "DIV(0)" if c is not None and c.get("skipped") else "-"
    x = c["metrics"].get(m)
    if not x or x.get("mean") is None: return "NA"
    s = f.format(x["mean"]) + ("+/-" + f.format(x["std"]) if x.get("n",0) > 1 and x.get("std") is not None else "")
    if c["n_completed"] < c["n_expected"]: s += f" (n={c['n_completed']})"
    return s
for m, title, f in [("feas","feas","{:.3f}"),("gap","gap (feasible-only)","{:+.2e}"),("n_feas","n_feas runs","{:.0f}")]:
    print(f"\n### {title}\n\n| arm | " + " | ".join(KAPPA) + " |\n|---|" + "---|"*8)
    for a in arms:
        print(f"| {a} | " + " | ".join(fmt(cells.get((a,k)), m, f) for k in KAPPA) + " |")
print("\n### completed seeds\n\n| arm | " + " | ".join(KAPPA) + " |\n|---|" + "---|"*8)
for a in arms:
    print(f"| {a} | " + " | ".join(f"{cells[(a,k)]['n_completed']}/10" if (a,k) in cells else "-" for k in KAPPA) + " |")
