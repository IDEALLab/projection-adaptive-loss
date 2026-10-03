"""Offline probe: DC3 generic-Newton completion on curvature_warp for random in-box Z,
per-sample convergence under damping settings (B=1 so the 1e3 gate is per-sample)."""
import sys, json, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import scripts.analyze_curvature as A
from pal.baselines.dc3._completion import CompletionDivergedError
torch.set_default_dtype(torch.float64)
bench_id = sys.argv[1]; N = int(sys.argv[2]); settings = json.loads(sys.argv[3])
bench = A.get_benchmark(bench_id, device="cpu")
torch.set_default_dtype(torch.float32); q = bench.eval_queries(0, n=N); torch.set_default_dtype(torch.float64)
X = q.conditions.double()
g = torch.Generator().manual_seed(0)
for st in settings:
    hp = {"completion_strategy_override": "generic_newton", **st}
    cfg = A._build_cfg_from_hparams("dc3", hp); cfg.device = "cpu"
    solver = A._build_solver("dc3", cfg, bench_id=bench_id)
    spec, use_compl, pv, ov, dim_out, bounds = solver._resolve(bench)
    shim = solver._build_shim(bench, spec, pv, ov, "cpu")
    lo = torch.tensor([b[0] for b in bounds]); hi = torch.tensor([b[1] for b in bounds])
    Z = lo + (hi - lo) * torch.rand(N, len(pv), generator=g, dtype=torch.float64)
    ok = div = 0; res = []
    for i in range(N):
        try:
            Y = shim.complete_partial(X[i:i+1], Z[i:i+1])
            r = bench.constraints(Y, X[i:i+1]).abs().max().item(); res.append(r); ok += r < 1e-4
        except CompletionDivergedError:
            div += 1
    res = torch.tensor(res) if res else torch.tensor([float("nan")])
    print(f"{st}: partial={pv} other={ov} conv(<1e-4)={ok}/{N} diverged(gate)={div}/{N} "
          f"resid med={res.median():.1e} p90={res.quantile(0.9) if res.numel()>1 else res[0]:.1e}")
