# IPOPT

Classical interior-point baseline via `cyipopt`. Each evaluation query is
solved independently, with optional multi-start (`--multi-start N`) where the
best start is chosen by feasibility first and objective second.

`cyipopt` is not a dependency of this package. Install it with

```bash
conda install -c conda-forge cyipopt
```

or with pip against a system IPOPT (`brew install ipopt pkg-config` or
`apt install coinor-libipopt-dev pkg-config`, then `pip install cyipopt`).
`docker/training_gpu_ipopt.Dockerfile` builds a container with IPOPT for the
GPU benchmarks, whose surrogate forward passes still run on the GPU.

```python
from pal.baselines.ipopt import IPOPTSolver, IPOPTConfig

solver = IPOPTSolver(IPOPTConfig(seed=0, multi_start=5, max_iter=500))
result = solver.train(bench, seed=0, logger=logger)
```

Constraints are passed as two-sided bounds: `cl = cu = 0` for equalities and
`cl = -inf, cu = 0` for inequalities `g(x) <= 0`. IPOPT is not batched, so
wall time scales linearly with the number of queries.
