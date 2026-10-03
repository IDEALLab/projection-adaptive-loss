# ENFORCE v1.0.4

ENFORCE (Lastrucci & Schweidtmann), vendored from
`https://github.com/process-intelligence-research/ENFORCE`, tag `v1.0.4`, commit
`acb51bb0bba00a3924d82c9b13d259193a53e507` (see `UPSTREAM_SHA`). MIT license,
copied to `upstream/LICENSE`.

`upstream/` holds the contents of upstream `enforce/core/` (`config.py`,
`model.py`, `fb_inequality_constraints.py`, `__init__.py`). The data loaders,
training engines, benchmark problems and scripts are not vendored.

Changes to the vendored files:

- Three absolute `enforce.core.*` imports in `__init__.py` and `model.py` are
  rewritten as package-relative imports.
- In `model.py`, the eigendecomposition fallback of `projection_tensors`
  (used when the Cholesky factorisation of the Gram matrix fails) adds
  `1e-6 * I` and runs in float64. Without it, CPU LAPACK fails on the
  rank-deficient e3 Jacobian. The Cholesky path is unchanged.

The adapter `solver.py` (method key `enforce_v4`) drives the vendored module's
`forward`, `compute_dc_dy`, `project` and `ada_np` from a self-supervised
training loop. Inequality benchmarks use `FischerBurmeisterReformulation`, so
the constraint vector is the equality residuals followed by the FB rows. The
FB duals stay inside the adapter. `predict()` returns the AdaNP-projected
output as `post`. s5_overdetermined is not applicable because upstream
`check_system` rejects more constraints than outputs.
