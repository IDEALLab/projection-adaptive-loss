# FSNet

FSNet (Nguyen & Donti 2025, arXiv:2506.00362), vendored from
`https://github.com/MOSSLab-MIT/FSNet` @ `911fcb2fdfa7488ac245ef9de535acc2a250915f` (see
`UPSTREAM_SHA`). MIT license, copied to `upstream/LICENSE`. Hyperparameters are
in `pal/baselines/hparams/fsnet.yaml` with the source of each value.

Changes to the vendored files: `upstream/utils/lbfgs.py` rejects non-finite
trial points in the line search and stops the L-BFGS loops at the last finite
iterate. On benchmarks with fast-growing nonlinear equalities the merit `1000 * (eq^2 + ineq^2)` grows faster
than quadratically in `|y|` and overflows float32 within a few iterations.
The checks have no effect on finite runs.

Adapter (`solver.py`):

- The MLP input is `[zeta; conditions]`, since unconditional benchmarks use
  `zeta` as their only source of variation.
- The output uses a sigmoid and is rescaled to the box `[L, U]`, as upstream.
  Box violations are part of the inequality residual, so L-BFGS cannot leave
  the box unpenalised.
- 100 epochs (paper Table C.1) with `steps_per_epoch=20`, which matches the
  number of optimizer steps upstream takes on a 10k-sample dataset.
- Steps with a non-finite loss or gradient norm are skipped.
- `final_x_on_eval` is the post-L-BFGS output. The pre-L-BFGS output is stored
  in `extras["final_x_pre_lbfgs"]`.
