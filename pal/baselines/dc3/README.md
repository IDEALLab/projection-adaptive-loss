# DC3 (Donti, Rolnick & Kolter 2021)

Completion and correction core of DC3 (ICLR 2021, arXiv:2104.12225), vendored
from `https://github.com/locuslab/DC3` @ `35437af7f22390e4ed032d9eef90cc525764d26f` (see
`UPSTREAM_SHA`). Only `method.py` is vendored, and only `grad_steps`,
`grad_steps_all` and `total_loss` are used from it.

Changes to the vendored file: `grad_steps_all` skips the equality-residual
check when a benchmark has no equality constraints, because `torch.max` on an
empty tensor raises.

The problem-specific parts of upstream `utils.py` are reimplemented against
the generic benchmark interface:

- `_completion.py`: closed-form completion for affine equalities, Newton
  completion with Tikhonov regularisation for nonlinear ones.
- `_completion_acopf.py`: two-step ACOPF completion from paper Appendix C.3
  (Newton on voltage magnitudes and angles, then closed form for slack real
  power and all reactive powers). Slack voltage angle is a known variable.
- `_partial_grad.py`: implicit-function gradient for the correction step.
- `bench_specs.py`: partial/other variable partition per benchmark.

The backbone is the shared `CoordinationMLP` instead of upstream's BN+Dropout
MLP, and its input is `[zeta; conditions]`. Hyperparameters follow upstream
`default_args.py`: `hparams/dc3.yaml` (nonconvex tier) for all benchmarks and
`hparams/dc3_acopf.yaml` (acopf tier) for e3. Failed completions raise
`CompletionDivergedError` and the run is recorded as failed.
