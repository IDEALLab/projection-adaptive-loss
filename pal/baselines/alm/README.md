# ALM (Basir & Senocak 2023)

Port of Algorithm 3 (augmented Lagrangian with adaptive penalty updates) from
S. Basir and I. Senocak, "An adaptive augmented Lagrangian method for training
physics and equality-constrained artificial neural networks", arXiv:2306.04904.
No standalone code release exists for this paper, so this is a paper-only port.

| Paper | Code |
|---|---|
| Algorithm 3 defaults (gamma=1e-2, alpha=0.99, eps=1e-8) | `_state.py`, `hparams/alm.yaml` |
| Eq. (5) augmented Lagrangian loss | `_state.py::compute_loss` |
| Eqs. (7)-(9) EMA, penalty reset, dual update | `_state.py::update` |

Differences from the paper:

- Inequalities enter as `max(0, g_i)`, so `(mu/2) C_i^2` becomes the standard
  quadratic inequality penalty. The paper treats equalities only.
- Residuals are the raw signed constraint values. The feasibility tolerance
  bands are applied only at evaluation time, as for every other method.
- The primal update is one Adam step per epoch instead of an L-BFGS inner
  loop, and the backbone is the shared `CoordinationMLP`, so all learned
  methods use the same optimizer and network.

`bolton_solver.py` implements ALM+Bolt-On: a trained ALM model followed by the
PAL projector at inference time only.
