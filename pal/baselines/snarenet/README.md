# SnareNet

Chu, Boukas & Udell, arXiv:2602.09317. Vendored from
`https://github.com/miniyachi/SnareNet` @ `203f1363370c908e125caf0484e8efcc34ab5831` (see
`upstream/UPSTREAM_SHA`). Only `models/snarenet.py` (repair layer and wrapper)
and `models/base_model.py` (MLP backbone) are vendored.

Changes to the vendored files:

- `snarenet.py`: the repair step clamps its initial candidate and every Newton
  iterate into the output box when the benchmark defines one
  (`spec.hard_output_box`). The e1 surrogates are undefined outside the box.
  Benchmarks without a hard box run the unmodified loop.
- `base_model.py`: the number of hidden layers is read from
  `cfg.model.num_hidden_layers` (upstream hardcodes 2) so that all methods can
  share the same backbone depth.

Adapter files: `solver.py` (config and train/predict), `data_shim.py`
(benchmark to SnareNet data interface), `_adaptive_relaxation.py` (port of
upstream `utils/utils.py` adaptive relaxation), `_cfg_proxy.py`,
`_upstream_loader.py`.

Hyperparameters follow the upstream `snarenet_noncvx` experiment (epochs 2000,
batch 200, lr 1e-4, lambda 1e-2, Newton maxiter 100, rtol 1e-8, adaptive
relaxation with 500 linear decay epochs), except the backbone, which uses 4
hidden layers of width 512 for parity with the other methods. To reproduce the
paper configuration set `num_hidden_layers=2` and `hidden_size=200`.
