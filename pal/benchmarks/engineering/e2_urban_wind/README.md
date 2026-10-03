# e2/urban_wind: urban layout under a pedestrian wind constraint

Place 10 buildings on a 750 x 750 m plot to maximise built volume. The wind
field over the site is predicted by WinDiNet, a video diffusion surrogate
(fine-tuned LTX-Video) that maps a rasterised footprint and inlet speed to 2D
velocity fields. The surrogate code is vendored under `_vendor/` and the
weights are downloaded with `python scripts/download_artifacts.py --benchmark
e2`.

| | |
|---|---|
| Design vector | 50: per building centre x, y and raw width, depth, height (decoded with softplus plus a minimum) |
| Conditions | none (zeta dim 16) |
| Objective | negative total volume divided by 1e6 m^3 |
| Constraints | 4 inequalities (site boundary, 10 m clearance between footprints, 250 m height cap, time-averaged wind speed below 15 m/s) and 1 equality (ground coverage 0.5) |
| Tolerance | 1e-4 |

`E2_N_BUILDINGS=20` selects a 100-dimensional variant (`city_20.yaml`).
The surrogate needs a GPU. On a 24 GB card run in bf16 (`E2_DTYPE=bf16`)
with a small batch.

```bash
pal run --method pal_loggap --benchmarks e2/urban_wind --seeds 0 --device cuda
```

The WinDiNet weights are derived from LTX-Video and are governed by the
LTX-Video open-weights license (see `THIRD_PARTY_NOTICES.md`).
