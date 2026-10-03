# e1/bwb: blended-wing-body aircraft design

Multidisciplinary design of a battery-electric blended-wing-body aircraft.
Four neural surrogates are composed into one differentiable evaluator:
aerodynamic coefficients (MLP), surface pressure and friction (FiLM MLP),
structural section properties (conditional VAE) and the outer-mold-line SDF.
A beam model turns the section properties and aerodynamic loads into strain
and tip deflection.

| | |
|---|---|
| Design vector | 36: shape (9), span scale L (1), structure (19), battery box (6), cruise angle of attack (1) |
| Conditions | 2: altitude (m), airspeed (m/s) |
| Objective | negative electric Breguet range, `R = (eta/g) E_bat (CL/CD) (m_bat/m_total)` |
| Constraints | 1 equality (lift balance), 2 inequalities (strain aggregated over 20 Sobol stations, tip deflection below 10% of the semi-span) |

All surrogate weights ship under `_artifacts_data/` (about 11 MB) and are
registered in `pal/artifacts/registry.py`, so no download is needed.
`reproduce/` holds the scripts that rebuild the training data and retrain each
surrogate from the BlendedNet dataset (`a_aero/`, `film_surface/`, `bwb_sdf/`)
or from sampled wingbox geometries (`structural/`: `sample.py`,
`generate_data.py`, `merge_chunks.py`, `train.py`).

```bash
pal run --method pal_loggap --benchmarks e1/bwb --seeds 0 --device cuda
PYTHONPATH=. python -m pal.benchmarks.engineering.e1_bwb.studies.smoke
```
