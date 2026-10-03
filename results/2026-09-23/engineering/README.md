# E3 (e3/acopf_ieee57) baseline rows and E1/E2 ENFORCE v4 rows

Protocol (e3): `--protocol paper-faithful`, 200x2 backbone, 2000 epochs, batch 200, seeds 0-9, objective divided by `bench.objective_scale` in every training loss, 64-query zeta-zero eval set, tol 1e-3, x86 CPU node with 8 threads. Launch `slurm/engineering/e3_baselines_x86.sbatch`, evaluate `slurm/engineering/e3_baselines_eval_x86.sbatch`, tabulate `scripts/e3_paper_table.py`. The first four rows come from the 10-seed PAL/DC3/ALM runs launched with `slurm/engineering/e3_gpu.sbatch` and `slurm/engineering/e3_gpu_alm_fill.sbatch`, tabulated by the same script. Metrics are computed after each method's own repair step and reported as mean (std) over seeds. Viol max is the per-query maximum over all 576 constraint rows, averaged over the 64 queries. Per-seed values for the v5 rows: `per_seed_metrics.csv` (rows with `config` set).

| Method | Seeds | Feasibility | Viol max | Max eq | Mean eq | Max ineq | Mean ineq | Obj |
|---|---|---|---|---|---|---|---|---|
| pal_loggap | 10 | 0.825 (0.072) | n/a | 2.318e-03 (1.1e-03) | 1.477e-04 (7.9e-05) | 1.318e-03 (1.0e-03) | 4.894e-06 (3.2e-06) | 4.277e+04 (4.6e+02) |
| dc3 | 10 | 0.708 (0.047) | n/a | 3.200e-03 (2.2e-03) | 4.315e-05 (3.0e-05) | 3.993e-03 (1.7e-03) | 2.098e-05 (1.2e-05) | 4.339e+04 (6.2e+02) |
| alm_bolton | 10 | 0.000 (0.000) | n/a | 1.854e-01 (2.9e-01) | 2.244e-02 (3.7e-02) | 3.357e-01 (4.8e-01) | 3.392e-03 (4.9e-03) | 4.520e+04 (2.6e+03) |
| alm | 10 | 0.000 (0.000) | n/a | 1.100e+01 (7.3e+00) | 1.402e+00 (4.5e-01) | 0.000e+00 (0.0e+00) | 0.000e+00 (0.0e+00) | 3.597e+04 (2.0e+04) |
| fsnet | 10 | 0.000 (0.000) | 1.377e+15 (4.4e+15) | 1.693e+06 (5.3e+06) | 5.498e+04 (1.7e+05) | 1.377e+15 (4.4e+15) | 4.067e+12 (1.3e+13) | 1.151e+04 (4.6e+03) |
| enforce_v4 (gate on) | 10 (10 div.) | 0.025 (0.024) | n/a | n/a | n/a | n/a | n/a | n/a |
| enforce_v4 (gate off) | 10 (10 div.) | 0.000 (0.000) | n/a | n/a | n/a | n/a | n/a | n/a |
| snarenet | 10 (10 div.) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

fsnet per-seed viol max has a median over seeds of 1.20. enforce_v4 (gate on) finite queries per seed (seeds 0-9, of 64): 51, 52, 52, 62, 53, 55, 53, 60, 56, 58, with 16 of 640 queries feasible. Obj for the four existing rows is the raw generation cost.

## Combined violation and training wall-clock (e3)

| Method | Combined viol (mean +/- std over seeds) | Median over seeds | Mean training wall-clock |
|---|---|---|---|
| fsnet (cap 50) | 1.38e+15 +/- 4.35e+15 | 1.20 | 3.48 +/- 0.39 h (2000 epochs) |
| enforce_v4 (gate on) | undefined (non-finite on 2-13 queries per seed) | n/a | 8.93 +/- 0.85 h (2000 epochs) |
| enforce_v4 (gate off) | undefined (no trained model) | n/a | 41 +/- 13 s to failure (epoch 1) |
| snarenet | undefined (no trained model) | n/a | 26 +/- 25 min to failure (epochs 6-252) |

## ENFORCE v4 on E1 (e1/bwb) and E2 (e2/urban_wind)

10 seeds x 2 arms (warm-up, no warm-up) on a 4-GPU node with 96 GB per GPU, launched with `slurm/engineering/enforce_v4_final_gpu.sbatch`. Every seed of both arms on both benches ran out of GPU memory in the constraint-Jacobian build of the projection except E1 warm-up seed 1, which finished training and was not evaluated. Engineering-figure cells for both benches: Obj n/a, Feas n/a, Mem >96 GB. Per-seed state, OOM flag and last-logged metrics: `per_seed_metrics.csv` (rows with `arm` set).
