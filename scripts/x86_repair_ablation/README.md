# Repair-step ablation

Compares three repair steps inside PAL training: the default Levenberg-Marquardt
projector (`pal_loggap`), an SQP step (`pal_sqp`) and an interior-point step
(`pal_ip`, run with `--set proj_ip_mu0=0.01`). Grid: 3 methods x S1-S6
(`s1`-`s6`) x seeds 0-9 = 180 CPU runs at 2000 epochs, one Slurm array task
per run. Feeds the repair-variants table of the paper.

`pal_sqp` needs `qpsolvers`, `osqp>=1.0` and `clarabel` in the environment.

```bash
# set REPO_ROOT, the module line and RUNS_ROOT in launch.sbatch, then
PAL_VENV=<venv>/bin/activate sbatch scripts/x86_repair_ablation/launch.sbatch
python scripts/x86_repair_ablation/aggregate.py --campaign-root <RUNS_ROOT>
```

`aggregate.py` prints the table below only when all 180 runs finished with
status `ok`, and a coverage report (exit code 1) otherwise.

## Results (mean +/- std over 10 seeds)

| bench | pal_loggap obj_post | pal_loggap viol_post | pal_loggap feas | pal_loggap inf_iters(mean) | pal_sqp obj_post | pal_sqp viol_post | pal_sqp feas | pal_sqp inf_iters(mean) | pal_ip obj_post | pal_ip viol_post | pal_ip feas | pal_ip inf_iters(mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_sphere_track | 1.675e-02 +/- 2.4e-02 | 4.530e-07 +/- 7.5e-08 | 1.0000 +/- 0.0000 | 2.00 | 6.619e-03 +/- 1.0e-02 | 5.484e-07 +/- 1.6e-07 | 1.0000 +/- 0.0000 | 2.00 | 7.488e-03 +/- 1.3e-02 | 5.960e-07 +/- 1.3e-07 | 1.0000 +/- 0.0000 | 2.60 |
| s2_active_set_switch | 2.895e-04 +/- 1.1e-04 | 1.669e-07 +/- 6.2e-08 | 1.0000 +/- 0.0000 | 1.00 | 4.171e-04 +/- 2.0e-04 | 1.431e-07 +/- 4.2e-08 | 1.0000 +/- 0.0000 | 1.00 | 2.083e-02 +/- 2.3e-03 | 5.722e-07 +/- 1.9e-07 | 1.0000 +/- 0.0000 | 2.00 |
| s3_illcond_tube | 1.178e-03 +/- 9.7e-04 | 1.073e-07 +/- 2.1e-08 | 1.0000 +/- 0.0000 | 1.10 | 6.640e-04 +/- 2.5e-04 | 1.412e-07 +/- 7.8e-08 | 1.0000 +/- 0.0000 | 1.00 | 8.749e-04 +/- 6.2e-04 | 6.221e-04 +/- 1.8e-04 | 0.8578 +/- 0.0457 | 10.00 |
| s4_qv_coupling | 9.654e-03 +/- 3.5e-03 | 5.364e-07 +/- 7.9e-08 | 1.0000 +/- 0.0000 | 2.20 | 8.827e-03 +/- 2.4e-03 | 5.603e-07 +/- 1.1e-07 | 1.0000 +/- 0.0000 | 2.00 | 5.274e-02 +/- 1.9e-02 | 4.888e-07 +/- 3.8e-08 | 1.0000 +/- 0.0000 | 3.00 |
| s5_overdetermined | 8.872e-06 +/- 1.9e-05 | 4.392e-07 +/- 1.8e-07 | 1.0000 +/- 0.0000 | 4.80 | 1.092e-05 +/- 2.0e-05 | 4.262e-07 +/- 1.5e-07 | 1.0000 +/- 0.0000 | 4.70 | 1.254e-05 +/- 1.6e-05 | 3.111e-07 +/- 2.0e-07 | 1.0000 +/- 0.0000 | 4.30 |
| s6_redundant_ineq | 7.747e-02 +/- 8.9e-03 | 3.338e-07 +/- 2.8e-07 | 1.0000 +/- 0.0000 | 2.70 | 6.799e-02 +/- 8.1e-03 | 2.444e-07 +/- 2.4e-07 | 1.0000 +/- 0.0000 | 1.00 | 2.104e-01 +/- 1.9e-02 | 1.729e-07 +/- 1.5e-07 | 1.0000 +/- 0.0000 | 3.00 |

`inf_iters(mean)` is the mean over seeds of each run's median number of inference repair iterations.
