# Curvature sweep (float64)

curvature_warp at 8 curvature levels (kappa 0 to 1e6) x 9 arms x 10 seeds, 2000 epochs, 512 eval queries, float64, x86 CPU nodes. Generated and submitted with `scripts/bp_campaign/driver.py`.

## Configurations

- ENFORCE v4 ran with the configuration in `configs/enforce_v4.json` (sha256 in `configs/MANIFEST.sha256`).
- ALM, ALM+Bolt-On, DC3, FSNet, SnareNet and PAL ran with the BO winners in `results/2026-07-26_bo_tuned_table/winners/` (PAL at tau 1e-4, 1e-2 and 1 layered on top of the PAL winner).

## Files

- `campaign_results.json`: all 9 arms, output of `scripts/analyze_curvature.py --campaign`.
- `pal_tau_1e2_1e4/campaign_results.json`: PAL at tau 1e2 and 1e4.
- `dc3_completion_diag/campaign_results_newton_{lm,yf}.json`: DC3 with Newton completion.

Render the table with `scripts/bp_campaign/render_target_table.py campaign_results.json`.
