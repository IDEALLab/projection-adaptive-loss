# BO-tuned synthetic Table 1 (s1-s6, report seeds 0-9)

7 BO-tuned methods x 6 synthetic benchmarks (s1-s6 = S1-S6) x 10 report seeds. Winner configs: `winners/<method>/confirm_winner.json` (checksums in `winners/MANIFEST.sha256`), search ranges in `scripts/bo/search_spaces.py`. Metrics by `pal.eval.table1_metrics`: seed mean +/- sample std, Obj = post-repair objective = optimality gap (f* = 0 on every benchmark), equal per-benchmark weighting. The ENFORCE v4 column of Table 1 and its winner config are in `results/2026-09-11_enforce_v4_no_warmup/`.

## Feasibility (%)

| Bench | ALM | ALM+Bolt-On | ENFORCE | DC3 | FSNet | SnareNet | PAL |
|---|---|---|---|---|---|---|---|
| s1 | 0+/-1 | 100+/-0 | 71+/-9 | div. [d] | 100+/-0 | 100+/-0 | 100+/-0 |
| s2 | 0+/-0 | 100+/-0 | 88+/-11 | 97+/-7 | 100+/-0 | 100+/-0 | 100+/-0 |
| s3 | 0+/-0 | 100+/-0 | 73+/-16 | 100+/-0 | 100+/-0 | 100+/-0 | 100+/-0 |
| s4 | 0+/-0 | 100+/-1 | div. [d] | div. [d] | 100+/-0 | 100+/-0 | 100+/-0 |
| s5 | 0+/-0 | 100+/-0 | 30+/-18 | n/a [x] | 51+/-31 | 100+/-0 | 100+/-0 |
| s6 | 0+/-0 | 100+/-0 | div. [d] | div. [d] | 100+/-0 | 100+/-0 | 100+/-0 |
| Mean | 0 | **100** | 44 [d] | 39 [d][x] | 92 | **100** | **100** |

## Objective (optimality gap)

| Bench | ALM | ALM+Bolt-On | ENFORCE | DC3 | FSNet | SnareNet | PAL |
|---|---|---|---|---|---|---|---|
| s1 | 0.01+/-0.01* | 0.07+/-0.12 | 0.00+/-0.00 | div. [d] | 0.00+/-0.00 | 0.78+/-0.10 | 0.03+/-0.04 |
| s2 | 0.21+/-0.17* | 0.27+/-0.17 | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 | 0.16+/-0.00 | 0.00+/-0.00 |
| s3 | 0.00+/-0.00* | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 | 0.01+/-0.00 | 0.00+/-0.00 |
| s4 | 3.93+/-1.07* | 19.47+/-5.25 | div. [d] | div. [d] | 0.00+/-0.00 | 0.09+/-0.01 | 0.00+/-0.00 |
| s5 | 0.00+/-0.00* | 0.00+/-0.00 | 0.00+/-0.00* | n/a [x] | 4.41+/-2.16 | 0.00+/-0.00 | 0.00+/-0.00 |
| s6 | 0.03+/-0.00* | 0.14+/-0.03 | div. [d] | div. [d] | 0.00+/-0.00 | 0.60+/-0.06 | 0.03+/-0.01 |
| Mean | 0.70* | 3.32 | 2.08 [d]* | 4.23 [d][x]* | 0.73 | 0.27 | 0.01 |

[d] Fully diverged cell: all 10 seeds NaN-diverged or the objective exceeded 1e2. Scored worst-case (feasibility 0, objective at the per-benchmark ceiling C_b taken from `results/2026-05-04_paper_cost_table/paper_cost_table.tex`) and included in the Mean.
[x] Structurally inapplicable, excluded from the Mean: DC3 x s5 (n_eq > dim).
\* Objective measured at mostly infeasible points (feasibility below 50%), not comparable to feasible methods. ALM feasibility is 0.31% on s1 and 0.08% on the Mean.
Every non-structural cell is either 10/10 seeds clean or 10/10 seeds diverged.

## Wall-clock (indicative)

Train = mean per-run training wall time, Total = mean end-to-end per-run wall time (setup and eval included), both over non-diverged runs. n = applicable seed-runs, Div = diverged runs under the table's rule. Runs shared x86 CPU nodes without isolation, so timings are indicative only.

| Method | Train (s) | Total (s) | n | Div |
|---|---|---|---|---|
| ALM | 79 | 171 | 60 | 0 |
| ALM+Bolt-On | (=ALM) [b] | 39 | 60 | 0 |
| ENFORCE | 120 | 237 | 60 | 20 |
| DC3 | 244 | 367 | 50 | 30 |
| FSNet | 5390 | 5603 | 60 | 0 |
| SnareNet | 449 | 526 | 60 | 0 |
| PAL | 164 | 286 | 60 | 0 |

[b] ALM+Bolt-On runs predict-only on the ALM checkpoints, so its training time is ALM's and its Total covers inference-time projection only.
