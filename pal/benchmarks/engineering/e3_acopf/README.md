# e3/acopf: AC optimal power flow

AC optimal power flow on the IEEE 30, 57 and 118 bus cases
(`e3/acopf_ieee30`, `e3/acopf_ieee57`, `e3/acopf_ieee118`). Constraint
functions come from ML4OPF (MIT license) and grid data from the MATPOWER cases
shipped with pandapower. The formulation is the one used by FSNet and DC3.

| | IEEE-30 |
|---|---|
| Design vector | 72: 6 pg, 6 qg, 30 vm, 30 va (per unit, baseMVA 100) |
| Conditions | 42: active and reactive load at 21 buses |
| Objective | generation cost `sum(c0 + c1 pg + c2 pg^2)` |
| Equalities | 60: real and reactive power balance at every bus |
| Inequalities | 248: voltage, generation, thermal line and angle difference limits |

Load conditions are drawn from a fixed pool of 1200 samples per case (1000
train, 200 eval) in `data/`, generated as in DC3: nominal loads are perturbed
with correlated uniform noise and a sample is accepted if MATPOWER ACOPF
(without thermal limits) converges. Regenerate with
`python scripts/gen_e3_pools.py --case ieee57`. The pool is not guaranteed
to lie in the feasible set of the full constraint set, which includes thermal
and angle limits.

Requires `pip install pandapower ml4opf`.

```bash
pal run --method pal_loggap --benchmarks e3/acopf_ieee30 --seeds 0 --device cuda
```
