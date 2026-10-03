# Quickstart

Two runs to check an environment.

## 1. Synthetic benchmark (CPU, about a minute)

Checks the install, the CLI and a full train, projection and eval cycle.

```bash
pal run --method pal_loggap --benchmarks s1_sphere_track --seeds 0 --epochs 200 --device cpu
```

Expected: `feasibility_post` equal to 1.0 and a finite `obj_mean_post` in
`runs/<timestamp>_pal_loggap_s1_sphere_track_seed0_*/final.json`. Replace
`pal_loggap` by `alm`, `dc3`, `fsnet`, `snarenet` or
`enforce_v4` to check the baselines.

## 2. Engineering benchmark (GPU)

Checks CUDA and the in-tree neural surrogates of the aircraft benchmark.

```bash
pal run --method pal_loggap --benchmarks e1/bwb --seeds 0 --epochs 5 --device cuda
```

Pass: `status.json` reports `ok` and `final.json` contains `obj_mean_post` and
`feasibility_post`. `e4/chip_layout` runs on CPU without a GPU.
