# e4/chip_layout: macro placement

Place N rectangular macros on a 200 x 200 canvas, simplified from FloorSet
(ICCAD 2024). One random instance is generated from a seed (default N = 30,
seed 42): block areas log-uniform in [20, 400], N random block-to-block nets
with weights in [0.5, 2], and max(5, N/3) fixed boundary pins.

| | |
|---|---|
| Design vector | 3N: block centres (cx, cy) and a raw width per block, `w = softplus(s) + 0.5`, `h = area / w` |
| Conditions | none |
| Objective | weighted half-perimeter wirelength (block-to-block and pin-to-block) plus 0.01 times the bounding-box area, with LSE-smoothed extrema (temperature 10) |
| Constraints | 1 inequality: summed pairwise AABB overlap area, margin 1e-4 |
| Tolerance | 1e-3 |

Compared with FloorSet, the dataset is replaced by one seeded instance and the
constraint set is reduced to non-overlap. Fixing the block areas keeps boxes
non-degenerate and the aspect ratio free.

```bash
pal run --method pal_loggap --benchmarks e4/chip_layout --seeds 0 --device cpu
```
