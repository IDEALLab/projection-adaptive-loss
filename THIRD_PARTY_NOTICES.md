# Third-party notices

The code in this repository is licensed under MIT (see `LICENSE`). The
following third-party code and data are included and remain under their own
licenses. Upstream licence files are kept next to the vendored code where the
upstream repository provides one.

| Location | Upstream | Commit | License |
|---|---|---|---|
| `pal/baselines/dc3/upstream/` | DC3 (Donti, Rolnick, Kolter, ICLR 2021), `locuslab/DC3` | `35437af` | Apache-2.0, `upstream/LICENSE` |
| `pal/baselines/fsnet/upstream/` | FSNet (Nguyen, Donti 2025), `MOSSLab-MIT/FSNet` | `911fcb2` | MIT, `upstream/LICENSE` |
| `pal/baselines/snarenet/upstream/` | SnareNet (Chu, Boukas, Udell 2026), `miniyachi/SnareNet` | `203f136` | |
| `pal/baselines/enforce_v4/upstream/` | ENFORCE v1.0.4 (Lastrucci, Schweidtmann), `process-intelligence-research/ENFORCE` | `acb51bb` | MIT, `upstream/LICENSE` |
| `pal/benchmarks/engineering/e2_urban_wind/_vendor/` | WinDiNet urban wind surrogate (fine-tune of Lightricks LTX-Video), `rabischof/windinet` on Hugging Face | | Apache 2.0 (code) |
| `pal/benchmarks/engineering/e3_acopf/` | constraint functions from ML4OPF (AI4OPT), grid cases from pandapower / MATPOWER, installed as dependencies | | MIT / BSD |

The commit of each vendored baseline is also stored in its `UPSTREAM_SHA`
file. The changes made to vendored files are listed in the README of each
baseline.

## WinDiNet model weights

The e2 benchmark downloads WinDiNet weights, which are derived from
Lightricks LTX-Video and are governed by the LTX-Video Open Weights License:

  https://huggingface.co/Lightricks/LTX-Video/blob/main/LTX-Video-Open-Weights-License-0.X.txt

The use-based restrictions in Attachment A of that license apply to any use
of the weights downloaded through this code. Commercial entities with annual
revenues of 10M USD or more need a separate agreement with Lightricks.
Academic and research use is permitted.

The weights are hosted at https://huggingface.co/rabischof/windinet. Cite the
WinDiNet paper as listed on that model card when using the e2 benchmark.
