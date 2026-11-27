## Plan (written before the run)

- Scope: workload short,mid,set (2000:400 2000:write:400 30000:200 at --ctx 140000; bench-set at --ctx 32768); A `b5afe` against B `b5afe`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: thread count of the CPU expert path: decode on short and set, prefill on mid; 2 to 4 rounds (docs/contribute/measurement.md, 2026-11-17-omp-threads.md) (decode tokens per prompt as in the targets, default 200)
- Duration: 8-16 min (2-4 rounds of 4.0 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: 2026-11-17-omp-threads.md expectation table):
  - decode: -0.5 to +4 %
  - prefill: -0.5 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1063-g96b4c2075c; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=16, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-1063-g96b4c2075c; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=12, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `b5afe` env {'OMP_NUM_THREADS': '16'}; B: `b5afe` env {'OMP_NUM_THREADS': '12'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1460.8 / 1460.6 / 1460.9 / 1465.0 | 1461.1 / 1457.4 / 1459.5 / 1461.7 | 1460.8 (1460.6-1465.0) | 1460.3 (1457.4-1461.7) | -0.13 % [-0.31, +0.06] | **equal** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 76.4 / 78.0 / 78.1 / 78.1 | 76.8 / 76.5 / 78.0 / 78.1 | 78.0 (76.4-78.1) | 77.4 (76.5-78.1) | -0.35 % [-1.68, +0.98] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1739.4 / 1739.7 / 1741.5 / 1740.3 | 1734.3 / 1733.1 / 1739.9 / 1739.0 | 1740.0 (1739.4-1741.5) | 1736.7 (1733.1-1739.9) | -0.21 % [-0.49, +0.08] | **equal** after round 4 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 131.7 / 131.5 / 131.5 / 131.7 | 132.0 / 132.1 / 132.1 / 132.2 | 131.6 (131.5-131.7) | 132.1 (132.0-132.2) | +0.38 % [-0.45, +1.21] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1978.6 / 1978.3 / 1978.7 / 1978.7 | 1979.2 / 1979.8 / 1981.2 / 1979.8 | 1978.6 (1978.3-1978.7) | 1979.8 (1979.2-1981.2) | +0.07 % [+0.01, +0.14] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 69.5 / 70.9 / 70.8 / 70.8 | 69.5 / 69.6 / 70.9 / 70.8 | 70.8 (69.5-70.9) | 70.2 (69.5-70.9) | -0.38 % [-1.22, +0.45] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 95.6 / 96.9 / 97.0 / 96.9 | 95.5 / 95.2 / 97.1 / 97.1 | 96.9 (95.6-97.0) | 96.3 (95.2-97.1) | -0.39 % [-1.67, +0.90] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 68.9 / 69.5 / 69.5 / 69.5 | 68.9 / 68.7 / 69.6 / 69.6 | 69.5 (68.9-69.5) | 69.2 (68.7-69.6) | -0.22 % [-1.22, +0.79] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AB | +0.02 % | +0.56 % | -0.29 % | +0.25 % | +0.03 % | +0.02 % | -0.10 % | +0.00 % | 241 |
| 2 | BA | -0.22 % | -1.93 % (outlier) | -0.38 % | +0.40 % | +0.07 % | -1.79 % (outlier) | -1.75 % | -1.15 % | 241 |
| 3 | AB | -0.09 % | -0.06 % | -0.09 % | +0.45 % | +0.13 % | +0.25 % | +0.10 % | +0.14 % | 240 |
| 4 | BA | -0.22 % | +0.03 % | -0.07 % | +0.43 % | +0.06 % | +0.00 % | +0.21 % | +0.14 % | 240 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | -0.13 % | -0.5 to +1 % | within the expectation |
| decode@2000 | -0.35 % | -0.5 to +4 % | within the expectation |
| prefill@2000:write | -0.21 % | -0.5 to +1 % | within the expectation |
| decode@2000:write | +0.38 % | -0.5 to +4 % | within the expectation |
| prefill@30000 | +0.07 % | -0.5 to +1 % | within the expectation |
| decode@30000 | -0.38 % | -0.5 to +4 % | within the expectation |
| decode-code | -0.39 % | -0.5 to +4 % | within the expectation |
| decode-text | -0.22 % | -0.5 to +4 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 131540, 21309 / 131540, 21309 / 131540, 21309 / 131540, 21309 | 131540, 21309 / 131540, 21309 / 131540, 21309 / 131540, 21309 | yes |
| 2000:write | 84912, 25365 / 84912, 25365 / 84912, 25365 / 84912, 25365 | 84912, 25365 / 84912, 25365 / 84912, 25365 / 84912, 25365 | yes |
| 30000 | 68879, 9490 / 68879, 9490 / 68879, 9490 / 68879, 9490 | 68879, 9490 / 68879, 9490 / 68879, 9490 / 68879, 9490 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 133.0 | 16.5 | 45.0 | 0.0 | 0.0 | 0.6 | 28256.2 | 0.6 | 122.5 | 44 | 0.0 | 0.0 | - | 0.0 | - | - | 20.0 |
| 2 | A | 223.3 | 9.9 | 2.9 | 0.0 | 0.0 | 1.6 | 28005.3 | 0.6 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 3 | A | 135.8 | 13.8 | 3.1 | 0.0 | 0.0 | 0.9 | 27587.5 | 0.5 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.6 |
| 4 | A | 222.9 | 9.9 | 5.2 | 0.0 | 0.0 | 1.7 | 27644.7 | 0.3 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 5 | B | 126.6 | 13.9 | 2.9 | 0.0 | 0.0 | 0.7 | 27650.7 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.6 |
| 6 | B | 199.9 | 9.6 | 3.3 | 0.0 | 0.0 | 1.8 | 27589.2 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 7 | B | 126.2 | 13.8 | 3.2 | 0.0 | 0.0 | 0.7 | 27633.0 | 0.1 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.7 |
| 8 | B | 200.9 | 9.5 | 2.7 | 0.0 | 0.0 | 1.7 | 27594.1 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 9 | A | 135.8 | 14.0 | 2.7 | 0.0 | 0.0 | 0.7 | 27618.3 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.7 |
| 10 | A | 223.7 | 9.9 | 2.9 | 0.0 | 0.0 | 1.7 | 27521.9 | 0.1 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |
| 11 | A | 136.1 | 14.1 | 3.0 | 0.0 | 0.0 | 0.7 | 27629.2 | 0.4 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.7 |
| 12 | A | 223.7 | 9.9 | 4.0 | 0.0 | 0.0 | 2.1 | 27584.8 | 0.1 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |
| 13 | B | 127.1 | 13.9 | 3.0 | 0.0 | 0.0 | 0.7 | 27641.8 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.7 |
| 14 | B | 202.9 | 9.8 | 3.3 | 0.0 | 0.0 | 1.9 | 27555.4 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 15 | B | 128.0 | 13.8 | 2.6 | 0.0 | 0.0 | 0.6 | 27621.3 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.7 |
| 16 | B | 201.9 | 9.9 | 3.4 | 0.0 | 0.0 | 2.2 | 27518.7 | 0.3 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 17 | A | 135.3 | 13.9 | 2.6 | 0.0 | 0.0 | 0.8 | 27600.9 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.7 |
| 18 | A | 224.7 | 9.9 | 3.0 | 0.0 | 0.0 | 1.8 | 27576.1 | 0.2 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | 0a2eac2f1502846f | 0a2eac2f1502846f | yes | yes | yes | - |

Runtime: 18.0 min for 4 rounds (4.0 min per round); a fixed 3 rounds would take 12.0 min (-4.0 min saved).
