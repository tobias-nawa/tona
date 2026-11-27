## Plan (written before the run)

- Scope: workload short,mid,set (2000:400 2000:write:400 30000:200 at --ctx 140000; bench-set at --ctx 32768); A `b5afe` against B `b5afe`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: thread count of the CPU expert path: decode on short and set, prefill on mid; 2 to 4 rounds (docs/contribute/measurement.md, 2026-11-17-omp-threads.md) (decode tokens per prompt as in the targets, default 200)
- Duration: 8-16 min (2-4 rounds of 4.1 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: 2026-11-17-omp-threads.md expectation table):
  - decode: -0.5 to +4 %
  - prefill: -0.5 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1093-g58d84ce829-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=16, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-1093-g58d84ce829-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=32, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `b5afe` env {'OMP_NUM_THREADS': '16'}; B: `b5afe` env {'OMP_NUM_THREADS': '32'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1449.7 / 1452.4 / 1451.7 / 1453.6 | 1450.0 / 1452.9 / 1452.1 / 1454.3 | 1452.0 (1449.7-1453.6) | 1452.5 (1450.0-1454.3) | +0.04 % [-0.09, +0.16] | **equal** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 77.1 / 77.1 / 77.1 / 77.0 | 76.7 / 76.8 / 76.7 / 76.7 | 77.1 (77.0-77.1) | 76.7 (76.7-76.8) | -0.46 % [-0.79, -0.14] | **B slower** after round 3 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1721.8 / 1726.1 / 1730.1 / 1728.8 | 1726.6 / 1728.0 / 1727.7 / 1726.8 | 1727.4 (1721.8-1730.1) | 1727.3 (1726.6-1728.0) | +0.03 % [-0.29, +0.36] | **equal** after round 3 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 128.1 / 128.6 / 128.5 / 128.7 | 127.0 / 127.0 / 127.1 / 127.0 | 128.6 (128.1-128.7) | 127.0 (127.0-127.1) | -1.14 % [-2.00, -0.28] | **B slower** after round 4 |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1913.2 / 1916.9 / 1917.7 / 1917.8 | 1915.9 / 1915.2 / 1916.9 / 1915.5 | 1917.3 (1913.2-1917.8) | 1915.7 (1915.2-1916.9) | -0.03 % [-0.13, +0.08] | open (undecided at the last round) |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 70.2 / 70.3 / 70.3 / 70.3 | 70.1 / 70.0 / 70.1 / 70.1 | 70.3 (70.2-70.3) | 70.1 (70.0-70.1) | -0.28 % [-0.66, +0.11] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 94.4 / 95.8 / 95.8 / 95.8 | 95.3 / 95.4 / 95.3 / 95.3 | 95.8 (94.4-95.8) | 95.3 (95.3-95.4) | -0.13 % [-1.22, +0.97] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 68.7 / 68.8 / 68.7 / 68.8 | 68.5 / 68.6 / 68.6 / 68.5 | 68.8 (68.7-68.8) | 68.5 (68.5-68.6) | -0.29 % [-0.99, +0.41] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AB | +0.02 % | -0.46 % | +0.28 % | -0.90 % (off plan) | +0.14 % | -0.21 % | +0.95 % | -0.29 % | 246 |
| 2 | BA | +0.04 % | -0.44 % | +0.11 % | -1.28 % | -0.09 % | -0.34 % | -0.42 % | -0.29 % | 246 |
| 3 | AB | +0.03 % | -0.55 % | -0.14 % | -1.03 % | -0.04 % | -0.23 % | -0.52 % | -0.15 % | 246 |
| 4 | BA | +0.05 % | -0.39 % | -0.12 % | -1.36 % | -0.12 % | -0.32 % | -0.52 % | -0.44 % | 246 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.04 % | -0.5 to +1 % | within the expectation |
| decode@2000 | -0.46 % | -0.5 to +4 % | within the expectation |
| prefill@2000:write | +0.03 % | -0.5 to +1 % | within the expectation |
| decode@2000:write | -1.14 % | -0.5 to +4 % | **OUTSIDE the expectation: finding** |
| prefill@30000 | -0.03 % | -0.5 to +1 % | within the expectation |
| decode@30000 | -0.28 % | -0.5 to +4 % | within the expectation |
| decode-code | -0.13 % | -0.5 to +4 % | within the expectation |
| decode-text | -0.29 % | -0.5 to +4 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 128514, 24335 / 128514, 24335 / 128514, 24335 / 128514, 24335 | 128514, 24335 / 128514, 24335 / 128514, 24335 / 128514, 24335 | yes |
| 2000:write | 81757, 28520 / 81757, 28520 / 81757, 28520 / 81757, 28520 | 81757, 28520 / 81757, 28520 / 81757, 28520 / 81757, 28520 | yes |
| 30000 | 67692, 10677 / 67692, 10677 / 67692, 10677 / 67692, 10677 | 67692, 10677 / 67692, 10677 / 67692, 10677 / 67692, 10677 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 133.5 | 23.9 | 3.8 | 0.0 | 0.0 | 0.9 | 32676.4 | 0.5 | 24.5 | 44 | 0.0 | 0.0 | - | 0.0 | - | - | 24.1 |
| 2 | A | 234.8 | 10.2 | 4.6 | 0.0 | 0.0 | 2.1 | 28131.4 | 0.2 | 0.2 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.6 |
| 3 | A | 139.7 | 12.7 | 3.7 | 0.0 | 0.0 | 1.1 | 24933.6 | 0.0 | 1.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 4 | A | 237.3 | 10.8 | 3.8 | 0.0 | 0.0 | 2.2 | 30669.9 | 0.2 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.1 |
| 5 | B | 169.2 | 14.0 | 3.2 | 0.0 | 0.0 | 1.0 | 27424.4 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.3 |
| 6 | B | 319.4 | 10.1 | 4.1 | 0.0 | 0.0 | 2.7 | 27287.0 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 7 | B | 168.8 | 14.1 | 4.4 | 0.0 | 0.0 | 1.2 | 27429.8 | 0.2 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.3 |
| 8 | B | 319.2 | 10.1 | 3.7 | 0.0 | 0.0 | 2.8 | 27424.7 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 9 | A | 139.7 | 13.6 | 3.9 | 0.0 | 0.0 | 1.0 | 27405.1 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 10 | A | 241.5 | 9.8 | 3.6 | 0.0 | 0.0 | 2.4 | 27435.5 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.1 |
| 11 | A | 139.3 | 13.8 | 3.5 | 0.0 | 0.0 | 0.9 | 27415.0 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 12 | A | 240.1 | 9.9 | 3.6 | 0.0 | 0.0 | 2.7 | 27436.8 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 13 | B | 168.9 | 14.1 | 3.1 | 0.0 | 0.0 | 0.8 | 27420.2 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 14 | B | 318.9 | 10.0 | 3.8 | 0.0 | 0.0 | 3.0 | 27353.4 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 15 | B | 168.9 | 13.9 | 3.3 | 0.0 | 0.0 | 1.0 | 27406.9 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 16 | B | 318.9 | 10.2 | 3.7 | 0.0 | 0.0 | 2.7 | 27427.6 | 0.1 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 17 | A | 140.7 | 13.9 | 4.5 | 0.0 | 0.0 | 1.1 | 27454.3 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 18 | A | 239.8 | 9.8 | 3.7 | 0.0 | 0.0 | 2.8 | 27431.1 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | 0a2eac2f1502846f | 0a2eac2f1502846f | yes | yes | yes | - |

Runtime: 18.5 min for 4 rounds (4.1 min per round); a fixed 3 rounds would take 12.3 min (-4.1 min saved).
