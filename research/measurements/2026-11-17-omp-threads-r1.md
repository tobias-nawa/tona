## Plan (written before the run)

- Scope: workload short,mid,set (2000:400 2000:write:400 30000:200 at --ctx 140000; bench-set at --ctx 32768); A `b5afe` against B `b5afe`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: thread count of the CPU expert path: decode on short and set, prefill on mid; 2 to 4 rounds (docs/contribute/measurement.md, 2026-11-17-omp-threads.md) (decode tokens per prompt as in the targets, default 200)
- Duration: 8-16 min (2-4 rounds of 4.0 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: 2026-11-17-omp-threads.md expectation table):
  - decode: -0.5 to +4 %
  - prefill: -0.5 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1063-g96b4c2075c; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-1063-g96b4c2075c; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=16, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `b5afe` env {'OMP_NUM_THREADS': '6'}; B: `b5afe` env {'OMP_NUM_THREADS': '16'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 4, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1464.3 / 1465.1 | 1463.7 / 1463.2 | 1464.7 (1464.3-1465.1) | 1463.5 (1463.2-1463.7) | -0.08 % [-0.31, +0.14] | **equal** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 76.7 / 76.6 | 78.0 / 78.1 | 76.7 (76.6-76.7) | 78.0 (78.0-78.1) | +1.78 % [+1.16, +2.40] | **B faster** after round 1 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1740.9 / 1740.1 | 1739.5 / 1740.8 | 1740.5 (1740.1-1740.9) | 1740.2 (1739.5-1740.8) | -0.02 % [-0.42, +0.38] | **equal** after round 2 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 126.7 / 126.0 | 131.8 / 131.6 | 126.4 (126.0-126.7) | 131.7 (131.6-131.8) | +4.25 % [+2.77, +5.73] | **B faster** after round 2 |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1979.9 / 1981.6 | 1981.3 / 1980.2 | 1980.7 (1979.9-1981.6) | 1980.7 (1980.2-1981.3) | +0.00 % [-0.11, +0.11] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 70.0 / 70.0 | 70.9 / 70.9 | 70.0 (70.0-70.0) | 70.9 (70.9-70.9) | +1.27 % [+0.71, +1.84] | **B faster** after round 1 |
| decode-code: Decode tok/s (bench-set code) | 94.0 / 95.5 | 97.0 / 97.0 | 94.8 (94.0-95.5) | 97.0 (97.0-97.0) | +2.38 % [+0.95, +3.81] | **B faster** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 68.3 / 68.9 | 69.5 / 69.5 | 68.6 (68.3-68.9) | 69.5 (69.5-69.5) | +1.31 % [+0.20, +2.42] | **B faster** after round 2 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AB | -0.04 % | +1.61 % | -0.08 % | +4.05 % (off plan) | +0.07 % | +1.31 % | +3.19 % | +1.76 % | 240 |
| 2 | BA | -0.12 % | +1.95 % | +0.04 % | +4.46 % | -0.07 % | +1.23 % | +1.57 % | +0.87 % | 239 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | -0.08 % | -0.5 to +1 % | within the expectation |
| decode@2000 | +1.78 % | -0.5 to +4 % | within the expectation |
| prefill@2000:write | -0.02 % | -0.5 to +1 % | within the expectation |
| decode@2000:write | +4.25 % | -0.5 to +4 % | **OUTSIDE the expectation: finding** |
| prefill@30000 | +0.00 % | -0.5 to +1 % | within the expectation |
| decode@30000 | +1.27 % | -0.5 to +4 % | within the expectation |
| decode-code | +2.38 % | -0.5 to +4 % | within the expectation |
| decode-text | +1.31 % | -0.5 to +4 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 131985, 20864 / 131540, 21309 | 131540, 21309 / 131540, 21309 | NO |
| 2000:write | 85325, 24952 / 84912, 25365 | 84912, 25365 / 84912, 25365 | NO |
| 30000 | 69049, 9320 / 68879, 9490 | 68879, 9490 / 68879, 9490 | NO |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 109.2 | 23.4 | 5.6 | 0.0 | 0.0 | 0.6 | 40939.9 | 0.7 | 16.4 | 44 | 0.0 | 0.0 | - | 0.0 | - | - | 24.0 |
| 2 | A | 163.6 | 9.8 | 4.5 | 0.0 | 0.0 | 1.8 | 27986.7 | 9.6 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.5 |
| 3 | A | 111.8 | 14.3 | 2.7 | 0.0 | 0.0 | 0.8 | 25173.8 | 0.1 | 0.9 | 44 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |
| 4 | A | 163.5 | 10.0 | 2.6 | 0.0 | 0.0 | 1.8 | 30157.4 | 0.1 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.6 |
| 5 | B | 138.2 | 13.1 | 2.6 | 0.0 | 0.0 | 0.8 | 27450.3 | 0.1 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 6 | B | 226.0 | 9.3 | 3.4 | 0.0 | 0.0 | 2.1 | 27493.3 | 0.1 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |
| 7 | B | 135.9 | 13.2 | 2.5 | 0.0 | 0.0 | 0.6 | 27435.7 | 1.7 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 8 | B | 225.6 | 9.5 | 2.9 | 0.0 | 0.0 | 1.7 | 27483.9 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |
| 9 | A | 114.1 | 13.1 | 2.6 | 0.0 | 0.0 | 0.7 | 27466.5 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 10 | A | 165.0 | 9.3 | 2.7 | 0.0 | 0.0 | 1.7 | 27479.2 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | 0a2eac2f1502846f | 0a2eac2f1502846f | yes | yes | yes | - |

Runtime: 10.1 min for 2 rounds (4.0 min per round); a fixed 3 rounds would take 12.0 min (+4.0 min saved).
