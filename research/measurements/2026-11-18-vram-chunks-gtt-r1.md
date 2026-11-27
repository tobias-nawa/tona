## Plan (written before the run)

- Scope: workload short,mid,set (2000:400 2000:write:400 30000:200 at --ctx 140000; bench-set at --ctx 32768); A `b164t` against B `b164t`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: layout change of the expert slots or the GTT margin: decode on short and set, prefill on mid (research/measurements/2026-11-18-vram-chunks-gtt.md) (decode tokens per prompt as in the targets, default 200)
- Duration: 8-16 min (2-4 rounds of 4.1 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: 2026-11-18-vram-chunks-gtt.md expectation table):
  - decode: -0.5 to +0.5 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1106-gbc2589bb8b; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-1106-gbc2589bb8b; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin, TONA_SLOTS_CHUNKED=1

## Result

A: `b164t` env -; B: `b164t` env {'TONA_SLOTS_CHUNKED': '1'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1433.9 / 1448.6 / 1451.5 / 1450.3 | 1449.7 / 1450.7 / 1454.6 / 1453.8 | 1449.5 (1433.9-1451.5) | 1452.2 (1449.7-1454.6) | +0.43 % [-0.13, +0.98] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 76.1 / 76.9 / 76.9 / 76.9 | 75.8 / 75.9 / 76.9 / 76.8 | 76.9 (76.1-76.9) | 76.4 (75.8-76.9) | -0.42 % [-1.20, +0.35] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1718.5 / 1726.5 / 1725.8 / 1725.2 | 1722.3 / 1727.0 / 1727.6 / 1725.6 | 1725.5 (1718.5-1726.5) | 1726.3 (1722.3-1727.6) | +0.09 % [-0.15, +0.34] | **equal** after round 2 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 126.8 / 127.8 / 127.7 / 127.9 | 128.0 / 128.0 / 128.3 / 127.9 | 127.8 (126.8-127.9) | 128.0 (127.9-128.3) | +0.42 % [-0.55, +1.39] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1911.4 / 1913.1 / 1914.7 / 1915.1 | 1912.8 / 1913.1 / 1914.9 / 1915.0 | 1913.9 (1911.4-1915.1) | 1914.0 (1912.8-1915.0) | +0.02 % [-0.04, +0.08] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 69.2 / 70.2 / 70.0 / 70.2 | 68.7 / 69.9 / 70.2 / 70.1 | 70.1 (69.2-70.2) | 70.0 (68.7-70.2) | -0.25 % [-0.77, +0.28] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 95.3 / 95.5 / 95.6 / 95.6 | 93.9 / 95.1 / 95.5 / 95.4 | 95.5 (95.3-95.6) | 95.2 (93.9-95.5) | -0.55 % [-1.56, +0.46] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 68.5 / 68.7 / 68.7 / 68.7 | 67.4 / 68.2 / 68.7 / 68.6 | 68.7 (68.5-68.7) | 68.4 (67.4-68.7) | -0.62 % [-1.72, +0.48] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AB | +1.10 % (off plan) | -0.36 % | +0.22 % | +0.99 % (off plan) | +0.07 % | -0.72 % (off plan) | -1.47 % | -1.61 % | 247 |
| 2 | BA | +0.14 % | -1.28 % | +0.03 % | +0.18 % | +0.00 % | -0.48 % | -0.42 % | -0.73 % | 247 |
| 3 | AB | +0.21 % | +0.03 % | +0.10 % | +0.49 % | +0.01 % | +0.35 % | -0.10 % | +0.00 % | 247 |
| 4 | BA | +0.24 % | -0.09 % | +0.02 % | +0.01 % | -0.00 % | -0.14 % | -0.21 % | -0.15 % | 247 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.43 % | -1 to +1 % | within the expectation |
| decode@2000 | -0.42 % | -0.5 to +0.5 % | within the expectation |
| prefill@2000:write | +0.09 % | -1 to +1 % | within the expectation |
| decode@2000:write | +0.42 % | -0.5 to +0.5 % | within the expectation |
| prefill@30000 | +0.02 % | -1 to +1 % | within the expectation |
| decode@30000 | -0.25 % | -0.5 to +0.5 % | within the expectation |
| decode-code | -0.55 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode-text | -0.62 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 128026, 24823 / 128026, 24823 / 128026, 24823 / 128026, 24823 | 128026, 24823 / 128026, 24823 / 128026, 24823 / 128026, 24823 | yes |
| 2000:write | 81310, 28967 / 81310, 28967 / 81310, 28967 / 81310, 28967 | 81310, 28967 / 81310, 28967 / 81310, 28967 / 81310, 28967 | yes |
| 30000 | 67469, 10900 / 67469, 10900 / 67469, 10900 / 67469, 10900 | 67469, 10900 / 67469, 10900 / 67469, 10900 / 67469, 10900 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 139.8 | 18.1 | 7.5 | 0.0 | 0.0 | 0.8 | 28862.5 | 0.1 | 0.3 | 44 | 0.0 | 0.0 | - | 0.0 | - | - | 20.0 |
| 2 | A | 236.6 | 10.0 | 3.8 | 0.0 | 0.0 | 2.1 | 27874.6 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.5 |
| 3 | A | 138.3 | 14.4 | 4.5 | 0.0 | 0.0 | 0.8 | 27489.1 | 0.0 | 2.5 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 4 | A | 238.8 | 9.7 | 4.5 | 0.0 | 0.0 | 2.9 | 27269.1 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 5 | B | 138.3 | 14.4 | 3.7 | 0.0 | 0.0 | 1.0 | 27391.6 | 0.0 | 2.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 6 | B | 235.5 | 9.7 | 5.1 | 0.0 | 0.0 | 2.4 | 27342.1 | 0.1 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.2 |
| 7 | B | 139.6 | 14.2 | 4.3 | 0.0 | 0.0 | 0.9 | 27331.5 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 8 | B | 238.5 | 9.7 | 4.3 | 0.0 | 0.0 | 2.6 | 27290.2 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.2 |
| 9 | A | 138.9 | 14.2 | 4.6 | 0.0 | 0.0 | 1.1 | 27298.6 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 10 | A | 238.4 | 9.8 | 4.4 | 0.0 | 0.0 | 2.5 | 27363.8 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.3 |
| 11 | A | 137.9 | 14.2 | 3.8 | 0.0 | 0.0 | 0.8 | 27329.3 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.6 |
| 12 | A | 238.7 | 9.8 | 4.5 | 0.0 | 0.0 | 2.9 | 27286.2 | 0.9 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.3 |
| 13 | B | 139.8 | 14.0 | 2.8 | 0.0 | 0.0 | 1.0 | 27250.3 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.6 |
| 14 | B | 239.2 | 9.6 | 3.7 | 0.0 | 0.0 | 2.9 | 27344.9 | 0.1 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.3 |
| 15 | B | 137.9 | 14.1 | 3.8 | 0.0 | 0.0 | 0.9 | 27358.6 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 16 | B | 238.8 | 9.8 | 4.0 | 0.0 | 0.0 | 2.7 | 27340.6 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.4 |
| 17 | A | 138.6 | 14.0 | 3.4 | 0.0 | 0.0 | 0.9 | 27355.0 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.6 |
| 18 | A | 239.7 | 9.7 | 4.3 | 0.0 | 0.0 | 2.8 | 27315.9 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.3 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | 0a2eac2f1502846f | 0a2eac2f1502846f | yes | yes | yes | - |

Runtime: 18.5 min for 4 rounds (4.1 min per round); a fixed 3 rounds would take 12.3 min (-4.1 min saved).
