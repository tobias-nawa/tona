## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `n3-main` against B `n3-pr`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: mid and long only: the two requests where q1 left a real or open difference (decode tokens per prompt as in the targets, default 200)
- Duration: 7-21 min (2-6 rounds of 3.4 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: q1 of this file: prefill@130000 -0.12 % [-0.22, -0.02] (inside the margin, B slower), decode and prefill at 30K equal; repeat to see whether the small loss reproduces):
  - decode: -0.5 to +0.5 %
  - prefill: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit d43a0b3fa7; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
  - B: build commit f50cd70c7a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin

## Result

A: `n3-main` env -; B: `n3-pr` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 2019.7 / 1996.6 / 1993.7 / 2007.3 / 2004.8 / 1990.9 | 1994.6 / 2008.6 / 2019.7 / 2013.9 / 2014.3 / 1991.1 | 2000.7 (1990.9-2019.7) | 2011.3 (1991.1-2019.7) | +0.25 % [-0.43, +0.92] | open (undecided at the last round) |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 73.1 / 74.5 / 74.9 / 74.5 / 74.4 / 74.7 | 73.5 / 74.7 / 75.1 / 73.7 / 74.7 / 74.7 | 74.5 (73.1-74.9) | 74.7 (73.5-75.1) | +0.08 % [-0.46, +0.63] | open (undecided at the last round) |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1900.7 / 1896.1 / 1902.6 / 1896.8 / 1900.0 / 1900.1 | 1900.2 / 1900.5 / 1900.5 / 1900.8 / 1901.1 / 1900.1 | 1900.1 (1896.1-1902.6) | 1900.5 (1900.1-1901.1) | +0.06 % [-0.05, +0.17] | **equal** after round 1 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 64.6 / 64.5 / 65.5 / 64.9 / 64.5 / 64.6 | 64.5 / 64.5 / 64.0 / 64.8 / 64.5 / 64.7 | 64.6 (64.5-65.5) | 64.5 (64.0-64.8) | -0.41 % [-1.26, +0.44] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -1.24 % (off plan) | +0.57 % (off plan) | -0.03 % | -0.17 % | 205 |
| 2 | BA | +0.60 % (outlier) | +0.28 % | +0.23 % | -0.00 % | 207 |
| 3 | AB | +1.30 % (outlier) | +0.34 % | -0.11 % | -2.24 % (outlier) | 202 |
| 4 | BA | +0.33 % | -1.01 % (outlier) | +0.21 % | -0.10 % | 206 |
| 5 | AB | +0.47 % | +0.31 % | +0.05 % | -0.05 % | 204 |
| 6 | BA | +0.01 % | +0.01 % | -0.00 % | +0.09 % | 207 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +0.25 % | -0.5 to +0.5 % | within the expectation |
| decode@30000 | +0.08 % | -0.5 to +0.5 % | within the expectation |
| prefill@130000 | +0.06 % | -0.5 to +0.5 % | within the expectation |
| decode@130000 | -0.41 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 64471, 9816 / 64471, 9816 / 64471, 9816 / 64471, 9816 / 64471, 9816 / 64471, 9816 | 64471, 9816 / 64471, 9816 / 64471, 9816 / 64471, 9816 / 64471, 9816 / 64471, 9816 | yes |
| 130000 | 72291, 9615 / 72291, 9615 / 72291, 9615 / 72291, 9615 / 72291, 9615 / 72291, 9615 | 72291, 9615 / 72291, 9615 / 72291, 9615 / 72291, 9615 / 72291, 9615 / 72291, 9615 | yes |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 30000 | 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) | 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) |
| 130000 | 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) | 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 132.5 | 20.0 | 13.1 | 0.0 | 0.0 | 0.3 | 37733.6 | 19.3 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | - | 14.2 |
| 2 | A | 134.9 | 18.8 | 13.1 | 0.0 | 0.0 | 0.4 | 32319.9 | 1.7 | 19.8 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 3 | B | 132.0 | 17.3 | 12.5 | 0.0 | 0.0 | 0.3 | 30276.5 | 4.4 | 0.8 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.7 |
| 4 | B | 135.5 | 20.9 | 12.5 | 0.0 | 0.0 | 0.4 | 32478.2 | 0.0 | 4.5 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 5 | A | 132.9 | 23.0 | 12.2 | 0.0 | 0.0 | 0.3 | 33903.9 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 14.4 |
| 6 | A | 135.9 | 16.2 | 12.5 | 0.0 | 0.0 | 0.4 | 27907.6 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 10.0 |
| 7 | B | 132.6 | 16.9 | 12.3 | 0.0 | 0.0 | 0.4 | 30736.9 | 0.1 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.3 |
| 8 | B | 135.1 | 20.5 | 13.2 | 0.0 | 0.0 | 0.4 | 32729.3 | 0.1 | 0.2 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 9 | A | 133.1 | 22.6 | 12.6 | 0.0 | 0.0 | 0.3 | 34034.3 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 14.1 |
| 10 | A | 135.5 | 20.8 | 12.6 | 0.0 | 0.0 | 0.4 | 32738.2 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 11 | B | 136.0 | 19.7 | 12.6 | 0.0 | 0.0 | 0.4 | 32068.0 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 12 | B | 135.2 | 18.7 | 12.5 | 0.0 | 0.0 | 0.5 | 30773.0 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.1 |
| 13 | A | 132.8 | 21.3 | 12.4 | 0.0 | 0.0 | 0.4 | 33031.8 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 14.1 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | - |
| 130000 | 6e4f01e0e47ec72c | 6e4f01e0e47ec72c | yes | yes | yes | - |

Runtime: 22.3 min for 6 rounds (3.4 min per round); a fixed 3 rounds would take 10.3 min (-10.3 min saved).
