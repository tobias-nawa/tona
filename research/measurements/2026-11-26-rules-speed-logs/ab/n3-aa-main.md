## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `n3-main` against B `n3-main`; rounds 4 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: control: is a difference of 0.1 % between two starts of the same binary possible (start order, page cache); same two requests as q1b (decode tokens per prompt as in the targets, default 200)
- Duration: 14-14 min (4-4 rounds of 3.4 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: A/A of the same build: 0 within the noise of bench/noise.toml; control for the -0.12 % of prefill@130000 in q1):
  - decode: -0.5 to +0.5 %
  - prefill: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit d43a0b3fa7; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
  - B: build commit d43a0b3fa7; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin

## Result

A: `n3-main` env -; B: `n3-main` env -. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1994.3 / 2000.0 / 2014.6 / 2003.7 | 1991.8 / 1998.9 / 1989.2 / 1996.0 | 2001.9 (1994.3-2014.6) | 1993.9 (1989.2-1998.9) | -0.46 % [-0.90, -0.02] | **B slower** after round 1 (real, but inside the margin) |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 74.4 / 74.4 / 74.5 / 74.4 | 74.4 / 74.7 / 74.5 / 74.7 | 74.4 (74.4-74.5) | 74.6 (74.4-74.7) | +0.24 % [-0.19, +0.66] | open (undecided at the last round) |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1895.3 / 1894.5 / 1899.2 / 1897.7 | 1901.0 / 1900.0 / 1898.7 / 1901.7 | 1896.5 (1894.5-1899.2) | 1900.5 (1898.7-1901.7) | +0.19 % [+0.07, +0.32] | **B faster** after round 1 (real, but inside the margin) |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 64.7 / 64.7 / 63.3 / 64.6 | 64.4 / 64.6 / 64.5 / 64.9 | 64.7 (63.3-64.7) | 64.6 (64.4-64.9) | +0.44 % [-0.57, +1.44] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -0.13 % | +0.02 % | +0.30 % | -0.48 % | 205 |
| 2 | BA | -0.06 % | +0.48 % | +0.29 % | -0.15 % | 207 |
| 3 | AB | -1.26 % (outlier) | +0.05 % | -0.03 % | +1.91 % (outlier) | 207 |
| 4 | BA | -0.39 % | +0.40 % | +0.21 % | +0.47 % | 204 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | -0.46 % | -0.5 to +0.5 % | within the expectation |
| decode@30000 | +0.24 % | -0.5 to +0.5 % | within the expectation |
| prefill@130000 | +0.19 % | -0.5 to +0.5 % | within the expectation |
| decode@130000 | +0.44 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 64471, 9816 / 64471, 9816 / 64471, 9816 / 64471, 9816 | 64471, 9816 / 64471, 9816 / 64471, 9816 / 64471, 9816 | yes |
| 130000 | 72291, 9615 / 72291, 9615 / 72291, 9615 / 72291, 9615 | 72291, 9615 / 72291, 9615 / 72291, 9615 / 72291, 9615 | yes |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 30000 | 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) | 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) / 3.37 (3.66) |
| 130000 | 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) | 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) / 3.03 (3.52) |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 132.8 | 21.2 | 13.1 | 0.0 | 0.0 | 0.4 | 33097.0 | 0.0 | 0.1 | 33 | 0.0 | 0.0 | - | 0.0 | - | - | 14.0 |
| 2 | A | 135.3 | 22.0 | 12.7 | 0.0 | 0.0 | 0.4 | 33387.4 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 3 | B | 134.6 | 19.7 | 12.8 | 0.0 | 0.0 | 0.3 | 32487.5 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 4 | B | 134.8 | 20.2 | 12.5 | 0.0 | 0.0 | 0.4 | 32240.3 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 5 | A | 132.7 | 22.5 | 12.2 | 0.0 | 0.0 | 0.4 | 34008.2 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 14.2 |
| 6 | A | 135.9 | 21.5 | 12.9 | 0.0 | 0.0 | 0.4 | 30956.3 | 0.1 | 0.3 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 7 | B | 132.1 | 21.8 | 12.7 | 0.0 | 0.0 | 0.4 | 35617.1 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 14.4 |
| 8 | B | 134.9 | 20.5 | 12.5 | 0.0 | 0.0 | 0.4 | 32137.0 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 9 | A | 136.1 | 21.4 | 12.5 | 0.0 | 0.0 | 0.4 | 33175.8 | 0.0 | 0.0 | 33 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | - |
| 130000 | 6e4f01e0e47ec72c | 6e4f01e0e47ec72c | yes | yes | yes | - |

Runtime: 15.5 min for 4 rounds (3.4 min per round); a fixed 3 rounds would take 10.3 min (-3.4 min saved).
