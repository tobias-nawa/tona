## Plan (written before the run)

- Scope: workload short,mid (2000:400 2000:write:400 30000:200 at --ctx 140000); A `b164t` against B `b164t`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: layout change of the expert slots or the GTT margin: decode on short and set, prefill on mid (research/measurements/2026-11-18-vram-chunks-gtt.md) (decode tokens per prompt as in the targets, default 200)
- Duration: 3-3 min (2-2 rounds of 1.7 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: 2026-11-18-vram-chunks-gtt.md expectation table):
  - decode: -0.5 to +0.5 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1106-gbc2589bb8b; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-1106-gbc2589bb8b; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin, TONA_SLOTS_CHUNKED=1

## Result

A: `b164t` env {'TONA_DETERMINISTIC': '1'}; B: `b164t` env {'TONA_DETERMINISTIC': '1', 'TONA_SLOTS_CHUNKED': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1450.4 / 1453.0 | 1450.4 / 1450.3 | 1451.7 (1450.4-1453.0) | 1450.4 (1450.3-1450.4) | -0.09 % [-0.33, +0.15] | **equal** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 73.2 / 72.9 | 73.2 / 73.2 | 73.0 (72.9-73.2) | 73.2 (73.2-73.2) | +0.17 % [-0.40, +0.74] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1727.4 / 1726.8 | 1727.3 / 1725.8 | 1727.1 (1726.8-1727.4) | 1726.5 (1725.8-1727.3) | -0.03 % [-0.38, +0.32] | **equal** after round 1 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 97.6 / 97.9 | 97.6 / 97.6 | 97.7 (97.6-97.9) | 97.6 (97.6-97.6) | -0.10 % [-1.42, +1.22] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1915.7 / 1913.6 | 1912.1 / 1912.8 | 1914.7 (1913.6-1915.7) | 1912.5 (1912.1-1912.8) | -0.11 % [-0.21, -0.01] | **B slower** after round 1 (real, but inside the margin) |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 72.4 / 71.7 | 71.8 / 71.7 | 72.0 (71.7-72.4) | 71.8 (71.7-71.8) | -0.36 % [-1.01, +0.29] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | +0.00 % | -0.01 % | -0.00 % | +0.02 % | -0.19 % | -0.80 % (off plan) | 104 |
| 2 | BA | -0.18 % | +0.35 % | -0.06 % | -0.22 % | -0.04 % | +0.08 % | 104 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | -0.09 % | -1 to +1 % | within the expectation |
| decode@2000 | +0.17 % | -0.5 to +0.5 % | within the expectation |
| prefill@2000:write | -0.03 % | -1 to +1 % | within the expectation |
| decode@2000:write | -0.10 % | -0.5 to +0.5 % | within the expectation |
| prefill@30000 | -0.11 % | -1 to +1 % | within the expectation |
| decode@30000 | -0.36 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 135492, 22183 / 135492, 22183 | 135492, 22183 / 135492, 22183 | yes |
| 2000:write | 111766, 26482 / 111766, 26482 | 111766, 26482 / 111766, 26482 | yes |
| 30000 | 68165, 9969 / 68165, 9969 | 68165, 9969 / 68165, 9969 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 134.5 | 16.0 | 9.0 | 0.0 | 0.0 | 1.4 | 28102.3 | 0.1 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | - | 20.1 |
| 2 | A | 131.9 | 14.2 | 10.4 | 0.0 | 0.0 | 1.3 | 27383.2 | 0.1 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.5 |
| 3 | B | 137.3 | 13.8 | 7.7 | 0.0 | 0.0 | 1.5 | 27482.9 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 4 | B | 136.9 | 13.6 | 4.5 | 0.0 | 0.0 | 1.5 | 27409.2 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 5 | A | 137.2 | 13.7 | 3.8 | 0.0 | 0.0 | 1.3 | 27436.5 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.6 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 11eacf1ee10017b4 | 11eacf1ee10017b4 | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | ee58880d3f5b558f (B = ref) |

Runtime: 4.3 min for 2 rounds (1.7 min per round); a fixed 3 rounds would take 5.2 min (+1.7 min saved).
