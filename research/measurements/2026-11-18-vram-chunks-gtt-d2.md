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
  - B: build commit v0.0.2-1106-gbc2589bb8b; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin, TONA_GTT_MARGIN_MB=3072

## Result

A: `b164t` env {'TONA_DETERMINISTIC': '1'}; B: `b164t` env {'TONA_DETERMINISTIC': '1', 'TONA_GTT_MARGIN_MB': '3072'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1446.4 / 1454.8 | 1454.1 / 1450.1 | 1450.6 (1446.4-1454.8) | 1452.1 (1450.1-1454.1) | +0.10 % [-0.58, +0.78] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 71.7 / 73.9 | 72.8 / 73.9 | 72.8 (71.7-73.9) | 73.4 (72.8-73.9) | +0.85 % [-0.41, +2.11] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1718.2 / 1712.0 | 1723.2 / 1726.3 | 1715.1 (1712.0-1718.2) | 1724.7 (1723.2-1726.3) | +0.56 % [+0.01, +1.11] | **B faster** after round 2 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 97.8 / 97.5 | 97.6 / 97.8 | 97.6 (97.5-97.8) | 97.7 (97.6-97.8) | +0.08 % [-1.26, +1.42] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1911.9 / 1914.7 | 1911.8 / 1917.0 | 1913.3 (1911.9-1914.7) | 1914.4 (1911.8-1917.0) | +0.06 % [-0.04, +0.15] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 71.5 / 73.1 | 72.4 / 73.1 | 72.3 (71.5-73.1) | 72.8 (72.4-73.1) | +0.61 % [-0.13, +1.35] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | +0.53 % | +1.61 % (off plan) | +0.29 % | -0.11 % | -0.01 % | +1.20 % (off plan) | 104 |
| 2 | BA | -0.33 % | +0.09 % | +0.83 % | +0.28 % | +0.12 % | +0.01 % | 104 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.10 % | -1 to +1 % | within the expectation |
| decode@2000 | +0.85 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| prefill@2000:write | +0.56 % | -1 to +1 % | within the expectation |
| decode@2000:write | +0.08 % | -0.5 to +0.5 % | within the expectation |
| prefill@30000 | +0.06 % | -1 to +1 % | within the expectation |
| decode@30000 | +0.61 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 135492, 22183 / 135492, 22183 | 135492, 22183 / 135492, 22183 | yes |
| 2000:write | 111766, 26482 / 111766, 26482 | 111766, 26482 / 111766, 26482 | yes |
| 30000 | 68165, 9969 / 68165, 9969 | 68165, 9969 / 68165, 9969 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 134.0 | 16.4 | 4.5 | 0.0 | 0.0 | 1.0 | 27926.8 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | - | 20.1 |
| 2 | A | 136.4 | 14.0 | 8.7 | 0.0 | 0.0 | 1.2 | 27602.5 | 22.7 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.5 |
| 3 | B | 135.7 | 13.7 | 5.8 | 0.0 | 0.0 | 1.3 | 27466.6 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 20.6 |
| 4 | B | 131.6 | 13.9 | 4.6 | 0.0 | 0.0 | 1.2 | 27402.6 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 5 | A | 131.3 | 14.1 | 3.3 | 0.0 | 0.0 | 1.2 | 27280.2 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.6 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 11eacf1ee10017b4 | 11eacf1ee10017b4 | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | ee58880d3f5b558f (B = ref) |

Runtime: 4.3 min for 2 rounds (1.7 min per round); a fixed 3 rounds would take 5.2 min (+1.7 min saved).
