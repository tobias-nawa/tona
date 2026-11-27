## Plan (written before the run)

- Scope: workload short (2000:400 2000:write:400 at --ctx 140000); A `ep175` against B `ep175`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: A/A of the short workload: the lookup gives the Coder its profile (decode tokens per prompt as in the targets, default 200)
- Duration: 1-1 min (2-2 rounds of 0.7 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: bench/noise.toml decode@2000, prefill@2000):
  - decode: -0.5 to +0.5 %
  - prefill: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1339-g5034f45806-dirty; ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
  - B: build commit v0.0.2-1339-g5034f45806-dirty; ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin

## Result

A: `ep175` env -; B: `ep175` env -. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1462.1 / 1414.4 | 1463.6 / 1414.9 | 1438.3 (1414.4-1462.1) | 1439.2 (1414.9-1463.6) | +0.07 % [-0.13, +0.27] | **equal** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 77.3 / 76.4 | 77.4 / 76.7 | 76.9 (76.4-77.3) | 77.1 (76.7-77.4) | +0.27 % [-0.27, +0.81] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1749.2 / 1715.0 | 1750.8 / 1712.8 | 1732.1 (1715.0-1749.2) | 1731.8 (1712.8-1750.8) | -0.02 % [-0.41, +0.37] | **equal** after round 2 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 130.2 / 130.3 | 130.4 / 130.4 | 130.3 (130.2-130.3) | 130.4 (130.4-130.4) | +0.09 % [-1.21, +1.40] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.10 % | +0.14 % | +0.09 % | +0.10 % | 43 |
| 2 | BA | +0.03 % | +0.40 % | -0.13 % | +0.09 % | 46 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.07 % | -0.5 to +0.5 % | within the expectation |
| decode@2000 | +0.27 % | -0.5 to +0.5 % | within the expectation |
| prefill@2000:write | -0.02 % | -0.5 to +0.5 % | within the expectation |
| decode@2000:write | +0.09 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 128937, 23912 / 128937, 23912 | 128937, 23912 / 128937, 23912 | yes |
| 2000:write | 82220, 28057 / 82220, 28057 | 82220, 28057 / 82220, 28057 | yes |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000 | 3.56 (3.96) / 3.56 (3.96) | 3.56 (3.96) / 3.56 (3.96) |
| 2000:write | 7.82 (1.24) / 7.82 (1.24) | 7.82 (1.24) / 7.82 (1.24) |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 1.9 min for 2 rounds (0.7 min per round); a fixed 3 rounds would take 2.2 min (+0.7 min saved).
