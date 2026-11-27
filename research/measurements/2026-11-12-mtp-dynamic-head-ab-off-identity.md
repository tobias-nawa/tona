## Plan (written before the run)

- Scope: workload short (2000:400 2000:write:400 at --ctx 140000); A `tona125base` against B `tona125`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: bit identity with the switch off needs one deterministic answer, short is the smallest (decode tokens per prompt as in the targets, default 200)
- Duration: 2-6 min (2-6 rounds of 1.1 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: switch off: the code path of the head is unchanged, kernels identical; noise of short in bench/noise.toml):
  - decode: -1 to +1 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-548-gf9b1629719; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-552-gadb0ca3fa0-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `tona125base` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin', 'TONA_DETERMINISTIC': '1'}; B: `tona125` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin', 'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1333.4 / 1357.3 / 1296.7 / 1360.2 / 1315.0 / 1306.7 | 1356.8 / 1359.2 / 1335.4 / 1309.3 / 1363.1 / 1355.5 | 1324.2 (1296.7-1360.2) | 1356.1 (1309.3-1363.1) | +1.42 % [-1.74, +4.58] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 71.4 / 71.5 / 72.4 / 72.2 / 72.5 / 72.6 | 72.4 / 72.6 / 72.5 / 72.2 / 72.9 / 72.8 | 72.3 (71.4-72.6) | 72.5 (72.2-72.9) | +0.62 % [-0.14, +1.37] | **B faster** after round 2 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1612.7 / 1592.5 / 1607.0 / 1614.5 / 1604.6 / 1562.9 | 1613.8 / 1614.5 / 1561.0 / 1558.2 / 1607.1 / 1589.8 | 1605.8 (1562.9-1614.5) | 1598.4 (1558.2-1614.5) | -0.50 % [-2.91, +1.90] | open (undecided at the last round) |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 95.2 / 95.0 / 95.5 / 95.3 / 95.1 / 95.5 | 95.3 / 95.1 / 95.0 / 95.2 / 95.3 / 95.0 | 95.3 (95.0-95.5) | 95.2 (95.0-95.3) | -0.11 % [-0.81, +0.59] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +1.75 % (off plan) | +1.41 % (off plan) | +0.07 % | +0.07 % | 65 |
| 2 | BA | +0.14 % | +1.44 % | +1.38 % (outlier) | +0.17 % | 65 |
| 3 | AB | +2.99 % (outlier) | +0.05 % | -2.86 % (outlier) | -0.51 % | 65 |
| 4 | BA | -3.74 % (outlier) | -0.01 % | -3.49 % (outlier) | -0.08 % | 65 |
| 5 | AB | +3.66 % (outlier) | +0.49 % | +0.16 % | +0.22 % | 65 |
| 6 | BA | +3.73 % (outlier) | +0.31 % | +1.72 % (outlier) | -0.54 % | 65 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +1.42 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode@2000 | +0.62 % | -1 to +1 % | within the expectation |
| prefill@2000:write | -0.50 % | -1 to +1 % | within the expectation |
| decode@2000:write | -0.11 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 140099, 16693 / 140099, 16693 / 140099, 16693 / 140099, 16693 / 140099, 16693 / 140099, 16693 | 140099, 16693 / 140099, 16693 / 140099, 16693 / 140099, 16693 / 140099, 16693 / 140099, 16693 | yes |
| 2000:write | 117662, 20586 / 117662, 20586 / 117662, 20586 / 117662, 20586 / 117662, 20586 / 117662, 20586 | 117662, 20586 / 117662, 20586 / 117662, 20586 / 117662, 20586 / 117662, 20586 / 117662, 20586 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 11eacf1ee10017b4 | 11eacf1ee10017b4 | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 7.1 min for 6 rounds (1.1 min per round); a fixed 3 rounds would take 3.2 min (-3.2 min saved).
