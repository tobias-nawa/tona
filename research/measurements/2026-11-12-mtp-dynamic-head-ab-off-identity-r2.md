## Plan (written before the run)

- Scope: workload short (2000:400 2000:write:400 at --ctx 140000); A `tona125base` against B `tona125`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: bit identity with TONA_MTP_SCORER=0 needs one deterministic answer, short is the smallest (decode tokens per prompt as in the targets, default 200)
- Duration: 2-6 min (2-6 rounds of 1.1 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: switch off: head path unchanged):
  - decode: -1 to +1 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-625-g1f17a936ef; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-629-g6d05ae3709; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin, TONA_MTP_SCORER=0

## Result

A: `tona125base` env {'TONA_DETERMINISTIC': '1', 'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin'}; B: `tona125` env {'TONA_DETERMINISTIC': '1', 'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin', 'TONA_MTP_SCORER': '0'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1349.5 / 1336.3 / 1340.5 / 1353.9 / 1333.5 / 1350.0 | 1363.4 / 1347.0 / 1361.5 / 1359.8 / 1363.3 / 1350.4 | 1345.0 (1333.5-1353.9) | 1360.7 (1347.0-1363.4) | +1.02 % [+0.14, +1.90] | **B faster** after round 2 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 72.7 / 71.9 / 72.5 / 71.5 / 72.9 / 73.0 | 72.4 / 72.8 / 73.2 / 73.1 / 73.3 / 73.1 | 72.6 (71.5-73.0) | 73.1 (72.4-73.3) | +0.80 % [-0.27, +1.87] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1612.1 / 1607.7 / 1608.4 / 1607.8 / 1605.5 / 1611.8 | 1614.8 / 1612.3 / 1607.8 / 1603.1 / 1613.1 / 1613.1 | 1608.1 (1605.5-1612.1) | 1612.7 (1603.1-1614.8) | +0.11 % [-0.22, +0.45] | **equal** after round 3 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 95.2 / 95.4 / 94.8 / 95.5 / 95.6 / 95.5 | 95.3 / 95.3 / 95.3 / 95.5 / 95.3 / 95.7 | 95.4 (94.8-95.6) | 95.3 (95.3-95.7) | +0.08 % [-0.60, +0.76] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +1.03 % (off plan) | -0.39 % | +0.17 % | +0.12 % | 65 |
| 2 | BA | +0.80 % | +1.22 % (outlier) | +0.29 % | -0.15 % | 65 |
| 3 | AB | +1.57 % (outlier) | +1.05 % | -0.04 % | +0.56 % | 65 |
| 4 | BA | +0.44 % | +2.32 % (outlier) | -0.30 % | -0.02 % | 65 |
| 5 | AB | +2.23 % (outlier) | +0.49 % | +0.47 % | -0.28 % | 65 |
| 6 | BA | +0.03 % | +0.12 % | +0.08 % | +0.25 % | 65 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +1.02 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode@2000 | +0.80 % | -1 to +1 % | within the expectation |
| prefill@2000:write | +0.11 % | -1 to +1 % | within the expectation |
| decode@2000:write | +0.08 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 139702, 17142 / 139702, 17142 / 139702, 17142 / 139702, 17142 / 139702, 17142 / 139702, 17142 | 139702, 17142 / 139702, 17142 / 139702, 17142 / 139702, 17142 / 139702, 17142 / 139702, 17142 | yes |
| 2000:write | 117123, 21125 / 117123, 21125 / 117123, 21125 / 117123, 21125 / 117123, 21125 / 117123, 21125 | 117123, 21125 / 117123, 21125 / 117123, 21125 / 117123, 21125 / 117123, 21125 / 117123, 21125 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 11eacf1ee10017b4 | 11eacf1ee10017b4 | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 7.0 min for 6 rounds (1.1 min per round); a fixed 3 rounds would take 3.2 min (-3.2 min saved).
