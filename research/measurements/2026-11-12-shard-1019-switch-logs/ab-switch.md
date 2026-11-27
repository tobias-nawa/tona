## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `shard` against B `shard`; rounds 2 to 3, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: mid (30K) for prefill at full groups, long (130K) where attention dominates; the shards share shapes and types, so these two rates decide whether the switch changed speed (decode tokens per prompt as in the targets, default 200)
- Duration: 8-12 min (2-3 rounds of 3.9 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: 2026-11-12-coder-1019.md section 2a (4 rounds, old against new shard)):
  - prefill: -0.1 to +0.1 %
  - decode@30000: +6 to +10 %
  - decode@130000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-490-gb6dd58cd44; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder-0929.bin
  - B: build commit v0.0.2-490-gb6dd58cd44; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `shard` env {'TONA_DETERMINISTIC': '1', 'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder-0929.bin'}; B: `shard` env {'TONA_DETERMINISTIC': '1', 'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin'}. Alpha 0.05, margin 0.5 %, rounds 3 of at most 3, stop: 3 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1878.3 / 1892.4 / 1893.4 | 1884.5 / 1884.8 / 1891.6 | 1892.4 (1878.3-1893.4) | 1884.8 (1884.5-1891.6) | -0.05 % [-0.33, +0.23] | **equal** after round 3 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 66.1 / 66.0 / 66.7 | 72.2 / 72.9 / 73.2 | 66.1 (66.0-66.7) | 72.9 (72.2-73.2) | +9.74 % [+9.07, +10.41] | **B faster** after round 1 |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1747.2 / 1753.4 / 1755.3 | 1749.6 / 1748.9 / 1755.2 | 1753.4 (1747.2-1755.3) | 1749.6 (1748.9-1755.2) | -0.04 % [-0.19, +0.11] | **equal** after round 3 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 63.8 / 63.9 / 64.4 | 63.4 / 64.9 / 64.7 | 63.9 (63.8-64.4) | 64.7 (63.4-64.9) | +0.49 % [-0.57, +1.56] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.33 % (off plan) | +9.09 % | +0.14 % (off plan) | -0.64 % | 235 |
| 2 | BA | -0.40 % (outlier) | +10.45 % (outlier) | -0.26 % (outlier) | +1.60 % (outlier) | 237 |
| 3 | AB | -0.09 % | +9.70 % | -0.00 % | +0.53 % | 232 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | -0.05 % | -0.1 to +0.1 % | within the expectation |
| decode@30000 | +9.74 % | +6 to +10 % | within the expectation |
| prefill@130000 | -0.04 % | -0.1 to +0.1 % | within the expectation |
| decode@130000 | +0.49 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 75602, 6927 / 75706, 6823 / 75706, 6823 | 70386, 7308 / 70386, 7308 / 70476, 7218 | NO |
| 130000 | 72219, 7691 / 72351, 7559 / 72351, 7559 | 77516, 6497 / 77516, 6497 / 77626, 6387 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | 5b4e7ef2b0fc2495 | ee58880d3f5b558f | yes | yes | NO | 5b4e7ef2b0fc2495 (B DIFFERS) |
| 130000 | 9340252c318b7217 | 7d2996546076d1b0 | yes | yes | NO | 9340252c318b7217 (B DIFFERS) |

**WARNING: deterministic mode, but the texts differ between rounds, arms or from the reference of main. The change is not bit-exact (or the run is not deterministic).**

Runtime: 13.7 min for 3 rounds (3.9 min per round); a fixed 3 rounds would take 11.7 min (+0.0 min saved).
