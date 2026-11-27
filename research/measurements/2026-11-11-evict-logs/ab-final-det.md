## Plan (written before the run)

- Scope: workload server (30000:200 130000:200 at --ctx 262144); A `w103b` against B `w103j`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: bit check at the overflow size; deterministic mode is slower, 2 rounds (decode tokens per prompt as in the targets, default 200)
- Duration: 8-8 min (2-2 rounds of 3.9 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: plan E13):
  - decode: -1 to +1 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-373-g551d7004b9-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-373-g551d7004b9-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `w103b` env {'TONA_DETERMINISTIC': '1'}; B: `w103j` env {'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1879.7 / 1878.5 | 1878.3 / 1879.2 | 1879.1 (1878.5-1879.7) | 1878.8 (1878.3-1879.2) | -0.02 % [-0.10, +0.07] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 66.4 / 66.3 | 66.4 / 66.3 | 66.4 (66.3-66.4) | 66.4 (66.3-66.4) | +0.00 % [-0.52, +0.53] | **equal** after round 2 |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1737.0 / 1737.6 | 1736.2 / 1737.4 | 1737.3 (1737.0-1737.6) | 1736.8 (1736.2-1737.4) | -0.03 % [-0.07, +0.01] | **B slower** after round 1 (real, but inside the margin) |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 63.6 / 63.5 | 63.5 / 63.7 | 63.6 (63.5-63.6) | 63.6 (63.5-63.7) | +0.08 % [-0.73, +0.90] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -0.07 % | -0.01 % | -0.05 % | -0.15 % | 234 |
| 2 | BA | +0.04 % | +0.02 % | -0.01 % | +0.32 % | 234 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | -0.02 % | -1 to +1 % | within the expectation |
| decode@30000 | +0.00 % | -1 to +1 % | within the expectation |
| prefill@130000 | -0.03 % | -1 to +1 % | within the expectation |
| decode@130000 | +0.08 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 75087, 7442 / 75087, 7442 | 75087, 7442 / 75087, 7442 | yes |
| 130000 | 71507, 8403 / 71507, 8403 | 71507, 8403 / 71507, 8403 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | 5b4e7ef2b0fc2495 | 5b4e7ef2b0fc2495 | yes | yes | yes | 5b4e7ef2b0fc2495 (B = ref) |
| 130000 | 9340252c318b7217 | 9340252c318b7217 | yes | yes | yes | 9340252c318b7217 (B = ref) |

Runtime: 9.8 min for 2 rounds (3.9 min per round); a fixed 3 rounds would take 11.7 min (+3.9 min saved).
