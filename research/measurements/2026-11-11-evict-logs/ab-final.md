## Plan (written before the run)

- Scope: workload server (30000:200 130000:200 at --ctx 262144); A `w103b` against B `w103j`; rounds 3 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 30K and 130K are the R8 sizes; ctx 262144 has 4.1 GB registered overflow (decode tokens per prompt as in the targets, default 200)
- Duration: 12-16 min (3-4 rounds of 3.9 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: plan E13 (E12 measured +-0.1 %)):
  - decode: -1 to +1 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-373-g551d7004b9; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-373-g551d7004b9; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `w103b` env -; B: `w103j` env -. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1881.0 / 1879.4 / 1878.4 / 1878.6 | 1879.4 / 1880.1 / 1877.9 / 1878.3 | 1879.0 (1878.4-1881.0) | 1878.9 (1877.9-1880.1) | -0.02 % [-0.09, +0.05] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 66.6 / 66.6 / 66.7 / 66.6 | 66.6 / 66.6 / 66.7 / 66.7 | 66.6 (66.6-66.7) | 66.7 (66.6-66.7) | +0.04 % [-0.34, +0.42] | **equal** after round 3 |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1737.1 / 1738.5 / 1736.6 / 1736.7 | 1737.0 / 1736.3 / 1736.5 / 1736.2 | 1736.9 (1736.6-1738.5) | 1736.4 (1736.2-1737.0) | -0.04 % [-0.09, +0.01] | **equal** after round 1 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 67.9 / 67.9 / 67.8 / 68.0 | 67.9 / 68.0 / 67.9 / 67.8 | 67.9 (67.8-68.0) | 67.9 (67.8-68.0) | +0.01 % [-0.57, +0.59] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -0.08 % | +0.07 % | -0.00 % | +0.06 % | 234 |
| 2 | BA | +0.04 % | +0.04 % | -0.13 % | +0.05 % | 234 |
| 3 | AB | -0.02 % | -0.02 % | -0.01 % | +0.12 % | 234 |
| 4 | BA | -0.01 % | +0.08 % | -0.03 % | -0.19 % | 234 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | -0.02 % | -1 to +1 % | within the expectation |
| decode@30000 | +0.04 % | -1 to +1 % | within the expectation |
| prefill@130000 | -0.04 % | -1 to +1 % | within the expectation |
| decode@130000 | +0.01 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 71501, 6925 / 71501, 6925 / 71501, 6925 / 71501, 6925 | 71501, 6925 / 71501, 6925 / 71501, 6925 / 71501, 6925 | yes |
| 130000 | 65599, 8523 / 65599, 8523 / 65599, 8523 / 65599, 8523 | 65599, 8523 / 65599, 8523 / 65599, 8523 / 65599, 8523 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | 3a5ff5033f8f119b | 3a5ff5033f8f119b | yes | yes | yes | - |
| 130000 | 971ebf40510eb261 | 971ebf40510eb261 | yes | yes | yes | - |

Runtime: 17.6 min for 4 rounds (3.9 min per round); a fixed 3 rounds would take 11.7 min (-3.9 min saved).
