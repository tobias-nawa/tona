## Plan (written before the run)

- Scope: workload server (30000:200 130000:200 at --ctx 262144); A `w103b` against B `w103h`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 30K and 130K are the R8 sizes; ctx 262144 is the service default with 4.1 GB of registered overflow (decode tokens per prompt as in the targets, default 200)
- Duration: 8-16 min (2-4 rounds of 3.9 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: placement change only (plan E8)):
  - decode: -1 to +1 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-369-g437b3ff687; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-369-g437b3ff687; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `w103b` env -; B: `w103h` env -. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1877.8 / 1877.4 / 1878.4 / 1878.1 | 1878.8 / 1877.7 / 1878.1 / 1877.6 | 1878.0 (1877.4-1878.4) | 1877.9 (1877.6-1878.8) | +0.01 % [-0.05, +0.07] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 66.6 / 66.5 / 66.7 / 66.7 | 48.7 / 46.4 / 52.6 / 46.6 | 66.6 (66.5-66.7) | 47.6 (46.4-52.6) | -27.07 % [-30.40, -23.74] | open (undecided at the last round) |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1736.2 / 1736.2 / 1737.3 / 1736.2 | 1736.3 / 1736.0 / 1737.3 / 1736.1 | 1736.2 (1736.2-1737.3) | 1736.2 (1736.0-1737.3) | -0.00 % [-0.03, +0.02] | **equal** after round 1 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 68.0 / 68.1 / 67.9 / 68.0 | 37.1 / 52.2 / 47.2 / 50.9 | 68.0 (67.9-68.1) | 49.1 (37.1-52.2) | -31.10 % [-38.95, -23.26] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.05 % | -26.88 % (off plan) | +0.00 % | -45.38 % (off plan) | 237 |
| 2 | BA | +0.02 % | -30.13 % (outlier) | -0.01 % | -23.40 % (outlier) | 236 |
| 3 | AB | -0.01 % | -21.16 % (outlier) | +0.00 % | -30.45 % (outlier) | 236 |
| 4 | BA | -0.03 % | -30.12 % (outlier) | -0.00 % | -25.18 % (outlier) | 236 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +0.01 % | -1 to +1 % | within the expectation |
| decode@30000 | -27.07 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| prefill@130000 | -0.00 % | -1 to +1 % | within the expectation |
| decode@130000 | -31.10 % | -1 to +1 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 71501, 6925 / 71501, 6925 / 71501, 6925 / 71501, 6925 | 71501, 6925 / 71501, 6925 / 71501, 6925 / 71501, 6925 | yes |
| 130000 | 65599, 8523 / 65599, 8523 / 65599, 8523 / 65599, 8523 | 65599, 8523 / 65599, 8523 / 65599, 8523 / 65599, 8523 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | 3a5ff5033f8f119b | 3a5ff5033f8f119b | yes | yes | yes | - |
| 130000 | 971ebf40510eb261 | 971ebf40510eb261 | yes | yes | yes | - |

Runtime: 17.7 min for 4 rounds (3.9 min per round); a fixed 3 rounds would take 11.8 min (-3.9 min saved).
