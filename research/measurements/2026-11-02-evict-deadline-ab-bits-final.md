## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `w85-base` against B `w85`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (decode) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: R4: the 30K and 130K requests of bench/sha.toml (decode tokens per prompt as in the targets, default 200)
- Duration: 8-8 min (2-2 rounds of 3.9 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: bits must be equal (bench/sha.toml)):
  - decode: -3 to +3 %
  - prefill: -3 to +3 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-204-g98dc08946e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DETERMINISTIC=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-214-g56811a4f2d; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DETERMINISTIC=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `w85-base` env {'IE_DETERMINISTIC': '1'}; B: `w85` env {'IE_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1801.3 / 1801.8 | 1802.6 / 1802.5 | 1801.6 (1801.3-1801.8) | 1802.6 (1802.5-1802.6) | +0.06 % [-0.02, +0.13] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 69.1 / 71.4 | 71.5 / 71.7 | 70.3 (69.1-71.4) | 71.6 (71.5-71.7) | +1.94 % [+0.43, +3.45] | **B faster** after round 2 |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1711.3 / 1712.1 | 1712.4 / 1712.2 | 1711.7 (1711.3-1712.1) | 1712.3 (1712.2-1712.4) | +0.03 % [-0.01, +0.07] | **B faster** after round 1 (real, but inside the margin) |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 64.4 / 65.3 | 65.6 / 64.9 | 64.8 (64.4-65.3) | 65.3 (64.9-65.6) | +0.70 % [-0.64, +2.04] | **B faster** after round 1 |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.07 % | +3.55 % (off plan) | +0.06 % | +1.93 % | 237 |
| 2 | BA | +0.04 % | +0.34 % | +0.01 % | -0.53 % | 237 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +0.06 % | -3 to +3 % | within the expectation |
| decode@30000 | +1.94 % | -3 to +3 % | within the expectation |
| prefill@130000 | +0.03 % | -3 to +3 % | within the expectation |
| decode@130000 | +0.70 % | -3 to +3 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 67129, 8953 / 67129, 8953 | 67129, 8953 / 67129, 8953 | yes |
| 130000 | 76065, 11270 / 76065, 11270 | 76065, 11270 / 76065, 11270 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | 08f0bd31e0c0277f | 08f0bd31e0c0277f | yes | yes | yes | 08f0bd31e0c0277f (B = ref) |
| 130000 | e9545d70f2400b87 | e9545d70f2400b87 | yes | yes | yes | e9545d70f2400b87 (B = ref) |

Runtime: 9.9 min for 2 rounds (3.9 min per round); a fixed 3 rounds would take 11.8 min (+3.9 min saved).
