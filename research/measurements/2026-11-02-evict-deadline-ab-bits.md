## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `w85-base` against B `w85`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (decode) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: R4: the 30K and 130K requests of bench/sha.toml (decode tokens per prompt as in the targets, default 200)
- Duration: 8-8 min (2-2 rounds of 3.9 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: bits must be equal (bench/sha.toml); rates of two rounds are not judged):
  - decode: -3 to +3 %
  - prefill: -3 to +3 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-204-g98dc08946e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DETERMINISTIC=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-210-ge338d14fbd; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DETERMINISTIC=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `w85-base` env {'IE_DETERMINISTIC': '1'}; B: `w85` env {'IE_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1793.6 / 1793.8 | 1793.3 / 1793.7 | 1793.7 (1793.6-1793.8) | 1793.5 (1793.3-1793.7) | -0.01 % [-0.09, +0.06] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 71.2 / 70.4 | 71.9 / 70.9 | 70.8 (70.4-71.2) | 71.4 (70.9-71.9) | +0.82 % [+0.28, +1.37] | **B faster** after round 1 |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1713.9 / 1712.1 | 1713.2 / 1712.6 | 1713.0 (1712.1-1713.9) | 1712.9 (1712.6-1713.2) | -0.00 % [-0.05, +0.04] | **equal** after round 1 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 65.0 / 65.3 | 65.7 / 65.7 | 65.2 (65.0-65.3) | 65.7 (65.7-65.7) | +0.81 % [+0.00, +1.61] | **B faster** after round 2 |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -0.02 % | +1.01 % | -0.04 % | +0.99 % | 237 |
| 2 | BA | -0.01 % | +0.64 % | +0.03 % | +0.62 % | 237 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | -0.01 % | -3 to +3 % | within the expectation |
| decode@30000 | +0.82 % | -3 to +3 % | within the expectation |
| prefill@130000 | -0.00 % | -3 to +3 % | within the expectation |
| decode@130000 | +0.81 % | -3 to +3 % | within the expectation |

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
