## Plan (written before the run)

- Scope: workload server (2000:400 13000:200 30000:200 at --ctx 140000); A `w100` against B `w100`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: same sizes as before; same build in both arms, only the switch differs; second attempt after fusing the weight conversion (dequant_fp8) and keeping the hc paths in f16 (decode tokens per prompt as in the targets, default 200)
- Duration: 3-10 min (2-6 rounds of 1.6 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: profile with the switch: window 6.40 s against 6.88 s (-7 %), k_gemm2 + k_gemm8 1.85 -> 1.29 s, extra k_dequant_fp8 0.14 s and k_to_fp8):
  - prefill: +5 to +10 %
  - decode: -1 to +1 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-240-g04843157d6-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-240-g04843157d6-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, IE_PREFILL_FP8=1, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `w100` env {'IE_DIAG': '1'}; B: `w100` env {'IE_DIAG': '1', 'IE_PREFILL_FP8': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1346.5 / 1347.8 / 1348.1 / 1339.5 / 1345.7 / 1346.3 | 1370.1 / 1370.3 / 1370.6 / 1366.1 / 1361.7 / 1366.9 | 1346.4 (1339.5-1348.1) | 1368.5 (1361.7-1370.6) | +1.64 % [+1.33, +1.94] | **B faster** after round 2 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 66.6 / 66.4 / 65.7 / 66.4 / 66.5 / 66.4 | 65.6 / 65.5 / 65.4 / 65.6 / 65.4 / 65.2 | 66.4 (65.7-66.6) | 65.5 (65.2-65.6) | -1.34 % [-1.86, -0.82] | **B slower** after round 2 |
| prefill@13000: Prefill tok/s after 13000 tokens of prompt | 1732.5 / 1731.8 / 1733.0 / 1731.9 / 1731.7 / 1733.0 | 1781.9 / 1782.5 / 1785.4 / 1784.0 / 1784.4 / 1783.3 | 1732.2 (1731.7-1733.0) | 1783.6 (1781.9-1785.4) | +2.96 % [+2.46, +3.46] | **B faster** after round 2 |
| decode@13000: Decode tok/s after 13000 tokens of prompt | 69.5 / 69.4 / 69.0 / 69.3 / 69.1 / 69.1 | 69.1 / 69.3 / 69.2 / 69.0 / 69.1 / 69.1 | 69.2 (69.0-69.5) | 69.1 (69.0-69.3) | -0.08 % [-0.68, +0.53] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1751.6 / 1752.5 / 1754.4 / 1785.1 / 1784.2 / 1785.0 | 1807.5 / 1808.7 / 1807.6 / 1841.6 / 1842.5 / 1842.2 | 1769.3 (1751.6-1785.1) | 1825.2 (1807.5-1842.5) | +3.18 % [+3.10, +3.25] | **B faster** after round 2 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 68.4 / 67.9 / 67.5 / 67.8 / 68.5 / 67.9 | 74.3 / 74.3 / 73.8 / 74.1 / 74.1 / 74.0 | 67.9 (67.5-68.5) | 74.1 (73.8-74.3) | +9.01 % [+8.52, +9.50] | **B faster** after round 2 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@13000 B vs A | decode@13000 B vs A | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | +1.75 % (off plan) | -1.46 % (off plan) | +2.85 % | -0.50 % | +3.19 % (off plan) | +8.58 % (off plan) | 98 |
| 2 | BA | +1.67 % | -1.34 % | +2.93 % | -0.10 % | +3.21 % | +9.44 % | 98 |
| 3 | AB | +1.67 % | -0.57 % | +3.02 % | +0.42 % | +3.03 % (outlier) | +9.41 % | 98 |
| 4 | BA | +1.99 % | -1.26 % | +3.01 % | -0.32 % | +3.17 % | +9.29 % | 98 |
| 5 | AB | +1.19 % (outlier) | -1.64 % | +3.04 % | -0.04 % | +3.26 % | +8.26 % | 98 |
| 6 | BA | +1.53 % | -1.76 % | +2.90 % | +0.06 % | +3.20 % | +9.07 % | 98 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +1.64 % | +5 to +10 % | **OUTSIDE the expectation: finding** |
| decode@2000 | -1.34 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| prefill@13000 | +2.96 % | +5 to +10 % | **OUTSIDE the expectation: finding** |
| decode@13000 | -0.08 % | -1 to +1 % | within the expectation |
| prefill@30000 | +3.18 % | +5 to +10 % | **OUTSIDE the expectation: finding** |
| decode@30000 | +9.01 % | -1 to +1 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 | 112921, 14181 / 112921, 14181 / 112921, 14181 / 112921, 14181 / 112921, 14181 / 112921, 14181 | NO |
| 13000 | 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 | 70124, 7456 / 70124, 7456 / 70124, 7456 / 70124, 7456 / 70124, 7456 / 70124, 7456 | NO |
| 30000 | 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 | 62958, 7489 / 62958, 7489 / 62958, 7489 / 62958, 7489 / 62958, 7489 / 62958, 7489 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | f6573f396c15300f | a1c43447d6e6c021 | yes | yes | NO | - |
| 13000 | 4d9fec82a483ee97 | d3a8e69dc4cfb94d | yes | yes | NO | - |
| 30000 | 1fad1ead2f1ac34c | 9aaec77fd9926cc1 | yes | yes | NO | - |

The texts differ (normal mode: sampling of the drafter and rounding may differ; no verdict).

Runtime: 10.7 min for 6 rounds (1.6 min per round); a fixed 3 rounds would take 4.9 min (-4.9 min saved).
