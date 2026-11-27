## Plan (written before the run)

- Scope: workload server (2000:400 13000:200 30000:200 at --ctx 140000); A `w100` against B `w100`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: same sizes as run 3: 13K is the profile size, 2K and 30K have known noise; same build in both arms, only the switch differs (decode tokens per prompt as in the targets, default 200)
- Duration: 3-10 min (2-6 rounds of 1.7 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: run 4 plan in research/measurements/2026-11-02-prefill-int8.md (dense = 26 % of the window, kernel 1.3-1.6x)):
  - prefill: +8 to +15 %
  - decode: -1 to +1 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-240-g04843157d6; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-240-g04843157d6; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, IE_PREFILL_FP8=1, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `w100` env {'IE_DIAG': '1'}; B: `w100` env {'IE_DIAG': '1', 'IE_PREFILL_FP8': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: all metrics decided after round 6. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1349.1 / 1350.3 / 1348.7 / 1347.7 / 1349.4 / 1347.6 | 1278.7 / 1278.5 / 1280.0 / 1280.8 / 1282.0 / 1279.9 | 1348.9 (1347.6-1350.3) | 1279.9 (1278.5-1282.0) | -5.10 % [-5.28, -4.92] | **B slower** after round 2 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 67.0 / 66.5 / 66.1 / 65.9 / 66.1 / 66.2 | 67.4 / 67.8 / 67.7 / 67.8 / 67.6 / 67.6 | 66.2 (65.9-67.0) | 67.7 (67.4-67.8) | +2.05 % [+1.19, +2.90] | **B faster** after round 4 |
| prefill@13000: Prefill tok/s after 13000 tokens of prompt | 1736.8 / 1731.6 / 1731.5 / 1735.0 / 1731.4 / 1733.2 | 1615.4 / 1612.8 / 1611.7 / 1614.0 / 1613.5 / 1610.5 | 1732.4 (1731.4-1736.8) | 1613.1 (1610.5-1615.4) | -6.94 % [-7.44, -6.44] | **B slower** after round 2 |
| decode@13000: Decode tok/s after 13000 tokens of prompt | 69.8 / 68.9 / 69.4 / 68.7 / 69.4 / 69.0 | 69.8 / 69.6 / 69.6 / 69.4 / 69.9 / 69.7 | 69.2 (68.7-69.8) | 69.7 (69.4-69.9) | +0.69 % [+0.01, +1.36] | **B faster** after round 6 |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1784.0 / 1780.7 / 1780.6 / 1782.0 / 1781.1 / 1784.4 | 1636.0 / 1631.2 / 1632.6 / 1633.4 / 1634.8 / 1635.0 | 1781.6 (1780.6-1784.4) | 1634.1 (1631.2-1636.0) | -8.32 % [-8.39, -8.25] | **B slower** after round 2 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 68.0 / 68.0 / 68.0 / 67.3 / 68.1 / 67.9 | 76.8 / 76.6 / 76.6 / 77.1 / 76.8 / 77.0 | 68.0 (67.3-68.1) | 76.8 (76.6-77.1) | +13.18 % [+12.53, +13.82] | **B faster** after round 2 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@13000 B vs A | decode@13000 B vs A | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | -5.22 % (off plan) | +0.67 % | -6.99 % | +0.04 % | -8.30 % (off plan) | +12.93 % (off plan) | 99 |
| 2 | BA | -5.32 % | +1.95 % (outlier) | -6.86 % | +1.03 % | -8.40 % | +12.69 % | 99 |
| 3 | AB | -5.09 % | +2.48 % (outlier) | -6.92 % | +0.32 % | -8.31 % | +12.71 % | 99 |
| 4 | BA | -4.96 % | +2.86 % | -6.98 % | +1.03 % | -8.34 % | +14.56 % (outlier) | 99 |
| 5 | AB | -5.00 % | +2.24 % | -6.81 % | +0.73 % | -8.21 % | +12.87 % | 99 |
| 6 | BA | -5.03 % | +2.10 % | -7.08 % | +0.97 % | -8.37 % | +13.30 % | 99 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | -5.10 % | +8 to +15 % | **OUTSIDE the expectation: finding** |
| decode@2000 | +2.05 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| prefill@13000 | -6.94 % | +8 to +15 % | **OUTSIDE the expectation: finding** |
| decode@13000 | +0.69 % | -1 to +1 % | within the expectation |
| prefill@30000 | -8.32 % | +8 to +15 % | **OUTSIDE the expectation: finding** |
| decode@30000 | +13.18 % | -1 to +1 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 | 95057, 11815 / 95057, 11815 / 95057, 11815 / 95057, 11815 / 95057, 11815 / 95057, 11815 | NO |
| 13000 | 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 | 69633, 7181 / 69633, 7181 / 69633, 7181 / 69633, 7181 / 69633, 7181 / 69633, 7181 | NO |
| 30000 | 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 | 60441, 7635 / 60441, 7635 / 60441, 7635 / 60441, 7635 / 60441, 7635 / 60441, 7635 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | f6573f396c15300f | a5590f5fc514a92a | yes | yes | NO | - |
| 13000 | 4d9fec82a483ee97 | 9be5085a06be0ba9 | yes | yes | NO | - |
| 30000 | 1fad1ead2f1ac34c | 23c47234fa088171 | yes | yes | NO | - |

The texts differ (normal mode: sampling of the drafter and rounding may differ; no verdict).

Runtime: 10.8 min for 6 rounds (1.7 min per round); a fixed 3 rounds would take 5.0 min (-5.0 min saved).
