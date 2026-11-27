## Plan (written before the run)

- Scope: workload server (2000:400 13000:200 30000:200 at --ctx 140000); A `merged` against B `w100`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 13K is the size of the profile and of agent turns; 2K and 30K are the sizes with known noise (noise.toml); decode is checked at every size (decode tokens per prompt as in the targets, default 200)
- Duration: 3-10 min (2-6 rounds of 1.7 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: run 3 plan in research/measurements/2026-11-02-prefill-int8.md (profile 7.41 s -> window -5 %)):
  - prefill: +2 to +8 %
  - decode: -1 to +1 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-234-gbcde1e5836; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-239-g83b28c9782; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `merged` env {'IE_DIAG': '1'}; B: `w100` env {'IE_DIAG': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1313.4 / 1328.2 / 1324.4 / 1327.0 / 1321.8 / 1323.7 | 1333.8 / 1344.6 / 1344.1 / 1340.9 / 1342.4 / 1340.0 | 1324.1 (1313.4-1328.2) | 1341.6 (1333.8-1344.6) | +1.35 % [+1.10, +1.60] | **B faster** after round 2 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 66.6 / 66.8 / 66.4 / 67.1 / 65.5 / 65.8 | 66.7 / 67.1 / 65.8 / 66.7 / 65.5 / 65.6 | 66.5 (65.5-67.1) | 66.3 (65.5-67.1) | -0.19 % [-0.75, +0.37] | open (undecided at the last round) |
| prefill@13000: Prefill tok/s after 13000 tokens of prompt | 1703.1 / 1716.1 / 1718.8 / 1719.0 / 1717.0 / 1717.0 | 1731.5 / 1735.3 / 1735.7 / 1735.1 / 1734.5 / 1733.9 | 1717.0 (1703.1-1719.0) | 1734.8 (1731.5-1735.7) | +1.12 % [+0.54, +1.70] | **B faster** after round 2 |
| decode@13000: Decode tok/s after 13000 tokens of prompt | 69.1 / 69.9 / 68.6 / 70.2 / 67.8 / 67.9 | 69.0 / 69.8 / 68.9 / 69.7 / 67.8 / 67.9 | 68.9 (67.8-70.2) | 69.0 (67.8-69.8) | -0.08 % [-0.75, +0.58] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1763.2 / 1765.9 / 1769.5 / 1769.3 / 1767.3 / 1769.1 | 1783.2 / 1783.8 / 1783.7 / 1784.2 / 1784.2 / 1782.2 | 1768.2 (1763.2-1769.5) | 1783.8 (1782.2-1784.2) | +0.91 % [+0.79, +1.04] | **B faster** after round 2 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 68.2 / 68.6 / 67.4 / 69.1 / 66.7 / 67.6 | 67.8 / 68.9 / 67.8 / 68.3 / 66.6 / 67.6 | 67.9 (66.7-69.1) | 67.8 (66.6-68.9) | -0.09 % [-0.69, +0.51] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@13000 B vs A | decode@13000 B vs A | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | +1.55 % (off plan) | +0.07 % | +1.67 % | -0.17 % | +1.13 % (off plan) | -0.59 % | 100 |
| 2 | BA | +1.24 % | +0.40 % | +1.12 % | -0.16 % | +1.01 % | +0.47 % | 100 |
| 3 | AB | +1.49 % | -0.86 % | +0.98 % | +0.51 % | +0.80 % (outlier) | +0.64 % | 100 |
| 4 | BA | +1.05 % (outlier) | -0.59 % | +0.94 % | -0.73 % | +0.84 % (outlier) | -1.08 % (outlier) | 99 |
| 5 | AB | +1.56 % | +0.04 % | +1.02 % | +0.11 % | +0.96 % | -0.05 % | 100 |
| 6 | BA | +1.23 % | -0.19 % | +0.99 % | -0.05 % | +0.74 % (outlier) | +0.08 % | 100 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +1.35 % | +2 to +8 % | **OUTSIDE the expectation: finding** |
| decode@2000 | -0.19 % | -1 to +1 % | within the expectation |
| prefill@13000 | +1.12 % | +2 to +8 % | **OUTSIDE the expectation: finding** |
| decode@13000 | -0.08 % | -1 to +1 % | within the expectation |
| prefill@30000 | +0.91 % | +2 to +8 % | **OUTSIDE the expectation: finding** |
| decode@30000 | -0.09 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 | 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 / 134222, 17474 | yes |
| 13000 | 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 | 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 / 70584, 7888 | yes |
| 30000 | 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 | 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 / 69329, 8203 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | f6573f396c15300f | f6573f396c15300f | yes | yes | yes | - |
| 13000 | 4d9fec82a483ee97 | 4d9fec82a483ee97 | yes | yes | yes | - |
| 30000 | 1fad1ead2f1ac34c | 1fad1ead2f1ac34c | yes | yes | yes | - |

Runtime: 10.8 min for 6 rounds (1.7 min per round); a fixed 3 rounds would take 5.0 min (-5.0 min saved).
