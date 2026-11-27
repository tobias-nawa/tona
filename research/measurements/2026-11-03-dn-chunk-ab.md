## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `w30dn` against B `w30dn`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: mid (30K) and long (130K): prefill at the piece sizes of the server where the DN share was equal at 12K and 128K; decode is run by the workload and must not move (decode tokens per prompt as in the targets, default 200)
- Duration: 8-23 min (2-6 rounds of 3.8 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: 2026-11-03-dn-chunk.md Plan 2: k_dn_wave 10.3 % of prefill GPU time, kernel ratio 0.5-0.7; decode untouched):
  - prefill: +2.5 to +4.5 %
  - decode: -1 to +1 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-291-g18ad38f6e8-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-291-g18ad38f6e8-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_DN_CHUNK=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `w30dn` env {'IE_DIAG': '1'}; B: `w30dn` env {'IE_DIAG': '1', 'IE_DN_CHUNK': '1'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 6, stop: all metrics decided after round 4. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1883.5 / 1884.3 / 1882.9 / 1884.4 | 1928.1 / 1927.6 / 1927.4 / 1928.5 | 1883.9 (1882.9-1884.4) | 1927.9 (1927.4-1928.5) | +2.34 % [+2.28, +2.40] | **B faster** after round 2 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 73.7 / 72.8 / 72.7 / 73.7 | 66.7 / 66.3 / 66.4 / 66.4 | 73.3 (72.7-73.7) | 66.4 (66.3-66.7) | -9.24 % [-9.84, -8.63] | **B slower** after round 2 |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1783.5 / 1783.4 / 1784.0 / 1784.6 | 1807.7 / 1818.1 / 1818.9 / 1818.6 | 1783.7 (1783.4-1784.6) | 1818.4 (1807.7-1818.9) | +1.79 % [+1.56, +2.03] | **B faster** after round 4 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 62.9 / 62.2 / 62.7 / 62.7 | 65.7 / 67.1 / 66.2 / 66.8 | 62.7 (62.2-62.9) | 66.5 (65.7-67.1) | +6.16 % [+4.87, +7.45] | **B faster** after round 3 |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +2.37 % (off plan) | -9.51 % (off plan) | +1.36 % (off plan) | +4.56 % (off plan) | 228 |
| 2 | BA | +2.30 % | -8.95 % | +1.95 % (outlier) | +7.89 % (outlier) | 228 |
| 3 | AB | +2.36 % | -8.61 % | +1.96 % (outlier) | +5.59 % | 228 |
| 4 | BA | +2.34 % | -9.89 % | +1.91 % | +6.57 % | 228 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +2.34 % | +2.5 to +4.5 % | **OUTSIDE the expectation: finding** |
| decode@30000 | -9.24 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| prefill@130000 | +1.79 % | +2.5 to +4.5 % | **OUTSIDE the expectation: finding** |
| decode@130000 | +6.16 % | -1 to +1 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 62127, 8324 / 62127, 8324 / 62127, 8324 / 62127, 8324 | 69040, 9260 / 69040, 9260 / 69040, 9260 / 69040, 9260 | NO |
| 130000 | 72498, 8374 / 72498, 8374 / 72498, 8374 / 72498, 8374 | 66393, 8255 / 66393, 8255 / 66393, 8255 / 66393, 8255 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | 9aaec77fd9926cc1 | 017faee6fabf2f4e | yes | yes | NO | - |
| 130000 | a15fb4624df41378 | 19a5c6e838eef9fb | yes | yes | NO | - |

The texts differ (normal mode: sampling of the drafter and rounding may differ; no verdict).

Runtime: 17.1 min for 4 rounds (3.8 min per round); a fixed 3 rounds would take 11.4 min (-3.8 min saved).
