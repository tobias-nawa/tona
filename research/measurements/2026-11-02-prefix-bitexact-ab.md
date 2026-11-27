## Plan (written before the run)

- Scope: workload mid,short (2000:400 2000:write:400 30000:200 at --ctx 140000); A `w81-base` against B `w81`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: mid (30K prompt) is the prefill the cold path of a long prompt takes; short (2K) is the decode loop; the change touches neither, so both must stay equal (decode tokens per prompt as in the targets, default 200)
- Duration: 3-10 min (2-6 rounds of 1.7 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: cold requests of a single message write no checkpoint (docs; section 3 of the plan), the code they run is unchanged; margin of docs/contribute/measurement.md):
  - prefill: -0.5 to +0.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-204-g98dc08946e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-208-ga810892b7a; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `w81-base` env -; B: `w81` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1329.6 / 1327.4 / 1328.9 / 1329.2 / 1328.8 / 1330.0 | 1328.0 / 1327.8 / 1329.7 / 1332.2 / 1328.6 / 1329.3 | 1329.1 (1327.4-1330.0) | 1329.0 (1327.8-1332.2) | +0.02 % [-0.14, +0.18] | **equal** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 66.1 / 66.2 / 66.2 / 66.1 / 66.1 / 66.1 | 66.1 / 66.0 / 66.1 / 66.0 / 66.1 / 66.1 | 66.1 (66.1-66.2) | 66.1 (66.0-66.1) | -0.08 % [-0.34, +0.19] | **equal** after round 4 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1543.8 / 1542.3 / 1540.4 / 1543.0 / 1544.1 / 1544.3 | 1543.9 / 1545.0 / 1542.8 / 1542.2 / 1541.2 / 1543.6 | 1543.4 (1540.4-1544.3) | 1543.2 (1541.2-1545.0) | +0.01 % [-0.21, +0.23] | **equal** after round 2 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 119.4 / 118.8 / 118.9 / 118.7 / 118.5 / 119.2 | 119.5 / 119.1 / 119.0 / 118.5 / 118.4 / 118.5 | 118.9 (118.5-119.4) | 118.8 (118.4-119.5) | -0.07 % [-0.74, +0.60] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1780.1 / 1779.2 / 1780.2 / 1779.9 / 1779.3 / 1779.1 | 1785.0 / 1783.8 / 1783.7 / 1784.0 / 1783.7 / 1784.5 | 1779.6 (1779.1-1780.2) | 1783.9 (1783.7-1785.0) | +0.25 % [+0.20, +0.30] | **B faster** after round 1 (real, but inside the margin) |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 69.9 / 70.2 / 70.1 / 70.1 / 70.3 / 70.0 | 66.2 / 67.3 / 67.3 / 67.1 / 67.3 / 67.3 | 70.1 (69.9-70.3) | 67.3 (66.2-67.3) | -4.31 % [-4.82, -3.81] | **B slower** after round 3 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | -0.12 % | +0.01 % | +0.01 % | +0.09 % | +0.28 % | -5.31 % (off plan) | 102 |
| 2 | BA | +0.03 % | -0.17 % | +0.18 % | +0.21 % | +0.26 % | -4.16 % (outlier) | 102 |
| 3 | AB | +0.06 % | -0.22 % | +0.16 % | +0.09 % | +0.20 % | -3.96 % | 102 |
| 4 | BA | +0.22 % | -0.07 % | -0.05 % | -0.12 % | +0.23 % | -4.29 % | 102 |
| 5 | AB | -0.01 % | +0.10 % | -0.19 % | -0.11 % | +0.25 % | -4.30 % | 102 |
| 6 | BA | -0.05 % | -0.11 % | -0.05 % | -0.58 % | +0.30 % | -3.87 % | 102 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.02 % | -0.5 to +0.5 % | within the expectation |
| decode@2000 | -0.08 % | -0.5 to +0.5 % | within the expectation |
| prefill@2000:write | +0.01 % | -0.5 to +0.5 % | within the expectation |
| decode@2000:write | -0.07 % | -0.5 to +0.5 % | within the expectation |
| prefill@30000 | +0.25 % | -0.5 to +0.5 % | within the expectation |
| decode@30000 | -4.31 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 | 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 | yes |
| 2000:write | 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 | 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 | yes |
| 30000 | 71024, 10531 / 71024, 10531 / 71024, 10531 / 71024, 10531 / 71024, 10531 / 71024, 10531 | 67023, 10509 / 67023, 10509 / 67023, 10509 / 67023, 10509 / 67023, 10509 / 67023, 10509 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | f6573f396c15300f | f6573f396c15300f | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | cde3f1093dce48a1 | 1fad1ead2f1ac34c | yes | yes | NO | - |

The texts differ (normal mode: sampling of the drafter and rounding may differ; no verdict).

Runtime: 11.1 min for 6 rounds (1.7 min per round); a fixed 3 rounds would take 5.1 min (-5.1 min saved).
