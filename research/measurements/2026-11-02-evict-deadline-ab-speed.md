## Plan (written before the run)

- Scope: workload short,mid (2000:400 2000:write:400 30000:200 at --ctx 140000); A `w85-base` against B `w85`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (decode) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: short = decode at 2K (the barrier code in the decode loop), mid = prefill through k_mega<8>, the kernel with the largest spill change (decode tokens per prompt as in the targets, default 200)
- Duration: 3-10 min (2-6 rounds of 1.7 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: ticket #85 step 3; risk 2026-10-28-mega-spills.md):
  - decode: -0.5 to +0.5 %
  - prefill: -0.5 to +0.5 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-204-g98dc08946e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-207-g1c88a9a06f; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `w85-base` env -; B: `w85` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1329.0 / 1313.6 / 1330.0 / 1329.1 / 1327.6 / 1328.5 | 1324.0 / 1312.3 / 1323.0 / 1329.8 / 1328.5 / 1328.4 | 1328.8 (1313.6-1330.0) | 1326.2 (1312.3-1329.8) | -0.15 % [-0.43, +0.13] | **B slower** after round 3 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 66.1 / 64.6 / 65.9 / 66.0 / 65.9 / 66.0 | 66.5 / 66.1 / 66.2 / 66.4 / 66.4 / 66.5 | 66.0 (64.6-66.1) | 66.4 (66.1-66.5) | +0.88 % [+0.11, +1.66] | **B faster** after round 5 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1543.9 / 1532.3 / 1544.7 / 1543.0 / 1543.6 / 1541.8 | 1541.9 / 1535.4 / 1540.7 / 1544.1 / 1542.8 / 1540.5 | 1543.3 (1532.3-1544.7) | 1541.3 (1535.4-1544.1) | -0.04 % [-0.28, +0.20] | **equal** after round 2 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 118.4 / 118.1 / 119.1 / 118.3 / 117.5 / 118.6 | 119.0 / 118.4 / 118.5 / 118.5 / 119.1 / 118.6 | 118.3 (117.5-119.1) | 118.6 (118.4-119.1) | +0.31 % [-0.58, +1.20] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1779.3 / 1777.6 / 1779.4 / 1779.7 / 1779.4 / 1778.5 | 1779.7 / 1777.7 / 1779.8 / 1779.4 / 1779.8 / 1780.4 | 1779.4 (1777.6-1779.7) | 1779.7 (1777.7-1780.4) | +0.03 % [-0.03, +0.08] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 69.8 / 68.9 / 70.0 / 70.1 / 70.1 / 69.9 | 70.6 / 69.5 / 70.5 / 70.2 / 70.5 / 70.5 | 70.0 (68.9-70.1) | 70.5 (69.5-70.6) | +0.73 % [+0.32, +1.14] | **B faster** after round 2 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | -0.38 % | +0.67 % (off plan) | -0.13 % | +0.56 % (off plan) | +0.02 % | +1.18 % (off plan) | 102 |
| 2 | BA | -0.10 % | +2.23 % (outlier) | +0.20 % | +0.30 % | +0.00 % | +0.95 % | 103 |
| 3 | AB | -0.52 % | +0.37 % | -0.26 % | -0.44 % | +0.02 % | +0.77 % | 102 |
| 4 | BA | +0.05 % | +0.59 % | +0.07 % | +0.15 % | -0.01 % | +0.16 % | 102 |
| 5 | AB | +0.06 % | +0.71 % | -0.05 % | +1.32 % | +0.02 % | +0.55 % | 102 |
| 6 | BA | -0.01 % | +0.73 % | -0.08 % | -0.04 % | +0.10 % | +0.78 % | 102 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | -0.15 % | -0.5 to +0.5 % | within the expectation |
| decode@2000 | +0.88 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| prefill@2000:write | -0.04 % | -0.5 to +0.5 % | within the expectation |
| decode@2000:write | +0.31 % | -0.5 to +0.5 % | within the expectation |
| prefill@30000 | +0.03 % | -0.5 to +0.5 % | within the expectation |
| decode@30000 | +0.73 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 | 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 / 131811, 19885 | yes |
| 2000:write | 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 | 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 / 82992, 25567 | yes |
| 30000 | 71024, 10531 / 71024, 10531 / 71024, 10531 / 71024, 10531 / 71024, 10531 / 71024, 10531 | 71024, 10531 / 71024, 10531 / 71024, 10531 / 71024, 10531 / 71024, 10531 / 71024, 10531 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | f6573f396c15300f | f6573f396c15300f | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | cde3f1093dce48a1 | cde3f1093dce48a1 | yes | yes | yes | - |

Runtime: 11.2 min for 6 rounds (1.7 min per round); a fixed 3 rounds would take 5.1 min (-5.1 min saved).
