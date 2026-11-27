## Plan (written before the run)

- Scope: workload short,mid (2000:400 2000:write:400 30000:200 at --ctx 140000); A `w85-base` against B `w85`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (decode) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: same workloads as the first run: short = decode at 2K, mid = prefill through k_mega<8> (decode tokens per prompt as in the targets, default 200)
- Duration: 3-10 min (2-6 rounds of 1.7 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: first run of this plan (2026-11-02-evict-deadline-ab-speed.md): +0.7..0.9 % decode; the final build differs by the tidy fixes (spills 111 -> 117 SGPR in k_mega<4>)):
  - decode: -0.5 to +0.5 %
  - prefill: -0.5 to +0.5 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-204-g98dc08946e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-213-gf97596c393; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `w85-base` env -; B: `w85` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1326.9 / 1323.8 / 1325.8 / 1321.5 / 1297.1 / 1289.6 | 1324.5 / 1324.5 / 1323.7 / 1328.0 / 1322.0 / 1287.6 | 1322.7 (1289.6-1326.9) | 1324.1 (1287.6-1328.0) | +0.33 % [-0.58, +1.24] | **equal** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 66.1 / 65.7 / 65.4 / 65.0 / 65.0 / 64.6 | 66.6 / 66.1 / 65.9 / 65.7 / 65.5 / 65.8 | 65.2 (64.6-66.1) | 65.9 (65.5-66.6) | +0.95 % [+0.36, +1.54] | **B faster** after round 2 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1541.0 / 1539.6 / 1543.3 / 1544.1 / 1538.9 / 1540.6 | 1541.6 / 1518.9 / 1543.6 / 1542.4 / 1537.8 / 1542.9 | 1540.8 (1538.9-1544.1) | 1542.0 (1518.9-1543.6) | -0.22 % [-0.86, +0.41] | open (undecided at the last round) |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 118.8 / 118.4 / 118.7 / 119.0 / 119.3 / 118.8 | 119.0 / 119.2 / 119.7 / 119.2 / 118.7 / 119.2 | 118.8 (118.4-119.3) | 119.2 (118.7-119.7) | +0.28 % [-0.49, +1.05] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1779.3 / 1776.8 / 1777.9 / 1778.7 / 1777.5 / 1779.4 | 1780.1 / 1760.4 / 1778.0 / 1779.2 / 1776.9 / 1776.0 | 1778.3 (1776.8-1779.4) | 1777.5 (1760.4-1780.1) | -0.18 % [-0.48, +0.13] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 70.1 / 69.3 / 69.2 / 68.9 / 69.1 / 69.9 | 70.5 / 70.7 / 69.6 / 69.6 / 69.9 / 70.1 | 69.3 (68.9-70.1) | 70.0 (69.6-70.7) | +0.93 % [+0.35, +1.50] | **B faster** after round 3 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | -0.18 % | +0.74 % (off plan) | +0.03 % | +0.19 % | +0.05 % | +0.47 % | 102 |
| 2 | BA | +0.05 % | +0.61 % | -1.34 % (outlier) | +0.60 % | -0.93 % (outlier) | +1.96 % (outlier) | 104 |
| 3 | AB | -0.16 % | +0.64 % | +0.02 % | +0.83 % | +0.01 % | +0.59 % | 104 |
| 4 | BA | +0.49 % | +1.09 % | -0.11 % | +0.13 % | +0.03 % | +1.06 % | 103 |
| 5 | AB | +1.92 % (outlier) | +0.70 % | -0.07 % | -0.45 % | -0.03 % | +1.19 % | 103 |
| 6 | BA | -0.15 % | +1.90 % (outlier) | +0.15 % | +0.40 % | -0.19 % | +0.29 % | 103 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.33 % | -0.5 to +0.5 % | within the expectation |
| decode@2000 | +0.95 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| prefill@2000:write | -0.22 % | -0.5 to +0.5 % | within the expectation |
| decode@2000:write | +0.28 % | -0.5 to +0.5 % | within the expectation |
| prefill@30000 | -0.18 % | -0.5 to +0.5 % | within the expectation |
| decode@30000 | +0.93 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

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

Runtime: 11.2 min for 6 rounds (1.7 min per round); a fixed 3 rounds would take 5.2 min (-5.2 min saved).
