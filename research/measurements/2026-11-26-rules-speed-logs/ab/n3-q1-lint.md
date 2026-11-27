## Plan (written before the run)

- Scope: workload short,mid,long (2000:400 2000:write:400 30000:200 130000:200 at --ctx 140000); A `n3-main` against B `n3-pr`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: short: decode loop at 2K; mid and long: prefill and decode at 30K and 130K where attention dominates; the three workloads of the generator, scheduler and prompt cache paths the diff touches (decode tokens per prompt as in the targets, default 200)
- Duration: 7-20 min (2-6 rounds of 3.3 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: bench/noise.toml A/A spreads (decode 0.23 to 0.42 %, prefill 0.02 to 0.09 %); the diff is host lint, no device code):
  - decode: -0.5 to +0.5 %
  - prefill: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit d43a0b3fa7; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
  - B: build commit f50cd70c7a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin

## Result

A: `n3-main` env -; B: `n3-pr` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1386.7 / 1378.8 / 1438.9 / 1403.4 / 1243.8 / 1427.2 | 1304.1 / 1361.1 / 1438.7 / 1308.4 / 1434.3 / 1430.8 | 1395.1 (1243.8-1438.9) | 1395.9 (1304.1-1438.7) | +0.26 % [-8.52, +9.03] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 72.0 / 76.0 / 75.8 / 75.9 / 74.9 / 75.6 | 75.6 / 76.0 / 75.5 / 76.0 / 76.2 / 76.2 | 75.7 (72.0-76.0) | 76.0 (75.5-76.2) | +1.20 % [-1.06, +3.47] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1651.6 / 1683.6 / 1734.0 / 1681.5 / 1667.8 / 1740.5 | 1737.3 / 1670.2 / 1752.9 / 1661.3 / 1710.4 / 1723.9 | 1682.6 (1651.6-1740.5) | 1717.1 (1661.3-1752.9) | +0.98 % [-1.81, +3.77] | open (undecided at the last round) |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 128.4 / 124.8 / 129.1 / 128.1 / 128.8 / 128.9 | 128.8 / 129.0 / 128.4 / 128.5 / 128.4 / 128.9 | 128.6 (124.8-129.1) | 128.6 (128.4-129.0) | +0.52 % [-1.18, +2.22] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1959.8 / 1977.9 / 1974.5 / 1980.4 / 1974.3 / 1986.3 | 1954.7 / 1976.9 / 1974.5 / 1963.1 / 1967.4 / 1990.6 | 1976.2 (1959.8-1986.3) | 1971.0 (1954.7-1990.6) | -0.22 % [-0.52, +0.08] | open (undecided at the last round) |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 69.5 / 69.2 / 69.0 / 69.5 / 69.5 / 69.5 | 69.5 / 69.3 / 68.9 / 69.2 / 69.3 / 68.8 | 69.5 (69.0-69.5) | 69.3 (68.8-69.5) | -0.25 % [-0.70, +0.20] | **equal** after round 3 |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1887.0 / 1885.8 / 1887.7 / 1888.2 / 1886.3 / 1889.3 | 1885.8 / 1882.5 / 1887.9 / 1881.9 / 1883.9 / 1889.0 | 1887.4 (1885.8-1889.3) | 1884.9 (1881.9-1889.0) | -0.12 % [-0.22, -0.02] | **B slower** after round 1 (real, but inside the margin) |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 67.1 / 67.0 / 66.9 / 67.1 / 67.0 / 66.1 | 67.0 / 67.0 / 66.8 / 66.9 / 66.6 / 66.9 | 67.0 (66.1-67.1) | 66.9 (66.6-67.0) | +0.02 % [-0.64, +0.68] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AB | -5.96 % (off plan) | +5.12 % (off plan) | +5.19 % (off plan) | +0.29 % | -0.26 % | -0.01 % | -0.06 % | -0.02 % | 199 |
| 2 | BA | -1.29 % (outlier) | +0.02 % | -0.80 % (outlier) | +3.39 % (outlier) | -0.05 % | +0.24 % | -0.17 % | -0.03 % | 199 |
| 3 | AB | -0.01 % | -0.45 % | +1.09 % (outlier) | -0.54 % | -0.00 % | -0.14 % | +0.01 % | -0.18 % | 196 |
| 4 | BA | -6.77 % (outlier) | +0.21 % | -1.20 % (outlier) | +0.25 % | -0.87 % (outlier) | -0.33 % | -0.33 % | -0.29 % | 203 |
| 5 | AB | +15.32 % (outlier) | +1.63 % (outlier) | +2.56 % (outlier) | -0.28 % | -0.35 % | -0.27 % | -0.13 % | -0.56 % | 197 |
| 6 | BA | +0.25 % | +0.70 % | -0.96 % (outlier) | +0.01 % | +0.22 % | -1.01 % | -0.02 % | +1.19 % | 196 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.26 % | -0.5 to +0.5 % | within the expectation |
| decode@2000 | +1.20 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| prefill@2000:write | +0.98 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode@2000:write | +0.52 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| prefill@30000 | -0.22 % | -0.5 to +0.5 % | within the expectation |
| decode@30000 | -0.25 % | -0.5 to +0.5 % | within the expectation |
| prefill@130000 | -0.12 % | -0.5 to +0.5 % | within the expectation |
| decode@130000 | +0.02 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 128937, 23912 / 128937, 23912 / 128937, 23912 / 128937, 23912 / 128937, 23912 / 128937, 23912 | 128937, 23912 / 128937, 23912 / 128937, 23912 / 128937, 23912 / 128937, 23912 / 128937, 23912 | yes |
| 2000:write | 82220, 28057 / 82220, 28057 / 82220, 28057 / 82220, 28057 / 82220, 28057 / 82220, 28057 | 82220, 28057 / 82220, 28057 / 82220, 28057 / 82220, 28057 / 82220, 28057 / 82220, 28057 | yes |
| 30000 | 67822, 10547 / 67822, 10547 / 67822, 10547 / 67822, 10547 / 67822, 10547 / 67822, 10547 | 67822, 10547 / 67822, 10547 / 67822, 10547 / 67822, 10547 / 67822, 10547 / 67822, 10547 | yes |
| 130000 | 73061, 9712 / 73061, 9712 / 73061, 9712 / 73061, 9712 / 73061, 9712 / 73061, 9712 | 73061, 9712 / 73061, 9712 / 73061, 9712 / 73061, 9712 / 73061, 9712 / 73061, 9712 | yes |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000 | 3.56 (3.96) / 3.56 (3.96) / 3.56 (3.96) / 3.56 (3.96) / 3.56 (3.96) / 3.56 (3.96) | 3.56 (3.96) / 3.56 (3.96) / 3.56 (3.96) / 3.56 (3.96) / 3.56 (3.96) / 3.56 (3.96) |
| 2000:write | 7.82 (1.24) / 7.82 (1.24) / 7.82 (1.24) / 7.82 (1.24) / 7.82 (1.24) / 7.82 (1.24) | 7.82 (1.24) / 7.82 (1.24) / 7.82 (1.24) / 7.82 (1.24) / 7.82 (1.24) / 7.82 (1.24) |
| 30000 | 3.12 (3.64) / 3.12 (3.64) / 3.12 (3.64) / 3.12 (3.64) / 3.12 (3.64) / 3.12 (3.64) | 3.12 (3.64) / 3.12 (3.64) / 3.12 (3.64) / 3.12 (3.64) / 3.12 (3.64) / 3.12 (3.64) |
| 130000 | 3.03 (3.59) / 3.03 (3.59) / 3.03 (3.59) / 3.03 (3.59) / 3.03 (3.59) / 3.03 (3.59) | 3.03 (3.59) / 3.03 (3.59) / 3.03 (3.59) / 3.03 (3.59) / 3.03 (3.59) / 3.03 (3.59) |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 157.6 | 25.0 | 11.5 | 0.0 | 0.0 | 0.8 | 36175.6 | 0.4 | 12.5 | 35 | 0.0 | 0.0 | - | 0.0 | - | - | 14.2 |
| 2 | A | 160.9 | 20.9 | 10.9 | 0.0 | 0.0 | 0.7 | 32291.6 | 0.1 | 0.3 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.1 |
| 3 | B | 158.3 | 22.0 | 11.2 | 0.0 | 0.0 | 0.9 | 33512.3 | 0.1 | 0.0 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 14.1 |
| 4 | B | 159.3 | 23.8 | 11.4 | 0.0 | 0.0 | 0.8 | 34427.1 | 0.8 | 0.3 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 14.3 |
| 5 | A | 163.8 | 23.5 | 12.5 | 0.0 | 0.0 | 0.8 | 32413.7 | 0.1 | 0.8 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 6 | A | 162.5 | 20.8 | 11.6 | 0.0 | 0.0 | 1.0 | 33218.5 | 0.1 | 0.8 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 7 | B | 161.4 | 19.5 | 13.2 | 0.0 | 0.0 | 0.8 | 30842.2 | 0.1 | 0.0 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 8 | B | 157.2 | 26.4 | 12.0 | 0.0 | 0.0 | 0.7 | 34841.6 | 0.1 | 0.3 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 16.1 |
| 9 | A | 159.9 | 22.8 | 12.8 | 0.0 | 0.0 | 0.9 | 32968.2 | 0.5 | 0.3 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 14.1 |
| 10 | A | 162.3 | 21.8 | 12.5 | 0.0 | 0.0 | 1.0 | 34883.5 | 0.3 | 0.0 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 11 | B | 162.9 | 22.6 | 14.4 | 0.0 | 0.0 | 0.9 | 33316.8 | 0.4 | 0.4 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.0 |
| 12 | B | 161.1 | 20.0 | 14.0 | 0.0 | 0.0 | 0.8 | 31049.4 | 0.1 | 0.4 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 12.1 |
| 13 | A | 161.0 | 19.6 | 12.7 | 0.0 | 0.0 | 0.9 | 31655.6 | 0.1 | 0.0 | 36 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 12.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | 0a2eac2f1502846f | 0a2eac2f1502846f | yes | yes | yes | - |
| 130000 | 7e79f79e83fade65 | 7e79f79e83fade65 | yes | yes | yes | - |

Runtime: 21.5 min for 6 rounds (3.3 min per round); a fixed 3 rounds would take 9.9 min (-9.9 min saved).
