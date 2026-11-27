## Plan (written before the run)

- Scope: workload short,mid,set (2000:400 2000:write:400 30000:200 at --ctx 140000; bench-set at --ctx 32768); A `tiers` against B `tiers`; rounds 3 to 3, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: repeat of step 3 with the budget counting the pinned KV and sequence buffers filled later (floor 3.0 GB kept); the effect is large, 3 alternating rounds (decode tokens per prompt as in the targets, default 200)
- Duration: 15-15 min (3-3 rounds of 5.1 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: research/measurements/2026-11-15-expert-tiers-step3-ab.md: first run +49 to +100 percent decode with 0.6 to 2 GB of page cache left; the fix pins 73 percent instead of 84 percent):
  - decode: +15 to +60 %
  - prefill: +0 to +100 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-818-ge3737b347f-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DECODE_DIAG=1, TONA_EXPERT_PROFILE=/var/tmp/claude/148/wt/work/profile/experts-coder.bin, TONA_MEGA_INFO=1, TONA_RAM_GB=26, TONA_VRAM_LOG=1
  - B: build commit v0.0.2-818-ge3737b347f-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DECODE_DIAG=1, TONA_EXPERT_PROFILE=/var/tmp/claude/148/wt/work/profile/experts-coder.bin, TONA_MEGA_INFO=1, TONA_RAM_GB=26, TONA_TIERS=1, TONA_VRAM_LOG=1

## Result

A: `tiers` env {'TONA_RAM_GB': '26', 'TONA_DECODE_DIAG': '1', 'TONA_VRAM_LOG': '1', 'TONA_MEGA_INFO': '1'}; B: `tiers` env {'TONA_RAM_GB': '26', 'TONA_DECODE_DIAG': '1', 'TONA_VRAM_LOG': '1', 'TONA_MEGA_INFO': '1', 'TONA_TIERS': '1'}. Alpha 0.05, margin 0.5 %, rounds 3 of at most 3, stop: 3 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 754.8 / 555.6 / 710.3 | 756.1 / 535.5 / 708.5 | 710.3 (555.6-754.8) | 708.5 (535.5-756.1) | -1.24 % [-3.70, +1.23] | **equal** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 44.7 / 44.2 / 44.5 | 44.8 / 43.7 / 44.1 | 44.5 (44.2-44.7) | 44.1 (43.7-44.8) | -0.55 % [-1.50, +0.40] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 774.9 / 707.1 / 732.0 | 780.0 / 710.6 / 726.8 | 732.0 (707.1-774.9) | 726.8 (710.6-780.0) | +0.15 % [-0.77, +1.07] | open (undecided at the last round) |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 58.9 / 58.3 / 58.4 | 58.7 / 58.1 / 58.1 | 58.4 (58.3-58.9) | 58.1 (58.1-58.7) | -0.40 % [-1.42, +0.62] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 721.2 / 676.4 / 680.9 | 723.0 / 691.7 / 678.2 | 680.9 (676.4-721.2) | 691.7 (678.2-723.0) | +0.71 % [-0.33, +1.74] | open (undecided at the last round) |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 42.7 / 42.3 / 42.7 | 42.5 / 42.3 / 42.7 | 42.7 (42.3-42.7) | 42.5 (42.3-42.7) | -0.15 % [-0.64, +0.34] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 61.1 / 62.6 / 62.5 | 88.5 / 84.3 / 85.5 | 62.5 (61.1-62.6) | 85.5 (84.3-88.5) | +38.77 % [+34.35, +43.19] | **B faster** after round 3 |
| decode-text: Decode tok/s (bench-set text) | 35.5 / 36.0 / 35.8 | 64.7 / 64.7 / 65.0 | 35.8 (35.5-36.0) | 64.7 (64.7-65.0) | +81.18 % [+79.97, +82.39] | **B faster** after round 3 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AB | +0.17 % | +0.29 % (off plan) | +0.65 % | -0.39 % (off plan) | +0.24 % | -0.49 % (off plan) | +44.84 % | +82.25 % | 301 |
| 2 | BA | -3.61 % (outlier) | -1.06 % (outlier) | +0.49 % | -0.38 % | +2.27 % | +0.03 % | +34.66 % | +79.72 % | 313 |
| 3 | AB | -0.26 % (outlier) | -0.87 % | -0.70 % (outlier) | -0.44 % | -0.39 % (outlier) | +0.00 % | +36.80 % | +81.56 % | 309 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | -1.24 % | +0 to +100 % | **OUTSIDE the expectation: finding** |
| decode@2000 | -0.55 % | +15 to +60 % | **OUTSIDE the expectation: finding** |
| prefill@2000:write | +0.15 % | +0 to +100 % | within the expectation |
| decode@2000:write | -0.40 % | +15 to +60 % | **OUTSIDE the expectation: finding** |
| prefill@30000 | +0.71 % | +0 to +100 % | within the expectation |
| decode@30000 | -0.15 % | +15 to +60 % | **OUTSIDE the expectation: finding** |
| decode-code | +38.77 % | +15 to +60 % | within the expectation |
| decode-text | +81.18 % | +15 to +60 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 83417, 69432 / 83417, 69432 / 83417, 69432 | 83417, 69432 / 83417, 69432 / 83417, 69432 | yes |
| 2000:write | 53918, 56359 / 53918, 56359 / 53918, 56359 | 53918, 56359 / 53918, 56359 / 53918, 56359 | yes |
| 30000 | 44325, 34044 / 44325, 34044 / 44325, 34044 | 44325, 34044 / 44325, 34044 / 44325, 34044 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | 0a2eac2f1502846f | 0a2eac2f1502846f | yes | yes | yes | - |

Runtime: 18.3 min for 3 rounds (5.1 min per round); a fixed 3 rounds would take 15.4 min (+0.0 min saved).
