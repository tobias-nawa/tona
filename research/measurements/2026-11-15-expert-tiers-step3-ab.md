## Plan (written before the run)

- Scope: workload short,mid,set (2000:400 2000:write:400 30000:200 at --ctx 140000; bench-set at --ctx 32768); A `tiers` against B `tiers`; rounds 3 to 3, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: effect is a factor (replay: 4.2 to 1.7-1.9 SSD reads per token); 3 alternating rounds of short, mid (30K prefill) and set give the spread (decode tokens per prompt as in the targets, default 200)
- Duration: 14-14 min (3-3 rounds of 4.8 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: research/measurements/2026-11-15-expert-tiers.md step 1 replay and step 3 plan):
  - decode: +15 to +40 %
  - prefill: +0 to +100 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-818-ge3737b347f-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DECODE_DIAG=1, TONA_EXPERT_PROFILE=/var/tmp/claude/148/wt/work/profile/experts-coder.bin, TONA_MEGA_INFO=1, TONA_RAM_GB=26, TONA_VRAM_LOG=1
  - B: build commit v0.0.2-818-ge3737b347f-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DECODE_DIAG=1, TONA_EXPERT_PROFILE=/var/tmp/claude/148/wt/work/profile/experts-coder.bin, TONA_MEGA_INFO=1, TONA_RAM_GB=26, TONA_TIERS=1, TONA_VRAM_LOG=1

## Result

A: `tiers` env {'TONA_RAM_GB': '26', 'TONA_DECODE_DIAG': '1', 'TONA_VRAM_LOG': '1', 'TONA_MEGA_INFO': '1'}; B: `tiers` env {'TONA_RAM_GB': '26', 'TONA_DECODE_DIAG': '1', 'TONA_VRAM_LOG': '1', 'TONA_MEGA_INFO': '1', 'TONA_TIERS': '1'}. Alpha 0.05, margin 0.5 %, rounds 3 of at most 3, stop: 3 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 750.3 / 442.9 / 726.1 | 721.6 / 1148.8 / 1134.5 | 726.1 (442.9-750.3) | 1134.5 (721.6-1148.8) | +70.59 % [-27.53, +168.72] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 44.6 / 43.7 / 43.7 | 71.8 / 75.0 / 74.1 | 43.7 (43.7-44.6) | 74.1 (71.8-75.0) | +67.49 % [+60.78, +74.20] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 774.4 / 718.9 / 738.7 | 1357.3 / 1413.1 / 1389.3 | 738.7 (718.9-774.4) | 1389.3 (1357.3-1413.1) | +86.63 % [+73.89, +99.38] | **B faster** after round 1 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 58.7 / 57.5 / 56.9 | 112.2 / 116.9 / 116.8 | 57.5 (56.9-58.7) | 116.8 (112.2-116.9) | +99.90 % [+90.80, +109.00] | open (undecided at the last round) |
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 713.8 / 685.0 / 689.7 | 1255.3 / 1244.4 / 1262.5 | 689.7 (685.0-713.8) | 1255.3 (1244.4-1262.5) | +80.19 % [+77.36, +83.02] | **B faster** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 42.3 / 42.1 / 42.6 | 68.6 / 69.2 / 69.2 | 42.3 (42.1-42.6) | 69.2 (68.6-69.2) | +63.05 % [+62.08, +64.02] | **B faster** after round 3 |
| decode-code: Decode tok/s (bench-set code) | 62.0 / 61.5 / 62.0 | 91.0 / 93.8 / 92.5 | 62.0 (61.5-62.0) | 92.5 (91.0-93.8) | +49.50 % [+47.07, +51.92] | **B faster** after round 3 |
| decode-text: Decode tok/s (bench-set text) | 35.6 / 35.3 / 35.7 | 66.0 / 67.0 / 66.5 | 35.6 (35.3-35.7) | 66.5 (66.0-67.0) | +87.16 % [+85.17, +89.14] | **B faster** after round 3 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | prefill@30000 B vs A | decode@30000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AB | -3.82 % (off plan) | +61.09 % (off plan) | +75.27 % | +91.18 % (off plan) | +75.86 % | +62.23 % (off plan) | +46.77 % | +85.39 % | 285 |
| 2 | BA | +159.36 % (outlier) | +71.70 % (outlier) | +96.57 % | +103.36 % (outlier) | +81.68 % | +64.39 % (outlier) | +52.52 % | +89.80 % | 299 |
| 3 | AB | +56.24 % | +69.67 % (outlier) | +88.06 % | +105.17 % (outlier) | +83.04 % | +62.53 % | +49.19 % | +86.27 % | 285 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +70.59 % | +0 to +100 % | within the expectation |
| decode@2000 | +67.49 % | +15 to +40 % | **OUTSIDE the expectation: finding** |
| prefill@2000:write | +86.63 % | +0 to +100 % | within the expectation |
| decode@2000:write | +99.90 % | +15 to +40 % | **OUTSIDE the expectation: finding** |
| prefill@30000 | +80.19 % | +0 to +100 % | within the expectation |
| decode@30000 | +63.05 % | +15 to +40 % | **OUTSIDE the expectation: finding** |
| decode-code | +49.50 % | +15 to +40 % | **OUTSIDE the expectation: finding** |
| decode-text | +87.16 % | +15 to +40 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 83417, 69432 / 83417, 69432 / 83417, 69432 | 130198, 22651 / 130198, 22651 / 130198, 22651 | NO |
| 2000:write | 53918, 56359 / 53918, 56359 / 53918, 56359 | 84165, 26112 / 84165, 26112 / 84165, 26112 | NO |
| 30000 | 44325, 34044 / 44325, 34044 / 44325, 34044 | 67993, 10376 / 67993, 10376 / 67993, 10376 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |
| 30000 | 0a2eac2f1502846f | 0a2eac2f1502846f | yes | yes | yes | - |

Runtime: 17.3 min for 3 rounds (4.8 min per round); a fixed 3 rounds would take 14.5 min (+0.0 min saved).
