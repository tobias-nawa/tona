## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: 6 rounds as the earlier A/Bs: effects of 0.2 to 2 % against a spread of about 1 % (decode tokens per prompt as in the targets, default 200)
- Duration: 25-75 min (2-6 rounds of 12.4 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-cache-hit.md):
  - decode-code: -0.5 to +0.5 %
  - decode-text: -0.5 to +0.5 %
  - decode@8000: -0.5 to +0.5 %
  - prefill@8000: +0 to +1.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1061-g4ad914d608-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-1061-g4ad914d608-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_METAL_KEEPALIVE_MS=500

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_METAL_KEEPALIVE_MS': '500'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 166.1 / 167.6 / 168.1 / 164.7 / 171.3 / 169.9 | 172.4 / 170.2 / 169.8 / 170.6 / 172.3 / 172.9 | 167.8 (164.7-171.3) | 171.4 (169.8-172.9) | +2.05 % [+0.50, +3.60] | **B faster** after round 2 |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.0 / 25.0 / 25.1 / 25.4 / 25.3 / 25.0 | 25.1 / 25.0 / 24.9 / 25.0 / 25.1 / 25.0 | 25.1 (25.0-25.4) | 25.0 (24.9-25.1) | -0.57 % [-1.47, +0.33] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 38.7 / 38.7 / 38.6 / 39.3 / 39.2 / 38.7 | 38.7 / 38.7 / 37.8 / 38.7 / 38.8 / 39.1 | 38.7 (38.6-39.3) | 38.7 (37.8-39.1) | -0.60 % [-1.96, +0.76] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.1 / 27.0 / 27.0 / 27.6 / 27.5 / 27.0 | 27.0 / 27.0 / 26.9 / 27.0 / 27.3 / 27.9 | 27.1 (27.0-27.6) | 27.0 (26.9-27.9) | -0.05 % [-2.12, +2.02] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +3.81 % | +0.29 % | +0.00 % | -0.37 % | 748 |
| 2 | BA | +1.54 % | -0.22 % | +0.00 % | +0.00 % | 747 |
| 3 | AB | +1.05 % | -0.55 % | -2.07 % | -0.37 % | 750 |
| 4 | BA | +3.57 % | -1.64 % | -1.53 % | -2.17 % | 747 |
| 5 | AB | +0.61 % | -1.05 % | -1.02 % | -0.73 % | 743 |
| 6 | BA | +1.72 % | -0.23 % | +1.03 % | +3.33 % | 742 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +2.05 % | +0 to +1.5 % | **OUTSIDE the expectation: finding** |
| decode@8000 | -0.57 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode-code | -0.60 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode-text | -0.05 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.7 | 22.7 | 17.5 | 90.9 | 0.7 | 1.3 | 5183.6 | 3.4 | 21398.2 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.5 | 37.8 | 8.4 | 93.7 | 0.2 | 0.2 | 3372.4 | 15.3 | 14621.6 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.4 | 28.2 | 8.2 | 94.2 | 0.3 | 0.2 | 3330.3 | 5.1 | 13919.0 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 6.4 | 14.6 | 7.9 | 94.6 | 0.2 | 0.3 | 2614.9 | 1.3 | 12331.1 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 6.5 | 18.4 | 9.7 | 95.0 | 0.3 | 0.3 | 2617.0 | 8.5 | 13781.7 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 6.4 | 14.5 | 7.6 | 94.9 | 0.3 | 0.3 | 2380.0 | 0.9 | 12420.7 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 6.4 | 16.0 | 7.8 | 94.7 | 0.3 | 0.3 | 2479.4 | 3.2 | 12771.9 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.3 | 14.2 | 8.5 | 94.4 | 0.3 | 0.3 | 2816.1 | 2.0 | 13404.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.3 | 15.2 | 7.8 | 94.3 | 0.3 | 0.3 | 2594.0 | 1.3 | 12909.4 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 5.3 | 15.5 | 7.8 | 94.6 | 0.3 | 0.3 | 2463.0 | 0.6 | 13059.3 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 5.3 | 14.4 | 7.6 | 94.3 | 0.3 | 0.3 | 2505.1 | 0.7 | 13100.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 6.5 | 16.7 | 7.7 | 94.5 | 0.3 | 0.3 | 2588.5 | 0.2 | 12968.1 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 7.4 | 16.0 | 7.6 | 93.4 | 0.6 | 0.3 | 2470.3 | 1.5 | 13193.9 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 7.0 | 16.3 | 7.8 | 94.1 | 0.5 | 0.3 | 2454.4 | 5.0 | 12917.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 6.4 | 17.7 | 8.0 | 94.8 | 0.3 | 0.3 | 2610.2 | 1.5 | 13458.5 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 6.0 | 29.9 | 9.3 | 93.3 | 1.1 | 0.3 | 2847.5 | 4.1 | 15025.2 | 7 | 0.33 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 6.1 | 52.8 | 8.8 | 93.6 | 0.3 | 0.3 | 3263.1 | 1.0 | 15863.1 | 6 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 18 | A | 6.5 | 72.6 | 9.8 | 92.5 | 0.2 | 0.4 | 4822.5 | 320.4 | 18417.0 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 19 | A | 5.7 | 45.0 | 8.0 | 94.5 | 0.3 | 0.2 | 2375.1 | 8.3 | 12887.8 | 8 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 20 | B | 6.8 | 45.6 | 8.2 | 95.0 | 0.2 | 0.3 | 2560.8 | 19.5 | 12631.2 | 8 | 0.37 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 21 | B | 7.0 | 19.2 | 8.2 | 94.9 | 0.2 | 0.3 | 2281.6 | 5.8 | 13169.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 22 | B | 6.7 | 40.3 | 9.4 | 94.7 | 0.2 | 0.3 | 2514.6 | 29.7 | 12525.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 23 | B | 6.7 | 53.0 | 46.1 | 95.1 | 0.2 | 0.2 | 2918.8 | 170.1 | 15828.1 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 24 | A | 5.0 | 15.3 | 19.4 | 95.2 | 0.2 | 0.3 | 2344.5 | 10.9 | 12364.6 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 25 | A | 5.2 | 16.3 | 7.7 | 94.6 | 0.3 | 0.3 | 2307.4 | 2.4 | 12645.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 77.8 min for 6 rounds (12.4 min per round); a fixed 3 rounds would take 37.3 min (-37.3 min saved).
