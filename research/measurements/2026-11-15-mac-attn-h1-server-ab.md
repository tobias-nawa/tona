## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: 2 to 6 rounds: an expected prefill gain of about 1 % against a spread of about 1 % (decode tokens per prompt as in the targets, default 200)
- Duration: 25-75 min (2-6 rounds of 12.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-attn-h1.md):
  - decode-code: -0.5 to +0.5 %
  - decode-text: -0.5 to +0.5 %
  - decode@8000: -0.5 to +0.5 %
  - prefill@8000: +0.5 to +2 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1100-gca29b36af2; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-1100-gca29b36af2; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_METAL_ATTN_H1_BELOW=1000000

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_METAL_ATTN_H1_BELOW': '1000000'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 162.0 / 161.6 / 163.7 / 160.4 / 173.5 / 169.8 | 166.0 / 169.3 / 163.9 / 166.5 / 171.3 / 172.2 | 162.8 (160.4-173.5) | 167.9 (163.9-172.2) | +1.89 % [-0.66, +4.44] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.2 / 25.2 / 24.9 / 25.4 / 25.5 / 25.1 | 25.0 / 25.1 / 25.0 / 25.0 / 25.5 / 25.2 | 25.2 (24.9-25.5) | 25.1 (25.0-25.5) | -0.27 % [-1.32, +0.77] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 38.9 / 38.9 / 38.2 / 39.4 / 39.2 / 38.6 | 38.8 / 38.7 / 38.8 / 38.8 / 38.8 / 38.7 | 38.9 (38.2-39.4) | 38.8 (38.7-38.8) | -0.25 % [-1.54, +1.04] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.2 / 27.2 / 27.2 / 27.6 / 27.6 / 27.0 | 27.1 / 27.1 / 27.3 / 27.2 / 27.1 / 27.0 | 27.2 (27.0-27.6) | 27.1 (27.0-27.3) | -0.60 % [-1.66, +0.45] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +2.51 % | -0.97 % | -0.26 % | -0.37 % | 751 |
| 2 | BA | +4.77 % | -0.37 % | -0.51 % | -0.37 % | 749 |
| 3 | AB | +0.12 % | +0.43 % | +1.57 % | +0.37 % | 754 |
| 4 | BA | +3.81 % | -1.52 % | -1.52 % | -1.45 % | 749 |
| 5 | AB | -1.28 % | +0.19 % | -1.02 % | -1.81 % | 738 |
| 6 | BA | +1.39 % | +0.60 % | +0.26 % | +0.00 % | 745 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +1.89 % | +0.5 to +2 % | within the expectation |
| decode@8000 | -0.27 % | -0.5 to +0.5 % | within the expectation |
| decode-code | -0.25 % | -0.5 to +0.5 % | within the expectation |
| decode-text | -0.60 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 7.2 | 34.0 | 11.3 | 92.9 | 0.2 | 0.2 | 5625.1 | 6.7 | 23819.8 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 6.7 | 40.6 | 13.6 | 93.7 | 0.2 | 0.3 | 4305.6 | 2.5 | 18569.4 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 6.9 | 76.6 | 10.2 | 93.5 | 0.3 | 0.3 | 4909.1 | 11.2 | 20126.7 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 6.9 | 47.9 | 9.4 | 93.8 | 0.3 | 0.2 | 3960.1 | 18.2 | 16897.3 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 6.5 | 40.5 | 9.2 | 93.9 | 0.2 | 0.2 | 3823.4 | 5.7 | 14669.2 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 6.3 | 40.0 | 8.5 | 94.3 | 0.2 | 0.3 | 3221.6 | 4.0 | 10833.1 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 6.3 | 24.1 | 9.1 | 94.3 | 0.2 | 0.4 | 3653.8 | 0.6 | 12434.4 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 6.6 | 29.9 | 8.8 | 93.9 | 0.2 | 0.3 | 3917.5 | 8.6 | 15875.7 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 6.9 | 81.5 | 10.4 | 93.4 | 0.2 | 0.2 | 5327.0 | 5.1 | 21279.8 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 7.6 | 59.1 | 13.5 | 92.1 | 0.6 | 0.5 | 5998.5 | 2.3 | 21334.2 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 7.9 | 64.8 | 13.6 | 91.6 | 0.8 | 0.8 | 5431.4 | 3.7 | 21477.3 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 7.4 | 40.1 | 14.6 | 92.6 | 0.6 | 0.5 | 4867.2 | 3.8 | 17102.7 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 6.8 | 52.1 | 11.9 | 92.7 | 0.4 | 1.0 | 5212.2 | 5.8 | 18417.4 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 7.2 | 55.2 | 11.5 | 93.2 | 0.5 | 0.4 | 4475.8 | 3.7 | 16777.6 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 6.6 | 42.2 | 9.6 | 93.9 | 0.2 | 0.2 | 4479.4 | 0.8 | 17717.0 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 7.2 | 77.9 | 10.4 | 93.1 | 0.2 | 0.3 | 5717.4 | 3.6 | 21756.0 | 5 | 0.86 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 7.6 | 109.5 | 10.1 | 92.7 | 0.3 | 0.2 | 7445.5 | 50.3 | 26576.3 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 18 | A | 6.8 | 47.6 | 8.2 | 94.9 | 0.2 | 0.3 | 2443.2 | 51.4 | 12561.0 | 10 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 19 | A | 7.1 | 63.3 | 19.2 | 94.6 | 0.2 | 0.3 | 2833.1 | 400.8 | 17109.6 | 8 | 0.61 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 20 | B | 7.4 | 25.6 | 8.1 | 94.1 | 0.2 | 0.3 | 2669.1 | 15.4 | 14403.4 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 21 | B | 6.4 | 22.1 | 7.7 | 94.7 | 0.2 | 0.2 | 2594.9 | 18.4 | 8532.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 22 | B | 6.1 | 25.5 | 7.7 | 94.8 | 0.2 | 0.3 | 2760.0 | 11.5 | 8557.0 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 23 | B | 6.1 | 16.8 | 9.0 | 94.7 | 0.2 | 0.4 | 2662.0 | 14.8 | 9255.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 24 | A | 6.0 | 16.4 | 7.6 | 94.6 | 0.2 | 0.4 | 2563.5 | 11.8 | 9127.3 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 25 | A | 6.4 | 17.4 | 7.6 | 94.5 | 0.3 | 0.3 | 2455.9 | 6.0 | 13581.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 77.9 min for 6 rounds (12.5 min per round); a fixed 3 rounds would take 37.4 min (-37.4 min saved).
