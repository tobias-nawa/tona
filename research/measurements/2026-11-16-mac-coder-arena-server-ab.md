## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: bench-set plus the 8000-token prompt as in the baseline A/B of the same pair, so the two files compare (decode tokens per prompt as in the targets, default 200)
- Duration: 33-99 min (2-6 rounds of 16.4 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-coder-arena.md (single diagnosis runs); round 1 of 2026-11-15-mac-coder-arena-server-ab.md within):
  - decode-code: -60 to -40 %
  - decode-text: -60 to -30 %
  - decode@8000: -60 to -30 %
  - prefill@8000: -20 to +5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1156-g58513e336f-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-1156-g58513e336f-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/toolchain/profiles/coder-iq1_m.bin

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_EXPERT_PROFILE': '<repo>/toolchain/profiles/coder-iq1_m.bin'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 6, stop: all metrics decided after round 2. All rounds are listed; 1 discarded, below.

Discarded (repeated) rounds:

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -16.49 % | -41.39 % | -49.11 % | -54.71 % | 1625 |

- AB: a system backup ran

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 170.5 / 170.3 | 144.0 / 142.7 | 170.4 (170.3-170.5) | 143.4 (142.7-144.0) | -15.88 % [-17.00, -14.76] | **B slower** after round 2 |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.1 / 25.0 | 17.6 / 16.4 | 25.0 (25.0-25.1) | 17.0 (16.4-17.6) | -32.09 % [-35.31, -28.87] | **B slower** after round 2 |
| decode-code: Decode tok/s (bench-set code) | 38.7 / 38.7 | 22.9 / 21.2 | 38.7 (38.7-38.7) | 22.0 (21.2-22.9) | -43.02 % [-46.38, -39.67] | **B slower** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 27.0 / 27.0 | 14.2 / 13.8 | 27.0 (27.0-27.0) | 14.0 (13.8-14.2) | -48.15 % [-49.64, -46.66] | **B slower** after round 2 |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -15.57 % | -29.99 % | -40.83 % | -47.41 % | 973 |
| 2 | BA | -16.19 % | -34.19 % | -45.22 % | -48.89 % | 997 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -15.88 % | -20 to +5 % | within the expectation |
| decode@8000 | -32.09 % | -60 to -30 % | within the expectation |
| decode-code | -43.02 % | -60 to -40 % | within the expectation |
| decode-text | -48.15 % | -60 to -30 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.4 | 27.0 | 9.1 | 91.4 | 0.3 | 0.4 | 14883.4 | 14.1 | 18403.1 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 6.3 | 41.7 | 58.3 | 93.5 | 0.4 | 0.2 | 3319.9 | 3.9 | 24017.3 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.9 | 50.0 | 8.8 | 94.1 | 0.2 | 0.3 | 3259.1 | 4.6 | 11225.8 | 7 | 0.87 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 72.5 | 16.7 | 6.8 | 68.0 | 0.2 | 0.2 | 7940.2 | 1.5 | 1773.7 | 23 | 0.0 | 0.0 | - | 0.0 | 2 | 540.7 | 4.0 |
| 5 | B | 72.4 | 17.5 | 7.7 | 71.0 | 0.2 | 0.2 | 301.1 | 13.2 | 4439.3 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 6 | A | 6.1 | 16.4 | 8.2 | 90.9 | 0.2 | 0.3 | 22747.8 | 0.5 | 18162.1 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 7 | A | 5.5 | 15.7 | 7.5 | 94.5 | 0.2 | 0.4 | 2641.4 | 1.1 | 9481.0 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 8 | B | 72.8 | 13.5 | 7.7 | 73.0 | 0.2 | 0.2 | 8103.2 | 0.4 | 15911.3 | 14 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 4.0 |
| 9 | B | 71.9 | 15.1 | 6.8 | 73.8 | 0.2 | 0.2 | 239.8 | 1.6 | 263.5 | 15 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 10 | B | 71.9 | 14.2 | 7.0 | 73.2 | 0.2 | 0.2 | 93.9 | 1.0 | 128.0 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 11 | B | 71.7 | 13.4 | 6.7 | 72.4 | 0.2 | 0.2 | 132.7 | 0.8 | 0.6 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 12 | A | 6.1 | 17.1 | 7.8 | 91.9 | 0.2 | 0.3 | 21499.6 | 22.9 | 16468.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 13 | A | 5.7 | 15.2 | 7.7 | 94.6 | 0.2 | 0.3 | 2660.4 | 0.3 | 10628.3 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | c317a70c633e2052 | yes | yes | NO | - |

The texts differ (normal mode: sampling of the drafter and rounding may differ; no verdict).

Runtime: 63.1 min for 2 rounds (16.4 min per round); a fixed 3 rounds would take 49.3 min (+16.4 min saved).
