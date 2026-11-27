## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: bench-set plus the 8000-token prompt as in the baseline A/B of the same pair, so the two files compare (decode tokens per prompt as in the targets, default 200)
- Duration: 32-95 min (2-6 rounds of 15.8 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-coder-arena.md (single diagnosis runs)):
  - decode-code: -60 to -40 %
  - decode-text: -60 to -30 %
  - decode@8000: -60 to -30 %
  - prefill@8000: -20 to +5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1152-g6cb5c55e72-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-1152-g6cb5c55e72-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/toolchain/profiles/coder-iq1_m.bin

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_EXPERT_PROFILE': '<repo>/toolchain/profiles/coder-iq1_m.bin'}. Alpha 0.05, margin 0.5 %, rounds 1 of at most 6, stop: interrupted (partial result). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 172.9 | 144.9 | 172.9 (172.9-172.9) | 144.9 (144.9-144.9) | - | open (one round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 24.9 | 18.4 | 24.9 (24.9-24.9) | 18.4 (18.4-18.4) | - | open (one round) |
| decode-code: Decode tok/s (bench-set code) | 38.7 | 23.6 | 38.7 (38.7-38.7) | 23.6 (23.6-23.6) | - | open (one round) |
| decode-text: Decode tok/s (bench-set text) | 27.0 | 14.9 | 27.0 (27.0-27.0) | 14.9 (14.9-14.9) | - | open (one round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -16.16 % | -26.41 % | -39.02 % | -44.81 % | 946 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 | 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 7.0 | 21.6 | 14.6 | 88.6 | 0.7 | 1.5 | 24003.8 | 23.7 | 16550.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.5 | 14.9 | 7.8 | 94.8 | 0.2 | 0.3 | 2518.5 | 4.6 | 8515.6 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.3 | 16.2 | 7.4 | 95.0 | 0.2 | 0.4 | 2434.0 | 3.7 | 7712.0 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 73.1 | 14.8 | 6.9 | 74.0 | 0.2 | 0.2 | 7713.0 | 3.9 | 16819.6 | 14 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 4.0 |
| 5 | B | 71.4 | 13.6 | 6.7 | 74.4 | 0.2 | 0.3 | 129.3 | 7.2 | 11.9 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 6 | B | 71.8 | 13.7 | 6.8 | 74.4 | 0.2 | 0.2 | 205.6 | 19.2 | 49.2 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 7 | B | 72.0 | 14.1 | 6.6 | 74.2 | 0.2 | 0.2 | 120.1 | 7.1 | 14.0 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 8 | A | 6.5 | 52.7 | 8.3 | 91.5 | 0.2 | 0.3 | 21894.7 | 23.2 | 17792.3 | 8 | 0.94 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 9 | A | 11.5 | 45.5 | 14.1 | 86.0 | 1.0 | 0.7 | 2989.3 | 15.7 | 13223.0 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | c317a70c633e2052 | yes | yes | NO | - |

The texts differ (normal mode: sampling of the drafter and rounding may differ; no verdict).

Runtime: 33.1 min for 1 rounds (15.8 min per round); a fixed 3 rounds would take 47.3 min (+31.5 min saved).
