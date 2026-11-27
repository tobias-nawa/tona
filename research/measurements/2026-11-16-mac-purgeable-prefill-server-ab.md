## Plan (written before the run)

- Scope: workload server (8000:16 at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (prefill@8000) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: prefill@8000 alone decides whether the purgeable arena's wakes cost prefill (cost A/B: -1.6 to -5.4 % in 3 of 4 calm rounds); 16 decode tokens only to end each request (decode tokens per prompt as in the targets, default 200)
- Duration: 7-21 min (2-6 rounds of 3.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/53-mac-purgeable-arena.md (v2: wakes 0.02 s per 10000 stops while requests stream; the cost A/B's prefill loss was v1's)):
  - prefill@8000: -1 to +1 %
  - decode@8000: -3 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1172-g3bcdae3336; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_CEILING=1, TONA_METAL_ARENA_GB=auto, TONA_METAL_ARENA_GOVERN=1
  - B: build commit stand-2026-10-25-1172-g3bcdae3336; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_GB=auto, TONA_METAL_ARENA_GOVERN=1, TONA_METAL_ARENA_PURGEABLE=1

## Result

A: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin', 'TONA_METAL_ARENA_GOVERN': '1', 'TONA_METAL_ARENA_CEILING': '1'}; B: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin', 'TONA_METAL_ARENA_GOVERN': '1', 'TONA_METAL_ARENA_PURGEABLE': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 6, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 168.4 / 169.6 | 165.6 / 163.9 | 169.0 (168.4-169.6) | 164.8 (163.9-165.6) | -2.50 % [-4.11, -0.89] | **B slower** after round 2 |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 11.9 / 12.8 | 12.0 / 11.7 | 12.4 (11.9-12.8) | 11.8 (11.7-12.0) | -4.01 % [-10.67, +2.64] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | -1.65 % | +0.51 % | 207 |
| 2 | BA | -3.35 % | -8.53 % | 207 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -2.50 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode@8000 | -4.01 % | -3 to +3 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 19.3 | 30.5 | 16.9 | 68.8 | 1.4 | 3.8 | 1798.7 | 11.5 | 20452.5 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | - | 4.0 |
| 2 | A | 13.1 | 19.6 | 13.1 | 81.9 | 0.6 | 1.1 | 995.2 | 50.4 | 11385.4 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 3 | A | 13.1 | 18.1 | 8.7 | 83.0 | 0.4 | 0.7 | 554.2 | 5.5 | 11321.4 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 4 | B | 12.6 | 25.0 | 8.7 | 83.3 | 0.3 | 0.8 | 338.1 | 7.0 | 9850.8 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 5 | B | 12.2 | 13.9 | 8.8 | 82.1 | 0.4 | 0.9 | 195.8 | 2.4 | 8849.0 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 6 | B | 12.5 | 14.7 | 8.3 | 82.0 | 0.4 | 0.8 | 291.5 | 1.4 | 8547.8 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 7 | B | 11.6 | 17.5 | 8.9 | 82.3 | 0.4 | 0.5 | 258.1 | 2.4 | 6990.6 | 13 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 8 | A | 12.4 | 18.1 | 9.7 | 83.4 | 0.5 | 0.7 | 298.0 | 19.7 | 10254.3 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 9 | A | 12.5 | 14.8 | 8.8 | 83.6 | 0.4 | 0.6 | 279.0 | 2.6 | 9817.1 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 307a5dbf3d13c067 | 307a5dbf3d13c067 | yes | yes | yes | - |

Runtime: 7.9 min for 2 rounds (3.5 min per round); a fixed 3 rounds would take 10.4 min (+3.5 min saved).
