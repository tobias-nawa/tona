## Plan (written before the run)

- Scope: workload server (8000:16 at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (prefill@8000) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: prefill@8000 alone decides whether the purgeable arena's wakes cost prefill (cost A/B: -1.6 to -5.4 % in 3 of 4 calm rounds); 16 decode tokens only to end each request (decode tokens per prompt as in the targets, default 200)
- Duration: 7-21 min (2-6 rounds of 3.4 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-11-16-mac-purgeable-prefill-server-ab.md (v2: -2.50 % [-4.11, -0.89] in 2 rounds) and a diagnosis start with the build that derives the idle time (-0.71 % [-2.02, +0.59]); the arena counters of B every 500 stops):
  - prefill@8000: -1 to +1 %
  - decode@8000: -3 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1172-g3bcdae3336-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_CEILING=1, TONA_METAL_ARENA_GB=auto, TONA_METAL_ARENA_GOVERN=1
  - B: build commit stand-2026-10-25-1172-g3bcdae3336-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_GB=auto, TONA_METAL_ARENA_GOVERN=1, TONA_METAL_ARENA_LOG=500, TONA_METAL_ARENA_PURGEABLE=1

## Result

A: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin', 'TONA_METAL_ARENA_GOVERN': '1', 'TONA_METAL_ARENA_CEILING': '1'}; B: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin', 'TONA_METAL_ARENA_GOVERN': '1', 'TONA_METAL_ARENA_PURGEABLE': '1', 'TONA_METAL_ARENA_LOG': '500'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 6, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 169.3 / 169.7 | 167.4 / 165.9 | 169.5 (169.3-169.7) | 166.7 (165.9-167.4) | -1.68 % [-2.97, -0.39] | **B slower** after round 2 |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 12.7 / 12.9 | 12.3 / 12.6 | 12.8 (12.7-12.9) | 12.5 (12.3-12.6) | -2.31 % [-3.43, -1.19] | **B slower** after round 2 |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | -1.14 % | -2.62 % | 205 |
| 2 | BA | -2.22 % | -2.01 % | 206 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -1.68 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode@8000 | -2.31 % | -3 to +3 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 13.1 | 14.4 | 9.5 | 82.0 | 0.5 | 0.8 | 362.5 | 1.8 | 10996.3 | 13 | 0.0 | 0.0 | - | 0.0 | 2 | - | 4.0 |
| 2 | A | 12.5 | 18.6 | 8.7 | 83.3 | 0.4 | 0.8 | 262.7 | 2.1 | 10367.8 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 3 | A | 12.6 | 14.2 | 9.6 | 83.0 | 0.5 | 0.9 | 304.9 | 0.9 | 10358.5 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 4 | B | 12.5 | 14.4 | 8.6 | 83.2 | 0.4 | 0.9 | 230.8 | 2.1 | 9603.0 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 5 | B | 12.7 | 15.0 | 9.1 | 82.7 | 0.4 | 0.7 | 262.7 | 1.7 | 9976.6 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 6 | B | 12.8 | 14.5 | 8.8 | 82.4 | 0.4 | 0.7 | 230.5 | 0.7 | 9631.3 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 7 | B | 12.8 | 14.5 | 8.4 | 82.3 | 0.4 | 0.7 | 204.6 | 1.3 | 9567.1 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 8 | A | 12.1 | 14.3 | 8.6 | 83.6 | 0.4 | 0.7 | 62.3 | 1.3 | 9143.5 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 9 | A | 12.0 | 14.5 | 8.6 | 83.7 | 0.4 | 0.5 | 237.1 | 1.4 | 9030.9 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 307a5dbf3d13c067 | 307a5dbf3d13c067 | yes | yes | yes | - |

Runtime: 7.7 min for 2 rounds (3.4 min per round); a fixed 3 rounds would take 10.3 min (+3.4 min saved).
