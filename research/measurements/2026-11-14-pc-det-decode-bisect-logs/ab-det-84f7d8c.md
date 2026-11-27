## Plan (written before the run)

- Scope: workload mid (30000:200 at --ctx 140000); A `b6dd` against B `bis-84f7d8c`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: same workload as the regression run (30000:200); mid only is the smallest that shows decode@30000 (decode tokens per prompt as in the targets, default 200)
- Duration: 3-3 min (2-2 rounds of 1.4 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: #125/#126 decode +1.5 to +2 % each, #129 prefill +6.9 to +8.1 % (2026-11-14-scale-run.md); regression run /var/tmp/claude/night/regr/ab.out found -5.65 %):
  - decode@30000: -2 to +4 %
  - prefill@30000: -1 to +9 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-490-gb6dd58cd44; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-712-g84f7d8cf34; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `b6dd` env {'TONA_EXPERT_PROFILE': '<repo>/work/profile/experts-coder.bin', 'TONA_DETERMINISTIC': '1'}; B: `bis-84f7d8c` env {'TONA_EXPERT_PROFILE': '<repo>/work/profile/experts-coder.bin', 'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1864.6 / 1865.4 | 1887.2 / 1887.4 | 1865.0 (1864.6-1865.4) | 1887.3 (1887.2-1887.4) | +1.20 % [+0.38, +2.01] | **B faster** after round 2 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 71.9 / 71.9 | 67.4 / 67.5 | 71.9 (71.9-71.9) | 67.5 (67.4-67.5) | -6.19 % [-7.00, -5.38] | **B slower** after round 2 |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +1.21 % | -6.21 % | 84 |
| 2 | BA | +1.18 % | -6.17 % | 78 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +1.20 % | -1 to +9 % | within the expectation |
| decode@30000 | -6.19 % | -2 to +4 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 68347, 9347 / 68347, 9347 | 70213, 9092 / 70213, 9092 | NO |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 83.1 | 18.1 | 2.2 | 0.0 | 0.0 | 0.6 | 27425.1 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | - | 20.3 |
| 2 | A | 84.0 | 18.1 | 2.1 | 0.0 | 0.0 | 0.5 | 27393.8 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |
| 3 | B | 130.2 | 15.3 | 2.0 | 0.0 | 0.0 | 0.7 | 27389.0 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 26.0 |
| 4 | B | 84.3 | 17.9 | 2.2 | 0.0 | 0.0 | 0.8 | 27417.6 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 5 | A | 84.0 | 18.0 | 2.1 | 0.0 | 0.0 | 0.6 | 27394.9 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | ee58880d3f5b558f (B = ref) |

Runtime: 3.4 min for 2 rounds (1.4 min per round); a fixed 3 rounds would take 4.1 min (+1.4 min saved).
