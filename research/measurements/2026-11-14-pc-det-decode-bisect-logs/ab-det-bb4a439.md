## Plan (written before the run)

- Scope: workload mid (30000:200 at --ctx 140000); A `b6dd` against B `bis-bb4a439`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: same workload as the regression run (30000:200); mid only is the smallest that shows decode@30000 (decode tokens per prompt as in the targets, default 200)
- Duration: 3-3 min (2-2 rounds of 1.3 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: #125/#126 decode +1.5 to +2 % each, #129 prefill +6.9 to +8.1 % (2026-11-14-scale-run.md); regression run /var/tmp/claude/night/regr/ab.out found -5.65 %):
  - decode@30000: -2 to +4 %
  - prefill@30000: -1 to +9 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-490-gb6dd58cd44; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-612-gbb4a439c03; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `b6dd` env {'TONA_EXPERT_PROFILE': '<repo>/work/profile/experts-coder.bin', 'TONA_DETERMINISTIC': '1'}; B: `bis-bb4a439` env {'TONA_EXPERT_PROFILE': '<repo>/work/profile/experts-coder.bin', 'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1866.1 / 1866.7 | 1865.0 / 1865.3 | 1866.4 (1866.1-1866.7) | 1865.2 (1865.0-1865.3) | -0.07 % [-0.88, +0.75] | open (undecided at the last round) |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 71.8 / 71.9 | 71.3 / 71.4 | 71.8 (71.8-71.9) | 71.3 (71.3-71.4) | -0.67 % [-1.49, +0.14] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | -0.06 % | -0.69 % | 78 |
| 2 | BA | -0.07 % | -0.65 % | 79 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | -0.07 % | -1 to +9 % | within the expectation |
| decode@30000 | -0.67 % | -2 to +4 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 68347, 9347 / 68347, 9347 | 67462, 9026 / 67462, 9026 | NO |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 82.8 | 18.4 | 6.3 | 0.0 | 0.0 | 0.4 | 27475.2 | 0.3 | 0.1 | 46 | 0.0 | 0.0 | - | 0.0 | - | - | 20.4 |
| 2 | A | 84.2 | 17.8 | 5.4 | 0.0 | 0.0 | 0.7 | 27453.1 | 0.0 | 0.1 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |
| 3 | B | 84.0 | 17.8 | 3.4 | 0.0 | 0.0 | 0.6 | 27335.3 | 0.6 | 0.1 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 4 | B | 83.6 | 18.0 | 3.9 | 0.0 | 0.0 | 0.7 | 27430.3 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |
| 5 | A | 83.7 | 18.1 | 2.3 | 0.0 | 0.0 | 0.6 | 27439.5 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.5 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | ee58880d3f5b558f (B = ref) |

Runtime: 3.3 min for 2 rounds (1.3 min per round); a fixed 3 rounds would take 3.9 min (+1.3 min saved).
