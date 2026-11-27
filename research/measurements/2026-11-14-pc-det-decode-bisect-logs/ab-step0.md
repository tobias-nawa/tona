## Plan (written before the run)

- Scope: workload mid (30000:200 at --ctx 140000); A `b6dd` against B `engine`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: same workload as the regression run (30000:200); mid only is the smallest that shows decode@30000 (decode tokens per prompt as in the targets, default 200)
- Duration: 3-3 min (2-2 rounds of 1.3 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: #125/#126 decode +1.5 to +2 % each, #129 prefill +6.9 to +8.1 % (2026-11-14-scale-run.md); regression run /var/tmp/claude/night/regr/ab.out found -5.65 %):
  - decode@30000: -2 to +4 %
  - prefill@30000: -1 to +9 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-490-gb6dd58cd44; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-735-g42e1beb752; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `b6dd` env {'TONA_EXPERT_PROFILE': '<repo>/work/profile/experts-coder.bin'}; B: `engine` env {'TONA_EXPERT_PROFILE': '<repo>/work/profile/experts-coder.bin'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1895.2 / 1865.2 | 1933.4 / 1933.2 | 1880.2 (1865.2-1895.2) | 1933.3 (1933.2-1933.4) | +2.83 % [+1.59, +4.07] | **B faster** after round 2 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 70.4 / 70.6 | 71.9 / 72.2 | 70.5 (70.4-70.6) | 72.0 (71.9-72.2) | +2.23 % [+1.42, +3.04] | **B faster** after round 2 |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +2.02 % | +2.20 % | 77 |
| 2 | BA | +3.64 % | +2.26 % | 78 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +2.83 % | -1 to +9 % | within the expectation |
| decode@30000 | +2.23 % | -2 to +4 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 67912, 9284 / 66759, 10437 | 65751, 8536 / 65751, 8536 | NO |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 82.6 | 18.6 | 2.6 | 0.0 | 0.0 | 0.5 | 28778.2 | 0.0 | 0.4 | 47 | 0.0 | 0.0 | - | 0.0 | - | - | 20.5 |
| 2 | A | 84.7 | 17.9 | 2.6 | 0.0 | 0.0 | 0.5 | 27474.5 | 0.0 | 0.5 | 47 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |
| 3 | B | 83.9 | 17.7 | 2.4 | 0.0 | 0.0 | 0.4 | 26182.2 | 0.0 | 0.3 | 47 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |
| 4 | B | 84.1 | 18.3 | 2.7 | 0.0 | 0.0 | 0.3 | 27456.4 | 0.0 | 0.2 | 47 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |
| 5 | A | 83.7 | 19.1 | 2.6 | 0.0 | 0.0 | 0.5 | 28886.7 | 11.6 | 0.2 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.6 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | c912d60cf2bbaa63 | ee58880d3f5b558f | yes | yes | NO | - |

The texts differ (normal mode: sampling of the drafter and rounding may differ; no verdict).

Runtime: 3.2 min for 2 rounds (1.3 min per round); a fixed 3 rounds would take 3.9 min (+1.3 min saved).
