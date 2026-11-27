## Plan (written before the run)

- Scope: workload server (2000:200 16000:200 64000:100 at --ctx 262144); A `scale1126` against B `scale1126`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: the offer.json rows of Bolt (2K, 16K, 64K code summary, no drafter, history drafts); targets as 2026-11-20-ornith-bolt-clean-state.md; 2 rounds is the minimum of ab.py (decode tokens per prompt as in the targets, default 200)
- Duration: 4-4 min (2-2 rounds of 2.2 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: A/A, same build: B against A is 0 within the known noise):
  - decode: -3 to +3 %
  - prefill: -3 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1468-g0f0736635d; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=<worktree>/work/diag/gpucore-%p.core, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/nemotron-bolt-74b-a9b-24gb.bin
  - B: build commit v0.0.2-1468-g0f0736635d; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=<worktree>/work/diag/gpucore-%p.core, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/nemotron-bolt-74b-a9b-24gb.bin

## Result

A: `scale1126` env -; B: `scale1126` env -. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1555.0 / 1438.6 | 1431.6 / 1558.2 | 1496.8 (1438.6-1555.0) | 1494.9 (1431.6-1558.2) | +0.19 % [-9.19, +9.56] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 41.0 / 41.1 | 41.0 / 41.1 | 41.0 (41.0-41.1) | 41.1 (41.0-41.1) | +0.11 % [-0.70, +0.92] | open (undecided at the last round) |
| prefill@16000: Prefill tok/s after 16000 tokens of prompt | 1625.1 / 1623.4 | 1623.0 / 1626.0 | 1624.3 (1623.4-1625.1) | 1624.5 (1623.0-1626.0) | +0.02 % [-0.81, +0.85] | open (undecided at the last round) |
| decode@16000: Decode tok/s after 16000 tokens of prompt | 39.6 / 39.8 | 39.7 / 39.7 | 39.7 (39.6-39.8) | 39.7 (39.7-39.7) | +0.06 % [-0.79, +0.92] | open (undecided at the last round) |
| prefill@64000: Prefill tok/s after 64000 tokens of prompt | 1479.8 / 1465.3 | 1466.3 / 1466.1 | 1472.5 (1465.3-1479.8) | 1466.2 (1466.1-1466.3) | -0.43 % [-1.41, +0.55] | open (undecided at the last round) |
| decode@64000: Decode tok/s after 64000 tokens of prompt | 36.8 / 36.8 | 37.0 / 36.9 | 36.8 (36.8-36.8) | 36.9 (36.9-37.0) | +0.34 % [-0.48, +1.16] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@16000 B vs A | decode@16000 B vs A | prefill@64000 B vs A | decode@64000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | -7.94 % | +0.10 % | -0.13 % | +0.28 % | -0.91 % | +0.44 % | 129 |
| 2 | BA | +8.31 % | +0.12 % | +0.16 % | -0.15 % | +0.05 % | +0.24 % | 130 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.19 % | -3 to +3 % | within the expectation |
| decode@2000 | +0.11 % | -3 to +3 % | within the expectation |
| prefill@16000 | +0.02 % | -3 to +3 % | within the expectation |
| decode@16000 | +0.06 % | -3 to +3 % | within the expectation |
| prefill@64000 | -0.43 % | -3 to +3 % | within the expectation |
| decode@64000 | +0.34 % | -3 to +3 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 45349, 26651 / 45349, 26651 | 45349, 26651 / 45349, 26651 | yes |
| 16000 | 43195, 28805 / 43195, 28805 | 43195, 28805 / 43195, 28805 | yes |
| 64000 | 24603, 12137 / 24603, 12137 | 24603, 12137 / 24603, 12137 | yes |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000 | 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) |
| 16000 | 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) |
| 64000 | 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 148.2 | 14.7 | 17.8 | 0.0 | 0.0 | 1.5 | 21812.4 | 7.6 | 0.0 | 54 | 0.0 | 0.0 | - | 0.0 | - | - | 8.0 |
| 2 | A | 146.1 | 16.9 | 19.0 | 0.0 | 0.0 | 1.5 | 22434.7 | 29.0 | 0.0 | 55 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 8.3 |
| 3 | B | 152.8 | 15.1 | 14.0 | 0.0 | 0.0 | 1.6 | 19558.2 | 18.5 | 0.0 | 55 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 6.0 |
| 4 | B | 147.6 | 16.1 | 12.6 | 0.0 | 0.0 | 1.5 | 23448.7 | 2.2 | 0.0 | 55 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 8.5 |
| 5 | A | 153.1 | 15.6 | 14.9 | 0.0 | 0.0 | 1.5 | 20405.8 | 1.0 | 0.0 | 55 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 6.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 929d9d0d30b0f75a | 929d9d0d30b0f75a | yes | yes | yes | - |
| 16000 | c5260d1b5f1fe082 | c5260d1b5f1fe082 | yes | yes | yes | - |
| 64000 | 4a5a7f1af555b2e8 | 4a5a7f1af555b2e8 | yes | yes | yes | - |

Runtime: 5.4 min for 2 rounds (2.2 min per round); a fixed 3 rounds would take 6.5 min (+2.2 min saved).
