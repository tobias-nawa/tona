## Plan (written before the run)

- Scope: workload server (2000:200 16000:200 64000:100 at --ctx 262144); A `work/build/clean175` against B `work/build/clean175`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: Bolt code at 2K/16K/64K, the other default row; 2 rounds is the minimum of ab.py (decode tokens per prompt as in the targets, default 200)
- Duration: 10-10 min (2-2 rounds of 5.1 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: A/A, same build: B against A is 0 within the known noise):
  - decode: -1 to +1 %
  - prefill: -2 to +2 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1382-g936dfdc44a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/nemotron-bolt-74b-a9b-24gb.bin
  - B: build commit v0.0.2-1382-g936dfdc44a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/nemotron-bolt-74b-a9b-24gb.bin

## Result

A: `work/build/clean175` env -; B: `work/build/clean175` env -. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1414.4 / 1416.2 | 1414.5 / 1417.7 | 1415.3 (1414.4-1416.2) | 1416.1 (1414.5-1417.7) | +0.06 % [-0.76, +0.87] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 40.1 / 40.1 | 40.1 / 40.1 | 40.1 (40.1-40.1) | 40.1 (40.1-40.1) | -0.06 % [-0.88, +0.76] | open (undecided at the last round) |
| prefill@16000: Prefill tok/s after 16000 tokens of prompt | 1571.7 / 1583.9 | 1570.0 / 1567.5 | 1577.8 (1571.7-1583.9) | 1568.8 (1567.5-1570.0) | -0.57 % [-1.54, +0.40] | open (undecided at the last round) |
| decode@16000: Decode tok/s after 16000 tokens of prompt | 38.6 / 38.5 | 38.6 / 38.5 | 38.6 (38.5-38.6) | 38.6 (38.5-38.6) | +0.01 % [-0.81, +0.83] | open (undecided at the last round) |
| prefill@64000: Prefill tok/s after 64000 tokens of prompt | 1424.9 / 1433.7 | 1424.2 / 1422.1 | 1429.3 (1424.9-1433.7) | 1423.2 (1422.1-1424.2) | -0.43 % [-1.35, +0.50] | open (undecided at the last round) |
| decode@64000: Decode tok/s after 64000 tokens of prompt | 36.3 / 36.3 | 36.3 / 36.4 | 36.3 (36.3-36.3) | 36.3 (36.3-36.4) | +0.03 % [-0.78, +0.85] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@16000 B vs A | decode@16000 B vs A | prefill@64000 B vs A | decode@64000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | +0.01 % | +0.05 % | -0.11 % | +0.08 % | -0.05 % | -0.00 % | 308 |
| 2 | BA | +0.11 % | -0.17 % | -1.03 % | -0.06 % | -0.81 % | +0.07 % | 308 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.06 % | -2 to +2 % | within the expectation |
| decode@2000 | -0.06 % | -1 to +1 % | within the expectation |
| prefill@16000 | -0.57 % | -2 to +2 % | within the expectation |
| decode@16000 | +0.01 % | -1 to +1 % | within the expectation |
| prefill@64000 | -0.43 % | -2 to +2 % | within the expectation |
| decode@64000 | +0.03 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 34459, 37541 / 34337, 37663 | 34459, 37541 / 34459, 37541 | NO |
| 16000 | 30030, 41970 / 29873, 42127 | 30030, 41970 / 30030, 41970 | NO |
| 64000 | 18397, 17603 / 18315, 17685 | 18397, 17603 / 18397, 17603 | NO |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000 | 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) |
| 16000 | 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) |
| 64000 | 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 929d9d0d30b0f75a | 929d9d0d30b0f75a | yes | yes | yes | - |
| 16000 | c5260d1b5f1fe082 | c5260d1b5f1fe082 | yes | yes | yes | - |
| 64000 | 4a5a7f1af555b2e8 | 4a5a7f1af555b2e8 | yes | yes | yes | - |

Runtime: 12.9 min for 2 rounds (5.1 min per round); a fixed 3 rounds would take 15.4 min (+5.1 min saved).
