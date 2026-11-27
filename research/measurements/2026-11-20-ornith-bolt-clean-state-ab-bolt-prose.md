## Plan (written before the run)

- Scope: workload prose,longdoc (2000:prose:400 16000:prose:400 at --ctx 32768; 65536:longdoc:200 at --ctx 98304); A `work/build/clean175` against B `work/build/clean175`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: Bolt's default workloads (2K and 16K prose, 64K document); the 2-round minimum of ab.py gives the A/A spread (decode tokens per prompt as in the targets, default 200)
- Duration: 18-18 min (2-2 rounds of 9.1 min, from round 1)
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
| prefill@2000:prose: Prefill tok/s after 2000:prose tokens of prompt | 1417.7 / 1416.0 | 1410.6 / 1407.7 | 1416.9 (1416.0-1417.7) | 1409.2 (1407.7-1410.6) | -0.55 % [-1.36, +0.27] | open (undecided at the last round) |
| decode@2000:prose: Decode tok/s after 2000:prose tokens of prompt | 40.4 / 40.3 | 40.3 / 40.4 | 40.3 (40.3-40.4) | 40.3 (40.3-40.4) | -0.00 % [-0.82, +0.82] | open (undecided at the last round) |
| prefill@16000:prose: Prefill tok/s after 16000:prose tokens of prompt | 1614.8 / 1615.8 | 1617.5 / 1616.8 | 1615.3 (1614.8-1615.8) | 1617.1 (1616.8-1617.5) | +0.11 % [-0.70, +0.93] | open (undecided at the last round) |
| decode@16000:prose: Decode tok/s after 16000:prose tokens of prompt | 39.4 / 39.3 | 39.4 / 39.4 | 39.4 (39.3-39.4) | 39.4 (39.4-39.4) | +0.01 % [-0.81, +0.83] | open (undecided at the last round) |
| prefill@65536:longdoc: Prefill tok/s after 65536:longdoc tokens of prompt | 1524.2 / 1522.8 | 1524.5 / 1523.2 | 1523.5 (1522.8-1524.2) | 1523.9 (1523.2-1524.5) | +0.02 % [-0.79, +0.84] | open (undecided at the last round) |
| decode@65536:longdoc: Decode tok/s after 65536:longdoc tokens of prompt | 36.7 / 36.7 | 36.7 / 36.7 | 36.7 (36.7-36.7) | 36.7 (36.7-36.7) | +0.01 % [-0.80, +0.83] | open (undecided at the last round) |

| Round | Order | prefill@2000:prose B vs A | decode@2000:prose B vs A | prefill@16000:prose B vs A | decode@16000:prose B vs A | prefill@65536:longdoc B vs A | decode@65536:longdoc B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | -0.50 % | -0.06 % | +0.17 % | -0.08 % | +0.02 % | +0.09 % | 544 |
| 2 | BA | -0.59 % | +0.06 % | +0.06 % | +0.11 % | +0.02 % | -0.06 % | 544 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000:prose | -0.55 % | -2 to +2 % | within the expectation |
| decode@2000:prose | -0.00 % | -1 to +1 % | within the expectation |
| prefill@16000:prose | +0.11 % | -2 to +2 % | within the expectation |
| decode@16000:prose | +0.01 % | -1 to +1 % | within the expectation |
| prefill@65536:longdoc | +0.02 % | -2 to +2 % | within the expectation |
| decode@65536:longdoc | +0.01 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000:prose | 69240, 74760 / 69240, 74760 | 69240, 74760 / 69240, 74760 | yes |
| 16000:prose | 68775, 75225 / 68775, 75225 | 68775, 75225 / 68775, 75225 | yes |
| 65536:longdoc | 36858, 35142 / 36858, 35142 | 36858, 35142 / 36858, 35142 | yes |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000:prose | 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) |
| 16000:prose | 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) |
| 65536:longdoc | 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000:prose | 03bc3042e3ebf8fa | 03bc3042e3ebf8fa | yes | yes | yes | - |
| 16000:prose | 4094a16a506c4e33 | 4094a16a506c4e33 | yes | yes | yes | - |
| 65536:longdoc | 147f5cba05557d02 | 147f5cba05557d02 | yes | yes | yes | - |

Runtime: 22.7 min for 2 rounds (9.1 min per round); a fixed 3 rounds would take 27.2 min (+9.1 min saved).
