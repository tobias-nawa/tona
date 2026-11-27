## Plan (written before the run)

- Scope: workload server (2000:200 16000:200 64000:100 128000:100 at --ctx 262144); A `work/build/clean175` against B `work/build/clean175`; rounds 2 to 3, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: clean absolute decode and prefill at the four context points of the README; 200/100 decode tokens give a stable rate (about 2 s of decode) (decode tokens per prompt as in the targets, default 200)
- Duration: 12-18 min (2-3 rounds of 5.9 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: A/A, same build: B against A is 0 within the known noise (bench/noise.toml); the absolute values are the result and are planned in the measurement file):
  - decode@2000: -1 to +1 %
  - decode@16000: -1 to +1 %
  - decode@64000: -1 to +1 %
  - decode@128000: -1 to +1 %
  - prefill@2000: -3 to +3 %
  - prefill@16000: -1 to +1 %
  - prefill@64000: -1 to +1 %
  - prefill@128000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1382-g936dfdc44a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/ornith-1.5-35b-a3b.bin
  - B: build commit v0.0.2-1382-g936dfdc44a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/ornith-1.5-35b-a3b.bin

## Result

A: `work/build/clean175` env -; B: `work/build/clean175` env -. Alpha 0.05, margin 0.5 %, rounds 3 of at most 3, stop: 3 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 2585.5 / 2580.2 / 2582.3 | 2579.6 / 2583.6 / 2582.2 | 2582.3 (2580.2-2585.5) | 2582.2 (2579.6-2583.6) | -0.03 % [-0.90, +0.83] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 114.3 / 114.3 / 114.4 | 114.4 / 114.2 / 114.4 | 114.3 (114.3-114.4) | 114.4 (114.2-114.4) | -0.01 % [-0.86, +0.85] | open (undecided at the last round) |
| prefill@16000: Prefill tok/s after 16000 tokens of prompt | 4082.8 / 4081.6 / 4077.9 | 4078.7 / 4089.5 / 4079.8 | 4081.6 (4077.9-4082.8) | 4079.8 (4078.7-4089.5) | +0.05 % [-0.81, +0.91] | open (undecided at the last round) |
| decode@16000: Decode tok/s after 16000 tokens of prompt | 95.5 / 95.5 / 95.4 | 95.5 / 95.3 / 95.5 | 95.5 (95.4-95.5) | 95.5 (95.3-95.5) | -0.02 % [-0.89, +0.84] | open (undecided at the last round) |
| prefill@64000: Prefill tok/s after 64000 tokens of prompt | 2964.8 / 2963.5 / 2963.8 | 2964.2 / 2963.8 / 2962.9 | 2963.8 (2963.5-2964.8) | 2963.8 (2962.9-2964.2) | -0.01 % [-0.85, +0.83] | open (undecided at the last round) |
| decode@64000: Decode tok/s after 64000 tokens of prompt | 64.7 / 64.7 / 64.7 | 64.6 / 64.6 / 64.8 | 64.7 (64.7-64.7) | 64.6 (64.6-64.8) | -0.02 % [-0.87, +0.83] | open (undecided at the last round) |
| prefill@128000: Prefill tok/s after 128000 tokens of prompt | 2191.6 / 2191.5 / 2191.0 | 2191.0 / 2192.7 / 2191.0 | 2191.5 (2191.0-2191.6) | 2191.0 (2191.0-2192.7) | +0.01 % [-0.83, +0.85] | open (undecided at the last round) |
| decode@128000: Decode tok/s after 128000 tokens of prompt | 47.5 / 47.5 / 47.4 | 47.5 / 47.4 / 47.4 | 47.5 (47.4-47.5) | 47.4 (47.4-47.5) | -0.07 % [-0.92, +0.78] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@16000 B vs A | decode@16000 B vs A | prefill@64000 B vs A | decode@64000 B vs A | prefill@128000 B vs A | decode@128000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AB | -0.23 % | +0.12 % | -0.10 % | +0.02 % | -0.02 % | -0.12 % | -0.03 % | -0.03 % | 352 |
| 2 | BA | +0.13 % | -0.08 % | +0.20 % | -0.23 % | +0.01 % | -0.03 % | +0.05 % | -0.20 % | 352 |
| 3 | AB | -0.00 % | -0.06 % | +0.05 % | +0.13 % | -0.03 % | +0.09 % | -0.00 % | +0.02 % | 352 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | -0.03 % | -3 to +3 % | within the expectation |
| decode@2000 | -0.01 % | -1 to +1 % | within the expectation |
| prefill@16000 | +0.05 % | -1 to +1 % | within the expectation |
| decode@16000 | -0.02 % | -1 to +1 % | within the expectation |
| prefill@64000 | -0.01 % | -1 to +1 % | within the expectation |
| decode@64000 | -0.02 % | -1 to +1 % | within the expectation |
| prefill@128000 | +0.01 % | -1 to +1 % | within the expectation |
| decode@128000 | -0.07 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 53986, 5176 / 53986, 5176 / 53986, 5176 | 53986, 5176 / 53903, 5259 / 53986, 5176 | NO |
| 16000 | 59073, 5275 / 59073, 5275 / 59073, 5275 | 59073, 5275 / 58975, 5373 / 59073, 5275 | NO |
| 64000 | 29552, 2495 / 29552, 2495 / 29552, 2495 | 29552, 2495 / 29499, 2548 / 29552, 2495 | NO |
| 128000 | 28777, 2472 / 28777, 2472 / 28777, 2472 | 28777, 2472 / 28735, 2514 / 28777, 2472 | NO |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000 | 2.38 (2.52) / 2.38 (2.52) / 2.38 (2.52) | 2.38 (2.52) / 2.38 (2.52) / 2.38 (2.52) |
| 16000 | 2.21 (2.43) / 2.21 (2.43) / 2.21 (2.43) | 2.21 (2.43) / 2.21 (2.43) / 2.21 (2.43) |
| 64000 | 2.08 (2.33) / 2.08 (2.33) / 2.08 (2.33) | 2.08 (2.33) / 2.08 (2.33) / 2.08 (2.33) |
| 128000 | 2.17 (2.30) / 2.17 (2.30) / 2.17 (2.30) | 2.17 (2.30) / 2.17 (2.30) / 2.17 (2.30) |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | db86c9ed5ae758cc | db86c9ed5ae758cc | yes | yes | yes | - |
| 16000 | 7d4d0e17363c8e06 | 7d4d0e17363c8e06 | yes | yes | yes | - |
| 64000 | 7ee9edb5706ccf03 | 7ee9edb5706ccf03 | yes | yes | yes | - |
| 128000 | ea7cf93e3fda55c7 | ea7cf93e3fda55c7 | yes | yes | yes | - |

Runtime: 20.6 min for 3 rounds (5.9 min per round); a fixed 3 rounds would take 17.6 min (+0.0 min saved).
