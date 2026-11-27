## Plan (written before the run)

- Scope: workload prose (2000:prose:400 16000:prose:400 at --ctx 32768); A `work/build/clean175` against B `work/build/clean175`; rounds 2 to 3, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: prose at 2K and 16K is Ornith's text workload; 400 decode tokens, as the workload is defined (decode tokens per prompt as in the targets, default 200)
- Duration: 7-11 min (2-3 rounds of 3.6 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: A/A, same build: B against A is 0 within the known noise):
  - decode: -1 to +1 %
  - prefill: -3 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1382-g936dfdc44a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/ornith-1.5-35b-a3b.bin
  - B: build commit v0.0.2-1382-g936dfdc44a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/ornith-1.5-35b-a3b.bin

## Result

A: `work/build/clean175` env -; B: `work/build/clean175` env -. Alpha 0.05, margin 0.5 %, rounds 3 of at most 3, stop: 3 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000:prose: Prefill tok/s after 2000:prose tokens of prompt | 2756.2 / 2682.0 / 2673.4 | 2676.2 / 2675.1 / 2680.3 | 2682.0 (2673.4-2756.2) | 2676.2 (2675.1-2680.3) | -0.97 % [-3.15, +1.22] | open (undecided at the last round) |
| decode@2000:prose: Decode tok/s after 2000:prose tokens of prompt | 116.0 / 116.5 / 116.8 | 116.6 / 116.7 / 116.5 | 116.5 (116.0-116.8) | 116.6 (116.5-116.7) | +0.12 % [-0.85, +1.08] | open (undecided at the last round) |
| prefill@16000:prose: Prefill tok/s after 16000:prose tokens of prompt | 4089.6 / 4088.7 / 4088.6 | 4088.1 / 4086.9 / 4092.6 | 4088.7 (4088.6-4089.6) | 4088.1 (4086.9-4092.6) | +0.01 % [-0.84, +0.85] | open (undecided at the last round) |
| decode@16000:prose: Decode tok/s after 16000:prose tokens of prompt | 101.2 / 101.2 / 101.2 | 101.1 / 101.1 / 101.1 | 101.2 (101.2-101.2) | 101.1 (101.1-101.1) | -0.06 % [-0.90, +0.78] | open (undecided at the last round) |

| Round | Order | prefill@2000:prose B vs A | decode@2000:prose B vs A | prefill@16000:prose B vs A | decode@16000:prose B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -2.90 % | +0.47 % | -0.04 % | -0.05 % | 216 |
| 2 | BA | -0.26 % | +0.19 % | -0.04 % | -0.05 % | 216 |
| 3 | AB | +0.26 % | -0.31 % | +0.10 % | -0.08 % | 216 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000:prose | -0.97 % | -3 to +3 % | within the expectation |
| decode@2000:prose | +0.12 % | -1 to +1 % | within the expectation |
| prefill@16000:prose | +0.01 % | -3 to +3 % | within the expectation |
| decode@16000:prose | -0.06 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000:prose | 116883, 7302 / 116883, 7302 / 116883, 7302 | 116883, 7302 / 116883, 7302 / 116883, 7302 | yes |
| 16000:prose | 118538, 6601 / 118538, 6601 / 118538, 6601 | 118538, 6601 / 118538, 6601 / 118538, 6601 | yes |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000:prose | 2.36 (2.60) / 2.36 (2.60) / 2.36 (2.60) | 2.36 (2.60) / 2.36 (2.60) / 2.36 (2.60) |
| 16000:prose | 2.20 (2.43) / 2.20 (2.43) / 2.20 (2.43) | 2.20 (2.43) / 2.20 (2.43) / 2.20 (2.43) |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000:prose | c612de3620abc451 | c612de3620abc451 | yes | yes | yes | - |
| 16000:prose | 831b87c6f1109eac | 831b87c6f1109eac | yes | yes | yes | - |

Runtime: 12.7 min for 3 rounds (3.6 min per round); a fixed 3 rounds would take 10.8 min (+0.0 min saved).
