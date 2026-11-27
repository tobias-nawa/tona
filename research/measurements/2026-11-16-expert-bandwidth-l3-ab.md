<worktree>/work/corpus/code.txt is missing, building it from e3660c0
## Plan (written before the run)

- Scope: workload short (2000:400 2000:write:400 at --ctx 140000); A `snap-base` against B `snap-l3b`; rounds 1 to 6, alternating AB/BA, stop when the primary metrics (decode) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: short: decode loop is the subject, 2K prompts, 400 tokens decode (code write request included) (decode tokens per prompt as in the targets, default 200)
- Duration: 1-7 min (1-6 rounds of 1.2 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: width curve l3 (2026-11-16-expert-bandwidth.md): -0.2 ms of 36 ms at W4):
  - decode: +0 to +1 %
  - prefill: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit unknown; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit unknown; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

waiting for the GPU lock <repo>/work/gpu.lock ...
gpu queue: 1 ahead of you (exclusive)
warm-up start of A (discarded)
round 1 (AB) done in 62 s
duration 2-6 min (2-6 rounds of 1.0 min, from round 1)
round 2 (BA) done in 62 s
round 3 (AB) done in 62 s
round 4 (BA) done in 62 s
round 5 (AB) done in 62 s
round 6 (BA) done in 62 s
raw data: /var/tmp/claude/v160/ab-l3.json
noise table with this run added: /var/tmp/claude/v160/ab-l3-noise.toml
not written; to take it: cp /var/tmp/claude/v160/ab-l3-noise.toml bench/noise.toml
## Plan (written before the run)

- Scope: workload short (2000:400 2000:write:400 at --ctx 140000); A `snap-base` against B `snap-l3b`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (decode) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: short: decode loop is the subject, 2K prompts, 400 tokens decode (code write request included) (decode tokens per prompt as in the targets, default 200)
- Duration: 2-6 min (2-6 rounds of 1.0 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: width curve l3 (2026-11-16-expert-bandwidth.md): -0.2 ms of 36 ms at W4):
  - decode: +0 to +1 %
  - prefill: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit unknown; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit unknown; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `snap-base` env -; B: `snap-l3b` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1431.1 / 1409.6 / 1408.4 / 1409.5 / 1418.8 / 1444.4 | 1463.8 / 1417.1 / 1414.2 / 1418.5 / 1463.2 / 1456.4 | 1414.2 (1408.4-1444.4) | 1437.5 (1414.2-1463.8) | +1.30 % [+0.06, +2.55] | **B faster** after round 6 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 74.7 / 74.6 / 74.5 / 75.4 / 74.7 / 74.7 | 74.8 / 74.9 / 74.7 / 75.3 / 74.9 / 74.5 | 74.7 (74.5-75.4) | 74.8 (74.5-75.3) | +0.16 % [-0.19, +0.51] | **equal** after round 6 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1731.2 / 1699.6 / 1701.9 / 1710.7 / 1728.9 / 1731.4 | 1681.5 / 1729.8 / 1708.1 / 1710.3 / 1744.1 / 1744.2 | 1719.8 (1699.6-1731.4) | 1720.1 (1681.5-1744.2) | +0.14 % [-1.62, +1.91] | open (undecided at the last round) |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 123.6 / 122.7 / 123.8 / 124.7 / 123.6 / 125.0 | 121.6 / 123.5 / 124.6 / 124.7 / 124.9 / 124.7 | 123.7 (122.7-125.0) | 124.7 (121.6-124.9) | +0.10 % [-1.12, +1.33] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +2.28 % (off plan) | +0.18 % | -2.87 % (off plan) | -1.63 % (off plan) | 62 |
| 2 | BA | +0.53 % (outlier) | +0.43 % | +1.78 % (outlier) | +0.71 % | 62 |
| 3 | AB | +0.41 % | +0.28 % | +0.36 % | +0.68 % | 62 |
| 4 | BA | +0.64 % | -0.04 % | -0.02 % | -0.01 % | 62 |
| 5 | AB | +3.13 % (outlier) | +0.31 % | +0.88 % (outlier) | +1.07 % | 62 |
| 6 | BA | +0.83 % | -0.22 % | +0.74 % | -0.21 % | 62 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +1.30 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode@2000 | +0.16 % | +0 to +1 % | within the expectation |
| prefill@2000:write | +0.14 % | -0.5 to +0.5 % | within the expectation |
| decode@2000:write | +0.10 % | +0 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 130767, 22082 / 132355, 20494 / 132773, 20076 / 132773, 20076 / 132773, 20076 / 132773, 20076 | 130767, 22082 / 132355, 20494 / 132773, 20076 / 132773, 20076 / 132773, 20076 / 132773, 20076 | NO |
| 2000:write | 83981, 26296 / 85766, 24511 / 86194, 24083 / 86194, 24083 / 86194, 24083 / 86194, 24083 | 83981, 26296 / 85766, 24511 / 86194, 24083 / 86194, 24083 / 86194, 24083 / 86194, 24083 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 6.8 min for 6 rounds (1.0 min per round); a fixed 3 rounds would take 3.1 min (-3.1 min saved).

