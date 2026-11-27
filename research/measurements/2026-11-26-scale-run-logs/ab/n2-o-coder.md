## Plan (written before the run)

- Scope: workload server (2000:200 16000:100 64000:50 at --ctx 262144); A `scale1126` against B `scale1126`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: the offer.json rows of the Coder (2K, 16K, 64K code summary, MTP); same targets as 2026-11-20-coder-decode.md; A/A gives the spread; 2 rounds is the minimum (decode tokens per prompt as in the targets, default 200)
- Duration: 4-4 min (2-2 rounds of 1.9 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: A/A, same build: B against A is 0 within the known noise):
  - decode: -3 to +3 %
  - prefill: -3 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1468-g0f0736635d; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=<worktree>/work/diag/gpucore-%p.core, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
  - B: build commit v0.0.2-1468-g0f0736635d; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=<worktree>/work/diag/gpucore-%p.core, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin

## Result

A: `scale1126` env -; B: `scale1126` env -. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1134.2 / 1188.1 | 1063.5 / 1124.5 | 1161.2 (1134.2-1188.1) | 1094.0 (1063.5-1124.5) | -5.80 % [-6.76, -4.84] | **B slower** after round 2 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 71.5 / 70.6 | 70.1 / 71.7 | 71.1 (70.6-71.5) | 70.9 (70.1-71.7) | -0.22 % [-2.47, +2.04] | open (undecided at the last round) |
| prefill@16000: Prefill tok/s after 16000 tokens of prompt | 1911.7 / 1929.0 | 1904.7 / 1901.9 | 1920.4 (1911.7-1929.0) | 1903.3 (1901.9-1904.7) | -0.89 % [-1.90, +0.12] | open (undecided at the last round) |
| decode@16000: Decode tok/s after 16000 tokens of prompt | 68.1 / 67.9 | 67.6 / 68.0 | 68.0 (67.9-68.1) | 67.8 (67.6-68.0) | -0.22 % [-1.17, +0.74] | open (undecided at the last round) |
| prefill@64000: Prefill tok/s after 64000 tokens of prompt | 1982.8 / 1990.6 | 1976.2 / 1975.4 | 1986.7 (1982.8-1990.6) | 1975.8 (1975.4-1976.2) | -0.55 % [-1.40, +0.30] | open (undecided at the last round) |
| decode@64000: Decode tok/s after 64000 tokens of prompt | 73.9 / 74.0 | 74.0 / 73.9 | 74.0 (73.9-74.0) | 73.9 (73.9-74.0) | -0.10 % [-0.93, +0.73] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@16000 B vs A | decode@16000 B vs A | prefill@64000 B vs A | decode@64000 B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | -6.24 % | -2.04 % | -0.36 % | -0.65 % | -0.33 % | +0.04 % | 115 |
| 2 | BA | -5.36 % | +1.61 % | -1.41 % | +0.22 % | -0.76 % | -0.25 % | 117 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | -5.80 % | -3 to +3 % | **OUTSIDE the expectation: finding** |
| decode@2000 | -0.22 % | -3 to +3 % | within the expectation |
| prefill@16000 | -0.89 % | -3 to +3 % | within the expectation |
| decode@16000 | -0.22 % | -3 to +3 % | within the expectation |
| prefill@64000 | -0.55 % | -3 to +3 % | within the expectation |
| decode@64000 | -0.10 % | -3 to +3 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 69867, 10833 / 69867, 10833 | 69867, 10833 / 69867, 10833 | yes |
| 16000 | 35866, 4503 / 35866, 4503 | 35866, 4503 / 35866, 4503 | yes |
| 64000 | 16383, 1790 / 16383, 1790 | 16383, 1790 / 16383, 1790 | yes |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000 | 3.33 (3.85) / 3.33 (3.85) | 3.33 (3.85) / 3.33 (3.85) |
| 16000 | 3.00 (3.18) / 3.00 (3.18) | 3.00 (3.18) / 3.00 (3.18) |
| 64000 | 3.33 (3.60) / 3.33 (3.60) | 3.33 (3.60) / 3.33 (3.60) |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 148.9 | 36.4 | 15.3 | 0.0 | 0.0 | 0.3 | 33776.4 | 18.9 | 1012.7 | 30 | 0.0 | 0.0 | - | 0.0 | - | - | 19.3 |
| 2 | A | 154.4 | 36.5 | 12.9 | 0.0 | 0.0 | 0.3 | 32575.2 | 1.6 | 1.2 | 31 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 16.2 |
| 3 | B | 152.3 | 38.4 | 13.5 | 0.0 | 0.0 | 0.3 | 34487.7 | 4.0 | 24.5 | 31 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 18.2 |
| 4 | B | 151.1 | 38.2 | 13.5 | 0.0 | 0.0 | 0.2 | 34120.5 | 7.7 | 52.5 | 31 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 19.5 |
| 5 | A | 157.0 | 35.3 | 12.6 | 0.0 | 0.0 | 0.4 | 32523.9 | 48.2 | 84.9 | 31 | 0.0 | 0.0 | - | 0.0 | - | 1.2 | 16.3 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | c84123c5356cbe1b | c84123c5356cbe1b | yes | yes | yes | - |
| 16000 | f8ad4e0401b3f627 | f8ad4e0401b3f627 | yes | yes | yes | - |
| 64000 | 038e48a132b20f47 | 038e48a132b20f47 | yes | yes | yes | - |

Runtime: 4.9 min for 2 rounds (1.9 min per round); a fixed 3 rounds would take 5.8 min (+1.9 min saved).
