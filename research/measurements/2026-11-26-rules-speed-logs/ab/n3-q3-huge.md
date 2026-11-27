## Plan (written before the run)

- Scope: workload huge (512000:200 1000000:200 at --ctx 1048576, TONA_SEQS=1); A `n3-main` against B `n3-pr`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); no warm-up start.
- Why this size: select_checks.py names R7 for attn_wmma2.h, mega_seq.cpp, prefill_plan.cpp, prompt_cache.cpp: 512K and 1M are past 262K where 32-bit index overflows and plan limits would show (decode tokens per prompt as in the targets, default 200)
- Duration: 88-88 min (2-2 rounds of 43.8 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-11-26-scale-run.md c4 (prefill 1,536 / 1,164, decode 60.8 / 57.2 at 512K / 1M); two rounds (ab.py minimum), no warm-up start, no noise table for huge):
  - decode: -1 to +1 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit d43a0b3fa7; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
  - B: build commit f50cd70c7a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin

## Result

A: `n3-main` env -; B: `n3-pr` env -. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@512000: Prefill tok/s after 512000 tokens of prompt | 1469.5 / 1454.5 | 1462.6 / 1463.6 | 1462.0 (1454.5-1469.5) | 1463.1 (1462.6-1463.6) | +0.08 % [-0.95, +1.10] | open (undecided at the last round) |
| decode@512000: Decode tok/s after 512000 tokens of prompt | 59.9 / 60.3 | 60.1 / 60.4 | 60.1 (59.9-60.3) | 60.3 (60.1-60.4) | +0.31 % [-0.52, +1.13] | open (undecided at the last round) |
| prefill@1000000: Prefill tok/s after 1000000 tokens of prompt | 1076.7 / 1076.3 | 1078.6 / 1077.2 | 1076.5 (1076.3-1076.7) | 1077.9 (1077.2-1078.6) | +0.13 % [-0.69, +0.94] | open (undecided at the last round) |
| decode@1000000: Decode tok/s after 1000000 tokens of prompt | 56.4 / 56.1 | 56.1 / 56.1 | 56.2 (56.1-56.4) | 56.1 (56.1-56.1) | -0.26 % [-1.10, +0.58] | open (undecided at the last round) |

| Round | Order | prefill@512000 B vs A | decode@512000 B vs A | prefill@1000000 B vs A | decode@1000000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -0.47 % | +0.42 % | +0.18 % | -0.45 % | 2625 |
| 2 | BA | +0.62 % | +0.19 % | +0.08 % | -0.07 % | 2627 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@512000 | +0.08 % | -1 to +1 % | within the expectation |
| decode@512000 | +0.31 % | -1 to +1 % | within the expectation |
| prefill@1000000 | +0.13 % | -1 to +1 % | within the expectation |
| decode@1000000 | -0.26 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 512000 | 73665, 10839 / 73665, 10839 | 73665, 10839 / 73665, 10839 | yes |
| 1000000 | 67467, 14390 / 67467, 14390 | 67467, 14390 / 67467, 14390 | yes |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 512000 | 2.78 (3.04) / 2.78 (3.04) | 2.78 (3.04) / 2.78 (3.04) |
| 1000000 | 3.12 (3.30) / 3.12 (3.30) | 3.12 (3.30) / 3.12 (3.30) |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 108.1 | 4.7 | 12.2 | 0.0 | 0.0 | 0.3 | 53432.3 | 20.8 | 6.3 | 18 | 0.0 | 0.0 | - | 0.0 | - | - | 25.8 |
| 2 | B | 107.8 | 3.9 | 12.4 | 0.0 | 0.0 | 0.3 | 46131.0 | 161.3 | 13.9 | 20 | 0.0 | 0.0 | - | 0.0 | - | 1.4 | 25.1 |
| 3 | B | 108.0 | 4.8 | 12.6 | 0.0 | 0.0 | 0.3 | 56889.4 | 5.7 | 118.9 | 19 | 0.0 | 0.0 | - | 0.0 | - | 1.5 | 29.3 |
| 4 | A | 108.3 | 4.8 | 12.3 | 0.0 | 0.0 | 0.3 | 59451.0 | 5.6 | 62.2 | 20 | 0.0 | 0.0 | - | 0.0 | - | 1.4 | 22.6 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 512000 | 32e65dd9bc402413 | 32e65dd9bc402413 | yes | yes | yes | - |
| 1000000 | dc6c12a50914afe6 | dc6c12a50914afe6 | yes | yes | yes | - |

Runtime: 87.5 min for 2 rounds (43.8 min per round); a fixed 3 rounds would take 131.3 min (+43.8 min saved).
