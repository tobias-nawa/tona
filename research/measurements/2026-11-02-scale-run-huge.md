## Plan (written before the run)

- Scope: workload huge (512000:200 1000000:200 at --ctx 1048576, IE_SEQS=1); A `merged` against B `w83`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (decode,prefill) are decided (alpha 0.05, margin 0.5 %); no warm-up start.
- Why this size: past 262K the registered share of the pinned experts grows (1.2 GB at 140K, 4.7 GB at 262K); 512K and 1M are the sizes where earlier notes have numbers (decode tokens per prompt as in the targets, default 200)
- Duration: 87-87 min (2-2 rounds of 43.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-10-31-long-context-topk.md (512K: prefill 1434, decode 60.5) and -yarn.md section 2 (1M); 2026-11-01-slow-decode.md (register route)):
  - decode: -2 to +0 %
  - prefill: -2 to +1 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-148-g8a08c6b73e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=./work/diag/gpucore-%p.core, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-167-g2a12ac4a5e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=./work/diag/gpucore-%p.core, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `merged` env {'IE_DIAG': '1'}; B: `w83` env {'IE_DIAG': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@512000: Prefill tok/s after 512000 tokens of prompt | 1398.3 / 1398.1 | 1408.0 / 1408.2 | 1398.2 (1398.1-1398.3) | 1408.1 (1408.0-1408.2) | +0.71 % [-0.10, +1.52] | open (undecided at the last round) |
| decode@512000: Decode tok/s after 512000 tokens of prompt | 60.6 / 60.6 | 60.5 / 59.8 | 60.6 (60.6-60.6) | 60.1 (59.8-60.5) | -0.80 % [-1.86, +0.26] | open (undecided at the last round) |
| prefill@1000000: Prefill tok/s after 1000000 tokens of prompt | 1099.9 / 1100.1 | 1101.8 / 1101.8 | 1100.0 (1099.9-1100.1) | 1101.8 (1101.8-1101.8) | +0.16 % [-0.65, +0.98] | open (undecided at the last round) |
| decode@1000000: Decode tok/s after 1000000 tokens of prompt | 63.9 / 63.4 | 63.7 / 63.1 | 63.6 (63.4-63.9) | 63.4 (63.1-63.7) | -0.37 % [-1.18, +0.45] | open (undecided at the last round) |

| Round | Order | prefill@512000 B vs A | decode@512000 B vs A | prefill@1000000 B vs A | decode@1000000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.70 % | -0.21 % | +0.17 % | -0.32 % | 2616 |
| 2 | BA | +0.72 % | -1.39 % | +0.15 % | -0.42 % | 2606 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@512000 | +0.71 % | -2 to +1 % | within the expectation |
| decode@512000 | -0.80 % | -2 to +0 % | within the expectation |
| prefill@1000000 | +0.16 % | -2 to +1 % | within the expectation |
| decode@1000000 | -0.37 % | -2 to +0 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 512000 | 65890, 10202 / 65890, 10202 | 65890, 10202 / 65890, 10202 | yes |
| 1000000 | 63776, 15081 / 63776, 15081 | 63776, 15081 / 63776, 15081 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 512000 | f7f5488a6a60d401 | f7f5488a6a60d401 | yes | yes | yes | - |
| 1000000 | f12ab3b48409a583 | f12ab3b48409a583 | yes | yes | yes | - |

Runtime: 87.0 min for 2 rounds (43.5 min per round); a fixed 3 rounds would take 130.6 min (+43.5 min saved).
