## Plan (written before the run)

- Scope: workload long (130000:200 at --ctx 140000); A `<worktree>/work/build/w7` against B `<worktree>/work/build/w7`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (prefill) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 130K is the long prompt (chunk 5120) with the token counts per expert where the wide tile pays or loses (decode tokens per prompt as in the targets, default 200)
- Duration: 6-19 min (2-6 rounds of 3.2 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: research/measurements/2026-11-16-prefill-lever7-state.md and qbench of 2026-11-16: 4 to 13 % faster at 100 and 200 tokens per segment, 5 to 13 % slower at 160, expert GEMM about a quarter of prefill):
  - prefill@130000: -1 to +1 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-951-g6b38221e1a; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin, TONA_GEMMQ_T128_MIN=0
  - B: build commit v0.0.2-951-g6b38221e1a; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `<worktree>/work/build/w7` env {'TONA_DETERMINISTIC': '1', 'TONA_GEMMQ_T128_MIN': '0'}; B: `<worktree>/work/build/w7` env {'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 1 of at most 6, stop: all metrics decided after round 1. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1856.3 | 1864.6 | 1856.3 (1856.3-1856.3) | 1864.6 (1864.6-1864.6) | +0.45 % [+0.39, +0.51] | **B faster** after round 1 (real, but inside the margin) |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 65.0 | 65.2 | 65.0 (65.0-65.0) | 65.2 (65.2-65.2) | +0.21 % [-1.10, +1.51] | open (undecided at the last round) |

| Round | Order | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +0.45 % | +0.21 % | 191 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@130000 | +0.45 % | -1 to +1 % | within the expectation |
| decode@130000 | +0.21 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 130000 | 76193, 7758 | 76193, 7758 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 130000 | 7d2996546076d1b0 | 7d2996546076d1b0 | yes | yes | yes | 7d2996546076d1b0 (B = ref) |

Runtime: 4.8 min for 1 rounds (3.2 min per round); a fixed 3 rounds would take 9.5 min (+6.4 min saved).
