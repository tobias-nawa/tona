## Plan (written before the run)

- Scope: workload mid (30000:200 at --ctx 140000); A `<worktree>/work/build/w7` against B `<worktree>/work/build/w7`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (prefill) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 30K is the shortest prompt whose chunks have the token counts per expert where the wide tile pays or loses (decode tokens per prompt as in the targets, default 200)
- Duration: 3-8 min (2-6 rounds of 1.3 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: research/measurements/2026-11-16-prefill-lever7-state.md and qbench of 2026-11-16: 4 to 13 % faster at 100 and 200 tokens per segment, 5 to 13 % slower at 160, expert GEMM about a quarter of prefill):
  - prefill@30000: -1 to +1 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-950-g5ea5e6418f-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin, TONA_GEMMQ_T128_MIN=0
  - B: build commit v0.0.2-950-g5ea5e6418f-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `<worktree>/work/build/w7` env {'TONA_DETERMINISTIC': '1', 'TONA_GEMMQ_T128_MIN': '0'}; B: `<worktree>/work/build/w7` env {'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 1 of at most 6, stop: all metrics decided after round 1. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1933.5 | 1945.5 | 1933.5 (1933.5-1933.5) | 1945.5 (1945.5-1945.5) | +0.62 % [+0.50, +0.75] | **B faster** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 70.5 | 70.5 | 70.5 (70.5-70.5) | 70.5 (70.5-70.5) | +0.08 % [-0.79, +0.94] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +0.62 % | +0.08 % | 77 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +0.62 % | -1 to +1 % | within the expectation |
| decode@30000 | +0.08 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 71275, 8030 | 71275, 8030 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | ee58880d3f5b558f (B = ref) |

Runtime: 1.9 min for 1 rounds (1.3 min per round); a fixed 3 rounds would take 3.8 min (+2.6 min saved).
