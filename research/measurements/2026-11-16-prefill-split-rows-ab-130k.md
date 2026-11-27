## Plan (written before the run)

- Scope: workload long (130000:200 at --ctx 140000); A `<worktree>/work/build/w7` against B `<worktree>/work/build/w7`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (prefill) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: the shortest runs that show the effect: 130000 tokens, gaps are 4 % of the window (decode tokens per prompt as in the targets, default 200)
- Duration: 6-19 min (2-6 rounds of 3.1 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: research/measurements/2026-11-16-prefill-idle-gaps.md: the four split kernels precede about 3,000 of the 9,658 gaps of 280 us at 130K (0.85 s of 72 s)):
  - prefill@130000: +0.1 to +1.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-956-gf8aadec8c9-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin, TONA_PF_SPLIT_ROWS=0
  - B: build commit v0.0.2-956-gf8aadec8c9-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `<worktree>/work/build/w7` env {'TONA_DETERMINISTIC': '1', 'TONA_PF_SPLIT_ROWS': '0'}; B: `<worktree>/work/build/w7` env {'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 1 of at most 6, stop: all metrics decided after round 1. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1866.3 | 1885.8 | 1866.3 (1866.3-1866.3) | 1885.8 (1885.8-1885.8) | +1.04 % [+0.99, +1.10] | **B faster** after round 1 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 64.7 | 64.8 | 64.7 (64.7-64.7) | 64.8 (64.8-64.8) | +0.25 % [-1.06, +1.55] | open (undecided at the last round) |

| Round | Order | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +1.04 % | +0.25 % | 189 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@130000 | +1.04 % | +0.1 to +1.5 % | within the expectation |
| decode@130000 | +0.25 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 130000 | 75832, 8119 | 75832, 8119 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 130000 | 7d2996546076d1b0 | 7d2996546076d1b0 | yes | yes | yes | 7d2996546076d1b0 (B = ref) |

Runtime: 4.7 min for 1 rounds (3.1 min per round); a fixed 3 rounds would take 9.4 min (+6.3 min saved).
