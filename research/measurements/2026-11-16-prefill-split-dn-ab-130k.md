## Plan (written before the run)

- Scope: workload long (130000:200 at --ctx 140000); A `<worktree>/work/build/w7a` against B `<worktree>/work/build/w7`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (prefill) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: the shortest runs that show the effect: 130000 tokens, gaps are 4 % of the window (decode tokens per prompt as in the targets, default 200)
- Duration: 6-19 min (2-6 rounds of 3.2 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: research/measurements/2026-11-16-prefill-split-dn.md: gaps after k_dn_conv_l2 are 51 ms of 15.5 s at 30K):
  - prefill@130000: +0 to +0.6 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-961-gbef521faa4-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-961-gbef521faa4-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `<worktree>/work/build/w7a` env {'TONA_DETERMINISTIC': '1'}; B: `<worktree>/work/build/w7` env {'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 1 of at most 6, stop: all metrics decided after round 1. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1799.4 | 1806.8 | 1799.4 (1799.4-1799.4) | 1806.8 (1806.8-1806.8) | +0.41 % [+0.36, +0.47] | **B faster** after round 1 (real, but inside the margin) |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 64.1 | 63.9 | 64.1 (64.1-64.1) | 63.9 (63.9-63.9) | -0.39 % [-1.70, +0.91] | open (undecided at the last round) |

| Round | Order | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +0.41 % | -0.39 % | 191 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@130000 | +0.41 % | +0 to +0.6 % | within the expectation |
| decode@130000 | -0.39 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 130000 | 74113, 9838 | 73884, 10067 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 130000 | 7d2996546076d1b0 | 7d2996546076d1b0 | yes | yes | yes | 7d2996546076d1b0 (B = ref) |

Runtime: 4.8 min for 1 rounds (3.2 min per round); a fixed 3 rounds would take 9.5 min (+6.4 min saved).
