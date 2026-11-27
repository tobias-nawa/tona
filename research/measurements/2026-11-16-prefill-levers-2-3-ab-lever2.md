## Plan (written before the run)

- Scope: workload long (130000:200 at --ctx 140000); A `<worktree>/work/build/w159` against B `<worktree>/work/build/w159`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (prefill) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 30K and 130K are the prompt lengths of coding agents; lever 3 acts at every group boundary (1 at 30K, 8 at 130K), lever 2 at every layer chunk (decode tokens per prompt as in the targets, default 200)
- Duration: 6-19 min (2-6 rounds of 3.1 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: research/measurements/2026-11-16-prefill-breakdown.md levers 2 and 3):
  - prefill@130000: +0.5 to +2.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-941-ga7f0f5d18b-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin, TONA_PF_WCACHE=0
  - B: build commit v0.0.2-941-ga7f0f5d18b-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `<worktree>/work/build/w159` env {'TONA_DETERMINISTIC': '1', 'TONA_PF_WCACHE': '0'}; B: `<worktree>/work/build/w159` env {'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 1 of at most 6, stop: all metrics decided after round 1. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1832.8 | 1855.5 | 1832.8 (1832.8-1832.8) | 1855.5 (1855.5-1855.5) | +1.24 % [+1.18, +1.29] | **B faster** after round 1 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 64.1 | 63.6 | 64.1 (64.1-64.1) | 63.6 (63.6-63.6) | -0.66 % [-1.97, +0.64] | open (undecided at the last round) |

| Round | Order | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +1.24 % | -0.66 % (off plan) | 187 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@130000 | +1.24 % | +0.5 to +2.5 % | within the expectation |
| decode@130000 | -0.66 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 130000 | 76007, 7944 | 76007, 7944 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 130000 | 7d2996546076d1b0 | 7d2996546076d1b0 | yes | yes | yes | 7d2996546076d1b0 (B = ref) |

Runtime: 4.7 min for 1 rounds (3.1 min per round); a fixed 3 rounds would take 9.4 min (+6.2 min saved).
