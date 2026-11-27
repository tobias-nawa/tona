## Plan (written before the run)

- Scope: workload short,set (2000:400 2000:write:400 at --ctx 140000; bench-set at --ctx 32768); A `w163b7` against B `w163p`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: decode only: short (2K x 400, noise 0.23 percent) and set; effect 0.3 to 1.5 percent needs 4 to 6 rounds; prefill reported, expected unchanged (no k_mega) (decode tokens per prompt as in the targets, default 200)
- Duration: 7-20 min (2-6 rounds of 3.4 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: research/measurements/2026-11-18-b7-step-prefetch.md plan: 1000 steps x 0.1 to 0.6 us of scalar-load wait):
  - decode: +0.3 to +1.5 %
  - prefill: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1069-g1101e80a54; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-1069-g1101e80a54; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `w163b7` env -; B: `w163p` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1457.3 / 1457.6 / 1462.6 / 1460.5 / 1460.9 / 1456.9 | 1462.2 / 1463.1 / 1459.8 / 1463.3 / 1456.6 / 1457.5 | 1459.1 (1456.9-1462.6) | 1461.0 (1456.6-1463.3) | +0.08 % [-0.24, +0.39] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 75.8 / 75.6 / 75.7 / 75.7 / 75.6 / 75.8 | 75.1 / 74.9 / 74.9 / 74.9 / 74.9 / 75.0 | 75.7 (75.6-75.8) | 74.9 (74.9-75.1) | -1.02 % [-1.26, -0.79] | **B slower** after round 2 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1739.3 / 1740.2 / 1739.5 / 1737.9 / 1739.9 / 1739.1 | 1739.2 / 1739.2 / 1739.1 / 1741.2 / 1738.6 / 1737.6 | 1739.4 (1737.9-1740.2) | 1739.1 (1737.6-1741.2) | -0.01 % [-0.21, +0.19] | **equal** after round 2 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 124.1 / 123.3 / 124.2 / 124.1 / 124.1 / 123.9 | 122.6 / 121.9 / 122.2 / 121.8 / 122.3 / 122.4 | 124.1 (123.3-124.2) | 122.3 (121.8-122.6) | -1.40 % [-2.07, -0.73] | **B slower** after round 3 |
| decode-code: Decode tok/s (bench-set code) | 94.3 / 94.3 / 94.4 / 94.4 / 94.3 / 94.3 | 93.0 / 93.0 / 93.0 / 93.0 / 93.0 / 93.1 | 94.3 (94.3-94.4) | 93.0 (93.0-93.1) | -1.40 % [-1.90, -0.90] | **B slower** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 68.2 / 68.2 / 68.2 / 68.2 / 68.2 / 68.2 | 67.6 / 67.5 / 67.5 / 67.6 / 67.6 / 67.5 | 68.2 (68.2-68.2) | 67.5 (67.5-67.6) | -0.95 % [-1.45, -0.45] | **B slower** after round 3 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | +0.34 % | -0.94 % (off plan) | -0.01 % | -1.21 % (off plan) | -1.38 % | -0.88 % | 202 |
| 2 | BA | +0.37 % | -1.02 % | -0.06 % | -1.12 % | -1.38 % | -1.03 % | 202 |
| 3 | AB | -0.19 % | -1.05 % | -0.02 % | -1.56 % | -1.48 % | -1.03 % | 202 |
| 4 | BA | +0.19 % | -1.07 % | +0.19 % | -1.84 % | -1.48 % | -0.88 % | 204 |
| 5 | AB | -0.29 % | -1.01 % | -0.07 % | -1.41 % | -1.38 % | -0.88 % | 202 |
| 6 | BA | +0.05 % | -1.05 % | -0.09 % | -1.26 % | -1.27 % | -1.03 % | 202 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.08 % | -0.5 to +0.5 % | within the expectation |
| decode@2000 | -1.02 % | +0.3 to +1.5 % | **OUTSIDE the expectation: finding** |
| prefill@2000:write | -0.01 % | -0.5 to +0.5 % | within the expectation |
| decode@2000:write | -1.40 % | +0.3 to +1.5 % | **OUTSIDE the expectation: finding** |
| decode-code | -1.40 % | +0.3 to +1.5 % | **OUTSIDE the expectation: finding** |
| decode-text | -0.95 % | +0.3 to +1.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 128514, 24335 / 128514, 24335 / 128514, 24335 / 128514, 24335 / 128514, 24335 / 128514, 24335 | 128514, 24335 / 128514, 24335 / 128514, 24335 / 128514, 24335 / 128514, 24335 / 128514, 24335 | yes |
| 2000:write | 81757, 28520 / 81757, 28520 / 81757, 28520 / 81757, 28520 / 81757, 28520 / 81757, 28520 | 81757, 28520 / 81757, 28520 / 81757, 28520 / 81757, 28520 / 81757, 28520 / 81757, 28520 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 22.0 min for 6 rounds (3.4 min per round); a fixed 3 rounds would take 10.1 min (-10.1 min saved).
