## Plan (written before the run)

- Scope: workload short,set (2000:400 2000:write:400 at --ctx 140000; bench-set at --ctx 32768); A `w163b7` against B `w163q`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: decode only; variant 2 of the prefetch after variant 1 lost 1 to 1.4 percent; the loss is large against the 0.23 percent noise, so 3 to 4 rounds decide whether it is gone (decode tokens per prompt as in the targets, default 200)
- Duration: 7-14 min (2-4 rounds of 3.5 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: research/measurements/2026-11-18-b7-step-prefetch.md plan; hot-variant spills +4 (NB=8), 0 (NB=1), -2 (NB=4) against +9..+14 in variant 1):
  - decode: -0.5 to +1.5 %
  - prefill: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1071-g0b5cbc92b5; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-1071-g0b5cbc92b5; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `w163b7` env -; B: `w163q` env -. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1450.0 / 1455.6 / 1457.6 / 1460.1 | 1457.1 / 1460.9 / 1450.2 / 1457.0 | 1456.6 (1450.0-1460.1) | 1457.1 (1450.2-1460.9) | +0.03 % [-0.54, +0.61] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 70.8 / 74.6 / 75.1 / 75.1 | 74.8 / 73.9 / 73.3 / 74.3 | 74.9 (70.8-75.1) | 74.1 (73.3-74.8) | +0.27 % [-4.04, +4.58] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1731.9 / 1735.9 / 1737.0 / 1737.6 | 1728.2 / 1736.0 / 1734.7 / 1737.0 | 1736.5 (1731.9-1737.6) | 1735.4 (1728.2-1737.0) | -0.09 % [-0.35, +0.16] | **equal** after round 2 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 123.4 / 123.3 / 123.7 / 123.6 | 118.9 / 122.0 / 121.9 / 121.8 | 123.5 (123.3-123.7) | 121.8 (118.9-122.0) | -1.92 % [-3.50, -0.33] | **B slower** after round 3 |
| decode-code: Decode tok/s (bench-set code) | 93.0 / 92.6 / 92.5 / 92.4 | 91.6 / 91.4 / 90.9 / 91.1 | 92.5 (92.4-93.0) | 91.2 (90.9-91.6) | -1.48 % [-2.20, -0.77] | **B slower** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 67.4 / 67.2 / 67.1 / 67.2 | 66.8 / 66.6 / 66.5 / 66.5 | 67.2 (67.1-67.4) | 66.5 (66.5-66.8) | -0.93 % [-1.62, -0.24] | **B slower** after round 3 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | +0.49 % | +5.64 % (off plan) | -0.22 % | -3.60 % (off plan) | -1.51 % | -0.89 % | 210 |
| 2 | BA | +0.36 % | -1.05 % (outlier) | +0.01 % | -1.05 % (outlier) | -1.30 % | -0.89 % | 210 |
| 3 | AB | -0.51 % (outlier) | -2.41 % (outlier) | -0.13 % | -1.52 % | -1.73 % | -0.89 % | 210 |
| 4 | BA | -0.21 % | -1.09 % | -0.03 % | -1.49 % | -1.41 % | -1.04 % | 210 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.03 % | -0.5 to +0.5 % | within the expectation |
| decode@2000 | +0.27 % | -0.5 to +1.5 % | within the expectation |
| prefill@2000:write | -0.09 % | -0.5 to +0.5 % | within the expectation |
| decode@2000:write | -1.92 % | -0.5 to +1.5 % | **OUTSIDE the expectation: finding** |
| decode-code | -1.48 % | -0.5 to +1.5 % | **OUTSIDE the expectation: finding** |
| decode-text | -0.93 % | -0.5 to +1.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 128026, 24823 / 128026, 24823 / 128026, 24823 / 128026, 24823 | 128026, 24823 / 128026, 24823 / 128026, 24823 / 128026, 24823 | yes |
| 2000:write | 81310, 28967 / 81310, 28967 / 81310, 28967 / 81310, 28967 | 81310, 28967 / 81310, 28967 / 81310, 28967 / 81310, 28967 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 15.8 min for 4 rounds (3.5 min per round); a fixed 3 rounds would take 10.5 min (-3.5 min saved).
