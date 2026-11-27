## Plan (written before the run)

- Scope: workload short,set (2000:400 2000:write:400 at --ctx 140000; bench-set at --ctx 32768); A `tona124` against B `tona124`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: short: SHA identity of the text per request; set: decode tok/s of code and text, the list steers MTP drafts only (decode tokens per prompt as in the targets, default 200)
- Duration: 7-22 min (2-6 rounds of 3.6 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: research/measurements/2026-11-12-mtp-vocab-own.md (old shard numbers do not apply; A is the reference)):
  - decode: -2 to +2 %
  - prefill: -2 to +2 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-540-gbdef57f94f-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin, TONA_MTP_VOCAB=<repo>/work/mtp-vocab-old.txt
  - B: build commit v0.0.2-540-gbdef57f94f-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `tona124` env {'TONA_DETERMINISTIC': '1', 'TONA_MTP_VOCAB': '<repo>/work/mtp-vocab-old.txt'}; B: `tona124` env {'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1367.1 / 1516.3 / 1365.5 / 1359.6 / 1514.1 / 1363.8 | 1453.6 / 1514.6 / 1513.2 / 1515.3 / 1513.9 / 1516.8 | 1366.3 (1359.6-1516.3) | 1514.2 (1453.6-1516.8) | +6.61 % [+0.55, +12.68] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 72.0 / 72.3 / 72.3 / 71.9 / 72.2 / 72.2 | 72.7 / 73.1 / 73.2 / 73.2 / 73.1 / 73.3 | 72.2 (71.9-72.3) | 73.2 (72.7-73.3) | +1.32 % [+0.89, +1.76] | **B faster** after round 2 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1621.2 / 1614.7 / 1616.7 / 1614.2 / 1614.9 / 1620.5 | 1747.3 / 1618.4 / 1617.4 / 1616.2 / 1615.6 / 1618.4 | 1615.8 (1614.2-1621.2) | 1617.9 (1615.6-1747.3) | +1.35 % [-2.13, +4.82] | open (undecided at the last round) |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 94.8 / 95.2 / 95.2 / 94.7 / 95.4 / 95.1 | 94.6 / 94.9 / 95.4 / 95.6 / 95.5 / 95.4 | 95.2 (94.7-95.4) | 95.4 (94.6-95.6) | +0.14 % [-0.63, +0.91] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 85.6 / 85.9 / 86.1 / 85.0 / 86.0 / 85.6 | 85.4 / 85.7 / 85.7 / 85.3 / 85.5 / 85.5 | 85.8 (85.0-86.1) | 85.5 (85.3-85.7) | -0.21 % [-0.82, +0.40] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 69.4 / 69.3 / 69.4 / 69.2 / 69.4 / 69.3 | 68.9 / 68.9 / 68.8 / 68.2 / 68.8 / 68.8 | 69.3 (69.2-69.4) | 68.8 (68.2-68.9) | -0.87 % [-1.46, -0.27] | **B slower** after round 4 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | +6.33 % (off plan) | +0.96 % | +7.77 % (off plan) | -0.23 % | -0.23 % | -0.72 % | 219 |
| 2 | BA | -0.12 % | +1.00 % | +0.23 % | -0.35 % | -0.23 % | -0.58 % | 216 |
| 3 | AB | +10.82 % (outlier) | +1.32 % | +0.04 % | +0.18 % | -0.46 % | -0.86 % | 216 |
| 4 | BA | +11.45 % (outlier) | +1.87 % | +0.12 % | +0.90 % | +0.35 % | -1.45 % | 217 |
| 5 | AB | -0.02 % | +1.32 % | +0.04 % | +0.08 % | -0.58 % | -0.86 % | 216 |
| 6 | BA | +11.22 % (outlier) | +1.48 % | -0.13 % | +0.28 % | -0.12 % | -0.72 % | 217 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +6.61 % | -2 to +2 % | **OUTSIDE the expectation: finding** |
| decode@2000 | +1.32 % | -2 to +2 % | within the expectation |
| prefill@2000:write | +1.35 % | -2 to +2 % | within the expectation |
| decode@2000:write | +0.14 % | -2 to +2 % | within the expectation |
| decode-code | -0.21 % | -2 to +2 % | within the expectation |
| decode-text | -0.87 % | -2 to +2 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 142122, 16434 / 142122, 16434 / 142122, 16434 / 142122, 16434 / 142122, 16434 / 142122, 16434 | 140360, 16432 / 140360, 16432 / 140360, 16432 / 140360, 16432 / 140360, 16432 / 140360, 16432 | NO |
| 2000:write | 117972, 20276 / 117972, 20276 / 117972, 20276 / 117972, 20276 / 117972, 20276 / 117972, 20276 | 117925, 20323 / 117925, 20323 / 117925, 20323 / 117925, 20323 / 117925, 20323 / 117925, 20323 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 11eacf1ee10017b4 | 11eacf1ee10017b4 | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 23.6 min for 6 rounds (3.6 min per round); a fixed 3 rounds would take 10.8 min (-10.8 min saved).
