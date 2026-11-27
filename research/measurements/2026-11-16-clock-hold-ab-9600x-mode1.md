## Plan (written before the run)

- Scope: workload short (2000:400 2000:write:400 at --ctx 140000); A `/var/tmp/claude/161/build` against B `/var/tmp/claude/161/build`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: prefill@2000 only: the effect to exclude is the 7 to 9 % queue effect of 2026-11-12 (A/A spread 0.2 %), so 1 to 2 rounds decide; 4 at most; decode must stay equal (decode tokens per prompt as in the targets, default 200)
- Duration: 2-5 min (2-4 rounds of 1.2 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: A/A of 2026-11-12-prefill-bimodal.md (prefill 0.2 %, decode 0.1 to 0.5 %)):
  - prefill: -0.5 to +0.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-952-gdc11332e8f; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=/var/tmp/claude/161/wt/work/profile/experts-coder.bin
  - B: build commit v0.0.2-952-gdc11332e8f; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_CLOCK_HOLD=1, TONA_EXPERT_PROFILE=/var/tmp/claude/161/wt/work/profile/experts-coder.bin

## Result

A: `/var/tmp/claude/161/build` env -; B: `/var/tmp/claude/161/build` env {'TONA_CLOCK_HOLD': '1'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1429.5 / 1280.1 / 1386.2 / 1377.0 | 1348.7 / 1176.7 / 1443.3 / 1401.4 | 1381.6 (1280.1-1429.5) | 1375.0 (1176.7-1443.3) | -1.96 % [-8.85, +4.93] | open (undecided at the last round) |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 70.5 / 75.1 / 75.1 / 75.3 | 72.7 / 73.9 / 74.9 / 75.3 | 75.1 (70.5-75.3) | 74.4 (72.7-75.3) | +0.30 % [-2.14, +2.74] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1707.8 / 1705.4 / 1701.8 / 1674.9 | 1696.6 / 1743.0 / 1730.1 / 1685.6 | 1703.6 (1674.9-1707.8) | 1713.3 (1685.6-1743.0) | +0.96 % [-0.54, +2.47] | open (undecided at the last round) |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 118.3 / 124.1 / 124.2 / 124.3 | 117.8 / 124.4 / 123.2 / 124.2 | 124.2 (118.3-124.3) | 123.7 (117.8-124.4) | -0.28 % [-1.27, +0.71] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -5.65 % (off plan) | +3.16 % (off plan) | -0.66 % (off plan) | -0.47 % | 85 |
| 2 | BA | -8.07 % (outlier) | -1.69 % (outlier) | +2.20 % (outlier) | +0.21 % | 67 |
| 3 | AB | +4.12 % (outlier) | -0.35 % | +1.66 % (outlier) | -0.82 % | 64 |
| 4 | BA | +1.77 % (outlier) | +0.09 % | +0.64 % (outlier) | -0.03 % | 62 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | -1.96 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode@2000 | +0.30 % | -0.5 to +0.5 % | within the expectation |
| prefill@2000:write | +0.96 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode@2000:write | -0.28 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 123067, 29782 / 132355, 20494 / 132355, 20494 / 132355, 20494 | 123067, 29782 / 132355, 20494 / 132355, 20494 / 132355, 20494 | NO |
| 2000:write | 76698, 33579 / 85766, 24511 / 85766, 24511 / 85766, 24511 | 76698, 33579 / 85766, 24511 / 85766, 24511 / 85766, 24511 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 5.5 min for 4 rounds (1.2 min per round); a fixed 3 rounds would take 3.5 min (-1.2 min saved).
