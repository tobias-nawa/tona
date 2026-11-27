## Plan (written before the run)

- Scope: workload short (2000:400 2000:write:400 at --ctx 140000); A `/var/tmp/claude/161/build` against B `/var/tmp/claude/161/build`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: prefill@2000 only: the effect to exclude is the 7 to 9 % queue effect of 2026-11-12 (A/A spread 0.2 %), so 1 to 2 rounds decide; 4 at most; decode must stay equal (decode tokens per prompt as in the targets, default 200)
- Duration: 2-4 min (2-4 rounds of 1.0 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: A/A of 2026-11-12-prefill-bimodal.md (prefill 0.2 %, decode 0.1 to 0.5 %)):
  - prefill: -0.5 to +0.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-952-gdc11332e8f; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=/var/tmp/claude/161/wt/work/profile/experts-coder.bin
  - B: build commit v0.0.2-952-gdc11332e8f; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_CLOCK_HOLD=2, TONA_EXPERT_PROFILE=/var/tmp/claude/161/wt/work/profile/experts-coder.bin

## Result

A: `/var/tmp/claude/161/build` env -; B: `/var/tmp/claude/161/build` env {'TONA_CLOCK_HOLD': '2'}. Alpha 0.05, margin 0.5 %, rounds 3 of at most 4, stop: error: server exited with 1 before it was ready (partial result). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1416.1 / 1364.0 / 1414.3 | 1429.9 / 1405.2 / 1439.4 | 1414.3 (1364.0-1416.1) | 1429.9 (1405.2-1439.4) | +1.92 % [+0.65, +3.20] | **B faster** after round 3 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 75.1 / 75.2 / 75.3 | 74.3 / 75.0 / 75.0 | 75.2 (75.1-75.3) | 75.0 (74.3-75.0) | -0.60 % [-1.30, +0.10] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1725.5 / 1676.1 / 1739.0 | 1708.7 / 1683.0 / 1742.4 | 1725.5 (1676.1-1739.0) | 1708.7 (1683.0-1742.4) | -0.12 % [-1.08, +0.83] | open (undecided at the last round) |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 124.3 / 124.7 / 124.2 | 123.4 / 123.7 / 124.1 | 124.3 (124.2-124.7) | 123.7 (123.4-124.1) | -0.53 % [-1.71, +0.64] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.97 % (off plan) | -1.12 % (off plan) | -0.98 % (off plan) | -0.73 % (off plan) | 62 |
| 2 | BA | +3.03 % (outlier) | -0.20 % | +0.41 % | -0.83 % | 62 |
| 3 | AB | +1.77 % | -0.48 % | +0.19 % | -0.04 % | 62 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +1.92 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode@2000 | -0.60 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| prefill@2000:write | -0.12 % | -0.5 to +0.5 % | within the expectation |
| decode@2000:write | -0.53 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 132355, 20494 / 132355, 20494 / 132355, 20494 | 132355, 20494 / 132355, 20494 / 132355, 20494 | yes |
| 2000:write | 85766, 24511 / 85766, 24511 / 85766, 24511 | 85766, 24511 / 85766, 24511 / 85766, 24511 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 3.8 min for 3 rounds (1.0 min per round); a fixed 3 rounds would take 3.1 min (+0.0 min saved).
