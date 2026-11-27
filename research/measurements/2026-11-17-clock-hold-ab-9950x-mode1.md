## Plan (written before the run)

- Scope: workload short (2000:400 2000:write:400 at --ctx 140000); A `/var/tmp/claude/161/build` against B `/var/tmp/claude/161/build`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: prefill@2000 only: the effect to exclude is the 7 to 9 % queue effect of 2026-11-12 (A/A spread 0.2 %), so 1 to 2 rounds decide; 4 at most; decode must stay equal (decode tokens per prompt as in the targets, default 200)
- Duration: 2-4 min (2-4 rounds of 1.1 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: A/A of 2026-11-12-prefill-bimodal.md (prefill 0.2 %, decode 0.1 to 0.5 %)):
  - prefill: -0.5 to +0.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-954-g0602d8be73; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=/var/tmp/claude/161/wt/work/profile/experts-coder.bin
  - B: build commit v0.0.2-954-g0602d8be73; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_CLOCK_HOLD=1, TONA_EXPERT_PROFILE=/var/tmp/claude/161/wt/work/profile/experts-coder.bin

## Result

A: `/var/tmp/claude/161/build` env -; B: `/var/tmp/claude/161/build` env {'TONA_CLOCK_HOLD': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 4, stop: error: server exited with 1 before it was ready (partial result). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1459.1 / 1461.1 | 1462.1 / 1454.9 | 1460.1 (1459.1-1461.1) | 1458.5 (1454.9-1462.1) | -0.11 % [-0.68, +0.46] | **equal** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 76.6 / 76.5 | 76.1 / 76.4 | 76.6 (76.5-76.6) | 76.2 (76.1-76.4) | -0.42 % [-1.20, +0.36] | open (undecided at the last round) |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1744.5 / 1744.4 | 1743.3 / 1732.7 | 1744.4 (1744.4-1744.5) | 1738.0 (1732.7-1743.3) | -0.37 % [-1.01, +0.27] | open (undecided at the last round) |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 126.0 / 126.9 | 126.3 / 126.3 | 126.5 (126.0-126.9) | 126.3 (126.3-126.3) | -0.10 % [-1.63, +1.43] | open (undecided at the last round) |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.20 % | -0.75 % (off plan) | -0.07 % | +0.20 % | 65 |
| 2 | BA | -0.43 % | -0.09 % | -0.67 % | -0.41 % | 65 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | -0.11 % | -0.5 to +0.5 % | within the expectation |
| decode@2000 | -0.42 % | -0.5 to +0.5 % | within the expectation |
| prefill@2000:write | -0.37 % | -0.5 to +0.5 % | within the expectation |
| decode@2000:write | -0.10 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 133160, 19689 / 133160, 19689 | 133160, 19689 / 133160, 19689 | yes |
| 2000:write | 86604, 23673 / 86604, 23673 | 86604, 23673 / 86604, 23673 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 3.0 min for 2 rounds (1.1 min per round); a fixed 3 rounds would take 3.2 min (+1.1 min saved).
