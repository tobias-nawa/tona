## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `w103b` against B `w103f`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 30K and 130K are the R8 sizes; the pieces change how 0.7 to 1.8 GB of the pinned memory is registered, which prefill and decode read (decode tokens per prompt as in the targets, default 200)
- Duration: 8-23 min (2-6 rounds of 3.9 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: placement change only; E4/E7 idle runs of 2026-11-11-evict.md, noise of mid/long in bench/noise.toml):
  - decode: -1 to +1 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-366-gb8a734d90e; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-366-gb8a734d90e; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `w103b` env -; B: `w103f` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1880.7 / 1881.0 / 1880.8 / 1880.7 / 1881.7 / 1881.3 | 1881.0 / 1881.0 / 1881.9 / 1881.7 / 1881.7 / 1880.2 | 1880.9 (1880.7-1881.7) | 1881.3 (1880.2-1881.9) | +0.01 % [-0.04, +0.07] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 66.8 / 66.8 / 66.8 / 66.8 / 67.0 / 67.0 | 66.9 / 66.9 / 66.9 / 66.8 / 66.9 / 67.0 | 66.8 (66.8-67.0) | 66.9 (66.8-67.0) | +0.04 % [-0.28, +0.35] | **equal** after round 4 |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1755.8 / 1755.3 / 1755.6 / 1753.4 / 1755.4 / 1755.3 | 1755.3 / 1755.5 / 1755.8 / 1753.2 / 1753.4 / 1752.5 | 1755.3 (1753.4-1755.8) | 1754.4 (1752.5-1755.8) | -0.05 % [-0.11, +0.01] | **equal** after round 1 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 68.2 / 68.1 / 68.1 / 68.2 / 68.4 / 68.2 | 68.4 / 68.4 / 68.4 / 68.3 / 68.3 / 68.3 | 68.2 (68.1-68.4) | 68.4 (68.3-68.4) | +0.21 % [-0.27, +0.69] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.01 % | +0.07 % | -0.02 % | +0.28 % | 232 |
| 2 | BA | +0.00 % | +0.22 % | +0.01 % | +0.33 % | 232 |
| 3 | AB | +0.06 % | +0.10 % | +0.01 % | +0.47 % | 233 |
| 4 | BA | +0.05 % | +0.01 % | -0.02 % | +0.27 % | 233 |
| 5 | AB | -0.00 % | -0.13 % | -0.11 % | -0.09 % | 233 |
| 6 | BA | -0.05 % | -0.05 % | -0.16 % | +0.02 % | 233 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +0.01 % | -1 to +1 % | within the expectation |
| decode@30000 | +0.04 % | -1 to +1 % | within the expectation |
| prefill@130000 | -0.05 % | -1 to +1 % | within the expectation |
| decode@130000 | +0.21 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 72075, 6351 / 72075, 6351 / 72075, 6351 / 72075, 6351 / 72075, 6351 / 72075, 6351 | 72075, 6351 / 72075, 6351 / 72075, 6351 / 72075, 6351 / 72075, 6351 / 72075, 6351 | yes |
| 130000 | 66396, 7726 / 66396, 7726 / 66396, 7726 / 66396, 7726 / 66396, 7726 / 66396, 7726 | 66396, 7726 / 66396, 7726 / 66396, 7726 / 66396, 7726 / 66396, 7726 / 66396, 7726 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | 3a5ff5033f8f119b | 3a5ff5033f8f119b | yes | yes | yes | - |
| 130000 | 971ebf40510eb261 | 971ebf40510eb261 | yes | yes | yes | - |

Runtime: 25.2 min for 6 rounds (3.9 min per round); a fixed 3 rounds would take 11.6 min (-11.6 min saved).
