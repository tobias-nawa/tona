## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `tona125base` against B `tona125`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: set is the production decode with the MTP drafter; B runs with the new defaults (16K list, rank 128, 1,024 rows) (decode tokens per prompt as in the targets, default 200)
- Duration: 5-15 min (2-6 rounds of 2.4 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: r2 run with the 16K list file (125-scorer-r2-16k): code +0.96 %, text +1.00 %; the built-in list holds the same ids in ascending order):
  - decode: +0.2 to +2 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-625-g1f17a936ef; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-629-g6d05ae3709; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `tona125base` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin'}; B: `tona125` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 6, stop: all metrics decided after round 4. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 91.4 / 90.7 / 90.8 / 91.1 | 91.9 / 92.9 / 92.0 / 92.7 | 90.9 (90.7-91.4) | 92.3 (91.9-92.9) | +1.51 % [+0.31, +2.71] | **B faster** after round 4 |
| decode-text: Decode tok/s (bench-set text) | 67.4 / 67.5 / 67.6 / 66.9 | 68.3 / 68.7 / 68.5 / 68.4 | 67.5 (66.9-67.6) | 68.5 (68.3-68.7) | +1.67 % [+0.78, +2.56] | **B faster** after round 2 |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +0.55 % | +1.34 % | 147 |
| 2 | BA | +2.43 % | +1.78 % | 146 |
| 3 | AB | +1.32 % | +1.33 % | 147 |
| 4 | BA | +1.76 % | +2.24 % | 147 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +1.51 % | +0.2 to +2 % | within the expectation |
| decode-text | +1.67 % | +0.2 to +2 % | within the expectation |


Runtime: 11.0 min for 4 rounds (2.4 min per round); a fixed 3 rounds would take 7.3 min (-2.4 min saved).
