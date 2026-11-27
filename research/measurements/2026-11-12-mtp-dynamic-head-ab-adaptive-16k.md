## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `tona125base` against B `tona125`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: set is the production decode with the MTP drafter (docs/contribute/measurement.md); the change touches only the drafter head, so decode of the 14 bench prompts answers it (decode tokens per prompt as in the targets, default 200)
- Duration: 5-15 min (2-6 rounds of 2.6 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: fixed-depth runs F4/G7/P4 of 2026-11-12-mtp-dynamic-head.md: MTP 8.1 -> 7.4 ms of a 66 ms pass = +1.1 %, tokens per pass +0.9 % code, text -0.6 %):
  - decode: +0.5 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-548-gf9b1629719; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-552-gadb0ca3fa0-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin, TONA_MTP_DYN=4096, TONA_MTP_SCORER=128, TONA_MTP_SCORER_ROWS=1024, TONA_MTP_VOCAB=<worktree>/work/mtp-vocab-125-n16k.txt

## Result

A: `tona125base` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin'}; B: `tona125` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin', 'TONA_MTP_SCORER': '128', 'TONA_MTP_SCORER_ROWS': '1024', 'TONA_MTP_DYN': '4096', 'TONA_MTP_VOCAB': '<worktree>/work/mtp-vocab-125-n16k.txt'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 6, stop: all metrics decided after round 4. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 87.3 / 87.2 / 87.0 / 87.3 | 87.8 / 87.7 / 88.1 / 88.1 | 87.2 (87.0-87.3) | 87.9 (87.7-88.1) | +0.83 % [+0.01, +1.65] | **B faster** after round 4 |
| decode-text: Decode tok/s (bench-set text) | 67.3 / 67.1 / 66.9 / 67.2 | 66.2 / 66.3 / 66.7 / 66.7 | 67.2 (66.9-67.3) | 66.5 (66.2-66.7) | -0.97 % [-1.97, +0.03] | **B slower** after round 2 |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +0.57 % | -1.63 % | 154 |
| 2 | BA | +0.57 % | -1.19 % | 153 |
| 3 | AB | +1.26 % | -0.30 % | 153 |
| 4 | BA | +0.92 % | -0.74 % | 153 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +0.83 % | +0.5 to +3 % | within the expectation |
| decode-text | -0.97 % | +0.5 to +3 % | **OUTSIDE the expectation: finding** |


Runtime: 11.5 min for 4 rounds (2.6 min per round); a fixed 3 rounds would take 7.7 min (-2.6 min saved).
