## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `tona130` against B `tona130`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: bench-set is the decode of the production path (code and text prompts, MTP widths as in service); a +-1 % effect needs the paired rounds (decode tokens per prompt as in the targets, default 200)
- Duration: 5-10 min (2-4 rounds of 2.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: pass curve of this file (W2..W4 within 0.1 ms, W5..W8 -0.2..-0.6 ms, i.e. 0 to +1 %)):
  - decode: -0.5 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-668-g6a8e859be1-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-668-g6a8e859be1-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin

## Result

A: `tona130` env -; B: `tona130` env -. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 88.4 / 88.7 / 88.7 / 89.0 | 89.7 / 89.3 / 89.6 / 89.4 | 88.7 (88.4-89.0) | 89.5 (89.3-89.7) | +0.90 % [+0.04, +1.76] | **B faster** after round 3 |
| decode-text: Decode tok/s (bench-set text) | 64.0 / 63.9 / 64.4 / 64.5 | 64.6 / 64.4 / 64.7 / 64.6 | 64.2 (63.9-64.5) | 64.6 (64.4-64.7) | +0.59 % [-0.21, +1.38] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +1.47 % | +0.94 % | 150 |
| 2 | BA | +0.68 % | +0.78 % | 150 |
| 3 | AB | +1.01 % | +0.47 % | 150 |
| 4 | BA | +0.45 % | +0.16 % | 150 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +0.90 % | -0.5 to +1 % | within the expectation |
| decode-text | +0.59 % | -0.5 to +1 % | within the expectation |


Runtime: 11.3 min for 4 rounds (2.5 min per round); a fixed 3 rounds would take 7.5 min (-2.5 min saved).
