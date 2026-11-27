# Prefill: row-wise kernels in pieces below 1 ms (#159, lever 5)

## Plan (written before the runs)

- Basis: research/measurements/2026-11-16-prefill-idle-gaps.md. A kernel of 1 ms or longer is followed by a stall of
  about 280 us before the next queued kernel starts (0 % of the kernels below 1 ms, 78 % from 1 ms on; chunk 2048
  removed half of the gaps, 1,139 against 2,188, because the chunk-sized kernels fell below 1 ms). 130K: 9,658 gaps,
  2.73 s; of them about 3,000 (about 0.85 s) follow `k_combine`, `k_combine_norm16`, `k_moe_sum` and `k_norm16`.
- Change (host code only, `engine/gpu/prefill_split.h`, `prefill_hc.hip`, `prefill_moe.hip`): the launches of these four
  kernels are cut into equal pieces of at most `TONA_PF_SPLIT_ROWS` rows (default 2048; 0 = one launch as before). A piece is
  the same kernel on row-shifted pointers, so every output is computed from the same values in the same order; the kernel
  code is unchanged (`isa-diff`: all kernels identical).
- Questions and decisions: (a) tokens identical (`TONA_DETERMINISTIC=1` SHAs of 30K and 130K equal to `bench/sha.toml`,
  goldens identical)? If not, drop. (b) Faster at 30K and 130K? Keep the default 2048 only if both are faster beyond noise;
  if equal or slower the default goes to 0 or the change is dropped. (c) Do the gaps after these kernels vanish (profile of
  the 30K prompt, the share of those kernels followed by a gap)? If the speed does not move but the gaps vanish, the stall
  moved elsewhere: a finding.
- Runs: `scripts/ab.py` same binary (`work/build/w7`), A `TONA_PF_SPLIT_ROWS=0`, B default, workloads mid (30K) and long
  (130K), `--env TONA_DETERMINISTIC=1`, `--primary prefill`, early stopping on (`--max-rounds` from bench/noise.toml: the
  known noise of prefill is 0.02 to 0.04 %, so 1 to 3 rounds decide). `hw_report.py` and `quiet_check.py` before and after
  each `ab.py` run.
- Why this size: the smallest runs that show the effect; 30K has 4.2 % of its window in gaps, 130K 3.8 %.
- Expectation (source: the gap counts above): the four kernels account for about 1.2 % of the 130K window and about the
  same at 30K; minus the cost of two extra launches per kernel (sub-percent). Prefill +0.8 % at 130K, +0.8 % at 30K,
  tolerance +-0.7 points; decode unchanged (+-0.5 %).
- Active tooling: `TONA_DETERMINISTIC=1`, `OMP_NUM_THREADS=6`, `ROCR_VISIBLE_DEVICES=0`,
  `TONA_EXPERT_PROFILE=work/profile/experts-coder.bin` (ab.py); no profiler.

## Results

`ab.py`, same binary (`work/build/w7`), A `TONA_PF_SPLIT_ROWS=0`, B default (2048), `TONA_DETERMINISTIC=1`, early stopping
decided after round 1 in both:

| Prompt | Prefill tok/s A -> B | B vs A | Decode | SHA |
|---|---|---|---|---|
| 30K | 1,942.6 -> 1,963.9 | **+1.10 % [+0.97, +1.22]** | -0.01 % (equal) | ee58880d3f5b558f = bench/sha.toml |
| 130K | 1,866.3 -> 1,885.8 | **+1.04 % [+0.99, +1.10]** | +0.25 % (equal) | 7d2996546076d1b0 = bench/sha.toml |

Both inside the planned +0.1 to +1.5 % band (the point estimate +0.8 was exceeded by 0.2 to 0.3 points: the stall after
these four kernels cost more than the 280 us counted, or the shorter kernels also run better). `isa-diff` against the
build before the change: all kernels identical (286 functions). Goldens (prefill dflash, prefill, server): 38 of 38
identical. `just check` green.

This is also the proof without a profiler that the stall exists in the production path and that it is removed by
keeping kernels below 1 ms. The remaining gaps are after the longer kernels that are not split yet (`k_gemm2` with
1.28 ms, `k_gemm8b<256>`, `k_dn_wave`, `k_qsa_prep`, `k_attn_wmma`); the 30K profile of this build lists them (`sp30`).

## Test conditions

Raw files: `2026-11-16-prefill-split-rows-ab-30k.md/json`, `-ab-130k.md/json`. `hw_report.py` and `quiet_check.py` were run
before and after each `ab.py` run (`/var/tmp/claude/w159/split/`): RX 9060 XT gfx1200, Ryzen 5 9600X, 60.5 GiB RAM with 53.7
to 54.0 GiB available, kernel 7.2.15-200.fc44, hip 7.16.26385, no `TONA_*` in the shell, GPU 0 % busy before the runs, other
programs 0.3 to 3.3 % of the CPU threads, swap in use 2.96 GiB. Quiet check: warning, not blocking: 1.53 to 1.58 GiB of VRAM held
by others (the desktop); before the 30K run also load average 4.48 (it lags a minute) and 1 page swapped in. The A and B arms
ran back to back on the same state.
