# Prefill: dense GEMM launches below 1 ms (#159, lever 5, part 2)

## Plan (written before the runs)

- Basis: research/measurements/2026-11-16-prefill-split-rows.md (the row-wise kernels in pieces gave +1.1 % at 30K and
  +1.04 % at 130K) and the 30K profile of that build (`sp30`): 1,703 gaps of 200 to 500 us (488 ms, down from 2,320 and
  654 ms), now after `k_gemm2` 561 (167 ms), `k_gemm8b<256>` 428 (120 ms), `k_dn_wave` 249 (70 ms), `k_dn_conv_l2` 183
  (51 ms), `k_attn_wmma` 95, `k_qsa_prep` 74. This step takes the two GEMM families: 288 ms of 15.5 s (1.9 %) at 30K.
- Change (host code only, `engine/gpu/gemm_split.h`, `gemm_f16.hip`, `gemm_fp8.hip`): `gemm_f16`, `gemm_f16_mix` and
  `gemm_fp8` cut a segment into pieces of token rows (multiples of the 128-row tile) so that one launch has at most
  `TONA_PF_SPLIT_GMAC` x 10^9 multiply-adds (default 28; the fp8 launches twice that; 0 = as before), and a launch that
  packs several segments stops at the same budget. 28 GMAC is 0.84 ms at 33.5 TMAC/s (67.1 TFLOPS of the 2560 -> 10240
  GEMM, research/measurements/2026-10-23-gemm-wmma.md); the fp8 rate of twice that is an assumption, to be read from the
  next profile (a fp8 kernel still above 1 ms or far below 0.5 ms moves the factor). Pieces start at row multiples of 128,
  every output tile keeps its K loop: identical results; the kernel code is unchanged.
- Questions and decisions: (a) tokens identical (`TONA_DETERMINISTIC=1` SHAs equal to `bench/sha.toml`, goldens identical)?
  (b) Faster at 30K and 130K? Keep the default only if both are faster beyond noise. (c) The cost: each piece re-reads the
  weights of its GEMM (one more read per piece of up to a few tens of MB from VRAM/L2); the gain must beat that.
- Runs: `scripts/ab.py` same binary (`work/build/w7`), A `TONA_PF_SPLIT_GMAC=0`, B default, mid (30K) and long (130K),
  `TONA_DETERMINISTIC=1`, `--primary prefill`, early stopping on; `hw_report.py` and `quiet_check.py` before and after each.
- Why this size: the smallest runs that show the effect (30K: gaps 3.1 % of the window after this GEMM family; 130K as the
  check on the long plan).
- Expectation (source: the `sp30` profile): gaps after the two GEMM families are 288 ms of 15.5 s = 1.9 % at 30K, minus the
  weight re-read and extra launches (assume a third of it): prefill +1.2 % at 30K and +1.2 % at 130K, tolerance +-0.8 points;
  decode +-0.5 %.
- Active tooling: `TONA_DETERMINISTIC=1`, `OMP_NUM_THREADS=6`, `ROCR_VISIBLE_DEVICES=0`,
  `TONA_EXPERT_PROFILE=work/profile/experts-coder.bin`; no profiler.

## Run 1 result (default 28 GMAC against 0): equal, and why (profile `sg30`, 30K)

- `ab.py`: 30K prefill 1,953.3 -> 1,952.2 tok/s, **-0.06 % [-0.14, +0.03]** (2 rounds, equal); 130K 1,886.0 -> 1,887.8,
  **+0.07 % [-0.01, +0.15]** (5 rounds, equal). SHAs equal to bench/sha.toml. Decode equal. Outside the planned
  +0.4 to +2 %: a finding.
- Profile of the same build: the pieces are not below 1 ms. `k_gemm8b<256>`: 672 launches (p50 1,965 us) -> 1,344 launches
  (p50 1,000 us, 680 of them at 1,000 us or longer); `k_gemm2` (the f16 pieces and the mix GEMM): p90 1,024 us, 715 of 4,300
  at 1,000 us or longer; the gaps after them: `k_gemm2` 561 -> 564, `k_gemm8b<256>` 428 -> 235. So the budget was set
  at the edge: the f16 rate is about 27 TMAC/s (28 GMAC = 1.04 ms), not the 33.5 assumed from the 2560 -> 10240 benchmark, and
  the fp8 rate is twice the f16 one as assumed (56 GMAC = 1.0 ms). Window 15.50 -> 15.47 s, gaps 488 -> 438 ms.
- Run 2 (same binary, no rebuild, the budget is an environment variable): A `TONA_PF_SPLIT_GMAC=0`, B 20 (f16 0.74 ms,
  fp8 40 GMAC 0.71 ms), 30K first. Expectation unchanged from the plan (+1.2 %, tolerance 0.8) because the pieces now fall
  below 1 ms; if the gaps after `k_gemm2` and `k_gemm8b<256>` vanish and the speed still does not move, the cost of the
  pieces (weights read again, tile tails) cancels the gain and the change is dropped.

## Run 2 result (20 GMAC against 0, 30K): slower, change dropped

- `ab.py` 30K, 4 rounds: prefill 1,937.5 -> 1,922.9 tok/s, **-0.93 % [-1.24, -0.62]** (B slower); SHA equal to bench/sha.toml.
  Outside the plan (+1.2 %): a finding.
- With the pieces below 1 ms (the stall after the GEMMs removed as far as the budget reaches) the cost of the pieces is larger
  than what the stall costs: each piece re-reads the weights of its GEMM and ends in a partial tile wave, and the dense
  GEMMs run at 67 TFLOPS only with full waves. The row-wise kernels, which are memory-bound and stream each row once, do
  not pay this (+1.1 %).
- Decision: the dense GEMM split is removed (code reverted, never committed: `gemm_split.h`, `gemm_f16.hip`,
  `gemm_fp8.hip`). The gaps after the GEMMs, `k_dn_wave`, `k_dn_conv_l2`, `k_attn_wmma` and `k_qsa_prep` stay; together they
  are 438 ms of 15.5 s (2.8 %) at 30K. What could still remove the stall without paying for pieces: keeping the GPU busy
  during it (a short independent kernel queued behind the long kernel, for example the shared-expert chain or the next
  layer's router, which are already in the stream and are the next candidates), or the cause of the stall itself (the
  scheduler quantum; no knob found). Not pursued now.
