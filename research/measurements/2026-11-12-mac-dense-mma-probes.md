# Mac: dense mat-vecs at 2 to 4 tokens: input loads and simdgroup MMA, probes (#108)

M1 Max 32 GB. Question: the dense mat-vecs of a verify pass cost 2 to 3 times as much at W = 4 as at W = 1 for the
same bytes (2026-11-12-mac-verify-profile.md). Is it the input loads (the PC's finding in #126), and does an 8x8
simdgroup MMA path (maintainer decision 2026-11-12: half precision allowed for W ≥ 4 on Metal after a quality
check) make the cost flat over the width?

**Deviation from the measurement rules:** exploratory probes, a few seconds each, no plan per probe; leads, not
claims, nothing in BENCHMARKS.md. The quality check of a half path was not run: no variant was fast enough.

## Setup

`bench/mac/densek.mm` (scratch copies of `densek.metal` and `decode.metal`; the input buffer widened to 8 tokens so an
MMA tile may read columns past B). All 737 dense mat-vecs of the Coder with real types and shapes, synthetic blocks;
"phased" is the steps of a pass with split-K below 2048 rows, median of 20 command buffers. Matrices below 2048 rows
keep the f32 split-K kernel in every variant, so their growth with B stays in all numbers. Every variant that keeps
the arithmetic passed the tool's check (f32 ≤ 3.2e-6, half 4.2e-4 relative).

Conditions: quiet check ok at 10:57 and 12:13; `tools/hw_report.py` 12:13: load 2.31 / 2.23 / 2.44, other programs
3.6 % of all CPU threads, commit `dcb8710` (clean tree), AC power. Raw:
[2026-11-12-mac-dense-mma-probes.jsonl](2026-11-12-mac-dense-mma-probes.jsonl) (`variant` field).

## Probes

| Variant | B1 (ms) | B4 (ms) |
|---|---|---|
| today's kernel (f32, 4 rows per simdgroup, lanes over groups of 8) | 21.7 | 40.2 |
| same, inputs not loaded (computed) | 20.9 | 35.6 |
| MMA f32, 8 rows per simdgroup, staged through threadgroup memory | 48.9 | 52.7 |
| MMA f32, fragments filled by the lanes (K permuted inside 32-weight chunks) | 44.7 | 48.5 |
| same with 2 / 4 row tiles per simdgroup | 48.3 / 63.9 | 52.3 / 67.6 |
| MMA half fragments, f32 accumulator | 45.2 | 48.8 |
| MMA half, K split over 4 / 8 / 16 simdgroups of a threadgroup | 34.8 / 34.8 / 35.6 | 38.8 / **38.6** / 39.5 |
| K split 8, inputs not loaded | 29.1 | 32.9 |
| K split 8, trivial decode | 23.6 | 27.4 |
| K split 8, a lane decodes 4 consecutive groups (128-weight chunks) | 51.4 | 55.7 |
| K split 8, Q6_K data of each 8-row tile in fragment order (synthetic, no check) | 30.1 | 33.9 |

## Reading

- **Input loads: at most 11 % at B4** (40.2 to 35.6 without any load). Staging the inputs in threadgroup memory was
  slower on 2026-11-11 (`XTG=1`, 25.4 against 21.5 ms at B1). The PC's mechanism exists here but is not the main
  cost.
- **MMA makes the cost flat over B** (every MMA variant grows by ~4 ms from B1 to B4, and that growth is the f32
  kernel of the small matrices), but its level is high: the best variant is 38.6 ms at B4, −4 % against today.
- **The cost inside the MMA kernel is the decode** (11 ms of 34.8), not the matrix units and not the inputs: with a
  trivial decode the kernel is at today's level. In the MMA layout the 32 lanes decode 4 groups each of 8 different
  rows, so a simdgroup touches short runs of 8 rows and loads every block header again; today's kernel reads 32
  consecutive groups of one row. Half or f32 does not matter (45.2 against 44.7).
- Prior art (2026-11-12): MLX #4712 (`qmv_mma`, merged 2026-10-22) and skinny-mlx win 1.5 to 3× at 6 to 16 rows
  with affine 4-bit formats whose decode is a shift and a multiply; skinny-mlx also repacks the weights into
  fragment lane order (at twice the memory). The Coder's dense matrices are Q4_K, Q5_K, Q6_K and BF16 with
  super-block scales.

## Not taken over: reason, research, attempt (principle 8)

Not taken over: every MMA variant, because none beats today's kernel by more than 4 % at B4 and all lose at B1
(where W = 1 stays f32 anyway). Researched 2026-11-12 (window 2026-08 to 2026-11; small-batch quantized mat-vec on
Apple GPUs, speculative verify kernels): MLX #4712, skinny-mlx, sglang-omni #3755, llama.cpp `mul_mv_mma` (local
checkout d9ee007). Attempts: K split over 4 to 16 simdgroups (the gain from 48.8 to 38.6 ms), 1 to 4 row tiles, half
fragments, contiguous 128-weight chunks per lane (slower). Open attempt: the weights of the dense matrices in a file
variant ordered by MMA fragment (as skinny-mlx, offline as TONA_IQ3_XXS_R, no runtime memory), so a lane reads its
fragment's weights contiguously; the probe "trivial decode" (27.4 ms at B4 with the small matrices included) bounds
what it can reach.

## Follow-up: Q6_K in fragment order

Q6_K holds 1.88 of 4.10 GB of the dense bytes (46 %; BF16 36 %, Q5_K 9 %, Q4_K 5 %, IQ4 5 %). Probe: the Q6_K
matrices' bytes of each 8-row tile laid out so a lane reads its 8 weights as one 32-bit low-bit word, one 16-bit
high-bit word and one scale byte (`d` per 32 groups), the other types unchanged. B4 38.6 to 33.9 ms (−4.7 ms), B1
34.8 to 30.1: about the Q6_K share of the 11 ms decode cost (46 % of 11 ms is 5 ms), so the layout removes most of it.
Taken over (coordinator 2026-11-12): a Tona type per K-quant in fragment order, Q6_K first. Constraint for the build:
the file holds one copy, so the W = 1 f32 kernel must read the same layout without getting slower.

W = 1 with the same layout (probe, synthetic, no check): today's R4 f32 kernel with the Q6_K matrices read in
fragment order, B1 21.71 to 19.96 ms, B4 40.09 to 40.00 (same run, rerun of today's kernel as A). An f32 kernel on
the MMA tile structure (8 rows, K split 8) is slower at B1 (26.8 GGUF, 25.7 fragment order), so W = 1 keeps the R4
kernel and only its Q6_K decoder changes.

**For other matrix-unit paths (e.g. WMMA on the reference, #128):** with K-quants in GGUF order the MMA kernel's
cost was the decode in fragment layout (each lane reads short runs of 8 rows and every block header again: 11 of 35
ms), not the matrix units; the same bytes in fragment lane order (one word of low bits, one of high bits, one scale
per lane and 8 weights, `TONA_Q6_K_T`) removed about Q6_K's share of it. Measured on Metal's 8x8 fragment only; the
IQ types were not tried in MMA (their decode is grid lookups, 2026-11-12-mac-expert-decode-probes.md).
