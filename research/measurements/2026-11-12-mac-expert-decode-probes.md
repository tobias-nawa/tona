# Mac: what limits the expert mat-vec decode? Probes and GPU counters (#108)

M1 Max 32 GB. Question: the experts run at ~54 GB/s in the server's verify pass (2026-11-12-mac-verify-profile.md)
and at 96 GB/s at B = 1 in isolation (2026-11-11-mac-expertk.md), against 334 GB/s measured peak. What limits them,
and which kernel change moves it?

**Deviation from the measurement rules:** these were exploratory probes, each a few seconds long, run one after the
other; no plan was written before each one. The numbers below are therefore leads, not claims: none goes into
BENCHMARKS.md, and any kernel change that follows gets its own planned A/B.

## Setup

`bench/mac/expertk.mm` (unchanged in the repository; probes used copies of `decode.metal` and `expertk.metal` in a
scratch directory, the tool compiles Metal at run time). Synthetic weights in the real formats, the Coder's shapes
(gate/up 640 × 2560, down 2560 × 640, 10 experts per layer, 48 layers, pool of 64 experts per matrix); R = 4 rows
per simdgroup. "Token" is the experts of 48 layers with the Coder's type mix in one command buffer, median of 20.
The probe variants that change the arithmetic (no input loads, trivial decode) skip the tool's correctness check;
every other variant passed it (relative error ≤ 1.8e-7).

Conditions: quiet check ok before (10:31:31) and after (10:54:40); `tools/hw_report.py` after: load 2.18 / 3.6 /
3.73, other programs 4.2 % of all CPU threads, commit `8d97c93` (clean tree), AC power. Raw:
[2026-11-12-mac-expert-decode-probes.jsonl](2026-11-12-mac-expert-decode-probes.jsonl) (`variant` field).

## Probes

| Variant | What changes | Token B1 (ms) | Token B4 (ms) | IQ3_XXS gate_up B1 / B4 (µs per layer) |
|---|---|---|---|---|
| base | today's kernel | 10.0 | 20.6 | 148 / 285 |
| nox | inputs not loaded (computed) | 9.3 | 17.3 | 143 / 245 |
| nodec | trivial decode, same bytes read | 5.5 | 15.4 | 71 / 198 |
| u16 | 32-bit sign word from two 16-bit loads | 10.0 | 20.7 | 144 / 284 |
| un2 | group loop unrolled by 2 | 11.3 | 70.5 | 159 / 996 (registers spill) |
| R2 / R8 | 2 or 8 rows per simdgroup | 10.6 / 11.7 | 21.0 / 26.8 | |
| sb | IQ3_XXS: 8 lanes per row, header once per 32 weights | 9.8 | 21.5 | 138 / 342 |
| xs | IQ3_XXS: sign by XOR on the sign bit, vector conversion | 10.1 | 20.6 | 152 / 283 |
| pk | IQ3_XXS repacked per row: index pairs as aligned ushort, sign indices, scales as half in own arrays (speed only) | 9.7 | 20.3 | 126 / 266 |

The threadgroup copy of the IQ grids (instead of constant memory) was tried in the engine itself: DM pass W1 44.6
against 44.3 ms, W4 91.6 against 92.3 (`tona-bench`, 2 rounds each), no gain.

## GPU counters

`xctrace record --instrument "Metal GPU Counters"` on IQ3_XXS gate_up alone, 2 s at B1, 1.5 s idle, 2 s at B4
(counter averages over 100 ms; steady over the 2 s):

| | ALU limiter | ALU utilization | F32 utilization | Buffer read limiter | Last-level cache limiter | Compute occupancy |
|---|---|---|---|---|---|---|
| B1 | 82 % | 60 % | 15 % | 62 % | 21 % | 27 % |
| B4 | 74 % | 51 % | 19 % | 58 % | 35 % | 15 % |

## Reading

- The kernel is ALU-limited, and the floating-point part is small (15 % F32 of 60 % ALU). The weight arithmetic
  itself is not the cost: the cheaper sign and conversion (xs) changed nothing, and decoding the header once per 32
  weights (sb) saved 7 % at B1.
- A trivial decode reading the same bytes halves B1 (nodec). What remains of the decode beyond the arithmetic is
  addressing: GGUF IQ blocks of 98, 82 and 110 bytes, 2-byte aligned, from which every lane picks single bytes
  (scale, sign word, two or more index bytes) per 8 weights.
- B4 halves the occupancy (27 to 15 %): the `acc[R][B]` and input registers. Fewer rows per simdgroup (R2) did not
  help, because every input load then serves fewer rows.
- Repacking tested on IQ3_XXS (pk, synthetic, no correctness check): −15 % at B1, −7 % at B4 for that type, −3 % on
  the whole token (IQ3_XXS is 17 of 48 layers). The addressing is part of the cost, not most of it; what remains
  beyond nodec is the grid lookups and the per-weight work. GGUF stays mapped without a copy today, so a repack is
  either a copy at load time (memory) or a Mac-side file variant, for ~3 to 8 % of the expert time: not worth that
  decision now.
- In the real verify pass the routed experts are 14.8 ms at W = 1 (66 GB/s on 0.98 GB) and 52.1 ms at W = 4 before
  the grouping (2026-11-12-mac-verify-profile.md); a large part of the growth with W is the distinct experts four
  tokens select, i.e. bytes, not the kernel.

## Not taken over: reasons, research, attempts (docs/contribute/measurement.md principle 8)

Researched 2026-11-12 (window 2026-08 to 2026-11; topics: Metal IQ mat-vec kernels, small-batch quantized mat-vec
on Apple GPUs, speculative verify kernels): llama.cpp #28086 and #28692 (row split in the IQ mul_mv kernels for
K < 1024; our K is 2560 and 640, every lane already busy), llama.cpp `mul_mv_mma` (local checkout d9ee007, half
fragments from 2 to 3 rows), MLX #4712 (`qmv_mma`, 8x8 simdgroup MMA from 6 rows, merged 2026-10-22), skinny-mlx
(bf16 MMA from 6 rows, weights in fragment order), sglang-omni #3755 (multi-row kernel for the verify pass). None
addresses the IQ decode cost at 1 to 4 rows; the MMA paths are for dense types and change numbers with the width.

| Probe | Reason | Attempt to remove the cost |
|---|---|---|
| threadgroup grids | no gain (44.6 / 44.3 ms) | none needed: constant memory is not the limit (counters: buffer reads, not constant) |
| 16-bit sign word, XOR sign | no gain | taken further as the repacked layout (TONA_IQ3_XXS_R, 2026-11-12-mac-iq3xxs-repack.md) |
| unroll by 2 | registers spill (B4 3.4× slower) | unrolling only at B1 would change the summation order between widths (check 2 of the backend) |
| R2 / R8 | slower at every B | R by width is allowed (same order per row) but no width wins with other than R4 |
| sub-block layout (B1 −7 %, B4 +20 %) | B4 loses more than B1 gains | it is a kernel layout, not a data layout: a width-dependent choice costs no memory, but sums each row in another order than R4, so widths would no longer be bit-identical (check 2). One layout for both: open (after the A/B: sub-block with the inputs of a group shared by the 4 rows' lanes) |
| repacked IQ3_XXS | −3 % per token in the probe | done offline as a file variant: TONA_IQ3_XXS_R |
