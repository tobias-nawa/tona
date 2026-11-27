# Ornith int8 prefill GEMM route (#175): prior art, micro test

Date: 2026-11-20. Base `d3f3ae0`. Build `work/build/q8`, logs `/var/tmp/claude/q8/`. Default of the route would be off
(`TONA_PREFILL_INT8`, not wired: the micro test comes first, see the stop rule).

## Test conditions (before the first run, 13:23)

`python3 tools/hw_report.py` (`/var/tmp/claude/q8/hw.txt`): Ryzen 9 9950X (16C/32T), RX 9060 XT gfx1200 15.9 GiB, RAM 60.5 GiB
(48.2 GiB available), kernel 7.3.4, hip 7.16, no `TONA_*` set, commit `d3f3ae0` (dirty: this change). `tools/quiet_check.py`:
**do not measure**: other programs 73 % of the CPU threads, 4.44 GiB of the 15.9 GiB VRAM held by others (the desktop and other
contributors). Measured anyway as the task allows; every GPU run is a queued job and the load of the machine is a source of
noise for the CPU side only (the micro test is a GPU kernel timing, 10 repetitions).

## Prior art

- **Our own int8 expert attempt (#100)**, `2026-11-02-prefill-int8.md` Run 2 and the finding "the 8-bit route does not pay for the
  experts": int8 WMMA, weights unpacked to int8 in the LDS, scale per K block of 32 applied in float after the int32 accumulation
  ran at **0.76 to 1.20x** of `k_gemm_q2` (IQ4_NL, a scale per 32 and no offset: 1.17 to 1.20x; IQ2_S, a scale per 16: 0.77x).
  Cause: the scale conversion (cvt, multiply, fma per output element and 32 values) costs about as much as the int8 WMMA saves.
  Status: stopped, code on branch `100-int8-experts-archive` (7e566ec); this work ports that kernel (namespace, tile helpers) and
  adds what Q4_K and Q6_K need that the IQ formats did not: an **offset per scale group** (w = s q + b), cancelled exactly by the
  per-group sum of the int8 activations (the decode path's formula in `matvec_q8_core.h`, `Dec`).
- **Our dense FP8 prefill (#100, #106, `2026-11-19-ornith-prefill.md`)**: kernel 1.35 to 1.78x, end to end +2.7 % at 16K, mean
  |d nll| 0.216 against 0.123 (f16 prefill).
- **Our FP8 expert micro test (#159, `2026-11-17-prefill-fp8-experts.md`)**: 1.27 to 1.52x on the expert GEMM with the scale folded
  into the weight value (no per-block scale math), but the quality budget was already used.
- **llama.cpp MMQ for RDNA4 (PR 17156)**: scales per 32-value sub-block, `v_wmma_i32_16x16x16_iu8`; it compares against dequantize +
  rocBLAS, not against a fused f16 kernel as here. Status of the PR and of newer work: see the search below.

## Plan 1: micro test of the int8 expert GEMM (written before the run)

**Question.** Is `gemm_q8` (int8 WMMA, Q4_K and Q6_K weights unpacked to int8 in the LDS, activations int8 per token and 32 values)
clearly faster than `k_gemm_q2` (f16 WMMA, unpack to f16) on the expert shapes of Ornith at the token counts the experts see in a
16K prefill? **Decision.** Median ratio of the GEMM alone (without the activation quantization, which is shared by gate and up) at
T = 100 to 320 of at least 1.3x: go on (wire, A/B, quality). Below 1.15x: stop and report. In between: report and ask.

**Scope.** One `exclusive` job, `tona-gemm-q8-bench` on `Ornith-1.5-35B-Q4_K_M.gguf`: gate (Q4_K, 512 x 2048), down (Q4_K and Q6_K,
2048 x 512), 32 experts of T = 64, 100, 200, 320 tokens, 10 repetitions after a warm-up, one process (the kernels are
deterministic; the repetitions are the spread). Prints the deviation of the int8 result from the f16 one. About 1 minute.

**Expectation (source: the cycle count below and the IQ4_NL result of #100).** Per 16 x 16 output tile and 32 values the int8 route
issues 2 WMMA (8 cycles each at 186 TOPS: 64 SIMDs, 2.9 GHz) and an epilogue of 8 values per lane x (cvt, mul, 2 fma) = 32 VALU
cycles; the f16 route issues 2 WMMA of 16 cycles with no epilogue. Without overlap of WMMA and VALU on one SIMD that is 48 against
32 cycles for the matrix part: int8 is slower there, and only the cheaper unpack (nibble kept as int8) can pay for it. Expected
ratio 0.8 to 1.15x for Q4_K, 0.6 to 0.9x for Q6_K (two groups of 16, twice the epilogue). Mean deviation of the int8 result from
the f16 one 0.5 to 1.5 % of mean |y| (int8 rounding of activations, #100: 0.65 to 0.73 %).

**Active tooling.** none (no diagnostics, no profiler).

Prior art search (2026-11-20, web): llama.cpp PR 17156 (RDNA4 WMMA-MMQ, merged late November 2025 per a third-party report, not
verified on the PR page), follow-ups 17495 / 17576 (RDNA3 int8 WMMA-MMQ), 17990 (RDNA mma refactor), 18537 (MMQ/WMMA switching
tuning). All compare against dequantize + hipBLAS, not against a fused unpack-in-LDS f16 kernel; no newer RDNA4 int8 prefill work
found that changes the per-32-scale epilogue cost. Search limited to one web query plus our own notes.

## Result 1: micro test (job `q8-micro`, `/var/tmp/claude/q8/micro1.txt`; 32 experts, 10 repetitions after a warm-up, kernel only)

Quiet check before: do not measure (CPU 73 % by others, 4.44 GiB VRAM by others); after: `/var/tmp/claude/q8/quiet-after.txt`.
Ratio = f16 time / int8 time (above 1: int8 faster). Mean deviation of the int8 result from the f16 result 0.53 % of mean |y| in every
case (correct kernel, as the #100 kernel: 0.65 to 0.73 %).

| Weight | T = 64 | T = 100 | T = 200 | T = 320 |
|---|---:|---:|---:|---:|
| gate/up Q4_K 512 x 2048 | 0.75x | 0.80x | 0.76x | 0.84x |
| down Q4_K 2048 x 512 | 0.95x | 0.90x | 0.91x | 0.97x |
| down Q6_K 2048 x 512 | 0.76x | 0.63x | 0.61x | 0.65x |

Absolute at T = 320: gate f16 0.728 ms (29.5 TFLOPS) against int8 0.864 ms (24.8); Q6_K down 0.846 against 1.311 ms. The separate
activation quantization costs another 0.02 to 0.24 ms per call (gate/up share one quantization).

**Finding.** Within the expectation (0.8 to 1.15x Q4_K, 0.6 to 0.9x Q6_K) and below the stop threshold of 1.15x at every token count:
**the int8 route is slower than the f16 expert kernel for Q4_K and Q6_K**, the same verdict as #100 for the IQ formats. Reason
(cycle count in the plan): the per-group float epilogue (cvt, scale, offset fma per output element and 32 values) costs more VALU
cycles than the halved WMMA time saves, and the offset term of the K-quants adds one more fma. The estimate of
`2026-11-20-ornith-prefill.md` (8 to 11 % at 16K) assumed the epilogue was free. The dense GEMMs would use the same epilogue
(per-32 scales) and the FP8 route (scale per row, no epilogue) already measured only +2.7 % at 16K, so int8 with per-32 scales
cannot beat it either; a per-row int8 scheme would be FP8's structure with a quality problem on activations with outliers.

**Decision per stop rule: not wired, no `TONA_PREFILL_INT8`, no A/B and no quality runs** (they would measure a slower route).
Not taken because the kernel is slower; searched 2026-11-20 (above): nothing new; tried: offset folded into a rank-1 term, Q4_K as
one chain of both WMMAs per block. Untried ideas (not expected to flip it): weights pre-converted to int8 with a per-row scale
(memory x2 for the cache, quality loss), epilogue by packed math. The kernel (`gemm_q8.hip`) and the tool
(`tona-gemm-q8-bench`) were not merged (no dead code in the engine); a re-test rebuilds them from the #100 int8 kernel plus the
per-group offset term described above.
