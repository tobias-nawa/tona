# Prefill GEMMs in 8-bit WMMA (#100): profile and measurements

## Plan (written before the first run)

### Run 1: fresh kernel profile of the prefill at 13K (R-profile, not a recipe of docs/contribute/measurement.md)

- Scope: one GPU block, `scripts/golden.sh`-equivalent command (`GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash`: the
  13,328-token prompt `scripts/bench-long-code.txt`, `--prefill --ctx 16384 --dflash`, 64 decode tokens, big model,
  `IE_DETERMINISTIC=1`) under `rocprofv3 --kernel-trace`. One run, about 1 min incl. 18 s load plus profiler
  overhead. Kernel time summed per name; the prefill part is the span before the first decode token.
- Why this size: the question is the share of the GEMMs at the length where agent turns live (13K); one run is
  enough because the shares of a deterministic run do not scatter (the kernel set is fixed); the rate is not
  measured here (the profiler distorts it).
- Expectation (source: levers.md section 5 and the 10-24 profile): GEMM (k_gemm2 + gemm_q + dequant) 45-55 %,
  DeltaNet (k_dn_wave) about 7 %, attention about 6 %, GPU idle below 5 %, each +-5 points.
- Active tooling: rocprofv3 kernel trace (rocpd), `IE_DIAG=1`, `IE_DETERMINISTIC=1` (as in the golden command).
  No test switches otherwise.

## Results

### Run 1: profile, 13,328-token prompt, big model, dflash, rocprofv3 kernel trace (1 run, build w100 = main bcde1e5)

Prefill window 7.41 s (first prefill kernel to first k_mega; the profiler slows it, so no rate is taken from this
run). Kernel time 6.84 s, GPU idle 0.56 s (7.6 %). Raw data: /var/tmp/claude/w100/prof/.

| Kernel | Time | Share of window | Share of busy | Calls |
|---|---:|---:|---:|---:|
| k_gemm2 (dense f16) | 1.942 s | 26.2 % | 28.4 % | 2028 |
| k_gemm_q2 (experts, unpack to f16 in LDS) | 1.805 s | 24.4 % | 26.4 % | 5196 |
| k_dn_wave | 0.684 s | 9.2 % | 10.0 % | 108 |
| k_attn_wmma | 0.512 s | 6.9 % | 7.5 % | 33 |
| k_moe_sum / k_combine_norm16 / k_combine | 0.706 s | 9.5 % | 10.3 % | 141 each |
| k_dn_conv + k_dn_norm + k_dn_l2 | 0.344 s | 4.6 % | | |
| k_qsa_prep | 0.198 s | 2.7 % | | |
| k_norm16, k_swiglu16, k_scatter, k_dequant | 0.511 s | 6.9 % | | |
| idx_score, topk, rest | < 0.2 s | < 3 % | | |

Against the expectation: GEMM (k_gemm2 + k_gemm_q2) 50.6 % of the window (expected 45-55 %: inside); DeltaNet 9.2 %
(expected 7 +- 5: inside, upper half); attention 6.9 % (expected 6 +- 5: inside); idle 7.6 % (expected < 5 +- 5: inside,
at the edge). No finding outside the expectation.

Expert GEMM work: 13,328 tokens x 8 experts x 3 matrices x 2 x 2560 x 640 x 47 MoE layers = about 49 TFLOP in 1.805 s =
about 27 TFLOPS (f16 peak 93, int8 183): the expert GEMM is bound by unpacking and LDS, not by the WMMA units.

### Run 2: micro test of the int8 expert GEMM, variant A (`ie-gemm-q8-check`, #100)

Plan (written before the runs): build `gemm_q8` (int8 WMMA, weights unpacked by `decode_q8` into the LDS, activations int8 with
a scale per token and 32 values, int32 accumulators converted and scaled per K block of 32 = variant A, plus the double-K
128-bit LDS loads of GPUOpen WMMA guide part 2) and compare it with the heated `k_gemm_q2` on the expert shapes of the model
(640 x 2560 for gate/up in types 18, 21, 22, 23; 2560 x 640 for down in types 20, 42). Expectation: >= 1.2x (the bar of the
maintainer for going on without asking); source: the WMMA peak (183 against 93 TFLOPS) and the 27 TFLOPS of run 1. Workload:
32 experts (the segment limit of one launch) with T tokens each, 10 timed repetitions after one warm-up, one run each (the
kernels are deterministic and the repetitions inside a call are the spread; no GPU time beyond about 1 min per run).
T = 416 (mean of a 13K prompt if the experts saw the whole prompt), then T = 48 and 128. The mean tokens per expert in the
real prefill follow from run 1: 3.54 M tiles of 64 tokens for 1.8 s and 49 TFLOP, i.e. the tiles are about 42 % full (about
27 tokens per tile) because the experts see one group of the prompt at a time. Active tooling: none (no diagnostics).
Test switches during the investigation (compile-time, removed again): EXP1 = no unpack, EXP2 = no scale math, EXP3 = both.

Correctness: kernel against a CPU sum in double of the same quantized activations and exactly unpacked weights: deviation
<= 2e-6 of the maximum for all 6 expert formats (types 18, 20, 21, 22, 23, 42). Quantization against the f16 GEMM: mean
deviation 0.65-0.73 % of the mean magnitude (not a quality number; the quality is measured in NLL).

Speed, `gemm_q8` against `k_gemm_q2` (ratio > 1: int8 faster), 32 experts:

| Type (shape) | T = 48 | T = 128 | T = 416 (8 experts) |
|---|---:|---:|---:|
| 18 IQ3_XXS (640 x 2560) | 0.99x | 1.03x | 1.01x |
| 20 IQ4_NL (2560 x 640) | 1.17x | 1.20x | 1.19x |
| 21 IQ3_S | 0.99x | 1.00x | 1.01x |
| 22 IQ2_S (scale per 16: two conversions per block) | 0.76x | 0.77x | 0.77x |
| 23 IQ4_XS | 1.05x | 1.10x | 1.09x |
| 42 Q2_0 (2560 x 640) | 1.02x | 1.07x | 1.05x |

The expectation (>= 1.2x) is missed: variant A is at 0.76-1.20x. Attribution at T = 48 (type 18, 32 experts, absolute: gemm_q
0.283 ms, gemm_q8 0.285 ms): skeleton without unpack and without scale math 0.117 ms (2.4x), without scale math only 0.201 ms
(1.40x). So of the 0.285 ms about 0.117 ms are the skeleton (loads of X, barriers, WMMA, store),
about 0.085 ms the unpack and about 0.084 ms the scale math of variant A. The scale math is 30 % of the kernel; variant B
(activation scale per 128 values, weight scale as small integer in the int32 accumulator, float conversion once per 128) would
remove about half of it for types 18, 21, 22, 23 and less for the formats with a float scale per 32 (20, 42, 8): estimate 1.1-1.2x,
not measured. The skeleton at 43 TFLOPS shows that the barriers and loads (no pipelining across K steps) are an equal part.

## Finding: the 8-bit route does not pay for the experts (decomposition)

The int8 expert GEMM (variant A) is correct (deviation <= 2e-6 of the maximum against a CPU sum of the same quantized
activations) but only 0.76-1.20x of the f16 kernel at the token counts the experts really see. Why:
- Run 1: the 3.54 M tiles of 64 tokens hold 49 TFLOP of useful work in 1.805 s. The "27 TFLOPS" are a work measure, not a
  rate of the matrix unit: the tiles are about 42 % full (the experts see about 27 tokens per pass), so the kernel
  streams and unpacks weights for few tokens.
- Ablation on the f16 kernel (type 18, 32 experts x 64 tokens, 0.290 ms): without WMMA 0.178 ms, without unpacking 0.188 ms,
  without both 0.081 ms. So unpacking (about 0.10 ms) and WMMA (about 0.11 ms) are additive on top of a 0.08 ms skeleton:
  the waves of a block do the two phases one after the other, and the 8-bit WMMA halves only the matrix part while its
  scale conversion (30 % of the int8 kernel, variant A) costs about as much as it saves.
- Variant B (activation scale per 128, integer weight scale in the accumulator) can only remove about half of the scale
  cost and only for the grid formats: estimated 1.1-1.2x, not built (the maintainer stopped the int8 experts).
- The code stays on branch `100-int8-experts-archive` (commits 7e566ec, 48e8112), not in the main-bound branch.

## Run 3: expert GEMM, bit-identical speed-ups of the f16 kernel (`k_gemm_q2`)

Plan (written before the runs):
- Change 1 (done): in a partial last token tile of up to 32 tokens the four waves of the upper token half skip the WMMAs.
  Same K sum per output element in the same order, so bit-identical. Tried and dropped (measured, no gain or loss):
  prefetching the weight lines of the next super-blocks (+8..15 % time), unpacking the next step into registers during the
  WMMAs (+2..4 %), producer and consumer waves with a double-buffered LDS (+45..60 %: 1 block per CU, the 4 producer waves
  are latency-bound).
- Micro test `ie-gemm-q-bench` (32 experts, T = 20, 27, 48, 64, 100; 10 repetitions after a warm-up; the hashes of the
  output of the base and the new build must be equal). Expectation: -20 % time for T <= 32, unchanged (+-3 %, the noise of
  the tool) for T >= 48; source: the WMMA share 0.11 of 0.29 ms in the ablation above.
- Bit-exactness: golden checks gpu, cpu plan, prefill, prefill dflash, server (all identical to the references), SHA of
  `ab.py` workloads `mid` and `long` equal to bench/sha.toml.
- Profile as run 1 (13,328 tokens, big model, dflash, rocprofv3, same command). Expectation: k_gemm_q2 1.805 s ->
  1.45-1.55 s (about 42 % tile fill means roughly 60 % of the tiles are partial and at most 32 tokens wide: -20 % on those), so
  the prefill window 7.41 s -> 7.0-7.1 s (+5 %).
- Speed: `scripts/ab.py --a merged --b w100 --workload server --targets 2000,13000,30000` (prefill tok/s; `merged` = main
  bcde1e5), rounds decided by `ab.py` (noise.toml has prefill@2000 and prefill@30000, no 13000: at least 2 rounds there).
  Expectation +3..6 % prefill, decode unchanged (+-0.5 %).
- Active tooling: IE_DIAG=1 for the ab.py arms, IE_DETERMINISTIC=1 for the goldens; no profiler in the speed runs.

### Run 3 results (expert GEMM, change 1)

- Micro test (`ie-gemm-q-bench` predecessor, 32 experts, base against change 1, same data): T = 20 / 27: 0.28 -> 0.22 ms
  (-22 %) for all six formats (IQ4_NL 0.251 -> 0.193, Q2_0 0.225 -> 0.160); T = 48 / 64 / 100: unchanged within +-3 %
  (0.283 -> 0.285 for IQ3_XXS at T = 48). The hashes of the output of base and new build are equal for all formats and
  T = 20, 27, 48, 64, 100. Within expectation.
- Goldens (`scripts/gpu-status.sh --own --build work/build/w100`): prefill dflash, gpu, cpu plan, server, prefill: all
  identical. Bit-identical as planned.
- Profile (as run 1, 13,328 tokens): window 7.41 -> 6.88 s, k_gemm_q2 1.805 -> 1.672 s (-7.4 %), k_gemm2 1.942 -> 1.849 s
  (not touched: drift of the GPU between the runs, about 5 %), idle 0.56 -> 0.38 s. The expectation for k_gemm_q2 was
  1.45-1.55 s: the gain on the real prefill is a third of the micro test's, so the partial tiles of at most 32 tokens
  are fewer than the 42 % fill suggested, or the tiles that matter are the full ones. A finding; the window difference
  is partly drift.
- `ab.py` (`research/measurements/2026-11-02-prefill-int8-ab-expert-skip.md`, `merged` = main bcde1e5 against this build, 6 rounds, IE_DIAG=1):
  prefill@2000 +1.35 %, prefill@13000 +1.12 %, prefill@30000 +0.91 % (every round positive, outliers named in the report);
  decode -0.19 / -0.08 / -0.09 % (equal within the noise); SHA of the text and the expert counters identical in both arms at
  all three sizes. Expected +2 to +8 % prefill: **below the expectation, a finding** (the server prefill is not only the
  kernel: PCIe copies of the experts, CPU waits). Small gains count (AGENTS.md memory): kept.

### Run 4 plan: fp8 dense GEMM on the K128/BN256 structure of `k_gemm2` (`k_gemm8b`, step B)

- Kernel: `k_gemm8b` in gemm_fp8.hip: tile 128 x BN2 (256 for outputs >= 4096, else 128), K step 128 bytes in one LDS
  buffer, loads through registers during the WMMAs, 128-bit LDS loads feeding two WMMAs (double-K). Same numbers as the
  old fp8 kernel (factor per row, E4M3); only the sum order over K differs. Switch IE_FP8_V1=1 selects the old kernel.
- Micro test: `ie-gemm-check` shapes qkv 2560->10240, q 2560->12288, ssm_out 6144->2560 at T = 2048 against `k_gemm2`,
  one run each (kernel timing of 10 repetitions). Expectation >= 1.25x (the bar for building the full path); source: the
  old fp8 kernel gave 1.1-1.2x at 27 % of the fp8 peak, the f16 K64/BN256 kernel reached 1.25-1.4x over the old f16 one.
- Speed in the prefill: `ab.py --a w100 --b w100 --b-env IE_PREFILL_FP8=1` (same build, the switch only), targets 2000, 13000,
  30000, IE_DIAG=1 both. Expectation: prefill +8 to +15 % (dense is 26 % of the window; kernel 1.3-1.6x; the weight
  conversion f16 -> fp8 per call is extra). Decode: unchanged (+-1 %); the KV changes, so the text SHA differs (counted
  and documented).
- Quality is not measured before the speed says it is worth it.

### Run 4 results (fp8 dense GEMM, step B)

- Micro test (`ie-gemm-check`, T = 2048, kernel only, 10 repetitions): `k_gemm8b` against `k_gemm2` (f16, this tool's
  numbers 42-43 TFLOPS): qkv 2560->10240 1.857 against 2.507 ms (57.8 TFLOPS, 1.35x), q 2560->12288 2.122 against 3.049 ms
  (60.7, 1.44x), ssm_out 6144->2560 0.852 against 1.514 ms (75.6, 1.78x); the old fp8 kernel gave 1.15-1.2x
  (49-52 TFLOPS at the start of this run). Above the 1.25x bar: the full path was built. Same numbers as the old fp8 kernel
  (mean deviation 3.74 % against f16 on random data, as before), only the order of the K sum differs.
- First end-to-end try (`2026-11-02-prefill-int8-ab-fp8-dense.md`, switch only): prefill **-5.1 / -6.9 / -8.3 %** at 2K / 13K / 30K. Profile: the
  per-call conversion of the weights (dequant to f16, then `k_to_fp8`) cost 0.73 s, the lost fused mix and pair paths
  (`k_mix` 0.47 s) the rest of what the GEMMs saved (k_gemm2 1.85 s -> k_gemm8b + k_gemm8 1.15 s). Expectation was +8..+15 %:
  a finding.
- Changes: `dequant_fp8` (weights straight to E4M3 in one pass, byte-identical to dequant_f16 + to_fp8_rows, checked in
  `ie-gemm-check` for every format of the model; 0.14 s instead of 0.8 s per prefill), fp8 only for dense weights with at
  least 512 rows and not for the router, the indexer, the position embeddings and the MTP head, the hyper-connection paths
  (pair, fused mix) stay f16 with the switch on.
- Second try (`2026-11-02-prefill-int8-ab-fp8-dense2.md`, same build, switch only, 6 rounds): prefill **+1.64 / +2.96 / +3.18 %** at 2K / 13K / 30K
  (every round positive), decode -1.34 / -0.08 / +9.0 % (the text changes: the expert counters and the SHA differ, the 30K
  decode rate follows the changed text, not the kernel). Expected +5..+10 %: **below the expectation, a finding**.
  The kernel profile of one run (window 6.88 s -> 6.40 s, -7 %) overstates it: single profiles drift by 5-10 % between
  runs (k_gemm_q2 1.67 / 1.86 / 1.61 s in three runs of the same kernel), only the alternating `ab.py` rounds decide.
- Not done: the quality check (docs/backends/README.md#checks) and the re-recorded references. Worth it only if the maintainer
  accepts +3 % prefill for the +0.5 % perplexity of the old measurement; see the report.

## GPU page fault of the server golden check (2026-11-02 19:17, build w100 at 89e855a)

- One abort of `scripts/golden-server.sh` inside `gpu-status.sh`: `HSA_STATUS_ERROR_MEMORY_FAULT`, kernel `k_moe_sum` (grid
  5,829,120 threads = 2,277 tokens x 2,560), read fault at 0x7f3c041c9000 to 0x7f3c041da000 ("page not present"),
  `IE_DETERMINISTIC=1`, ctx 32768. All `results/` of the cases base, sampled, thinking and reuse were written, the server
  that died was the first of the `ssd` case (a fresh server after the reuse server): the request in flight was the cold
  prompt of `ssd-1` with checkpoints (`checkpoint_at -> prefill_slices -> prefill -> moe_group`, where the stream
  synchronize reports the fault).
- Reproduction attempts on the same build (allocation trace on, `IE_DIAG=1`): `base,sampled,thinking` 5 times; the full
  case list 6 times; the sequence of `gpu-status.sh` (cpu-plan golden, prefill dflash golden, full server golden) 3 times;
  the `ssd` case alone 30 times. No fault in 44 server sessions after the first. One failure of the repro loop itself was my
  own edit of the script while it ran (syntax error), not a fault. The build `work/build/merged` could not be used for
  comparison (the freshness check refuses it: its sources are in the main checkout).
- What the tile skip can and cannot do: it removes WMMA instructions of waves that have no tokens and adds no read; the
  faulting kernel is `k_moe_sum`, which reads the expert output rows through the rowmap; a fault there means a rowmap value
  or an output buffer outside what is mapped. Nothing in the diff of this branch touches them (the diff against main:
  `gemm_q.hip`, the fp8 files, `gemmq` dispatch, the plan flag `pf_mix_fused` which only changes with the fp8 switch on).
  This is an argument, not a measurement of main.
- Added: `scripts/golden-server.sh` now always starts its servers with `IE_ALLOC_TRACE` (a 110 KB file per server in the
  run folder) and, on a page fault in the server log, runs `tools/alloc_trace_lookup.py` for the faulting address, so the
  next occurrence names the buffer.

## The gap of the fp8 dense path (from the existing profiles, no new run)

Dense GEMM time at 13K, profile 2 (f16, 6.88 s window) against profile 4 (fp8 switch on, 6.40 s window):
- f16: k_gemm2 1.849 s + k_dequant 0.097 s = 1.946 s (28 % of the window).
- fp8 on: k_gemm8b 0.691 s + k_gemm2 0.599 s (the weights that stay f16: hyper-connection, indexer, position embeddings,
  router, MTP) + k_to_fp8 0.127 s (the activations, once per GEMM call) + k_dequant_fp8 0.142 s + k_dequant 0.020 s = 1.579 s.
- Saved 0.367 s = 5.3 % of the window; the expected gain by the plan formula, share x (1 - 1/factor) with the 28 % share and
  the micro-test factors 1.35-1.78x (mean 1.5), would be 9 %. What eats the rest: 31 % of the dense time stays f16 by design
  (0.60 s), the conversions cost 0.27 s (0.13 s of them converting the same activations again for each of the 3-5 GEMMs of
  a layer that share one input), and k_gemm8b gets 1.8x on the large shapes only. Measured by `ab.py` (alternating,
  drift-free): +3.0 % at 13K; the single profiles differ by 5-10 % in drift, so 5.3 % against 3.0 % is inside that.
- A bit-identical correction inside the fp8 path would be to convert the shared activations once per layer (saves about
  0.08 s = 1.2 %); the larger remainder (0.60 s f16) is the sensitive part and not touched.
