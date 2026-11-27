# Prefill breakdown on the reference PC (#159)

## Plan (written before the first run)

- Question: where does the **prefill** time of a 30K and a 130K prompt go, per kernel family, against the ceiling of each
  family? Decision it feeds: which prefill lever is built first (ranked list in the last section).
- Scope: one server start of the production path (`tona-server`, MTP drafter, `--ctx 140000`, expert profile as `ab.py`),
  build `work/build/engine` of the main checkout (built 2026-11-16 11:18, sources of main, no code change here), under
  `rocprofv3 --kernel-trace --memory-copy-trace` (rocpd). Three requests in order, 4 decode tokens each:
  1. `sum` 2K tokens (warm-up of the GPU clock ramp, discarded; research/measurements/2026-11-12-prefill-bimodal.md,
     Result 2),
  2. `write` 30K tokens (warm: the GPU runs at 3.1 GHz when it starts),
  3. `sum` 130K tokens (warm). `write` and `sum` have different heads so the prompt cache does not reuse the 30K prefix
     for the 130K request (the log line `Prefill N tokens` of `TONA_PF_DEBUG` shows the tokens really prefilled).
  Expected duration: load 25 s, prefill 30K about 20 s, 130K about 95 s, profiler overhead and conversion a few minutes.
- Why this size: the question is the split of the time at the two prompt lengths where coding agents live; attention
  (and indexer, top-k) grow with the context, the rest is linear, so two lengths give both coefficients. One start is
  enough: the kernel set and its times are deterministic to <1 % in the same warm state (2026-11-12 bimodal, Result 2:
  5128 kernels per request, identical gaps), the rate is **not** taken from the profiled run (the profiler changes the
  queues and the dispatch gaps).
- Expectation (source: 2026-11-02-prefill-int8.md run 1 at 13K; 2026-10-29-prefill-topk.md, 2026-10-30-prefill-attn.md
  at 130K): dense f16/fp8 GEMM 25-30 %, expert GEMM (`k_gemm_q2`) 25-30 %, DeltaNet 9-10 %, attention 7 % at 13K and
  much larger at 130K (~6.2 s of 92 s = 7 % from the 130K file, in VRAM), combine/moe_sum/norm element-wise 15-20 %,
  idle gaps 5-8 %; tolerance +-5 points per family. Prefill rate unprofiled: about 1,700 tok/s at 30K, about 1,400 tok/s
  at 130K (**prefill**, 2026-10-29-prefill-topk.md, 2026-11-02-86-ab-prefill.md).
- Ceilings: compute family = FLOP / measured WMMA peak (f16: 93 TFLOPS nominal; the measured peaks come from
  2026-11-02-prefill-int8.md: expert GEMM micro skeleton 43 TFLOPS), memory family = bytes / 315 GB/s
  (2026-10-22-micro.md: 316 GB/s). Gap = measured ms - ceiling ms.
- Active tooling: `rocprofv3 --kernel-trace --memory-copy-trace`, `TONA_PF_DEBUG=1` (logs only), `OMP_NUM_THREADS=6`,
  `ROCR_VISIBLE_DEVICES=0`, `TONA_EXPERT_PROFILE=work/profile/experts-coder.bin`. No `TONA_DETERMINISTIC`.
- Queue: the run goes through `scripts/gpu_queue.py --kind exclusive` of the main checkout (one lock).
- Quiet check: warning before the run (see conditions); the trace gives kernel times, which a busy CPU does not move
  (the GPU path is the megakernel and the prefill kernels; the host feeds the stream), idle gaps may be affected and
  are read with that caveat.

## Test conditions

Before the run (`tools/hw_report.py`, `tools/quiet_check.py`):

```text
- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.49 GiB held by others before the run, 0 % busy before the run
- RAM: 60.5 GiB, 51.9 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.1.52802-9999 (shell), the engine runs with the ROCm 10.1 toolchain of work/rocm-10.1
- TONA_* variables: none set in the shell
- Load average (1, 5, 15 min): 1.39, 0.33, 0.16; CPU use by other programs over 0.5 s: 96.7 % of all threads, swap in use 2.1 GiB
- Commit: e4a4f41 (main), worktree clean
- Quiet check (2026-11-16 12:35:39): do not measure (other programs use 67.2 % of all CPU threads); warning: 1.49 GiB VRAM held by others
```

Quiet check after the runs (12:48:50): warning, 1.68 GiB VRAM held by others, the system swapping (5 pages in 1 s). Before and between
the runs other programs used 67 to 76 % of all CPU threads (load average up to 9.9); the GPU lock was held by this job alone. Names of
programs are not recorded.

## Results

Raw data (not in the repository): `/var/tmp/claude/w159/run1/trace_results.db` (rocpd; 2K, 30K and 130K request in one start), `run2/`
(repeat of the 30K request with `--hip-runtime-trace`). The tool is `tools/prefill_breakdown.py`. Prompt tokens really prefilled (log of
`TONA_PF_DEBUG`): 29,913 (layer by layer, chunk 5,120, groups of 15,360) and 129,769 (chunk 4,096, groups of 16,384). The first request (2K)
was cold, as expected (1,239 tok/s **prefill**); 30K and 130K are warm.

Rates inside the profiled run, **prefill**: 30K 1,881 tok/s (repeat 1,889), 130K 1,780 tok/s; unprofiled `ab.py` rates of 2026-11-03
(older build) 1,884 and 1,784: the profiler does not move the rate, so the idle gaps below are real (not a profiler artefact).
The kernel time of the repeat of 30K equals the first one to 0.1 % (family times within 0.5 %).

### Breakdown, 130K prompt (window 72.93 s, 129,769 tokens)

Window = first to last kernel of the request (4 decode tokens included, 0.3 s). Idle = window minus the union of kernel intervals.
Useful TFLOP from the weight counts in `tools/prefill_breakdown.py` (padding of partial tiles not counted). Ceiling at the measured WMMA
peaks (f16 93, fp8 186 TFLOPS; `docs/backends/reference-platform.md`); **under sustained load the GPU runs at the 182 W limit with 2.6 to 2.8 GHz,
so the realistic ceiling is about 0.85 of these** (research/measurements/2026-10-24-gemm-v2.md); the gap column uses the full peak.
Rows marked (bytes) use a byte count estimated from the buffers (fp32 residual of 4 x 2560 values = 41 KB per token and sublayer, expert
outputs 10 x 2560 fp32) at 315 GB/s; rows marked (est.) use a FLOP estimate of the kernel's work (top-k 2048 keys for attention, one score
per 4 tokens for the indexer). Both are estimates, not measurements.

| Family | ms | % | calls | useful TFLOP | TFLOPS (peak) | ceiling ms | gap ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| MoE expert GEMM (`k_gemm_q2`: unpack to f16 in LDS, f16 WMMA) | 16,995 | 23.3 | 58,921 | 612 | 36 (93) | 6,584 | 10,411 |
| DeltaNet recurrence (`k_dn_wave`) | 6,276 | 8.6 | 1,188 | | | | |
| dense f16 GEMM (`k_gemm2`: ssm_out, attn_output, hc down, router, indexer, alpha/beta) | 6,142 | 8.4 | 11,479 | 293 | 48 (93) | 3,152 | 2,991 |
| residual combine, norm (`k_combine`, `k_combine_norm16`, `k_norm16`) | 5,966 | 8.2 | 8,070 | | | ~4,000 (bytes) | ~2,000 |
| attention (`k_attn_wmma`, 12 layers, top-k 2048) | 5,780 | 7.9 | 396 | ~78 (est.) | ~13 | ~900 | ~4,900 |
| dense fp8 GEMM (`k_gemm8b`: qkv, gate, q, k, v, shared expert) | 5,551 | 7.6 | 8,316 | 559 | 101 (186) | 3,004 | 2,547 |
| indexer score + top-k (`k_idx_score_wmma`, `k_topk`) | 5,358 | 7.3 | 95,832 | ~26 (est.) | ~5 | ~300 | ~5,000 |
| MoE sum, swiglu, scatter, route | 4,801 | 6.6 | 28,075 | | | ~4,900 (bytes) | ~0 |
| hc up GEMM + mix (`k_gemm2<256,true>`) | 3,150 | 4.3 | 3,233 | 82 | 26 (93) | 884 (compute), ~2,200 (bytes) | ~1,000 |
| DeltaNet conv, norm (`k_dn_conv_l2`, `k_dn_norm`) | 2,826 | 3.9 | 2,376 | | | ~2,400 (bytes, 2026-11-03-dn-fuse.md) | ~400 |
| QSA prep (norm, rope, cache write) | 1,910 | 2.6 | 2,143 | | | | |
| fp8 weight unpack (`k_dequant_fp8`) and input convert (`k_to_fp8`) | 1,722 | 2.4 | 13,068 | | | | |
| f16 weight unpack (`k_dequant`) | 467 | 0.6 | 17,880 | | | | |
| other (PLE, MTP head, fills) | 309 | 0.4 | 8,709 | | | | |
| idle between kernels | 5,670 | 7.8 | | | | | 5,670 |
| **window** | 72,925 | 100 | 259,686 | | | | |

Idle gaps (130K), by size: 285 gaps of 0.5 ms and more total 2.05 s (the largest: 9 at the group boundaries, 129 to 348 ms each, the
GPU waits while the host copies 3.76 GB of loaned expert slots back, 1,704 H2D copies at 29 GB/s); 9,286 gaps of 50 to 500 us total
2.58 s (mean 280 us, 1,000 to 1,400 per kind of kernel pair such as `k_dn_wave` to `k_dn_norm`, `k_combine_norm16` to `k_dequant`); 239,000
gaps below 10 us total 0.88 s (launch floor, 3.7 us each). The 50 to 500 us gaps do not follow a copy (28 of 9,286 end right after one)
and are not host-bound (30K repeat with `--hip-runtime-trace`: `hipLaunchKernel` 101 ms in total, the host waits in
`hipStreamSynchronize` for 15.5 of 16.5 s, so the queue is far ahead). **Cause unknown (ungeprueft).**

Copies: 98,365 H2D copies, 216.6 GB (expert weights streamed per layer and group, `layer_buffers`), copy time 4.56 s on the SDMA engines,
hidden behind the compute (PCIe 55 GB/s would need 3.9 s).

Tile fill of the expert GEMM (`grid_x` / 256 = blocks of every `k_gemm_q2` launch): 746 TFLOP executed in 16.995 s = 44 TFLOPS against
612 TFLOP useful = 82 % fill. The kernel unpacks the 128 weight rows of its tile for **every** 64-token tile, so a weight is unpacked
ceil(tokens / 64) times per chunk of 4,096 tokens (about 160 tokens per expert on average); dense weights are unpacked once per chunk too
(`layer_order()` runs all chunks of a layer in a row, `gemmq_fp8` unpacks per call).

### Breakdown, 30K prompt (window 15.80 s, 29,913 tokens)

| Family | ms | % |
|---|---:|---:|
| MoE expert GEMM | 3,817 | 24.2 |
| DeltaNet recurrence | 1,472 | 9.3 |
| dense f16 GEMM | 1,439 | 9.1 |
| residual combine, norm | 1,378 | 8.7 |
| attention | 1,309 | 8.3 |
| dense fp8 GEMM | 1,254 | 7.9 |
| MoE sum, swiglu, scatter, route | 1,140 | 7.2 |
| hc up GEMM + mix | 729 | 4.6 |
| DeltaNet conv, norm | 659 | 4.2 |
| QSA prep | 442 | 2.8 |
| indexer score + top-k | 391 | 2.5 |
| fp8 weight unpack and input convert | 341 | 2.2 |
| f16 weight unpack | 85 | 0.5 |
| other | 117 | 0.7 |
| idle between kernels | 1,231 | 7.8 |

Per token, 30K: 0.528 ms; 130K: 0.562 ms (+6 %). Everything but the attention and the indexer is linear in the prompt length: the
quadratic part is 13.6 % at 30K and 17.8 % at 130K (attention + indexer + QSA prep).

### Against the plan

| Item | Expected | Measured | Verdict |
|---|---|---|---|
| all GEMMs (expert, dense f16, dense fp8, hc up), 130K | about 55 to 60 % | 43.6 % (23.3 + 8.4 + 7.6 + 4.3) | **OUTSIDE (lower): finding**: since the layered prefill and the FP8 route the GEMMs are less than half of the window |
| DeltaNet (recurrence + conv/norm) | 9 to 10 % (recurrence alone) | recurrence 8.6 %, with conv/norm 12.5 % at 130K, 13.5 % at 30K | within for the recurrence, conv/norm not in the plan |
| attention | about 7 % | 7.9 % (plus indexer 7.3 %, QSA prep 2.6 %) | within; indexer not in the plan |
| element-wise (combine, moe_sum, norm, swiglu, scatter, hc mix, dn conv/norm, QSA prep, fp8 convert) | 15 to 20 % | 31 % | **OUTSIDE (higher): finding** |
| idle | 5 to 8 % | 7.8 % | within |
| rate unprofiled | about 1,700 / 1,400 tok/s | 1,881 / 1,780 tok/s (profiled equals unprofiled) | OUTSIDE (higher: the plan quoted older files) |

The two families that dominate the gap to the ceiling are the expert GEMM (10.4 s at the f16 peak) and the sum of the bandwidth-bound
element-wise and idle parts. A prefill with every compute family at its peak and every element-wise kernel at its byte ceiling would take
about 13.6 s (compute at 93 / 186 TFLOPS) + 14 s (bytes) = 28 s against 73 s now (**prefill** about 4,600 tok/s; an upper bound, not a plan).

## Levers, ranked

Gains are the share of the window the lever can remove, as prefill tok/s at 130K (now 1,780) and 30K (now 1,881). Estimates, not
measurements. Prior art in the last column.

| # | Lever | Bit-exact? | Removes (130K / 30K) | tok/s 130K / 30K | Prior art and state |
|---|---|---|---|---|---|
| 1 | Expert weights unpacked once per layer and group into fp8 (the dense FP8 route: `k_dequant_fp8` + `k_gemm8b`, 101 TFLOPS in the same trace), then one plain GEMM for all chunks; today every 64-token tile unpacks its 128 weight rows again | **no** (e4m3 weights: quality check R6 of `docs/backends/README.md`; dense FP8 was accepted 2026-11-03) | 7 s / 1.8 s (unpack about 1.9 s + GEMM 6.1 s at 101 TFLOPS and 82 % fill = 9.7 s against 17.0 s) | +10 % / +13 % (about 1,960 / 2,130) | int8 expert GEMM (#100) failed because it unpacked per tile at 42 % fill (2026-11-02-prefill-int8.md: 0.76 to 1.2x); this design avoids exactly that. pat883/w4a8-fp8-wmma (HF, unmeasured, gfx1200 in the fat binary) |
| 2 | Same unpack-once, but to f16 and the f16 GEMM path | **yes** (same f16 values and K order; `TONA_DETERMINISTIC=1`) | only the redundant unpack, est. 1.5 to 3 s; the GEMM stays at 36 to 48 TFLOPS | +2 to +4 % | 2.5 GB per layer for 256 experts, loan limit 3.9 GB: slices only |
| 3 | Dense weights unpacked once per layer and group, not once per chunk (4 chunks per group at 130K, 3 at 30K) | **yes** | 1.4 s / 0.28 s (of 1.9 s / 0.43 s) | +2.0 % / +1.8 % | structure of `layer_order()`; nothing external |
| 4 | Do not copy the loaned slots back between groups (only after the last): the GPU idles 129 to 348 ms at each of the 9 boundaries | **yes** (scheduling) | 1.2 to 1.5 s / 0.13 s | +1.7 to +2.1 % / +0.85 % | Layered Prefill arXiv 2510.08055 (our basis) |
| 5 | Attention and indexer: indexer score 4.1 s + top-k 1.2 s at about 5 TFLOPS, attention at about 13 TFLOPS: fewer syncs, larger tiles, INT8/FP8 QK (`research/34` ideas 5, 6) | tile and sync changes yes, FP8 QK no | 3 to 4 s / 0.4 s | +4 to +5 % / +1 to +2 % | DeepSeek DSA lightning indexer (FP8); StreamIndex arXiv 2605.02568 (top-k streaming); not read in depth |
| 6 | Find and remove the cause of the 50 to 500 us gaps (2.58 s) | likely yes | up to 2.6 s / 0.6 s | up to +3.6 % / +4 % | cause unknown; first step is a diagnosis (hipGraph of one layer-chunk, queue and fence settings), not a build; see the queue finding of 2026-11-12-prefill-bimodal.md |
| 7 | Expert GEMM token tile 128 (halves the repeated unpack; fill drops from 82 %) | **yes** | 1.7 to 2.5 s / 0.4 to 0.6 s | +2.4 to +3.5 % / +2.6 to +3.9 % | part of lever 1 if lever 1 is taken |
| 8 | DeltaNet chunk form (WY/WMMA): measured +2.3 % at 30K, +1.8 % at 130K in the prototype of #30 | no (rounding; paired -log p within noise) | 2.1 s / 0.4 s | +1.8 % / +2.3 % | arXiv 2406.06484, FLA `chunk_gated_delta_rule`, FlashQLA (QwenLM); llama.cpp #20340 (fused beat chunked for Qwen3.5); the prototype is not in the tree |
| 9 | Fewer bytes for the fp32 residual and the expert outputs (fusing combine into GEMM epilogues, fp16 intermediates): these kernels sit at 65 to 100 % of 315 GB/s, only fewer bytes help | pure fusions yes, fp16 residual no | est. 2 to 3 s | +3 to +4 % | `research/16-prefill-fusion.md` (our fusions listed; DeepSeek mHC kernels are CUDA only) |

Levers 1 and 2 do not stack, 7 is part of 1. Order if FP8 experts are accepted: 1, 4, 3, 5, then 6. Bit-exact only: 3 and 4 first
(together +3.7 % at 130K, +2.7 % at 30K, small), then 7 or 2, then 5.
Not worth it by the numbers: the f16 GEMM kernel itself (48 TFLOPS is about 0.6 of the sustained ceiling already), the hc-up GEMM (near
its byte ceiling).

Prior-art search for this list (read, no external scan with the weekly tool was run for this step): `research/papers.md` (DeltaNet
chunking, Layered Prefill, StreamIndex, FlashQLA, pat883 FP8 WMMA), `research/papers-screened.tsv`, `research/16-prefill-fusion.md`,
`research/measurements/2026-11-02-prefill-int8.md`, `2026-11-03-dn-chunk-ab.md`, `2026-11-03-dn-fuse.md`, `2026-11-12-prefill-bimodal.md`.
Still to search before building lever 1: hipBLASLt/CK grouped GEMM and RDNA4 FP8 MoE kernels in vLLM and llama.cpp since 2026-11-01.

Search on lever 1 (2026-11-16, one web search, topics: RDNA4 gfx1200 FP8 WMMA MoE grouped GEMM, vLLM, llama.cpp): vLLM supports FP8 on gfx12xx
with native WMMA and tuned Triton FP8 MoE configs per shape (discuss.vllm.ai threads "Native FP8 WMMA Support for AMD RDNA4" and "RDNA4 FP8
support"; forum reports, not benchmarks); VulkanForge (third party, Vulkan) reports +33 to +39 % MoE prefill from native FP8 WMMA (self-reported).
Nothing found in llama.cpp for gfx1200 FP8 grouped MoE. No cheaper route than the FP8 unpack-once design found.
