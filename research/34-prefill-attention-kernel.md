# 34 — The prefill attention kernel `k_attn_wmma`: cost, ceiling, state of the art, ideas

Agent research (Sonnet), 2026-10-29 (issue #78)

Method: read of `engine/gpu/prefill_attn.hip` (complete), `research/29`, the measurement files of #71/#74, and **re-analysis of the
existing rocprofv3 traces** under `/var/tmp/claude/` (`w71/prof-p128-w71`, `w73/prof-p250-main`, `ctx-curve/prof-p64`; read
only, the GPU was not used, nothing was built or run). Web search, `gh` for PR/issue pages, arXiv HTML pages. Marking per entry:
- **[code]**: source of our repo read. **[trace]**: derived by this note from the existing rocprofv3 CSVs (not a new run).
- **[PR page]** / **[docs]** / **[abstract]** / **[HTML]**: page read; numbers are the authors', not re-measured by us.
- **[snippet]**: search hit only. **[estimate]**: own calculation. **[unconfirmed]**: could not be verified.

Rates are named: **prefill** (prompt tokens per s). "token-layer" = one token in one QSA layer (12 QSA layers).

---

## 0. Summary

1. **`k_attn_wmma` is 26-27 % of the GPU time at 128K and 250K and runs at 4-6 % of the WMMA peak** [trace + estimate]:
   24.0 s of 92.7 s at 128K, 51.9 s of 191.4 s at 250K, 9.7 s at 64K (section 1). Per token-layer it costs 12.4 / 15.3 / 17.3 us
   (64K / 128K / 250K average) and **grows with the position from ~6 us (positions 3K-9K) to ~20 us (>100K)** although
   every token attends to the same 2,051 cells. The cause is the gather, not the arithmetic.
2. **Roofline: the kernel is memory-bound by construction** [estimate]. Useful work is 21 FLOP per gathered byte;
   the ridge point is 295 FLOP/B (VRAM) or 1,670 FLOP/B (PCIe). Without cache reuse the gather alone needs 7.5 us per
   token-layer from VRAM and 42 us from host RAM. The measured 15-20 us is *below* the host figure, so **at least 52-64 %
   of the gathered bytes are already served by L2 through neighbouring tokens** (section 2.3). The compute floor is
   0.7-0.9 us per token-layer (1.1-1.4 s at 128K). Distance between today and the compute floor: 15-25x.
3. **Within the kernel the cause is probably a latency chain, not bandwidth** [estimate, not measured]: each iteration of 32 cells
   has three dependent global round trips (cell index -> K row -> [S, sync] -> V row), 2 blocks per CU (16 waves), no prefetch. The
   iteration takes 3 us (data near) to 10 us (data far), against ~0.6-0.9 us of arithmetic. Section 3.
4. **Best ideas** (section 5): (a) issue the V load together with the K load and prefetch the next iteration into registers
   (bit-identical, est. -35 to -55 % of the kernel); (b) token union with T = 4 (48 rows = three full WMMA tiles, K/V staged and
   dequantized once per tile; numeric, needs the overlap statistic first); (c) `IE_KV_HOST=0` up to 128K as a cheap
   control and fallback (bit-identical). The first thing to measure is the split latency / PCIe / compute (section 6).
5. **State of the art offers no ready kernel for gfx1200.** The closest published work is aiter #6576 (union kernel for QSA,
   MI308X, still open), QUILT and CompactAttention (neighbour sharing, not on GPUs of this class), and aiter #6390/#6627
   (INT8-QK/FP8-PV on gfx1201, dense). Section 4.

---

## 1. What `k_attn_wmma` costs today (question 1)

### 1.1 Shape [code]

Grid = 2 blocks per token (one per KV head), 256 threads (8 waves, wave32), LDS 30,208 B and VGPR 176 per thread (trace field;
LDS limit per workgroup is 64 KiB on gfx1200/1201 [PR page: aiter #4385], so 2 blocks per CU). 24 Q heads / 2 KV heads = 12
rows in a 16-row tile (75 %). Per token and layer: 2,051 cells (512 blocks x 4 + up to 3 tail cells) = 65 iterations of 32
cells (the last one holds 3), so 2,080 cell slots. One row of the KV cache is 1,152 B (K0 V0 K1 V1 at 512 B per head, then 2 x 16 f32 scales); a block reads 512 B + its
scales per cell, the two blocks of a token together read 1,152 B per cell.

Per iteration: cell index load -> K row (2 x 16 B per thread) + scales -> dequant to f16 into LDS -> sync -> S by 8 waves
(4 WMMA each: wave = one 16-cell tile x a quarter of the dimensions, partial sums through LDS `part[4][16][32]`) -> sync ->
**V load (only now)** + transposed f16 stores (32 single-halfword LDS writes per thread) -> softmax (16 threads per row)
-> sync -> P V (4 WMMA per wave, wave = 32 dimensions) -> sync. Four `__syncthreads`, no overlap between the load of iteration i+1
and the compute of iteration i.

### 1.2 Measured time [trace]

Sums per kernel (rocprofv3 kernel trace, one prompt each; 128K and 250K are on main after #71, the 64K trace is from before
#71, the kernel is unchanged; #74 changed only decode):

| Prompt | `k_attn_wmma` total | share | calls (chunk) | per token-layer (average) | executed WMMA rate |
|---:|---:|---|---|---:|---:|
| 64K (65,253) | 9.72 s | 22 % of 43.2 s time to first token | 180 incl. a 16K prompt; 6,144 tokens | 12.4 us | 5.5 TFLOPS (5.9 % of 93) |
| 128K (130,787) | 24.03 s (264 calls, 91 ms each) | 26 % of 92.7 s GPU | 6,144 tokens | 15.3 us | 4.5 TFLOPS (4.8 %) |
| 250K (249,730) | 51.87 s (732 calls, 71 ms each) | 27 % of 191.4 s GPU | 4,096 tokens | 17.3 us | 3.9 TFLOPS (4.2 %) |

The "91 ms per call" of `measurements/2026-10-29-prefill-topk.md` is one call over a chunk of 6,144 tokens x 2 blocks, not a
call per 32-token tile. Executed FLOPs are 68.2 MFLOP per token-layer (section 2.1).

**Cost per token-layer over the position** (each call mapped to its position by the launch order: group-major, then layer,
then chunk; plan from `server.log`: 128K chunk 6,144 / group 12,288, 250K chunk 4,096 / group 16,384; mean over the 12 layers):

| Position (tokens) | 128K run: us per token-layer | 250K run: us per token-layer |
|---:|---:|---:|
| 3K-9K | 5.95 | 8.35 (2K-22K) |
| 15K-22K | 10.9 | |
| 28K-34K | 13.7 | 14.1 (27K-51K) |
| 40K-52K | 14.6 | |
| 58K-65K | 17.0 | 16.9 (55K-76K) |
| 83K-89K | 18.2 | 17.9 (80K-100K) |
| 108K-114K | 19.2 | 18.8 (104K-129K) |
| 120K-132K | 18.4 | 18.3 (133K-154K) |
| 158K-260K | | 20.0 / 19.3 / 20.2 / 20.6 |

Both runs agree where they overlap (e.g. 17.0 and 16.9 at ~60-75K, 18.4 and 18.8-19.2 at ~110-130K), so this is the
position and not the plan. The curve rises steeply to ~60K and then flattens (~20 us). The number of cells is 2,051 from
position 2,051 on, so everything above ~6 us per token-layer is paid for *where* the cells lie. (A busy desktop would add
noise; the traces were taken on 2026-10-29 with an idle desktop per the notes of #71/#73.)

Expert streaming is not the cause of anything seen here: the host-to-device copies (7.1 s of copy time at 128K, 9.8 s at 250K)
have **zero overlap** with `k_attn_wmma` in the traces, so the PCIe link is free during attention and the kernel does not
compete with expert copies. (They also overlap nothing else of note; whether that is a lever of its own is outside this issue.)

Host KV against VRAM KV: `measurements/2026-10-28-long-context.md` ran 128K with `IE_KV_HOST=0`: 1,222 tok/s with 48 expert
slots against 1,119-1,136 tok/s with KV in RAM and 84-88 slots (one run each, before #71): **the host path costs at least
~10 % of the whole prefill, i.e. about 12 s of the 24 s of attention at 128K** [estimate from that comparison; the VRAM run
had fewer expert slots, so the true penalty is larger]. No per-kernel time with `IE_KV_HOST=0` exists yet.

---

## 2. Ceiling (question 1, theory)

### 2.1 FLOPs [estimate]

Per token-layer: Q K^T = 2 x 2,051 x 24 heads x 256 = 25.2 MFLOP, P V the same: **50.4 MFLOP useful**. Executed by the kernel:
64 WMMA (16x16x16, 8,192 FLOP) per block-iteration x 65 iterations x 2 blocks = **68.2 MFLOP** (padding 16/12 and the 3-cell tail).
At the measured 93 TFLOPS f16 peak: 0.54 us useful, **0.73 us executed**; at the sustained 2.6-2.8 GHz clock under the 182 W limit
(reference platform: 93 TFLOPS is at ~3.1 GHz): 0.8-0.9 us. Over all token-layers: 128K 1.57 M x 0.73 us = 1.1 s (1.4 s sustained),
250K 3.0 M x 0.73 us = 2.2 s. The dequantization (K and V to f16, ~200 VALU instructions per thread per iteration, [estimate] from the
source) costs as much as the WMMA again: together ~2.4 us per token-layer for this kernel design if every operand were in LDS
(estimate: 8 waves x ~300 VALU instr per block-iteration on 2 SIMDs per CU, plus the WMMA time).

### 2.2 Bytes [estimate]

Gathered per token-layer without reuse: 2,051 x 1,152 B = **2.36 MB**. Time at: L2 (~3.6-4 TB/s) 0.6 us; Infinity Cache (1.18 TB/s)
2.0 us; **VRAM (315 GB/s) 7.5 us**; **PCIe (55.7 GB/s) 42 us**. Chunk-wide, the unique bytes are far fewer than the gathered ones:
the KV of the whole prefix is 100K x 1,152 B = 115 MB per layer at position 100K, while a chunk of 6,144 tokens gathers
6,144 x 2.36 MB = 14.5 GB per layer, a **126x** potential reuse if the data of a chunk could be arranged to be loaded once
(this is what a KV-outer kernel does; see 4.1).

### 2.3 Roofline and what limits the kernel [estimate]

- Arithmetic intensity: 12 rows x 1,024 FLOP per cell and head / 576 B = **21 FLOP/B**. Ridge point 93e12 / 315e9 = 295 FLOP/B (VRAM),
  1,670 FLOP/B (PCIe). So at no reuse the ceiling is 21 x 315 GB/s = 6.6 TFLOPS from VRAM and 1.2 TFLOPS from host RAM.
- Measured 3.9-5.5 TFLOPS (4.5 TFLOPS executed = 3.9 useful at 128K). Consumed bytes: 2.36 MB / 15.3 us = **154 GB/s** at 128K
  (137 GB/s at 250K, 115 GB/s in the 20 us region; 397 GB/s near 6 us). With the KV in host memory (above 1 GiB total [research/29])
  the link delivers 55.7 GB/s at most, so **>= 52-64 % of the gathered bytes come from L2** (reuse factor >= 2.1-2.8x), assuming the link is
  saturated at all. Reuse by concurrently running neighbours is therefore already large; it falls as the position grows (the
  selections of neighbours diverge or the working set of 32 simultaneous tokens leaves the 4 MiB L2), which fits the growth of the
  curve. This is a reading of the numbers, not a measurement.
- **Where does the time go in the kernel?** Per block, 64 blocks are in flight (32 CUs x 2). Block duration = 32 x time per token-layer:
  554 us at 17.3 us, i.e. **8.5 us per iteration** (65 iterations), 2.9 us per iteration at 6 us, 10 us at 20 us. Arithmetic per
  iteration is 0.6-0.9 us [estimate above]. The rest is waiting. Three dependent global round trips per iteration (index, K, V)
  explain 3 us if each is ~1 us (L2 / VRAM) and 8-10 us if each is 2.5-3 us (host memory over PCIe); a PCIe round trip of 1.5-3 us
  is typical but **not measured on this machine** (section 6, M4). Little's law does not contradict: bytes in flight per block-iteration
  are 8 KB (K) or 8 KB (V), x 64 blocks = 512 KB, far above the 111 KB needed to fill 55.7 GB/s at 2 us, so the link could be saturated
  by *independent* loads; the chain just does not keep them in flight continuously (K and V are never in flight together, nothing
  is loaded during the compute). **Latency-bound with a bandwidth cap behind it** is the working hypothesis; it must be separated
  from a plain PCIe cap by the measurements M1/M2 below.
- Lower bound for the kernel in any design: max(compute 1-2.4 us, unique bytes / bandwidth). With reuse r the gather per token-layer is
  42 us / r (host) or 7.5 us / r (VRAM): r = 10 gives 4.2 us (host), r = 20 gives 2.1 us. Anything below ~3 us needs both
  deep reuse and a latency-free pipeline.

---

## 3. What the code does that costs time [code + estimate]

| Item | What | Cost / remark |
|---|---|---|
| Dependent loads | `cells[c0 + cc]` -> row pointer -> K bytes + scale floats (different cache line) -> (S, sync) -> V bytes (`fa_stage_v` loads only now) | 3 serial round trips per iteration; nothing prefetched |
| Occupancy | LDS 30,208 B and VGPR 176 -> 2 blocks (16 waves) per CU | low latency hiding; `KV` buffer 20.5 KB + `part` 8 KB + `Ps` 1.3 KB |
| 4 `__syncthreads` per iteration | stage K, S, softmax, P V | waves wait for the slowest load; at 8 waves per block each sync stalls on the slowest lane's load |
| V transposed store | 32 `ds_write_b16` per thread per iteration (8,192 per block) | many LDS instructions; bank pattern chosen by hand (comment in code) |
| Dequant redundancy | each of the 2,051 cells is dequantized per token although neighbours share most cells | VALU work equal to the WMMA work; the union (4.2) removes it |
| Q fill 75 % | 12 of 16 rows | -25 % WMMA efficiency; only matters once the kernel is near the compute floor |
| Tail | last iteration holds 3 cells but costs a full iteration (1.5 %) | negligible |

Not an issue: split-K / Flash-Decoding-style splitting. The grid has 12,288 (128K) or 8,192 (250K) blocks per call against 64 slots, so
the SIMDs are fully fed; there is no under-filled grid to repair (it would be for the last, short chunk only: 902,656 threads = 3,526
blocks, still 55 rounds).

---

## 4. State of the art (question 2) and RDNA4 (question 3)

### 4.1 Sparse / top-k attention prefill kernels

| Work | What | Numbers | Relevance for us |
|---|---|---|---|
| FlashMLA `flash_mla_sparse_fwd` (DSA, MLA 512-wide) [docs: README via `gh`] | token-level sparse prefill, indices `[s_q, h_kv, topk]`, up to 1,350 TFLOPS on B200 (CUDA 13.3), 410 TFLOPS = 95 % of peak on Ascend 950; older figure 640 TFLOPS on H800 (2025-09-29) | MLA absorbs the heads: 128 q heads share one KV (MQA-like), so each gathered cell serves 128 rows; arithmetic intensity is ~10x ours | no use for 24/2 GQA; shows what reuse buys: 95 % of peak |
| NSA, arXiv 2502.11089, Triton kernel [abstract + search] | per GQA group: the group's queries and the shared block list loaded together; outer loop over query tokens, inner over selected KV blocks | batching only pays with many heads per group | our group is 12 heads/KV head; this is the structure we have already |
| FSA (Flash Sparse Attention), arXiv 2508.18224 [HTML read] | reverses the loops: outer over KV blocks, inner over batches of query tokens that selected the block; no padding of the head dimension; partial results in a buffer plus a reduction kernel (no atomics); precomputed softmax statistics | kernel up to 3.5x (1.6x average) vs NSA, prefill up to 1.36x (1.11x avg) on H20/H200 | the "KV-outer" family; avoids padding; partial-sum merge changes rounding |
| MiniMax Sparse Attention, arXiv 2606.13392 [HTML read, first 100K characters] | KV-outer persistent kernel over (kv_block, kv_head), reverse index finds the queries, queries loaded via TMA, hot blocks split in chunks of <= 2 k B_k queries with a preassigned output slot, combine kernel merges with LSE; with G = 16 it packs ceil(128/G) query positions x G heads into a 128-row MMA; B_k = 128, k = 16 (2,048 tokens per query); index top-k with per-lane min-heaps | 1M context: 14.2x prefill, 7.6x decode vs GQA; top-k 779 us vs 3,970 us `torch.topk` (H800, N = 128K) | block size 128 makes KV-outer efficient; **ours is 4 tokens per block** (pool 4:1), so KV-outer would see 4-cell tiles. The 'pack ceil(128/G) positions' trick is our T-union with 12 heads: T = 4 -> 48 rows |
| aiter #6576 (open; opened 2026-10-15, ready 2026-10-22; **state 2026-10-29: OPEN, review decision empty, updated 2026-10-29**) [PR page; status via `gh`] | FlyDSL bf16 D256 QSA prefill; planner chooses per query tile: **union kernel** (rows share the union of blocks, exact per-query masks) or direct kernel | MI308X: 1,024 tokens 88 vs 283 us, 16,384 tokens 4,240 vs 13,906 us, 32,768 tokens 9,932 vs 28,948 us; chunked prefill 31K-132K context 2.3-7.4x | the only published AMD design for exactly this layer; not gfx12; numbers vs a Triton kernel, not vs ours |
| QUILT, arXiv 2610.11134 [HTML read] | neighbour query tokens share the intersection of their token-level top-K sets (group of 8 by default, cascaded: shared entries loaded once, per-query residuals); shared tails padded to a full tile if >= 50 % full (tau), padding masked, rest cascades down; only the final clipping step is approximate | neighbour overlap (group size 2): > 85.9 % Qasper, > 78.1 % GovReport over all layers, > 90 % in layer 0 (K = 2,048); kernel latency -17 to -55 %, processed KV entries -30 to -56 %, TTFT up to -36.8 %; naive set intersection 22-34x *slower* than baseline; **Ascend 910C, not GPUs** | direct evidence that neighbours overlap 80-90 % in DSA; the cost of building the sharing structure can exceed the gain (22-34x slower naive) |
| CompactAttention, arXiv 2605.16839 [HTML read] | union over query blocks and heads of a group -> per-group KV block table, KV read in place; exact over the selected pages | 128K: attention 2.74-2.79x, end-to-end 1.75-1.98x (RTX PRO 6000 / H200); keep ratio 0.098 -> 0.182 after union over a 1,024-token chunk -> 0.303 after intra-group union (RULER -0.5 to -2.4 points, due to the mask pattern not the union) | the union costs 1.9x more cells at chunk size 1,024; for T = 4 much less (QUILT overlap) |
| SpecSA, arXiv 2605.19893 [HTML read] | adjacent queries into one thread block; exact variant masks with -inf per query; approximate variant applies one representative's set | block-set overlap of adjacent queries 50-90 % (8K context, decode verification); exact grouping C = 2: up to 1.13x on H100; approximate C = 4: 1.22x | confirms the exact-mask variant gives small gains alone (1.13x); big gains come from the reuse layers |
| Gist-token sparse attention, arXiv 2604.20920 [search result only] | union per KV group, K/V tile staged once in shared memory | no numbers read | same idea |
| LongCat-Flash (research/29) | `k_attn` not covered | | |

**Token sharing: yes, and the literature says overlap is high** (QUILT 78-90 % pairwise, SpecSA 50-90 %), but **no number exists for
Qwen3.8's indexer** (pool of 4 tokens, 512 blocks). Mind the difference between *pairwise* overlap and the *union size* u(T) of T tokens;
only u(T) decides the cost (section 5, idea 3, measurement M3). Block granularity matters: DSA models select single tokens (K = 2,048),
ours selects 512 blocks of 4 consecutive tokens, so each selected block already brings 4 cells with it.

### 4.2 Gather strategy (loading)

- **Group-wise loading / union:** above.
- **Async copy to LDS.** NVIDIA has `cp.async`/TMA. For AMD, LLVM's DMA documentation lists LDS DMA (`global_load ... lds`,
  `buffer_load ... lds`, sizes 1/2/4 B and 12/16 B on gfx950) **only for GFX9 and GFX1250** [docs: llvm.org AMDGPUDMAOperations];
  gfx12 (gfx1200) is not mentioned, so **no TMA-like async copy on gfx1200** [unconfirmed beyond that page; check with a one-line
  inline-asm / `-S` test before relying on it]. Substitute: prefetch into **registers** (loads issued one iteration earlier; waitcnt
  only at use). Our K/V raw bytes are 64 B per thread per iteration = 16 VGPR: affordable (176 now; the compiler may spill beyond ~255).
- **`global_load_tr`**: gfx12 has transposing global loads (`global_load_tr_b64/b128`) [PR page: MLIR/LLVM commit, which is
  inconsistent between "gfx1200+" and "RDNA4 (gfx1250+)"; element widths: 16-bit named, 8-bit not mentioned]. Our V is int8 with
  per-32-dimension scales, so the transpose must happen after dequantization; no use unless V is stored as f16/f8 and the transposed
  fragment is loaded directly. [unconfirmed]
- **Software pipelining with `s_waitcnt` control** is the usual AMD substitute; the compiler schedules loads inside the loop body only,
  so the next iteration's loads have to be written explicitly.

### 4.3 fp8/int8 attention, quality

- SageAttention (arXiv 2410.02367), SageAttention2 (arXiv 2411.10958 [abstract read]): Q and K to INT4 (Q smoothed; 2-8b variant INT8), P and V to FP8 with a two-level accumulation; ~3x FlashAttention2 on RTX 4090,
  Hopper equal to FlashAttention-3 FP8 "with much higher accuracy"; claims negligible end-to-end loss in language, image, video models.
  SageAttention3 (arXiv 2505.11594 [search result]): FP4 microscaling, 1,038 TOPS on RTX 5090 (~5x FlashAttention); the maintainers recommend
  SageAttention2 for precision-sensitive use. Dense attention, diffusion-heavy evidence.
- **On gfx1201 (RDNA4), aiter**: #6390 (open; 2026-10-10) INT8 QK + FP8 PV, head dim 128, non-causal: **150-179 TFLOPS** (S = 4,096-118,808, Radeon AI PRO
  R9700, 64 CUs, 300 W) against BF16 SDPA 67-70 TFLOPS and Triton `fav3_sage` 14-19 TFLOPS; accuracy vs FP32: cosine 0.9992-0.9993, relative L2
  0.038-0.039 on random inputs; on real MiniMax-H3 tensors relative L2 0.004-0.014 for layers 0-40 and 0.07-0.19 for layers 45/49, against 0.0017-0.0023
  for BF16 [PR page]. #6627 (open): INT8 K, FP8 V, BF16 Q quantized in-kernel, hand-written code object, **277.7 TOPS** at S = 114,660, H = 28, D = 128, 676.6 ms;
  shares one INT8 scale over 64 key rows, `rel_l2` ratio 1.07 against the 32-row variant [PR page]. #5787 (open): CK fmha with LLC-sized head groups,
  137.3 TFLOPS fp8bf16 at S = 114,660 H = 28 (1.73x vs one launch; bitwise identical output) [PR page]. #4188 (open): FlyDSL gfx1201 BF16 and FP8 attention
  (no numbers on the page read). R9700 has twice our CU count (32), so absolute TFLOPS do not carry over; the *ratios* (INT8/FP8 path 2.2-2.6x BF16) do.
- FlashAttention-3 FP8 (block quantization + incoherent processing) [from memory of the paper, not re-read; not applicable on gfx1200 anyway].
- **Relevance for us:** our KV is already int8 with per-(cell, 32-dim) scales; the Q side would be the new rounding. The quality bar here is not
  diffusion PSNR but code at 250K; any change must pass the NLL/KL check (research/31). Gain exists only after the latency is fixed (section 5).

### 4.4 RDNA4 / gfx1200 facts

| Item | Fact | Source |
|---|---|---|
| WMMA | 16x16x16 only on gfx12 ("MMA_M must be 16"); same unified layout for f16, INT8, INT4 (8 contiguous elements per lane, K-major A/B, D M-major); intrinsic `..._w32_gfx12`; wave32 | GPUOpen WMMA guide part 1 [docs read] |
| Throughput | f16 93, bf16 97, fp8 186 TFLOPS, int8 183, int4 341 TOPS, FMA f32 24.3 (measured, 32 CUs, ~3.1 GHz, 174 W) | reference-platform.md |
| LDS | 64 KiB per workgroup on gfx1200/1201 (Triton: "Required: 65792, Hardware limit: 65536") | aiter #4385 [PR page] |
| Async copy | no TMA; LDS DMA documented only for GFX9 and GFX1250 | LLVM docs [see 4.2] |
| Transposing loads | `global_load_tr_b64/b128` introduced with gfx12 | LLVM commit [see 4.2] |
| Community FA | llama.cpp: the WMMA fused-GEMM technique of the GPUOpen guide "has been deployed in llama.cpp to implement Flash Attention on RDNA 4" [docs]; PR #28102 (merged 2026-09-11) "CUDA/HIP: Flash Attention tuning (gfx1201)" for head size 256 on R9700: pp512 at depth 40,000 426 -> 639 t/s, at depth 150,000 164 -> 399 t/s (Qwen 27B IQ4_XS; *prefill* t/s of the whole model) [PR page]; open: #31323 (Vulkan FA for RDNA3/4), #28507 (Vulkan FA shmem staging on RDNA) | |
| CK / aiter | CK fmha has gfx12 instances (aiter #5787 measured them on gfx1201); aiter's C++/ASM paths need patches for RDNA4, Triton works after arch-allowlist patches [search result; aiter #4535, #4385 open]. Triton on RDNA4: `fav3_sage` untuned (14-19 TFLOPS) | PR pages |

No source gives a TFLOPS figure for *sparse gather* attention on gfx12. The best dense figures on gfx1201 are 67-70 TFLOPS (BF16 SDPA), 137 (CK fp8bf16)
and 150-278 (INT8/FP8 Sage), on a card with twice our CUs.

---

## 5. Ideas for our kernel (question 4)

Expected gains are for `k_attn_wmma` per token-layer (today 15-17 us on average) and then converted into prefill time:
128K 24.0 s of 92.7 s (TTFT 93.3 s, 1,401 tok/s), 250K 51.9 s of 191.4 s (TTFT 194.2 s, 1,286 tok/s). **All gains are estimates, none measured.**

| # | Idea | Expected effect (calculation) | Bit-identical? | Effort | Risk |
|---:|---|---|---|---|---|
| 1a | **Load V together with K** (move the V global load from `fa_stage_v` into `fa_stage_k`; V raw bytes held in 8 VGPR until after S). One-line experiment | removes one of three dependent round trips: iteration 8.5 us -> ~5.7 us at 17 us/token-layer: **-30 %** (if round trips are equal) = 128K -7 s (-7.5 % of prefill), 250K -15 s (-8 %) | **yes** (same values, same order) | low (2 h) | register pressure (+8 VGPR); compiler may reorder |
| 1b | **Software pipeline of the gather**: before the compute of iteration i, issue the cell indices for i+2 and the K and V rows of i+1 into registers (16 VGPR); wait only at use | chain of 3 round trips collapses to ~1 hidden behind 0.6-0.9 us compute: iteration -> max(compute, bytes) ~ 2-3 us => per token-layer **~6-8 us (-50 to -60 %)**: 128K -12 to -14 s, 250K -26 to -31 s; prefill 128K 1,401 -> ~1,600-1,650 tok/s, 250K 1,286 -> ~1,500-1,560 tok/s [all estimate] | **yes** | medium (1 day; spills must be checked with `kernel-regs.py`) | VGPR 176 -> ~195; if the real limit is PCIe bandwidth, the gain stops where reuse stops (M1/M2 decide) |
| 2 | **Token union, T = 4** (48 rows = 3 full WMMA tiles instead of 75 % fill; the union of the cell sets of 4 consecutive tokens, ascending; per-token mask bits; K/V staged and dequantized once per tile) | per token: gather and dequant x u/T (u = |union|/|set|: 1.3 -> x0.33; 2.0 -> x0.5; 3.0 -> x0.75); WMMA x0.75u of today (0.98 / 1.5 / 2.25: 0.7-1.7 us, small). If gather-bound: 17 us -> 6-9 us on its own; with 1b the floor moves to the unique-byte limit. **Break-even u ~ 3.5 at T = 4** | **no**: each row sees its cells in different 32-groups, so the running max and the f16 rounding of P change (masked cells give exactly 0). Numeric ~1e-3 relative; needs `docs/backends/README.md#checks` | high (new union builder op next to `select_blocks`, bitmask format, LDS budget: `part` x3 -> remove it by idea 5) | depends on u(T); QUILT's 78-90 % pairwise overlap suggests u(4) ~ 1.3-2 [estimate], unknown for our indexer; the set-building can eat the gain (QUILT: naive intersection 22-34x slower) |
| 3 | **Union vs separate** | separate (today) keeps exact numbers and per-token control; union needs the statistics M3: choose union only if u(4) < 2.5 and gather-bound | | | |
| 4 | **KV in VRAM up to ~128K (`IE_KV_HOST=0`)**, automatic when the budget allows (KV = 12 x N x 1,152 B: 0.9 GB at 64K, 1.8 GB at 128K, 3.6 GB at 250K) | removes PCIe from the chain: round trips ~2.5 us -> ~1 us; at least the ~10 % of the whole prefill (12 s at 128K) seen in the one A/B, measured *with fewer expert slots*; after 1b the difference shrinks | **yes** | trivial for the switch; the policy against expert slots (VRAM 16 GB, desktop 1.4 GB) needs thought | costs expert slots (48 vs 84-88 at 128K) - which then costs MoE streaming time; net effect is what was measured (+10 %) |
| 5 | **Fewer syncs and no `part` buffer**: each wave computes S for one 16-cell tile over all 256 dims as 4 separate accumulator chains (one per 64-dim quarter), then adds them in the same order ((s0+s1)+s2)+s3 -> **same bits**; 128-cell super-iterations for S with softmax and P V still per 32 cells in the same order | frees 8 KB LDS (30.2 -> 22 KB); with VGPR <= 128 a third block per CU (+50 % latency hiding); syncs 4 -> ~2.5 per 32 cells. Est. -10 to -20 % on top of 1b | **yes** if the order of operations is kept (needs a test: tokens identical under `IE_DETERMINISTIC=1`) | high (LDS layout: K for 128 cells as f16 does not fit 64 KB; needs int8 K in LDS and dequant when loading the B fragment: same ops per element, same bits) | VGPR budget; only worthwhile after 1b |
| 6 | **INT8 WMMA for QK** (K stays int8: no dequant of K, Q to int8 per row and 32-dim group; scale per lane after the accumulator since the cell is the N index) | removes ~1/3 of the VALU work and half the QK WMMA time; max ~0.4 us of 15; gain only at the floor | **no** (Q rounding) | medium-high | quality (research/31 bar); aiter #6390 shows INT8 QK/FP8 PV works on gfx1201 for diffusion (cos 0.9992), not for code at 250K |
| 7 | **Overlap with other kernels (streams)**: attention of chunk c while projections, scoring and top-k of chunk c+1 run (legal: only KV rows of c+1 and the earlier pool are needed) | attention uses ~5 % of WMMA and is latency-bound, so the CUs have room, but both want the same CU slots (2 blocks per CU) and the phase buffers (`Carve p{B.phase}`) would need a second copy within the loan limit (3.9-4.2 GB). Gain 0-25 % of attention [guess] | **yes** | medium-high | memory; with KV in host memory the gather (55.7 GB/s) competes with the expert copies if those are overlapped too (today there is zero overlap with attention) |
| 8 | **Q-tile fill 75 % -> 100 %** without union: not possible per token (12 heads); only via T tokens (idea 2) or by processing two KV heads in one block (24 rows = 1.5 tiles, 75 % again) | at the compute floor only (0.73 -> 0.55 us) | | | not before the latency problem is solved |
| 9 | **KV-outer / FSA-style** (outer loop over KV blocks, inner over the queries of a chunk that selected the block) | unique bytes 126x lower than gathered (2.2), compute unchanged: the only design that approaches the compute floor (~1-2 us) | no | very high (reverse index, partial-result buffers + LSE merge kernel, 4-cell blocks make tiles tiny: needs 8-32 blocks grouped, i.e. masks) | block size 4 is hostile to KV-outer (MiniMax uses 128, NSA 64); revisit only if idea 2 shows high u-sharing |

Not recommended: split-K / flash-decoding style split of the 65 iterations (grid already oversubscribed), TileLang/Triton port (no
gfx12 sparse example found; our kernel is already hand-written WMMA), fp8 P (additional rounding of the probabilities, no gain until the
floor).

**The three most promising**
1. **1a + 1b (prefetch / earlier V)**: bit-identical, -30 % then -50 to -60 % of the kernel, i.e. about -7 / -13 s at 128K and -15 / -28 s at 250K.
2. **Token union T = 4 (idea 2)**: numeric; potentially the same size of gain as 1b and the only way to push below it (reuse); decide after M3.
3. **`IE_KV_HOST=0` as automatic policy up to the budget (idea 4)**: bit-identical, almost free, measured +10 % total at 128K in a confounded A/B; its benefit shrinks if 1b lands, but it is the control experiment for the whole latency theory.

---

## 6. What to measure first (all need the GPU; none of it was done)

Rules as in research/29 section 6: idle desktop, 3 runs, one file per run in `research/measurements/`, no build during a measurement.

| # | Measurement | Decides |
|---|---|---|
| M1 | `rocprofv3 --kernel-trace` with `--kernel-include-regex k_attn_wmma` at 128K plus counters (`--pmc`): `TCP_*` / `TCC_HIT`, `TCC_MISS`, `TCC_EA0_RDREQ` (host vs VRAM if separable), `SQ_WAVE_CYCLES`, `SQ_BUSY_CYCLES`, `SQ_INSTS_VALU`, `SQ_INSTS_LDS`, bank conflicts (`SQ_LDS_BANK_CONFLICT`), `SQ_WAIT_INST_LDS`, `SQ_INST_LEVEL_VMEM` (latency per request) [counter names unconfirmed on gfx1200; list with `rocprofv3 -L`]. For calls at positions ~5K, ~60K, ~120K | latency vs bandwidth vs LDS; L2 hit rate vs position (reuse r) |
| M2 | A/B `IE_KV_HOST=0` vs `1` at 64K and 128K with **the same number of expert slots**, `k_attn_wmma` per-call time from the rocprof trace (the analysis here can be re-used: `Grid_Size_X/256/2` tokens, call order -> position) | the pure PCIe/host penalty on the kernel (today only an indirect total) |
| M3 | Selection statistics from a debug dump of `cells` (as in research/29 step 6): per layer pairwise overlap `|S_t ∩ S_t+1| / 2,051`, **union size u(T) = |∪| / |S| for T = 2, 4, 8, 16, 32 consecutive tokens**, the same over a chunk (unique blocks vs 512 x tokens), and the unique KV bytes of a window of 32 tokens | idea 2 and 9: u(4) < 2.5? the actual reuse available; whether the observed r >= 2.1-2.8 is explained by L2 |
| M4 | Throwaway micro-kernel (not engine code): one block-iteration of this kernel (a) with K/V already in LDS (compute + LDS floor), (b) loading from VRAM, (c) loading from the mapped host pointer, with 64 blocks in flight; plus a pointer-chase latency test VRAM vs host over PCIe | iteration cost split: 0.6-0.9 us compute vs round trip; confirms or kills the latency chain hypothesis in one afternoon |
| M5 | Experiment 1a (V load moved) as a throwaway build: `k_attn_wmma` per call at 128K, bit-compare tokens with `IE_DETERMINISTIC=1` | cheapest test of the theory; if -30 % shows, 1b is justified |
| M6 | Position curve of M1/M2 for 250K once at the end (one 200 s run) | does the ~20 us plateau continue or does it rise again with the pool? |

Decision rule: if M4(c) is ~8-10 us per iteration and M4(a) < 1 us -> latency chain: 1a, 1b. If M2 shows the host penalty >= 50 % of the
kernel -> idea 4 first as policy, then 1b. If M3 gives u(4) <= 2 and M1 shows L2 hit rate falling with position -> idea 2 (with 5 for LDS).
If M1 shows LDS bank conflicts or VALU-bound at 5K -> idea 5/6 before union.

---

## 7. Open points / not clarified

- All gains in section 5 are estimates. The latency-chain theory is consistent with the numbers (section 2.3) but unproven; a PCIe cap with
  high L2 reuse produces similar numbers.
- Whether the Qwen3.8 indexer's selections overlap like those of DSA models (QUILT) is not known; QUILT ran on Ascend NPUs only.
- The 22 %-27 % shares: the 64K trace is from before #71 and includes a 16K prompt in the same file (separated by call count: 48 calls = 1.32 s).
- gfx1200 LDS DMA and `global_load_tr` width support: read only from LLVM documentation pages that do not name gfx1200; check with a compile test.
- aiter #6576 was opened a day after its sibling PRs and is still open; the K1-K3 planner code was not read (as in research/29).
- FlashAttention-3 FP8 and FlashInfer sparse/paged were not re-read for this note; no FlashInfer or TileLang kernel for 24/2 GQA, D256, top-k blocks of 4 was found.
- The per-position table is derived from the launch order of the calls; it assumes the group-major / layer / chunk order printed in `server.log`
  (confirmed by monotone growth and by the agreement of the two runs, not by a position field in the trace).

## Sources

- https://github.com/ROCm/aiter/pull/6576 (open), /pull/6390 (open), /pull/6627 (open), /pull/5787 (open), /pull/4188 (open), /pull/4385 (open)
- https://github.com/ggml-org/llama.cpp/pull/28102 (merged 2026-09-11), /pull/31323, /pull/28507 (open)
- https://github.com/deepseek-ai/FlashMLA (README read via `gh`)
- https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/
- https://llvm.org/docs/AMDGPUDMAOperations.html, https://llvm.googlesource.com/llvm-project/+/bda0016aa2b43f0817dddbd790ce4abd3df22e99
- https://arxiv.org/abs/2502.11089 (NSA), https://arxiv.org/html/2508.18224v2 (FSA), https://arxiv.org/html/2606.13392v1 (MiniMax Sparse Attention)
- https://arxiv.org/html/2610.11134 (QUILT), https://arxiv.org/html/2605.16839 (CompactAttention), https://arxiv.org/html/2605.19893v1 (SpecSA), https://arxiv.org/pdf/2604.20920 (gist tokens, search result)
- https://arxiv.org/abs/2411.10958 (SageAttention2), https://arxiv.org/html/2505.11594v3 (SageAttention3, search result)
- Own: `/var/tmp/claude/{w71/prof-p128-w71, w73/prof-p250-main, ctx-curve/prof-p64}/trace_*.csv` (not in the repository)
