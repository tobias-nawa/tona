# 29 — Prefill at 64K–256K: sparse-attention indexer, top-k and KV gather

Agent research (Sonnet), 2026-10-28

Method: web search and page retrieval (arXiv abstracts and HTML, GitHub PR pages, vendor docs/blogs) plus a read of our own
prefill code (`engine/gpu/prefill_qsa.hip`, `prefill_attn.hip`, `prefill_run.h`, `prefill_dev.h`, `prefill_layer.cpp`,
`prefill_entry.cpp`, `mega_build_bufs.cpp`). Nothing was measured or built here. Marking per entry:
- **[code]**: source read directly (our repo).
- **[PR page]**: PR description/review read on GitHub (numbers are the author's, not re-measured by us).
- **[docs]**: vendor docs/blog/README read.
- **[abstract]**: abstract page only (or HTML text of a paper, stated per entry).
- **[snippet]**: only a search-result snippet seen.
- **[estimate]**: own calculation from the figures given, not measured.
- **[unconfirmed]**: secondary source or not findable.

Rates are always named: **prefill** (prompt tokens per s) or **decode** (generated tokens per s).

Our shape: QSA layers: 12 of 48; indexer 4 heads x 128, one pooled key per 4 tokens, score = sum over heads of
ReLU(q.k)/sqrt(128); top-512 pool blocks (= 2,048 cells) plus up to 3 tail cells; GQA 24/2, head dim 256; KV int8, row
1,152 B per position and layer (`kv_row_`) in pinned host RAM above 1 GiB total (`IE_KV_HOST`, `hipHostMallocMapped |
NonCoherent`); pool f32 in VRAM. Platform: 32 CUs, 4 MiB L2, 32 MiB Infinity Cache (IC), 315 GB/s VRAM, PCIe 55.7 GB/s per direction
(`docs/backends/reference-platform.md`).

---

## 0. Summary

1. **Nobody has published a prefill breakdown for QSA at 64K–256K.** Not found: any engine number for QSA prefill
   throughput at these lengths (SGLang's Qwen3.8 blog has none [docs]; aiter PRs report kernel microbenchmarks on MI308X/MI355X only).
   Closest data: LongCat-Flash-Lite (DSA, flat token-level indexer): indexer = ~67 / 80 / 89 % of DSA time in prefill at
   64K / 128K / 256K [abstract-HTML, derived from their Table 3, qlen 2048] (section 1.3). Our indexer is 4x smaller (pooled),
   so the share is lower, but the trend is the same: the indexer is the quadratic part.
2. **Our scoring FLOPs are not the problem; the memory and latency structure around them probably is** [estimate]. Scoring
   needs 512 N^2 FLOP per layer (4 heads x 128 x 2 x N^2/8): 3.2e13 per layer and 3.8e14 over 12 layers at N = 250K, i.e.
   4 s at the 93 TFLOPS peak, ~16 s at 25 %. The measured excess over the 64K rate is ~277 s (section 2). So a faster MMA
   format (fp8/int8) cannot be the main lever.
3. **The measured curve has a cliff between 64K and 128K, then grows ~quadratically** (section 2). That fits a cache-capacity
   effect (per-tile score block of 32 tokens exceeds the 4 MiB L2 at ~128K; f32 pool exceeds the 32 MiB IC at ~250K),
   combined with the structure of `k_topk` (32 one-bit radix passes with block sync and atomic each, strided per-thread
   slices). This is a **hypothesis**, not measured. Two other candidates (PCIe gather, plan change with context) are listed.
4. **Best candidates** (section 5): rewrite `k_topk` as a multi-bit radix (bit-exact), shrink/relocate the score working set
   (f16 pool shadow, smaller tile; bit-exact), avoid materializing scores via histogram + recompute (bit-exact,
   StreamIndex-like), overlap top-k with the next scoring tile (bit-exact), union gather for the KV (aiter #6576 design;
   numerically near-identical, needs a check).

---

## 1. How others handle the prefill of DSA/QSA-style sparse attention (question 1)

### 1.1 AMD: aiter / ATOM

| Item | State | What it is | Numbers |
|---|---|---|---|
| ROCm/aiter #6515 | merged 2026-10-26 [PR page] | FlyDSL `qsa_paged_mqa_logits_flydsl`: MFMA scorer, bf16 Q/K only, 4 index heads, head dim 128, per-head ReLU, compress ratio 4, 16 compressed rows per page. ATOM uses it for **prefill-sized scoring (> 256 query rows)**. Writes logits per row's visible window (`visible_groups`, `row_starts`); where top-k runs is not stated | op test on gfx942 passes, error <= 2.4e-6; **no kernel speed numbers**; e2e TPOT (decode) 12.0-14.8 ms vs 14.3-14.6 ms stock, noise |
| ROCm/aiter #6469 | merged 2026-10-15 [PR page] | Triton/Gluon QSA kernels: `_qsa_paged_mqa_logits_kernel`, `_qsa_paged_mqa_logits_mfma_kernel`, `_expand_selected_groups`, `_qsa_sparse_paged_gqa_kernel`; `bench_qsa.py` in the PR | description fields empty, no numbers on the page; top-k method not described |
| ROCm/aiter #6576 | **open** (opened 2026-10-15, ready 2026-10-22, approved by one reviewer 2026-10-27, still waiting for more reviews) [PR page] | FlyDSL bf16 D256 **sparse attention for QSA prefill/extend**, replaces SGLang's Triton sparse GQA kernel. Input: per query row T four-token blocks (T = 256/512/768/1024) plus 0-3 tail tokens. Planner K1-K3 picks per query tile: **union kernel** (rows of a tile share the union of their blocks, exact per-query masks) or **direct kernel** (one query per wave) | MI308X, kernel incl. planning: 1,024 tokens 88 vs 283 us; 4,096 tokens 687 vs 2,743 us (TP2) up to 256 us (TP8); 16,384 tokens 4,240 vs 13,906 us; 32,768 tokens 9,932 vs 28,948 us; chunked prefill (1,024 queries, 31K-132K context) 2.3-7.4x faster. Indexer scoring and top-k not part of this PR (upstream) |
| ROCm/aiter #6675 | merged 2026-10-26 [PR page] | `mla_sparse_prefill_v41` for **MLA** (512-wide e4m3 rows with E8M0 scales, fp8 scaled MFMA for QK, bf16 PV), gfx950 | H = 64, MI355X: 1.93-2.11x vs bf16 kernel (N 1,024-8,192, topk 512-2,048); not QSA |
| ROCm/aiter #6670 | merged 2026-10-23 (from research/15) | decode only | see research/15 |

Take-away: the **union-of-blocks tile** in #6576 is the only published prefill answer for the attention part of QSA; the
scorer in #6515 is an MFMA GEMM with a dense logits output, and no AMD PR shows the top-k for long prefill. gfx12/gfx1200 is
in none of them (MI308X = gfx942, MI355X = gfx950).

### 1.2 NVIDIA/DeepSeek ecosystem

- **DeepGEMM** `fp8_fp4_mqa_logits` (non-paged, for prefill): fp8 (and on SM100 MXFP4) Q and K with scales, per-head bf16 weights, per-row KV start/end,
  dense logits output of width `max_seqlen_k`; top-k is done afterwards by the caller [docs]. **No MQA-logits TFLOPS in
  the README** (only the generic GEMM claim of up to 1,550 TFLOPS on H800) [docs]. Not found: per-kernel numbers.
- **TensorRT-LLM** (Tech Blog 15) [docs]: indexer MQA runs in MQA mode for prefill too; low-rank projections bf16, weight projection fp32, **indexer K cache and MQA math in blockwise FP8**, top-k in FP32.
  Top-k = radix select with 8-bit digits (4 passes: histogram, prefix sum, target digit, filter); below 2,048 candidates switch to parallel sort; PR-9255 bins on the leading 11 bits to halve the candidate set.
  B200, input [64, 9,295]: `torch.topk` 106.067 us vs custom `TopKPerRow` 14.327 us (7.41x). End-to-end 25-40 % (low latency), 14-24 % (throughput) from replacing torch.topk (ISL 8K/OSL 1K).
  Chunked prefill for DSA (PR-9376). Fast path: skip indexer and top-k for sequences <= 2,048 (all tokens selected anyway), 1.03x on 1K/1K. Long sequences (e.g. 32K) are listed as future work. Fused FP8 quantization into the indexer K-cache write: +32.64-64.20 % throughput (PR-8701).
- **vLLM #37421** (merged 2026-04-08) [PR page]: persistent TopK scheduler for DSA **decode**: per row dispatch: trivial (<= topk), <= 8,192: 2,048-bin fp16 histogram + fp32 radix refine, <= 64K: 256-bin, **> 64K: multi-CTA cooperative radix select**. Microbench batch 1: 163,840 tokens 54.10 -> 20.97 us (2.58x). The page has no prefill numbers.
- **Guess-Verify-Refine (GVR)**, arXiv 2604.22312 and TensorRT-LLM Tech Blog 21 [abstract + docs]: **exact** top-k for **decode** that reuses the previous step's top-k to guess the threshold, 1-2 global passes instead of 3-4; 1.88x average (up to 2.42x) over radix select; consecutive-step overlap on a 68,665-token prompt ~44 % in layers 20-60, ~1.5 % in layers 0-1. The blog states prefill is not applicable (no previous top-k) and falls back to radix. Production gate: >= 12,288 tokens.
- **FlashInfer** [docs, snippet]: `top_k`, `top_k_ragged_transform` (fuses top-k with an index offset for the second sparse-attention stage), `top_k_varlen` with backends `radix`, `gvr`, `radix_filter`. No FlashInfer-own speedup numbers found; a third-party kernel (UNIQUE, arXiv 2605.27740) claims up to 2.0x over `flashinfer.top_k` and 4.8x over `torch.topk` [snippet].
- **FlashMLA** `flash_mla_sparse_fwd` [docs]: sparse prefill attention only (indices given), up to 1,350 TFLOPS on B200 (CUDA 13.3); no indexer.
- **SGLang** Qwen3.8-Flash-Next blog (2026-08-26) [docs]: QSA prefill = custom index-score kernel, "fast top-k kernel" for 512 blocks, Triton expand + sparse GQA; **the indexer overlaps the QKV projection on a second CUDA stream**. No prefill numbers, no chunked prefill discussion.
- **llama.cpp** #29901, release b11405 (2026-10-05) [unconfirmed: secondary blog, rohitai.com]: tiled CUDA scoring kernel for 4-head/128 indexers, one key tile (64 keys) reused for 8 tokens, f16 keys, f32 accumulation; indexer op at kv = 65,536, batch 2,048: 16.4 -> 6.4 ms (4.76 ms later), on RTX PRO 6000; **prefill at 128K only +4.4 % (3,867 -> 4,038 tok/s)**, i.e. the indexer scoring was not the dominant prefill cost there.
- **TileLang** indexer/sparse-MLA prefill: not found beyond the line already in research/15.

### 1.3 Papers

| Paper | Claim | Exact? | Note for us |
|---|---|---|---|
| IndexCache, arXiv 2603.12201 [abstract] | drop 75 % of indexers, reuse top-k of the nearest "Full" layer; 30B DSA: prefill 1.82x, decode 1.48x; training-free greedy layer search or multi-layer distillation; "negligible quality loss" (no numbers on the abstract page) | no (changes selection) | 12 QSA layers 3 DeltaNet layers apart; model not trained for it |
| LongCat Sparse Attention, arXiv 2608.01662 [abstract + HTML first 100K chars] | profile (LongCat-Flash-Lite, K = 2,048): indexer share of DSA time **decode** 45 / 61 / 75 % at 64K / 128K / 256K; **prefill** (qlen 2,048) ~67 / 80 / 89 %. Hierarchical indexer (pages of 128, sub-block means of 8, top M = 1,024 pages, then token-level): prefill 0.79x at 128K (slower), 1.47x at 256K, 2.56x at 512K, 4.11x at 1M. Cross-layer indexing groups of N = 2 (N = 4 loses long-context accuracy); adjacent layers share ~57.4 % of top-k, keep ~93.2 % of attention mass; trained. Full-layer prefill speedup of their sparse attention + streaming indexer at 128K only 1.08x | no | hierarchical only pays from ~200K; our pooled index is already 4:1 |
| HISA, arXiv 2603.28458 [abstract + HTML] | training-free two-stage replacement for the token-level indexer (block size 128, 64 candidate blocks, then token level); A100, qlen 1,024, k = 2,048: 2.16x (4:1 candidate ratio) to 3.75x (8K fixed budget) at 64K; DeepSeek-V3.2 LongBench 50.78 vs 51.05 DSA; **no top-k recall reported**, quality only via NIAH/LongBench | no | selection changes; our indexer already pools 4 tokens |
| StreamIndex, arXiv 2605.02568 [abstract] | chunked partition-merge top-k, never materializes the full score tensor (CSA V4-shape full FP32 tensor would be ~256 GB at S = 65,536); H200, indexer step only: up to S = 1,048,576 with 6.21 GB; recall mean ~1.0, min >= 0.998 over sweeps; "matches the materialize baseline exactly" at small sizes; with TileLang attention at S = 262,144: 1.97 s, 18.56 GB | exact at small sizes, recall >= 0.998 in sweeps | we avoid materializing by 32-token tiles, but a tile still writes 32 x n_pool floats |
| DeepSeek-V4, arXiv 2606.19348 (research/papers.md) | indexer QK in FP4 with QAT, top-k 2x faster, recall 99.7 % | no | Qwen's indexer not trained for low precision |
| Self-Indexing Attention 2609.13205 (papers.md) | 1-bit index signs | no | low |

---

## 2. What our own numbers say (question 1, our side)

Measured yesterday (ie-server, MTP, one run per point; **prefill**):

| Context | prefill tok/s | time s (N / rate) | excess over the 64K rate (0.7236 ms/token) |
|---:|---:|---:|---:|
| 8K | 1,459 | 5.6 | |
| 16K | 1,457 | 11.2 | |
| 32K | 1,499 | 21.9 | |
| 64K | 1,382 | 47.4 | 0 (reference) |
| 128K | 808 | 162 | ~67 s |
| 250K | 545 | 458 | ~277 s |

[estimate] from these rows:
- Up to 64K the per-token cost is flat within ~8 %. Pure N^2 growth with the coefficient fitted at 250K (4.7e-3 s per K^2) would already cost ~19 s extra at 64K (prefill ~1,040 tok/s); it is not seen. So **the quadratic term switches on between 64K and 128K** (a cliff), after which excess(250K)/excess(128K) = 4.1 vs (250/128)^2 = 3.8. Consistent with a threshold effect, not a smooth FLOP term.
- Candidate thresholds [estimate]: per-tile score block `SCORE_TOK x n_pool x 4 B` = 2 MiB at 64K, 4.2 MB at 128K (~ L2 4 MiB), 8 MB at 250K; f32 pool per layer `n_pool x 512 B` = 8 MB at 64K, 16 MB at 128K, 32 MB at 250K (= IC 32 MiB).
- Also possible: a change of the prefill plan with context (chunk/group sizes; `choose_plan` in `prefill_entry.cpp` [code] searches chunk and group against the loan cap and falls back to smaller chunks); the experts are then streamed more often. `IE_PF_DEBUG=1` prints chunk/group per prefill. **Check before blaming the indexer.**

### 2.1 What the code does today [code]

- Scoring per layer and 32-token tile (`SCORE_TOK = 32` in `prefill_run.h`; `select_blocks` in `prefill_layer.cpp`): `k_idx_score_wmma` computes S = Q(128 rows = 32 tokens x 4 heads) x Pool^T(128 blocks) with `wmma_f32_16x16x16_f16` in two K steps; **Q and pool are converted f32 -> f16 on load** (`ix_load`), accumulate f32, ReLU-sum over the 4 heads in the epilogue, scale 1/sqrt(128), **one f32 score per (token, block) written to the global score buffer** `score[tt * sstride + pb]`. Tiles behind the last visible block exit early (causal triangle handled per tile). The scalar fallback `k_idx_score` is selected by `idx_scalar` or other head/dim shapes.
- Top-k `k_topk` per token (one block of 512 threads per token, grid = 32 blocks per launch -> at most one block per CU on 32 CUs): thread `t` owns the **contiguous** slice `[t * chunk, (t+1) * chunk)` of the n_pool scores (chunk = n_pool / 512, ~122 at 250K); threshold search `topk_threshold` = **32 binary passes** (bit 31 down to 0), each: full scan of the slice, three `__syncthreads`, one block `atomicAdd`; then a count pass (gt/eq), exclusive scans done by thread 0 over 512 entries (three of them), and an emit pass; tie rule = earlier blocks (`op_topk`). Output: 512 x 4 cell ids ascending + tail.
- Attention `k_attn_wmma`: block per (token, KV head); 12 q heads = rows of a 16-row tile (75 % fill); per 32 cells: K int8 -> f16 into LDS (+ scales), S via WMMA, online softmax, V transposed into the same buffer, O += PV. About 64 iterations of 4 `__syncthreads` per (token, kv head), gather of cells directly from the (host-pinned) KV through the mapped pointer; no union, no prefetch.
- Everything is per token: no sharing of selections between neighbouring tokens.

### 2.2 First-order cost model [estimate]

Quantities for one QSA layer at N = 250K, sum over tokens; average n_pool = N/8 = 31K:

| Quantity | Formula | Per layer | x 12 layers | Time at |
|---|---|---:|---:|---|
| Scoring FLOPs | 512 N^2 | 3.2e13 | 3.8e14 | 4.1 s at 93 TFLOPS peak |
| Pool reads for scoring (re-read per 32-token tile) | N/32 x avg n_pool x 512 B | 125 GB | 1.5 TB | 1.3 s (IC) .. 4.8 s (VRAM) |
| Score writes | 4 B x N^2/8 | 31 GB | 0.37 TB | 1.2 s (VRAM) |
| Top-k reads (~35 full scans of the slice) | 35 x 4 B x N^2/8 | 1.1 TB | 13 TB | 3.6 s (L2) .. 11 s (IC) .. 41 s (VRAM) |
| KV gather over PCIe (nominal, no reuse) | 2,051 cells x 1,152 B (one row holds both KV heads) = ~2.4 MB per token and layer | 0.6 TB | 7 TB (28 MB per token) | 130 s at 55 GB/s; 28 MB/token alone caps prefill at ~1,960 tok/s |

None of these ideal-traffic numbers reaches the ~277 s excess. Two readings: (a) the real cost is **inefficiency of the
access pattern**, not traffic: in `topk_threshold` each lane reads 4 B at a stride of `chunk` x 4 B = ~490 B, so one wave load
touches 32 cache lines; the 512 threads' lines (64 KB) exceed the vector L0 cache, and 32 passes x 3 syncs per token are latency-bound
on one block per CU (and 32 blocks cannot fill 32 CUs x 2+ waves worth of latency hiding); (b) the cliff comes from the score block leaving L2 so every pass hits IC/VRAM.
Neither is verified. The PCIe row is the largest nominal number but **linear** in N and present at 8K too, so by itself it does not explain a quadratic excess; it can still amplify it if
neighbouring tokens stop sharing cells in L2 at long context (the selected set is then less local than at 8K-64K, where the dense/near-dense
part is in the window).

---

## 3. Techniques with source and measured gain (question 2) and exactness/fit (question 3)

"Exact" = identical selected cells and identical numbers (can be tested with `IE_DETERMINISTIC=1`); "numeric" = same selection, different rounding; "changes" = different selection or approximation, quality check per `docs/backends/README.md#checks`.
gfx1200: the prefill kernels may use more than 32 KB LDS (limit is per workgroup 64 KB on RDNA4, to be confirmed with `hipDeviceProp`) [unconfirmed: only the task statement says more LDS is allowed in prefill].

| # | Technique | Source and measured gain | Exact? | Fits gfx1200? |
|---|---|---|---|---|
| T1 | **Multi-bit radix select** (8-11 bit digits, shared-memory histograms, 2-4 passes instead of 32 single-bit passes; coalesced/interleaved lane access; tie rule kept = earlier blocks) | TRT-LLM Blog 15: 8-bit radix, 4 passes, +11-bit binning step; 7.41x vs `torch.topk` (not vs a 32-pass kernel) [docs]; vLLM #37421 multi-CTA radix for > 64K: 2.58x at 163,840 [PR page] | **exact** if threshold + tie rule are preserved (scores are non-negative floats compared as integers: same threshold value) | yes: LDS atomics/histogram, wave32 ballots; 2,048-bin histogram = 8 KB |
| T2 | **Fused scoring with histogram, no score materialization** (epilogue of `k_idx_score_wmma` adds a per-token histogram of the score's top bits with LDS/global atomics; a second pass recomputes the scores of tiles and emits only values >= threshold bin; exact boundary-bin sort among few candidates). Recompute is cheap: scoring is ~3 % of peak FLOPs at 250K | StreamIndex (chunked partition-merge, never materializes the tensor; recall >= 0.998 min, exact at small sizes) [abstract]; TRT-LLM "bin by leading 11 bits halves candidates" [docs]; the recompute idea itself is **own design**, not from a source | exact if the recompute is the same kernel with the same reduction order (deterministic) | yes |
| T3 | **Smaller score working set**: (a) f16 shadow of the pool for prefill (the kernel converts to f16 on load anyway, so results are identical; halves pool traffic, 16 MB at 250K < IC; +192 MB VRAM for 12 layers at 250K), (b) `SCORE_TOK` / `TT` smaller so the score block (32 x n_pool x 4 B) stays < L2, or store scores as 2-D tiles that top-k reads while still resident | own design from the cliff analysis (section 2); no external source | exact | yes. Cost of smaller `TT`: Q rows per tile fall below 128, pool is re-read more often |
| T4 | **Overlap** top-k of tile i with scoring (WMMA) of tile i+1 on a second stream (top-k is latency-bound with 32 blocks) or indexer on a second stream next to the Q/K/V projections | SGLang overlaps the indexer with QKV projection on a 2nd stream [docs]; no gain number given | exact | yes (streams; no blocking HIP calls inside a megakernel wait, prefill kernels are ordinary launches) |
| T5 | **Union gather + per-query masks** (token tile shares the union of selected blocks; K/V of the union staged once; rows = tokens x heads fill WMMA M) | aiter #6576 union kernel: chunked prefill 2.3-7.4x faster than SGLang's Triton kernel on MI308X [PR page]; research/15 section 2 (decode, own idea) | numeric (different summation/softmax order; masks make it mathematically identical) -> measured | yes; 64 KB LDS for a 256-dim union tile is the constraint; K and V of a 2-head union tile of 32 cells = 32 x 1,152 B ~ 37 KB int8, f16 staging doubles that |
| T6 | **Pre-stage the KV rows to VRAM** (copy selected rows of the tile's union from host to VRAM on a copy stream, a double buffer, then attend from VRAM) | no engine found that does this for QSA/DSA; LongCat's KV-offload analysis: chunk overlap between steps 65 -> 82 % with their streaming layout [abstract-HTML]; the DMA path (`hipMemcpyAsync` of gathered rows) is **own idea** | exact (attention math unchanged) | yes; PCIe 55.7 GB/s measured. Needs an index-driven gather, e.g. by a kernel writing a contiguous staging buffer in VRAM |
| T7 | **Neighbour-token threshold guess** (GVR adapted to prefill: take token t-1's selection or a shared "anchor" token's selection to estimate the threshold, verify with counts, fall back to radix) | GVR is decode-only and exact: 1.88x average vs radix, 35-50 % consecutive-step overlap in layers >= 20 [docs, abstract]; NVIDIA states prefill is not applicable. The adaptation (within a tile the tokens are processed in parallel, so the guess must come from a different token) is **own idea** | exact | yes, but T1/T2 probably remove most of the benefit |
| T8 | **Index sharing across QSA layers** | IndexCache: 75 % of indexers dropped, 1.82x prefill / 1.48x decode on a 30B DSA [abstract]; LongCat CLI N = 2, trained, adjacent layers share ~57 % of top-k [abstract-HTML]. N = 4 loses accuracy | **changes**; training-free greedy search needs a calibration run | yes. Saves half of scoring + top-k + the iq projection of the shared layers; our QSA layers are 4 layers apart |
| T9 | **Hierarchical (coarse-to-fine) indexer** | HISA 2.16-3.75x at 64K (A100, token-level DSA indexer) [abstract-HTML]; LongCat HI 0.79x at 128K, 1.47x at 256K [abstract-HTML] | **changes** | yes. Already 4:1 pooled; payoff only near 250K |
| T10 | **fp8/int8 WMMA scoring** (pool and Q in fp8 or int8) | DeepGEMM (fp8, scales per row) and TRT-LLM (blockwise FP8 indexer) [docs]; V4 FP4 needs QAT (papers.md); gfx1200 fp8 186 / int8 183 TOPS (reference platform) | **changes** (scores differ; top-k boundary moves; the model was trained with bf16 indexer keys per vllm-qwen38-flash-next-kv-quant: "indexer caches stay bf16", papers.md) | yes, but compute is not the bottleneck |
| T11 | **Skip indexer/top-k when all cells are selected** (<= 2,048 cells) | TRT-LLM PR-9524, 1.03x on 1K/1K [docs] | exact | already in (`qsa_sel`) |
| T12 | **Drop the sort of cells / cells ascending after top-k** | no source | exact only if attention does not depend on cell order (it does: online-softmax order) | skip |

Not found: a published algorithm that picks the union of selections over a chunk of queries for the **indexer** step (only for
attention, #6576); any engine doing KV staging from host RAM for sparse attention prefill; a TileLang or FlashInfer prefill
benchmark of the indexer on any AMD GPU; DeepGEMM `mqa_logits` kernel speed.

---

## 4. Hypotheses to separate (before building)

| H | Statement | Evidence for | Test |
|---|---|---|---|
| H1 | Excess is in `k_idx_score_wmma` + `k_topk` and is a cache-capacity cliff of the score block / pool | cliff between 64K and 128K, quadratic after; sizes above | kernel time per call vs position; L2/IC hit rates |
| H2 | `k_topk` is latency/occupancy bound (32 passes, strided, one block per CU) | code read | `k_topk` time per call vs n_pool; wave occupancy and `SQ_WAIT_*` counters |
| H3 | The KV gather over PCIe dominates at long context (less cell locality) | 28 MB/token nominal vs 55 GB/s; `k_attn_wmma` serial load-compute chain | `IE_KV_HOST=0` at 64K/128K (KV 12 x N x 1,152 B = 1.8 GB at 128K, 3.5 GB at 250K) and compare; PCIe counters |
| H4 | The prefill plan or MoE streaming changes with context | `choose_plan` falls back to smaller chunks | `IE_PF_DEBUG=1` per context length |
| H5 | `k_attn_wmma` itself grows with context (cells not contiguous, L2 misses) | none direct | kernel time per call vs position |

---

## 5. Candidates for us (sorted by expected gain / effort, expectation not measured)

| Rank | Candidate | Expected effect | Effort | Exact | Evidence |
|---:|---|---|---|---|---|
| 1 | **T1: multi-bit radix top-k** (rewrite `k_topk`: 8-bit digits, LDS histograms, 3-4 passes, coalesced reads, scan with all threads instead of thread 0) | removes ~90 % of top-k scans if H2 holds; top-k becomes ~1 read of the scores plus 3 cheap histogram passes | low-medium (single kernel, tie rule testable bit-exact) | exact | TRT-LLM Blog 15, vLLM #37421 |
| 2 | **T3: f16 pool shadow + score tile sized for L2** | removes the cliff if H1 holds; pool traffic halves, score block 2 MB class | low | exact | own analysis (section 2) |
| 3 | **T4: overlap** of top-k(tile i) with scoring(tile i+1), or indexer next to projections | hides the latency-bound top-k on idle CUs | low-medium | exact | SGLang blog |
| 4 | **T2: histogram in the scoring epilogue + recompute emit** | removes score writes and most top-k reads (O(N^2) bytes -> O(N x bins)); decisive only if T1/T3 do not remove the cliff | high | exact | StreamIndex, own design |
| 5 | **T5/T6: union of the tile's selections, staged once** (also lifts WMMA fill from 12/16) | removes the dependence on PCIe latency; nominal 28 MB/token falls by the overlap factor (to be measured, `|union| / sum|set|` per layer at 128K) | high | numeric | aiter #6576 |
| 6 | T8 index sharing (greedy, training-free) | halves scoring + top-k in prefill and decode; LongCat shows training was needed for N = 4 | medium + quality run | changes | IndexCache, LongCat |
| 7 | T7 neighbour-token guess | small once T1 is in | medium | exact | GVR |
| 8 | T9 hierarchical, T10 fp8 scoring | only near 250K / compute not binding | high | changes | HISA, LongCat |

---

## 6. Measurement plan (question 4): measure first, then build

Rules: no build while a benchmark runs; same prompt, 3 runs per point with spread; desktop idle; record the plan (`IE_PF_DEBUG=1`); one file per run under `research/measurements/` and one row in `BENCHMARKS.md`. Name the rate: **prefill**. No tool here was run (no GPU in this task).

1. **Plan and layering (H4)**: `IE_PF_DEBUG=1` at 8K, 32K, 64K, 128K, 250K: print chunk, group, loan cap. If the plan changes with context, fix the comparison before anything else.
2. **Kernel time per kernel name** with `rocprofv3 --kernel-trace --stats` (CSV), runs at 16K, 64K, 96K, 128K (prefill only, short output): sum of `k_idx_score_wmma`, `k_topk`, `k_attn_wmma`, the GEMM kernels, the DeltaNet kernels, the MoE copies and `hipMemcpyAsync` (use `--memory-copy-trace`). From the kernel trace compute **time per call as a function of token position** (grid of `k_topk` = 32 blocks; calls in launch order within a layer map to position): plot t(`k_topk`) and t(`k_idx_score_wmma`) per call vs n_pool. Expected if H1/H2: a knee near 128K; linear in n_pool below it. 250K only after the 128K result (458 s run).
3. **Cache and memory counters for exactly the three kernels** (`rocprofv3 --pmc` with `--kernel-include-regex`, a counter list from `rocprofv3 -L` for gfx1200; names to be confirmed on the machine [unconfirmed]): L2 hit/miss (TCC_HIT, TCC_MISS), TCC reads to EA/DRAM (TCC_EA0_RDREQ*), vector L0 cache hit/miss (TCP_*), `SQ_WAVES`, `SQ_WAVE_CYCLES`, `SQ_BUSY_CYCLES`, VALU and wait-instruction counters (`SQ_WAIT_INST_ANY`, `SQ_INSTS_VALU`), `GRBM_GUI_ACTIVE`; at 64K and 128K. Tells cache cliff (H1) from latency/occupancy (H2). Infinity Cache counters may not be exposed [unconfirmed].
4. **PCIe/host gather (H3)**: (a) A/B `IE_KV_HOST=0` vs `1` at 64K and 128K (KV fits VRAM there); difference = cost of the host path; (b) PCIe throughput from `amd-smi`/`rocm-smi` pcie metrics or kernel-side counters during `k_attn_wmma` only (`--kernel-include-regex k_attn_wmma`) [unconfirmed which exist on gfx1200]; (c) `k_attn_wmma` time per call vs position.
5. **Ablations** (small env switches in a throwaway branch; each is a measurement, none is a candidate): (i) top-k stub (selection = the 512 most recent blocks, no scoring) at 128K: time without the indexer path; (ii) scoring only, top-k skipped; (iii) `SCORE_TOK` 8 / 16 / 32 at 64K and 128K (the tile score block 0.5 / 1 / 2 MB at 64K): if t(`k_topk`) per token falls, H1 is confirmed; (iv) `IE_ATTN`, `idx_scalar`, `attn_scalar` toggles for reference.
6. **Selection statistics (for T5/T6/T7/T8)**, from a debug dump of cells at 128K, per layer: overlap between neighbouring tokens `|S_t ∩ S_{t+1}| / 2,048`; union over 16 and 32 tokens `|U| / sum|S|`; fraction of selected cells inside the last 4,096 positions; overlap of top-k between QSA layer k and k+1 for the same token (T8). All on CPU from the dump.
7. **Check against the model** (cheap, no new data): effective PCIe bytes per token = unique cells x 1,152 B x 12; compare to the measured copy rate.

Decision rule after step 2-4: if `k_topk` + `k_idx_score_wmma` explain > 60 % of the excess -> candidates 1, 2, 3 (T1, T3, T4) first; if `k_attn_wmma` with host KV explains it -> T6/T5; if neither (plan change, MoE) -> back to the layered-prefill plan (research/18).

---

## 7. Open points / not clarified

- No external numbers for QSA prefill at 64K-256K on any engine; the aiter PRs measure on MI308X/MI355X (gfx942/gfx950) with other tile sizes; no gfx12 data.
- DeepGEMM MQA-logits throughput, FlashInfer top-k PR numbers, TileLang indexer prefill: not found.
- #6576 is still open (approved once on 2026-10-27); the K1-K3 planner code was not read.
- GVR overlap in **prefill** (neighbouring tokens) is not published; the paper's 35-50 % is for decode steps.
- LongCat paper HTML read only to character 100,000 of 188,157; the quality tables beyond that were not read.
- The 64 KB LDS limit on gfx1200 in prefill is taken from the task statement, not verified here.
- Whether Qwen3.8's indexer tolerates fp8/int8 keys: no data (papers.md open question 4).

## Sources

- https://github.com/ROCm/aiter/pull/6576 (open), /pull/6515 (merged 2026-10-26), /pull/6469 (merged 2026-10-15), /pull/6675 (merged 2026-10-26)
- https://github.com/vllm-project/vllm/pull/37421 (merged 2026-04-08)
- https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.html
- https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog21_Temporal_Correlation_Meets_Sparse_Attention.html and https://arxiv.org/abs/2604.22312
- https://github.com/deepseek-ai/DeepGEMM, https://github.com/deepseek-ai/FlashMLA
- https://www.lmsys.org/blog/2026-08-26-qwen-flash-next
- https://docs.flashinfer.ai/api/topk.html (snippet), https://arxiv.org/pdf/2605.27740 (snippet)
- https://rohitai.com/blog/llama-cpp-b11405-cuda-lightning-indexer-qwen4exp (secondary; llama.cpp #29901)
- https://arxiv.org/abs/2603.12201 (IndexCache), https://arxiv.org/abs/2608.01662 and https://arxiv.org/html/2608.01662 (LongCat), https://arxiv.org/html/2603.28458v3 (HISA), https://arxiv.org/abs/2605.02568 (StreamIndex)
