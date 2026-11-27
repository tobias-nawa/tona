# 15 — Decode attention for sparse top-k attention (QSA/DSA)

Agent research (Sonnet), 2026-10-24

Method: web search and page retrieval (arXiv abstracts, GitHub PR pages, docs). Source code was **not** checked out or read. Marking per entry:
- **[PR page]**: PR description and review read on GitHub (design and measured numbers come from the author, not re-measured by us).
- **[docs]**: vendor docs/blog read.
- **[abstract]**: summary only read.
- **[unconfirmed]**: secondary source or not findable.

Our shape: 24 Q heads, 2 KV heads (GQA 12:1), head dim 256, KV int8 with fp32 scale per 32, ≤ 2,051 cells per token, 4 tokens per pass, gfx1200 (WMMA 16x16x16, wave32).

---

## 0. Most important finding: QSA has its own, current kernels (Oct 2026)

### 0.1 ROCm/aiter PR #6670, FlyDSL fused QSA sparse decode (gfx950), merged 2026-10-23 [PR page]
https://github.com/ROCm/aiter/pull/6670
Files: `aiter/ops/flydsl/qsa_sparse_decode.py`, `aiter/ops/flydsl/kernels/qsa/sparse_decode_attn.py`, `.../sparse_decode_reduce.py`, `op_tests/test_flydsl_qsa_sparse_decode.py`. This is **exactly our shape** (Qwen3.8, Hq/Hkv = 24/2, top-k 2051, head dim 256).

Core idea according to the PR:
- One CTA per (row, KV head, KV split). **All Q heads of the GQA group in one CTA** (group up to 16, specialized per group size; tested 3, 12, 16).
- **Gather in the kernel:** the 2,051 positions are resolved directly via `req_to_token` into pool slots, no separate gather pass, no scratch buffer.
- 4 waves per CTA, each takes 16-token tiles with its own online softmax, a single merge in LDS at the end.
- **MMA in decode:** QK with `mfma_f32_16x16x32_bf16`, **tokens on the M axis, Q heads on N** (the 12 heads almost fill N=16, this is the trick against waste at small M). P stays in registers because the QK accumulator layout matches the B-operand layout of the second MFMA. V is placed in a wave-private LDS tile and read transposed with `ds_read_b64_tr_b16`.
- Split-K: `num_splits` CTAs share the selection of a row; `recommended_num_splits(batch)` chooses so that the GPU is filled; separate reduce kernel, omitted for 1 split. `workspace_partials` for CUDA graph.
- Speculative verification: according to the PR **each row is its own length-1 request**. So **no** sharing of KV reads between the 4 tokens.
- KV: bf16 or fp8 e4m3 **without scales**; Q always bf16.
- Measurement (MI355X, 6.3 TB/s HBM, Qwen3.8 shape 24/2, 2051): 1 row 19.0 µs (bf16) / 18.4 µs (fp8); 128 rows 86.6 / 69.4 µs. A single call is thus at ~19 µs, **latency-bound, not bandwidth-bound**.

Expected benefit for us: reference design for our attention. Roughly computing our load: 2,051 cells × 256 dim × 2 KV heads × 2 (K+V) = 2.1 MB (int8) per token and layer, at 315 GB/s ≈ 6.7 µs per token and layer; 4 tokens × 12 layers ≈ 320 µs per pass as pure read time **without** sharing. We stand at 3,800 µs. Even at 40 % bandwidth utilization (including L2/IC hits) there remains a factor ~3–5 of headroom (own calculation, not measured; cache hits between the tokens lower the real load further).
Effort: medium to high. The core structure (group as N axis, tokens as M axis, WMMA instead of wave-shuffle dot product) is transferable to gfx1200 with `wmma_f32_16x16x16_f16/bf16`; head dim 256 = 16 K steps per tile. The PR is MFMA-specific and runs only on gfx950 (skipped on other archs), so adopt the design, not the code.

### 0.2 ROCm/aiter PR #6469, Triton/Gluon paged sparse attention (QSA), merged 2026-10-15 [PR page, thin]
https://github.com/ROCm/aiter/pull/6469
Kernels: `_qsa_paged_mqa_logits_kernel`, `_qsa_paged_mqa_logits_mfma_kernel` (indexer scoring), `_expand_selected_groups` (block indices → cells), `_qsa_sparse_paged_gqa_kernel`. Source ROCm/ATOM `qwen4_exp/ops/qsa.py`. Description fields in the PR are empty; details only obtainable from the code (not read). Related: #6576 (open, FlyDSL QSA prefill MI308), #6515 (FlyDSL split). Relevance: the Triton variant is a readable comparison to #6670 and may also run on gfx12 (not checked).

### 0.3 SGLang: QSA path and IndexShare for MTP (2026-08-26) [docs]
https://www.lmsys.org/blog/2026-08-26-qwen-flash-next
- Decode: backend TRTLLM-Gen (Blackwell) or packed FlashAttention; paged indexer scorer, then **compaction of the selected original K/V**, then attention. The indexer runs on a second stream in parallel to the QKV projection.
- **IndexShare MTP:** A draft-extend pass computes the indexer over the tokens just accepted by the target, the selection is recorded and reused for the whole draft loop (missing columns for positions drafted in the meantime are added). Indexer calls per MTP iteration N → 1, acceptance length unchanged according to the blog. This concerns the **drafter**, not directly applicable to us (our drafter is DFlash with a window, without an indexer).
- On target verification and on sharing KV between verify tokens the blog says nothing. No PR for QSA decode named.

### 0.4 NVIDIA TensorRT-Edge-LLM QSA plugin [docs]
https://nvidia.github.io/TensorRT-Edge-LLM/latest/cpp_api/plugins/qsaAttentionPlugin.html
Decode in three indexer stages (B1 query preparation and tail, B2 block scores from kbar, B3 **deterministic radix top-512**), then **single-launch split-K** sparse attention (`CuteDslQsaSparseDecodeRunner`). Speculative decode explicitly rejected (S == 1). The kbar of the block lies in the V-row tail of token 4g, the index K of the incomplete block in the K tail. Useful as a layout idea: indexer state next to the KV entry, so that one cache-line access delivers both.

### 0.5 vLLM MiniMax-M3 sparse attention [docs]
https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/common/sparse_attention/
Triton `minimax_m3_sparse_attn_decode`: split-K over the selected blocks (request-major chunks), `decode_query_len` = 1 + `num_speculative_tokens` uniform, FP8 scales as argument. On MI355 AITER sparse PA is used. Evidence that multi-token queries are provided for in sparse decode kernels; the docs do not show whether KV accesses are shared.

---

## 1. How current kernels compute sparse decode (question 1)

| Kernel | Source / state | Group (GQA/MQA) | Gather | MMA | Split-K |
|---|---|---|---|---|---|
| aiter FlyDSL QSA decode | PR #6670, 2026-10-23 [PR page] | all heads of the group in one CTA, heads = N of the MFMA | in the kernel via `req_to_token` | MFMA 16x16x32 bf16, P in registers | `recommended_num_splits`, reduce kernel |
| FlashMLA sparse decode | deepseek-ai/FlashMLA, README state 2026-09-2026-10-09-30 [docs] | MQA (all h_q share one KV), `s_q` = MTP length, `indices (B, s_q, topk)` | in the kernel, -1 = invalid | Tensor Cores (SM100/Ascend; Hopper removed since 2026-09-30) | Split-KV with metadata (`get_mla_metadata`) |
| FlashMLA SM100 (Modular docs) | docs.modular.com mla_decode_sparse [abstract/unconfirmed] | MQA | **gather4 TMA** with index per tile | UMMA, FP8→BF16 in SMEM | – |
| TileLang sparse_mla_fwd | NeMo-Automodel docs: `block_I = 64`, 256 threads, 2 stages [docs line, source not read] | heads as M of the GEMM | gather top-k directly, no [T,T] mask | Tensor Core | – |
| vLLM MiniMax-M3 Triton | see 0.5 | – | block indices | – | request-major chunks |
| TRT-Edge-LLM QSA | see 0.4 | – | paged pool | – | single-launch split-K |

Summary of the common patterns (statement across several sources, not specifically verified in the code):
1. **GQA sharing is standard**: the heads of the group form the N or M dimension of the MMA, the K/V tile is loaded once and used by all 12 heads. Here each wave (token, **head**, 256 cells) reads the same K/V cells 12 times (L2 absorbs much, but dequantization and load instructions occur 12 times). This is the biggest lever.
2. **MMA instead of dot product**: 12 heads fit into N=16 (75 % utilization), with 4 tokens × 12 heads = 48 rows even into three full 16 tiles, see section 2.
3. **Gather inside the tile loop**, not as its own pass.
4. Split-K with a heuristic by batch size (goal: fill the GPU). Here: 32 blocks × 16 waves; a token pass with 4 tokens × 2 KV heads = 8 (token, KV head) pairs, 2,051 cells each; splitting into pieces of 128–256 cells yields 64–128 work packages, fits 32 CUs.

---

## 2. Verifying several tokens (question 2)

Finding: **No kernel found shares KV reads between verify tokens via a top-k union.**
- aiter #6670: each row a length-1 request [PR page].
- FlashMLA: `s_q` > 1 for MTP, indices per token `(B, s_q, topk)` [docs]; whether reads are shared internally is not visible.
- TRT-Edge-LLM: spec decode rejected.
- vLLM MiniMax-M3: `decode_query_len` uniform, details unclear.
- Papers on sparse verification (**abstracts only**): Dustin, arXiv 2606.24957 (2026-06-23, ICML 2026): draft lookahead plus target history select critical KV for the whole verification window, 9.17x end-to-end at 32k (Qwen2.5-72B, top-k estimated per head subset). Vegas/SpecAttn, arXiv 2602.07223 (ICML 2026): verification provides the sparse selection for drafting. SparseSpec-L, arXiv 2607.27735: statistics of the last verification reconstruct the sparse index for the next draft round. All three are **self-speculative** and choose the sparse set for the drafter, not a union for target verification. Only indirectly useful for us.

Conclusion (own reasoning, not from a source, not measured): A union strategy is plausibly possible here and would be a differentiator:
- Per (KV head, pass): form the union U of the 4 top-k sets (with high overlap ~2,051–2,600 cells instead of 4 × 2,051 = 8,204). Load the K/V tile once, dequantize, then compute **rows = 4 tokens × 12 heads = 48** against the tile (3 WMMA M tiles), with a **mask per token** (cell ∈ set_t, plus causal within the pass: token t sees the cells of tokens < t in the pass).
- Mask as a 4-bit field per cell in U (bitmap over the union), generated when merging the top-k lists. Cost: merge of the 4 lists (sorted/hash in LDS), ~4 × 512 blocks.
- Benefit: read volume and dequantization drop by the overlap factor (up to ~3–4x), MMA utilization rises from 12/16 to 48/48. Whether the overlap is really that high **has to be measured** (measurement: |U| / Σ|set_t| over real 13K runs, per layer).
Effort: high (new kernel plus list merge), but can be built on top of the path in 0.1.

---

## 3. Reuse of top-k indices (question 3)

### 3.1 IndexCache, arXiv 2603.12201 (2026-03-12, Bai et al., Tsinghua/Z.ai) [abstract + vLLM docs]
Layers split into Full (own indexer) and Shared (uses indices of the nearest Full layer). Training-free: greedy search over layer configurations by LM loss on calibration data; training-based: multi-layer distillation. 30B DSA model: **75 % of the indexer computations are dropped**, up to 1.82x prefill and **1.48x decode**; preliminary confirmation on GLM-5 (numbers not read). Code link not found.
vLLM: feature `use_index_cache` via `--hf-overrides`, `index_topk_freq` (N: top-k only on every N-th layer) or `index_topk_pattern` as an F/S string per layer. https://docs.vllm.ai/en/latest/features/index_cache/ (PR number not given).
Benefit for us: We have only 12 QSA layers, separated by 3 DeltaNet layers each (3:1), so far apart; the similarity of neighboring layers is probably lower than in DSA with 61 dense layers (conjecture). Testable training-free: greedy search by perplexity (we have perplexity measurement) with a pattern such as FSFS… (6 indexers instead of 12). Savings = half the indexer plus half the top-k. The indexer is cheap for QSA (block scores, 4×128, 1 key per 4 tokens), so the gain is limited; first measure how many ms the indexer really costs per pass. Effort: low (pattern switch), risk: quality, the model was not trained for this.

### 3.2 Index reuse across steps
- SGLang IndexShare MTP (0.3): reuse across draft steps [docs]. For us: **across the 4 tokens of a pass** the reuse is basically covered by the union idea (section 2), but the indexer still has to compute per token because it has a different Q.
- Between consecutive passes: top-k sets change with strong overlap according to earlier observation. A cache of the last selection as a candidate set (checking only deltas) is **not documented by a source**, only an idea. A similar idea (SparseSpec-L, 2607.27735 [abstract]) reconstructs the selection from the verification statistics.

---

## 4. AMD-specific (question 4)

- **aiter #6670** (0.1): gfx950 only (MFMA, `ds_read_b64_tr_b16`); gfx12 has no MFMA, WMMA instead. Source for the design, not for runnable code.
- **aiter, open QSA/sparse work** [PR titles from a list, contents not read]: #6841 (Mha v4 sorted-sparse, gfx950), #6840 (Triton/Gluon sparse, rotating buffers), #6743 (HipKittens H32 sparse MLA gfx950), #6675 (MLA v4.1 sparse prefill gfx950). Issue #6892 (open, 2026-10-23): fp8 MLA decode yields NaN for 512-wide rows. No gfx1200/gfx1201 entries on the first results page; #6462 (merged 2026-10-21) concerns gfx1250, not gfx1200.
- **llama.cpp**: `HIP: add fattn-mma-f16 for RDNA4 (#18481)` [commit title via mirror, merge date unconfirmed]; #14323 (vec FA on RDNA4); #12032 (rocWMMA FA, `GGML_HIP_ROCWMMA_FATTN`); #22880 (MMA FA for RDNA3, b9158, 2026-05-14 [secondary source]); #26419 (MMA FA head dim 256 on RDNA, per a forum post [unconfirmed]). All dense, not sparse. Useful: the head-dim-256 variant and gfx12 operand layout as a code template for WMMA attention on RDNA4. Concrete files (`ggml/src/ggml-cuda/fattn-mma-f16.cuh`, `fattn-vec.cuh`) **not read**.
- **GPUOpen WMMA guide RDNA4** (https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/ [docs]): `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, per lane 8 contiguous elements (128-bit load), A/B K-major, D M-major; swapping A/B yields D N-major and thus directly usable as the A operand of the following GEMM (P·V without an LDS detour). Exactly the trick from #6670 (P in registers). Open: the exact lane layout for P→A operand in gfx12 (there the two half-waves share the K halves) would have to be tested.
- **int8 KV in the kernel**: No sparse kernel found has int8 with group scales; #6670 fp8 without scales, FlashMLA fp8/fp4 with e8m0 scale per 32 (SM100) or e2m1 with e4m3 scale per 16 [docs]. For us: gfx1200 has **int8 WMMA (183 TOPS)**. Option: quantize Q to int8 as well (scale per head), QK directly as int8 WMMA, apply the scales (fp32 per 32 values = 1 K block ... with head dim 256 eight scales per cell) per block while accumulating. This saves the dequantization to f16 entirely. Loss: Q quantization error, check accuracy. P·V with fp8/f16 (V dequantized). Not from a source, own idea; effort high.

---

## 5. Drafter: fp32 K/V, head dim 128, window 2048, 32 Q / 8 KV heads (question 5)

This is dense and window-based, not top-k. No sparse kernel was found for it; standard FlashDecoding with GQA group 4 (32/8) applies.
- Read volume: 2048 × 128 × 8 × 2 × 4 B = 16.8 MB per layer and token in fp32 (**8x more than our QSA access**; at 315 GB/s 53 µs per layer and token). **Storing K/V as bf16/fp16 or fp8** halves/quarters this directly. The benefit is the largest in this category, effort low (only change the cache write path and read path). The drafter's accuracy is uncritical because wrong drafts only cost acceptance, not correctness (re-measure the acceptance rate).
- Tile structure as in 0.1: group of 4 heads × (tokens of the draft) as M, 16 rows = 4 heads × 4 tokens fits **exactly into one WMMA M tile** if the drafter computes several positions at once (only sensible for block drafting; to be checked for DFlash block drafting).
- Ring-buffer layout for the sliding window (no copying) and split-K only if window × heads do not fill the CUs: 8 KV heads × 2048/256 = 64 packages, sufficient.
Effort: low to medium.

---

## Assessment (benefit for us, descending)

| # | Step | Expectation | Effort | Evidence |
|---|---|---|---|---|
| 1 | **GQA sharing + WMMA**: load/dequantize the K/V tile once, 12 heads as N axis, tokens/cells as M; P in register, A/B swap for PV | biggest lever, up to factor 2–4 on 3.8 ms (estimate, not measured) | medium-high | aiter #6670 [PR page], GPUOpen guide |
| 2 | **Union of the 4 top-k sets** with bitmask per cell, 48 rows (4 tokens × 12 heads) per tile | reading/dequantizing / overlap factor; measure the overlap first | high | no source, own reasoning; #6670 does not share |
| 3 | **Drafter K/V from fp32 to fp16/fp8**, ring buffer | 2–4x fewer bytes in the drafter | low | calculation above |
| 4 | Check IndexCache pattern (6 of 12 indexers) by perplexity | indexer+top-k halved; value depends on the indexer share, measure first | low | arXiv 2603.12201 [abstract] |
| 5 | Split-K heuristic like `recommended_num_splits` (packages = multiple of 32 CUs), single-launch merge in the megakernel | smaller | low | #6670, TRT-Edge-LLM |
| 6 | int8 WMMA for QK (quantize Q) | saves dequantization | high, accuracy risk | own idea |

Open gaps: kernel source code (aiter files, llama.cpp fattn-mma, FlashMLA `csrc`) not read; Qwen3.8-Next paper (arXiv 2608.30320) captured only via title/snippets, the PDF was saved locally but not extracted; FlashInfer sparse and TileLang source not found; GLM-5 index-sharing details only via the IndexCache abstract.
