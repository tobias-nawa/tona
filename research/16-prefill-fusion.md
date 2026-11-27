# 16 – Prefill: fusing elementwise kernels, hyper-connections, GDN, MoE, megakernel

**Agent research (Sonnet), 2026-10-24.** Complements `13-deltanet-chunked.md` (not repeated) with new material.
Marking: **[code]** = source read directly (raw file from GitHub, state of 2026-10-24),
**[docs]** = project documentation/release notes read, **[abstract]** = only summary or search hit seen,
**[estimate]** = own calculation, not measured. Nothing was measured here.

## 0. Key findings

1. **Our hyper-connection is not a Sinkhorn mHC.** The Qwen3.8-Flash-Next scaffold in llama.cpp
   (`src/models/qwen4exp.cpp`, architecture name `qwen4exp`) **[code]** shows: `hc = 4` streams, grouped RMS norm per stream
   (gamma `[2560, 4]`), down projection `4·2560 → 320` (+ 4 "inject" rows), SiLU, up projection `320 → 10240`, sigmoid gate,
   mean over the 4 streams (`mixed`, `[2560]`); write-back `R += block_out · 2σ(inject/4)` per stream. **No
   comb matrix, no Sinkhorn** (llama.cpp #28901: "a NULL comb in `hc_comb`", "per element gate in `hc_pre`").    DeepSeek's Sinkhorn kernels are therefore a model only in structure.
2. **Our three elementwise kernels are already at the bandwidth ceiling** **[estimate]**, assumption: R fp32
   `[T, 4, 2560]`, T = 12,000, 96 mixes (48 layers × attn/FFN), 315 GB/s:
   `k_combine` reads R (491 MB) + `bo` (123 MB), writes R (491 MB) = 1.1 GB → 3.5 ms × 96 = **0.34 s** (measured 0.39);
   `k_norm16` reads R, writes xn16 (246 MB) = 0.74 GB → **0.23 s** (measured 0.28);
   `k_mix` reads `gated` (T × 10240 fp32 = 491 MB) + R, writes `mixed` = 1.04 GB → **0.32 s** (measured 0.38).
   This only gets faster through **less traffic** (fusion, smaller data types), not through better kernels.
3. **A megakernel is irrelevant for prefill** (section 4): prefill is compute-bound, launch overhead at 2.1 µs per kernel
   is ≈ 2.5 ms at ~1,200 kernels per chunk against 10.6 s.

## 1. Hyper-connections / mHC: fusion

### 1.1 DeepSeek mHC (original)
- **arXiv 2512.24880** (Dec 2025, v2 Jan 2026) **[docs, paper HTML read]**: n = 4 streams, `H_pre`, `H_post`, `H_res` (Sinkhorn,
  t_max = 20) from `φ ∈ ℝ^{4C × 24}` (TF32). Three kernels for the coefficients (1: projection+norm in two scans with the matrix unit,
  2: small coefficient operations, 3: Sinkhorn in one kernel), plus two for applying: pre-mapping, and **post-mapping +
  residual merge fused**. Effect according to the paper: reads `(3n+1)C → (n+1)C`, writes `3nC → nC` elements per token. Additional training cost
  6.7 %. TileLang. Recompute in the backward pass (irrelevant for us).
- **deepseek-ai/TileKernels** `tile_kernels/mhc/` (files including `pre_big_fuse_*`, `pre_split_mixes_*`, `pre_apply_mix_*`, `post_*`,
  `sinkhorn_*`, `norm_fn_*`, `head_compute_mix_*`; variants `_cuda`, `_kernel`, `_asc`) **[code, directory + `pre_big_fuse_kernel.py`
  read]**. CUDA and Ascend only, **no HIP path**. The repo itself says it is not best-practice code.
- **Benefit for us:** structural model (traffic per token as the target quantity, "post + next pre" in one pass). Kernels not transferable.
  Effort: nothing to port.

### 1.2 vLLM (DeepSeek V4/V4.1)
- `vllm/model_executor/layers/mhc.py` **[code]** (last commits: #58655 2026-09-29, #56513 2026-09-12, #56503 2026-09-10, all "[ROCm][DSv4.1][Perf]").
  Ops: `mhc_pre`, `mhc_pre_delayed`, `mhc_post`, `hc_head`, `mhc_fused_post_pre`. TileLang kernels
  (`kernels/mhc/tilelang_kernels`, **[docs]**): `mhc_fused_tilelang` = **post-mapping + pre-norm GEMM FMA in one kernel** (write the new
  residual, sum of squares and projection `x·W` in the same pass, split-K, 256 threads, only for ≤ 16 tokens);
  `mhc_pre_big_fuse_tilelang` = norm partial sums + coefficients + Sinkhorn + stream collapse; `hc_head_fuse_tilelang` = two passes, without
  intermediate tensors in memory. For > 16 tokens: post kernel, then TF32 pre-norm GEMM (DeepGEMM), then `pre_big_fuse`.
- **ROCm:** TileLang wheels have no HIP target, hence `mhc_pre`/`hc_head` fall back to the **torch reference**; on gfx942 TileLang mHC
  is faulty. Instead **AITER** (`aiter/ops/mhc.py`, 1,230 lines **[code]**): `mhc_pre_gemm_sqrsum` (GEMM with sum of squares in the same
  pass), `mhc_pre_big_fuse`, `mhc_fused_post_pre` (Triton/Gluon), configurations for gfx942, gfx950, **gfx1250**. gfx1250 is the
  Instinct line, **not gfx1200**; the commit "[GFX12] gluon backend for mhc_fused_post_pre" (aiter #6442, 2026-10-12) thus does not target
  the 9060 XT. Only for `hc_mult == 4` and `hidden % 256 == 0`. Also `aiter/ops/mega_mhc.py` (HIP, "delayed Mega-mHC … in a single launch", 504 lines).
- **Benefit:** shows the fusion boundaries vendors choose: (a) post of the previous block + sum of squares/projection of the next in one pass
  over the residual stream, (b) small coefficient arithmetic in one kernel. Both are passes over R that save us traffic.
  Directly usable: nothing on gfx1200.

### 1.3 SGLang
- The day-0 post (LMSYS, 2026-04-25) names "TileLang mHC" in the V4 stack **[abstract]**; source not checked in code. The MThreads library `mate`
  adopts the "SGLang MHC pre convention" **[abstract]**. Nothing of its own on hyper-connections found for Qwen3.8.

### 1.4 llama.cpp (the most relevant finding for us)
- **#28901 "qwen4exp: add hc ops"** (merged 2026-09-16) **[code]**: ggml ops `GGML_OP_DSV4_HC_PRE` (gated: `σ(gate)·xn`, mean over streams in one op) and
  `…_HC_POST` (`residual + block_out·w`, optional comb). CUDA/HIP kernel `ggml/src/ggml-cuda/dsv4-hc.cu` (`dsv4_hc_pre_f32<gated>`, `dsv4_hc_post_f32<has_comb,gated>`).
  Measurement there (DGX Spark, **CPU**): pp2048 964 → 1,097 tok/s (prefill, +14 %). Only the **elementwise part after the GEMMs** is fused, neither norm nor GEMM.
- **#25421 `GGML_OP_SINKHORN_NORM`** (not merged) for DeepSeek-V4 mHC: MI250X, DeepSeek-V4-Flash, pp64 111.5 → 119.8 (+7.4 %), tg16 +61 % **[docs]**
  (Sinkhorn was ~30 % of the GPU kernel time due to 250k tiny launches). Not relevant for us (no Sinkhorn).
- **#29520** (Vulkan, closed) fuses `SCALE → SIGMOID → SCALE → hc_post`; **#29187** (open, CUDA) fuses the GDN alpha/beta projection with softplus/sigmoid **[docs]**.
- Open model PRs for DeepSeek V4.1: #30817, #28696, #31086 (all open) – no kernel news for us.

### 1.6 Conclusion and plan for us (estimate)
HC path per mix: norm (R → xn16) → down+inject GEMM (`[T,10240] × [10240,324]`) → SiLU → up GEMM (`[T,320] × [320,10240]`) → `mix` → ... → `combine`.
Order by benefit/effort:
1. **Up GEMM with the mix in the epilogue** (on gfx12 WMMA): reorder the up weight rows so that one block computes column `d` of all 4 streams
   (permute the weights once when loading). Epilogue: `Σ_c xn[c,d]·σ(gate[c,d]) / 4` in registers, writes only `mixed` (f16/bf16, 61 MB).
   Saves writing+reading `gated`: ~0.98 GB → ~3.1 ms × 96 ≈ **0.30 s** (estimate). Effort medium; the gfx12 WMMA operand layout (8 elements per lane instead of the gfx11 duplication) has to be written anew.
2. **Combine + norm + xn16 in one pass**: R row (4 × 2560) in registers, sum of squares per stream, write `xn16`. Instead of 1.1 + 0.74 GB only 1.35 GB → **~0.15 s** (estimate). Effort low.
3. **Keep the residual R in bf16/fp16** (DeepSeek keeps bf16): halves the R traffic, still about 0.2–0.25 s (estimate), but changes the numerics over 96 additions; quality measurement needed.
   vLLM/DeepSeek keep bf16. Effort low, risk quality.
4. Together: from ~1.05 s to roughly **0.4–0.6 s**, i.e. 4–6 % of the prefill time [estimate].
5. **To check:** The down and up GEMMs of the HC themselves cost `2·(10240·324 + 320·10240)·T·96 ≈ 15 TFLOP` at T = 12K (estimate), which is ~0.3 s at 50 TFLOPS. In our GPU time breakdown it is not clear
   whether they run under "dense GEMMs 2.4 s". If they run poorly there (K = 320 is small, N = 324 narrow), a dedicated measurement is worthwhile.
6. Norm in the down GEMM prologue: possible (fold gamma into the weights, `rs` per stream as scaling per K block of 2560), but it only saves traffic if xn16 is not needed elsewhere; the up epilogue needs `xn`
   (from xn16 or `R·rs·w`). Hence ranks 1 and 2 first.

## 2. Chunked Gated DeltaNet (only new relative to `13`)

### 2.2 New work 2026
- **arXiv 2607.16831** (2026-07-18, MLSys 2026 FlashInfer contest, B200) **[docs, HTML read]**: the best *scalar* recurrence kernel (MSInfer, 239.5 µs prefill) got no further after register tiling,
  V-column splitting and algebraic fusion ("plateau"); the chunked WY submissions (Kachua, Neumann-like inverse, split/flat/direct WY; UW SyFI: bf16 WY with **C = 16, WMMA fragments, `cp.async`**; "LLM-CUDA" uses upstream PR #3001) were at 51.99–61.5 µs, **3.9–4.6× faster**.
  Failed for MSInfer: software prefetch, shared q/k broadcasts, fast math. Tiny contest shapes (4 QK, 8 V heads), not our size.
- **arXiv 2606.06034** (June 2026) **[docs, HTML read]**: *Multiplication-Only Matrix Inversion* for quantized Gated DeltaNet. `T = (I−A)⁻¹` ≈ truncated Neumann series (N = 3, with band mask `i−j ≤ N`) plus residual correction `T⁽⁰⁾ Σ E^s` (S = 8; for C = 32 S = 4 suffices).
  **Matmuls only**, 7 (C = 32) or 11 (C = 64) of them, instead of serial forward substitution; kernel 5.2× / 4.2× / 4.6× (C = 32/64/128) on a Snapdragon NPU, layer −18 % to −25 % cycles; perplexity on Qwen3.5 (0.8–9B) equal to FLA (numbers: 8.98 baseline; N = 4 gives 9.54 – sensitive).
  A 128×128 direct inverse overflows in fp16/int16, hence blockwise.
  **Benefit for us:** turns the part that runs serially in fp32 on the vector units for us (64 serial steps per chunk) into WMMA matmuls. Risk: precision (`13` §3 names the inverse as critical). Measurement against exact fp32 substitution needed.
- **New cell variants** (search-hit snippets only, **[abstract]**, not applicable to a fixed model): MDN (arXiv 2605.05838, momentum parallel), Preconditioned DeltaNet (2604.21100), Semidirect Fourier Delta Attention (2607.11897, reference construction only, no optimized kernel).
- **arXiv 2607.14568** (Fermi GPU assistant): chunked scan 2.8× faster than sequential after a forward-substitution kernel was corrected **[abstract, PDF not readable]**.
- **DeltaProduct** (2502.10297): still unchecked, irrelevant for a fixed model anyway.

### 2.3 Fusing conv/norm/gating
- **vLLM** `vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv` **[docs]**: `fused_post_conv_prep` = "split + l2norm + gating in one kernel" (delivers `q,k,v,g,beta`, optionally `exp(g)` for FlashInfer). Source not read (path 404 on `main`).
  Likewise `fused_sigmoid_gating_delta_rule_update` (decode). On AMD all of this runs via Triton; **no** RDNA numbers found.
- **llama.cpp** #29187 (open): alpha/beta projection + softplus/sigmoid in one launch **[docs]**.
- **Benefit for us:** small (these elementwise parts are not in the list of small kernels); only take along if the chunk path is built.

### 2.4 Recommendation GDN (estimate)
Build the chunk path only if it pushes the 0.59 s down to ≤ 0.25 s: **C = 16/32** (WMMA tile, short inverse, least precision worry), `T`/`P`/`W` in one `prep` over all chunks, scan with the state in accumulators and the norm in the scan,
optionally a Neumann inverse (2606.06034). Experience reported there: two launches cost more than they save below ~100 tokens; on V100 it was latency, not compute, that ate the gain. Expectation: **−0.2 to −0.35 s**, effort high.

## 3. MoE prefill: grouped GEMM, gather/scatter

| Project | What | Source / status |
|---|---|---|
| **llama.cpp** `mul_mat_id` (CUDA/HIP MMQ) | `mm_ids_helper` (GPU, `mmid.cu`) builds `ids_src1`, `ids_dst`, `expert_bounds` from the router IDs. **Gather in the quantization** (`quantize_mmq_q8_1` with `ids_src1`, activations expert-compact), grid z = expert, tile width `J` 8…128 in steps of eight by average tokens per expert (own branch for RDNA3/4), **scatter in the epilogue** via `ids_dst[j]·stride`. WMMA path for RDNA4 in the same `mmq.cuh` (`AMD_WMMA_AVAILABLE`, `GGML_CUDA_CC_IS_RDNA4`). | **[code]** `ggml/src/ggml-cuda/{mmq.cu,mmq.cuh,mmid.cu}` master 2026-10-24 |
| **vLLM** `fused_moe.py` | `moe_align_block_size` (sorted, padded per expert to `BLOCK_SIZE_M`) → `sorted_token_ids`; one Triton kernel per GEMM, gather via `offs_token`, **router weight in the epilogue** (`MUL_ROUTED_WEIGHT`), SwiGLU is its own kernel in between. ROCm branches present; the repo has tuned configurations for `AMD_Radeon_R9700` (fp8_w8a8, E = 128 or 64), **not** for E = 256 and not for our quantization. | **[code]** `vllm/model_executor/layers/fused_moe/fused_moe.py`; configuration list via GitHub API |
| **vLLM on RDNA4 (community)** | Report: AITER C++/ASM kernels do not run on RDNA4, Triton FP8 with native WMMA after patching the architecture detection; tuned fused-MoE configurations fix a performance cliff at M ≥ 96. | **[abstract]** (vLLM forum, Docker image `vllm-radiance`, not independently checked) |
| **DeepGEMM Mega MoE** | EP dispatch + Linear1 (FP8×FP4) + SwiGLU + Linear2 + EP combine in one kernel, April 2026 (PR #304). On AMD: Primus "MegaMoE" (FlyDSL, gfx950, bf16, EP only). | **[abstract]**; multi-device idea, no benefit for one card |
| cuDNN grouped GEMM + SwiGLU | SM100, SwiGLU in the epilogue | **[abstract]** |
| AMD RDNA4 WMMA | GPUOpen part 2: "Wide-K" for fp8/int8: two 16×16×16 WMMAs give K = 32, hence 128-bit loads; bit-identical to the normal order; llama.cpp adopted it for quantized GEMMs. | **[docs]** gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2 |

**Conclusions for us:**
- The 2026 pattern is the same everywhere: sorting on the GPU (counters/offsets), gather on load (A operand via index list or in the quantization), **weighting and scatter in the epilogue**, SwiGLU either in the gate/up epilogue (cuDNN, Mega MoE) or as its own pass (vLLM). On a card with 32 CUs the decisive point is the **tile utilization at ~12,000·10/256 ≈ 470 rows per expert** (estimate): with 64 tiles ~7.3 tiles per expert, i.e. up to 1/8 idle due to the remainder tile; a tile list or stream-K-like splitting of the remainder tiles helps.
- **Largest single lever: int8 WMMA instead of f16 dequant** (183 TOPS int8 against 93 TFLOPS f16 measured here). Q8 activations per 32 values, expert codes unpacked to int8 in registers, SwiGLU+requantization in the epilogue. The rounding loss is real, so measure perplexity/KL. Effort high, expectation **−0.4 to −0.9 s** of the 2.35 s [estimate, not transferable without measurement].
- Measure first: achieved TFLOPS/TOPS of the expert GEMMs against the ceiling (rows per expert, tile utilization), then decide whether tile lists or int8 make the difference. This is the only place where **no** estimate is available here.
- Missing publicly: a tuned grouped-GEMM kernel for **gfx1200** (E = 256, top-10). AITER is not usable on gfx12.

## 4. Persistent kernels / megakernel

- **Mirage MPK**, arXiv 2512.22219 (Dec 2025) **[abstract]**: compiler+runtime generates from the model a megakernel with an SM task graph, up to 1.7× lower latency. Target: **decode latency**.
- **Hazy Research "Look Ma, No Bubbles"** (2025-05-27) **[docs]**: Llama-1B, batch 1, decode, 78 % of the bandwidth, > 1.5× vLLM/SGLang. Follow-up "How Many Llamas…" (2025-09-28): batch size 1, multi-GPU. **No prefill.**
- **Cohere megakernel blog** (2026-09-08) **[docs]**: decode megakernel (30B-A3.3B, H100, up to batch 8); "Prefill runs as ordinary PyTorch kernels and pauses decoding"; overall gain 1.25–1.41× (decode alone 1.58×).
- **Ada-MK**, arXiv 2605.11581 (2026-05-12) **[abstract]**: megakernel for Ada (L20), decode; prefill runs separately in TensorRT-LLM.
- **HipKittens**, arXiv 2511.08083 (Nov 2025) **[abstract]**: validated on CDNA3/CDNA4; **RDNA not mentioned**.
- **Assessment for us:** prefill is compute-bound. Launches cost 2.1 µs (measured), with ~25 kernels × 48 layers ≈ 2.5 ms per chunk, i.e. under 0.1 %. A megakernel only adds register pressure and scheduling. Benefit **zero to negative** for prefill.
  What is right about the idea is the *epilogue/prologue route* (sections 1 and 3), not the persistence. Only a persistent tile scheduler in the grouped GEMM (distribution of remainder tiles) would be relevant.

## 6. Prioritization (by benefit for the 13,431-token prefill)

| Rank | Step | Expected benefit [estimate] | Effort | Risk |
|---|---|---|---|---|
| 2 | Expert GEMM: measure against the ceiling, then int8 WMMA with gather/SwiGLU/requantization in the epilogue, tile lists | −0.4 to −0.9 s | high | Perplexity (rounding) |
| 3 | HC fusion: up GEMM+mix epilogue, combine+norm+xn16 in one pass, possibly R in bf16 | −0.45 to −0.65 s (4–6 %) | medium | low; bf16 R: quality |
| 4 | GDN chunk path (C = 16/32, WMMA, possibly Neumann inverse) | −0.2 to −0.35 s | high | Precision of the inverse |
| – | Megakernel / persistent kernel for prefill | 0 | – | – |


## 7. Open points
1. Where does the work of the HC down/up GEMMs stand in our time breakdown (section 1.6, point 5)?
3. Achieved expert GEMM performance (TFLOPS or TOPS) per expert row count, at E = 256, top-10, 12K tokens.
4. Does the mix epilogue layout (4 streams per column) work on the gfx12 WMMA fragment layout without an LDS detour?
5. Triton/fla on gfx1200: no measurement found; use as a reference for correctness only.

## 8. Sources
- mHC: arXiv 2512.24880 (HTML v2 read) <https://arxiv.org/html/2512.24880v2>
- DeepSeek TileKernels `tile_kernels/mhc/` <https://github.com/deepseek-ai/TileKernels>
- vLLM `vllm/model_executor/layers/mhc.py` (main, 2026-10-24), documentation `…/kernels/mhc/tilelang_kernels` <https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/mhc/tilelang_kernels/>
- vLLM commits #58655, #56513, #56503 (ROCm DSv4.1 mHC via AITER), AITER `aiter/ops/mhc.py`, `aiter/ops/mega_mhc.py`, aiter commit #6442 <https://github.com/ROCm/aiter>
- SGLang/LMSYS DeepSeek-V4 day 0 (2026-04-25) <https://www.lmsys.org/blog/2026-04-25-deepseek-v4>
- llama.cpp #28901 (merged 2026-09-16), #25421 (not merged), #29520, #29187, #30817, #28696, #31086; `src/models/qwen4exp.cpp`, `ggml/src/ggml-cuda/{dsv4-hc.cu,mmq.cu,mmq.cuh,mmid.cu}` <https://github.com/ggml-org/llama.cpp>
- GDN: arXiv 2607.16831 <https://arxiv.org/html/2607.16831v1>, 2606.06034 <https://arxiv.org/html/2606.06034v1>, 2607.14568 <https://arxiv.org/abs/2607.14568>, 2605.05838, 2604.21100, 2607.11897 (abstract only)
- vLLM `fla/ops` (v0.21.0 docs) <https://docs.vllm.ai/en/v0.21.0/api/vllm/model_executor/layers/fla/ops/>
- vLLM `fused_moe.py` <https://github.com/vllm-project/vllm>; vLLM forum "Native FP8 WMMA Support for AMD RDNA4" <https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900>
- DeepGEMM Mega MoE (PR #304, Apr 2026); AMD Primus MegaMoE <https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/mega-moe.html>
- GPUOpen WMMA guide RDNA4 part 2 <https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/>
- Megakernel: arXiv 2512.22219 (MPK), 2605.11581 (Ada-MK), 2511.08083 (HipKittens), Hazy "No Bubbles" <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>, Cohere <https://cohere.com/blog/megakernels>
