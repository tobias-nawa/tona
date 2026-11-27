# 18 – MoE prefill with experts in RAM: streaming, layer-major, VRAM split

Agent research (Sonnet), 2026-10-24

Evidence: **[code]** = file:line in GitHub source; **[docs]** = docs/blog/issue text; **[abstract]** = only abstract or search hit read, not the paper itself; **[estimate]** = own calculation, not measured. Nothing was run or re-measured. Numbers from papers are the authors' claims. A source for which the page only provided summaries is marked as such.

## Summary

1. **Streaming the complete non-resident set in a fixed order, before routing, through a ring that continues across layer boundaries** lets the copy engine work during the attention half too. By the measurement we wait 0.81 s in the MoE for copies. That is the indication that here copying starts only after routing.
2. **PCIe bandwidth is not the limit here, the schedule is.** [estimate] 166 missing experts per layer × 2.03 MB = 337 MB, that is 6 ms at 55.7 GB/s, against ~34 ms compute time per layer and piece. The link is ~18 % utilized.
3. **Layer-major exists as an idea in the literature, with numbers:** Layered Prefill (2510.08055, MLSys 2026), MoE-Gen (2503.09716), MoE-CORE (2610.01950), FreeToken (2608.16157), MoE-Prefill (2605.02960). Keeping the residual streams in VRAM is cheap (0.49 GB at 12K, 1.3 GB at 32K). Offloading them to RAM almost eats the gain at 12K (see 2.3).
4. **The measurement point fits a simple model:** prefill time = n · (1 + r/chunk) with r ≈ 405 token equivalents reproduces +9 % (4,096) and +12 % (6,144). For layer-major (one pass) this implies a gain of about +16 to +20 % at 12K and 32K [estimate, two parameters, three points, read with caution].
5. Compression during transfer, CPU experts and predictors bring nothing in prefill here (reasons in section 5).

---

## 0. Base numbers for the estimates (all [estimate] from the given figures)

| Quantity | Value | Calculation |
|---|---:|---|
| Expert | 2.03 MB | 520 MB / 256 |
| Missing per layer (90 of 256 in cache) | 166 experts = 337 MB | |
| One full pass over all 48 layers | 7,968 experts = **16.2 GB** | if all missing ones are needed |
| Duration of one pass at 55.7 GB/s | 0.29 s | |
| Compute time per chunk (2,048) and layer | ~34 ms | 2,048 / 1,264 tok/s (prefill) = 1.62 s, /48 |
| Transfer per layer and chunk | 6.0 ms | 337 MB / 55.7 GB/s |
| Residual streams | 40 KB/token | 4 × 2,560 × fp32 |
| State 12K / 32K tokens | 0.49 / 1.31 GB | |

"Almost all experts are needed" rests on two sources: with uniformly distributed routing the chance that an expert is never chosen by 2,048 tokens is (1 − 10/256)^2048 ≈ e^-80; and llama.cpp issue #25859 measures the union at pp2048 as "essentially all 256" [docs]. Selective copying (llama.cpp PR #15346, merged 2025-08-20) pays off only with very small batches.

**Transfer volume per prompt**, if all missing experts are needed in each pass:

| Method | 12K (12,288 tok) | | 32K (32,768 tok) | |
|---|---:|---:|---:|---:|
| | Passes, volume | pure DMA time | Passes, volume | pure DMA time |
| chunk-major 2,048 (today) | 6 → 97 GB | 1.74 s | 16 → 259 GB | 4.65 s |
| chunk-major 4,096 | 3 → 49 GB | 0.87 s | 8 → 130 GB | 2.33 s |
| chunk-major 6,144 | 2 → 32 GB | 0.58 s | 6 → 97 GB | 1.74 s |
| chunk-major 8,192 | 2 → 32 GB | 0.58 s | 4 → 65 GB | 1.16 s |
| chunk-major 16,384 | 1 → 16 GB | 0.29 s | 2 → 32 GB | 0.58 s |
| **layer-major, state in VRAM** | **1 → 16 GB** | **0.29 s** | **1 → 16 GB** | **0.29 s** |
| layer-major, state in RAM | 16 GB + 47 GB state = 63 GB | ~1.0 s, see 2.3 | 16 GB + 126 GB = 142 GB | ~2.3 s |

Important: the DMA time does not stand for the prefill time. At 12K the prefill takes ~9.7 s; the 1.74 s of DMA are largely hidden under compute. Of this, 0.81 s are measured as not hidden.

---

## 1. How others do it

### 1.2 llama.cpp

- **`-ncmoe` / `--n-cpu-moe N`**: puts the experts of the first N layers on the CPU [docs, HF guide Doctor-Shotgun]. Static, no dynamic cache.
- **Op offload** [code]: `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32)`. From 32 tokens in the batch the scheduler copies the CPU weights to the GPU for that batch and computes there, otherwise the CPU computes. Since PR #15346 (merged 2025-08-20) only the experts used in the batch are copied [docs, PR title "copy only the used experts"]. ik_llama.cpp: threshold 32 · experts/active experts [docs, HF guide, not checked in code].
- **ubatch is the lever.** The batch is copied piece by piece (`-ub`) per layer, i.e. **chunk-major** with a new copy per ubatch. Field reports: ubatch 512 → 8,192 lifts prefill from ~380 to ~2,091 tok/s (gpt-oss-120b, RTX 3090, `-ncmoe` has to rise so that the compute buffer fits; Reddit, May 2026, single measurement [docs]); Vulkan/Vega 512 → 4,096 roughly doubles (comment in #25859).
- **Issue #25859** ("Offloaded-MoE prefill leaves the GPU idle waiting on serial expert H2D copies", 2026-07-18, **open**): RTX 3060, Qwen3.6-35B-A3B Q4_K_M, pp2048: GPU 42 % idle. Page-lock +21 %, copy on a second stream parallel to compute → ~1,143 → ~1,880 tok/s (prefill, +64 %), idle down to "a few percent". In addition: omit the D2H readback of the routing IDs for large batches, because the selection is almost everything anyway. [docs, author's claim, fork, no PR]. A commenter found no gain with double buffering per **single** expert (copies too small), so prefetch per layer is necessary, not per expert.
- **PR #21067** ("ggml: allow prefetching tensor overrides", 2026-03-27, **open, draft**, CUDA only, `--prefetch-weights`, `--no-mmap` required): layer n+1 copies while n computes. The author writes himself that with MoE the selective copy stands in the way of prefetch (one does not know which experts the next layer chooses), but that prefetch wins with larger ubatch (gpt-oss-20b, ubatch 1024/2048 "much better", chart, no numbers in the text) and becomes rather slower with large, wide MoE models. [docs]. That is no problem here because we need the whole layer anyway.
- **Benefit for us:** confirmation of the pattern prefetch + large batches. Nothing to adopt; llama.cpp is weaker here. Effort: as a reference only.

### 1.3 KTransformers / SGLang / vLLM

- **KTransformers** (SOSP '25, Seoul 2025-10-13 to 16; Tsinghua MADSys): experts stay on the **CPU**, AMX kernels compute them in prefill too (LMSYS blog 2025-10-22: ~21.3 TFLOPS per Xeon socket). "4.62–19.74× prefill" against other systems; **expert deferral** (part of the expert output goes only into the layer after next, so that CPU and GPU overlap: CPU utilization <75 % → ~100 %, up to 1.45× throughput, ≤0.5 % accuracy loss) [abstract/blog]. A "layerwise prefill" mode that pushes experts layer by layer to the GPU was **not** found; the docs name only `--chunked-prefill-size` to lower VRAM and `--kt-gpu-prefill-token-threshold` (example 512) for dynamic redistribution of the GPU experts [docs, experts-sched tutorial]. SGLang integration: issue #11425 (2025-10-10, **closed** 2026-02-08, reason not checked).
  **Benefit for us: low.** The approach relies on server CPUs with AMX; a Ryzen 9600X (6 cores, 58 GB/s) computes ~16 MFLOP per token and expert (IQ1/IQ2) nowhere near as fast as the GPU fetches the 2 MB in 36 µs (section 5.2). Expert deferral changes the computation (model output) and is a decode topic.
- **vLLM** [docs]: static CPU offload (PR #34535 "Selective CPU Weight Offloading", merged 2026-02-14), prefetch offloader v2 (PR #29941, merged 2026-02-26, hides the onloading behind compute), both layer-wise on whole weights, not expert-wise. **RFC #38256** ("Incremental MoE Expert Offloading", 2026-03-26, **open**; PR #37190 open): GPU cache with LFRU, CPU pinned backing, in prefill the expert IDs of the batch are deduplicated and loaded once each (author's estimate: 32 instead of ~12,000 loads at 3K context, tinyserve, not vLLM). If the batch does not fit in the cache, PR 1 aborts; community fallbacks (split tokens = bit-identical; group experts = not bit-identical, error grows, token agreement 39–58 %) [docs]. Benefit: warning that "summing experts in groups" is not numerically identical.
- **FreeToken** (arXiv 2608.16157, 2026-08-17, Yang … Stoica; edge MoE serving): **the closest counterpart to us**. Prefill loads the **complete** expert set of layer l+1 into one of two full-layer buffers while layer l computes, **before** the routing is known. The buffers come from the same slot pool as the decode cache, "no separate prefill cache and no phase handoff", leftovers from prefill feed the decode cache; if two layers do not fit, it falls back to loading on demand. RTX 5090, PCIe 5.0 x16, Qwen3.6-35B BF16: 64.4 GB expert pool per chunk at **52.7 GB/s**, chunk 8,192 in 1.19–1.22 s, 6.7k tok/s (prefill) at 16K. Without the second buffer −19 % (4K), −25 % (8K), −26 % (16K) [abstract/HTML page, summarized via web fetch, numbers not checked against the PDF]. **Also chunk-major there.**

### 1.4 Papers on offloading (2024–2026)

Classification: most target **decode** or small batches. For prefill here only the rows with "prefill" count.

| Source (date) | Core idea | Prefill-relevant? | Benefit for us | Effort |
|---|---|---|---|---|
| **MoE-Lightning** 2411.11217 (2024-11; ASPLOS '25 per a third-party page, unchecked) [abstract] | CGOPipe: CPU-GPU I/O pipeline with paged weights, hierarchical roofline model chooses the batch size; up to 10.3× on T4 (Mixtral 8x7B). | Yes, batch throughput | The roofline thinking (transfer vs. compute per expert) confirms the calculation in section 0; code not transferable (offline batch, CPU attention). | low (model only) |
| **MoE-Gen** 2503.09716 (2025-03-12; code `github.com/EfficientMoE/MoE-Gen`) [abstract + HTML page] | **Module-based batching**: tokens accumulate in host memory, each module (attention, experts) gets its own large batch size; experts are fetched once per large batch; KV entirely in the host. A5000, PCIe 4.0 (32 GB/s). Roughly 2^10 tokens per expert saturate the GPU, >2^11 hide the transfer. | Yes | **Expert-major over the whole prompt** (not only layer-major). For us: at 12K tokens × 10 / 256 ≈ 480 rows per expert instead of ~80 at 2,048 → better GEMM utilization. Risk: intermediate buffer T·10 rows (12K: 123K rows × 2,560 × 2 B = 630 MB fp16 for the inputs alone). | high |
| **Layered Prefill** 2510.08055 (MLSys 2026; vLLM, 2×H100) [abstract + HTML] | Model split vertically into G layer groups; a prompt runs **without chunking** through one group per iteration, each layer "exactly once". Redundant expert loads: Qwen, arXiv prompts 35.6 → 21.7 TB (−39 %), ShareGPT 28.5 → 25.1 TB (−12 %); chunked traffic falls with chunk size: 512 → 955 GB/request, 1,024 → 631, 2,048 → 304, 4,096–8,192 <100 GB (8K prompt). TTFT up to −70 %. | Yes, **layer-major in serving** | Documents the mechanism (traffic ∝ number of chunks) with measurement; the paper says nothing on the intermediate state between the groups. | medium |
| **MoE-Prefill** 2605.02960 (2026-05) [HTML, partly read] | Prefill serving with expert backing in pinned CPU RAM; window of the next k layers on the GPU; NVLink AllGather of the next layer plus PCIe prefetch further ahead; optionally "KV-cache-free". 1.35–1.59× throughput. | Yes, but multi-GPU | Sliding-window idea; window size k not given. | low |
| **MoE-CORE** 2610.01950 (2026-10-01; Ascend NPU) [HTML] | In prefill two buffers B0/B1 with **one whole MoE layer each**, layer l+2 loads while l+1 computes; threshold N* = ⌊K/k⌋ tokens (otherwise selective residency). Prefill and decode region **physically separate** (named as a gap in the paper: "prefill and decode caches are still separate"). TTFT +1.4–4.1 % compared to load-on-demand, decode much better. | Yes | Confirms the two-buffer pattern; the separate regions are the counterexample to FreeToken. | low |
| **FreeToken** 2608.16157 (2026-08-17) | see 1.3 | Yes | Best evidence that double buffering brings 19–26 %, and for a shared slot pool. | medium |
| **Apex (Adaptive Expert Assignment)** 2610.23006 (2026-10-16; builds on KTransformers and SGLang, RTX A6000, PCIe 4.0) [HTML] | High token load → copy the expert to the GPU, low load → compute on the CPU; chooses p so that max(CPU path, GPU path) is minimal; throughputs by moving average. Prefill +31.2 % on average, up to +51.9 %. | Yes | Decision rule usable, benefit small here (5.2). | medium |
| **DALI** 2602.03495 (2026-02-03; EPYC 7532, RTX 3090, PCIe 4.0) [HTML] | Greedy CPU/GPU assignment per expert by load, prefetch, cache; PCIe "up to 78.1 % of execution time"; prefill 7.62× vs. llama.cpp, 2.00× vs. HybriMoE. No prefill-specific mechanism. | partly | Like Apex. | medium |
| **HybriMoE** 2504.05897 (2025-04, PKU, code `PKU-SEC-Lab/HybriMoE`) [abstract] | Dynamic intra-layer CPU/GPU planning, inter-layer prefetch, score-based cache; 1.33× prefill, 1.70× decode versus hybrid planners. | yes | Like Apex. | medium |
| **Fiddler** 2402.07033 (ICLR 2025) [abstract] | Small inputs: expert on the CPU; large: copy weights to the GPU. 8.2× (Quadro RTX 6000), 10.1× (L4) for single-batch latency. | yes (rule "large → copy") | The rule applies here in the extreme form: always copy. | trivial |
| **DAOP** 2501.10375 (DATE 2025) [abstract] | Dynamic CPU/GPU assignment per sequence, precomputation of predicted experts on the CPU; numbers from decode. | no | – | – |
| **Klotski** 2502.06888 (2025-02-09) [abstract] | Overlap several batches so that the compute of the current layer covers the loading of the next; more batches = more active experts. | indirectly | Same principle as MoE-Gen, weaker. | – |
| **FloE** 2505.05950 (ICML 2025) [abstract] | Compress expert matrices (~9.3× per expert on Mixtral, sparsification plus quantization), thin predictor; 11 GB VRAM. | decode | See 5.1: we are already at ~2 bit. | – |
| **MoE-Infinity** 2401.14361 (v3 2025-03), **ProMoE** 2410.22134, **Pre-gated MoE** (ISCA 2024), **SeqMoE** 2609.12978 (2026-09-11; 45 % residency → 96.97 % hit rate, 80.22 % of full-load performance, decode) [abstract] | Predicting the next experts for cache/prefetch | no, decode | In prefill almost all experts are needed, a predictor saves nothing. | – |

None of the above sources was read in full text unless "HTML" is stated explicitly. For **ProMoE**, **MoE-Infinity** and **Pre-gated MoE** there were only search hits; their classification (decode) follows from the titles and from other papers that cite them.

---

## 2. Layer-major instead of chunk-major

### 2.1 Who does it?

- **Layered Prefill** (2510.08055): yes, in serving, GPU-resident. "Each layer is traversed exactly once." Measures fewer expert loads (above).
- **MoE-Gen**: expert-major over the whole prompt, state in the host.
- **MoE-CORE / FreeToken / MoE-Prefill**: full-layer buffers with prefetch, but the batch is per request a chunk run or the whole prompt in one step; chunk processing across several layers is not described.
- **llama.cpp, vLLM, KTransformers**: chunk-major (ubatch).

### 2.2 Costs

| Item | 12K | 32K | Note |
|---|---:|---:|---|
| Residual streams (4 × 2,560 × fp32) | 0.49 GB | 1.31 GB | fp16 would halve, but changes the computation (bit-identity lost) |
| Expert ring: one missing layer double-buffered | 0.67 GB | 0.67 GB | 2 × 337 MB |
| Sum of additional VRAM | ~1.2 GB | ~2.0 GB | against what larger chunks (6,144) cost: unknown, **to be measured** |
| Returning the loan after the prompt (refill) | 21 ms | 36 ms | 1.2 or 2.0 GB / 55.7 GB/s, [estimate], matches the 20–40 ms |

The intermediate storage thus need **not** be a problem as long as it stays in VRAM. The argument is a comparison of numbers: 1.3 GB at 32K is about 640 expert slots of 4,320 (15 %) for the duration of the prompt.

### 2.3 State in RAM instead of VRAM

If the state per layer travels over PCIe and back, this costs at 12K 2 × 0.49 GB × 48 = **47 GB**, at 32K 126 GB, plus the 16 GB of experts. The link manages only 60.8 GB/s in both directions together. At 12K layer-major with host state (63 GB) thus lies at the level of chunk-major with 4,096 (49 GB): **no gain**. At 32K (142 GB against 259 GB at 2,048) it is better, but the VRAM state (1.3 GB) is clearly cheaper. The other engine reaches the same trade-off in `lm_begin`. Furthermore expert DMA and state DMA would compete for the same link.

### 2.4 DeltaNet states, attention KV

Both are separate per layer. Layer-major needs at any time only the running DeltaNet state of **one** layer (recurrence over the pieces in order, otherwise unchanged) and writes the KV of this layer for all T tokens in one go. The memory for KV and states of all layers stays as today. To check are (a) the snapshots of the DeltaNet state at chunk boundaries for the prefix cache (research/10): in layer-major they arise layer by layer and must be stored before the next layer, (b) checkpoints on abort, (c) the attention half of every layer with the QSA/indexer ring, whose state runs across pieces. Evidence in the literature that someone has done this for hybrid models (DeltaNet + attention) was **not found**.

### 2.5 Forecast from the measurement points [estimate]

Model: `t = n · (1 + r/c)`, c = chunk size, r = fixed overhead per pass in token equivalents.
From 2,048 → 4,096 (+9 %) follows r ≈ 405. With that 6,144 gives +12.4 % (measured +12 %). This is a consistency check (three points, one parameter after fitting), not a rule.
- 405 token equivalents ≈ 0.32 s per pass ≈ the pure DMA time of one pass (0.29 s). That means the **overhead per chunk is almost a whole expert transfer, hardly hidden** (measured, however, only 0.13 s waiting time per chunk; the rest may be kernel boundary effects, unexplained).
- layer-major (one pass, c = n): 12K → 1.033 n instead of 1.198 n → **+16 %** (~1,470 tok/s, prefill). 32K → +18 %.
- Overhead removed completely (perfect prefetch): +20 %.
- Kernel quality is a further factor.

**What the model does not separate:** how much of the +9/+12 % comes from less transfer and how much from better-filled expert GEMMs (more rows per expert). Measurement proposal: compare prefill with all experts resident (e.g. 8–12 layers without streaming) at 2,048 against 6,144. If a difference already shows there, layer-major with 2,048 slabs brings less than the model says; then expert-major (MoE-Gen) or larger slabs.

---

## 4. Redistributing VRAM between prefill buffer and expert cache

| Source | Mechanism | State |
|---|---|---|
| **FreeToken** (2026-08) | One slot pool for prefill full-layer buffers and decode cache; leftovers from prefill stay in the decode cache. | paper |
| **MoE-CORE** (2026-10) | **Separate**: prefill region and decode region fixed; named as a gap in the paper. | paper |
| **KTransformers/SGLang** | `--kt-gpu-prefill-token-threshold`: from N prefill tokens dynamic redistribution of which experts sit on the GPU [docs]. The redistribution concerns the expert selection, not the buffer size. | in docs |
| **vLLM RFC #38256** | Open question 2: "Whether slot buffers should be counted during memory profiling". | open |
| **llama.cpp** | none; `-ncmoe` and `-ub` are fixed at start time, the compute buffer grows with ubatch (field report: ub 8,192 forces `--n-cpu-moe` from 26 to 28). | – |

Benefit for us: the idea ("prefill borrows slots, refills afterwards, ~20–40 ms") is confirmed by FreeToken. There is no argument against it in the literature. The drawback remains: as long as the loan is out, decode lacks the slots. Effort: medium, if our cache is addressable by segment.

---

## 5. Further levers for less transfer in prefill

### 5.1 Compression during transfer
Our experts are already IQ1_M/IQ2 (~2 bit). FloE (9.3× per expert) uses sparsification plus quantization on FP16 experts (Mixtral), the factor cannot be had here. A lossless entropy coding on the GPU (nvCOMP-like) brings little for already packed codebook values; this is an [inference], not measured. **Benefit: low, effort: high. Do not pursue.**

### 5.2 CPU computes rare experts (Fiddler, DALI, HybriMoE, Apex, KTransformers)
[estimate] One expert = 2.03 MB, ~8 M weights, ~16 MFLOP per token. Transfer: 36 µs at 55.7 GB/s. CPU (6 Zen 5 cores, optimistically 0.5 TFLOPS effective with IQ dequantization): 32 µs per token, i.e. for n tokens 32 µs · n. The CPU wins only at **n ≤ 1 token**; in a 2,048 chunk the smallest expert is mostly at n ≥ 5. In addition the CPU shares the 58 GB/s RAM read bandwidth with the DMA (55.7 GB/s). Apex (+31 % on average) and DALI work with PCIe 4.0 and 12 MB+ BF16 experts (Apex: 3.80 ms transfer per layer against 0.73 ms GPU compute at batch 1). Here the experts are small and the link fast. **Benefit: ~0 in prefill.** The other engine's own CPU share applies only to chunks <1,024 tokens (and is off on HIP).

### 5.3 Order of the experts by occupancy
In prefill almost everything is chosen; sorting by occupancy helps only where the transfer does not finish: **frequent ones first into the ring** (the ring is worked off in ID order, compute follows the ID order). Alternatively choose the set of **resident** experts by prompt profile: the other engine has a "hot-expert profile" and a routing-prior option (opt-in, investigated on an RTX 3060, not default). Benefit: small, because only 90 of 256 are resident and the rest is fetched anyway. **Do not tackle before the schedule rework.**

### 5.4 Overlapping transfer and compute
- **Full-layer prefetch without routing** (FreeToken, MoE-CORE, llama.cpp #25859): **the most important lever.** Measured +64 % (llama.cpp, RTX 3060, fork), −19…−26 % without a second buffer (FreeToken). Here the 0.81 s wait (8.5 % of 9.5 s at 12K) will almost completely disappear if layer l+1 is loaded completely (337 MB) already during the attention/DeltaNet half of l [estimate].
- **Offset the copy order against the compute:** compute starts with resident experts (no transfer needed) while the ring fetches the missing ones.
- **Pinned RAM, no intermediate buffer**: according to our data the path is pinned (55.7 GB/s). On systems with partly unpinned RAM the host copies slow things down; with 64 GB this should not happen.
- **No D2H readback of the routing IDs for a large batch** (llama.cpp #25859: hard synchronization). To check here: does our router read the top-k IDs to the CPU?
- **Do not push state over PCIe at the same time** (see 2.3), because only 60.8 GB/s in both directions together.

---

## 6. Recommendation

Order by benefit/effort. Numbers are [estimate] and have to be measured.

1. **Preliminary measurement (small):** (a) prefill with all experts resident at 2,048 against 6,144, to separate GEMM effect and transfer effect; (b) `nsys` timeline of one layer at 12K: is the copy job started only **after** the routing? How long is the DMA gap per layer (expected ~3 ms of 6 ms)?
2. **Prefetch of the complete miss set of the next layer** through a ring of ≥ 1.5 layers (≈ 500 MB, ~250 slots), in fixed order, running across layer boundaries (like FreeToken). Expected: wait time 0.81 s → ~0, i.e. **up to +8 %** at 12K, independent of chunk size. Effort medium, risk small, output bit-identical.
3. **Layer-major with state in VRAM** (not in RAM) and **loan from the expert cache** for state and ring (≈ 1.2 GB at 12K, 2.0 GB at 32K, return 21–36 ms): transfer 97 → 16 GB (12K), 259 → 16 GB (32K); forecast **+16 to +20 %** compared to today, more than the +9…+12 % of the larger chunks, without enlarging the chunk buffers. Effort high (loop inversion, snapshots for the prefix cache, DeltaNet/QSA state per layer), compute precision unchanged because the slab size stays the same; output should stay bit-identical provided the order of rows in the expert GEMMs is the same.
   **Fallback if (1a) shows that the larger GEMMs carry the gain:** layer-major with larger slabs (4,096 to 6,144) for the MoE part, attention/DeltaNet at the 2,048 pace, or expert-major like MoE-Gen.
4. **Loan as the default also for the chunk buffers (if 3 is dropped):** 4,096–6,144 by loan instead of fixed reservation. This removes the −3…−6 % from decode, because the slots come back after the prompt.
5. **Do not pursue:** CPU experts in prefill, transfer compression, predictors (MoE-Infinity, ProMoE, SeqMoE: decode topics).

**Best combination:** 2 (full-layer prefetch) first, then 3 (layer-major, state in VRAM, loan). Together the overhead per pass should fall from ~405 token equivalents to ~0. Step 1 tells whether the model holds.

## Open points

- The FreeToken numbers (52.7 GB/s, 19–26 %) come from a web summary of the HTML page, not from the PDF.
- Whether KTransformers has a "layerwise prefill" mode (earlier docs mention something like it) was not confirmed.
- papers.md is not updated (only this file); entries for 2610.01950, 2608.16157, 2610.23006, 2510.08055, 2609.12978, 2605.02960, 2602.03495 are still to be added.
