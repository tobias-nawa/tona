# 38: MoE inference on Apple silicon, unified memory and SSD: paper survey

For the Mac backend (#107); short list in [papers.md](papers.md). Agent research (Opus); arXiv IDs and titles of the
nine entries the proposal relies on re-checked by hand against the arXiv API (all exist, titles match). **Search: 2026-11-11** (agent research). Method: arXiv API queries (`arxiv.org/api/query`, sorted by
submission date, at most 40 hits per query) for SSD/flash MoE offloading, expert cache, prefetching/prediction, miss
handling (skipping, substitution, top-k reduction), speculative decoding with offloading, Apple silicon / MLX / Metal /
unified memory, and mixed-precision quantization, pruning and merging of MoE experts. Abstracts were read through the API
for every entry. Full texts were read through arXiv HTML with WebFetch summaries, plus `pdftotext` for 2610.31479. Each
row is marked "abstract" or "full text". "Full text" means a WebFetch summary of the HTML page, with targeted questions,
not a line-by-line read. Pages over 100K characters were read only up to 100K. **All numbers are the authors' claims**;
own derivations are marked [estimate].

**Gaps:**
- Each query returned at most 40 hits, so older hits inside a topic may be missing.
- No search outside arXiv: no ACM DL, USENIX or OpenReview full search, and no Semantic Scholar citation chasing.
- The engines and products (MLX, llama.cpp Metal, Flash-MoE, Metal watchdog) were left to the other agent.
- Papers already in `papers.md` are not listed again, except where this draft adds new numbers (marked "already listed").
- Classic 2023-2024 work on PC offloading (MoE-Infinity, ProMoE, Pre-gated MoE, HOBBIT, FATE, …) is covered in
  `14-expert-prefetch.md`. Only the entries with Mac or flash relevance are listed here.

Our numbers, which every "Relevance" verdict below uses (measurements 2026-11-11, research/21-backends.md):
- M1 Max with 32 GB; the GPU may use 26.8 GB, reads at 334 GB/s; the CPU reads at 58-115 GB/s.
- SSD: 13 GB/s for 2 MB reads at queue depth 16, 3.5 GB/s and 0.6 ms at depth 1, ~7 GB/s for 0.68 MB blocks.
- Metal: ~2 µs per dependent dispatch, 170 µs per command-buffer round trip.
- qwen4exp:
  - experts: 48 × 256 = 12,288 experts at 2.046 MB = 25.1 GB; top-10 + shared means 480 routed expert reads per token.
  - dense weights: 4.1 GB.
  - [estimate] After dense weights, KV and buffers, at most about 20 GB is left for experts, i.e. at most ~80 % of the
    experts resident. With the desktop and other programs on a 32 GB Mac the safe size is smaller (MawForge below;
    [measurements/2026-11-11-mac-ssd.md](measurements/2026-11-11-mac-ssd.md) assumes 14 to 18 GB, 56 to 72 %).

## 1. Flash/SSD offloading of experts and weights

| Date | Paper | Key statement | Relevance | Status |
|---|---|---|---|---|
| 2023-12 (ACL 2024) | [LLM in a flash, arXiv 2312.11514](https://arxiv.org/abs/2312.11514) (full text, 8 auth.) | **M1 Max with 1 TB SSD**. Flash: 6.10 GB/s for dense reads, 1.25 GB/s for sparse reads, 2.25 GB/s with row-column bundling (32 KiB chunks); "32 KiB or larger random reads across multiple threads" needed; uncached reads via `fcntl(F_NOCACHE)`. OPT-6.7B on Metal M1: I/O 2,196 → 92 ms/token (window k=4-5) | **high**: same machine; F_NOCACHE and parallel reads carry over. Bundling targets dense FFN neurons, not needed for 2 MB experts | new |
| 2026-11 | [PagedMoE, arXiv 2611.03173](https://arxiv.org/abs/2611.03173) (full text, 6 auth.) **already listed (abstract, "low")**, new numbers | Experts stored once as **bitplanes**: lower precision is a prefix of higher precision, so a cached low-bit expert is extended by reading only the missing planes. Precision per token: greedy over Σ g²·ε(e,b), with Σb ≤ cK. Qwen3.5-397B, 24 GiB cache, 9950X + RTX 5090: Uniform4 5.71 → Dyn Avg2.5 8.89 tok/s (1.56×); SSD traffic 2.08 → 0.69 GiB/token; GSM8K 97.04 → 96.89, GPQA 51.01 → 52.53, WikiText-2 PPL 3.223 → 3.240. Byte hit rate 47.9 → 68.5 % from cross-precision reuse | **high** for the Mac (SSD tier, 13 GB/s), unlike the PC (RAM tier). Needs its own kernels and format; IQ quants are not bitplanes | re-rated |
| 2026-10 | [DeepInfer-Mac, arXiv 2610.36564](https://arxiv.org/abs/2610.36564) (full text, 1 auth.) | DeepSeek-V4.1-Flash (748B, 338 GiB on disk) on a **MacBook Air M4, 24 GB**: 7.6 GiB resident, 4.5 GiB LRU expert slots. Each expert is one contiguous record padded to 16 KiB, read uncached **directly into a reusable GPU buffer** (no staging copy). 2.0 GiB of expert reads per token; 0.21-0.72 tok/s (decode). Next-layer prefetch +21.0 % over overlap alone (0.92 → 1.11 tok/s), for +5.4 % reads | **high**: closest public design to ours (UMA, direct read into Metal slots, prefetch into the next layer). Hit rate and SSD bandwidth not reported | new |
| 2026-09 | ["The Other Half of the Memory Wall" (called Edge0 in its text, per the agent), arXiv 2609.18063](https://arxiv.org/abs/2609.18063) (full text, 5 auth.) | **Mac mini M4 Pro, 24 GB** (A/B test on M2, 16 GB). Qwen3.6-35B-A3B int4, 1.77 MB per expert, served at K=4 instead of 8. A trained per-layer "prerouter" predicts next-layer routing one token ahead, and **the prediction becomes the routing**, plus a recovery LoRA. 20.4 tok/s (decode) in 2.9 GiB vs. 3.9 tok/s for resident mlx-lm; quality −3.9 points on average (AIME 92.7 → 86.6, HumanEval 95.1 → 90.9). On-demand reads average 0.27 MiB vs. 1.32 MiB with the prerouter; load cost 1.17 ms + 1.33 ms × cold fraction | medium: lossy and needs training; confirms that whole-expert reads (≥1 MB) beat per-tensor reads on Apple SSDs | new |
| 2026-09 | [SSD-LLaMA, arXiv 2609.18110](https://arxiv.org/abs/2609.18110) (full text, 7 auth.) | Expert gate/up/down **back to back in one aligned block, one O_DIRECT read per expert** via io_uring into pinned slots; rANS compression of the packed 4-bit weights cuts SSD traffic by 33 % (8.89 → 5.93 GB/token), decompressed on the GPU. Up to 77.7 % of 9 GiB/s sequential. Packed layout 2.6-4.1× faster than GGUF layout. Running misses on the CPU was slower (TPOT 514 → 530-574 ms). Kimi-K2.7 at 1.03 tok/s (decode), RTX 5090 + 32 GB RAM | **high** (layout); medium (rANS: only pays off if decode is cheaper than 1/3 of an SSD read) | new |
| 2026-09 | [Paging the Experts (Routide), arXiv 2609.29032](https://arxiv.org/abs/2609.29032) (full text, 1 auth.) | Swift/MLX runtime on **iPhone 17 Pro Max**: Qwen3.6-35B-A3B, expert 1.6875 MiB, positional reads (no mmap). Fixed-route replay: LRU at 512 MiB **0.00 %** hits, random eviction 18.8 %, LRU at 576 MiB 38.6 %, Belady-like ("pinned farthest-next-use") 54.5 %. Prefetch accuracy 61.3 %. 1.8-3.2 tok/s (decode). Mac controls (M1 Ultra) token-exact across eviction and prefetch | medium: shows the LRU cliff when the cache is barely above the per-token working set. We are at ~80 % resident, far from it | new |
| 2026-07 | [Budgeting Bytes, arXiv 2609.04238](https://arxiv.org/abs/2609.04238) (full text, 1 auth., workshop, single runs) | Windowed roofline: I/O is hidden iff Σ bytes whose fetch window lies inside a compute interval ≤ B·C(I); exposed I/O with predictor accuracy p is (1−p)·s/B. RK3588 eMMC: prefetch, even oracle, does not help when the bus is saturated (0.19 → 0.12 tok/s). **Apple M4 16 GB**: Qwen3-30B Q2_K resident 11.5 tok/s vs. 4-bit overflowing 1.2 tok/s. Pre-attention probe predicts the top-8 at 91.2 % recall (0.84-0.97 per layer) | **high** as a model: tells when prefetch pays (latency-bound) and when it does not (bandwidth-bound) | new |
| 2026-08 | [Kernel-Managed Tiering ("Who Should Own the Expert Cache?"), arXiv 2608.12103](https://arxiv.org/abs/2608.12103) (full text, 4 auth.) | OS page cache as the expert tier vs. a user-space oracle-pinned arena: arena only 1.09-1.11× faster, same bytes read. With a domain shift the static frequency table falls to 21-45 % hits, LRU holds 70-71 %. `posix_fadvise(WILLNEED)` with 64.7 % recall: +0.3 % (noise); perfect one-layer advice +5.0 %; **blocking prefetch reads were slower than none**. Includes a Darwin run (M-series, 128 GB, 16 KB pages) | medium: argues against a frequency-pinned cache; on macOS the page cache cannot be bounded per process (see MawForge), so we keep our own cache | new |
| 2026-06 | [MawForge, arXiv 2607.09686](https://arxiv.org/abs/2607.09686) (full text, 1 auth., 7 pages) | **MacBook Pro M5 Pro, 24 GB**, 18 GiB budget, experts from a pack file into a bounded cache. Gemma-4-26B Q8: cache 35 % → 13.9 tok/s at 86.7 % hits, but **cache 50 % → 2.3 tok/s at 92.9 % hits, 65 % → 1.3 tok/s** (system memory 84 %). Qwen3.6-35B Q8: 15 % cache 8.5-12.1 tok/s, 45 % cache 1.3-1.8 tok/s. MTP-4 speculation: hits 78 → 48 %, 9.5 → 8.3 tok/s | **high (warning)**: past a point a larger expert cache makes macOS compress or page memory and decode collapses. Our 20 GB cache must sit inside a wired budget measured under pressure | new |
| 2025-05 (ICML 2025) | [FloE, arXiv 2505.05950](https://arxiv.org/abs/2505.05950) (abstract, 8 auth.) | Compresses expert matrices for transfer (9.3× per expert on Mixtral), 4.4-7.6 % quality loss | low: lossy and aimed at PCIe; our experts are already at 3.3 bpw | new |
| 2024-06 | [PowerInfer-2, arXiv 2406.06282](https://arxiv.org/abs/2406.06282) (abstract, 6 auth.) | Neuron-cluster I/O pipeline on a phone, 47B at 11.68 tok/s | low: neuron granularity is too fine for 2 MB experts and our SSD sweet spot | new |

## 2. Expert cache and eviction under limited fast memory

| Date | Paper | Key statement | Relevance | Status |
|---|---|---|---|---|
| 2026-06 (v2) | [WiSP, arXiv 2606.21868](https://arxiv.org/abs/2606.21868) (abstract, 4 auth.) | Routing-aware expert pager, byte-identical outputs; RTX 3090 2.0× vs. static offload. **Prefetch does not help single-stream decode** (PCIe-bound: speculative transfers compete with demand transfers). Instead splits memory between experts and KV by marginal latency per byte, up to 1.19× | medium: for us the KV vs. expert split matters for long agent contexts | new |
| 2026-02 | [SpecMD (Least-Stale), arXiv 2602.03921](https://arxiv.org/abs/2602.03921) (abstract, 4 auth.) | Expert access does not follow temporal locality; "Least-Stale" eviction cuts collision misses up to 85× vs. LRU, 88 % hits at 5 % (0.6 GB) cache on OLMoE | low: tiny cache share; 2608.07911 (already listed) found LFRU best under correct replay | new |
| 2026-05 (ICML 2026) | [ReMoE, arXiv 2605.27081](https://arxiv.org/abs/2605.27081) (abstract, 6 auth.) | Router fine-tuning biased towards recently used experts: +26 % reuse, **Jetson Orin NX** llama.cpp TPOT −43.6 to −49.8 % (UFS storage) | low: changes the model (training) | new |
| 2026-09 | [Cache-Aware Joint Router Adaptation, arXiv 2609.04895](https://arxiv.org/abs/2609.04895) (abstract, 9 auth.) | Post-training of backbone plus auxiliary routers for cache retention: hit rate +1.15-18.03 points and traffic −4.6-53.3 % vs. the best prefetch baseline (Qwen3) | low (training) | new |
| 2026-08 | [Cacheable by Design?, arXiv 2608.18261](https://arxiv.org/abs/2608.18261) (abstract, 1 auth., pre-registered) | Qwen3-30B: adjacent-token reuse 2.0× chance, 95 % of traffic on 52.5 % of experts, LRU with 13.4 % of experts serves 66 %. Locality-trained routers miss the ≤ 1 % PPL gate; training-free cache-aware rerouting plus locality gives ~80 % fewer misses at ≤ 3.4 % PPL. Qwen3-235B Q4 from SSD on an 8 GB GPU: 0.44 tok/s | low-medium: reference values for routing statistics | new |
| 2024-11 (TMLR 2025) | [Cache-Conditional Experts, arXiv 2412.00099](https://arxiv.org/abs/2412.00099) (abstract, 8 auth.) | Training-free cache-aware routing (bias towards resident experts) on phones, 2× decode | medium: lossy; superseded by PSR (section 4) | new |
| 2025-05 | [Not All Models Suit Expert Offloading, arXiv 2505.16056](https://arxiv.org/abs/2505.16056) (abstract) | 20 MoEs: **shared experts lower local routing consistency**; domain-specialized experts raise it | medium: qwen4exp has a shared expert, so expect weaker locality than shared-free models | already in 14-expert-prefetch.md |

## 3. Expert prediction and prefetching

| Date | Paper | Key statement | Relevance | Status |
|---|---|---|---|---|
| 2026-08 (CODES 2026) | [APEX, arXiv 2608.11688](https://arxiv.org/abs/2608.11688) (abstract, 3 auth.) | Prefetch router **before attention**, confidence model decides how many extra experts to fetch: "> 99 % overlap accuracy"; correctness-preserving mode −26 % latency per token; a stall-free mode runs on whatever has arrived | **high**: pre-attention prediction gives a whole attention block (12 QSA + 36 DeltaNet layers) of lead time; overbooking by confidence fits 13 GB/s at QD16 | new |
| 2026-06 | [SpecPrefetch, arXiv 2607.24787](https://arxiv.org/abs/2607.24787) (abstract, 7 auth.) | Shared light adapter predicts next-layer experts **only for transfer**; native router still decides, so errors cost time, not quality. Snapdragon 8 Elite: +20 % decode | medium: training needed; same principle as our lossless prefetch | new |
| 2026-09 | [Mira, arXiv 2609.38090](https://arxiv.org/abs/2609.38090) (abstract, 2 auth.) | Per-layer predictors **two layers ahead**, HOT+STAGE two-tier cache (frequent ones kept, predicted ones staged), 5.71× throughput on a memory-limited GPU | medium: the separate staging area protects hot experts from prefetch pollution (cf. `TONA_PRE_KEEP`) | new |
| 2025-08 | [MoE-Beyond, arXiv 2508.17137](https://arxiv.org/abs/2508.17137) (abstract, 4 auth.) | Small transformer over activation traces (DeepSeek-V2-Lite): GPU cache hits 17 → 72 % at 10 % cache (simulation) | low: tiny cache share, simulation only | new |
| 2026-02 | [MoE-SpAc, arXiv 2603.09983](https://arxiv.org/abs/2603.09983) (abstract, 4 auth.) | Speculative-decoding drafts as a "lookahead sensor" for expert demand; prefetch and eviction in one utility space; +42 % TPS over the best SD-based baseline | medium: our MTP/DFlash drafts could feed the SSD queue | new |
| 2025-12 | [OD-MoE, arXiv 2512.03927](https://arxiv.org/abs/2512.03927) (abstract, 6 auth.) | "Emulative" predictor several layers ahead, 99.94 % prediction accuracy (ten-node edge testbed) | low: distributed; the claim needs checking | new |
| 2023-08 | [EdgeMoE, arXiv 2308.14352](https://arxiv.org/abs/2308.14352) (abstract, 6 auth.) | First flash-resident MoE on mobile: per-expert bit width plus preloading | low (historic) | new |

Predictor numbers from section 1, for comparison: Budgeting Bytes, pre-attention probe, Qwen3-30B top-8 recall 91.2 %
(0.84-0.97 per layer); Routide 61.3 %; Kernel tiering PILOT 64.7 % recall. Already in 14-expert-prefetch.md: FATE
78.8 % training-free, 97.2 % with overbooking; quasi-hidden state ~90 %.

## 4. Handling a miss inside the forward pass

| Date | Paper | Key statement | Relevance | Status |
|---|---|---|---|---|
| 2026-09 | [Training-Free Halving of Activated Experts, arXiv 2609.04575](https://arxiv.org/abs/2609.04575) (abstract, 2 auth.) | Use top-k₁ experts but renormalize by the probability mass of top-k₂. Qwen3.6-35B-A3B 8 → 4 experts: MMLU −4.65 with standard renormalization, **−0.35 with k₂=16**. **Qwen3.5-397B-A17B (top-10) 10 → 5: −0.55**. PPL and accuracy prefer different k₂ | **high**: same model family and the same top-10. A dropped miss renormalized with k₂ > 10 should cost less than our `TONA_MISS_DROP` (+1 % NLL at w=0.1). To test | new |
| 2026-10 | [Don't Skip the Baseline, arXiv 2610.28369](https://arxiv.org/abs/2610.28369) (abstract, 6 auth.) | Dynamic expert skipping (4 methods) rarely beats a **static smaller k** at matched compute (Gemma4-26B, Qwen3-30B, Qwen3-VL) | medium: if we drop experts, compare against a static k=8/9 baseline first | new |
| 2026-10 | [Prefetch-Steered Routing (PSR), arXiv 2610.34721](https://arxiv.org/abs/2610.34721) (full text, 1 auth.) | Router bias towards a **deterministic logical cache** (at most n=1-2 predicted experts inserted per layer and token), so outputs do not depend on transfer timing. OLMoE 25 % resident: 19.2 vs. 42.5 ms/token exact LRU (2.2×), +0.5 ± 0.8 points. **Qwen3.6-35B-A3B at 25 %: 30.0 → 24.1 ms**, KL 0.029 vs. 0.051 for Cache-Prior. DFlash block drafter (k=15) on Qwen3.6: 2.1-2.9× vs. the best autoregressive setup at the same memory | medium: lossy but deterministic, so `TONA_DETERMINISTIC` stays possible. Only worth it if the miss rate at ~80 % residency stays high | new |
| 2024-11 | [HOBBIT, arXiv 2411.01433](https://arxiv.org/abs/2411.01433) (abstract, 8 auth.) | Replace less critical **missed experts with low-precision versions**, on llama.cpp: up to 9.93× decode on edge devices | medium: a resident 1-2-bit copy of every expert (~8-10 GB) does not fit next to an 80 % cache; PagedMoE's prefix form does | in 14-expert-prefetch.md (citation only) |
| 2024-08 | [AdapMoE, arXiv 2408.10284](https://arxiv.org/abs/2408.10284) (abstract, 6 auth.) | Sensitivity-based dynamic number of experts: −25 % activated experts, 1.35×, "without accuracy degradation" | low (older, superseded by 2609.04575) | new |
| 2026-08 | [Launch-Bound and Substitutable, arXiv 2608.26612](https://arxiv.org/abs/2608.26612) (abstract, 4 auth.) | INT4 changes 0.53 of 8 experts per token, yet replaying those changed routes in FP16 recovers only 2.7 % of the quality loss, i.e. experts are **substitutable**; leaving routers in FP16 lowers drift by 20 % but raises loss | medium: supports substitution or dropping low-weight misses | new |
| 2026-09 | [ACE, arXiv 2609.05228](https://arxiv.org/abs/2609.05228) (abstract, 8 auth.) | Calibration-free skipping from offline spectral expert statistics plus router gate; at 50 % skipping on Qwen3.6-35B-A3B +4.15 points vs. the best competitor | low-medium: a better "which miss may be dropped" score than router weight | new |
| 2025-12 (ASP-DAC 2026) | [CPU-GPU Collaborative MoE, arXiv 2512.16473](https://arxiv.org/abs/2512.16473) (abstract, 3 auth.) | GPU expert cache, misses computed on the CPU | low on the Mac: in unified memory the CPU does not avoid the SSD read (see "What transfers") | new |

## 5. Speculative decoding combined with expert offloading

| Date | Paper | Key statement | Relevance | Status |
|---|---|---|---|---|
| 2026-09 | [BigMoMo, arXiv 2609.14643](https://arxiv.org/abs/2609.14643) (full text, 10 auth.) | Phones (Snapdragon 8 Elite Gen 5 / 8 Elite / 8 Gen 3, UFS 4.x). Verification window lets experts be read contiguously; **experts are relocated on flash by co-loading** (reads per token, Qwen3-30B: 131 → 94 → 83). Node plus expert pruning: expert requests 0.59×, TPOT 0.77×, GSM8K −0.13, HumanEval −0.05. 1.82× (abstract) / 1.76× (text) vs. the best SD baseline | medium: relocation pays on UFS; on our SSD 2 MB random reads already reach 13 GB/s, so contiguity buys little | new |
| 2026-08 | [S2-MoE, arXiv 2608.15018](https://arxiv.org/abs/2608.15018) (abstract, 3 auth.) | Self-speculation in llama.cpp on edge devices; routing-aware expansion, reuse-aware gating; up to 5.3× (≈2.0× mean) | medium | new |
| 2026-08 (MICRO 2026) | [EdgeXpert, arXiv 2608.05303](https://arxiv.org/abs/2608.05303) (abstract, 5 auth.) | Hardware accelerator: instead of loading the union over draft tokens, load only salient channels plus calibration; −56.3 % latency | low (ASIC) | new |
| 2026-08 | [The Limits of Speculation, arXiv 2609.22156](https://arxiv.org/abs/2609.22156) (abstract, 2 auth.) | Oracle over Qwen3-Coder + EAGLE-3: rejected draft candidates obey a linear rule ΔE[cost]/ΔE[accepted] | low-medium: a theory backdrop for SpecFit/ECOSPEC (already listed) | new |
| 2025-08 | [SpecMoEOff, arXiv 2508.21706](https://arxiv.org/abs/2508.21706) (abstract, 11 auth.) | Speculation enlarges the work per loaded expert; roofline-tuned CPU/GPU split, 2.5× decode | low (PCIe) | new |

Mac-specific results from other sections: MawForge (MTP-4 slower, hits 78 → 48 %); PSR (DFlash block drafter 2.1-2.9× on
Qwen3.6 at 25 % residency); 2610.31479 (section 6: every speculation method lowers throughput on dense 4B models on Apple).

## 6. Papers that evaluate on Apple silicon / Metal / MLX

| Date | Paper | Key statement | Relevance | Status |
|---|---|---|---|---|
| 2026-10 (NeurIPS 2026 ODI workshop) | [Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon, arXiv 2610.31479](https://arxiv.org/abs/2610.31479) (full text via pdftotext, 8 auth.) | M2, M2 Pro, M3 Ultra, M4 Pro, MLX. **Metal megakernel probe** (one LFM2.5-8B-A1B MoE block fused into one dispatch): 174 µs vs. 96 µs stock (1.8× slower). GPU-side dispatch boundary **3.04 µs**, 15 µs per MoE block in total. Hand-rolled grid barrier works, but **plain device stores 96 % stale** across threadgroups; no occupancy query. 24 dependent matvecs run at 77 GB/s vs. 114 GB/s independent (−32.7 %). Speculation (Classic, N-gram, EAGLE-3, DFlash, DDTree) lowers decode on Qwen3-4B 4-bit by 25-70 % (M2 Pro) and 9-62 % (M3 Ultra); DFlash accepts 2.57 but 156.7 → 60.0 tok/s. Private `air.simdgroup_async_copy` exists, public MSL has no async copy. MoE gather kernel +3-12 % | **high**: matches our 2 µs/dispatch; argues for many dispatches with dependency-aware scheduling, not a grid-barrier megakernel. The speculation result is for a small dense model; with experts on SSD the balance shifts | new |
| 2026-07 | [BaseRT, arXiv 2607.00501](https://arxiv.org/abs/2607.00501) + [M5 follow-up, arXiv 2607.19438](https://arxiv.org/abs/2607.19438) (abstracts, 3 auth.) | Native Metal runtime: decode up to 1.56× vs. llama.cpp, 1.35× vs. MLX (M3, M4 Pro); larger margins on MoE prefill. M5 Neural Accelerators (Metal 4 tensor API): prefill up to 6.4× vs. llama.cpp | medium: yardstick for a native Metal path; M1 Max has no tensor units | new |
| 2026-04 | [NPUMoE, arXiv 2604.18788](https://arxiv.org/abs/2604.18788) (abstract, 2 auth.) | MoE prefill on the Apple Neural Engine with static capacity tiers and grouped experts; 1.32-5.55× lower latency, 1.81-7.37× energy | low-medium: ANE as a prefill helper, not tested for decode | new |
| 2026-06 | [Above the Inner Loop (M1 AMX), arXiv 2606.25426](https://arxiv.org/abs/2606.25426) (abstract, 1 auth.) | Direct AMX fp32 GEMM on M1: second AMX block via multi-thread panels plus pre-packed weights beats Accelerate (1.17× vs. BNNS Graph); llama.cpp prefill 291 → 420 tok/s (128 tokens) | low: fp32 only, CPU side; not for IQ experts | new |
| 2026-09 | [SiliconBench, arXiv 2609.19169](https://arxiv.org/abs/2609.19169) (abstract, 5 auth.) | 9 Apple serving engines; **"explicit memory budgets do not guarantee memory headroom"**: two stacks run close to physical capacity and slow down | medium: same warning as MawForge | new |
| 2026-08 (CCS 2026) | [SLAC, arXiv 2608.09075](https://arxiv.org/abs/2608.09075) (abstract, 5 auth.) | Reverse-engineered M1 SLC set indexing; GPU accesses visible in the shared SLC | low: shows CPU and GPU share the SLC, which matters if the CPU writes SSD reads while the GPU computes | new |
| 2026-08 | [RotaryQuant, arXiv 2608.08081](https://arxiv.org/abs/2608.08081) (abstract, 3 auth.) | Fused Metal pipeline: 4-bit dense, **2-bit routed experts, 8-bit shared expert**, LRU disk paging, 3-bit KV; Qwen3-30B in 16 GB, Nemotron-H 120B in 32 GB at 9-19 tok/s | medium: role-based bit allocation (shared expert high) | new |
| 2026-05 | [Silicon Showdown, arXiv 2605.00519](https://arxiv.org/abs/2605.00519) (abstract, 2 auth.) | Apple UMA vs. NVIDIA for 70B+; energy up to 23× | low | new |

## 7. Low-bit quantization, pruning and merging of MoE experts

| Date | Paper | Key statement | Relevance | Status |
|---|---|---|---|---|
| 2025-10 | [REAP, arXiv 2510.13999](https://arxiv.org/abs/2510.13999) (abstract, 6 auth.) | Pruning beats merging for generation; router-weighted activation criterion; **near-lossless code generation at 50 % pruning on Qwen3-Coder-480B and Kimi-K2** | **high**: 20 % pruned experts (25.1 → 20 GB) would let all experts fit; needs a coding check on qwen4exp | new |
| 2026-07 | [Half the Experts, All the Code, arXiv 2607.16721](https://arxiv.org/abs/2607.16721) (abstract, 1 auth.) | Qwen3.6-35B-A3B and Gemma-4-26B: 50 % of experts removed with no detectable loss on the main code benchmark; the best strategy differs per model; **pruning beats quantization only where quantization would drop below 3 bpw**; PPL can rank a broken model above an intact one | **high**: we are at 3.3 bpw, near the crossover; pruning 20 % vs. requantizing to ~2.6 bpw must be measured on code | new |
| 2026-11 | [Prune, Don't Merge, arXiv 2611.12240](https://arxiv.org/abs/2611.12240) (abstract, 2 auth.) | Qwen3-30B at 25 % removed: REAP keeps 93.6 % of generative quality, REAM 62.5 %; losses mostly format breaks (invalid JSON, runaway generation) | **high** for agents: check tool-call JSON validity, not only benchmarks | new |
| 2026-10 | [ProQuant, arXiv 2610.30779](https://arxiv.org/abs/2610.30779) (abstract, 5 auth.) | Joint expert pruning plus bit allocation, where higher-precision experts substitute for badly degraded ones; RMSNorm finetune against routing drift | medium | new |
| 2026-10 | [Routing Is Not a Proxy for Quant Sensitivity, arXiv 2610.20826](https://arxiv.org/abs/2610.20826) (abstract, 1 auth., negative) | Qwen3.6-35B-A3B in ik_llama.cpp: layer bit reallocation from routing statistics does not beat uniform (~4.0 and ~2.7 bpw); at 2.7 bpw random allocation equals routing-guided | medium: saves a dead end (routing frequency → bits) | new |
| 2026-05 (ICML 2026) | [GEMQ, arXiv 2605.23078](https://arxiv.org/abs/2605.23078) (abstract, 7 auth.); [Q-Strata (EMNLP 2026), arXiv 2608.30564](https://arxiv.org/abs/2608.30564); [NatScale, arXiv 2610.20353](https://arxiv.org/abs/2610.20353); [AlphaQ, arXiv 2606.04980](https://arxiv.org/abs/2606.04980) (abstracts) | Expert-level mixed-precision allocators: global LP plus router finetune (GEMQ), bi-level per-block budgets (Q-Strata), contrast-calibrated weights with best KL in 9/9 settings (NatScale), calibration-free heavy-tail spectra (AlphaQ) | medium: only if we build our own expert quants; no numbers on our model | new |
| 2026-11 (ICCAD 2026) | [MatMoE, arXiv 2611.01704](https://arxiv.org/abs/2611.01704) (abstract, 6 auth.) | Matryoshka experts at several precisions in one representation, bit width per routing score, on llama.cpp | medium: alternative to PagedMoE bitplanes | new |
| 2026-08 | [Tied Trit-Planes, arXiv 2608.08910](https://arxiv.org/abs/2608.08910) (abstract, 1 auth.) | One 4.06-bpw format for disk, cache and kernel (no repacking), DeepSeek-V4-Flash streamed from SSD on a 64 GB laptop; 9 % smaller than Q4_K, 6.7 % faster decode | low-medium: principle "disk bytes = cache bytes = kernel input" holds for us already with GGUF IQ blocks | new |

## What transfers to the Mac backend

1. **One read per expert, 2 MB, uncached, straight into a Metal slot.** We measured 13 GB/s for 2 MB reads vs. ~7 GB/s for
   0.68 MB reads. That favors one contiguous 2.046 MB record per expert (gate+up+down, 16 KiB aligned), as in SSD-LLaMA
   (2.6-4.1× over the GGUF layout) and DeepInfer-Mac. Per-tensor reads (2609.18063 on demand: 0.27 MiB) are slower. Read with
   `F_NOCACHE` (LLM in a flash) into page-aligned no-copy `MTLBuffer` slots, so unified memory needs no staging copy and the
   page cache keeps no second copy.
2. **Queue depth is the lever, not bandwidth.** [estimate] At ~80 % residency and LRU-like hit rates of ≥ 95 %, a token has
   ~15-25 misses, i.e. ~30-50 MB, ~3-4 ms at 13 GB/s. But that is only 0.3-0.5 misses per layer, so on-demand reads run at
   QD1: 0.6 ms each, ~9-15 ms per token if serial. The windowed roofline (2609.04238) says lookahead pays exactly in this
   latency-bound regime, unlike PCIe-saturated WiSP or eMMC. Prefetch must therefore issue reads for several layers at once
   so the queue reaches depth 8-16.
3. **Predict before attention, overbook by confidence.** Our lead time is attention plus the DeltaNet block per layer. APEX
   reports > 99 % overlap with a pre-attention router and confidence-based extra experts; a probe on Qwen3-30B reaches 91 %
   top-8 recall (0.84-0.97 per layer); FATE-style next-gate reaches 79 % (97 % with overbooking). DeepInfer-Mac measured
   +21 % from next-layer prefetch on an M4. Overbooking is cheap here: a wasted 2 MB read costs ~0.16 ms of SSD time at
   QD16, but its slot needs a staging area so hot experts are not evicted (Mira HOT+STAGE, `TONA_PRE_KEEP`).
4. **Size the cache under memory pressure, not to the limit.** MawForge (M5 Pro, 24 GB) went from 13.9 to 2.3 tok/s when the
   cache grew from 35 to 50 % (hits 87 → 93 %), and SiliconBench reports the same. Our 26.8 GB GPU limit is not a safe cache
   size on a 32 GB Mac with a desktop. Measure decode vs. cache size with `vm_stat`/compressor counters, and keep the cache
   wired.
5. **The CPU cannot rescue a miss.** On the PC the CPU computes a miss to avoid the PCIe copy. On the Mac the expert is on
   the SSD either way, and the CPU reads RAM at only 58-115 GB/s vs. 334 GB/s for the GPU. A miss has three options: wait
   for the read, drop it, or use a low-bit stand-in. SSD-LLaMA also measured CPU misses as slower.
6. **Drop low-weight misses with top-k₂ renormalization.** On Qwen3.5-397B (top-10), 10 → 5 experts costs only 0.55 MMLU
   points if renormalized by the top-k₂ mass (2609.04575). Dropping only the 1-2 late, low-weight misses per token should cost
   far less than our `TONA_MISS_DROP` (+1 % NLL, w=0.1). Test it against a static-k baseline (2610.28369). PSR is the
   deterministic variant if the miss rate stays high.
7. **Dispatch structure: many small dispatches, no grid barrier.** 2610.31479 measured 3.04 µs per dispatch boundary on an
   M4 Pro (we measured ~2 µs on M1 Max). A fused MoE block was 1.8× slower. Plain device stores were 96 % stale across
   threadgroups without a cooperative launch. Dependent matvecs lose 33 % bandwidth (77 vs. 114 GB/s), so overlap
   independent work (expert matvecs of one layer) within one encoder instead of fusing.
8. **Fitting everything beats streaming, if the quality holds.** Budgeting Bytes reports 22× from fitting the fast tier
   (M4 16 GB, Q2_K). For us, 25.1 → ~20 GB means either ~20 % of experts pruned (REAP: near-lossless code at 50 % on
   Qwen3-Coder-480B; pruning wins over quantization below 3 bpw), or ~2.6 bpw on average (PagedMoE Dyn Avg2.5 ≈ INT4
   quality on Qwen3.5-397B). Agent checks need tool-call JSON validity (2611.12240), not PPL.

## Gaps

- **No paper measures an M1/M2 Max SSD for expert-sized reads** apart from LLM in a flash (6.1 GB/s dense, 2023). DeepInfer-Mac,
  2609.18063, MawForge and Routide report no bandwidth or queue depth. Our 13 GB/s at QD16 has no external reference.
- **No paper measures expert hit rates at ~80 % residency** on a 256-expert top-10 model. The Mac studies sit at 15-65 %
  (MawForge) or below the per-token working set (Routide), so our miss rate must come from our own traces (routing logs from
  the PC backend can be replayed at 9,800 slots).
- **Speculation plus SSD experts on Apple** has no clean result: 2610.31479 tests small dense models, MawForge one MTP-4 run
  (negative), PSR runs on an RTX 4090. Whether the expert union of 3-7 drafts fits the SSD budget is open.
- **Low-bit fallback copies** (HOBBIT, PagedMoE) were not evaluated with IQ-type quants or on Metal. A bitplane format would
  mean new kernels.
- **MTLIO / `MTLIOCommandQueue`** (Metal fast resource loading) appears in no paper; whether it beats `pread` + `F_NOCACHE` into
  no-copy buffers is unmeasured.
- Not read in full: all rows marked "abstract". The full text of PagedMoE, Kernel-tiering and PSR was read only up to 100K
  characters.
