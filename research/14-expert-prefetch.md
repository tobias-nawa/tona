# Prefetching and predicting experts (especially with speculative decoding)

**Agent research, state of 2026-10-24.** Sources are search results, arXiv pages and
GitHub pages; papers were only partly read in full text (see "Verification status" at the end).
Numbers are the authors' claims unless "independent" is stated. Own derivations and
estimates are marked **[own estimate]**. None of it was measured on our platform.

---

## Key findings

1. **Applying the router of layer l+1 to the state of layer l is usable without training,
   but not exact.** Measured on Qwen3-30B-A3B (128 experts, top-8): 78.8 % without training
   (FATE, arXiv 2502.12224); with a "quasi-hidden state" (normalized residual stream plus a learned
   default vector) about **90 % recall@k** from layer 3 on, clearly worse in the first two layers and the
   last (arXiv 2603.19289); in the vLLM RFC 83.6–99.1 % depending on the model,
   weakest in early layers (arjvnv, Apple Silicon, independent). Misses
   mostly affect **low-weighted** experts. A small learned estimator
   (4–5 M training tokens) lifts Qwen3-30B-A3B to ~90 % on average and early layers by up to
   ~25 points. These methods barely reach beyond 1–3 layers
   (SeqMoE, 2609.12978).
2. **Speculation makes the problem more expensive and easier at the same time.** More expensive: the union of
   experts over the verified tokens grows (Qwen3-30B-A3B with EAGLE-3, 7 nodes: cost per
   expert to load 0.21 ms with cache, 0.41 ms without; SpecFit 2610.15417; on another
   model ~3.7× the experts of one token at k=5, jmeadlock in vLLM RFC). Easier: the verified tokens
   are **known in advance**, the prediction for all of them together holds for the whole pass.
3. **The biggest documented lever with speculation is not the prefetch but the choice
   of what to verify.** SpecFit chooses subtree and depth by the cost of the expert union:
   +7 to +31.6 % over EAGLE-3 (A100, B=1, 48 slots/layer), and stays close to the optimum with a route predictor that is only
   60 % accurate (2.26× instead of 2.27× vs. AR). DraftExpert
   lowers the verify cost from 2.35× to 1.25× of a single step, of which prefetch ~0.2×.
4. **Cache policy: LFRU/SLRU with an admission filter beats plain LRU, but only by a few
   points; no known causal or learned policy closes the gap to Belady (~45 % fewer misses).**
   A learned next-use predictor was worse than LFRU in simulation (arXiv 2608.07911). The situation depends strongly on the model: Qwen3.8-Flash-Next (Coder)
   has concentrated routing according to llama.cpp reports (~80 % of hits in ~28 % of the
   experts), gpt-oss flat routing.
5. **CPU-GPU distribution: the goal is equal completion time of both paths**, not "rare
   → CPU, frequent → GPU" (DALI 2602.03495, arXiv 2610.23006; Fiddler 2402.07033 as starting point).
   What matters for us is that the load per expert is small at B=1..4 (CPU path lightly loaded,
   GPU waits), see detail 4.
6. **llama.cpp merged a GPU expert cache on 2026-10-07 (PR #29887, LRU, only
   batches ≤ 32, no prefetch).** Prefetch work there is open or postponed (#28414
   left out because prefill aborted in cuBLAS). None of the findings is an engine that
   combines drafter information for expert prefetch with CPU compute on consumer hardware.

---

## The three most implementable ideas for us (ordered by effort/benefit)

### A. Router lookahead on the CPU via the mailbox, with overbooking

The GPU already deposits the residual stream for the CPU at router l. The CPU additionally applies
router l+1 (optionally l+2) to this state (one 256×d matrix-vector multiplication per
token, negligible on Zen 5) and starts loading the predicted, non-resident experts **before** the GPU reaches layer l+1. With a learned default vector
and a small estimator for layers 1–2 and the last (Hybrid-PF from 2603.19289).
- Hit rate: 78.8 % (without training) to ~90 % (quasi-hidden state/estimator) on Qwen3-30B-A3B;
  overbooking (more candidates than top-k) lifts FATE from 78.8 to 97.2 % (Qwen3-30B-A3B).
- Reach: one layer ≈ 80 ms / 48 ≈ 1.7 ms lead time **[own estimate]**. At 55 GB/s
  that is up to ~90 MB ≈ 45 experts of 2 MB, so the cap of "2 refills per layer" can
  be raised if the copy runs asynchronously by DMA instead of synchronously from the CPU.
- Expectation **[own estimate]**: if 70–85 % of today's ~30 ms reload cost per pass are hidden,
  the pass drops from ~80 to ~55–60 ms (≈ 1.35–1.45×). Upper bound from the literature for context:
  prefetch alone gave 9–14 % TPOT (A6000, Qwen3-30B-A3B, no speculation, 2603.19289; compute time
  and copy time of equal size would be max. 2×), with speculation more (DraftExpert 1.45× overall).
- **Risk:** Qwen3.8-Flash-Next has a "gated residual" with four branches (arXiv 2608.30320);
  whether the residual stream l is similar enough to the router input l+1 is open. **Measure first**
  (recall@10 per layer, a trace from prefill suffices) before building code.
- Objection from practice: the vLLM RFC (arjvnv, Apple Silicon) found prefetch at B=1 net slower
  because the staging copy competed with weight streaming for bandwidth. Here
  the load sits on VRAM (315 GB/s) and PCIe separately; still **tune reloading and miss compute
  on the CPU to the RAM bandwidth (58 GB/s)**, both read from the same RAM.

### B. Cost-aware verify width instead of a fixed 3 draft tokens

The planner knows hit rate and reload cost per expert. Before the verify pass, estimate the expected
union of experts (with A) and choose draft depth/nodes so that
expected tokens per cycle time are maximal (SpecFit cost model: time = node term +
β · Σ_layers |non-resident union|; selection overhead 0.6–0.7 % of decode time).
- Evidence: SpecFit +7 to +31.6 % (Qwen3-30B-A3B, A100, B=1, 48 slots/layer, EAGLE-3);
  EVICT up to 2.35× vs. AR or ~1.21× vs. EAGLE-3 on average (2605.00342); EcoSpec 2607.12696;
  "Limits of Speculation" (2609.22156, Qwen3-Coder-30B-A3B + EAGLE-3, A100) shows: the rejection boundary is
  linear in the ratio Δcost/Δexpected tokens, i.e. a simple threshold suffices.
- Lossless. Lossy variant (owner decision): expert budget per layer in verification
  (MoE-Spec 2602.16052: +10–30 % throughput; AcceptMoE 2608.02989 per papers.md: 2.06×, −0.27 pp.).
- For our pass time: with a cycle time of ~80 ms a rejected draft token with 8–10 new
  miss experts per layer is expensive; the threshold should come from our measurements.

### C. Cache admission only for experts of accepted/likely tokens, LFRU with profile preload

Do not fill into VRAM experts that only a later rejected draft token needs
(bypass admission; in the Belady model bypass is the smaller part of the gap, 3.4–15.7 %, but
here it additionally costs PCIe time and CPU fill slots). Eviction by LFRU
(`frequency / (clock − last_access + 1)`, vLLM PR #37190) or SLRU+admission filter
(+8–15 points hit rate over LRU with mixed traffic according to a proof of concept in
llama.cpp #20757, independent). Fill the cache at startup from a profile (an existing engine does this
with a profile file); our value of ~81–84 % is already above what the reports
give for plain LRU (~60 %, one report in PR #29887).
- Expectation: small improvement, a few points of hit rate, **[own estimate]** 1–3 points;
  the literature sees LFRU 1.3 points (absolute miss rate) ahead of LRU (18.0 vs 19.3 %).

---

## Details

### 1. Predicting future experts (new work 2025/2026)

New relative to `papers.md` (DraftExpert, SP-MoE, AcceptMoE, SpecMoE, MoBiLE already there):

| Date | Paper | Core and numbers | Hardware | Note |
|---|---|---|---|---|
| 2026-03 | [Speculating Experts, 2603.19289](https://arxiv.org/abs/2603.19289) | Router l+1 on a "quasi-hidden state" (norm. residual + default vector); ~90 % recall@k on Qwen3-30B-A3B from layer 3, GPT-OSS 83/88 %; −9–14 % TPOT (A6000), −5–8 % (A100/GH200). Executing the speculated experts instead of reloading: accuracy mostly preserved, on Qwen3 math it drops, Hybrid-PF (estimator only in early layers) recovers ~37 % of the gap | A6000 PCIe 4, A100, GH200 | Full text read; code in YALIS |
| 2025-10 (v3 2026-10-22) | [MoE-SpeQ/Specter, 2511.14102](https://arxiv.org/abs/2511.14102) | A 4-bit draft model provides multi-stage routing lookahead, PCIe copies during draft and verify; 1.50× TPOT on average, max. 1.88× (ASPLOS 2027); ">90 %" top-k accuracy per secondary source | Single GPU, models not named on the page | abstract only |
| 2026-08 | [SPICE, 2608.21240](https://arxiv.org/abs/2608.21240) | Draft model with adaptive lookahead depth by confidence; low-confidence misses approximated by low-rank substitute experts, exact remainder asynchronously on the CPU; up to 3.12× TPOT (ASP-DAC 2027) | not named; DeepSeek-V2-Lite, Qwen2-57B-A14B | abstract only; lossy |
| 2026-09 | [SeqMoE, 2609.12978](https://arxiv.org/abs/2609.12978) | Sequence-to-sequence prediction of activations over several layers/steps, prefetch as "job sequencing with deadlines", Belady-like eviction from the forecast, graph-compatible (prefetch without synchronization); hit rate 96.97 % at 45 % residency; 80.22 % of full-load performance | not named | Abstract; reviewer note on circularity (oracle comparison) |
| 2026-09 | [Mira, 2609.38090](https://arxiv.org/abs/2609.38090) | Per-layer predictor predicts two layers ahead (i→i+2), HOT+STAGE cache (STAGE as probation area), INT8 experts; prefetch share only 1.07× at 48 GB, 1.08–1.49× at 16–48 GB; 1.55× vs. Fiddler | RTX 6000 Ada, A5000 | Mixtral/DeepSeek-V2-Lite |
| 2025-11 | [Pre-Attention Expert Prediction, 2511.10676](https://arxiv.org/abs/2511.10676) | Per-layer predictor on the input before attention, runs parallel to attention; Qwen3-30B-A3B 94.69 % exact (layer 1), over-provisioned 98.81 %; 0.15 ms, also covers layer 1 | V100/A100 (timing), TITAN RTX (training) | no end-to-end; training data MMLU |
| 2025-02 | [FATE, 2502.12224](https://arxiv.org/abs/2502.12224) | Next gate on the current gate input: 78.79 % without training, with confidence extension 97.15 % | – | Numbers from a secondary source |
| 2026-05 | [Fast MoE Inference via Predictive Prefetching and Expert Replication, 2605.11537](https://pith.science/paper/2605.11537) | title only seen | – | not checked |
| 2026 | [DoMoE, IJCAI 2026](https://www.ijcai.org/proceedings/2026/657) | Domain routing tables for prediction; 1.32× throughput, 1.22× hit rate | – | snippet only; close to "frequency across sessions" |
| 2025-05 | [Local Routing Consistency, 2505.16056](https://arxiv.org/abs/2505.16056) | 20 MoE LLMs: domain-specialized experts contribute more to local routing consistency than vocabulary-specialized ones; cache ≈ 2× active experts is mostly best for segment caching | – | Snippet |

Older foundations (the agent's prior knowledge, **not re-checked**, IDs from memory except where linked):
Pre-gated MoE (Hwang et al., ISCA 2024, 2308.12066), AdapMoE (2408.10284), MoE-Infinity (2401.14361),
[HOBBIT 2411.01433](https://arxiv.org/abs/2411.01433) (mixed precision, LRU/LFU mix; hit rate ~55 % on Mixtral per third-party citation),
[ProMoE 2410.22134](https://arxiv.org/abs/2410.22134) (proactive caching), [ExpertFlow 2410.17954](https://arxiv.org/abs/2410.17954) (up to 95 % prediction accuracy),
[Fiddler 2402.07033](https://arxiv.org/abs/2402.07033), DAOP (2501.10375), MoE-Lightning (2411.11217), Klotski (2502.06888),
eMoE (2503.06823), FloE (2505.05950), PowerInfer-2 (2406.06282), DuoServe-MoE (as cited in 2603.19289: 54–67 % top-2 on Mixtral).
They mostly target batch throughput or older models (Mixtral 8×7B with only 8 large experts);
transferability to 256 small experts is poorly documented.

**With speculation/drafter information specifically:**

| Paper | What the draft provides | Numbers |
|---|---|---|
| [SP-MoE, 2510.10302](https://arxiv.org/abs/2510.10302) (full text read) | Applies the **gates of the target** to states of the **structurally identical small draft model** (cosine similarity of the states up to 94.6 % on DeepSeek-Lite, 56–60 % on Mixtral/Phi); a "cutoff layer" bounds the prefetch depth so that prefetch finishes before the end of the draft; critical experts k=1–2 | TPOT −29–43 % vs. Mixtral offloading, mean 1.35×, peak 1.75× (RTX 3090); over models avg. 2.3×, peak 3.5×; RTX 3090/4090 PCIe 4, Mixtral 8×7B, Phi-3.5-MoE, DeepSeek-V2-Lite |
| [DraftExpert, 2607.24434](https://arxiv.org/abs/2607.24434) (HTML read) | One trained extra expert per layer in the draft (self-distillation); a router-agreement loss lifts prefetch hits from 73 to 88 % (DS) | 1.29–1.61×, avg. 1.45×; RTX 4090 (BF16, experts in CPU RAM) 1.36×/1.29× vs Snapdragon 8 Elite (flash) 1.52×/1.61×; verify cost 2.35×→1.48× (confidence+expansion)→1.25× (+prefetch) |
| SpecMoEOff, [2508.21706](https://arxiv.org/abs/2508.21706) | Speculation increases the load per expert; no predictor of its own | up to 2.5× vs. MoE-Lightning (Mixtral 8×7B FP16, A30/4090D, batch throughput) |
| [SpecFit, 2610.15417](https://arxiv.org/abs/2610.15417) | Cost model by expert union, subtree selection | see idea B |
| [EVICT, 2605.00342](https://arxiv.org/abs/2605.00342), [EcoSpec, 2607.12696](https://arxiv.org/abs/2607.12696), [MoE-Spec, 2602.16052](https://arxiv.org/abs/2602.16052), [AcceptMoE, 2608.02989](https://arxiv.org/abs/2608.02989) | Prune the tree or set an expert budget in verification | see idea B |
| [Limits of Speculation, 2609.22156](https://arxiv.org/abs/2609.22156) | Upper bound and local rule; in a case study 1.4–3.7 new experts per layer at the first divergence | Qwen3-Coder-30B-A3B + EAGLE-3, A100 |

Relation to us: SP-MoE and DraftExpert need a draft of **identical architecture** (target gates
applicable) or trained draft experts. Our DFlash2 draft (603M, block-wise
diffusion) has no expert layers; its hidden states are not in the space of the target routers.
A transferable route would be a small learned mapping from draft hidden states to
target router logits (see 5b). Without training, what remains is the verify pass's own prediction (idea A),
because there all 4 tokens and the real target state are available.

### 2. Accuracy of "router of the next layer on the current state"

| Source | Model | Measure | Value |
|---|---|---|---|
| FATE 2502.12224 (citation in 2511.10676) | DeepSeek-V2-Lite/Qwen3-30B | Prefetch accuracy | 78.79 % (without training); 97.15 % with extended candidate count |
| 2603.19289, fig. 4/10 | Qwen3-30B-A3B | Recall@k | ~90 % from layer 3, weak in layers 1–2 and the last; estimator ~90 % on average |
| 2603.19289 | GPT-OSS-20B/120B | Hits | estimator 88/83 % |
| vLLM RFC #38256 (arjvnv, independent) | Qwen3.6-35B-A3B and others | Router reuse | 83.6–99.1 %, weaker early |
| 2511.10676 | Qwen3-30B-A3B | exact, layer 1, predictor before attention | 91.8–94.7 % |
| SP-MoE 2510.10302 | Mixtral/Phi-3.5 | Draft→target gate | ~88 % |

For DeepSeek-V3 size, Qwen3-Next or Qwen3.8-Flash-Next the search found **no** measurement;
the numbers apply to smaller models. Our own measurement on Qwen3.8 is therefore mandatory.
Notes: The loss in state space arises mainly in the first layers (large
residual change) and in MoE models with strong drift (GPT-OSS); Qwen3-30B-A3B has
hardly any drift beyond layer 2 according to the authors. With **several layers of lookahead** the
accuracy drops quickly (SeqMoE: 1–3 layers is the limit of residual-based methods).

Distinguish: "recall of the top-k" (we need all 10) versus "weight mass" (missed
experts are mostly lightly weighted, executing the wrong experts barely changes the output
but is lossy).

### 3. Cache policy

- [Reproducible Evaluation of MoE Expert Caching, 2608.07911](https://arxiv.org/abs/2608.07911): simulation, Qwen3-30B-A3B (4 bit), ρ=40 %, B=8: miss rate
  LFRU 18.01 %, LFU 19.14 %, LRU 19.30 %, static 19.10 %, **Belady 9.93 %** (gap ≈ 45 % relative,
  stable at 44–46 % over 13 workloads). 84–97 % of the gap comes from knowing the expert "needed furthest
  in the future", not from bypass. A learned next-use predictor was
  worse than LFRU (picked the optimal victim in 3.4 % of cases, LRU/LFRU ~21–22 %).
  **The result depends on the replay:** sequential replay order inflates LRU/LFRU by 27–29 %
  and reverses rankings; prefetch was excluded. Simulation only, no hardware run.
- SeqMoE 2609.12978: probabilistic Belady policy from forecast; own ablation finding LRU ≈ LFU,
  future-aware better (reviewer note on circularity).
- llama.cpp [#20757](https://github.com/ggml-org/llama.cpp/issues/20757) (issue, closed 2026-04-08): SLRU+admission filter
  +8–15 points over LRU with mixed traffic (proof of concept "tinyserve", GPT-OSS-120B on 8 GB: 98–100 % hits);
  cumulative LFU best for one tester (Qwen3-30B-A3B, Metal); Qwen3.8-Flash-Next-Coder routing:
  ~80 % of hits in ~28 % of the experts; martinalderson (RX 9070 XT, Vulkan, Gemma 4 26B-A4B): Gini ≈ 0.76,
  44 % overlap of consecutive tokens, LRU 47→67 % hits, 27.9→35.1 tok/s.
- vLLM [#37190](https://github.com/vllm-project/vllm/pull/37190) (PR 1 of RFC #38256, state of the page: open, CI green): LFRU;
  a third-party test found LFU without aging 1.4–2.3× worse than LRU; a Belady oracle (OLMoE, B=1) only
  1.068× → many misses are "cold" (not avoidable).
- llama.cpp [#29887](https://github.com/ggml-org/llama.cpp/pull/29887): plain LRU; a tester with a fork reported that LRU is poor, frequency-based
  admission/eviction reaches ~60 % hits (single report).
- According to llama.cpp reports Qwen3.8 has a favorable ratio; **at 81–84 % we are already above
  what was reported**. Also reliable: do not raise the residency budget beyond the working set
  (koren1712, JigSawPT: loss with too large a share).

### 4. CPU-GPU hybrid

- **Fiddler** (ICLR 2025): latency model; CPU time grows almost linearly with the number of input tokens,
  GPU time flat plus transfer; 1.26× (B=1), 1.30× (long prefill).
- **KTransformers** (SOSP 2025, [paper](https://dl.acm.org/doi/10.1145/3731569.3764843)): AMX kernels, asynchronous scheduler, 1.25–4.09× decode;
  **Expert deferral**: the outputs of some experts are added to the residual stream only in a later layer,
  so that CPU and GPU overlap; CPU utilization from <75 % to ~100 %, up to 1.45× additional, accuracy loss ≤ 0.5 %.
  SGLang adoption: [sgl-project/sglang#11425](https://github.com/sgl-project/sglang/issues/11425).
- **DALI** ([2602.03495](https://arxiv.org/abs/2602.03495), 2026-02): expert assignment as a 0-1 integer problem, solved greedily, **equal completion time** of CPU and GPU;
  PCIe transfer up to 78.1 % of total time; residual-based prefetch for experts with high load;
  cache replacement by temporal workload; decode 3.97×/2.16×/1.48×/1.32× vs. llama.cpp/KTransformers/MoE-Lightning/HybriMoE
  (RTX 3090/4090, PCIe 4.0 x16). Observation: with a small batch the loads are light, the static assignment pushes almost everything to the CPU, the GPU waits.
- **[2610.23006](https://arxiv.org/abs/2610.23006)** (2026-10-16): experts with high load on the GPU, with low load on the CPU, then balanced so that both paths finish simultaneously; prefill up to +51.9 %, decode up to +39.5 %.
- **CoX-MoE** ([2605.17889](https://arxiv.org/abs/2605.17889), DAC 2026): AMX CPU; finds that AVX matmul throughput per core is insufficient, for throughput operation (not B=1).
- llama.cpp break-even (RFC discussion [#28248](https://github.com/ggml-org/llama.cpp/discussions/28248), single claim): hit rate ≥ 1 − PCIe/RAM bandwidth
  (with PCIe 3 + DDR4 ≈ 91 %). On our platform with measured 55.7 vs 58 GB/s the value would be purely arithmetically ≈ 4 % **[own calculation]**; more important is
  that CPU compute and upload claim **the same RAM bandwidth**.
- llama.cpp PRs: [#29887](https://github.com/ggml-org/llama.cpp/pull/29887) GPU cache, **merged 2026-10-07** (am17an), Qwen3.8-Flash-Next Q4_0, RTX 5090: 30.8→54.5 tok/s (77 % hits) and 67.8 tok/s with `-cmoe` (89 %), RTX 4090 1.57–1.62×
  (author, EPYC 7742, PCIe 4); independent: R9700 (HIP) 1.3× at 33 %, 2.5× at 67 % hits, RTX 3090 only 1–5 % above a tuned `--n-cpu-moe`; GLM-5.3-Flash 33→16 tok/s at 43.5 % hits (single report);
  only batches ≤ 32 (prefill on the regular path, there up to −27 % on 2×4090). [#30112](https://github.com/ggml-org/llama.cpp/pull/30112) multi-GPU, one day later. #27861 (JigSawPT, asynchronous worker, +26–31 %; status not given on the page),
  #26563/#26824 (heatmap/LFU with aging, open per secondary source), #25294 (SSD streaming, status unknown), #28414 (expert prefetch, left out in a fork: prefill aborted in cuBLAS), #21067 (prefetch of the next layer via the scheduler copy engine, blog 2026-04-28, status not checked).
  Review request in #29887: compute misses on the CPU while uploading in the background — **we already do this**.
- vLLM: [#37190](https://github.com/vllm-project/vllm/pull/37190) open (PR 1/3); PR 2 (asynchronous H2D, temporal prediction) planned; #34535 (static CPU offload) merged; consensus in the thread: prediction helps more in batch operation than at B=1.
- **ik_llama.cpp:** the search results contained neither an expert cache nor prefetch (only `-ot`/`--cpu-moe`); not checked in the repos themselves.

### 5. New for this constellation

a) **Expert deferral for CPU misses** (from KTransformers): the result of the miss experts computed by the CPU
is added to the residual stream not at the end of the same layer but one layer later; the GPU then never waits for the CPU. Lossy (≤ 0.5 % accuracy at KTransformers, a different model);
because of our low miss count per layer (16–19 % × 10) only part of it could be deferred (e.g. the lowest-weighted). Owner decision, measure quality first.

b) **Learned mapping draft hidden state → target router logits**: DraftExpert shows that a router-agreement loss lifts the prefetch hit rate from 73 to 88 % (same draft, only a different loss).
For DFlash2 there is no indication that anyone has done this. **[own idea, unchecked]** Only sensible if we retrain the draft anyway.

c) **Verification as prefetch lead for the next cycle**: The experts of the accepted tokens of one cycle are the best predictors for the experts of the next
(44 % overlap of consecutive tokens on Gemma 4 according to martinalderson; Gini 0.76). Pin over one cycle, no eviction of those just used (the vLLM bug "freshly loaded expert is evicted immediately" has been fixed, transferable to us: exclude experts of the running pass from victim selection).

d) **Quality on a wrong prediction**: executing the predicted instead of the actually chosen experts (2603.19289) saves reloading but is lossy; for our coder use (math/code sensitive) only after measurement.

e) For context: Specter (ASPLOS 2027) 1.50× on average, SpecFit +7–31 %, DraftExpert 1.45×. These are orders of magnitude that seem reachable for us with combined implementation of A+B **[own estimate]**, without our having measured it.

---

## Open measurements (before code)

1. Recall@10 and weight mass of "router l+1 on residual l" per layer on Qwen3.8-Flash-Next-Coder, with and without a learned default vector; influence of the gated residual.
2. Distribution of the expert union per verify pass and the share of misses that fall on **rejected** draft tokens.
3. Actual cost per reload (µs per expert, synchronous from the CPU vs. DMA copy), RAM bandwidth conflict between CPU compute and copying.
4. LRU vs LFRU vs LFU with aging on recorded traces **from real decode** (not a replay of one session), Belady as upper bound.

## Verification status

- Read in full text (PDF text): 2603.19289, 2510.10302 (partly), 2602.03495 (partly), 2609.22156 (beginning).
- Via arXiv abstract/HTML summary: 2607.24434, 2608.21240, 2511.14102, 2609.12978, 2608.07911, 2610.15417, 2508.21706, 2609.38090, 2511.10676, 2610.23006, 2605.17889, 2608.30320.
- Search result snippet only: 2605.00342, 2607.12696, 2602.16052, 2605.11537, DoMoE, 2505.16056, KTransformers details, everything on ik_llama.cpp.
- GitHub pages via WebFetch summary (not raw text): llama.cpp #20757, #29887, discussion #28248; vLLM #38256. Benchmarks there are claims by the PR authors and individual testers, unchecked.
- The supplementary list "Older foundations" relies on agent knowledge; IDs without a link are from memory.
