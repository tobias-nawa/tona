# Expert co-activation as a predictor (#154)

**Agent research, state of 2026-11-15** (a Sonnet agent read the sources; [V-text] = full text read, [V-fetch] = page
summarized by a fetch tool, [snippet] = abstract or search result only). Numbers are the authors' claims. The replay
on our own traces is in [measurements/2026-11-15-expert-coactivation.md](measurements/2026-11-15-expert-coactivation.md);
the results section at the end of this file is written from it.

## Question

Vertical: given expert X in layer l, which (layer, expert) pairs follow in the next k layers, k up to about 20 (a
prediction 10 to 20 layers ahead would hide SSD reads completely, the next-layer lookahead leaves about one layer).
Horizontal: once one expert of a layer is predicted, which experts of the same layer come with it and which almost
never. Reference: the router lookahead (router of layer l+1 on the residual of layer l), 69 % recall and precision
on the Mac ([measurements/2026-11-15-mac-coder-arena.md](measurements/2026-11-15-mac-coder-arena.md)), earlier
notes in [14-expert-prefetch.md](14-expert-prefetch.md).

## Result of the literature search

**No source measures a statistical vertical chain at distance 5 or more, and none measures within-layer partner
completion.** The question is open in the literature; everything below is nearest evidence.

### Co-activation tables (the idea as asked)

| Source | Status | Method | Distance | Numbers | Stability across workloads |
|---|---|---|---|---|---|
| ST-MoE, arXiv 2606.15453 (2026-06) [V-fetch] | simulator, no code | cross-layer correlation table: per expert of layer L up to 8 candidates of layer L+1, 2-bit confidence; plus a table of the previous token's routing | 1 | about 85 % (abstract), "mostly above 80 %" (figure); models 60 to 160 experts, top-4 to 8; consecutive-token overlap about 2x the random value | not measured (CNN/DM, MATH, HumanEval) |
| Klotski, arXiv 2502.06888 (2025-02) [V-text] | paper, code not stated | table of frequency relations between experts of different layers from a warm-up run, path length 1, top-K of a batch group prefetched | 1 | prefetch accuracy of the hot experts 58.9 % (batch group), 42.2 % for one sequence; Mixtral-class (2 of 8) | table is not persisted so tasks do not contaminate each other; not measured |
| ExFlow, arXiv 2401.08383 [snippet] | code on GitHub | inter-layer expert affinity (conditional routing probability) for placement by ILP | all layers | up to 2.2x throughput; placement for expert parallelism, not prefetch | n/a |
| fMoE / FineMoE, arXiv 2502.05370 [V-fetch] | EuroSys 26, on the MoE-Infinity base | "expert maps" (gate distributions per iteration), trajectory search over stored maps, semantic search for early layers | 3 to 6 (3 Mixtral, 6 Qwen1.5-MoE, 4 Phi-3.5-MoE; the paper contradicts itself, 3 in section 6.1) | hit rate +14 % over Mixtral-Offloading, +37 % ProMoE, +68 % MoE-Infinity (figure); 1K maps default, 32K maps under 200 MB | 7:3 split of LMSYS-Chat-1M and ShareGPT, no variance |
| 2609.22471, Apple (2026-09-18) [V-text] | paper, no code | **training-time** router changes that raise co-activation among the tokens of a verify window (global balance loss, shared experts, neighbour-consistency loss, autoregressive selection) | within a window | 4 tokens need 9 unique experts normally, 5 with their router; +21 % throughput at 1B/353M active | not a predictor; cannot be applied to a trained model |
| Doubleword blog [V-fetch] | no code | batch-level co-activation, reorder prompts for expert overlap | n/a | oracle 21.3 %, trained BGE 15.6 %, plain clustering 12.4 % fewer expert loads; out of domain 12.3 % | yes, 15.6 to 12.3 % |

### Learned and router-based predictors (the baseline family)

| Source | Method | Distance | Numbers | Notes |
|---|---|---|---|---|
| Speculating Experts, 2603.19289 [V-text] | next layer's router on a quasi-hidden state, learned estimator (17M parameters) for early layers | 1 | Qwen3-30B-A3B (128 experts, top-8): about 90 % recall@k from layer 3; code `axonn-ai/yalis` branch `offload_prefetch` | executing mispredicted prefetches costs accuracy (0.817 to 0.641 router-PF, 0.766 hybrid) |
| PreScope / LayerScope, 2509.23638 [V-fetch] | learned layer-aware predictor on the PCA-compressed hidden state, 0.5 to 2.8 MB and 0.12 to 0.48 ms per layer | 1 (two-layer mentioned, no numbers) | hot top-4 hit 94 to 99 % (lenient metric), never below 80 % | ShareGPT with DPO and LLaVA as generalization tests, no drift numbers |
| Mira, 2609.38090 [V-text] | per-layer learned "MixtureHead" on pooled hidden states, router logits and expert histograms | 2 | "same accuracy as i+1"; coverage only as a figure (Mixtral); 5.71x throughput over MoE-Infinity | no distance sweep, no stability |
| ExpertFlow, 2410.17954 [V-fetch] | routing path for all layers from the input tokens in one pass, 7.2 MB feed-forward | full depth (12 to 32 layers) | up to 95 %, over 90 % in domain; 80 to 90 % Switch-64 | **cross-domain loss 5 to 10 points**; 8 to 64 experts only |
| SeqMoE, 2609.12978 [V-fetch] | Mamba2 over generation steps (token-ahead, not layer-ahead) | token | recall at top-(k+3) 85.8 to 90.9 % (Qwen3-30B-A3B 90.9 %); 8 recursive steps lose 3.6 points | no cross-workload test |
| DoMoE, IJCAI-26 0657 [V-text] | domain routing tables with semantic matching, one prediction amortized over several layers | several (value not in text) | hit ratio 67 to 70 % on Qwen3-30B-A3B; adjacent-layer embedding cosine 0.96, decays with distance | domain-split design, mixed workload still 66 to 68 % |
| SPICE, 2608.21240 [V-fetch] | recursive low-rank multi-layer rollout, halts when confidence drops | H_min to H_max | no recall in text; confidence "varies substantially across layers and workloads" | |
| SpecPrefetch, 2607.24787 [V-fetch] | low-rank adapter per layer pair | 1 | Qwen3-VL-30B-A3B recall at 4/8/10 predicted for top-8: 48.8 / 85.6 / 92.3 % | **asking for 10 for top-8 lifts recall from 49 to 92 %**, the cheapest "completion" |
| Fate, 2502.12224 [V-fetch] | training-free next-layer prefetch from a cross-layer gate | 1 | 78.8 % at top-8 of 128 (Qwen3-30B-A3B); 97.15 % at a confidence threshold | |
| MoE-CORE, 2610.01950 [V-fetch] | next router on the hidden state; "depth" = number of experts prefetched, not layers | 1 | hit rate 89.1 / 92.4 / 93.8 / 94.1 % at depth 0 to 3 (DeepSeek-V4-Flash) | code `LookAround0301/vllm-ascend` |
| Edge0, 2609.18063 [V-fetch] | per-layer pre-router head, one token ahead | token | adjacent tokens agree on only about a quarter of a layer's experts (Qwen3.6-35B-A3B, 256 experts) | |
| 2608.07911 [V-text] | evaluation of expert caching | n/a | probe sets from one template per category move the early-window effect by 19.4 to 31.9 points; a causal next-use predictor reaches R^2 0.237 | **measure workload-diverse traces** |
| 2604.23150 [V-fetch] | activation patterns for multi-node placement | n/a | popularity is strongly workload-dependent; a classifier on prefill patterns tells the request type with over 98 % | tables from one workload may not transfer |

Not opened (names only): MoE-Infinity, ProMoE, HOBBIT, AdapMoE, Pre-gated MoE (changes the model), DAOP 2501.10375,
Lina, eMoE 2503.06823, Lynx 2411.08982, Bevy 2611.04794 (abstract: prediction-based prefetch can hurt when a GPU
cache exists). The agent found no source for "Occult".

### Issue trackers

- **exllamav3 #254** [V-fetch]: the next layer's router on the layer's MoE input, top-16 predicted to cover the real
  top-8: recall 0.85 to 0.90 at one layer ahead, **0.76 to 0.83 at two layers ahead**; net loss because speculative
  copies compete with demand misses on the same PCIe link; misses per token at 170 slots/layer: static 100.9, LRU 31.5,
  LFU 29.3, Belady 13.6.
- **kekzl/imp #3052** [V-fetch]: top-10 recall on a Qwen-style model: 66.0 % (MoE input of L to L+1), 71.4 % (normed hc
  streams), 78.8 % (L+1's full MLP read), **56.8 % for L to L+2**. Server speed moved by 1 %. Close to our 69 %.
- ongunm/llama-moe-cache (llama.cpp fork, LRU, cache hit 99.5 % on Qwen3-30B-A3B after a GPU pool; not a recall);
  vLLM RFC #38256 (prefetch planned as a second step). Nothing on co-activation or lead time in ktransformers, SGLang,
  mlx-lm or Fiddler (one combined search only, coverage limited).

### What this means for the two questions

1. **Distance >= 5.** Router reuse decays by 7 to 10 points per additional layer at the first step (exllamav3,
   imp); the table methods were only tested at distance 1 (ST-MoE about 85 % on 60 to 160 experts, Klotski 59 % on 8).
   fMoE uses distance 3 to 6 but gives hit rates only in figures. A statistical chain at distance 5 to 20 on 256
   experts with top-10 is a new measurement.
2. **Horizontal partners.** No paper measures it. The nearest evidence is that over-fetching candidates is already
   a cheap completion (SpecPrefetch, 49 to 92 % for 8 vs 10 candidates; our Mac numbers: 69 % at 10, 82 % at 14,
   89 % at 18, at falling precision), and that adjacent tokens share only a quarter of a layer's experts (Edge0).
3. **Stability.** Cross-domain losses of 5 to 10 points (ExpertFlow), 15.6 to 12.3 % (Doubleword), and a warning that
   one-template probe sets distort by 19 to 32 points (2608.07911): the tables must be tested on other prompts and
   other workloads, which the replay does.
4. **Cost.** exllamav3 #254 and Bevy say a wrong prefetch costs the link a demand miss needs, so precision matters as
   much as recall; a prefetch only pays above a break-even that depends on the link load.

## Ideas from other fields

"Who goes with whom, and who not" is solved in other fields. Each idea is adapted here; the replay measures all of
them in the same way (name = predictor name in the replay).

| Origin | Idea | Maps to | Name |
|---|---|---|---|
| Market-basket analysis (Apriori, FP-Growth; Agrawal et al. 1993/94) | support, confidence = P(Y given X), lift = P(XY)/(P(X)P(Y)); rules above a confidence; lift < 1 as "rarely together" | horizontal (pruning) and vertical (rules from X in l to Y in l+d) | `chain`, `rule0.5/0.7/0.9`, `look+prune` |
| Information theory | PMI and NPMI, sum over the known experts is naive Bayes in log form | both | `pmi`, `npmi` |
| Item-item collaborative filtering (Sarwar et al. 2001) | cosine similarity of the co-occurrence vectors, neighbors of the routed experts | both | `cosine` |
| User-user CF / temporal streaming prefetchers (STMS, Domino) | find the most similar earlier tokens by the routed sets of the last layers and replay what followed | vertical | `knn` |
| Markov / correlation prefetchers (Joseph and Grunwald 1997) | table indexed by the top-1 expert (the "miss address"), top-N successors | vertical | `markov1` |
| PPM, variable-order Markov chains, n-gram backoff; TAGE (Seznec) | contexts of 1 to 3 layers of top-1 / top-2, interpolated by the context's support (Witten-Bell style), backoff to shorter | vertical | `ppm` |
| Perceptron branch predictor (Jimenez and Lin) | one linear unit per target expert over the routed-set history, batch ridge instead of online updates | vertical | `ridge_set`, `perceptron2`, `ridge_logit` (router logits as input) |
| Rank fusion (Cormack et al. 2009) | reciprocal rank fusion of several scores, nothing fitted | combination | `look+chain`, `mix`, `look*partnerM` |

Borrowed from the literature: the table of cross-layer co-occurrences (ST-MoE, Klotski), the next-layer router
lookahead (Pre-gated MoE, Fate, Speculating Experts, our own `TONA_ROUTE_TRACE_P`). Own mapping of the ideas above to
(layer, expert) paths, to the cold part of a layer, and to verify-pass unions: new here and measured against the
borrowed ones with the same replay.

## Result of the replay on the Coder (2026-11-15)

Details, tables and the decision against the plan's thresholds:
[measurements/2026-11-15-expert-coactivation.md](measurements/2026-11-15-expert-coactivation.md). 14,115 token rows of 33
requests (code, text, two 19k and 23k contexts), counting and test set always different requests.

- The router lookahead reproduces the Mac's number on the PC: 70.3 % recall and precision at K = 10 (72 on a shifted
  workload, stable). The published table methods are far weaker on this model: the plain cross-layer count chain
  (the ST-MoE / Klotski idea) has 42.6 % at d = 1.
- Own predictor: a linear map from the 256 router logits of layer l reaches 59.4 / 54.9 / 49.0 % at d = 1 / 5 / 20
  (the marginal control 20.2): the decay with distance is flat, because most of it is workload state. Its cold
  precision is 12 to 29 % (the lookahead 22 to 40 % at the same K), i.e. 3 to 7 wasted reads per useful one.
- Horizontal: "rarely together" is a real structure (27 % of the supported pairs have lift < 0.2, test lift 0.12), but
  completing or pruning the lookahead with it, or a fitted mixer, gains at most +0.1 point (all experts) or +1.6 to +2.2
  (cold-only): no build.
- Stability: a table counted on code loses 20 to 28 points on text (and the reverse); long contexts are not the
  problem. An online chain with decay (half-life 500 tokens) from an empty start reaches 44 to 48 % at d = 1 on every
  workload, +20 to +25 points over an online popularity control, and beats a shipped table.
- Recommendation: no build of a co-activation prefetch now; if one is built it learns online, ships no table, and
  needs a measured case where the lookahead's one layer of lead is not enough.
