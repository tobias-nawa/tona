# 30 — Reusing the sparse-attention selection across layers, decode steps and draft tokens

Agent research (Sonnet), 2026-10-29

Method: web search and page retrieval (arXiv abstracts/HTML, GitHub README, vLLM docs). Nothing was measured or built
here. No repository code of ours was changed. Marking per entry:
- **[HTML]**: paper HTML read (numbers are the authors').
- **[abstract]**: abstract page only.
- **[docs]**: vendor docs / README read.
- **[snippet]**: only a search-result snippet seen.
- **[estimate]**: own reasoning, not measured.

Rates are named: **prefill** or **decode**. All speedups below are the authors' on their hardware (H100/A100/B300), not on
our RDNA4 platform.

Our setup (see research/29): 12 QSA layers between 36 Gated-DeltaNet layers; indexer 4 heads x 128 over pooled blocks
(4 positions), top-512 blocks (= 2,048 cells) + up to 3 tail cells; KV int8, in pinned host RAM at large context; decode runs
in the persistent megakernel. First own measurement: 6 of 12 indexers -> decode pass time at 250K -18 %, dNLL +0.001 to
+0.003 nats.

---

## 0. Summary

1. **IndexCache has public code (patches for SGLang and vLLM) and is documented in vLLM**, but the released tooling is
   only the patch and a pattern string; the greedy search itself is described, not shipped [docs]. Our 6/12 result
   (1/2 retention) is in line with the paper: at 1/2 the searched pattern loses nothing measurable (30B: Long Avg 50.3 vs
   50.2; GLM-5: 78.7 vs 78.4) [HTML].
2. **The pattern matters far more than the ratio.** Uniform interleaving at 1/4 loses 7 points on long-context averages
   (50.2 -> 43.0; GLM-5 78.4 -> 72.7), the searched pattern loses 0.3-0.4. Cosine/overlap proxies did not work as a
   selection criterion; only end-to-end LM loss did [HTML]. This confirms our greedy-by-loss approach.
3. **Cross-layer reuse is a family**, not one paper: Kascade (anchor layers by DP over similarity, head remapping),
   OmniKV (filter layers), TidalDecode (2 selection layers, re-selection), LServe (reuse over decode steps), the
   speculative-verification paper 2512.21911 (anchor layers and shared selection across draft tokens). All are
   training-free except the training-aware IndexCache variant and the YOCO-style CLSA.
4. **Reuse over decode steps (temporal) is much less proven than reuse over layers.** SPIN (2610.09025) measured that
   the same layer's previous-step scores correlate better than the previous layer's (Pearson 0.849 vs 0.519, top-K recall
   0.577 vs 0.306), but recall 0.58 is poor for exact reuse; it uses the history only as a *pre-filter* for the indexer,
   with exploration. LServe reuses page selection over consecutive query tokens (4x less selection cost [snippet]).
   Nobody has published "re-run the DSA indexer only every n tokens" with quality numbers on a DSA model.
5. **For MTP verification the draft tokens naturally share one selection**: 2512.21911 reports >0.8 block overlap among
   draft tokens in a step. This fits our verify pass (one top-k per pass instead of per token) but costs recall for the
   later draft positions; it must be measured.
6. **Risk is long-context retrieval, not reasoning.** IndexCache's degradation at aggressive ratios appears in
   long-context averages (MRCR, GraphWalks, LongBench v2, RULER, AA-LCR); AIME/GPQA/LiveCodeBench stayed flat or rose
   [HTML]. Our code-agent use is retrieval-heavy at 100K-250K, so the benchmark to watch is a retrieval one.

---

## 1. Cross-layer reuse of the selection

### 1.1 IndexCache (arXiv 2603.12201, Tsinghua / Z.ai, March 2026)

- **Idea:** Layers split into F (keep their indexer) and S (reuse the top-k of the nearest preceding F layer). The
  DSA indexer is O(L^2) per layer, and top-k overlap between adjacent layers is 70-100 % [HTML][docs].
- **Training-free:** yes (greedy search); **training-aware:** also (multi-layer distillation: each kept indexer is
  trained against the averaged attention distributions of all layers it serves).
- **Greedy search [HTML]:** start with all F; per step flip each remaining F layer (not the first) to S, evaluate LM loss on
  a calibration set (SFT data, batch 768, 200K context), commit the lowest-loss flip. Cost N(N-1)/2 forward passes.
  Not globally optimal. Proxies (cosine, top-k overlap) failed.
- **Pattern (30B, 47 layers, 1/4 searched):** `FSFSFSSSSFSSSFSSFFSSFSSFSSSSFSSSFSSSSFSSSSSSSSS` (as returned by the page fetch; verify against Appendix B before use) as printed in the
  paper [HTML]. It is irregular: dense F at the start (layers 0-5), then sparser, a few adjacent FF. Early and late layers
  have overlap <= 0.4; clusters of similar layers at 3-5, 6-8, 17-30, 31-36.
- **Quality, training-free, 30B DSA [HTML]** (Long Avg / General+Reasoning Avg): DSA 50.2/74.6; 1/2 searched 50.3/74.4;
  1/4 searched 49.9/74.9; 1/8 searched 46.1/73.7; uniform 1/2 47.4, 1/4 43.0, 1/8 35.3. AIME-25 at 1/4 searched 92.6 vs 91.0.
  Long benchmarks: MRCR v2, GraphWalks, LongBench v2, RULER, AA-LCR (no needle-in-haystack).
- **Quality, GLM-5 (744B) training-free [HTML]:** 78.4 -> 1/2 uniform 78.1, 1/2 searched 78.7, 1/4 uniform 72.7, 1/4
  searched 78.0.
- **Training-aware (30B) [HTML]:** 1/4 uniform reaches 50.6 vs 51.0 for the shortened-pipeline DSA; dropping the
  cross-layer loss drops AA-LCR 49.8 -> 44.0. Training-aware on GLM-5 is "planned".
- **Speed (30B, H100, SGLang) [HTML]:** prefill 200K 19.5 s -> 13.7 s (1/2) -> 10.7 s (1/4) = 1.82x; decode
  single-stream 200K 58 -> 73 (1/2) -> 86 tok/s (1/4) = 1.48x (decode rate, per request); full-KV throughput 297 vs 197
  tok/s at 200K. The indexer is 81 % of prefill at 200K. GLM-5: >= 1.3x prefill and decode beyond 100K at 1/2 [HTML];
  the README says ~1.2x end to end [docs].
- **Code:** github.com/THUDM/IndexCache: `indexcache.patch` (SGLang, tested on commit b638b25b) and
  `indexcache_vllm.patch` (vLLM, commit 4508532fb), DeepSeek-V3.2 and GLM-5, Apache 2.0, config
  `index_topk_freq` / `index_topk_pattern` (F/S string) [docs]. vLLM documents the feature (`use_index_cache`,
  `--hf-overrides`) in its docs for several releases, i.e. it is in vLLM main [docs]; I did not find the PR numbers. For
  SGLang I could not confirm a merge (only the patch). llama.cpp: no IndexCache PR found; the lightning indexer op itself
  is a separate PR (#24231, open per a third-party doc, unconfirmed) [snippet].
- **MTP/spec decode:** neither README nor vLLM docs mention it [docs].
- **Transfer to us:** direct. Our dNLL +0.001..0.003 at 6/12 is the same regime as "1/2 searched is lossless". The
  paper's decode gain (1.48x at 1/4) comes from a model where the indexer is a large part of decode; ours is pooled
  (4x fewer keys), so the share is smaller and -18 % pass time at 1/2 is plausible. Open: 1/3 and 1/4 (3 or 4 of 12 indexers).
  Differences from the paper: our layers are 12 widely separated QSA layers (DeltaNet in between), so the "adjacent layer"
  similarity of a 47-layer pure-DSA stack does not transfer automatically; the hidden state passes through 3 DeltaNet
  layers between two QSA layers. Our own measurement of overlap between consecutive QSA layers is the first thing to log.

### 1.2 Kascade (arXiv 2512.16391, Dec 2025) [abstract]

- **Idea:** exact top-k on a few anchor layers, intermediate layers reuse the indices. Anchors chosen by dynamic
  programming that maximizes cross-layer similarity on a dev set. **Head remapping** (each reuse-layer head takes the
  indices of the best-matching anchor head) is called critical for accuracy. Reported: top-k of layer 16 covers 99 % of the
  top-k attention mass of layers 17 and 18 [snippet].
- **Training-free:** yes. LongBench, AIME-24. Up to 4.1x decode attention, 2.2x prefill attention vs FlashAttention-3 (H100).
  Accuracy "close to dense", no number on the abstract page. Code: not confirmed.
- **Transfer:** DP over a similarity matrix is cheaper than IndexCache's greedy LM-loss search (no forward passes), but
  IndexCache reports similarity proxies failed as the criterion; use Kascade's DP only to propose candidates. Head
  remapping is irrelevant for the shared selection in our indexer (4 indexer heads are summed to one score per block;
  the top-k is per query, not per head), but the 2 KV heads of our GQA could be handled separately if we ever select per head.

### 1.3 TidalDecode (arXiv 2410.05076, ICLR 2025) [snippet+abstract]

- **Idea:** position-persistent sparse attention: a few token-selection layers (one early, one in the middle) compute
  full attention to find top tokens; the rest attend only to that set. A re-selection layer in the middle corrects drift.
  Up to 2.1x decode latency, matches full-attention quality on the paper's tasks. Training-free. Code:
  github.com/DerrickYLJ/TidalDecode (Llama-3, GQA, per-head top-k).
- **Transfer:** the lesson is that a *second selection layer in the middle* is needed because selection drifts with depth,
  consistent with IndexCache's cluster structure. Not a DSA method (selection by exact attention, not an indexer).

### 1.4 OmniKV (ICLR 2025) [snippet]

- **Idea:** high similarity of important tokens across consecutive layers; a few "filter" layers pick the tokens, the
  rest use them; KV offloaded to CPU and only selected tokens are fetched. Training-free. 1.7x at 128K on an A100;
  Llama-3-8B max context 128K -> 450K on one A100. Code: github.com/antgroup/OmniKV.
- **Transfer:** same offload structure as ours (KV in host RAM, fetch only selected cells). It is the closest published
  design to "S layers need no selection, only the gather"; with shared selection the gather for shared layers can be
  prefetched as soon as the F layer finishes (see 4.3).

### 1.5 InfiniGen (arXiv 2406.19707) [abstract]

- Predicts the next layer's needed tokens with a cheap preview (current layer input, partial next-layer query weights
  and key cache) and prefetches only those KV rows from host memory. Up to 3.0x over prior offloading methods. Training-free
  (offline weight skewing).
- **Transfer:** this is the *prefetch* version of cross-layer prediction. For us: the F layer's selection (or the next
  layer's cheap indexer) can start the PCIe gather one layer early (we have 55.7 GB/s per direction). Relevant to the
  S layers: their selection is known as soon as the F layer finishes, so the gather for several following QSA layers could be
  issued together.

### 1.6 Others on the same axis

| Paper | Idea | Training-free | Numbers | Note for us |
|---|---|---|---|---|
| HySparse2 (arXiv 2609.26368) [snippet] | full-attention layers do top-k; following sparse layers reuse the selected tokens' *KV entries* (two-level KV sharing) | architecture, not retrofit | not read | needs training; shows direction of next model generations |
| CLSA / "You Only Index Once" (arXiv 2606.06467) [abstract] | YOCO-style: one indexer computes token-level top-k once, shared across cross-decoder layers that share KV | no (architecture) | 7.6x decode, 17.1x throughput at 128K | not retrofittable |
| Autonomy-of-Heads (arXiv 2608.06849) [abstract] | anchor layer records per-head top-k, reuse layers borrow; heads classified data-free from W_K^T W_Q rank | yes | 96.5 % of full at 50 % sparsity, prefill -41 %, decode -66 % | head-level reuse; needs per-head selection, not our indexer |
| 2512.21911 (spec-verification) [HTML] | anchor layers = lowest Jaccard with predecessor (30 of 61 on DeepSeek-R1) | yes | no wall-clock; FLOPs only | see section 3 |

---

## 2. Reuse over decode steps and tokens (temporal)

| Paper | What is reused | Numbers | Caveat |
|---|---|---|---|
| SPIN (arXiv 2610.09025, NVIDIA) [alphaxiv abstract] | EMA of the same layer's past indexer scores (vertical by absolute slot, diagonal by relative offset); max-pooled in 64-slot blocks; only the top-predicted blocks go to the real indexer; sink block and newest block always kept; part of the budget random exploration | DeepSeek-V4 Flash, vLLM, 8x B300, 507K cache: +14.9 % output throughput, ITL -13.2 % at 50 % target block sparsity; tau2-airline 73.5 % vs 72.5 % | one prompt per diagnostic; history can miss sudden shifts; single serving setup. Previous-step recall 0.577 is not enough for blind reuse |
| LFPS (arXiv 2506.15704) [abstract] | earlier steps' retrieval results predict the current step's top-k (vertical + slash pattern + positional expansion) | 22.8x vs full attention (RTX 4090), 9.6x vs exact top-k on one Xeon core; LongBench/RULER on Llama-3.1-8B "preserving accuracy" | not DSA; not clear if training-free |
| LServe (arXiv 2502.14866) [abstract+snippet] | hierarchical page selector; reuses page selection across consecutive query tokens (4x less selection overhead [snippet]) | 2.9x prefill, 1.3-2.1x decode vs vLLM | the reuse interval and its quality effect I could not read; check the paper before use |
| Dynamic Sparse Attention: Access Patterns (arXiv 2603.13430) [abstract] | analysis of per-layer KV indices over decode steps; proposes LLC reservation + token-level LRU | no numbers in abstract | notes that per-query top-k can differ completely between queries (argument against reuse); the data is useful for our host-RAM gather cache |

Findings:
- No paper reports "re-run the DSA indexer only every n tokens" with quality on a DSA model. Closest are SPIN
  (prediction as pre-filter, not reuse) and LServe.
- **Why it could work for us:** the pooled indexer works on blocks of 4 positions; consecutive decode tokens change the
  query slowly, and most of the 512 blocks should persist. The new tokens enter through the always-included tail cells
  (up to 3) and the new pooled block, so staleness is bounded to *old-context* blocks that become relevant. [estimate]
- **Why it could fail:** code agents jump (a tool-call result, a symbol lookup) and the relevant block changes at
  exactly those tokens. SPIN's authors state this risk themselves.
- A safe variant is *validated reuse*: reuse the last selection, but also score only the *delta*: new blocks since the
  last full run plus a random/rotating subset of the rest; trigger a full re-score when the (cheap) score of the kept
  boundary block moves. This is SPIN's idea reduced to a trigger. [estimate]

---

## 3. Combination with speculative decoding (MTP verify, several tokens per pass)

- **2512.21911 "Accelerate Speculative Decoding with Sparse Computation in Verification"** [HTML]: draft tokens of one step
  retrieve highly similar blocks (average overlap > 0.8 on Llama3.1-8B); the first draft token's query selects the blocks
  for the whole verify batch; plus inter-layer anchor reuse. Tested with EAGLE-3 (Llama3.1-8B, Qwen3-30B-A3B) and with
  DeepSeek-R1's MTP heads (hybrid; 30 of 61 layers anchors). Training-free. Sparse-attention accuracy loss 0.3-1.0
  ROUGE/F1 points at moderate sparsity; mean acceptance length slightly reduced; hybrid drops on CollegeMath 58 -> 53. No
  wall-clock results, no code.
- **SSV (arXiv 2605.19893)** [snippet]: sparse verification on NSA; either per-query semantics or an approximate shared-index layout
  with one representative query.
- Vegas / SpecAttn (arXiv 2602.07223) are the opposite direction (verification information chooses the KV for drafting).
- None of IndexCache's materials mention MTP.
- **For us:** a verify pass with m tokens currently needs m indexer runs per QSA layer (or one batched). Options, in
  rising risk: (1) F/S layer reuse is independent of m and stacks with everything else; (2) *one selection per pass* from
  the first (or last accepted-candidate) token's query, shared by the m tokens, a second saving of (m-1)/m on the
  remaining F layers; the later positions in the pass attend to some blocks that their own query would not choose, but the
  tail cells (up to 3) and the in-pass causal part are always exact. It is rejection-safe in the sense that a wrong
  selection only lowers the acceptance rate (the verified distribution is still the target model's *with* the shared
  selection, so it changes the target distribution slightly: the output is no longer identical to single-token decode with
  per-token selection; this must be listed as a quality-affecting change, not bit-exact). [estimate]

---

## 4. Transfer to our engine

### 4.1 How to pick the Full layers (question b)

- IndexCache evidence: search by LM loss, not by similarity; greedy removal from all-F; first layer always F; irregular
  pattern with denser F early. With only 12 QSA layers the full greedy search costs 12*11/2 = 66 evaluations, trivially
  affordable (the paper's cost problem does not exist for us).
- Do the search with **our metric (dNLL on code and long retrieval text at 64K-250K)**, not only short text; the paper
  calibrates at 200K context for a reason: selection similarity is context-length dependent.
- Search separately for **prefill** and **decode**: in prefill the indexer cost is 67-89 % of DSA time at 64K-256K (research/29,
  LongCat-Flash-Lite data) so the gain there is larger; the pattern that is lossless for decode may not be for prefill
  (queries in prefill are batched, and recall errors accumulate in the KV state of later layers; this is a risk, measure it).
- Keep the first QSA layer F (required) and check that the last QSA layer is F or has high overlap with its donor (paper:
  overlap of late layers <= 0.4 in a pure-DSA model).
- Training-aware (distillation of kept indexers) is out of scope without training infrastructure; note that it makes even
  uniform 1/4 lossless in the paper, so if we ever fine-tune the indexers, this is the cheapest quality win.

### 4.2 Risks for code and long-context tasks (question c)

- IndexCache's own long-context metrics show where the loss lands: at uniform 1/4, MRCR/GraphWalks/AA-LCR-type multi-hop
  retrieval suffers (-7 points avg); reasoning is untouched [HTML]. A needle test would likely pass (single, easy target)
  while multi-needle / aggregation fails first. So test with: RULER (multi-key NIAH, variable tracking, aggregation), a
  code-specific retrieval (find definition of symbol X across a 150K repo dump, edit-location accuracy), and our csv/rust/lang
  code benchmarks.
- A mean dNLL of a few thousandths of a nat hides rare large errors; also report top-1 agreement and the worst-position dNLL (max over
  windows), not only the mean.
- Prefill-sharing changes the stored KV of all later layers; decode-sharing only changes the current token's path. Hence
  decode is the safer first target (which is what we measured).

### 4.3 Megakernel and host-RAM KV (question on fit)

- S layers skip scoring, the pool read (f32 in VRAM) and top-k. Because the selection is known one or more QSA layers
  earlier, the **host-RAM gather for the S layer can be issued early** (InfiniGen/OmniKV style) and overlapped with the
  DeltaNet layers in between. This is likely the larger win at 250K when the gather is PCIe-bound; the measured -18 % comes
  only from removing the indexer. [estimate]
- In the megakernel the selection result is a fixed-size index buffer (2,048 cells + tail): sharing means one
  extra read of an existing buffer, no new synchronization. The S layer needs only its own tail cells (they
  depend on the current position and are layer-independent) and its own KV rows.
- Worth logging in our own runs: overlap of top-512 between each QSA pair (all 12x12), per step and per context length.
  It gives the Kascade/IndexCache similarity matrix on our model for free and tells whether the clusters in the paper exist
  here.

---

## 5. What to try next for us (ranked)

1. **Full greedy search over the 12 QSA layers by dNLL at 128K-250K on code + retrieval text, for 1/2, 1/3, 1/4
   retention (6, 4, 3 indexers), decode and prefill separately.** Why: 66 evaluations; IndexCache shows pattern choice is
   worth +7 points at 1/4 versus uniform; our 6/12 result is only one pattern. Also log the 12x12 overlap matrix.
   Gate: dNLL <= +0.003, top-1 agreement, RULER multi-key + a code retrieval test.
2. **Early gather for S layers** (issue the host-RAM KV fetch right after the donor F layer's top-k, overlap with the
   DeltaNet layers). Why: the indexer is not the only cost at 250K with KV in host RAM; the reuse makes the selection
   available early for free. Bit-exact relative to IndexCache alone, so it only needs a speed measurement.
3. **Apply the same F/S pattern in prefill** and measure prefill time and dNLL. Why: prefill indexer share is 67-89 % in the
   DSA comparison and IndexCache reports its largest speedups there (1.82x at 200K); it combines with the radix k_topk work
   (#71). Risk: errors enter the KV of later layers; hence rank 3, after the decode gate.
4. **One selection per MTP verify pass** (shared by the m tokens, from the first token's query) on top of the F/S pattern.
   Why: 2512.21911 reports > 0.8 overlap among draft tokens; gain scales with m. Not bit-exact with per-token decode: measure
   acceptance length and dNLL, compare against per-token selection.
5. **Temporal reuse with a trigger** (re-run the indexer every n = 4..16 tokens, keep tail cells exact, full re-score when
   a cheap drift signal fires). Why: biggest unexplored gain after layer reuse, but no paper shows it works on a DSA model and
   SPIN's recall numbers (0.58 previous-step) argue for caution; run offline first: replay recorded decode traces, compute the
   recall of the stale selection against the fresh one per n, by token type (code symbol, whitespace, tool result).
6. **Training-aware indexer distillation** (IndexCache's multi-layer loss) only if rank 1 shows that 1/4 retention is the
   limit and we want it lossless. Why last: needs training pipeline and data; paper shows the benefit (uniform 1/4 -> 50.6),
   but the training-free search already gets close.

Not recommended now: HashAttention (learned Hamming signatures, needs a trained mapping, solves a different problem than
our already-pooled indexer), Twilight (top-p pruning on top of a base selector; could shrink the 2,048 cells adaptively
but changes the selection, a separate topic), RetroInfer/ShadowKV (cluster index and low-rank key cache for dense models;
our pooled indexer already plays this role).

---

## Sources

- IndexCache: https://arxiv.org/abs/2603.12201 (HTML https://arxiv.org/html/2603.12201v1), code https://github.com/THUDM/IndexCache, vLLM docs https://docs.vllm.ai/en/latest/features/index_cache/
- Kascade: https://arxiv.org/abs/2512.16391
- TidalDecode: https://arxiv.org/abs/2410.05076 , https://github.com/DerrickYLJ/TidalDecode
- OmniKV: https://proceedings.iclr.cc/paper_files/paper/2025/hash/da1131a86ac3c70e0b7cae89c3d4df22-Abstract-Conference.html , https://github.com/antgroup/OmniKV
- InfiniGen: https://arxiv.org/abs/2406.19707
- ShadowKV: https://arxiv.org/abs/2410.21465
- Quest: https://arxiv.org/abs/2406.10774
- Twilight: https://arxiv.org/abs/2502.02770
- LServe: https://arxiv.org/abs/2502.14866
- RetroInfer: https://arxiv.org/abs/2505.02922
- HashAttention: https://arxiv.org/abs/2412.14468
- SPIN: https://arxiv.org/abs/2610.09025
- LFPS: https://arxiv.org/abs/2506.15704
- Dynamic Sparse Attention access patterns: https://arxiv.org/abs/2603.13430
- CLSA / You Only Index Once: https://arxiv.org/abs/2606.06467
- Autonomy-of-Heads: https://arxiv.org/abs/2608.06849
- HySparse2: https://arxiv.org/abs/2609.26368
- Sparse verification in speculative decoding: https://arxiv.org/abs/2512.21911 ; SSV https://www.alphaxiv.org/abs/2605.19893 ; Vegas/SpecAttn https://arxiv.org/abs/2602.07223
- llama.cpp lightning indexer op PR #24231 (status unconfirmed): https://github.com/ggml-org/llama.cpp/pulls
