# 68 CECAR: reuse-distance-aware expert routing from SSD (#172), 2026-11-18

arXiv 2611.20119 (Kim et al.), code github.com/hoonki2002/CECAR (MIT). This is a lossy lever (it changes which experts run, so the output changes). **The engine's routing was not changed.** the maintainer decides. Offline replay on our recorded routing traces: `research/measurements/2026-11-18-cecar-offline-replay.md`.

## 1. Decision page

**What it is.** Two separable parts. (a) A learned eviction score (small network on recency and frequency). Lossless: it only chooses what to keep. (b) Routing that prefers resident experts: a bonus on the router logits of cached experts, plus dropping selected experts that are not cached and rank low. This one is lossy.

**Gain (replay, Coder, 90 of 256 slots per layer, nine of 48 layers, two traces that agree):**

| Variant | Hit rate against LRU 76 to 78 % (demand only) | Misses per layer-pass | Selections replaced | Routing weight lost | Decode estimate* |
|---|---|---|---|---|---|
| learned eviction, exact routing (lossless) | +1.5 to +2.4 points | -7 to -10 % | 0 | 0 | +0.2 to +0.7 % realistic, +1.5 % at replay gain |
| pruning only, tau 8 | +5 to +6 | -30 % | 5 % | 2.6 % | +2.3 % (1.4 to 6) |
| pruning only, tau 6 | +9 to +11 | -50 % | 11 % | 5.7 % | +4 % (2 to 10) |
| bonus m 3 + pruning tau 6 | +14 to +16 | -70 to -75 % | 21 to 24 % | 12.6 to 14 % | +5 % (3 to 13) |
| paper-like (bonus m 10, tau 6) | +19 to +21 | -88 to -91 % | 38 % | 26 to 27 % | +6.5 % (4 to 16) |
| oracle (Belady, exact) | +11 to +12 | -50 % | 0 | 0 | (not reachable) |

\* Decode estimate, from our counters, unmeasured for these variants: the real engine already preloads the next layer (LRU about 90 % hits, `2026-10-25-cache-policy.md`), about 158 misses per pass (`2026-11-02-classlru.md`), i.e. about 3.3 per layer-pass; this table applies the replay's relative miss reduction to that baseline. Cost per miss: 15 to 60 microseconds (plan of `2026-11-02-classlru.md`), **23 microseconds central from its server A/B** (6.3 % fewer misses, about 10 per pass, gave +0.5 % decode). Example, pruning tau 6: about 72 fewer misses x 23 us = 1.7 ms of a 46 ms pass (+4 %); range from 15 to 60 us in brackets. Removing every miss would be worth at most about 3.5 ms at 23 us (+8 %). On the 16 and 32 GB classes, where missing experts come from the SSD (decode 10 to 63 tok/s, `2026-11-16-pc-dm-vs-coder.md`), the same relative reduction is worth more; not estimated here.

**Cost.** The price is paid in output. Our Coder has 256 fine-grained experts with a flat router (top-10); to beat the exact oracle at all the replay has to replace 17 % or more of the selections (pruning tau 4: 17 % replaced, 10 % of routing weight lost, 91 % hits). The paper's model (Qwen3-30B-A3B, top-8 of 128) keeps 7.13 of 8 experts on average with no accuracy loss beyond noise (HumanEval 67.9 against 67.7, MBPP 61.1 against 64.4, GPQA-D 36.4 against 36.7, MATH-500 56.4 against 56.7, three seeds); the nearest row for us is pruning tau 8 or bonus m 2, tau 10 (5 to 11 % replaced). The authors' own ablation shows the risk: bonus routing alone cost 3 points on HumanEval (47.56 to 44.51) on DeepSeek-V2-Lite until the learned score was added. They did not test long context, quantised experts (ours are IQ1_M: 1.x bit, where the expert difference is already noisy), batched serving or MTP verify passes with several tokens per pass, all of which Tona uses.

**Quality check needed before any build (rule 3, `docs/backends/README.md`):** teacher-forced NLL over the 25 stored Claude Code turns (the #106 method, 0.01 nat/token limit) with the changed routing against exact routing, plus the code benchmarks (csv, rust, lang) and the 128K needle test; sweep tau 8 / 6 first, bonus only with the lossless eviction in. Replays cannot see it: the following tokens change when the routing does. A real run needs a Tona switch (`TONA_ROUTE_*`), decoder-side only, default off; prefill unchanged (the paper applies it in decode only).

**Our current eviction policy (read from the code, not the docs).** LRU, per layer: `TONA_EXPERT_POLICY` defaults to `lru` (`engine/gpu/mega_build_bufs.cpp:91-95`); `classlru` (`engine/backend/class_lru.h`: CRF value, 4 classes, half-life 128 tokens) is an opt-in value and not the default. In steady state a missed expert does **not** enter the cache: refill happens only while slots are free (`TONA_FILL` 0, `fill_cap_` 4, `mega_build_bufs.cpp`, `mega_select_cpu.cpp:174`); the cache changes by hits (promotion) and by preloading 8 experts per layer and pass at the back of the list (`mega_prefetch.cpp`). So the victim policy only decides which preloaded or promoted entry leaves first.

**Candidate for #137 (lossless, no routing change): learned or frequency-aware eviction, +2 points hit rate.** Cost of a miss from our counters: `research/measurements/2026-11-02-classlru.md` (server A/B, hits and misses per request from `/status`): 6.3 % fewer misses (about 10 of 158 misses per pass, 46 ms pass) gave **+0.5 % decode** (six paired rounds, interval includes 0 in 7 of 8 metrics), which is about 23 microseconds per miss; the plan's range there is 15 to 60 microseconds per miss. 158 misses per pass at about 90 % hits means about 1,500 expert requests per pass (48 layers), so **+2 points = about 30 fewer misses per pass = 0.45 to 1.8 ms of a 42 to 46 ms pass, central 0.7 ms = +1.5 % decode (range +1 to +4 %)**. Two cautions that pull it down: (1) the +1.5 to +2.4 points are replay numbers at **demand only**, without the preloading that already lifts LRU to about 90 %; for ClassLRU the same kind of replay gave +0.76 / +1.12 points and the server gave +0.1 to +0.6 % (same file), so the realistic server gain of a learned score is about **+0.2 to +0.7 %**; (2) the learned score costs host time per victim (the old LRFU scan cost 0.3 to 1 ms per pass, same file), which a table lookup of a tiny network can keep small but has to be measured. Small gains count (no 5 % hurdle), the change is bit-identical in output only if it is the same routing (it is: eviction does not change which experts run, but `TONA_DETERMINISTIC` runs must give identical tokens, which they do since the expert set per token is unchanged). Next step if wanted: replay with the server's preloading and fill rules (`tools/cache_sim/sim.py` has them) with the learned score as the victim order, on a workload-diverse trace; build only if it beats LRFU/ClassLRU there.

**What a quality check of routing pruning would cost (tau 8: -30 % misses for 5 % of selections replaced):** a default-off decode-only switch plus, for each of tau 8 and tau 6, the teacher-forced NLL over the 25 stored turns (limit 0.01 nat/token), the three code benchmarks (csv, rust, lang), a 128K needle run and the draft-acceptance counters, 2 arms x 3 rounds on the GPU, plus one A/B for speed; the decode gain it would have to justify is about +2 % (estimate above).

**Recommendation.** (1) Do not build the lossy routing for the Coder now: the mildest useful row replaces 5 % of selections for about +2 %; the Coder is already at 55 to 110 tok/s. (2) The lossless learned eviction is a small candidate (+0.2 to +0.7 % realistic, up to +1.5 % at the replay's gain), needs no maintainer decision, and is listed on #137 for a replay with preloading first. (3) Revisit routing for the 16 and 32 GB classes where misses hit the SSD, with a quality run of tau 8 and tau 6 only; the decision would rest on the class's decode gain, which has to be measured there.

See "The method as published" below, and the replay table (trace A / trace B) in the measurement file. Summary of the replay: learned eviction +1.5 (A) / +2.4 (B) points over LRU, LRFU +0.9 / +2.0, Belady +11.2 / +12.0; the routing rows are in section 1.

### The method as published

CECAR ("Cache Expert Co-Aware Routing") fetches MoE experts from SSD into a per-layer GPU expert cache during decode and changes the routing so that it prefers experts that are resident. Three parts:

1. **Learned reuse predictor (eviction score).** Per expert and layer two scalars: recency `r_norm = 1 / (steps since last use)` and frequency `f_norm = uses / max uses`. Two small expert-independent feed-forward networks (hidden size 32), shared by all experts of a layer, map them to a score: FFN-Hot (short-term target, `T_hot = (D-1)/(D_hot-1)` for reuse distance `D < D_hot`, else 1) and FFN-Cold (long-term target 0 / 0.5 / 1 around `D_cold`, `2 D_cold`). The scores are merged with weights that depend on the cache size (hot detector weighted more for caches up to 25 % of the experts, cold detector above). Trained per layer on router traces of 512 TriviaQA sequences of 512 tokens (Belady-style reuse distances). The scorer's size does not matter (sweep over 15 sizes, 1000x range: OOD hit rate changes by less than 0.5 %).
2. **Cache-aware routing ("bonus").** Per layer and token: margin `Delta = z[1] - z[m]` of the sorted router logits (top-1 minus the m-th); every resident expert gets `z'_e = z_e + Delta * delta_e` with `delta_e` derived from its eviction score (the best-to-keep resident experts get close to the full margin); top-k selection on `z'`. The final softmax weights use the original logits, normalised over the original top-k, so a dropped expert's weight is lost, not redistributed.
3. **Cache-conditioned pruning.** Of the selected experts, keep one if it is resident or if its rank `j <= tau`; `tau` is fixed per model (4 for Qwen3-30B-A3B at top-8, 3 for DeepSeek-V2-Lite at top-6). Misses that stay are loaded and replace the resident expert with the highest eviction score.

Reported results (own cache of 24 of 128 experts per layer, SSD offload, host DRAM pinned to about 1 GB, temperature 0.7, 2,048 generated tokens, accuracy over 3 decoding seeds; accuracy measured with a virtual cache with all weights resident, speed on the real offload setup):

| Qwen3-30B-A3B | HumanEval | MBPP | GPQA-D | MATH-500 | avg experts used (of 8) | hit rate | decode |
|---|---|---|---|---|---|---|---|
| baseline, exact routing (LRU) | 67.7 | 64.4 | 36.7 | 56.7 | 8.00 | 68.3 % | 2.8 tok/s |
| MoCCE (earlier cache-aware routing) | 67.5 | 61.8 | 38.0 | 55.3 | 8.00 | 92.3 % | 8.6 |
| CECAR | 67.9 | 61.1 | 36.4 | 56.4 | 7.13 | 96.6 % | 12.5 |

- The learned eviction policy alone gains only 3 to 4 points over LRU at exact routing (Table 2: 73.6 against 69.5 at 24 experts on TriviaQA; Belady 82.6; 57.95 against 54.83 in the DeepSeek ablation). Its value is robustness out of distribution and across cache sizes: the earlier learned policy (FlashMoE) falls below LRU on HumanEval and MBPP (44.8 against 65.5 at cache 24).
- The big hit-rate gain comes from the routing change: the ablation on DeepSeek-V2-Lite shows bonus routing 54.8 -> 75.9 % hit and 3.40 -> 5.45 tok/s but HumanEval 47.56 -> 44.51; with the learned score 51.83 (accuracy back up). Pruning alone 63.5 % hit, 7.35 tok/s. All three 86.9 %, 8.98 tok/s, 49.39.
- The authors state: with exact routing even Belady cannot exceed 84.2 % at 24 of 128 experts, so replacement alone cannot fix an I/O-bound decode; they say routing must change.
- Not evaluated by the authors: batched serving, long-context generation, quantised expert storage, composition with exact-routing prefetchers ("future work"); abrupt topic shifts within a request; single-seed hit rates; sensitivity to `D_hot`, `D_cold`, cache size.

## 3. Comparison with our expert cache

- Ours (`engine/backend/class_lru.h`, `mega_prefetch.cpp`): per-layer LRU or ClassLRU / LRFU admission, next-layer preloading from the lookahead (8 candidates, protected). Exact routing, bit-identical output. With preloading LRU is already level with Belady without preloading (`2026-10-25-cache-policy.md`: 91.9 against 92.3 %, 90.1 against 90.5 %); the gap to Belady with preloading is 5 to 7 points.
- Theirs: no preloading (the paper lists composition with exact-routing prefetchers as future work); eviction by the learned score; routing and eviction share the score.
- The replay (demand only) puts the learned score +1.5 to +2.4 points over LRU, LRFU +0.9 to +2.0: LRFU captures most of it, which agrees with the paper's own statement that the learned policy mostly buys robustness out of distribution.
- Preloading is orthogonal to the bonus: the bonus raises hits on what a pass needs, preloading brings the next layer's experts early. The estimate above does not credit any interaction; it is the weakest number in this note.

## 4. Prior art and sources

- arXiv 2611.20119 (read in full from the PDF), repo hoonki2002/CECAR (README only), FlashMoE (Belady-trained ML eviction, cited in the paper), MoCCE (binary cache-aware bonus, Skliar et al. 2025), DES/ODP (weight-based pruning), related: profile-then-pin llama.cpp #32212 (not read here); own: `research/measurements/2026-10-24-misses.md`, `2026-10-25-cache-policy.md`, `2026-11-15-expert-coactivation.md`, `docs/architecture/expert-tiers.md`, `tools/cache_sim/`.
- Not done: the repository's code (the README names the flags `--mode`, `--bonus-strategy cecar` and the checkpoints, not the routing sources), training their networks, measuring on the engine.
