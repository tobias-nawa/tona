# 32 — 1M context with YaRN: switching, quality cost, memory and time on the reference platform

Agent research (Sonnet), 2026-10-29, issue #77

Method: web search and page retrieval (model cards, transformers docs and source, YaRN/LongRoPE2/MrRoPE/DroPE
abstracts), a read of our own code (`engine/gpu/mega_dev.h`, `mega_build_bufs.cpp`, `mega_build_layer.cpp`,
`mega_build_mtp.cpp`, `mega_seq.cpp`, `prefill_qsa.hip`, local llama.cpp `src/models/qwen4exp.cpp`) and our
measurements. Nothing was measured or built here, no engine code changed. Marking per entry:
- **[card]**: model card / config read (numbers are the vendor's).
- **[code]**: source read directly (ours, HF transformers, llama.cpp).
- **[abstract]** / **[snippet]**: abstract page only / search-result snippet only.
- **[calc]**: own calculation from sizes in the code. **[estimate]**: extrapolation of our measured curves.

Rates are named: **prefill** (prompt tokens per s) or **decode** (generated tokens per s, MTP, ~2.6-3.0 tokens per pass).

Scope (coordinator, 2026-10-29): YaRN/NTK/Dynamic for our model, memory, product option. The survey of newer
methods (2025-2026) is research/33-context-extension-survey.md; the findings I already had are only listed in the
appendix.

---

## 0. Summary

1. **What the vendor recommends.** Qwen3.8-Flash-Next [card]: 262,144 native, "extensible up to 1,000,000" with
   `rope_parameters = {rope_type: yarn, factor: 4.0, original_max_position_embeddings: 262144, rope_theta: 1e7,
   partial_rotary_factor: 0.25, mrope_interleaved: true, mrope_section: [11,11,10]}`. The card calls the frameworks'
   YaRN **static** ("scaling factor remains constant regardless of input length", may hurt shorter texts), says to
   add it only when needed and to lower `factor` to the typical length (2.0 for 524,288). The card has **no RULER or
   other text benchmark beyond 262K**. The only vendor numbers are for the predecessor Qwen3-Next-80B-A3B-Instruct
   with YaRN: RULER 93.5 at 256K, 86.9 at 512K, 80.3 at 1M (avg 91.8; section 3).
2. **YaRN touches little in this model.** With theta = 1e7 and 64 rotated dims (32 frequency pairs) only pairs 15-21
   get a partial interpolation and pairs 22-31 the full /4; pairs 0-14 (the fast ones, local order) are unchanged
   (section 1.2, [calc]). Plus `mscale` = 0.1 ln 4 + 1 = 1.1386 on cos and sin.
3. **Dynamic is possible and cheap here, because RoPE is a rotation.** Three ways (section 2): (A) decide per request
   from the known prompt length (no cache change, prefix cache keyed by mode); (B) re-rotate the stored K and the
   indexer pool once when a running conversation crosses 262,144 (about 0.4 GB of K data in RAM and 0.4 GB pool in
   VRAM, seconds, approximate because of int8 requantisation and because the content of deeper layers was computed
   under native RoPE); (C) store K unrotated and rotate on read (any factor free, but changes numbers and the kernels).
   No engine (llama.cpp, vLLM, SGLang, transformers) does (B) or (C) for YaRN today; all are static.
4. **Quality cost: unknown for this model.** Published: short-text loss of static YaRN is "may" in all Qwen cards,
   never quantified; Qwen3-Next-with-YaRN RULER falls 98.5 (4K) -> 93.5 (256K) -> 80.3 (1M), but that mixes length
   effect and YaRN effect. We must measure (plan in section 6).
5. **Memory at 1,048,576 tokens [calc]:** KV int8 15.7 GB (pinned RAM), indexer pool 1.74 GB (VRAM, f32), per
   sequence slot. **Fits only with one slot** (`IE_SEQS=1`), all pinned experts (25.3 GB) and a small prompt-RAM
   budget; costs ~13 expert slots per layer in VRAM (80 -> ~67, prefill about -5 %). Not with the default 2 slots (31 GB).
6. **Time [estimate]:** prefill of 1M about **16-23 min** (average 720-1,040 tok/s; 250K today: 194 s at 1,285
   tok/s). Decode pass about **65-80 ms** after the 1M prompt (47 ms at 250K today), i.e. ~35-45 tok/s;
   `IE_IDX_SHARE` (8 full layers) saves ~8 ms. PCIe is **not** the limit: the gather is 2,051 cells per token and
   layer, independent of length.
7. **Recommendation:** first measure whether 262K -> 524K (factor 2) is cheaper and better than 1M (factor 4); build
   mode A (server option `--ctx` + `--rope-yarn auto|off|N`, decided per request) before B/C; no 1M claim before a
   retrieval + code test beyond 262K exists.

---

## 1. What the model does with positions

### 1.1 In our engine [code]

- RoPE is NeoX style: pair (i, i+32) of the first **64** of 256 dims (`rope_pair(..., n_rot=64, pos, base)`,
  `engine/gpu/mega_dev.h:24`), angle `pos * base^(-2i/64)` in double, base `rope_base` = 1e7 from the GGUF. For text
  tokens the three M-RoPE sections have equal positions, so interleaved M-RoPE equals plain partial RoPE
  (research/06 section 5). There is **no** YaRN, scaling or `mscale` code in the engine: `rope_scaling` keys of the
  GGUF are not read (`engine/model/qwen4exp.cpp:60`).
- RoPE is used in four places, all with absolute position: attention q (after RMSNorm), attention k (stored rotated),
  indexer q, and the pooled indexer key (rotated at the position of the block's first token, stored f32 in the
  pool). The raw indexer keys (ring) and the DeltaNet layers do **not** use positions (the conv is causal and
  position-free; the recurrence carries order itself).
- Both reference implementations apply the same scaled cos/sin to the indexer: HF `Qwen4ExpTextQSAIndexer` gets the
  model's `position_embeddings` and rotates q and the pooled block keys with them [code]; llama.cpp `qwen4exp.cpp`
  passes `n_ctx_orig, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow` to `ggml_rope_multi` for the pooled
  key (l. 802), the indexer q (l. 818) and attention q/k (l. 1007-1016) [code]. So YaRN applies to attention **and**
  indexer. That the HF rotary class multiplies cos/sin by `attention_scaling` is the standard behaviour of the
  inherited class; I could not see that file (`Qwen3_5TextRotaryEmbedding`) in the fetch.
- Consequence of partial RoPE: the 192 non-rotated dims carry content only and are not changed. `mscale` scales
  cos and sin of q and of k, i.e. it multiplies only the **rotary part** of q.k by 1.1386^2 = 1.296 (llama.cpp
  folds the same factor into `mscale` in the rope kernel: `cos * mscale`, `ggml-cuda/rope.cu:36`) [code].

### 1.2 What YaRN changes at theta = 1e7, 64 dims, factor 4, original 262,144 [calc]

YaRN (arXiv 2309.00071): per frequency pair, wavelength lambda_i = 2 pi theta^(2i/64); r_i = original / lambda_i
rotations inside the native window. r > 32 (`beta_fast`): unchanged. r < 1 (`beta_slow`): frequency / 4. Linear
ramp in between. For us:

| Pairs i | lambda (tokens) | rotations in 262K | Treatment |
|---|---|---|---|
| 0-14 | 6 ... 7,260 | 41,700 ... 36 | unchanged |
| 15-21 | 12K ... 247K | 22 ... 1.06 | ramp, 88 % ... 12 % kept |
| 22-31 | 408K ... 38M | 0.64 ... 0.007 | frequency / 4 |

Reading: after the change the slowest pairs advance 4x slower, so position 1M looks like 262K to them (phase
0.026 turns instead of 0.1); the fast pairs, which carry exact local order, keep their native phases. The model
spent all training up to 262K with native frequencies; the damage to short texts therefore comes from pairs 15-31
(half of all pairs, but the slow ones) and from the 1.296 logit gain on the rotary part, which every query and every
distance sees. Whether the factor 1.296 hurts short code is exactly what the cards only call "may".
`mscale` is a free parameter (`attention_factor` in HF, `--yarn-attn-factor` in llama.cpp); the YaRN paper
fitted 0.1 ln s + 1 on Llama, nothing is published for Qwen3.5+ with partial RoPE [no source].

---

## 2. Static versus dynamic, and the cache problem

### 2.1 What the others do

| Engine | Behaviour | Source |
|---|---|---|
| Qwen cards (all since Qwen2.5) | "static YaRN, factor constant regardless of input length"; add `rope_scaling` only when needed; Qwen3-4B-era cards put the loss at "average inputs under 32,768 tokens" | [card], vLLM-doc quote seen via search |
| transformers | rope types `default, linear, dynamic, yarn, longrope, llama3, proportional`; `dynamic` is **NTK-aware base rescaling**, not YaRN; YaRN is static (its `factor`, `beta_fast=32`, `beta_slow=1`, `attention_factor`). The commit that added YaRN to transformers **removed Dynamic-YaRN** | [docs], semanticdiff of the commit (search hit) |
| vLLM | only static YaRN ("presently only supports static YARN"). Forum thread: dynamic schemes do not work because rotation is applied before the KV cache is written, so cached K goes stale | [card], discuss.vllm.ai thread 631 (user report, may be dated) |
| SGLang | same JSON override as vLLM (`--json-model-override-args`), static | [card] |
| llama.cpp | `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 [--yarn-attn-factor ..]`, from GGUF metadata otherwise; static per context. `cparams.yarn_attn_factor *= 1/(1+0.1 ln factor)` is a compensation so that the kernel's own `mscale` is not applied twice. Open: issue #17459 (server refuses the longer context with yarn and scale 4; snippet only, cause not read) | [code], search snippets |

YaRN paper (2309.00071, section on Dynamic Scaling): the scale factor is recomputed per forward pass from the current
length, s = max(1, l'/L); for KV caching "the correct approach caches key-value embeddings **before** rotary
embedding", because each token's rotation changes with s [abstract/snippet]. A 2026 survey (arXiv 2608.10021
"Position Encoding in Transformers", snippet only, PDF text not readable by the fetch tool) names the two fixes:
re-rotate cached keys, or hold the scale fixed within a session. Nobody ships either for YaRN.

### 2.2 Why a switch is mathematically simple here

A stored key is K~ = R(p f_old) k. Under another frequency set the key would be R(p f_new) k = R(p (f_new - f_old))
K~: a rotation of the stored value per pair by the angle p (f_new - f_old); for pairs 0-14 the angle is 0 (nothing
to do), `mscale` multiplies the rotary dims. With QK-norm before RoPE, k has no other dependency on position.
This holds equally for the pooled indexer key (f32, rotated at the first position of its block) and not at all for
anything in the DeltaNet layers.

What is **not** exact: (a) the cached values and keys of layers >= 1, and the DeltaNet state, were computed from
hidden states that were themselves produced under the old attention. A fresh prefill under YaRN would give other
KV. The re-rotation gives the conversation "native content, YaRN geometry" ([estimate]: the error is of the order of
the YaRN-vs-native difference on the first 262K, which is the quantity we need to measure first). (b) K is stored
int8 with a scale per 32 values (`kv_row_` = 1,152 B: 1,024 B K+V int8 + 128 B scales); rotation mixes pair (i,
i+32), i.e. groups 0 and 1, so these two groups need requantisation (one more rounding of ~1/254 relative).

### 2.3 Options for the engine

| | A: per request, static | B: re-rotate once at 262,144 | C: unrotated K, rotate on read |
|---|---|---|---|
| Rule | prompt + max_tokens <= 262,144 -> native; else YaRN from token 0 | native until a request would cross 262,144, then rotate cache + pool, continue in YaRN | K stored before RoPE (+ mscale), rotation by the gather kernel; any factor, any time |
| Cache | none; prefix cache and saved conversations carry the mode; a mode change = miss (full prefill) | K of 12+1 layers, positions < 262K: 128 B of 1,152 B per row touched = ~0.4 GB; pool 0.43 GB; ~1-3 s [estimate] | none at switch |
| Bit-exact to a fresh run | yes (same as static YaRN run) | no (see 2.2) | no (K quantised before rotation, today after) |
| Decode cost | none | none | rotate 2,051 cells x 2 heads x 32 pairs per layer and token: small against reading 1.1 KB rows, needs a cos/sin table up to 1M |
| Prefill cost | none | none | attention kernel reads K and rotates (WMMA path changes) |
| Engine work | RoPE tables per mode in 4 places (decode, prefill, MTP, drafter), GGUF keys, API | A + rotation kernel + state flag + mscale handling | rewrite of KV layout and 3 attention paths |
| Risk | agents that grow across 262K lose the prefix cache once | unmeasured quality; MTP KV needs the same treatment | speed and new numerical path |

My ranking: A first (small, exact, enough for "give me a 400K prompt"); B only after the measurement in section 6
says the approximate state is as good as a fresh YaRN prefill; C only if we want continuous factors (e.g.
factor = ceil(length / 262,144) in steps of 0.5), which the card's own advice ("factor 2.0 for 524,288") already
approximates with mode A and a server-wide factor.

Prefix cache and saved conversations (`IE_PROMPT_RAM_GB`, research/10): the cached keys depend on the rope mode and
factor, so mode and factor belong in the cache key; a conversation of 700K tokens holds ~10 GB of KV and does not fit
the 12 GB default budget with others [calc].

---

## 3. Quality: what is known

### 3.1 YaRN, the paper and the model families

- YaRN [abstract]: fine-tuning on ~0.1 % of the pre-training data extends Llama 2 to 64K/128K with small
  perplexity loss; the dynamic variant extends >2x without any fine-tuning. Our case is the zero-training case.
- Qwen3-Next-80B-A3B-Instruct [card, same family, DeltaNet hybrid; evaluated with YaRN]: RULER (accuracy %; 260
  samples per length): 4K 98.5, 32K 98.7, 128K 96.0, 256K 93.5, 384K 91.7, **512K 86.9, 768K 81.7, 1M 80.3**,
  average 91.8. For comparison in the same table: Qwen3-235B-A22B-Instruct-2507 (DCA) 256K 91.0, 512K 90.9, 1M
  84.5; Qwen3-30B-A3B-2507 256K 82.5, 1M 72.8. The 2507 models are evaluated with Dual Chunk Attention, Qwen3-Next
  with YaRN. Beyond 262K the score falls about 6-7 points per doubling; RULER is a retrieval test, not code.
- No number anywhere for the loss of YaRN **below** the native length for a Qwen model. A vision-language long-context
  study (arXiv 2505.16928, snippet) found YaRN fine at moderate factors and clear degradation below 64K at factor
  16 (a different model, an extreme factor). LongRoPE2 reports that YaRN and NTK "fall behind past 32-64K on RULER at
  128K" on Llama-3-8B (secondary summary, not the paper's table).
- Qwen2.5-1M tech report (arXiv 2501.15383): the 1M models are **trained** (staged ABF to long contexts) and use DCA
  (training-free, +4x) plus MInference-like sparse attention at inference; Qwen2.5-14B-Instruct-1M RULER 92.2 at
  128K. Not applicable to us without training; DCA is the training-free alternative to YaRN that Qwen itself used for
  the 2507 line (appendix).

### 3.2 Hybrid with linear attention beyond the training length

- Gated DeltaNet has a fixed state (108 MiB), no position input; the recurrence does not know the length. The
  Gated DeltaNet paper tests length extrapolation only to 20K (arXiv 2412.06464, results "mixed but better than
  Mamba2"). The only 2026 study found: Liu, He, Qiu, "Mechanics of Long-Context Hybrid Models Part 1.1"
  (arXiv 2610.10114, 2026-10-07, abstract only): hybrids of full attention with SWA extrapolate better without
  training, hybrids with GLA/GDN gain more from long-context continual training, and a "Matthew effect of hybrid
  position extrapolation" is named; their sliding-window-linear variant gets 16x training-free. No GDN-specific
  number in the abstract; the paper (60 pp.) must be read before relying on it.
- For us this means: the 36 DeltaNet layers are a risk that RoPE scaling cannot repair (no positions in them),
  and that neither the card nor any paper measures for Flash-Next; the card's YaRN statement covers QSA only. Qwen3-Next
  (same 3:1 pattern) reaches RULER 80.3 at 1M with YaRN, so the state does not collapse there.

### 3.3 Sparse attention beyond the training length

DSA/QSA was trained and measured at 128K (DeepSeek-V3.2-Exp) and at 262K (here). Nothing found on top-k=2,048 and an
indexer beyond that length (snippets); the indexer scores more blocks (262K at 1M), so a distracting-block risk
grows with length and the position scaling of the indexer is part of the question (YaRN changes indexer scores, see
1.1). Our own IndexCache result is relevant: retrieval is where selection errors show first
(research/30, measurements/2026-10-29-indexcache.md).

---

## 4. Memory at 1M tokens on the reference platform [calc]

Sizes from the code. 1M = 1,048,576 (2^20); the card's 1,000,000 is smaller.

| Item | Formula | Per token | At 262,144 | At 1,048,576 | Where |
|---|---|---:|---:|---:|---|
| KV, 12 QSA layers | `kv_row_` 1,152 B x 12 | 13,824 B | 3.62 GB | **14.50 GB** | pinned RAM (`IE_KV_HOST`, auto above 1 GiB) |
| KV, MTP layer | 1,152 B | 1,152 B | 0.30 GB | 1.21 GB | same |
| Indexer pool, 12 layers | (ctx/4 + 1) x 128 f32 = 128 B per token | 1,536 B | 0.40 GB | **1.61 GB** | VRAM |
| Pool, MTP layer | | 128 B | 0.03 GB | 0.13 GB | VRAM |
| Scores buffer | `sstride_` floats | 1 B | ~0 | 1 MB | VRAM |
| Total per sequence slot | | 16.6 KB | 4.35 GB | **17.45 GB** (15.7 RAM + 1.74 VRAM) | |

`IE_SEQS` default is 2 and each slot has its own KV and pool (`mega_seq.cpp`: `alloc_seq_buffer` allocates every
registered buffer again), so 1M with the default is 31.4 GB RAM + 3.5 GB VRAM. Documented as forced to 1 with a
drafter and with `IE_STATE_HOST`; whether the MTP path forces it is not stated in `docs/use/environment.md`, so
set `IE_SEQS=1` explicitly for 1M. DeltaNet state (108 MiB), conv state, ring: independent of length.

**VRAM:** the pool grows by 1.31 GB. Today at 250K the engine holds 14.47 GB with 80 expert slots per layer
(measurements/2026-10-28-long-context). One slot index over all layers is ~98 MB (0.98 GB per token / 480 expert
reads = ~2.0 MB per expert x 48 layers) [calc], so the pool costs ~13 slots (80 -> ~67): prefill -5 % judging by
the slot curve 70 -> 91 slots = 1,619 -> 1,746 tok/s (2026-10-25-slots), decode "hardly any marginal benefit
left at 80-85 slots" (2026-10-26-slot-memory). Alternative: pool as f16/fp8 (0.9 / 0.45 GB at 1M), not bit-exact,
needs the quality check of docs/backends/README.md. Pool in host RAM is not an option: the decode indexer reads the whole
pool every pass (1.74 GB; 32 ms at 55 GB/s versus 5.5 ms from VRAM).

**RAM (64 GB, `MemoryMax` = RAM - 6 = 58 GB as in the measurements):** pinned experts 25.3 GB with "all or none"
(2026-10-27-ram-limit; 29.2 GB pinned in the 250K run) + KV 15.7 GB = 41-45 GB, leaving 13-17 GB for the process,
the page cache of the N-gram table and the prompt store (`IE_PROMPT_RAM_GB` default 12 should be lowered; a
second stored conversation at this length is impossible). Fits, tightly, with the desktop and agent sessions
outside the cgroup. KV at int4 would halve it (7.9 GB) but K per channel/rotation questions come first
(research/papers.md KV quantisation); not needed to fit.

**Decode traffic does not grow with context:** each token reads 2,048 + up to 3 cells per layer = 2,051 x 1,152 B =
2.36 MB per layer, 28.3 MB for 12 layers (x ~3 tokens of the verify pass, less where cells overlap) = ~85 MB per
pass -> 1.5 ms at 55 GB/s; it grows with context only through the indexer scan (below) and the scatter of the cells
across more pages (measured at 128K -> 250K: attention gather per call grew, research 2026-10-28).

Result: **fits with `IE_SEQS=1`, all experts pinned and a reduced prompt store; costs ~13 expert slots in VRAM; does not
fit with two slots.**

---

## 5. Time at 1M [estimate]

### 5.1 Prefill

Measured curve (main after #71, 3 rounds, `--ctx 262144`): mean time per prompt token a(N) = 0.600 ms (8K), 0.668
(65K), 0.721 (131K), 0.778 (250K), i.e. 1,666 / 1,497 / 1,386 / 1,285 tok/s (measurements/2026-10-29-prefill-topk).
Two extrapolations of a(N) = a(250K) + s (N - 250K):
- s = 5e-7 ms/token^2 (the 131K -> 250K slope): a(1M) = 1.15 ms -> **~19 min** (870 tok/s average).
- s = 8e-7 (the 65K -> 131K slope): a(1M) = 1.38 ms -> **~23 min** (720 tok/s).

Cross-check by component (rocprofv3 at 250K, 191.4 s GPU: indexer scoring 14.5 s + `k_topk` 4.2 s are quadratic, the
rest linear): quadratic x16 = 232 s + 67 s = ~300 s, linear (191 - 19) x 4 = 690 s: ~990 s = **16.5 min**, before
any growth of the attention gather (`k_attn_wmma`, 24 s of 93 s at 128K) which makes the real number larger. So 16-23 min,
about 19 as a planning figure. The quadratic terms are 30 % of it; the indexer scoring kernel runs at only ~7 % of
WMMA peak (1.6e15 FLOP at 1M [calc] against 232 s), so a better scoring kernel is the first lever; IndexCache in prefill
(not built, research/30 rank 3) is the second. Unmeasured risks: temporary buffers of the layer-major prefill that
scale with context, and the GPU reset risk of very long single kernels (research/23; prefill is chunked, so
each pass stays short).

### 5.2 Decode pass after the prompt

Measured pass times (server log, MTP; measurements/2026-10-29-indexcache, before the #74 scoring work): 128K 51.8 ms
(12 indexers) / 43.0 ms (layer 0 only); 250K 65.0 / 46.5 ms. The indexer cost per layer was 0.8 ms at 128K and
1.7 ms at 250K: linear in the number of blocks (32K / 65K). You report ~47 ms at 250K after #74, i.e. the 12
indexers now cost ~5.7 ms in total (0.48 ms each) instead of ~20 ms. Model: pass(L) = base(L) + 12 x idx(L).
- idx(L) scales with blocks: 4x at 1M -> ~23 ms for 12 layers (0.1 ms per layer is the VRAM bound: pool 134 MB
  per layer at 1M = 0.43 ms at 315 GB/s, times the multi-token pass read once since #74).
- base(L) ~41.3 ms at 250K (47 - 5.7); it grew 42.2 -> 44.8 ms from 128K to 250K before #74 (+2.6 ms per
  122K, cause not isolated: gather spread over pages of a larger pinned region); linear extrapolation +17 ms at 1M,
  flat extrapolation +0.
- Pass at 1M: **41-58 + 23 = ~64-81 ms**. With `IE_IDX_SHARE` pattern (d) (8 of 12 indexers): -8 ms.
- Decode rate ~ tokens per pass / pass time = 2.8 / 0.072 = **~35-45 tok/s** (250K today: 43-46 tok/s measured,
  text; the pass time to rate depends on MTP acceptance, which at 1M is unknown).

Both numbers are extrapolations of two to three points; a run at 524K (`--ctx 524288`, factor 2) would turn them
into interpolations.

---

## 6. Plan to answer the open questions (all offline-measurable, no code change to the reference path)

1. **YaRN cost below the native length.** Add RoPE tables with YaRN parameters behind a flag (default off, bit-identical
   when off), then the quality rule of docs/backends/README.md (teacher-forced NLL on py/cpp/sh, 32K and 128K tails, as in
   measurements/2026-10-29-indexcache): native vs YaRN factor 4 vs factor 2, and with `mscale` 1.0 vs 1.1386.
   Gate: if the 95 % interval of dNLL is below +0.01 nats, "always YaRN" is possible and the switch needs no logic.
2. **Beyond 262K.** Context of 400K / 524K / 1M tokens (concatenated own repositories) with a retrieval test like
   `retr.py` (8 keys over the whole range, single and multi) plus a code-task set; factor 4 for all, factor 2 at 524K.
   Without this there is no quality number for Flash-Next; the Qwen3-Next RULER curve is only a hint.
3. **Dynamic B vs fresh YaRN prefill** at 300K: same text, (a) static YaRN from token 0, (b) native prefill of 262K +
   re-rotation + 40K more, (c) native only. NLL of the last 4,096 tokens and retrieval of keys placed before and
   after the switch.
4. **Memory:** `IE_SEQS=1 --ctx 1048576` start, `IE_VRAM_LOG`, check slots and RAM; a 524K prefill run as the time
   interpolation (about 6-9 min predicted).

## 7. Product option [proposal]

- **Server start:** `--ctx N` stays the capacity (memory planning needs it at start). New `--rope-yarn
  auto|off|<factor>` (env `IE_ROPE_YARN`): `auto` = factor max(1, ctx / 262,144) rounded up in steps of 0.5, applied
  per request only if prompt + max_tokens exceeds 262,144 (mode A); `off` = native, the request fails above 262,144.
  Default: `off` and ctx 262,144 (reference quality). A model whose GGUF carries rope-scaling keys keeps them as the
  default factor.
- **Per request:** API extension is not needed with `auto`; an explicit field would only be useful for evaluation.
  The mode is part of the prefix-cache key; server log and `/v1/models` should show the active mode and factor.
- **Display:** the installer text and README must say that above 262K the quality is not measured by us (until step 2
  of section 6) and that short texts in a server started with a fixed factor lose quality as the cards warn.
- **How others do it:** llama.cpp `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144` (server-wide, static);
  vLLM `--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}'` or
  `--hf-overrides`, SGLang `--json-model-override-args` (server-wide, static). Per-request mode A would be new.
- **Risks:** (1) quality unknown beyond 262K and below it (section 6); (2) every mode change invalidates cached keys;
  (3) two sequence slots impossible at 1M, so a second conversation evicts the first; (4) prefill time of 16-23 min is
  an agent-session stall (the 100 ms yield rule and `IE_MEGA_BUDGET_MS` apply per pass, not per request); (5) desktop
  VRAM headroom shrinks by 1.3 GB; (6) a 4-hour GPU job is a bigger reset exposure than 3 minutes (research/9, 12);
  (7) MTP acceptance and the DeltaNet state beyond 262K are unmeasured for this model.

---

## Appendix: newer methods found on the way (not evaluated; the survey is research/33)

Training-free or cheap, found in this session (abstract level; numbers are the authors'):
- MrRoPE, arXiv 2601.22181 (2026-01-28): radix-conversion view of RoPE extension, two training-free variants; Pro keeps
  over 85 % recall at 128K NIAH and more than double YaRN's accuracy on InfiniteBench retrieval/dialogue; claimed stable
  RULER to 128K where YaRN drops sharply beyond 64K (secondary summary).
- Dual Chunk Attention in Qwen2.5-1M, arXiv 2501.15383, and used for the Qwen3-2507 RULER-1M numbers [card].
- DroPE, arXiv 2512.12167 (ICLR 2026): drop RoPE after pre-training plus short recalibration (0.5-2 % of the
  budget reported); needs training.
- LongRoPE2, arXiv 2502.20082 (ICML 2025): evolutionary search of per-dimension factors + mixed-window fine-tuning, Llama3-8B
  to 128K with over 98.5 % short-context performance retained, ~10B tokens; needs training.
- Hybrid position extrapolation, arXiv 2610.10114 (2026-10-07): section 3.2.
- Position-encoding survey, arXiv 2608.10021 (2026-08): context for the dynamic-scaling and cache conflict.
- How 1M models are made (secondary sources, search): MiniMax-M1 (arXiv 2506.13585, 7 lightning : 1 softmax, 1M, trained),
  UltraLong (arXiv 2504.06214, 1M/2M/4M from Llama-3.1-8B with ~1B tokens of continued pre-training), Llama 4 Scout
  (iRoPE, trained at 256K, 10M claimed, no independent numbers found).

---

## Sources

- Qwen3.8-Flash-Next card: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- Qwen3-Next-80B-A3B-Instruct card (YaRN settings, RULER 1M table): https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- HF modular `qwen4_exp` (indexer rotation): https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modular_qwen4_exp.py
- transformers RoPE utils: https://huggingface.co/docs/transformers/main/en/internal/rope_utils
- YaRN: https://arxiv.org/abs/2309.00071
- vLLM forum, DynamicYaRN and KV reuse: https://discuss.vllm.ai/t/issue-with-dynamicyarn-and-key-value-cache-reuse-in-vllm/631
- llama.cpp YaRN PR #2268: https://github.com/ggml-org/llama.cpp/pull/2268 ; issue #17459: https://github.com/ggml-org/llama.cpp/issues/17459
- Qwen2.5-1M: https://arxiv.org/abs/2501.15383
- LongRoPE2: https://arxiv.org/abs/2502.20082
- MrRoPE: https://arxiv.org/abs/2601.22181
- DroPE: https://arxiv.org/abs/2512.12167
- Mechanics of Long-Context Hybrid Models: https://arxiv.org/abs/2610.10114
- Position Encoding in Transformers (survey): https://arxiv.org/abs/2608.10021
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- MiniMax-M1: https://arxiv.org/abs/2506.13585 ; UltraLong: https://arxiv.org/abs/2504.06214
- Own: research/29, research/30, measurements/2026-10-28-long-context, 2026-10-29-prefill-topk, 2026-10-29-indexcache,
  2026-10-25-slots, 2026-10-26-slot-memory, 2026-10-27-ram-limit
