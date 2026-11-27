# Self-speculative drafting by skipping layers (#155)

Question: can a cheaper pass of the same model (layers skipped, or fewer experts per layer) draft token chains that one
full pass verifies, with identical output, alone or behind the MTP head? Status 2026-11-16: measured offline, closed negative ([plan and
results](measurements/2026-11-15-self-spec-layer-skip.md)): the drafts are good (16 of 48 layers skipped: survival 0.75 over
8 tokens) but a step costs 4 to 6 times the break-even; no engine code.

Method: arXiv pages read through summaries (full text unless marked abstract; tails of some papers unread), trackers of
llama.cpp, vLLM, SGLang searched with `gh` (TensorRT-LLM and MLC: nothing relevant). `tools/papers/scan.py` coverage is
the maintainers' book (#120); this note adds the entries as a block at the end.

## What the literature says

| Work | Draft | Training | Reported gain | Condition |
|---|---|---|---|---|
| LayerSkip 2404.16710 (Meta) | exit at layer E (8 of 32), shared head, verify reuses KV | needs layer dropout + exit loss (52B tokens continual) | 1.34 to 2.16x, acceptance 55 to 78 % | H100/A100, dense Llama |
| Draft&Verify 2309.08168 | fixed attention/MLP skip set (Bayesian opt., 2.5 h for 13B) | none | 1.43 to 1.99x; 56.3 -> 36.8 ms/token (draft 25.5, verify 10.7) | A100, dense Llama-2; fp4 weights 1.36x vs bf16 1.53x |
| SWIFT 2410.06916 | skip set tuned online (r 0.45 to 0.5) | none | 1.3 to 1.6x own; **0.72 to 1.33x** in KnapSpec Table 2, 1.06 to 1.21x in CAS-Spec | 4x A6000, batch 1, dense |
| Kangaroo 2404.18911 | layer 2 to 3 plus a trained 67M adapter | adapter | 1.5x (7B), 1.44x (13B) | V100, Vicuna |
| KnapSpec 2602.20217 | knapsack over attention and MLP blocks, tokens per time, cost model per hardware | none | 1.06 to 1.47x, acceptance 85 to 94 % on 32B/70B; 1.1 to 1.33x on 1B/8B | dense; MoE, hybrid, quantized not tested |
| ConfLayers 2604.14612 (AMD) | confidence-driven skip | none | 1.11 to 1.35x | MI300X, dense |
| SpecBound 2604.12247 | trained heads at intermediate layers | heads | 1.93 to 2.33x | H800, Vicuna/CodeLlama |
| SEED 2609.36590 | last 2 layers as a draft decoder | fine-tuning | 2.6x (1.7B) | merges early-exit with MTP, dense, trained |
| CAS-Spec 2510.26843 | cascade: layer sparsity, early exit, n-gram | adapter optional | 1.48 to 1.70x; plain n-gram lookup already 1.39 to 1.54x | H100, Vicuna |
| Component-Aware Self-Spec 2605.01106 | **Qwen3.5 hybrid**: drop attention, keep DeltaNet | none | acceptance **0.019** (k=4); uniform 33 % skip **0.233**; early exit 50 % **0.000**; top-1 agreement 0.203 | 0.8B dense hybrid, six attention layers; wall-clock below 1 (Python) |
| SpecMoE 2604.10152 | **same model, MoE restricted to N resident hot experts**, no layer skip | none | up to 4.30x over on-demand offload; 7.2 tokens per step at gamma 10, batch 256; **at batch 1 plain expert caching is faster** | H100 + host DRAM; 2 to 128 experts, top-2 |
| S2-MoE 2608.15018 | reduced top-k plus reuse-aware gating (llama.cpp) | none | 1.2 to 5.3x on edge, but **not lossless** (top-1 agreement 89 to 93 % with the unmodified target); naive layer-sparsity drafts 0.23x, expert-sparsity 0.74x of autoregressive in places | Jetson, 4090 with offload |
| ELMoE-3D 2604.14626 | hot-expert subset plus 4-bit MSB slice of INT8 | none | alpha 0.66 to 0.91, 4.6x (simulated hardware) | INT8 targets; bit slicing needs bits to slice |
| Cassandra 2605.26558 | pruned weights, 4-bit mantissas, new hardware | none | 1.78 to 2.41x simulated | bf16 targets; no layer skip |
| Packed-SS 2610.33379 (abstract) | diffusion-mode drafter in the same call | model needs both modes | no numbers | not applicable |
| Limits of Speculation 2609.22156 | EAGLE-3 on Qwen3-Coder-30B-A3B | | verify cost grows with unique experts: 12.1 per layer (K=1) to 30.2 (K=10), per-expert cost 3.2x higher from 5 tokens | A100, batch 1: long chains are expensive on MoE |

Independent reading of the same numbers: training-free layer skipping gives 1.05 to 1.6x on dense Transformers, with
the unfavourable cells below 1. The larger gains (1.8 to 2.6x) all need training.

## What the trackers say

- llama.cpp [#3565](https://github.com/ggml-org/llama.cpp/pull/3565), open since 2023, demo only: perplexity per skipped
  layer, no acceptance or speed. Skipping whole layers breaks beyond 10 to 20 % of the layers (Mistral 7B: 3 skipped
  layers +1.4 ppl, 8 skipped +9.2 ppl); skipping attention alone hurts less than skipping MLP.
- llama.cpp [#10090](https://github.com/ggml-org/llama.cpp/issues/10090), [#10820](https://github.com/ggml-org/llama.cpp/issues/10820):
  closed stale, no data.
- vLLM [#18994](https://github.com/vllm-project/vllm/pull/18994) (LayerSkip): closed unmerged after eight months; its own
  docs: "greedy typically yields very low acceptance rates", eager mode only, 15 to 20 % slower than other methods; no
  numbers shipped. SGLang [#25745](https://github.com/sgl-project/sglang/pull/25745): per-request exit layer
  infrastructure, closed stale, no drafting.
- vLLM [#64556](https://github.com/vllm-project/vllm/pull/64556) (open): static layer skip and reduced top-k on a hybrid
  Mamba+MoE (Nemotron-H); +8 % decode at C=1 for 7 skipped layers, +3 % for top-k 4, +12 % both; **lossy**, serving
  not drafting. It shows the draft-side cost of a hybrid MoE falls roughly with the layers and experts removed.
- Qwen3-Next prototype (angerybob/S2-MoE PR #2): expert-only self-drafting; hybrid recurrent state needs a **snapshot of
  conv and DeltaNet state per prefix** to undo draft writes; greedy tokens equal on 9 of 9 sequences, no speed number.
  ik_llama.cpp keeps 149 MB of GPU state per step for the same reason and reports a net loss on a dense 27B.
- No tracker item and no paper reports layer-skip drafting on a MoE with 256 experts, a hybrid DeltaNet model, a 1 to
  3 bit quantized model, or combined with an MTP head. That is the gap this ticket would fill, and also a warning: the
  closest data (Component-Aware, S2-MoE) is negative for naive skipping.

## What we take, and what we do differently

| Idea | Taken | Ours, and why |
|---|---|---|
| SWIFT/Draft&Verify: skip attention/MLP sublayers | the skip mask as the draft | Online Bayesian search costs hours or steals decode time and was below 1.0x in two independent tables. We pick the mask offline once, per model, from teacher-forced agreement with the full model on our own bench-set (the set the drafter is tuned on), and ship it as data like the MTP vocabulary. |
| KnapSpec: knapsack on tokens per time | the objective | Their cost model is attention-vs-MLP time. Ours is bytes per layer from the GGUF (`tools/layer_cost.py`: dense, per-expert, head), because decode here is bandwidth-bound and quantization differs per layer (layer 1: 156 MB, others 82 to 103 MB). The skipped layers also skip their expert reads, which dominates on the offloaded classes. |
| Confidence early stop (SWIFT, KnapSpec, vLLM #63030) | stop the chain on low top-1 probability | Calibrated like the MTP chain (`mcal`), not a fixed threshold: the engine already chooses depth by expected tokens per ms (`chain_push`). |
| SpecMoE/S2-MoE: restrict experts in the draft | draft variant `topk=n` | Both are batched or lossy. We keep the target pass untouched (their S2-MoE gating changes the target too) and apply the restriction to the draft only, so the output is identical by construction. |
| MTP + layer-skip chain (SEED idea) | MTP drafts first, a skip draft extends | SEED trains a 2-layer decoder. We train nothing: the extension is the model itself with a mask, and only the survival after the MTP chain's end matters, which the offline run measures directly. |
| LayerSkip shared KV, exit query cache | no | needs training; and DeltaNet state cannot be rebuilt from a query cache. Draft writes go to scratch state that the verify pass recomputes (snapshot per prefix, as the Qwen3-Next prototype does). |
| Component-Aware: do not drop attention in a hybrid | the warning | Our QSA layers (every fourth) are kept in the first candidates; masks that drop DeltaNet layers instead are tested too (the reverse of their draft). |

## Cost arithmetic before any measurement

Per layer a decode token reads 81 to 156 MB (mean ~94 MB, 1.85 % of 5.1 GB); the head and final hyper-connection add
0.53 GB (10 %) however many layers run (`tools/layer_cost.py`). The engine's measured defaults (`generator.h`): one pass
23.7 ms, verify over 2..8 positions 28.9 to 52.6 ms, MTP step 1.15 ms, fill 1.3 ms. Tokens per verify pass today: code 4.7,
text 2.8 (BENCHMARKS.md, MTP-256 with scorer). Time per generated token is therefore about 45 ms / 4.7 = 9.6 ms (code).

A layer-skip step with half the layers kept reads about 55 % of a full token (2.8 of 5.1 GB), ~13 ms at 23.7 ms per
full pass, **more than the 9.6 ms the engine already spends per token**. A drafted token can add at most one expected
token, so a step must cost less than the current time per token times its survival: with the verify increment (~4.5 ms)
a step breaks even only if survival exceeds ~1.8, impossible. The same holds on the offloaded classes, where both the
draft step and the verify pass are expert-bound: the verify pass amortizes the expert reads over 4 to 5 tokens, a draft
step does not. The idea can only pay when the draft step costs well under the time per token: roughly at most 20 % of a
full token (about 8 layers plus a cut head) with survival of 0.4 or more beyond the MTP chain, or an extension that
runs for many tokens per step (not a sequential chain). The offline run (2026-11-16) tested whether any mask comes near that: none does. Contrary to the expectation (S_4 below 0.4)
the draft quality was high; the cost, not the acceptance, closes the idea.

## Rows proposed for research/papers.md (not edited here)

New to the book, abstract or full text as marked: SEED 2609.36590 (full), SpecBound 2604.12247 (full), ConfLayers
2604.14612 (full), ELMoE-3D 2604.14626 (full), Limits of Speculation 2609.22156 (full), ECHO 2609.17241 (title), DEL
2504.05598 (baseline numbers only); llama.cpp #3565, #10090, #10820, vLLM #18994 (closed), #64556 (open), SGLang #25745
(closed), angerybob/S2-MoE PR #2 (experimental). Already in the book: KnapSpec, SpecMoE, Cassandra, Packed-SS,
S2-MoE, Component-Aware Self-Spec, CAS-Spec, SPEQ.

Scan (2026-11-15): `scan.py arxiv --from 2025-11-15 --to 2026-11-14 --topic speculative --no-book`, 651 hits, complete
except for 2026-11-14 (younger than 3 days), 3 not in papers.md (none relevant). Title/abstract filter for layer skip,
early exit, self-speculation: 50 hits, all already in the book. Three read as abstracts for this note:

- DraftExpert 2607.24434: expert-offloaded MoE (DeepSeek-V2-Lite, Moonlight-16B), draft = shared expert + top-1 + one
  **trained** lightweight draft expert per layer, self-distilled; acceptance 84 to 87 %, 1.45x on average, tokens verified
  exactly. Says what we found: a bigger draft expert set triggers extra loads, and the verify pass touches the union of
  target experts. Relevant as the trained variant of the expert-reduced draft; needs training, 64 experts or fewer.
- FlexEE 2609.17008: early exit for offloaded inference with layer-wise exit supervision and a top-K local vocabulary for
  the exit decision; 1.27x without and 3.16x with 50 % weight offload (Llama2-7B). Trained; the gain grows with
  offloading, the one report that matches our offloaded classes. The local vocabulary matches our MTP draft vocabulary.
- Audit 2608.28846: three-seed audit on Qwen2.5-0.5B/1.5B: SWIFT's search overhead up to 28.7 % of cost; routing-based
  skips 1.08 to 1.33x with much lower accuracy. Supports picking the mask offline.

Search: tools/papers/scan.py topics v1; topic: speculative (plus by-hand `gh search issues/prs` for layer skip, early
exit, self-speculative in llama.cpp, vLLM, SGLang); window 2025-11-15..2026-11-14 (12 months) plus the named older papers;
sources: arXiv, GitHub PRs and issues, research/papers.md.
