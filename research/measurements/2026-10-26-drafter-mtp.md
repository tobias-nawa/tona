# 2026-10-26: drafter comparison DFlash2 against MTP

Own measurement. The hit rate depends on the models, not on the engine. Therefore with llama.cpp b11822 (Vulkan,
because HIP can hang when loading Qwen3.8, llama.cpp#29149), target `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M` (experts on the CPU,
`--n-cpu-moe 48`), `scripts/bench-set.py` (11 code, 3 text prompts, greedy, up to 400 tokens each). Tokens per verify pass
= tokens / (tokens − accepted drafts).

| Drafter | Drafts | Code tokens per pass | Text tokens per pass |
|---|---:|---:|---:|
| DFlash2 Q8_0 (0.65 GB) | 3 | 3.30 (2220/2854) | 2.53 (647/1260) |
| MTP Q8_0 (4.14 GB) | 3 | **3.47** (2266/2717) | **2.67** (669/1194) |
| DFlash2 | 7 | 5.06 (2633/4442) | 2.88 (637/2349) |
| MTP | 7 | **5.68** (2660/3874) | **3.11** (679/2229) |

Against the model card (DFlash2 2.94 versus MTP 2.52 on HumanEval), MTP hits better in our measurement: +5 % at 3, +12 % at 7
drafts (code). One run each, greedy, so no scatter between runs; averaged over 14 prompts.

## Cost per MTP step (tensors of the GGUF)

MTP is a full layer (QSA, 512 experts, top-10, shared expert, hyper-connections) with its own KV and LM head.
Per step in Q8_0: dense ~102 MB, 10 experts ~52 MB, LM head 248,320 × 2560 = 675 MB. Together ~0.83 GB, at
315 GB/s ~2.6 ms. The steps run one after another. All 512 experts in Q8_0 are 2.67 GB.

## Projection (code, 7 drafts, verify pass with 8 tokens 59.4 ms from `--batch-bench`)

| | Drafter per pass | Tokens per pass | ms per token |
|---|---:|---:|---:|
| DFlash2 (measured 6.3 ms per block 8) | 6.3 ms | 5.06 | 12.98 |
| MTP, LM head full in Q8_0 | ~20 ms | 5.68 | ~14.0 |
| MTP, LM head cut to 106K tokens and 4 bit, experts 4 bit | ~12.6 ms (~1.8 per step) | 5.68 | ~12.7 |

Even in the favorable case ~2 % remain. Against that stand 0.9–1.4 GB VRAM for MTP (experts in 2–4 bit, head, KV), i.e.
9–14 expert slots fewer (~0.14 % decode per slot, measurements/2026-10-25-slots), and several days of integration (own
QSA layer with KV and indexer, MoE with 512 experts, sequential steps in the megakernel). Result: DFlash2 stays.

Side finding: MTP hits better on long drafts because every step sees the previous draft token. That speaks
rather for a second, cheap drafting path for long hits (suffix matching over prompt and history, Baseten +34 %
acceptance on code edits, research/11-suffix-decoding.md) than for MTP.

## Comparison: our engine against llama.cpp with DFlash2

ie-server with `IE_DRAFT_FULL=1 IE_DRAFT_BIG_ALWAYS=1 IE_SUFFIX_MIN=0` (always 7 drafts) and `IE_DRAFT_N=3`, same
14 prompts, tokens per pass from the server log:

| | llama.cpp | ie-server |
|---|---:|---:|
| code, 3 drafts | 3.30 | 3.36 |
| code, 7 drafts | 5.06 | 5.26 |
| text, 3 drafts | 2.53 | 2.53 |
| text, 7 drafts | 2.88 | 2.88 |

So llama.cpp is suitable as a measuring instrument for the hit rate. Addendum: with a shortened draft LM head
(llama.cpp#31031, 32K rows, ~87 % fewer head bytes) an MTP step falls to ~0.25 GB. Projected
59.4 + 7 × ~1 ms = ~66.4 ms per pass at 5.68 tokens, i.e. ~11.7 ms per token against 12.98 (+11 %). Decision
(2026-10-26): build in MTP.
