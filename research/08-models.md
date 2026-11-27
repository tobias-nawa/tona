# Models besides Qwen3.8-Flash-Next

Agent research, as of 2026-10-23. None of it measured by us. Benchmark values are, unless stated
otherwise, **vendor claims** with each vendor's own harness. They are only roughly comparable. tok/s and
bytes per token are **estimates** by the agent, derived from `config.json` or file sizes. Calculated
with 315 GB/s (measurements/2026-10-22-micro). Complements `02-engines-compared.md` ("Smaller models with more headroom").
Dates without a year are 2026 (MM-DD).

Newer survey (2026-11-14): [49-moe-models-2026-11.md](49-moe-models-2026-11.md) (MoE releases since 10-01, fit per memory class, rumors).

Question: Are there other current models that could be interesting for us?

## Summary

1. **Qwen itself has not released a new language model for coding since 08-24.** Newest Qwen weights on HF:
   Flash-Next (08-24), 27B (08-05), 2.4T-A95B (08-08) ([HF API Qwen](https://huggingface.co/Qwen)).
   There is no official coder finetune of Flash-Next.
2. **The cheapest gain lies in derivatives of our own model.**
   - ISTA rebuilt the coder on 10-17: SWE-bench Verified 75.6 → 79.2.
   - Gyro-Coder (agentionai, 10-21) keeps 256 experts at ~2.1 bpw: 19.9 GiB instead of 29.6 GB resident.
   - Swift 1.5 and Signal 3.8 are finetunes with the same architecture. They think 26–56% shorter.
3. **Qwen3.8-27B (dense, Apache-2.0) is almost level on SWE-bench Pro** (61.7 versus 62.5) and fits
   entirely in VRAM as IQ3 (10.1 GB). It uses the same DeltaNet shapes as we do. At 256K context,
   however, it fails on full attention: the KV cache does not fit in VRAM.
4. **The strongest small MoE is Ornith-1.5-35B-A3B (MIT, 08-18).** Architecture Qwen3.5/3.6 MoE, i.e.
   DeltaNet + gated attention + MoE. Per the vendor SWE-bench Verified 79, Terminal-Bench 2.1 67.8. On
   long agent tasks it stays clearly behind Flash-Next: DeepSWE 22 versus 58.7, Toolathlon 48.7
   versus 73.5.
5. Larger novelties are above our budget of ~60 GB: GLM-5.3-Flash 320B, DeepSeek V4.1 Flash, MiniMax
   M3.1-Flash, Step 3.7 Flash. Laguna S 2.1 (118B-A8B) would be barely feasible. But only Q4_K_M with
   96 GB exists for it, and the KV cache is large.

## Sources

| Key | Source |
|---|---|
| QFN | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), model card |
| ISTA-C | [ISTA-DASLab/…-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF), section update 10-17 |
| ISTA-R | [r/LocalLLaMA post by the ISTA authors, 09-29](https://prismix.dev/news/5915fe52d0cc) (mirror) |
| OCDC | [openclawdc.com, 09-20](https://openclawdc.com/blog/qwen-3-8-flash-next-2-bit-96gb-ram/), sizes of the ISTA files |
| GYRO | [agentionai/…-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) |
| GYRO-C | [agentionai/…-Gyro-Coder-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF) |
| SWIFT | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF), [GSQ-RCO version](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| SIGNAL | [agentionai/Signal-3.8-Flash-Next-GGUF](https://huggingface.co/agentionai/Signal-3.8-Flash-Next-GGUF) |
| REAP288 | [sh0wie/Qwen3.8-Flash-Next-REAP-288-GGUF](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-GGUF) |
| ROCMFP4 | [agentionai/…-ROCmFP4-FAST-imatrix-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF) |
| Q27 | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), model card and [config.json](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json) |
| Q27-ISTA | [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) |
| Q27-US | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), file list via the HF API |
| Q27-DF | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) |
| QUESMA | [Quesma blog, 08-26](https://quesma.com/blog/qwen38-27b-quantizations-benchmarked/), Terminal-Bench 2.1 per quant |
| ORN | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), model card, config.json, [GGUF file list](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) |
| LAG-S | [poolside/Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1), model card, config.json, [GGUF](https://huggingface.co/poolside/Laguna-S-2.1-GGUF) |
| LAG-XS | [poolside blog Laguna XS 2.1](https://poolside.ai/blog/introducing-laguna-xs-2-1), config.json [Laguna-XS-2.1](https://huggingface.co/poolside/Laguna-XS-2.1) |
| LING | [inclusionAI/Ling-3.0-flash-FP8](https://huggingface.co/inclusionAI/Ling-3.0-flash-FP8), config.json [Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash); numbers from [benchlm.ai](https://benchlm.ai/md/compare/ling-3-0-flash-vs-step-3-7-flash.md) (aggregator) |
| MUSE | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) |
| NMC | [noqta.tn on North Mini Code](https://noqta.tn/en/news/cohere-north-mini-code-open-source-agentic-coding-model-2026) (secondary source), HF API [CohereLabs/North-Mini-Code-1.0](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) |
| LCS | [meituan-longcat/LongCat-Flash-Lite-Sparse](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite-Sparse) |
| NEMO | [Artificial Analysis on Nemotron 3.5 Lightning, 08-11](https://artificialanalysis.ai/articles/nemotron-3-5-lightning-launch) |
| GLM | [Transformers docs GLM-5.3-Flash](https://huggingface.co/docs/transformers/en/model_doc/glm5_next), [Unsloth GLM-5.3-Flash](https://unsloth.ai/docs/models/glm-5.3-flash) |
| STEP | [stepfun-ai/Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) |
| DS | [DeepSeek V4 Flash locally](https://computingforgeeks.com/run-deepseek-v4-flash-locally/), [V4.1 Flash Skeleton](https://huggingface.co/agentionai/DeepSeek-V4.1-Flash-Skeleton-GGUF), [V4.1 announcement](https://thenextweb.com/news/deepseek-v4-1-flash-launch-v4-pro-retired-price-cut) |
| MMX | config.json [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) |
| HFAPI | `huggingface.co/api/models?author=…`, retrieved 10-23: creation date, license tag, parameter count from safetensors |

## Overview

"P/s" = forward passes per second as a ceiling: bytes per pass divided by 315 GB/s, everything assumed in VRAM.
This is an **estimate**, neither a measurement nor a forecast. Our engine currently reaches
39–58 tok/s decode on Flash-Next with DFlash2.

| Model | Date | License | total / active | Architecture | File for us | Bytes/pass (est.) | Ceiling P/s | SWE-V | SWE-Pro | TB 2.1 | Effort |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| **Flash-Next ISTA coder (today)** | 09-29, rebuilt 10-17 | qwen-community | 125B+51B n-gram / 6B, 256 exp. | DeltaNet + QSA, MoE 10/256 | 58.4 GB (29.6 resident) | 5.1 GB (06) | ~62 | 79.2 (ISTA-C) | – | – | – |
| Flash-Next BF16 (reference) | 08-24 | qwen-community | as above, 512 exp. | | | | | 82.8 (ISTA-C) | 62.5 (QFN) | 67.6 (SWIFT) | |
| Gyro-Coder | 10-21 | qwen-community | 256 exp., ~2.125 bpw | same, trellis quant (QTIP) | 19.9 GiB resident + 50.7 GiB n-gram | ~4.7 GB | ~67 | – | – | – | M (trellis kernel) |
| Swift 1.5 (finetune) | 09-24 (HFAPI) | Swift Open License + qwen | 512 exp. | same | 66.6–68.2 GB (GSQ-RCO Q2_0/IQ2_XS) | ~4.1 GB | ~77 | – | – | 69.7 (SWIFT) | S |
| Signal 3.8 (finetune) | upd. 10-12 (HFAPI) | qwen-community | 512 exp. | same | from 84 GiB (too large) | – | – | – | – | – | S + own quant |
| **Qwen3.8-27B** | ~08-13 (repo 08-05) | Apache-2.0 | 27.8B dense | 48 DeltaNet + 16 gated attn (GQA 24/4), MTP | 10.1 GB (ISTA IQ3_XXS) | ~9.6 GB + KV | ~33 (short) | – | 61.7 (Q27) | 73.0 (Q27) | M |
| **Ornith-1.5-35B-A3B** | 08-18 | MIT | 36B / 3B | Qwen3.5 MoE: 30 DeltaNet + 10 gated attn, MoE 8/256, MTP | 21.7 GB Q4_K_M | ~1.8 GB + KV | ~180 | 79 (ORN) | 59.6 (ORN) | 67.8 (ORN) | M |
| Qwen3.6-35B-A3B | 04-15 | Apache-2.0 | 36B / 3B | like Ornith | ~21 GB Q4 | ~1.8 GB | ~180 | 73.4 (ORN) | 49.5 (ORN) | 52.5 (ORN) | M |
| Laguna S 2.1 | 07-13 | OpenMDW-1.1 | 118B / ~8B | 36 SWA-512 + 12 GQA (8 KV), MoE 10/256 | only Q4_K_M 96 GB, quantize ourselves | ~3.3 GB (3 bpw) | ~95 | – | 59.4 (LAG-S) | 70.2 (LAG-S) | L |
| Ling-3.0-flash | 08-02 (HFAPI) | MIT | 127B / 5.1B | 35 KDA + 7 MLA, MoE 8/512 | community GGUFs, fork | ~2.5 GB (2.5–3 bpw) | ~125 | – | 56.6 (LING) | 50.2 (LING) | L |
| Muse Glimmer-30B | 08-09 | Apache-2.0 | 29.6B dense | SWA-2048 : global = 3 : 1, GQA 32/2 | ~11 GB IQ3 (est.) | ~10.8 GB | ~29 | 76.0 (MUSE) | 51.2 (MUSE) | 51.7 (MUSE) | M–L |
| North Mini Code 1.0 | 06-05 | Apache-2.0 | 30B / 3B | MoE 8/128, details not checked | – | – | – | 67.6 (NMC) | 40.2 (NMC) | TB2: 36 | L |
| LongCat-Flash-Lite-Sparse | 07-31 | MIT | 69B / 3B | MLA + DSA indexer (top-k 2048), n-gram embedding | – | – | – | 68.2 (LCS) | 40.6 (LCS) | TB2.0: 33.7 | L |
| Nemotron 3.5 Lightning | 08-11 | OpenMDW-1.1 | 31.6B / 3.6B | Mamba-Transformer hybrid | ~20 GB Q4 | – | – | – | – | 24 (NEMO) | L |

TB = Terminal-Bench. Ornith and Qwen3.6 measure with Terminus-2, Qwen3.8-27B with Terminus, Swift
measures itself with its own setup.

## 1. Derivatives of Qwen3.8-Flash-Next (same architecture)

### ISTA coder, rebuild of 10-17

- The first shard was replaced (ISTA-C). Per the card there are fewer loops. SWE-bench Verified rises
  from 75.60 to **79.20**, LiveCodeBench v6 from 86.28 to 87.43. BF16 base: 82.80 and 87.43, measured
  at xhigh. That retains 95.7% of the base.
- This matches README item 14: "first shard replaced on HF on 10-19". **Check whether our local
  GGUF is the rebuild.** If not, it costs only a 29.6 GB download, zero code.

### Gyro-Coder (agentionai)

- 256 of 512 experts per layer, selected by evolutionary search over ~2,200 candidate sets.
  Trellis quantization with Hadamard rotation (QTIP family), **~2.125 bpw** (GYRO-C).
- **19.92 GiB resident** (21.1 GiB with 64K context), plus the n-gram table at 50.7 GiB on disk.
  For comparison ISTA: 28.8 GB.
- Quality only as KLD/top-1 against Q8_0, Python 0.165 / 93.1%. Own task collection: 13/16 versus 15/16
  at 8 bit. **No SWE-bench**, no comparison with the ISTA coder.
- The sibling models Gyro-S/M/L (GYRO) beat ISTA at equal memory in KLD. Gyro-M: 34.98 GiB,
  KLD 0.291. ISTA IQ3_XXS: 43.80 GiB, KLD 0.311.
- Measured by agentionai on RDNA4, namely R9700 with Vulkan: Gyro-S decode 57.8 tok/s, with MTP on code
  70.6, prefill 1,246 tok/s (GYRO).
- **For us:** One expert shrinks from 2.046 to ~1.31 MB. Calculated as 4.915 M parameters ×
  2.125 bit, an **estimate**. With today's VRAM budget for experts (100 slots × 48 layers ×
  2.046 MB ≈ 9.8 GB), **~157 instead of 100 slots per layer** then fit, i.e. 61% instead of 39%. Per llama.cpp #29887
  (33% → 85%, 67% → 94% hits) that would be roughly 92% instead of today's 81–84%. Expert bytes per pass
  drop from 0.98 to ~0.63 GB.
- **Drawbacks:**
  - new format: trellis decoding, own kernels needed
  - "experimental, format may change"
  - runs only in the agentionai fork of llama.cpp, upstream does not load it
  - no agent benchmarks
  - n-gram table 76% larger

### Swift 1.5 (UkisAI) and Signal 3.8 (agentionai): finetunes for shorter thinking

- **Swift 1.5** (SWIFT), measured against the BF16 base with its own setup:
  - LiveCodeBench v6 90.39 versus 88.40, thinking tokens −44.8%
  - Terminal-Bench 2.1 69.66 versus 67.64: total output on average +11.9%, median −17.9%
  - GPQA: −55.8% tokens at −0.2 points
  - **No SWE-bench, no pruned coder version.** GSQ-RCO quants with 512 experts: 66.55 GB (Q2_0)
    to 83.74 GB (IQ3_S).
  - License: free up to USD 1 million annual revenue.
- **Signal 3.8** (SIGNAL): self-distillation without external data.
  - −26% reasoning tokens
  - **MTP acceptance 90% versus 69%**
  - Terminal-Bench 2.0: 10/10 versus 8/10, but only 10 tasks with one run each
  - Only Q4/Q8 files from 84 GiB. We would have to prune and quantize ourselves.
- **For us:** zero kernel work, the tensors are the same. Shorter thinking saves real wall time for
  agents. Higher MTP/draft acceptance acts directly on tokens per pass. Our current draft acceptance
  on prose is 2.38 tokens/pass.
- **Drawback:** With 512 experts only ~27% fit in the cache at the same VRAM: 9.8 GB / (48 × ~1.4 MB)
  ≈ 140 slots of 512, **estimate**. More misses, as long as nobody builds a coder prune of these.
  Rebuilding the expert selection with RCO ourselves is open.

### Other derivatives, less interesting

- **REAP-288** (REAP288): 288 of 512 experts, calibrated on agentic coding traffic. Only Q4_K_M from 78 GB.
  HumanEval 91.5 versus 93.9.
- **ROCmFP4-FAST** (ROCMFP4): 4.23 bpw, 87 GiB. Measured only on Strix Halo, needs the agentionai fork.
  Format undocumented. Too large for 16 GB + 64 GB.
- **Qwen3.8-Omni** ([arXiv 2609.25611](https://arxiv.org/abs/2609.25611), 09-22): same MoE base,
  1M context. Open weights not found.

## 2. Qwen3.8-27B: dense, entirely in VRAM

- **Architecture** (Q27 config.json): `qwen3_5`, 64 layers = 16 × (3 DeltaNet + 1 gated attention),
  hidden 5120, FFN 17,408.
  - DeltaNet 16 QK / 48 V heads × 128: **exactly our DeltaNet shapes**.
  - Attention 24 Q / 4 KV × 256, RoPE 64 dim., output gate: like QSA, but without indexer and with 4 instead of 2
    KV heads.
  - 1 MTP layer, context 262,144. No gated residual, no n-gram table, no MoE.
- **Benchmarks** (Q27, vendor, compared with Flash-Next from QFN):

  | | 27B | Flash-Next |
  |---|---:|---:|
  | SWE-bench Pro | 61.7 | 62.5 |
  | Multilingual | 73.8 | 81.0 |
  | DeepSWE 1.1 | 42.2 | 58.7 |
  | Toolathlon | 67.1 | 73.5 |
  | LiveCodeBench v6 | 90.3 | 91.9 |
  | Terminal-Bench 2.1 | 73.0 | not given on the card |

  SWE-bench Verified is not published. Note: our Flash-Next is the pruned 1.89-bpw coder.
  The gap to the 27B in IQ3 should therefore be smaller than the table shows (**not proven**).
- **Quants:**
  - ISTA GSQ-RCO: IQ2_XS 8.4 GB, IQ3_XXS 10.1 GB, IQ3_S 11.8 GB; MTP +0.35 GB (Q27-ISTA)
  - IQ3_S "task-lossless" per ISTA on AIME, GPQA and LCB. Agent tasks were not checked.
  - Quesma (QUESMA): Q4_K_M = BF16 on Terminal-Bench 2.1, UD-Q2_K_XL "noticeable drop". IQ3 not tested.
- **Drafter:** DFlash2 for the 27B (Q27-DF), Q4_K_M 1.1 GB, acceptance length 5.3 on GSM8K. A
  coding value is missing. Our engine already uses DFlash2.
- **Memory, estimate by the agent:**
  - KV per token: 16 layers × 4 × 256 × 2 = 32,768 values = 64 KB in fp16, 32 KB in int8
  - VRAM: 14 GB − 10.1 (weights) − 0.35 (MTP) − 1.1 (drafter) ≈ 2.5 GB for KV and state
  - That is enough for ~36K tokens in fp16 or ~75K in int8. 256K would need 8.4 GB in int8 and would then sit in
    RAM. Every pass would read it there at 58 GB/s: ~145 ms, i.e. below 7 P/s.
  - **Full attention at 256K is the crux.** QSA solves exactly that for Flash-Next.
- **Decode, estimate:**
  - ~9.6 GB per pass → ceiling ~33 P/s at short context, ~27 P/s at 32K (fp16 KV)
  - With DFlash2 at 3–4 tokens per pass, ~80–120 tok/s decode as a ceiling
  - At the megakernel's current bandwidth utilization (check pass 35.8 ms for 5.1 GB ≈ 45%), rather ~40–55 tok/s decode
- **Prefill:** ~52 GFLOP per token versus ~13 for Flash-Next (06, section 11), i.e. about 4× more expensive.
  A clear drawback for agents with long prompts.
- **Effort: M.**
  - reusable: DeltaNet kernels, the QSA attention core without selection, DFlash2 path, tokenizer (same
    vocabulary 248,320)
  - new: dense FFN matvecs at width 5120, flash decoding over the full KV, pre-norm instead of GR
  - Upstream llama.cpp knows `qwen35` as a reference.
- **Variants:** Swift-1.5-Qwen3.8-27B-GSQ-RCO (HFAPI, 09-24) with shorter thinking, analogous to Swift above.

## 3. Ornith-1.5-35B-A3B and Qwen3.6-35B-A3B: small MoE with our building blocks

- **Architecture** (ORN config.json): `qwen3_5_moe`, 40 layers (30 DeltaNet + 10 gated attention),
  hidden 2048.
  - DeltaNet 16 QK / 32 V × 128, attention 16 Q / 2 KV × 256
  - MoE 8 of 256 + 1 shared, intermediate 512; 1 MTP layer, 262K context
  - Parameter count identical to Qwen3.6-35B-A3B (35,951,822,704, HFAPI), so a finetune of it. The card
    says: built on Qwen3.5 and Gemma 4.
- **Benchmarks** (ORN, vendor, mean of 5 runs):

  | | Ornith-1.5 | Qwen3.6-35B-A3B | Flash-Next |
  |---|---:|---:|---:|
  | SWE-bench Verified | 79 | 73.4 | |
  | SWE-bench Pro | 59.6 | 49.5 | |
  | Terminal-Bench 2.1 (Terminus-2) | 67.8 | | |
  | Terminal-Bench 2.1 (Claude Code) | 68.5 | | |
  | SWE-bench Multilingual | 71.4 | | 81.0 |
  | DeepSWE | 22 | | 58.7 |
  | Toolathlon Verified | 48.7 | | 73.5 |

  Not independently confirmed. Ornith AI is a new vendor.
- **Bytes, estimate from the config:** active 2.94 B parameters, of which 1.0 B routed experts.
  - Q4_K_M (21.7 GB): ~1.8 GB per pass → ceiling ~180 P/s
  - Experts total ~19 GB in Q4: About half fit in VRAM, ~90% hits per llama.cpp #29887.
  - KV: 10 layers × 2 × 256 × 2 = 20 KB per token in fp16. **256K = 5.2 GB in fp16, 2.6 GB in int8**,
    fits in VRAM. At 256K, ~8 ms per pass are added.
- **Prefill:** ~6 GFLOP/token, half as much as Flash-Next.
- **Effort: M.**
  - same building blocks as ours: DeltaNet, gated attention with 256-wide heads, MoE with offload, MTP
  - different widths: hidden 2048, V heads 32 instead of 48, experts 512 instead of 640 wide
  - dropped: n-gram table, gated residual, indexer
  - Upstream llama.cpp loads the GGUFs (`llama-server -hf ornith-ai/Ornith-1.5-35B-A3B-GGUF`, ORN). That
    gives us a reference.
- **For us:** several slots in parallel, i.e. several workers. Very high decode. 256K without a RAM path.
- **Drawbacks:** clearly weaker on long tasks (DeepSWE, Toolathlon). Benchmarks from the vendor only.

## 4. Larger MoE that might barely fit in 16 + 64 GB

### Laguna S 2.1 (poolside)

- **Key data** (LAG-S): 118B-A8B, OpenMDW-1.1, 1M context
  - 36 sliding-window layers (512) + 12 global GQA (8 KV × 128)
  - MoE 10 of 256 + shared, intermediate 1024, hidden 3072
- **Benchmarks** (vendor):
  - Terminal-Bench 2.1 70.2, SWE-bench Pro 59.4, Multilingual 78.5, DeepSWE 40.4, Toolathlon 49.7
  - Thus close to Flash-Next on terminal and SWE, weaker on DeepSWE and Toolathlon
  - DFlash drafter available
- **Hurdles:**
  - Only Q4_K_M (96 GB) and Q8_0 officially. For ≤ 60 GB we would have to go to ~2.5–3 bpw ourselves
    (37–44 GB, **estimate**).
  - KV of the global layers: 12 × 8 × 128 × 2 = 48 KB per token in fp16. 256K would be 12.6 GB, in int8
    6.3 GB.
  - Upstream llama.cpp: support still in review (#25165), poolside fork needed.
- **Bytes, estimate:** ~3.3 GB per pass at 3 bpw. Experts, however, fit in VRAM only to ~20%. The
  RAM path (58 GB/s) then limits to rather ~50–60 P/s.
- **Effort: L.** SWA attention is new, as is softplus gating. No DeltaNet, so our strongest
  kernels go unused.

### Ling-3.0-flash (inclusionAI)

- **Key data** (LING): 127B / 5.1B, MIT
  - 35 KDA layers (Kimi Delta Attention, a DeltaNet relative with channel-wise gate) + 7 MLA
  - MoE 8 of 512 with intermediate 768, hidden 2560 (like Flash-Next); MTP
- **Benchmarks:** SWE-bench Pro 56.6, Terminal-Bench 2.1 50.2, BFCL v4 73.0. The numbers come from an
  aggregator because the card shows them only as an image. **Not checked directly.**
- **Runnability:** llama.cpp support (`bailing_hybrid`) inconsistent. Community GGUFs name forks.
- **Effort: L.** KDA can be derived from DeltaNet, MLA is new. Clearly weaker than Flash-Next, so
  not worthwhile.
- **Successor:** Ling-3.1-flash (10-13) has 572B parameters (HFAPI). Too large for us.

## 5. Small models for simple worker tasks (entirely in 14 GB)

| Model | VRAM file | KV per token (fp16) | Assessment |
|---|---|---:|---|
| **Qwen3.8-27B** IQ3_XXS / IQ3_S | 10.1 / 11.8 GB (Q27-ISTA) | 64 KB | strongest small model; context practically ≤ 64–75K in VRAM |
| Ornith-1.5 or Qwen3.6-35B-A3B, ~3 bpw | ~13.5 GB (est.) | 20 KB | tight, rather with experts partly in RAM (section 3) |
| Muse Glimmer-30B | ~11 GB IQ3 (est.); Meta: "K-quant 17GB" for 24 GB | 13 KB (13 global layers × 2 × 128) | SWE-V 76, Pro 51.2, TB 2.1 51.7 (MUSE); only 131K context; DFlash drafter, 3.1× on RTX 5090 |
| North Mini Code 1.0 | ~12–13 GB at ~3.3 bpw (est.) | not checked | SWE-V 67.6 (NMC, secondary source), positioned as sub-agent |
| Gemma 4 26B-A4B / 31B (04-02) | QAT-q4_0 from Google | – | LCB v6 77.1 and 80.0; SWE-V 31B: 52 (ORN) or 66.6 (MUSE), contradictory |
| Nemotron 3.5 Lightning | ~20 GB in Q4, does not fit without offload | low (Mamba) | Terminal-Bench 2.1 only 24 (NEMO), very fast |

For a "cheap" second worker next to Flash-Next, none of these fits in VRAM at the same time. Flash-Next
already occupies it almost entirely. In parallel, only alternating operation or a CPU worker would work.

## 6. Discarded: too large for ~60 GB

| Model | Date | total / active | smallest known file | Reason |
|---|---|---|---|---|
| GLM-5.3-Flash | 08-25 | 320B / 18B; KDA + DSA indexer (top-k 2048), mHC (GLM) | 1-bit ~93 GB (GLM, Unsloth) | too large; architecturally related, though (indexer, hyper-connections) |
| DeepSeek V4 Flash | April, 0731 | 284B / 13B | IQ3_XXS ~103 GB, needs 110 GB (DS) | too large |
| DeepSeek V4.1 Flash | 09-10 | 552B + ~200B n-gram | ~510 GB actual (DS) | too large |
| MiniMax M3.1-Flash | 10-17 | 220B / ~23B active, calculated from the config (MMX) | – | too much active |
| Step 3.7 Flash | end of May | 198B / 11B (STEP) | Q3_K_L 102.5 GB | 2 bit would be ~50 GB (estimate), too tight next to desktop and sessions |
| Kimi K3, GLM-5.2, Nemotron 3 Ultra, Tencent Hy3, Ling-3.1-flash | | ≥ 295B | | too large |
| gpt-oss, Devstral | | | | no new open version found in 2026 |

## Recommendation for us

Opinion of the agent, not a decision.

1. **Stay with Flash-Next and take the derivatives along. Highest priority, smallest effort.**
   - First check whether our coder GGUF is the ISTA rebuild of 10-17: SWE-V +3.6 points, zero code.
   - Check Gyro-Coder as the next quant candidate. Smaller experts bring ~60% instead of 39% cache share,
     exactly where our current gap lies (16–19% misses).
   - **Drawbacks:** new trellis kernel, experimental format, no SWE numbers. Before building the kernel, compare KLD and
     own coding tasks against the ISTA coder (the file lives in the fork, not upstream).
   - Watch Swift 1.5 and Signal. Shorter thinking and higher MTP acceptance act directly on the time
     per agent step. A coder prune of these is still missing.
2. **Qwen3.8-27B as a second model, for the fast worker with short to medium context.**
   - Same DeltaNet shapes, same tokenizer, DFlash2 drafter available, Apache-2.0
   - Per Qwen on par with Flash-Next on SWE-Pro and Terminal-Bench. It runs entirely in VRAM, without a
     RAM/PCIe path and thus without misses.
   - **Drawbacks:** full attention, from ~64–75K context the KV has to go to RAM; 256K works only slowly.
     Prefill ~4× more expensive. On long tasks (DeepSWE, Toolathlon) clearly behind Flash-Next.
3. **Ornith-1.5-35B-A3B as an option for several parallel workers.**
   - Our building blocks (DeltaNet, gated attention, MoE offload, MTP), ~3B active
   - 256K in VRAM feasible, ceiling clearly above Flash-Next
   - **Drawbacks:** benchmarks from the vendor only, new vendor, clearly weaker on long agent
     tasks. Worthwhile only if we want several workers at once.

Not recommended: Laguna S 2.1 and Ling-3.0-flash. Both need new attention kernels, own quants
and forks, for at most the same quality.

## Open questions

- Is our local ISTA coder GGUF the rebuild of 10-17? Compare the checksum of the first shard against HF.
- Gyro-Coder: How expensive is trellis decoding in the matvec on gfx1200? Does quality hold on agent tasks?
- Qwen3.8-27B in IQ3_S against our Flash-Next coder on our own coding tasks: measure ourselves.
- Does Qwen have a smaller model with the Flash-Next architecture? Paper [arXiv 2608.30320](https://arxiv.org/abs/2608.30320)
  names none.
