# 49: MoE models since 2026-10-01, and what is expected next

Research of 2026-11-14. Complements `08-models.md` (survey of 2026-10-23). Nothing here was run or measured by us.
Source marks: [V] page opened (Hugging Face model card, `config.json` or the HF API listing, retrieved 2026-11-14; the
page text was read through a summarizer, so numbers are as the page states them), [S] search snippet only, [K] from
memory or own arithmetic. Benchmark numbers are **vendor claims** unless marked otherwise. Dates without a year are 2026.
"Created" dates are the `createdAt` of the HF repository, not the announcement.

Question from the maintainer: "We have two models that run for us; everything else on the market is too big. What can the
newest MoE models do, and what rumors or expectations exist for future ones?"

Ranking criteria (maintainer's rule, 2026-11-14): quality for coding agents, memory fit, speed potential and license.
Engineering work is not a criterion; for each model the file states **what Tona needs** as a fact.

## Summary

1. **The open frontier moved to 300 B to 3 T parameters with 1 M context.** Everything released since 10-01 that is
   state of the art is too large for 16 GB + 64 GB: Kimi K3.1 Pro (2.8 T / 104 B), LongCat 2.5 (1.6 T / 48 B), Mistral
   Large 4 (1 T / 52 B), Meta Muse Spark Open (816 B / 42 B). None has a quantization below ~200 GB [K from parameter
   counts at 1.6 to 2 bpw].
2. **The class "Flash" (300 to 760 B total, 8 to 27 B active) is where a 32 GB card and 128 GB RAM start to matter.**
   DeepSeek V4.1 Flash-1017 (763 B stored, 552 B backbone, 8 B active in prefill / 16 B in decode, MIT) has the best
   evidence for coding: SWE-bench Verified 77.0 after ISTA's expert pruning (base 81.2, measured by ISTA, 94.8 %
   retained) and DeepSWE 77.0 (vendor) [V]. Its smallest usable file is 72.1 GB plus a 110.6 GB lookup table that can
   stay on SSD [V].
3. **For the reference box (16 GB + 64 GB, file budget about 60 GB) nothing new that is stronger than the ISTA Coder
   fits.** What fits is weaker or unproven: Ornith-1.5-35B-A3B (21.7 GB), Nemotron 3.5 Lightning 30B-A3B (SWE-bench
   Verified 51.6 vendor [S]), Laguna S 2.1 only with an own 3 bpw requantization (about 44 GB [K]), Nemotron 3.5 Bolt
   74B-A9B (about 52 GB of int4 experts [V], quality not found).
4. **A 32 GB card changes the ranking only together with RAM.** With 32 GB VRAM + 64 GB RAM (about 80 GB of file
   budget [K]) DeepSeek V4.1 Flash Coder (72 GB backbone) and Ornith-1.5-122B (69.7 GB) become possible; with 128 GB RAM
   MiMo-V2.6-Flash and GLM-5.3-Flash class files (100 to 115 GB) too.
5. **Rumors worth tracking:** Qwen 4 (in training per Alibaba; the "Flash" tier is speculated to be the Flash-Next
   architecture), Meta Muse Spark 1.2 weights ("soon"), Nemotron 4 (1 T+, reported), poolside's larger Laguna (in
   pretraining), DeepSeek V5 (code references only), MiniMax M4 (rumor). Several trackers contradict the primary
   sources; see "Source conflicts".

## Source conflicts (read before quoting any tracker)

| Claim by an aggregator | What the primary source shows | Verdict |
|---|---|---|
| Qwen 4 weights under Apache-2.0 "by Dec 20", Flash 30B/3.3B (local-ai-zone.github.io) [V] | Qwen HF org has no Qwen 4 repo (HF API, 11-14) [V]; the Alibaba press release only says "in training" (orcarouter.ai summary of it) [V]; a search found no such date, size or license [S] | **Unsupported.** Do not plan on it |
| Mistral Large 4 weights "overdue, no repository" as of 11-11 (local-ai-zone) [V] | `mistralai/Mistral-Large-4-1T-A52B` created 09-30, NVFP4 on 10-15 [V HF API] | Tracker wrong, or the repo is gated; the repo exists |
| Kimi K3.1 Pro license Apache-2.0 (mindstudio, local-ai-zone) [V] | Model card: "Kimi K3.1 Pro License"; HF tag `other` [V] | Not Apache-2.0 |
| Kimi K3.1 Pro "published Nov 4" [V tracker] | HF repo created 11-02 [V] | Minor |
| LongCat 2.5 "week of Nov 12" [V tracker] | HF repo created 11-04, MIT [V] | Minor |
| Inkling (Thinking Machines) open weights | Only trackers mention it; weights said to be closed with an Apache-2.0 release planned [S] | Not checked on HF |

## Models since 2026-10-01 (and relevant ones from September)

Coding scores are not comparable across columns: SWE-bench Verified is nearly saturated and vendors moved to DeepSWE,
Terminal-Bench 2.1 / 4.0 and in-house suites. A Terminal-Bench 2.1 of 84 to 90 and a Terminal-Bench 4.0 of 31 to 42 are
the same models on two scales of one benchmark family.

| Model | Created | Total / active | Experts, top-k | Attention | Context | MTP / draft | Native low-bit | License | Coding (source) |
|---|---|---|---|---|---|---|---|---|---|
| **DeepSeek V4.1 Flash-1017** [V] | 10-17 | 763 B stored (552 B backbone, rest Engram) / 8 B prefill, 16 B decode | 384 + 1 shared, top-6 | one KV head (MQA), head dim 512, sliding window 128, compressed KV, indexer top-512; hyper-connections (4 streams); Engram n-gram tables | 1 M | 3 MTP layers, DSpark draft (block 5) | FP8 + FP4 experts | MIT | DeepSWE 77.0, Terminal-Bench 4.0 42.3, ProgramBench 29.1 (vendor); SWE-bench Verified 81.2 base, 77.0 pruned (ISTA, 11-02) |
| **MiMo-V2.6-Flash-MOPD** (Xiaomi) [V] | 09-27 | 309 B / 15 B | 256, top-8, no shared | 39 sliding-window (128, 8 KV) + 9 global (4 KV), head dim 192, V 128, partial RoPE 0.33, sink bias | 1 M | 3 MTP layers, 5-layer SWA drafter (7 tokens) | FP8 + MXFP4 storage | MIT | none on the card |
| **Ornith-1.5-122B** (ornith-ai) [V] | 10-16 | 125 B / not stated | arch `qwen35moe`; the 35B sibling is 8/256 [V in 08] | DeltaNet + gated attention | 262 K, 1 M with YaRN | MTP in the community GGUF name; not verified | none | MIT | claims SOTA for its size, no numbers on the card |
| **GLM-5.3-Flash** (Z.ai) [V] | 08-25 | 320 B / 18 B | not stated | hybrid sparse + linear (first in GLM; Transformers `glm5_next`) | up to 1 M in evals | not stated | – | MIT | Terminal-Bench 2.1 84.3, DeepSWE 63.4 |
| **MiniMax-M3.1-Flash** [V] | 10-17 | 428 B / 23 B | not stated | MiniMax Sparse Attention throughout | 1 M | no MTP; DSpark draft head | – | `minimax-community` (text not read) | DeepSWE 73.7, Terminal-Bench 4.0 38.1, NL2Repo 61.2 |
| **Nemotron 3.5 Bolt 74B-A9B** (NVIDIA) [V via a community repo; NVIDIA's own page gave HTTP 401] | before 11-14 | 74 B / 9 B | 512 routed + shared, 40 MoE layers | 40 Mamba-2 + 8 attention layers, non-uniform | not stated (32 K in example) | – | int4 experts in the community quant | OpenMDW-1.1 | none found |
| **Nemotron 3.5 Lightning 30B-A3B** [S] | 08-11 | 30 B / 3 B | – | Mamba-2 + MoE + attention | 1 M | – | NVFP4 build | OpenMDW-1.1 | SWE-bench Verified 51.6, Terminal-Bench 2.1 24.6 (one source repeated) |
| **Step-5-Preview** (StepFun) [V] | 09-20 | 600 B / 27 B | not stated | not stated | 1 M | MTP used for speculation | NVFP4 build | StepFun Model License 1.0 | DeepSWE 67.7, Terminal-Bench 33.3 |
| **Mistral Large 4** [V] | 09-30 | ~1 T / 49 to 52 B | 128 + 1 shared, top-4 | not stated for the trunk | 1 M | DSpark draft (MLA layers) | NVFP4 build | Mistral Ecosystem Open-Weight License (modified MIT; Model-as-a-Service needs a license) | Artificial Analysis Coding Agent Index 48, Terminal-Bench 41 |
| **Kimi K3.1 Pro** [V] | 11-02 | 2.8 T / 104 B | 896 + 2 shared, top-16 | 69 KDA (linear) + 24 gated DSA (sparse) layers | 1 M | not mentioned | MXFP4 weights, QAT | Kimi K3.1 Pro License | Terminal-Bench 4.0 65.7 (vendor; 67 by Artificial Analysis per a tracker), DeepSWE 75.5 |
| **LongCat 2.5** (Meituan) [V] | 11-04 | 1.6 T (HF: 1.8 T) / 48 B | not stated | "LSA" | 256 K to 1 M | not mentioned | W4A4 build with learned transforms | MIT | Terminal-Bench 2.1 88.4, 4.0 38.5, DeepSWE 64.6 |
| **Muse Spark Open** (Meta) [V] | 09-15 | 816 B / 42 B | 256, top-8 | GQA 128/16, gated | 1 M | – | NVFP4 build | Muse Model License ("Built with Muse Spark Open", MaaS clause above USD 20 M revenue) | DeepSWE 60, Terminal-Bench 2.1 76.4 |
| Laguna S 2.1 (poolside), in 08 [V] | 07-13 | 118 B / ~8 B | 256, top-10 | 36 SWA-512 + 12 GQA | – | DFlash drafter | – | OpenMDW-1.1 | Terminal-Bench 2.1 70.2, SWE-bench Pro 59.4 |

Not opened or no new fact: Granite Switch 4.1 (8 B, 30 B; IBM, 10-14, GGUF exists) [V listing only]; Microsoft has only
small models in October (AesCode-8B/32B, listing only) [V]; Google has released no new Gemma since Gemma 4 in April
(E2B, E4B, 26B MoE, 31B dense, Apache-2.0) [S]; OpenAI has no open weights since gpt-oss in 2025-08 [S]; xAI: nothing
found [S]; Tencent Hy4 and Inkling are named by trackers only [S].

## Smallest usable quantization, fit and what Tona needs

Budgets for the file (weights that must be reachable per token; lookup tables read per token from SSD are listed
apart). Reference 16 GB + 64 GB: about 60 GB, because the 58.4 GB coder runs there [V README]. 32 GB + 64 GB: about
30 GB VRAM + about 50 GB RAM = about 80 GB [K, from docs/hardware/pc-24gb-and-larger.md]. 32 GB + 128 GB: about
130 GB [K]. Mac 64 GB: GPU working set 51 to 53 GB [V docs/hardware/mac-64gb-and-larger.md]. "Marginal" means within
10 % of the budget.

| Model | Smallest quant on HF | Size | 16+64 | 32+64 | 32+128 | Mac 64 | What Tona needs (facts) |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash-1017 | ISTA GSQ-RCO Coder IQ2_XXS, 192 of 384 experts kept [V]; unpruned IQ1_M [V] | 72.1 GB + Engram IQ4_NL 110.6 GB (SSD, one row per token); unpruned IQ1_M 120.7 GB + 110.6 GB | no | marginal | yes | no | New model code: MQA with head dim 512 and compressed KV plus top-512 indexer, sliding window 128, hyper-connections (4 streams, Sinkhorn), Engram table lookup (Tona already reads an n-gram table from RAM/SSD for Flash-Next), 384 + 1 shared experts with sqrtsoftplus routing, 3 MTP layers or the DSpark draft, FP4 or IQ2 expert kernels. llama.cpp support: ISTA says build b11870 or newer [V]; another card says it needs an open PR (#30817) [V]: conflict, unchecked |
| MiMo-V2.6-Flash-MOPD | mradermacher i1-Q2_K [V]; Baekpica VQ GGUF (1.63 / 2.63 bpw experts) [V] | 113.4 GB; 98.5 GB (embeddings on SSD) | no | no | yes | no | New model code: 39 SWA-128 + 9 global layers (head dim 192, V 128, partial RoPE, sink bias), 256 experts top-8 sigmoid routing without shared expert, 3 MTP layers. The VQ format needs its own runtime (ds4-dfm-rs) [V]; Q2_K loads in llama.cpp |
| Ornith-1.5-122B | MXFP4_MOE (prithivMLmods) [V]; ornith-ai GGUF exists, sizes not read | 69.7 GB | no | yes | yes | no | Same family as Qwen3.5/3.6 MoE (DeltaNet + gated attention + MoE), not the `qwen4_exp` family of Flash-Next: no QSA, no n-gram embeddings; shapes differ from the fixed shapes in the README; MXFP4 expert kernels |
| GLM-5.3-Flash | Unsloth and community GGUFs exist, sizes not read; "GLM 5.3 Slim" IQ2_M is 13.8 GB, probably heavily pruned [S] | about 80 GB at 2 bpw [K] | no | marginal [K] | yes [K] | no | New model code: hybrid sparse + linear attention layers (`glm5_next`); MoE details not read |
| MiniMax-M3.1-Flash | not checked | about 107 GB at 2 bpw [K] | no | no | yes [K] | no | New model code: MiniMax Sparse Attention, DSpark draft head instead of MTP |
| Laguna S 2.1 | only Q4_K_M [V in 08] | 96 GB; own 3 bpw about 44 GB [K] | with own quant | yes | yes | with own quant | New model code: 36 SWA-512 + 12 GQA, 256 experts top-10; DFlash drafter exists |
| Nemotron 3.5 Bolt 74B-A9B | community W4A16 (experts int4) [V] | expert modules 51.6 GB, total not stated | marginal | yes | yes | marginal | New model code: 40 Mamba-2 + 8 attention + 40 MoE layers with 512 experts, non-uniform layer config |
| Ornith-1.5-35B-A3B (08) | Q4_K_M [V in 08] | 21.7 GB | yes | yes | yes | yes | Same family as Ornith-122B; 30 DeltaNet + 10 gated attention, 8/256 experts, MTP |
| Step-5-Preview | NVFP4 only seen | 600 B, at least ~150 GB [K] | no | no | no | no | Card does not state experts or attention |
| Mistral Large 4, Kimi K3.1 Pro, LongCat 2.5, Muse Spark Open | NVFP4 / W4A4 / MXFP4; community Kimi "2bit" EXL3 and a "12GB SuperExpert" GGUF exist, quality unknown [V listing] | 1 T to 2.8 T | no | no | no | no | – |

Speed note [K, estimate]: DeepSeek at 2 bpw reads about 4 to 5 GB per decode pass (16 B active), the same order as
Flash-Next (5.1 GB, research/08). With a 32 GB card at 640 GB/s the bandwidth ceiling would be comparable; whether
the experts that do not fit in VRAM can be fed over PCIe 5.0 at that rate is unmeasured.

## Trends

- **Total size grew, active size stayed small.** Flash class: 300 to 760 B total with 8 to 27 B active (sparsity 1:20
  to 1:40); frontier: 1 to 3 T with 40 to 104 B active. Expert counts 128 to 896, top-k 4 to 16; routed experts with
  0 to 2 shared. Experts are small (DeepSeek 2304, MiMo 2048 intermediate).
- **Hybrid or sparse attention is now universal for long context:** linear layers (Kimi KDA, Qwen DeltaNet,
  Mamba-2), sparse indexers (DeepSeek top-512, Kimi gated DSA, GLM, MiniMax MSA), sliding windows of 128 to 512 with
  few global layers (MiMo 39:9, Laguna 36:12). 1 M context is the default claim. KV cache per token is small, so
  context no longer decides the memory class; the expert weights do.
- **Native low bit:** FP4/MXFP4 experts shipped by the vendor (DeepSeek FP4, MiMo MXFP4, Kimi MXFP4 QAT, LongCat W4A4).
  Because 4 bit is the original, 2-bit pruned and requantized builds (ISTA) lose little on vendor-style tests (94.8 %
  to 100 % retained [V]).
- **Multi-token prediction and draft heads are standard** (3 MTP layers in DeepSeek and MiMo; DSpark/DFlash drafters
  in DeepSeek, Mistral, MiniMax, poolside). Tona's use of the model's own head fits this.
- **N-gram / Engram lookup tables** (DeepSeek Engram 110 GB, Flash-Next n-gram embeddings) move parameters out of the
  per-token path: they sit on SSD or in page cache and are read per token. This favors machines with fast SSD and RAM,
  and is what lets a 763 B model run next to 12 GB of VRAM at all [V].
- **Coding/agent specialization:** vendor benchmarks are Terminal-Bench 2.1 / 4.0, DeepSWE, ProgramBench, NL2Repo;
  community builds specialize by pruning experts with coding data (ISTA Coder, Gyro-Coder, Kimi "Coding-720E").
- **Where the "16 to 32 GB GPU + 64 GB RAM" class is heading:** away from whole-model residency, toward
  "backbone in RAM/VRAM + big table on SSD". The strongest open coder within reach of 32 GB + 128 GB is DeepSeek V4.1
  Flash; with 64 GB RAM only the 60 to 80 GB class (Flash-Next, Ornith-122B, Laguna S, Bolt) is reachable. A vendor
  Flash model at 100 to 130 B total (the rumored Qwen 4 Flash, Laguna-class, Ornith-122B) is the class that fits;
  nothing at 30 to 60 B total with frontier coding quality was found.

## Rumors and expectations (next 3 to 6 months)

Status: **announced** (vendor said so), **leaked** (reporting from inside), **speculated** (extrapolation).

| Who | Item | Status | Source |
|---|---|---|---|
| Alibaba Qwen | Qwen 4 "currently in training" (Apsara, 09-22); tiers Max, Plus, Flash, 27B on a slide; Qwen4-Max preview in products from 11-11 | announced (training, tier names) | [V] orcarouter.ai summary of the press release; [S] |
| Alibaba Qwen | Qwen3.8-Flash-Next is "the preview of the architecture that will underpin Qwen4" (its model card, `qwen4_exp`) | announced | [V] model card as quoted by orcarouter.ai |
| Alibaba Qwen | Flash tier = 125 B / 6 B; 27B tier open weights; open weights at all | speculated (27B from release history) | [S] |
| Alibaba Qwen | Weights by Dec 20 under Apache-2.0, Flash 30B/3.3B | unsupported (single tracker, contradicted) | local-ai-zone.github.io [V] vs [S] |
| DeepSeek | V5 | speculated: references in code (AIBase, 10-28); earlier dates (July, September) passed; V4.x updates are the usual cadence | [S] |
| Moonshot | successor of K3.1 Pro | nothing found | [S] |
| Z.ai | GLM-6 named on an earnings call (08-31) per aggregators; no date, no open-weight statement as of 11-05 | speculated | [S] qcode.cc, orcarouter.ai |
| MiniMax | M4 in an "October to November" window | rumor from community posts | [S] cometapi.com explainer |
| Mistral | Large 4 weights announced for "late October"; repo exists since 09-30 | announced, done (modified MIT) | [V] HF, [S] TechCrunch 10-06 |
| Meta | open weights of Muse Spark 1.2 "soon" (Zuckerberg, Wang); `Muse-Spark-Open-816B-A42B` appeared 09-15 | announced, partly done | [S] gcn.com, [V] HF |
| NVIDIA | Nemotron 4 with a model above 1 T, built with the Nemotron Coalition (Mistral and others), "as early as this fall" | leaked (The Information, via vktr.com / orcarouter.ai) | [S] |
| poolside | larger Laguna model in pretraining | announced (no date; open weights not stated) | [S] VentureBeat, The Decoder |
| Google | Gemma 5 | speculated: "late 2026 or early 2027" extrapolation | [S] |
| OpenAI | gpt-oss successor | nothing found; only a forum request | [S] |
| xAI | open weights of a Grok model | nothing found | [S] |
| Microsoft, IBM | Phi / Granite | no large MoE found; Granite Switch 4.1 30B in October | [V listing] |
| agentionai, ISTA | DeepSeek, Kimi (K3-P48NVFP4) and Flash-Next derivatives; Kimi K3.1 community conversions appear daily | ongoing, observed | [V] HF listings |
| Thinking Machines | Inkling 2 Small (594 B MoE), open weights planned | tracker only | [S] |

## What this means for Tona

Ranking by quality for coding agents, fit, speed potential and license. Engineering work is not a criterion.

| # | Model | Why | Fit | License |
|---|---|---|---|---|
| 1 | **DeepSeek V4.1 Flash-1017 (ISTA Coder IQ2_XXS)** | Best coding evidence of anything within reach: SWE-bench Verified 77.0 (ISTA, 94.8 % of base), DeepSWE 77.0 and Terminal-Bench 4.0 42.3 (vendor); 8 to 16 B active, so bytes per pass are close to Flash-Next's; Engram table matches the n-gram path Tona already has; MTP and draft head present | 32+128 yes, 32+64 marginal; 16+64 and Mac 64 no | MIT |
| 2 | **Qwen 4 Flash, when it exists** | Announced to share the Flash-Next architecture lineage, so the present kernels are the likely match; size, date and license unknown | probably the reference class if 125 B / 6 B | unknown |
| 3 | **Ornith-1.5-122B** | MIT, 69.7 GB MXFP4; same DeltaNet family as the Flash-Next trunk; quality unproven (no numbers on the card; its 35B sibling is the strongest small MoE in 08) | 32+64 yes | MIT |
| 4 | **GLM-5.3-Flash** | Terminal-Bench 2.1 84.3 and DeepSWE 63.4 (vendor), 18 B active, hybrid linear + sparse attention | 32+128 yes; 32+64 marginal [K] | MIT |
| 5 | **MiMo-V2.6-Flash-MOPD** | 15 B active, 3 MTP layers, SWA-128 hybrid (small KV), MIT; no coding score on the card; quants start at 98 GB | 32+128 only | MIT |
| 6 | **Laguna S 2.1** | Terminal-Bench 2.1 70.2, SWE-bench Pro 59.4 (vendor); the only 100 B+ coder that reaches the reference class, with an own 3 bpw quant [K]; quality of such a quant unmeasured | 16+64 with own quant; all others | OpenMDW-1.1 |
| 7 | **Nemotron 3.5 Bolt 74B-A9B** | Fits best (51.6 GB int4 experts), 9 B active, but no coding number found | 16+64 marginal | OpenMDW-1.1 |
| 8 | **MiniMax-M3.1-Flash** | DeepSWE 73.7, Terminal-Bench 4.0 38.1 (vendor); 23 B active; license text unread | 32+128 | `minimax-community` |
| small tier | **Ornith-1.5-35B-A3B** | 21.7 GB, fits everywhere, SWE-bench Verified 79 (vendor, 08) | all | MIT |

- **Out of reach in all four memory classes:** Kimi K3.1 Pro, LongCat 2.5, Mistral Large 4, Muse Spark Open,
  Step-5-Preview. They matter as teachers and as markers of the trend, not as targets.
- **Does a 32 GB card change the ranking?** Not the order, but the cut line. With 16 GB + 64 GB the list reduces to
  Laguna S (own quant), Bolt and the small tier, and the ISTA Coder stays the best model. With 32 GB + 64 GB, #1
  (marginal) and #3 become possible; with 32 GB + 128 GB everything down to #8. RAM, not VRAM, decides for the 100 B+
  files, so the upgrade question is "32 GB card and 128 GB RAM", which is also what the hardware concept file for that
  class assumes.
- **Shared across the top candidates:** FP4/MXFP4 expert kernels, an SWA-plus-global or sparse attention path with a
  small KV cache, a 3-layer MTP head and a draft head (DSpark/DFlash), and tables bigger than VRAM read from RAM/SSD
  per token.

## Unchecked

- NVIDIA's own HF pages answered 401; Bolt facts come from a community requantization card. Its coding scores, release
  date and native context are unknown.
- Qwen 4 statements come from a press-release summary on orcarouter.ai and search snippets, not from Alibaba's pages.
- GLM-5.3-Flash, MiniMax-M3.1-Flash, Step-5-Preview: expert counts, GGUF file sizes and exact attention layouts not read
  (sizes above are arithmetic [K]); MiniMax license text unread.
- Ornith-1.5-122B: active parameters, official GGUF sizes, benchmark numbers, whether its MTP head is present.
- DeepSeek V4.1 llama.cpp support: two cards disagree (b11870 versus an open PR); not tested.
- Independent benchmarks (Artificial Analysis and others) were not read first-hand except as quoted by trackers.
- Not checked on HF or elsewhere: Alibaba's other orgs, Cohere, Tencent Hy4, Inkling, the Granite Switch card, quality of
  Kimi community quants, Reuters-style reporting on Mistral and NVIDIA timelines.
- Nothing here was run on our hardware; all speeds and "marginal" fits are estimates.

## Sources

- HF API listings, retrieved 2026-11-14 [V]: `huggingface.co/api/models?author=` for Qwen, moonshotai, deepseek-ai,
  zai-org, MiniMaxAI, mistralai, nvidia, google, poolside, meituan-longcat, stepfun-ai, tencent, inclusionAI,
  XiaomiMiMo, meta-models, ibm-granite, microsoft, agentionai, ISTA-DASLab, ornith-ai; searches for MiMo-V2.6-Flash,
  Nemotron-3.5, Kimi-K3.1.
- Model cards and configs [V]: deepseek-ai/DeepSeek-V4.1-Flash and -1017 (with `config.json`),
  ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO(-Coder)-GGUF, agentionai/DeepSeek-V4.1-Flash-Skeleton-GGUF,
  XiaomiMiMo/MiMo-V2.6-Flash-MOPD (with `config.json`), Baekpica/MiMo-V2.6-Flash-MOPD-VQ-GGUF,
  mradermacher/MiMo-V2.6-Flash-MOPD-heretic-i1-GGUF, zai-org/GLM-5.3-Flash, MiniMaxAI/MiniMax-M3.1-Flash,
  stepfun-ai/Step-5-Preview, moonshotai/Kimi-K3.1-Pro, meituan-longcat/LongCat-2.5,
  mistralai/Mistral-Large-4-1T-A52B, meta-models/Muse-Spark-Open-816B-A42B,
  prithivMLmods/Ornith-1.5-122B-MXFP4-MoE-MTP-GGUF, aidevco/Nemotron-3.5-Bolt-74B-A9B-W4A16-g128.
- Aggregators [V], treated as unreliable where they conflict with HF: local-ai-zone.github.io (November 2026 AI Model
  Updates), mindstudio.ai (new open-weight models roundup), orcarouter.ai/blog/qwen-4.
- Search snippets [S]: Nemotron 3.5 Lightning (datacamp.com), Nemotron 4 (vktr.com, cryptobriefing.com), DeepSeek V5
  (AIBase, thomas-wiegold.com), GLM-6 (qcode.cc), MiniMax M4 (cometapi.com), Mistral Large 4 (TechCrunch 10-06),
  Meta Muse (gcn.com), poolside (VentureBeat, The Decoder), Gemma 4 (The Register, 2026-04-02).
