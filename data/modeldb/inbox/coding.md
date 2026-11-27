# Inbox from the coding area (working notes, last pass 2026-11-19)

The first-pass items are folded into records (`data/modeldb/models`): Muse Glimmer, Nemotron 3.5 Bolt and Lightning,
GLM-5.3-Flash, MiniMax-M3.1-Flash, MiMo-V2.6, Ling-3.0, LongCat-Flash-Lite-Sparse, Step-3.7-Flash, Zeta, AesCode,
Granite Switch 4.1 and Qwen3.8-27B all have records. Open:

- **Qwen3.8-Flash-Next (base, unpruned) and its GSQ-RCO unpruned releases**: no record for the base yet (125B total, 6B
  active, 512 experts, qwen4_exp, qwen-community-1.0; vendor SWE-bench Pro 62.5, Multilingual 81.0, DeepSWE 58.7,
  LiveCodeBench v6 91.9, Toolathlon 73.5; card https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Tona has the Q2_0 repack
  in test (`flash-next-512-q2_0`, toolchain/models.toml). The record `qwen3.8-flash-next-gsq-rco-coder-iq1_m` holds the
  architecture reasons; a general-area record may link to it.
- **Hugging Face downloads are unreliable as a quality signal** (the top "coder" listing has merge repos with 1.7 to 2.8M
  downloads and 3,000+ likes next to vendor repos with 60 to 300k).

## Qwen3.8-Flash-Next vision (documents and vision area, 2026-11-19)
- config.json of Qwen/Qwen3.8-Flash-Next (HF raw, read 2026-11-19): architectures = Qwen4ExpForConditionalGeneration and a vision_config block (deepstack_visual_indexes, depth, hidden_size, ...). The model class has a vision tower; whether the vision weights ship in the repo and whether the ISTA coder GGUFs carry an mmproj is unchecked (research/52 saw none).
- **Qwen2.5-Coder-1.5B (base, FIM)**: released 2024-09 (predates 2025); not recorded. Decision for the maintainers: record legacy FIM models as a baseline for autocomplete, or keep the database to 2025+ releases.
- **Community coder finetunes with very high download counts** (Qwopus3.6-35B-A3B and 27B Coder lines by Jackrong, Tiel/Cyber-Tiel lines, DavidAU and nerkyor Qwen3.8-27B merges): only Tiel-Coder-35B-A3B has a record; the others have self-reported numbers only. Decision: whether finetunes get their own records or stay variants of their base record.
## From the general / knowledge / research area (gap-closing pass, 2026-11-19)

- **Nemotron 3.5 Bolt 74B-A9B (IQ2) German evidence**: independent PeerBench DACH index
  (https://dach.peerbench.ai/api/leaderboard.json, runs 2026-10-15/16): INCLUDE DE 73.8, GermEval NER 73.4, ScaLA 58.9,
  SB10K 52.8, MuSR DE 76.9; Qwen3.8-27B BF16 scores 87.9 / 81.5 / 66.9 / 62.6 / 86.1 on the same tasks. License text read:
  OpenMDW-1.1 (Lightning repo LICENSE and openmdw.ai), no restriction on commercial use; the Bolt repos carry no LICENSE file.
- **Qwen3.8 Flash** (PeerBench DACH, provider-internal route, 2026-10-15): INCLUDE DE 89.7, GermEval 79.1, SB10K 62.5,
  ScaLA 67.6, overall German Elo 1574 (Qwen3.8 27B BF16: 1634). Not the Flash-Next weights, but the same family.
- **Granite 4.2 30B** (dense, Apache-2.0, official GGUF) now has a record `granite-4.2-30b` (SWE-bench Verified 57.0,
  Terminal-Bench 2.1 29.24, vendor); `granite-switch-4.1-30b` was extended (license text read).
- **Mistral Medium 3.5 128B** (SWE-Bench Verified 77.6 vendor, dense, Modified MIT with a USD 20M monthly revenue cap) and
  **Leanstral 1.5 119B-A6B** (Lean 4 agent, Apache-2.0) now have records in the general area.
- **Qwen3.8-27B**: the general area extended `qwen3.8-27b` (license text read, independent German scores, verdict offer).

## From the 2026-11-19 new-release sweep

- **Not recorded, general area**: NousResearch Arete-27B (2026-11-06, SFT of Qwen3.8-27B for value alignment, Apache-2.0 tag, https://huggingface.co/NousResearch/Arete-27B); inclusionAI Ling-3.1-flash (2026-10-13) and Ling-3.1-flash-base; DeepSeek-V4.1-Flash base (2026-09-10, 1.48M downloads, only the gsq-rco-coder finetune has a record); LiquidAI LFM2.5-24B-A2B and LFM3-1.7B; openbmb MiniCPM-V-4.7-35B-A3B (869k downloads) and MiniCPM5-2B; Kimi-K3.1-Pro, Mistral-Large-4 and LongCat-2.5 (too large for every class).
- **Math**: no new small math specialist with a post-RL release found; openbmb JustRL-II-base-model is only the RL starting checkpoint (about 61 percent AIME 2025, blog claims 81 percent after RL), not recorded. Sage-S2.1-Flash (internlm) is 643 GB.
- Mellum2.1 Thinking (record added) lists AIME 25/26 83.3 and LiveCodeBench v6 82.0 at 2.5B active; Nanbeige4.3-33B lists AIME 2026 95.2 and GPQA Diamond 89.5 (vendor).
- `frognano-4b-2609` (Microsoft, coding: Qwen3.5-4B RL-trained on 1,500 synthetic SWE tasks, SWE-bench Verified 61.5 vendor, Python only) and `mimo-v2.6-distill-qwen-9b` (agentic SFT of Qwen3.5-9B) are candidates for a Rust/Go eval; `hy-memory-30b-a3b` (agent long-term-memory writer, Apache-2.0) and `qwen3.8-27b-drafters` (DSpine/NVIDIA drafters, acceptance 5.6 vs 3.8 for the MTP head) are relevant prior art for coding agents.
- Not recorded (no time or gated): Hy4-flash (440 GB), Ling-3.1-flash (574 GB), DeepSeek-V4.1-Flash (510 GB), ERNIE-5.5-Flash-Base (288 GB), LongCat-2.5, Kimi-K3.1-Pro, Mistral-Large-4, Step-5-Preview, Muse-Spark-Open (gated), Sage-S2.1-Flash (643 GB); all far beyond any hardware class.

## From the 2026-11-19 gap pass 2
- `muse-spark-open-816b-a42b` (Meta, 816B/42B, 1644 GB BF16, smallest GGUF UD-IQ1_S 175.64 GB): reject for every class; the license (MIT-style, "Built with" line, US$20M Model-as-a-Service clause) is permissive and the coding scores are vendor-run (Terminal-Bench 2.1 76.4, DeepSWE 60). The official repo is `meta-models/...`, not `nvidia/...`; its NVIDIA DFlash2 drafter (4.38B, 5 layers, block size 8, pairwise path selector) is relevant prior art for speculative decoding.
- `deepseek-ai/DeepSeek-V4.1-Flash-Base` does not exist (API 401, absent from the author listing); only V4-Flash-Base and V4-Pro-Base do. `DeepSeek-V4.1-Flash` itself is public (MIT tag, not gated).
- Text-LLM sweep (worker 2026-11-19): new records `flint` (Recursive, 39.8B-A2.5B agentic coder, Apache-2.0 read, unsloth GGUF 25 GB, strongest fit for 16 GB class), `xing4.0-29b-a4b`, `k2-horizon-mova-36b-a4b`, `nex-n2.5-mini`, `neohorse-1-4b`, `spark-x2.6-4b`, `raon-1-mini-preview`, `ling-3.0-tiny`, `beam-1.0-501b-a23b` and `qwen3.8-2.4t-a95b` (frontier size, fit no). The meshllm "Qwen3.8-2.4T-A95B" repos are community layer splits of the real official Qwen model.
- Quantization-technique items to study, not offer: `thor-t-35b-a3b` (ternary multi-plane GGUF of Ornith-1.5-35B-A3B with ternary MTP head, forked llama.cpp), `edge0-35b-a3b-preview` (learned prerouter predicting experts one step ahead for SSD streaming), `underdog-saluki-27b-1.0` (2-bit Qwen3.8-27B in 7.9 GB, but CC BY-NC).
- `twil-8b` (Granite-4.2-8B RL fine-tune with strong vendor math/LiveCodeBench numbers) is non-commercial licence (text read): reject.
- Trend: typed "decision models" (Cloudflare Clef, `autotrust-decider-pro-vision-27b`, `wald-26b-a4b`, `laplace-1-atlas`, H2O-Lightning-4B) return per-option probabilities, not text; irrelevant for a coding worker. `autotrust` has 991k downloads for a LoRA on Qwen3.8-27B with no explanation.
- `shashanth-27xA9/*-SuperExpert-*` repos claim 320B-class models in 8-12 GB (SVD/'zero-drift' marketing, widget text, no independent evidence): treated as not a real release.
- `deepseek-v4.1-flash` (MIT text read, 552B backbone + 196B Engram, 16B active in decode, 1017 update: DeepSWE 77.0, Terminal-Bench 4.0 42.3, vendor): the 1017 GSQ-RCO GGUFs of ISTA-DASLab have an unpruned backbone of 120.7 GB (IQ1_M) to 138.2 GB (IQ2_XXS) plus a 110.6 GB Engram file read from SSD, so the unpruned model fits the 32 GB VRAM + 128 GB RAM class; quantiser-run AIME25/GPQA/LCB v6 recovery 98.7 % (IQ1_M) and 100.0 % (IQ2_XXS). No separate Base repo exists. The pruned `deepseek-v4.1-flash-1017-gsq-rco-coder` is the 24 GB class option.
- Rejected by size in this pass (records written): `kimi-k3.1-pro` (2.8T/104B, smallest GGUF 593 GB), `longcat-2.5` (1.6T/48B, MIT read, no community quant), `mistral-large-4-1t-a52b` (1T/49B, smallest GGUF 233 GB), `step-5-preview` (600B/27B, smallest GGUF 182 GB), `sage-s2.1` (744B class, 1.5 TB), `ling-3.1-flash` (560B/25B, smallest GGUF 131 GB would only just fit 128 GB RAM + 32 GB VRAM), `sage-s2.1-flash` (320B/18B, no llama.cpp-compatible GGUF; same architecture size as `glm-5.3-flash`).
- `hy4-flash` (Tencent, 220B with 64B Engram tables, 12B active, Tencent Hy License read: MaaS above US$50M revenue needs a licence, no territory clause): watch. 49 GB resident at 2.3 bits (AtomicChat), 105 GB at Q4_K_M plus Engram on SSD; unusually quantisation-sensitive (Q8_0 only 91.8 % top-token agreement vs BF16); llama.cpp support is the unmerged PR #32517; the vendor itself recommends the larger Hy4 for long-horizon software engineering.
- `intern-s2-397b` (Apache-2.0 read, Qwen3.5 MoE architecture, 397B): 2-bit GGUFs of 119 to 155 GB fit the 128 GB class, but it is science-tuned and no coding score was found; its 35B siblings `internlm/Intern-S2-Preview` and `Intern-S2-Mobius` (73 GB BF16 each) are the ones to evaluate for small classes (no records yet).
- `ernie-5.5-flash-base` licence (text read): grants no rights in the EU, UK or South Korea and bars use of outputs there; base model only. Reject.
- Gap pass 2: `ripple-65b-preview` (Ricursive, dense 65B, Apache-2.0, vendor Terminal-Bench 2.1 80.52; community GGUF Q4_K_M 40.4 GB) and `ternary-bonsai-2-27b` (ternary Qwen3.8-27B, vendor 14-benchmark thinking average 84.78, coding average 89.42) were written by the audio/architecture worker; both need our own agentic eval before any verdict above candidate.
