# Inbox from the general / knowledge / research area

Findings for models owned by other areas. Sources opened 2026-11-19 (Hugging Face API listings unless noted).

## Nemotron 3.5 Bolt (coding agent owns the file)
- Family listing (nvidia, createdAt): Bolt-74B-A9B-24GB (2026-10-07, 45,474 downloads), its GGUF (2026-10-07, 225,574 downloads),
  Bolt-74B-A9B-BF16 (2026-10-09), Bolt-74B-A9B-24GB-LiLiCorr and its GGUF (2026-10-12, 35,895 downloads), Bolt-E4B-4GB (2026-10-05,
  172,088 downloads) with a DFlash drafter, Bolt-E4B-BF16-DFlash-NVFP4 (2026-10-09). NVIDIA's own GGUF repos exist for the 74B,
  which research/49 listed as "community repo only".
- Same nemotron_h_moe family as Nemotron 3.5 Lightning (30B-A3B) and Super VL (120B-A12B), see `nemotron-3.5-lightning-30b-a3b.json`
  and `nemotron-3.5-super-vl-120b-a12b.json` (general area): the nemotron_h_moe backend of Tona also covers these.

## Qwen3.8-Flash-Next (coding agent owns the Coder)
- HF `Qwen/Qwen3.8-Flash-Next`: created 2026-08-24, 2,852,539 downloads, tags image-text-to-text, conversational; FP8 repo 265,742
  downloads. Not read in detail by this area. Sibling `Qwen/Qwen3.8-2.4T-A95B` (2026-08-08) is far too large for any class.
- Qwen3.8-27B (general area file `qwen3.8-27b.json`) shares the vocabulary (248,320) and DeltaNet shapes with Flash-Next (research/08).

## Ornith
- No new information. Qwen3.6-35B-A3B (`qwen3.6-35b-a3b.json`, general area) has the same parameter count that Ornith-1.5-35B-A3B finetunes.

## Other finds not in any file
- mistralai/Leanstral-1.5-119B-A6B (2026-07-01, Apache-2.0, vLLM only): name suggests Lean proving; not opened.
- mistralai/Mistral-Medium-3.5-128B (2026-03-31, license "other", 128B): not opened; too large for most classes.
- Qwen/Qwen-AgentWorld-35B-A3B (2026-06-22, "world-model, agent"): not opened.
- google/diffusiongemma-26B-A4B-it (2026-06-09, Apache-2.0, 598,561 downloads): diffusion language model; not opened.
- nvidia/Nemotron-Labs-Sigma-distill-* (2026-11-09, 3B/8B, nfe2 to nfe16 variants), Nemotron-3.5-Embed and Rerank (embeddings area).

## Vision/document facts for general-area records (from the documents and vision area, 2026-11-19)
- Gemma 4 26B-A4B-it (google/gemma-4-26B-A4B-it): vision encoder about 550M parameters, text and image in, text out; image token budgets 70/140/280/560/1120 (high budgets for OCR); card: MMMU Pro 73.8, OmniDocBench 1.5 edit distance 0.149 (lower is better), MATH-Vision 82.4; no DocVQA/OCRBench on the card. llama.cpp docs/multimodal.md lists Gemma 4 (E2B, E4B, 26B-A4B, 31B). Vendor claims; German OCR unmeasured. Report arXiv 2607.02770.
- Qwen3.5-35B-A3B: vendor card MMMU 81.4, OmniDocBench 1.5 89.3, CharXiv RQ 77.5, OCRBench 91.0, MMLongBench-Doc 59.5; mmproj 1.79 GB F32 (research/52). Qwen3.5-9B: OCRBench 89.2, OmniDocBench 1.5 87.7, CharXiv RQ 73.0, MMMU 78.4. Same architecture family Tona already runs (qwen35moe); the server drops image parts today.
- Independent (MDPBench, 17 languages, photographed and digital pages; https://huggingface.co/datasets/Delores-Lin/MDPBench README table, read 2026-11-19): Qwen3.5-Instruct-9B overall 65.7, German 72.8 (digital 74.8, photographed 62.7); Qwen3-VL-Instruct-8b overall 68.3, German 73.7; Claude-Sonnet-4.6 overall 73.1, German 79.8; Gemini-3-pro-preview 86.4, German 91.2. Dedicated parsers beat the general 8-9B VLMs on German pages (GLM-OCR 82.7, PaddleOCR-VL-1.5 84.8, Chandra 86.6): use a parser for scans, a general VLM for questions about content.
- New document records created by this area: dots.mocr.json, holo-3.1.json.
- From the audio area (2026-11-19): Open ASR Leaderboard (Space config read 2026-11-19) has audio-capable general models: google/gemma-4-E4B-it English avg WER 13.2 (rank 68 of 69), German 7.59 (7 of 8 sets): weak as an ASR; microsoft/Phi-4-multimodal-instruct German 3.92 (5 of 8 sets); OpenLLM-France/Luciole-8B-Audio-1.0 German 3.76 (6 of 8 sets), English 8.73. Add to the Gemma 4 E4B record and consider Phi-4-multimodal.
- From the audio area: Gemma 4 26B-A4B is the decoder of salesforce SOV-Transcribe-1; Ministral 3 3B is the decoder of Voxtral Mini 4B Realtime; Nemotron 3.5 Lightning 30B-A3B is the decoder basis of Nemotron 3.5 Transcribe.
## From the coding area (gap pass 2026-11-19)
- Independent anchor (mirror of the Artificial Analysis Coding Index, https://benchlm.ai/benchmarks/aacodingindex, read 2026-11-19, no per-row dates): Qwen3.6-35B-A3B 41.9, Mistral Small 4 26.6, Nemotron 3.5 Lightning 30B A3B NVFP4 26.8, Qwen3.8-27B 68.1, Qwen3.8-Flash-Next 73.0. Records `qwen3.6-35b-a3b` and `mistral-small-4-119b-a6b` can add these as independent rows (coding only).
- New records in the coding area that carry general facts: `glm-5.3-flash`, `minimax-m3.1-flash`, `mimo-v2.6-flash-mopd`, `ling-3.0-flash`, `longcat-flash-lite-sparse`, `step-3.7-flash` (general and knowledge benchmarks such as HLE, GPQA, MMLU-Pro are partly in the vendor tables; only coding rows were entered).
- `glm-5.3-flash` has its MIT text verified; `minimax-m3.1-flash` carries the MiniMax Community License (non-commercial by default, notice duty for commercial use): relevant for any "offer" list.
- OpenMDW-1.1 text was read for NVIDIA Nemotron 3.5 (Lightning, Bolt, LiLiCorr drafter): records updated in the coding pass; the Super-VL 120B record may reuse the same text note.
Empty: every item of the 2026-11-19 file was folded into records (see `models/`) or forwarded to `coding.md`.

## From the 2026-11-19 new-release sweep
- Translation specialists now in the database: `plamo-3-translate` (PFN, 0.8B-31B, custom license with revenue tiers), `index-translate-9b` (Bilibili, Qwen3.5 base, Apache-2.0, strongest published instruction-following numbers) and `hy-simulmt-30b-a3b` (Tencent, zh-en simultaneous speech translation only). `hunyuan-mt-7b` is the only Hy-MT record; Hy-MT2 (1.8B/7B/30B-A3B, Apache-2.0, 33 languages incl. German, released 2026-05-21) falls before the sweep window and has no record yet; Index-Translate's card compares against it.
- `pplx-embed-v2-context` and `hs-rerank-1` (114M multilingual reranker, MIT) are new retrieval components for the research workers' local memory/RAG; `yuanbao-reranker` (Apache-2.0, Qwen3 base) claims +1.5 nDCG over Qwen3-Reranker-8B, builder-measured only.

## From the 2026-11-19 gap pass 2
- `tiny-aya-thinkers`: L2-Thinker thinks in the prompt language (44 languages plus English, German included), base is `tiny-aya-base-32K`; all CC-BY-NC-4.0, gated, no benchmark numbers on the cards read.
- Language-specific open models added: `kolibri-1` (Aleph Alpha, German/EU, 78B-A3.5B, Apache-2.0 read; its own table has Qwen3.8-27B ahead in DE 79.9 vs 70.8), `aliceai-foundation-80b-a3b-base` (Yandex, Russian, base only), `jais-3-32b` (Arabic), `llm-jp-4.1-32b-a3b-thinking` (Japanese, plain qwen3moe), `raon-1-mini-preview` (Korean). `memt-1.0` and `sage-a1.5`, `hydrogen-1.0` are research/science variants.
- `mimo-v2.6-distill-qwen-9b` reconciled: added the card's internal-set scores (MiMo Code mini 51.6, Cyber 31.3, General 62.2, OfficeQA 19.5, JobBench 18.3), generation_config sampling (0.6 / top_k 20 / top_p 0.95), a user GSM8K audit (96/100 like base but think chains of 150 to 400 tokens instead of 1800 to 3500), MTP user numbers and all GGUF repo sizes; corrected the earlier claim that prithivMLmods ships an MTP repack (the MTP-patched Q8_0 is IHaveNoClueAndIMustPost's).
- `sage-s2.1` repository carries the card of "Atria Dawn" (744B GLM-5.2 base, MIT statement, no LICENSE file); `sage-s2.1-flash` is 320B/18B on the GLM-5.3-Flash architecture. Both are science/agent models, rejected for size or lack of a usable GGUF.
- Gap pass 2 architecture trace (llama.cpp b11042-b12549 titles, vLLM v0.29-v0.34) wrote: `ternary-bonsai-2-27b` (Prism ML, 5.95-7.63 GB, 98.2% of FP16 per vendor; fastest packs need the vendor llama.cpp fork, Q2_0 runs on stock b11556+; strong candidate for 12/16 GB classes), `ripple-65b-preview` (dense 65B, Apache-2.0, Terminal-Bench 2.1 80.5 vendor), `pplx-decider-v1.2` (decision-model family with Liquid d1, Clef, CLM), `limite-1b-violetto` (1B math), `domyn-edge` (4B-class, 12 languages), `sensenova-u1.5-8b-mot` (unified image generation and editing).
- Architectures seen in the notes and not found as real models on HF: Qwen4Exp (only tiny-random fixtures; the qwen4exp graph appears to back Qwen3.8-Flash-Next), GLM5Next (tiny fixtures only), Cohere Compass (tiny-random only), maion-coder (no repo found), "mistral le chonk" (no repo found), Metis-1.6 (Lernex Logos/Praxis, under 2k downloads: skipped), ZAYA1 (Zyphra, May 2026, older than the window: not recorded), nimble decision (one 14-download repo).
