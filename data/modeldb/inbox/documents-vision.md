# Inbox: documents and vision area

Emptied 2026-11-19 (gap-closing pass): Gemma 4 and Qwen3.5 vision facts moved to the general inbox, Qwen3.8-Flash-Next vision_config to the coding inbox, Holo3.1 and dots.mocr became records, German OCR evidence is in the document records (MDPBench README table, MORE paper).

Needs a human decision: none.
## Gemma 4 26B-A4B-it (google/gemma-4-26B-A4B-it), general area
- Report arXiv 2607.02770 (published Jul 2, 2026). Apache-2.0 (card), 25.2B total, 3.8B active, 30 layers, 128 experts with 8 active plus 1 shared, sliding-window (1024) plus global attention, 256K context, vocabulary 262K.
- Vision encoder about 550M parameters; text and image in, text out (no audio on this model). Image token budgets 70/140/280/560/1120; the card says high budgets suit OCR and small text.
- 140+ languages pretrained, 35+ out of the box (relevant for German documents, unmeasured).
- Vision/document numbers: MMMU Pro 73.8, OmniDocBench 1.5 average edit distance 0.149 (lower is better), MATH-Vision 82.4. No DocVQA or OCRBench on the card.
- llama.cpp docs/multimodal.md lists Gemma 4 (E2B, E4B, 26B-A4B, 31B) as supported vision models.
- Open: Tona has no Gemma 4 architecture; 25 GB total means expert offload on 16 GB.

## Qwen3.5-35B-A3B and Qwen3.5-9B (Qwen), general area
- Released February 2026 (card citation), Apache-2.0, vision encoder included, 262K context (YaRN to about 1M), 201 languages.
- 35B-A3B: hybrid Gated DeltaNet + Gated Attention with MoE (256 experts, 8 routed + 1 shared), 40 layers. Vision/document: MMMU 81.4, OmniDocBench 1.5 89.3, CharXiv RQ 77.5, OCRBench 91.0, MMLongBench-Doc 59.5.
- 9B: 32 layers, OCRBench 89.2, OmniDocBench 1.5 87.7, CharXiv RQ 73.0, MMMU 78.4.
- This is the architecture family Tona already runs (qwen35moe), so the document/image branch could reuse the resident model if the vision tower is loaded; research/52 notes the mmproj (1.79 GB F32 for 35B-A3B) and that the server drops image parts today.
- Card caveats: thinking on by default, no /think soft switch, high presence_penalty may mix languages.
- Open: whether our engine can load the vision tower; independent document accuracy.

## Qwen3.8-Flash-Next coder (Qwen), coding area
- research/52 records a `vision_config` in its config.json, with no mmproj in the ISTA GGUF. Not re-checked here. If the coder reads images itself, no side model is needed for screenshots.

## Holo3.1 (H company), vision/UI grounding
- Could not open the card (HTTP 401 from the fetch tool). Facts from research/52: 0.8B/4B/9B/35B-A3B, Qwen 3.5 based, Apache-2.0, 4B GGUF Q4_K_M 3.07 GB. Purpose: computer-use and UI grounding, not document reading. Needs a new record by an area owner who can open the card.

## Notes for the documents area (not records)
- No German-specific OCR benchmark was found for any model above. A multilingual parsing benchmark (MORE, arXiv 2607.02956) exists but was not readable; worth opening before choosing a German-capable parser. A German blog (wz-it.com, secondary) discusses local invoice OCR and says its table mixes vendor figures from different benchmark versions.
- Text PDFs: Docling-style text-layer extraction needs no vision model; the OCR/VLM branch is only for scans and photos. The app can pick the branch by testing whether a page has a text layer.

## From the coding area (gap pass 2026-11-19)
- `aescode-8b-32b` (Microsoft, Apache-2.0 by card, Qwen3-VL based): generates slides, posters and dashboards as editable HTML/CSS with an image-generator reference; created in the coding area, but it is a document and visual-design model, so the documents area may take ownership and judge it for the document role.

## From the 2026-11-19 new-release sweep

- Records added: unlimited-ocr-2, minicpm-ocr-1.0, jina-ocr-v1, wevisdoc-2b-4b, youtu-parsing-omni, lensvlm-9b, wevisone-vl-2b-4b, mage-vl-cowork, lbs-27b.
- **Not recorded**: MiniCPM-OCR-Jev (2B Qwen3.5-based OCR correctness scorer, AUROC 93.6 on JevOCRBench, Apache tag, https://huggingface.co/openbmb/MiniCPM-OCR-Jev), mentioned in the minicpm-ocr-1.0 record; a Xiaomi-OCR-0 model appears in the Jev comparison table and was not looked up; inclusionAI ZwZ2-4B/9B/27B (visual-perception RL on Qwen3.5/3.8, +4.96 points over Qwen3.5-9B on 25 benchmarks), ZoomGain-4B/9B and SingVL_S1_4B (decision model) are research artifacts without a document or coding use; tencent/EVIE not checked.
- WeVisDoc-4B vendor table (OmniDocBench v1.6): HunyuanOCR-1.5 94.74, Unlimited-OCR 93.92, Qianfan-OCR 93.90, dots.mocr 92.57; Youtu-Parsing-Omni claims 96.96 but excludes EU use in its license.

## From the 2026-11-19 new-release sweep
- `evie-visual-retriever` (Tencent, Apache-2.0): ViDoRe V3 nDCG@10 66.75 (8B) and 66.02 (4.5B, Prefix-MRL 64-2048 dim) per the builder; the late-interaction multi-vector index is the cost to plan for. Candidate for the documents area's OCR-free retrieval.
- `pplx-embed-v2-context` (Perplexity, MIT per card): contextual chunk embeddings plus image bands; 0.6B and 9B share a 1024-dim space.
- `meta-encoder-v1` (28B, 57 GB BF16, license unstated): rejected for size; its base architecture (`muse_glimmer`) matches the existing `muse-glimmer-30b` record.
- `index-translate-9b` (Bilibili, Apache-2.0) translates text inside images without OCR via mmproj; relevant to the vision area as a translation-by-image option.
- `moss-image-sft` and `qwen-image-2.1-turbo`: image generation records (MOSS Apache-2.0, Qwen research-only license); text-rendering benchmarks are builder-made.

## From the 2026-11-19 gap pass 2
- Image/video records added: `ming-image-0.1-design` (MIT, UI/poster design with RGBA and a layer-decomposition sibling, vendor-validated on 80 GiB), `flux3-image-klein-base` (training base of the distilled Klein), `qwen-image-2.1-pe` (9B prompt rewriters, research-only license; Turbo-PE repo does not seem to exist), `fastframes` (Apple, research-only), `invoker-render-1.0` (33B, non-commercial), `genfirst` (ByteDance research, CC BY-NC).
- `meta-encoder-v1`: card of the first release `facebook/meta-encoder` read (Apache-2.0 by card summary, 30B bi-encoder, vendor retrieval table); still rejected for size.
- Ming-Image and Qwen-Image-2.1 both recommend a separate VLM prompt enhancer (Ling-3.0-flash-VL or qwen3.8-27B; Qwen-Image-2.1-PE 9B); an LLM prompt-rewrite step already available in Tona would cover that for any image path.

## From the 2026-11-19 gap pass 2 (vision-language, OCR, GUI agents)
- Records added: surya-ocr-2.5, xiaomi-ocr-0, typhoon-ocr2, nemotron-3.5-parse, openthai2.1-ocr-markdown, lightonocr-3, minicpm-ocr-jev, holo4, sensenova-6.8-code2video, sensenova-loop-9b-preview, ling-3.0-flash-vl, zwz2-zoomgain, nv-reason-ct, mirage-cactus, s5-nano, molmo2-8b-minimax-h3, wevisforensics.
- New model class "decision models" (typed questions in, calibrated probability per option out in one forward pass, no generation; Jev / SystemOne API, community Decision Index boards): records clef (Cloudflare 27B/9B/30B-A3B), jev-27b-vl (AutoTrust, 992k downloads), winnow (Gemma-4 12B/E4B/E2B, runs on 16 GB), intern-decision (0.8B/2B/4B), xing-vl2.0-decision, singvl-s1-4b, glide-27b, liquid-d1, h2o-lightning-4b. For Tona they are router/judge/guardrail components, not workers; none has a quantized release except Winnow (GGUF). Boards: https://huggingface.co/spaces/multimodalart/jev-decision-index (not opened, numbers quoted from cards).
- Best document candidates this pass (all vendor numbers): xiaomi-ocr-0 (0.8B, OmniDocBench v1.6 96.83), nemotron-3.5-parse (0.6B, OpenMDW, ParseBench 75.20, but olmOCR-bench 62.7 per a third party), surya-ocr-2.5 (0.7B, licence caps users above 5M USD revenue or funding), lightonocr-3 (third-party ParseBench 75.10 for the 4B, 74.60 for the 0.8B). No German-specific benchmark found for any.
- GUI/computer-use: holo4 (35B-A3B Apache-2.0 and 27B CC BY-NC; OSWorld 85.2 and OSWorld 2.0 61.7 for the 27B, vendor harness) is the relevant release for coding-agent workers; official Q4_K_M GGUFs 16.9 and 21.3 GB.
- Not recorded in this pass (large text models that also carry a vision tag, left to the general/coding areas): moonshotai/Kimi-K3.1-Pro (1.56 TB), meituan-longcat/LongCat-2.5 (3.5 TB, MIT), mistralai/Mistral-Large-4-1T-A52B (1.06 TB); none fits any hardware class. meta-models/Muse-Spark-Open-816B-A42B, thinkingmachines/Inkling-2-Small, tencent/Hy4-flash were already recorded by others.
