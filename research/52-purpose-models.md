# 52: Purpose-bound models next to the coding model (image, speech, embeddings, autocomplete, vision)

Research of 2026-11-15 for #151. Nothing here was run or measured by us; no GPU, no server, no download of weights.

Source marks: **[V]** a primary page was opened on 2026-11-15 (Hugging Face model card, `config.json`, tokenizer config or
the HF file-tree API, an arXiv abstract, a GitHub README or release API); sizes in bytes are the file sizes of the HF tree.
**[K]** claimed by a secondary source (search-result summary of a blog or aggregator) or a vendor claim not checked against
a primary page; treat as unverified. **[E]** own estimate, with the arithmetic shown, from the measured numbers of
[docs/backends/reference-platform.md](../docs/backends/reference-platform.md) (93 TFLOPS f16 WMMA, 315 GB/s VRAM, 55.7 GB/s PCIe 5.0 x16,
58 GB/s RAM read). **Unchecked** = not verified, with the check that would settle it. Benchmark numbers are vendor claims
unless a line says otherwise. HF "license" values are the repository tag; where the tag and the license text could
disagree (see [42-model-licenses.md](42-model-licenses.md)) the text is named.

Question (#151): purpose-bound models next to the coding model (in the product: selectable one at a time), for example image generation. Are there MoE models for
that, is it necessary, feasible and sensible, and which options do we have?

## Summary

1. **MoE models exist for most purposes, but MoE does not mean what it means for the coding model.** Image: HunyuanImage 3.0
   (80B total, 13B active, 64 experts) [V], Nucleus-Image (17B, about 2B active, 64 experts, Apache-2.0) [V], HiDream-I1
   (17B, "sparse DiT with dynamic MoE") [V abstract], WeLM-Image (31.7B stored, 5.4B active) [V abstract]. Video: Wan 2.2 A14B
   [V], but its two experts are split by timestep, one is active at a time. Speech: Nemotron 3.5 Transcribe 30B-A4B [V],
   ZONOS3-Mini and ZONOS2 (TTS) [V], Ming-3.0-tts-tiny 7.9B-A1.3B [V]. Embeddings: nomic-embed-text-v2-moe [V]. Vision-language:
   Qwen3.5-35B-A3B, Qwen3-VL-30B-A3B, Gemma-4-26B-A4B, Holo3.1-35B-A3B [V].
2. **Diffusion is compute-bound, not bandwidth-bound; MoE only cuts FLOPs and expert offload does not help.** A diffusion step
   processes thousands of tokens at once, so every expert is hit in every layer (Nucleus-Image even uses expert-choice
   routing, which forces balanced use [V]). [E] Z-Image at 1024 px: about 6e13 FLOP per step against 12 ms of weight reads at
   Q4; the step is about 100 times compute-bound. The megakernel and the expert arena are built for the opposite case
   (batch 1 decode, 5.1 GB read per token) and give nothing here.
3. **Models that decode autoregressively with small batches are memory-bound like our coder** and Tona's machinery would apply
   to them: the vision-language MoEs (Qwen3.5-35B-A3B is the architecture family `qwen35moe` that Tona already runs, #143),
   the MoE speech decoder Nemotron 3.5 Transcribe, and the AR backbone of MoE TTS. Only for those is "inside Tona" cheap.
4. **Vision is the one purpose that sits inside the coder's own family.** The base of our coding model,
   `Qwen/Qwen3.8-Flash-Next`, ships a vision encoder (`vision_config` in `config.json`, `image-text-to-text`) [V], but the
   ISTA GGUF we use contains no `mmproj` file [V file list], and `engine/server/api.cpp` replaces images by the text
   "[image omitted]" [V, lines 19-20]. Screenshots and UI checks are the highest-value gap for a coding agent.
5. **Memory is no longer a sharing problem, but a switching one.** The coder takes the whole VRAM budget (14.6 GB on the 16 GB
   card); a purpose model replaces it for the time it runs. What counts is the file size (load time: seconds for every small
   model, [E]), the engine behind the control point and the API per purpose. Embeddings and FIM suffer most from the rule
   because they are only useful while the coder works (batch job or one CPU-side exception, see "Switching").
6. **Product model: one model at a time** (maintainer, 2026-11-15; see "Switching"): purposes are installer entries (coding, general, vision, image, TTS, STT, transcription of recordings, embeddings, FIM, video) and a control point starts exactly one backend.
7. **Recommendation (details at the end):** per purpose and hardware class one recommendation and at most one alternative.
   Inside Tona: general (same engine) and vision (the coder's own encoder, to check). Managed existing engine behind the
   control center: image generation and editing (stable-diffusion.cpp, FLUX.2 klein 4B / Z-Image-Turbo), TTS, STT and
   transcription of recordings (Parakeet v3 / `parakeet-primeline` + Sortformer), embeddings and reranker, FIM. Video: leave out.
   MoE helps only where decode is memory-bound (vision-language MoEs, the MoE STT, MoE TTS backbones); diffusion gains FLOPs only.

8. **UI (final decision): per purpose a small local web app plus `tona` subcommands; no per-purpose TUI.** The server TUI stays
   and shows a login link (OSC 8 hyperlink + plain URL) carrying a one-time token that the server exchanges for an HttpOnly,
   SameSite=Strict session cookie and redirects to a clean URL (Jupyter's scheme, tightened: Jupyter's token is reusable, not
   redirected away and compared with `==` [V]). Weaknesses: cookies are not isolated by port [V RFC 6265], `SameSite` ignores
   ports [V MDN], the server has no Origin/Host check today [V]; see "Access". Order: routes + login, subcommands, web app.

## Reading guide for the hardware classes

Classes follow [docs/hardware/README.md](../docs/hardware/README.md): PC 16 GB VRAM (RX 9060 XT, gfx1200) with 16/32/64 GB
RAM, PC 12 GB VRAM, Macs M1 to M6 with 16 to 64+ GB unified memory. Budgets from the class files: PC 16 GB: about 14.6 GB of
VRAM for the engine (16,304 MiB driver-usable minus 1.4 GB desktop) [V pc-16gb-vram.md]; Mac 32 GB: 26.8 GB GPU working set,
measured; the medium coder tiers take 21 to 23.5 GB, so about 3 GB are left [V mac-32gb.md]. "Fits" below means: the file
(not counting activations) is small enough to be resident beside the coder at the given class, or fits when the coder gives
the memory back for the duration of the job (see "Switching").

## Engine landscape (status on 2026-11-15)

| Engine | Purposes | AMD ROCm / RDNA4 | Apple Metal | Notes |
|---|---|---|---|---|
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | image, edit, video | **yes**: `-DSD_HIPBLAS=ON`, release `master-1104-5dcfa79` (2026-11-14) ships `…Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip` and a Windows ROCm zip; wave32 WMMA flash attention [V docs/build.md, releases API]. gfx1200 appears in issue/PR titles (#1927, #2351, #2353) [V search], no benchmark on gfx1200 found | **yes**, release asset `…Darwin-macOS-arm64.zip` [V] | GGUF; `sd-server` with web UI [V README]. Lists FLUX.1/FLUX.2, Qwen-Image (2.1), Z-Image, SD/SDXL/SD3, Wan 2.1/2.2, LTX-2.3/2.5, HunyuanVideo 1.5 [V README]. **Not** listed: HunyuanImage 3.0, HiDream. Nucleus-Image (MoE): open PR #2315 and issue #1421, not merged [V]. README says "API and command-line option may change frequently" [V] |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | image, edit, video | **yes**: README has `device-gfx1200` for RX 9060 / XT, ROCm 7.2 wheels (and a ROCm 10.0 Windows section) [V]; AMD has a "ComfyUI on Radeon 9000" guide [K, search result title] | yes (README: macOS, Apple silicon) [V] | PyTorch based, Python stack of several GB; node graph, HTTP API |
| [mflux](https://github.com/filipstrand/mflux) (MLX) | image, edit | no | **yes**, native MLX: Z-Image, FLUX.2 4B/9B, FLUX 3 Image Klein, Krea 2, Ming-Image, MOSS-Image, Iris-3B, ERNIE-Image [V README table] | Mac only |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | STT (Whisper only) | **yes** ("AMD ROCm GPU support" section) and Vulkan [V] | **yes**, Metal and Core ML [V] | pushed 2026-11-13; OpenAI-style server example exists [K] |
| [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) | STT, 16 model families (Parakeet, Qwen3-ASR, Voxtral, Granite Speech, Moonshine, Cohere Transcribe, Whisper, Canary-Qwen, diarization, VAD) | **Vulkan**, CPU; ROCm not listed [V README: "Metal, Vulkan, and CUDA backends"] | **yes** (Metal) [V] | ggml based, GGUF, pushed 2026-11-14, 2.2 k stars |
| [zonos2.cpp](https://github.com/Zyphra/zonos2.cpp) | TTS (ZONOS2/3) | **Vulkan** (Linux x64 prebuilt), CPU [V README] | **yes** (macOS arm64, Metal) [V] | `zonos2-server` HTTP server [V] |
| [MLX-Audio](https://github.com/Blaizzy/mlx-audio) | TTS, STT | no | yes (MLX); Qwen3-TTS, Kokoro [V README, partially read] | Mac only |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` | embeddings, reranking, FIM, vision, audio input | yes (ROCm/HIP, Vulkan) | yes (Metal) | `/v1/embeddings`, `/reranking` (`--reranking`), `/infill`, multimodal via `libmtmd` with `--mmproj` for image, audio and video [V server README, docs/multimodal.md] |
| [llama-swap](https://github.com/mostlygeek/llama-swap) | model swapping proxy for any local server | n/a | n/a | "Reliable model swapping for any local AI server", pushed 2026-11-15, 6.3 k stars [V]; a model of what a managed side service needs |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) / SGLang diffusion | image (MiniMax-H3-Image recommended frameworks) | Instinct class (MI350X named); RDNA4 not named [V MiniMax card] | no | server-class, not for these hardware classes |

Our own gap: the Tona server exposes `/v1/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`,
`/health`, `/status` only [V grep of `engine/server`]. No embeddings, no `/infill`, no images, no audio.

## Image generation

| Model | Size on disk (usable quant) | Architecture | License | Fits | Claim, source |
|---|---|---|---|---|---|
| **Z-Image-Turbo** (Alibaba Tongyi) | 6.15B params [V]; GGUF Q4_K_M **3.85 GB**, Q8_0 not listed above Q5 in the rows read; BF16 12.3 GB [V, unsloth GGUF tree] | single-stream DiT, dense, 8 NFEs [V] | Apache-2.0 [V] | 12 GB VRAM yes; 16 GB yes alone; beside the coder only after the coder gives memory back; Mac 16 GB+ yes | "fits in 16 GB", "sub-second on H800", Elo-based human preference result by the authors [V card, vendor claim] |
| **FLUX.2 [klein] 4B** (Black Forest Labs) | 4B; GGUF Q4_K_M **2.60 GB**, Q8_0 4.30 GB [V unsloth tree] (+ text encoder, size not read) | rectified flow transformer, 4 steps, `guidance_scale=1.0` [V] | **Apache-2.0** [V]; the 9B klein is `flux-non-commercial-license` [V tag] | as above | "fits in ~13 GB VRAM" (BF16), "under a second" [V card, vendor claim]; also does editing (pipeline tag image-to-image) [V] |
| **FLUX 3 Image Klein** (Oct 2026) | 5.55B [V]; GGUF Q4_K_M **3.67 GB**, Q8_0 6.09 GB [V unsloth tree], plus a 4B text encoder [V mflux table] | text-to-image and editing with up to 10 reference images [V mflux table] | `flux-kommunity-license` (custom, not read; repository gated) [V tag] | as above | only the mflux table was read; no model card (gated) |
| **Nucleus-Image** (MoE) | 17B total, about 2B active [V]; BF16 repo 51.7 GB including an 8B text encoder (Qwen3-VL-8B) [V]; no GGUF found | sparse MoE DiT, 32 layers, 29 with 64 routed experts + 1 shared, expert-choice routing [V] | Apache-2.0 [V] | 16 GB VRAM only with quantization and offload; **no engine runs it** except PyTorch/diffusers (stable-diffusion.cpp PR #2315 open) [V] | "matches or exceeds leading models on GenEval, DPG-Bench, OneIG-Bench at 2B active" [V abstract]; it is a **base model without post-training** [V card] |
| Qwen-Image-2.1 (Sep 2026) | 7.1B [V]; GGUF Q4_K_M **4.20 GB**, Q8_0 7.64 GB [V] | single-stream DiT, 32 layers, generation + editing + RGBA in one model [V] | **`qwen-research`: non-commercial, "research or evaluation purposes only"** [V LICENSE text] | size fits | the licence is the problem; see [42-model-licenses.md](42-model-licenses.md) |
| Qwen-Image 20B / 2512 | 20.4B [V]; Edit-2511 GGUF Q4_K_M **12.84 GB** [V] | dense MMDiT | Apache-2.0 [V] | 16 GB VRAM only alone; not beside the coder | "exceptional text rendering" [V card] |
| HiDream-I1-Full | 17.1B [V]; BF16 repo 47.2 GB incl. T5-XXL and Llama-3.1-8B encoders [V] | "dual-stream sparse DiT with dynamic MoE", then single-stream sparse DiT [V abstract] | MIT for the transformer, text encoders keep own licences (Llama 3.1 Community) [V] | too large without a quantized engine; not in stable-diffusion.cpp's list | "SOTA quality within seconds" [V abstract, vendor claim] |
| HunyuanImage 3.0 (Instruct, Instruct-Distil) | 83B total, ~13B active [V]; BF16 168.7 GB; community MXFP4 (Intel AutoRound) repo 52.7 GB [V] | autoregressive-multimodal MoE, 64 experts [V] | Tencent Hunyuan Community: **excludes EU, UK and South Korea** [V LICENSE text] | no (>= 53 GB at 4 bit, needs the whole machine) | "largest open image MoE" [V card] |
| MiniMax-H3-Image (Nov 2026) | 33.1B dense DiT, 68.65 GB BF16 plus a 66.7 GB text encoder [V] | dense single-stream DiT, 6 steps, up to 16 MP [V] | `minimax-h3-image-community-license` (custom, not read) [V tag] | no | stable-diffusion.cpp supports it (day-0 news item) [V], Q6/Q8 on ROCm still an open request (#1927) [V] |

**Image editing** is mostly the same models: FLUX.2 klein 4B and FLUX 3 Image Klein edit in the same checkpoint [V],
Qwen-Image-Edit-2511 (20B, Apache-2.0, Q4_K_M 12.84 GB) [V], Qwen-Image-2.1 (research licence) [V], HunyuanImage 3.0-Instruct
(MoE) [V], MiniMax-H3-Image [V]. The smallest usable option is FLUX.2 klein 4B (Apache-2.0).

**MoE verdict for images.** Real, current (Nucleus-Image 2026-03, HiDream-I1 2025-04, HunyuanImage 3.0 2025-09, WeLM-Image
2026-11-05 [V abstracts]); but quality per file size is led in practice by dense 4 to 7B distilled models (Z-Image-Turbo, FLUX.2
klein), and the MoE ones either do not fit (HunyuanImage) or have no engine (Nucleus-Image). Nucleus-Image is the one to
watch: Apache-2.0, 2B active, and ComfyUI/llama.cpp-style ports are in flight (stable-diffusion.cpp PR #2315 open) [V].

**Bound by.** [E] Per step: FLOP = 2 x active params x tokens. Z-Image at 1024 px: 4096 image tokens (8x VAE, 2x2 patch, an
assumption taken from the FLUX-style design, unchecked for Z-Image) x 6.15e9 x 2 = 5e13, plus attention about 8e12, so about
6e13 FLOP. At half of the measured 93 TFLOPS: about 1.3 s per step, 8 steps about 10 s plus text encoder and VAE. Weight
traffic at Q4 (3.85 GB / 315 GB/s) is 12 ms per step. The ridge point is 93e12 / 315e9 = 295 FLOP per byte; a diffusion step
at 4096 tokens sits near 4000, LLM decode at batch 1 near 2. Nucleus-Image per step: 2 x 2e9 x 4096 = 1.6e13 FLOP (about
0.35 s) against about 9 GB of Q4 weights read once per step (29 ms): still compute-bound, and with expert-choice routing
all 64 experts are read in every layer, so a hot-expert cache has nothing to cache. Qwen-Image-2.1 at 1024 px and 40 steps:
about 6e13 x 40 / 46e12 = 50 s [E]; at the 2048 px of its card example 4 times the tokens, minutes. FP8 (186 TFLOPS
measured) could halve these if an engine used it; none of the engines above is known to. No measured seconds-per-image for
gfx1200 with FLUX or Z-Image was found (search result only: SD 1.5/SDXL review figures and a Windows ROCm bug report [K]).
Unchecked: run `sd-server` with Z-Image-Turbo Q4_K_M on the reference box.

## Video generation (briefly)

| Model | Size | MoE? | License | Notes |
|---|---|---|---|---|
| Wan 2.2 T2V/I2V-A14B | 126 GB BF16 repo; GGUF Q4_K_M **two files of 9.65 GB** (HighNoise, LowNoise) [V QuantStack tree] | **yes, but by timestep**: a high-noise expert and a low-noise expert, "separating the denoising process cross timesteps" [V card]; only one runs per step | Apache-2.0 [V] | official note: "at least 80 GB VRAM" for the plain command [V card]; sd.cpp supports Wan 2.2 [V] |
| Wan 2.2 TI2V-5B | 5B | no | Apache-2.0 [V] | the practical one; 8 GB with offload [K] |
| HunyuanVideo 1.5 | 8.3B [K] | no | Tencent community [V tag] | 14 GB with offload [K] |
| LTX-2.3 | n/a | n/a | `ltx-2-community-license-agreement` [V tag], paid above USD 10M revenue [K] | updated 2026-10-22 |

Video is compute-bound like images, with 10s to 100s of times more tokens per step; on a 16 GB card it means minutes to
hours per clip [E, no measurement]. **Leave out.** Nothing about it helps a coding agent.

## Speech to text (dictation, live input)

| Model | Size | MoE? | License | Fits | Claim, source |
|---|---|---|---|---|---|
| **Parakeet-TDT-0.6B-v3** (NVIDIA) | 0.63B; GGUF Q8_0 **0.74 GB**, Q5_K_M 0.55 GB [V handy-computer GGUF] | no | CC-BY-4.0 [V] | everywhere, CPU is plausible | "speed option"; long-form WER 13.37 in NVIDIA's own table [V Nemotron card] |
| **Qwen3-ASR-1.7B** | 2.35B incl. audio encoder [V]; GGUF Q4_K_M **1.32 GB**, Q8_0 2.19 GB [V] | no | Apache-2.0 [V] | everywhere | English mean WER 4.94 on the Open ASR set, per NVIDIA's table [V]; 52 languages and dialects [V] |
| **Whisper large-v3-turbo** | 0.81B; GGUF Q5_K_M **0.62 GB** [V] | no | MIT [V] | everywhere | no longer the accuracy leader; most deployed [K] |
| **Nemotron 3.5 Transcribe 30B-A4B** (NVIDIA, 2026-10) | 33.7B total, 4.3B active incl. encoder [V]; third-party GGUF **13.6 GB** (Q2_0_M) to 19.7 GB (IQ4_NL) [V AtomicChat tree] | **yes**: Mamba-2/Transformer hybrid MoE speech-augmented LM [V] | OpenMDW-1.1 [V] | 16 GB VRAM at 4 bit; no, beside the coder | English mean WER **4.71** (best listed, 2026-10-12), multilingual 4.62, long-form 10.98, all NVIDIA's own table [V]; 40 languages, 1 h recordings [V] |
| Voxtral-Mini-4B-Realtime-2602 (Mistral) | 4.4B [V] | no | Apache-2.0 [V] | 8 to 16 GB class | streaming; in transcribe.cpp [V] |
| Cohere Transcribe 03-2026, Granite Speech 4/5 | 2.1B / 2.3B / 0.47B [V] | no | Apache-2.0 [V] | everywhere | in transcribe.cpp [V]; Granite 5.0 TurboCTC 470M is non-autoregressive [V names] |

**Bound by.** [E] Whisper-style models run a compute-heavy encoder over 30 s windows (about 1500 frames) and a small
bandwidth-bound decoder; Parakeet/CTC/TDT models are non-autoregressive or nearly so, so a minute of audio costs well under
a second of GPU time on any listed card. Nemotron Transcribe is an LLM-style decoder (4.3B active): the same bandwidth-bound
regime as the coder, relevant only if Tona grows a Nemotron-H (Mamba-2 hybrid) backend; its architecture class is the one
of Nemotron 3.5 Lightning 30B-A3B [V base_model in the card], which [49-moe-models-2026-11.md](49-moe-models-2026-11.md)
lists as a coding candidate, so one implementation could serve both. Unchecked: whether any engine runs the 30B-A4B
checkpoint outside vLLM/NeMo (the community GGUF has no stated engine).

**Engines.** AMD: whisper.cpp (ROCm, Vulkan) for Whisper; transcribe.cpp (Vulkan) for the rest. Mac: both with Metal; Whisper
also Core ML. A coding agent needs dictation and not diarization; **Parakeet-TDT-0.6B-v3 or Qwen3-ASR-1.7B** are enough and
small enough to run next to anything.

## Transcription of recordings and video (its own purpose, next to dictation)

**Decision: separate purpose, not merged with the speech-to-text item above.** Reasons: (1) different models win (long-form
accuracy, language coverage and timestamps instead of latency and streaming), (2) it needs a pipeline around the model (audio
extraction from video, chunking or windowed attention, speaker diarization, output formats), (3) the API differs (a file in,
a document with segments out, minutes of runtime, instead of a short utterance), (4) the speed number that matters is batch
throughput as a multiple of real time. The installer entry "Speech to text" stays for dictation/streaming; "Transcription
(recordings, video)" is its own entry. The two share model files (Parakeet, Qwen3-ASR, Whisper), so installing one model can
serve both.

**Pipeline.** Video files: extract mono 16 kHz audio first. Checked here with the installed ffmpeg 8.1.3: `ffmpeg -i in.mp4 -vn
-ac 1 -ar 16000 -c:a pcm_s16le out.wav` produced the expected 16 kHz mono WAV [V, 2 s test file]. transcribe.cpp's diarizer states
the same input contract ("16 kHz mono WAV") [V]. Then: (voice-activity detection, optional) -> ASR with timestamps -> diarization
-> merge speaker segments with word/token timestamps -> SRT/VTT/JSON. ffmpeg is an external dependency (a system package, or
shipped with the control center); it is not part of any model engine.

**Models for recordings** (sizes of the GGUF in transcribe.cpp or the HF file tree [V]):

| Model | Size | Long files | Timestamps | German | Diarization | License |
|---|---|---|---|---|---|---|
| **Parakeet-TDT-0.6B-v3** | Q8_0 740 MB [V catalog] | "no practical per-call length limit": windowed local attention in transcribe.cpp, a 25 min talk runs on a 16 GB laptop, multi-hour files fit; WER on full TED-LIUM 3 talks (10 to 25 min) 2.87 full attention vs 2.71 windowed [V transcribe.cpp doc] | word and segment [V NVIDIA card]; token timestamps in the port [V] | yes, 25 European languages incl. German [V card] | none built in | CC-BY-4.0 [V] |
| **`parakeet-primeline`** (German fine-tune of v3 by primeLine) | Q8_0 740 MB [V] | same | token timestamps [V] | **FLEURS German WER 5.98 %** (NeMo reference on the same manifest 5.98 %) [V transcribe.cpp doc]; the v3 base number on German was not extracted | none | CC-BY-4.0 [V HF tag] |
| **Whisper large-v3-turbo** (whisper.cpp / transcribe.cpp) | Q5_K_M 0.62 GB [V] | 30 s windows, chunked by the engine; hallucination and repetition on silence is a known failure class [K] | segment timestamps; word-level "experimental" (`-ml 1`) [V whisper.cpp README] | yes (99 languages [K]) | `tinydiarize` "experimental" [V README] | MIT [V] |
| **Qwen3-ASR-1.7B** + Qwen3-ForcedAligner-0.6B | Q4_K_M 1.32 GB (ASR) [V] | forced aligner "up to 5 minutes of speech" per call [V card]; longer files need chunking | via the aligner, 11 languages incl. German [V] | yes [V] | none | Apache-2.0 [V] |
| **Nemotron 3.5 Transcribe 30B-A4B** (MoE) | 13.6 to 19.7 GB (third-party GGUF) [V] | "recordings up to one hour", 65,536-token context [V card] | timestamp prediction [V] | yes, in its 40 languages [V tag `de`] | **speaker-attributed transcription**, multi-speaker cpWER 15.80 on AMI/CALLHOME/Mixer 6/NOTSOFAR-1 vs 16.83 for DiCoW + DiariZen [V NVIDIA table] | OpenMDW-1.1 [V] |
| Granite Speech 4.1 2B / MOSS-Transcribe-Diarize | in transcribe.cpp [V] | not read | "diarize, word timestamps" listed as capabilities [V] | not read | listed [V] | Apache-2.0 (Granite) [V tag] |

**Diarization models** (who spoke when; they emit segments, no text):

| Model | Size / runtime | Speakers | License |
|---|---|---|---|
| **Streaming Sortformer 4spk v2.1** (NVIDIA) in transcribe.cpp (GGUF, Metal/Vulkan/CPU) | not read | **hard cap of 4 concurrent speakers**, numbered by first appearance [V doc] | `nvidia-open-model-license` [V tag] |
| **pyannote speaker-diarization community-1** | PyTorch, Python | not read | **CC-BY-4.0** [V tag]; 4.9 M downloads [V]; the model would run as a Python child, not ggml |
| Nemotron 3.5 Transcribe (built-in) | see above | speaker-attributed output [V] | OpenMDW-1.1 |

The honest weak point: no single small ggml model does ASR **and** diarization with word timestamps for German. The practical
chain is Parakeet (primeline for German) + Sortformer for up to 4 speakers, or the heavy MoE Nemotron Transcribe (speaker
attribution built in, 14 to 20 GB, needs an engine that does not exist for us: see the STT section). Meetings with more than
4 speakers need pyannote (Python) or the MoE model. Alignment of speaker segments to words is our own merge code (small).

**Languages incl. German.** Parakeet v3: 25 European languages with German [V]. Qwen3-ASR: 30 languages and 22 Chinese dialects
with German [V]. Whisper: broad coverage [K]. German-specific: `parakeet-primeline` [V]. Nemotron Transcribe: 40 languages [V].

**Speed as multiple of real time (x RT).** No measurement for our hardware classes exists in the sources read; the
transcribe.cpp docs state numbers "from single Metal-backed runs" for accuracy only. Own estimates [E] (compute ceilings, then
a derating): a FastConformer encoder at 8x subsampling sees 12.5 frames per second of audio (assumption from the NeMo design,
unchecked here); 2 x 0.6e9 x 12.5 = 1.5e10 FLOP per audio second, which at 20 TFLOPS (about 20 % of the measured 93 TFLOPS)
is 0.75 ms, a ceiling near **1000x RT** on the reference GPU; the TDT decoder adds little. With ggml kernels and a realistic
derating, expect **100 to 300x RT** (a 1 h recording in 12 to 36 s, plus 2 to 3 s model load). Whisper turbo: encoder about 0.64B
at 50 frames per second = 6.4e10 FLOP per audio second, ceiling near 300x RT, plus an autoregressive decoder of about 3 to 4
tokens per second of speech: expect **50 to 150x RT**. On the CPU (Zen 5, 6 cores): the encoder FLOPs are about 40 to 90 times
lower in throughput than the GPU [E, from the CPU's assumed 0.5 to 1 TFLOPS], so a 0.6B model lands near **5 to 20x RT**
(whisper.cpp's own docs call the smaller Whisper models "near-realtime on CPU" [V transcribe.cpp whisper doc]); Macs: no
TFLOPS figure exists in the repo's Mac files, so no estimate is made (to measure). The MoE Nemotron Transcribe at 4.3B active
is bandwidth-bound in the decoder: at a Q4 file of 20 GB with the expert arena, a transcript of 1 h (about 9,000 words, 12,000
tokens [E]) costs 12,000 passes: at the coder's 60 passes/s that is about 200 s, so roughly **15x RT**, no better than the CPU
Parakeet. Everything here is unmeasured; the check in "Open checks" decides.

**Switching and API.** Same one-at-a-time rule: transcribing a recording is a job (seconds to minutes) with a clean start and
end, which fits the model well. Endpoint: `/v1/audio/transcriptions` (OpenAI; supports `response_format` verbose_json with
segment and word timestamps and `srt`/`vtt` [K, from the OpenAI API definition]; diarization is not part of that API, so
speaker labels would be an extension field). Engines: transcribe.cpp or whisper.cpp behind the control point, plus an adapter
for the route (see "Switching", table). Size cap: video files can be GBs; the upload path must stream to disk, and the control
point calls ffmpeg before the engine.

## Text to speech

| Model | Size | MoE? | License | Claim, source |
|---|---|---|---|---|
| **Kokoro-82M** | 82M, `kokoro-v1_0.pth` 0.33 GB [V] | no | Apache-2.0 [V] | 12.7 M downloads [V]; tiny, CPU-feasible [E, unmeasured] |
| **Qwen3-TTS 12Hz 0.6B / 1.7B** | 1.9B (CustomVoice) [V]; GGUF 1.7B Q4_K_M **1.22 GB**, Q8_0 2.08 GB [V Serveurperso tree] | no | Apache-2.0 [V] | on MLX-Audio [V] |
| **ZONOS3-Mini** (Zyphra, 2026-11-09) | 800M total, ~100M active; **latent MoE, 64 experts, top-2**, 11 layers [V card] | **yes** | Apache-2.0 [V] | "real time on phones and laptops", WebGPU in the browser; time to first audio 80 ms at 8 concurrent requests on an H100 [V card, vendor claim]; zonos2.cpp covers CPU, Vulkan, Metal [V] |
| ZONOS2 | 8B total / 900M active [K: web summary; card says "MoE", numbers not in the text read]; GGUF Q4_K **4.92 GB**, Q8_0 8.54 GB [V] | **yes** | Apache-2.0 [V] | Tier 1 languages English, Mandarin, Japanese [V] |
| Ming-3.0-tts-tiny (Ant Group) | 7.9B total, 1.3B active, 9.0B params in the repo [V] | **yes** | MIT [V] | MoE on a Ling V3 backbone [V] |
| Voxtral-4B-TTS-2603 | 4B | no | **CC-BY-NC-4.0** [V] | not usable |
| MOSS-TTS Local Transformer | 3.06B [V] | no | Apache-2.0 [V] | |

**Bound by.** AR codec TTS emits about 12 frames per second of speech and decodes one frame per pass: bandwidth-bound at batch
1, like LLM decode; a 100M-active MoE is tiny (a few hundred MB read per pass at Q4 [E]) and is limited by launch and
vocoder cost rather than by weights. **Value for a coding agent is low** (reading answers aloud). If wanted, ZONOS3-Mini or
Kokoro via an existing CPU/Vulkan/Metal engine; no GPU needed.

## Embeddings and rerankers for code search

| Model | Size | License | Claim, source |
|---|---|---|---|
| **Qwen3-Embedding-0.6B** | 0.60B; GGUF Q8_0 **0.64 GB** [V Qwen GGUF repo] | Apache-2.0 [V] | 32K context, 1024-d, instruction-aware; MTEB multilingual v2 mean 64.33 (0.6B), 69.45 (4B), 70.58 (8B) [V card, vendor] |
| Qwen3-Embedding-4B / 8B | 4B / 7.57B [V] | Apache-2.0 [V] | as above; 8B is a dense model of about 4 GB at Q4 [E] |
| **EmbeddingGemma 2** (Google, 2026-09) | 740M total, text backbone 270M, vision 170M and audio 300M loadable separately [V] | Apache-2.0 (tag) [V] | **MTEB code v1: 78.68** vs 68.76 for its predecessor (as read from the card table), 8K context, "about 14 % better on code" [V card, vendor] |
| **Qwen3-Reranker-0.6B** / 4B / 8B | 0.6B; GGUF Q8_0 (ggml-org) [V repo exists] | Apache-2.0 [V] | MTEB-Code rerank score 73.42 (0.6B) vs 81.22 (8B), over the card's retrieval pool [V card, vendor] |
| nomic-embed-code | 7.07B [V] | Apache-2.0 [V] | built on Qwen2.5-Coder-7B [V] |
| CodeRankEmbed | 137M [V] | MIT [V] | code retrieval, small |
| jina-code-embeddings 0.5B / 1.5B | 0.49B / 1.54B [V]; GGUF IQ4_NL **0.35 GB** [V] | **CC-BY-NC-4.0** [V] | not usable for a free project that others may redistribute without checking |
| nomic-embed-text-v2-moe | 475M, **MoE** [V] | Apache-2.0 [V] | multilingual text, not code-specific |

MoE here is irrelevant: the useful models are 0.1 to 0.7B and read in a few ms. **Bound by:** [E] indexing a repository is
compute-bound batch work (tokens x 2 x params: 1 M tokens x 1.2e9 FLOP = 1.2e15 FLOP = about 25 s at 46 TFLOPS on the GPU,
about 20 min on the CPU at an assumed 1 TFLOPS); a query is one short sequence and negligible. llama.cpp `llama-server`
already serves `/v1/embeddings` and `/reranking` on ROCm, Vulkan and Metal [V]. Qwen3-Embedding is a plain dense Qwen3
transformer; whether `llama.cpp` needed special bidirectional attention handling is visible in its PR #30607 "qwen3:
bidirectional attention and dense_2 output projection" [V title], i.e. an own implementation in Tona would need that change.

## Fill-in-the-middle (FIM) autocomplete and next-edit

| Model | Size | License | Notes |
|---|---|---|---|
| **Qwen2.5-Coder-1.5B** (base) | GGUF Q8_0 **~1.6 GB** (Q2_K 0.75 GB listed for the instruct repo) [V] | Apache-2.0 for 1.5B and 7B; **3B is `qwen-research`** [V tags] | FIM tokens in the tokenizer [V]; the model card names FIM as a use of the base model [V] |
| **Sweep Next-Edit 1.5B** | GGUF Q8_0 **1.54 GB** [V] | Apache-2.0 [V] | predicts the next edit, "under 500 ms with speculative decoding" on a laptop, "outperforms models over 4x its size" [V card, vendor claim]; 8K context |
| Zed `zeta` | not read | not read | 519 k downloads [V listing only] |
| Mellum2 12B-A2.5B (JetBrains) | 12.1B, **MoE** (64 experts, 8 active) [V] | Apache-2.0 [V] | base model has the FIM tokens [V]; Q4_K_M GGUF exists in the Instruct repos [V listing]; 12B at Q4 is about 7 GB [E] |
| Qwen3-Coder-Next / Qwen3-Coder-30B-A3B / Qwen3.5-35B-A3B | 80B/3B, 30B/3B, 35B/3B | Apache-2.0 [V] | **their tokenizers define `<fim_prefix/middle/suffix/pad>`** [V tokenizer_config] |
| **Our own coder** `Qwen3.8-Flash-Next` (base of the ISTA GGUF) | 180B total | `qwen-community-1.0` [V] | tokenizer also defines the four FIM tokens [V]. Whether the model was trained on FIM and whether the IQ1_M quant still does it well is **unchecked** (check: 200 infill prompts through the engine, exact-match against the base model) |

Tona already runs the model that could do FIM: with the FIM tokens in the vocabulary, a `/infill` route on the existing
server needs no second model, only a route and a latency budget. But the coder decodes at about 89 tok/s with MTP [V
reference platform] and a fresh prefill of a 2K-token editor context takes seconds on the large tier, while editor
autocomplete wants under 300 ms [K as common practice]; a 1.5B dense model prefills 2K tokens in about 0.1 s on the GPU
[E: 6e12 FLOP / 46e12] but it needs the GPU while the megakernel owns it (see below). The llama.cpp client side exists:
`llama-server /infill` [V] and the `llama.vscode` extension, pushed 2026-11-13 [V]. VS Code itself "cannot connect to a local
model for inline suggestions" as of August 2026 [K, secondary quote], so a third-party extension is required.

## Vision-language for screenshots and UI

| Model | Size | MoE | License | Notes |
|---|---|---|---|---|
| **Our coder's own vision encoder** (`Qwen3.8-Flash-Next`: `vision_config`, `image_token_id`, `Qwen4ExpForConditionalGeneration`) | not in the ISTA GGUF; `mmproj` absent [V] | the coder is MoE | `qwen-community-1.0` [V] | the first thing to check: can the vision tower be converted to a small mmproj and fed into our engine? sizes unchecked (the sibling Qwen3.5-35B-A3B has a 1.79 GB F32 mmproj [V unsloth tree]) |
| **Qwen3.5-35B-A3B** | Q4_K_M **22.0 GB** + mmproj 1.79 GB [V] | yes, `qwen35moe` family | Apache-2.0 [V] | image-text-to-text; the architecture Tona already runs ("small" tier Ornith-1.5-35B-A3B is the same arch, #143) |
| **Holo3.1** (H company, Qwen 3.5 based) | 0.8B, 4B, 9B, **35B-A3B**; 4B GGUF Q4_K_M **3.07 GB**, 35B Q4_K_M **21.3 GB** [V] | the 35B yes | Apache-2.0 [V] | computer-use and UI-grounding models; numbers are in images on the card (not read) |
| Qwen3-VL-8B | GGUF Q4_K_M **5.03 GB** + mmproj Q8_0 0.75 GB [V] | no | Apache-2.0 (Qwen3-VL-30B-A3B is Apache-2.0 [V]) | generic |
| Qwen3-VL-30B-A3B, Gemma-4-26B-A4B | 31B / 25.8B total [V] | yes | Apache-2.0 [V] | |
| UI-TARS-1.5-7B | 7B | no | not read | 228 k downloads, GGUF exists [V listing] |

**Bound by.** The vision encoder is a ViT: compute-bound prefill, fixed cost per image (hundreds to a few thousand image
tokens), then ordinary decode of the language model. The ViT of 0.4 to 1.5 GB is tiny against the decoder. If the coder itself
reads the image, **no second resident model is needed**: only the encoder (a few hundred ms to a few seconds per screenshot
[E, unmeasured]) and the image tokens in the context (KV cost: each screenshot spends context). llama.cpp supports image,
audio and video input through `libmtmd` on ROCm and Metal [V docs/multimodal.md]. Our server currently drops images [V].

## Switching: one model at a time

Product (maintainer's correction, 2026-11-15): the installer asks for a purpose first (coding, general, image generation,
text-to-speech, speech-to-text, ...), shows the models for it per hardware class (one recommendation, at most one
alternative, with strengths, weaknesses and expected speed), and the user may install several. Later, in the server or
control center, he picks one installed model and starts it. **One model at a time, switched freely.** There is no co-residency
requirement, so the whole question of VRAM sharing with a running coder (and the GPU-safety risk of a second process beside
the megakernel) disappears. What switching needs instead:

**1. Start and stop time.** [E] Load time is bounded by reading the file: reference NVMe 2.65 GB/s sequential [V
reference-platform.md], Mac SSD 13 GB/s for 2 MB blocks on the M1 Max [V mac-32gb.md]. A Z-Image-Turbo Q4_K_M (3.85 GB) is about
1.5 s cold on the PC and 0.3 s on the Mac, a 0.6 GB embedding model 0.3 s, a 22 GB Qwen3.5-35B-A3B Q4_K_M about 8 s, the 58 GB
large coder about 22 s plus arena fill and pinning (unmeasured; the engine's own start time is a number we have to read from
the start log). Warm page cache is faster. Process start, GPU init and kernel compile come on top and are unmeasured. One
known first-run cost: stable-diffusion.cpp benchmarks hipBLASLt candidates for each INT8 matrix shape on first use and caches
them in the process only [V docs/build.md], so every start repeats it. **Stop = process exit**, the only release that is known
to give back all VRAM and pinned RAM (`tona stop` already does this [V README]); engines with in-process unload (vLLM sleep
mode, CUDA only [V research/44]) do not apply to ROCm or Metal. Plan: the switch is "stop A, wait until
`mem_info_vram_used` is back at the desktop's 1.4 GB, start B" and takes a few seconds for every small model.

**2. One runtime or several engines behind one control point.** No single runtime covers all purposes: the coder family runs in
Tona's own backend (megakernel, expert arena); diffusion, STT and TTS run in ggml-based or PyTorch/MLX engines. The realistic
design is a **control point (Tona's control center) that owns the model catalogue, the single GPU lease and the lifecycle of
exactly one backend process**, and exposes one HTTP port with one route set per purpose. Behind the port:

| Purpose | API a client expects | Backend process | API of that backend today |
|---|---|---|---|
| Coding, general | `/v1/chat/completions`, `/v1/messages` | Tona engine | present [V] |
| Embeddings | `/v1/embeddings` | `llama-server --embedding` | present [V server README] |
| Reranking | `/v1/reranking` | `llama-server --reranking` | present [V] |
| FIM | `/infill` (llama.vscode convention) | `llama-server` | present [V] |
| Image generation and editing | `/v1/images/generations`, `/v1/images/edits` (OpenAI) | `sd-server` (stable-diffusion.cpp) | README states an OpenAI-compatible `/v1/...` API plus a WebUI-compatible and a native async API [V]; which image routes exactly is **unchecked** |
| Text to speech | `/v1/audio/speech` | `zonos2-server` | OpenAI-compatible `/v1/audio/speech` [V zonos2.cpp README]; `llama-server` also has `/v1/audio/speech` [V server README] |
| Speech to text, transcription of recordings | `/v1/audio/transcriptions` (+ ffmpeg audio extraction and a diarization step for recordings) | transcribe.cpp or whisper.cpp | whisper.cpp's server uses `/inference` [V]; no OpenAI-style transcription route found for it or for transcribe.cpp (**a thin adapter in the control point is needed**; unchecked for transcribe.cpp's own server) |
| Vision | image parts in `/v1/chat/completions` | Tona engine (if the coder's own encoder) or `llama-server --mmproj` | engine drops images today [V] |

All the engines in the table except Tona's backend are built on ggml and ship as prebuilt binaries for Linux ROCm/Vulkan and
macOS Metal (stable-diffusion.cpp ROCm and Metal assets [V releases]; zonos2.cpp Vulkan and Metal [V]; whisper.cpp ROCm and Metal
[V]; transcribe.cpp Vulkan and Metal [V]). Several binaries of different release cadence mean several pins in
`toolchain/models.toml`-style (version, SHA-256), several licences to carry and several update paths; the control center needs a
per-engine health check, a port per child and a proxy. ComfyUI and mflux are Python stacks of several GB: not suitable as managed
children; mflux is the only Metal-native route for FLUX 3 Image Klein, Krea 2 and others [V], so on Mac it stays an optional
alternative to stable-diffusion.cpp.

**3. Where each purpose lives (a/b/c).** (a) inside Tona: needs the model family in our backend (gates in
[AGENTS.md](../AGENTS.md): measured, prior art, 500-line files). (b) existing engine as managed child behind the control
point: inherits the engine's bugs and its "API may change frequently" (stable-diffusion.cpp [V]), needs the download, licence
check ([42-model-licenses.md](42-model-licenses.md)) and hash pin. (c) leave out.

| Purpose | (a) inside Tona | (b) managed engine | (c) leave out |
|---|---|---|---|
| General (chat LLM) | **yes, same engine and families as the coder** | none needed | no |
| Vision-language | cheap if the coder's vision tower exists as loadable weights | llama.cpp `--mmproj` | no: images are dropped today |
| Embeddings / reranker | dense Qwen3 0.6B, new kernels + routes | `llama-server` | possible |
| FIM / next-edit | `/infill` route on the coder (FIM tokens present [V]) | `llama-server /infill` | possible |
| STT, transcription | no | transcribe.cpp / whisper.cpp + adapter + ffmpeg | possible |
| TTS | no | zonos2.cpp or `llama-server` | possible |
| Image generation / editing | no (compute-bound, new kernels) | stable-diffusion.cpp | possible, but suits the one-at-a-time model best of all (it wants the GPU alone anyway) |
| Video | no | stable-diffusion.cpp supports Wan 2.2 [V] | leave out |

**4. What one-at-a-time costs the purposes.** Embeddings, reranking and FIM are only useful **while** an agent or editor is
working, which is when the coder runs; under the rule they become batch jobs (build the repository index, then switch back to
the coder) or need an exception (a small CPU-side model beside the coder). Query-time retrieval without a resident embedding
model is not possible. Image, TTS and STT jobs fit the rule well (short, exclusive, GPU-hungry or tiny). Decision for the
maintainer: whether a CPU-only side process for embeddings and FIM is allowed as the one exception; in the RAM budget of the
PC classes this costs 0.6 to 1.6 GB (the 16 GB RAM class runs with `MemoryMax=12G` [V pc-16gb-vram.md]).

**5. Work programs first still applies** to the one running model: the elastic shrink of research/44 (PSI on Linux, memory
pressure source on macOS) is the same for any backend process; for a child engine it is `MemoryHigh`/`MemoryMax` on its unit
and an exit on pressure [research/44, design]. None of the engines found has a pressure-driven unload [V research/44].

## Footnote: the server TUI and terminal image display (dropped)

Final UI decision (maintainer, 2026-11-15): where a purpose has the web app there is **no separate TUI** for it. Per purpose: (1)
the small local web app, reached by a login link that the server TUI shows, and (2) `tona` subcommands for direct, scriptable
use by agents and the shell. The existing server TUI (`tona-top`) stays as it is: status, model start/stop, and the link. What
this research found before the decision, kept short because it no longer drives anything:

- `tona-top` is hand-written ANSI without a library (raw `termios`, alternate screen, full-frame redraw every 250 ms) and has no
  image code; the installer uses FTXUI v7.0.3, whose README lists a Braille canvas but no graphics protocol [V `engine/tools/
  tona_top*.cpp`, `engine/CMakeLists.txt` line 431, `installer/CMakeLists.txt`, FTXUI README].
- Terminal image display exists as the kitty graphics protocol (PNG passes through unchanged, `f=100`) and iTerm2 inline images
  (OSC 1337), plus sixel (own encoder needed) [V kitty docs, iTerm2 docs]. Support: kitty, Ghostty, WezTerm, iTerm2 (and Konsole
  for kitty graphics) yes; sixel in WezTerm, iTerm2, Konsole, foot, xterm, tmux (if built with it); **none in GNOME Terminal
  (VTE), Alacritty and macOS Terminal.app** [V Are We Sixel Yet, kitty docs, Ghostty docs, WezTerm docs]. A terminal preview
  would therefore always need a "save the file and open it" fallback, which is the reason the web app displays media instead.
- Audio playback from a terminal needs an external player (`pw-play`, `paplay`, `aplay`, `ffplay` exist here [V `which`];
  `afplay` on macOS [K]); the web app has `<audio>`. File picking in a terminal is a path argument or a path input; the web app
  has a file drop.

## Local web app, `tona` subcommands and the shared backend API

Shape per purpose (maintainer's final decision, 2026-11-15): **(1) a small local web app** served by `tona-server`, opened from a
login link that the existing server TUI shows (and by `tona ui`), localhost only, no external CDN at runtime; **(2) `tona`
subcommands** for direct, scriptable integration (agents, shell). No separate per-purpose TUI. Both are thin clients of the
same backend endpoints, so the endpoints are the real deliverable.

### What each does better, per purpose

| Purpose | Web app | `tona` subcommands |
|---|---|---|
| Control (pick an installed model, start, stop) | model catalogue with sizes, strengths, weaknesses per class (the installer's list), install progress | `tona start <model>`, `tona stop`; the server TUI keeps status and the link |
| Image generation / editing | images display in every browser; zoom, history grid, drag and drop of up to 10 reference images [V], mask painting for edits, copy to clipboard | `tona image "prompt" -o x.png`, `tona image edit in.png "prompt" -o out.png`: pipelines, agents generating assets |
| Transcription of recordings | file drop for GB-sized files, `<video>`/`<audio>` preview with **seek by clicking a transcript line**, speaker renaming, text correction, live progress, SRT/VTT/JSON download | `tona transcribe file.mp4 --lang de --diarize --format srt -o out.srt`: batch, cron, agents; progress on stderr |
| Text to speech | built-in `<audio>` player, voice preview, no audio dependency in our binary | `tona speak "text" -o x.wav`, `--play` through a system player (`pw-play`/`paplay`/`ffplay`; `afplay` on macOS [K]) |
| Dictation (STT) | microphone capture in the browser (`getUserMedia` needs a secure context; `http://127.0.0.1` counts as one [K], unchecked per browser) | `tona transcribe -` from stdin |
| Embeddings, reranking, FIM, vision, general, coding | none (they serve other programs or the agents) | the OpenAI-compatible routes and `tona claude` / `tona opencode` |
| Remote / headless | needs a port forward and the login link | works over SSH as is |
| Cost to build and gate | **HTML/CSS/JS is a new language in a repo whose gates cover C++, Python and shell** [V `AGENTS.md` gate table, "no exception lists"]: a JS formatter and linter must become gates; no npm build chain (plain ES modules, vendored files with pinned hashes like `engine/third_party`) | bash plus `curl` in `scripts/tona` for the first version, a C++ client later (cpp-httplib is vendored [V]); testable by `ctest` |

### Access: a one-time login link, exchanged for a session cookie (Jupyter style)

**Prior art.** Jupyter Server logs "tokenized URLs" at start; visiting one **sets a cookie**, and the token is also accepted as
`Authorization` header or form field; the token is **reusable, not one-time**, and token auth is on by default because access
means arbitrary code execution [V Jupyter Server docs]. Reading its login handler [V `jupyter_server/auth/login.py`]: the cookie
holds a random `uuid4().hex`, not the token; `HttpOnly` defaults to true; `Secure` only on HTTPS; **`SameSite` is not set** unless
configured; the token is compared with plain `==` (not constant time); for `?token=` the handler **sets the cookie and continues
without redirecting, so the token stays in the address bar and history**; the login form POST keeps it out of the URL. Our
design keeps the idea and tightens those points. (The MCP specification gives the matching rule for local HTTP servers: validate
`Origin` against DNS rebinding, bind to localhost, authenticate every connection [V MCP spec 2025-06-18, transports].)

**Facts about `tona-server` today** [V `server_main.cpp`, `routes_status.cpp`, `SECURITY.md`, grep]: binds `127.0.0.1` by default;
an optional API key (`--api-key-file`) is compared in constant time, **but only for paths under `/v1/`**; `/status` and
`/health` are open; no `Origin` or `Host` validation, no CORS headers; cpp-httplib v0.51.0 (MIT) vendored.

**Design [E].**

1. **New secret per server start**: a login token of at least 128 bits from the OS random source, held in memory; sessions die
   with the process. A restarted server therefore invalidates old links and old cookies (the page shows "session ended, run
   `tona ui`").
2. **The server TUI shows the complete link**: `http://127.0.0.1:<port>/ui/login?token=<token>` as an **OSC 8 hyperlink**
   (`ESC ] 8 ; ; URI ESC \` text `ESC ] 8 ; ; ESC \`), with the plain URL printed next to it as well, because "there's no way of
   detecting whether the terminal emulator supports hyperlinks" and a link must not be the only way to the target [V OSC 8 spec
   by E. Bortolotti, gist]. Support [V Alhadis' OSC 8 adoption table]: yes in VTE/GNOME Terminal (0.50+), kitty, Ghostty, WezTerm,
   iTerm2, Alacritty, foot, Windows Terminal; **Konsole partial ("disabled by default")**; **tmux partial** (needs `set -ga
   terminal-features "*:hyperlinks"`); macOS Terminal.app and xterm not listed. The URI must be ASCII 32 to 126 and under about
   2000 bytes (VTE and iTerm2 limit it to 2083) [V]: a hex token is fine. `tona ui` opens the same link directly with `xdg-open`
   (Linux) or `open` (macOS), without a click.
3. **Login handler** `GET /ui/login?token=T`: constant-time compare (as the existing `authorized()` does [V]); on success the token
   is **consumed** (one-time) and the server answers **`303 See Other` to `/ui/`** with
   `Set-Cookie: tona_session=<256-bit random, not the token>; HttpOnly; SameSite=Strict; Path=/` (no `Domain` attribute, so it is
   host-only; no `Expires`, a browser-session cookie; also `Max-Age` on the server's own session table), plus `Cache-Control: no-store`.
   The address bar then shows the clean URL; whether the browsers also keep the one-time URL in history is unchecked (browsers
   normally record the redirect target; test in Firefox, Chrome, Safari).
4. **`Referrer-Policy: no-referrer` on every response** of the server, so the link never leaks through a Referer header.
5. **Sessions in a server-side table**, idle timeout (for example 12 h), `POST /ui/logout`. `tona ui` or the TUI can mint a
   **fresh one-time link** at any time (the TUI refreshes its link line after each use), so a consumed link costs one command.
6. **Cookie authenticates, so CSRF matters**: validate `Host` (exactly `127.0.0.1:<port>` or `localhost:<port>`; defeats DNS
   rebinding) and `Origin` (absent, or exactly `http://127.0.0.1:<port>`) on every request; require a per-session **CSRF
   header** on state-changing requests (the page reads the value from `GET /ui/session`, which another origin cannot read);
   accept only `application/json` and `multipart/form-data`. Third-party API clients keep using `Authorization: Bearer <api key>`
   on the `/v1/` routes.
7. **No external resources**: all JS, CSS, fonts and icons ship in the release; `Content-Security-Policy: default-src 'self'`;
   no analytics, no web fonts from a CDN. Media arrives as uploads streamed to disk with a size limit; the server never reads a
   path that the browser names; `ffmpeg` and decoders run as limited child processes on untrusted files.
8. **Alternative link delivery without the token in argv**: `tona ui` can write a 0600 temporary HTML file that auto-submits a
   `POST /ui/login` form (the form-POST variant Jupyter's login page uses [V]) and open that file; the token then never appears in
   the browser's command line.

**Weaknesses of this scheme (to accept or mitigate; all unchecked in practice).**

- **Cookies are not isolated by port** ("Cookies do not provide isolation by port" [V RFC 6265, section 8.5]): the browser sends
  `tona_session` to every service on the same host name, whatever its port, and any of them can overwrite or delete it. HttpOnly
  hides the cookie only from page JavaScript, not from a server on another port that receives the request. A local dev server or a
  malicious `npm` script therefore sees a valid session id and can replay it against the Tona port (the Origin/CSRF gate of point
  6 does not stop a non-browser client that has the id). Treat every local web server on the same host name as inside the trust
  boundary of the session, or bind the session to a client fingerprint that a replay cannot copy cheaply (unchecked what is
  practical). A cookie name prefix `__Host-` is no way out: it requires `Secure`, and an `http:` page cannot set a `Secure`
  cookie; localhost is an exception only in some browsers and **not in Safari** [V MDN Cookies].
- **`SameSite=Strict` does not separate ports either**: "same-site" is decided by the registrable domain and the scheme, not the
  port [V MDN]; a page served from another local port is same-site and its requests carry the cookie. The `Origin` check and the
  CSRF header of point 6 carry the weight there, not `SameSite`.
- **Loopback is shared by all local users.** Another account on the machine can connect to `127.0.0.1:<port>`; the token and the
  session id are the only barrier (rate-limit failed logins). On a multi-user host a Unix socket would be safer but a browser
  cannot use one.
- **The link is a bearer secret on screen and in the terminal's scrollback**: visible in screen sharing or a recording, in tmux
  history, on hover; one-time use and expiry (for example 10 minutes unused) limit it; the plain-URL fallback line should be
  omitted when the user asks for a hidden mode.
- **One-time links can be burned by a prefetch or a link preview** (an application that fetches the URL before the browser);
  then the user's click fails. Mitigations: accept the exchange only for top-level navigations (`Sec-Fetch-Mode: navigate`,
  `Sec-Fetch-Dest: document` [K, Fetch Metadata headers, unchecked per browser]) or allow a short grace window (60 s) for a
  second exchange from the same peer; the cheapest recovery is the fresh link.
- **Browser extensions** with cookie or history access can read the cookie and the link; out of scope.
- **Terminal link click behaviour varies** (modifier keys, disabled by default in Konsole [V]); the plain URL fallback covers it.
- **DNS rebinding** is blocked by the Host check, not by the binding to `127.0.0.1` [V MCP]; without that check a visiting
  website could address the server through its own name.

### Minimal shared backend API (both clients are thin)

Reuse the OpenAI-compatible routes where a standard exists, so third-party clients work too, and add one small job layer
because image generation (10 to 60 s [E]) and transcription of a recording (tens of seconds to minutes [E]) do not fit a
blocking request for a terminal progress bar or a browser tab.

| Route | Purpose | Notes |
|---|---|---|
| `GET /v1/models` | OpenAI model list (exists [V]) | add `purpose`, `size_bytes`, `installed`, `running` fields |
| `GET /tona/v1/catalog?purpose=P` | the installer's list for the class: recommendation + alternative, strengths, weaknesses, expected speed | same data source as `tona-install` |
| `POST /tona/v1/models/{id}/start`, `POST /tona/v1/stop` | switch the one running model (stop, wait for VRAM release, start) | returns a job; at most one model at a time |
| `GET /status` | state, phases with `done/total` (exists [V]) | add `purpose` and the active model; same shape as job progress |
| `POST /v1/chat/completions`, `/v1/messages` | coding, general, vision (image parts) | exist [V] |
| `POST /v1/images/generations` | text to image: `prompt`, `size`, `n`, `seed`, `steps`, `response_format` (`b64_json` or `url`) | OpenAI shape [K, from the OpenAI API definition]; engine: stable-diffusion.cpp's `sd-server` states an OpenAI-compatible `/v1/...` API [V README] |
| `POST /v1/images/edits` | multipart: `image[]` (up to 10 for the Klein models [V]), optional `mask`, `prompt` | OpenAI shape [K] |
| `POST /v1/audio/speech` | `input`, `voice`, `response_format` (wav, mp3), `speed`; streams bytes | zonos2-server and llama-server implement the route [V] |
| `POST /v1/audio/transcriptions` | multipart `file`, `language`, `response_format` (`json`, `verbose_json`, `srt`, `vtt`), `timestamp_granularities[]`; extension fields `diarize`, `max_speakers` | OpenAI shape [K]; diarization is our extension; video is accepted, the control point extracts audio with ffmpeg |
| `Prefer: respond-async` (or `?async=1`) on any POST above | returns `202` and `Location: /tona/v1/jobs/{id}` | so the same route serves blocking third-party clients and our two UIs |
| `GET /tona/v1/jobs/{id}` | `{state, progress:{done,total,unit}, rate, eta, error, results:[{name,mime,bytes}]}` | `unit` is `steps`, `audio_seconds` or `bytes`; x real time for transcription is `rate` |
| `GET /tona/v1/jobs/{id}/events` | Server-Sent Events: progress and partial segments (transcript lines as they arrive) | SSE is the pattern MCP's HTTP transport also uses [V]; cpp-httplib streaming support: unchecked for v0.51.0 |
| `GET /tona/v1/jobs/{id}/results/{name}` | the PNG, WAV, SRT, JSON | `Range` support for the browser's media seeking |
| `DELETE /tona/v1/jobs/{id}` | cancel (stop the step loop, kill the child) and delete the results | |

Thin clients: the **`tona` subcommands** are HTTP clients (bash plus `curl` for the first version, C++ with cpp-httplib [V
vendored] later); `tona image "prompt" -o x.png` = POST with `respond-async`, poll or SSE for a progress line on stderr, GET the
result, write the file. They authenticate with the API key (`Authorization: Bearer`) read from the state directory, not with the
browser cookie. The **web app** is `fetch()` against the same routes with the session cookie and the CSRF header, plus `<img>`,
`<audio>` and `<video>` elements pointing at the result URLs (same-origin requests carry the cookie, so no URL signing is
needed). The job layer lives in the control point, so a child engine without progress output only needs the control point to
estimate it (steps from the log line, seconds from the segment timestamps).

### Recommendation (UI)

1. **Backend routes first** (job layer plus the OpenAI-shaped routes, the token/cookie login, Host/Origin checks): the part both
   clients and third-party tools share, and the part that decides the security model.
2. **`tona` subcommands second** (`image`, `transcribe`, `speak`, `start`, `stop`, `ui`): scriptable, testable by `ctest`, the
   first client of the API.
3. **Web app third**, for image, transcription and TTS: three pages plus the model switcher; plain ES modules, no build chain, no
   CDN; a JS format/lint gate added to the repository.
4. **Server TUI unchanged** apart from one line: the login link (OSC 8 plus plain text) and the active model/purpose.

## General purpose (chat/assistant LLM)

Same engine, same MoE machinery as the coder; only the weights differ. Candidates with sizes [V HF trees]:

| Model | Size (Q4_K_M) | MoE | License | Notes |
|---|---|---|---|---|
| **Qwen3.5-35B-A3B** | 22.0 GB (+ mmproj 1.79 GB) | yes, `qwen35moe` | Apache-2.0 | architecture Tona already runs (#143); vision-capable |
| **Gemma-4-26B-A4B-it** | 16.95 GB (UD-Q4_K_M) | yes (25.8B total, 4B active) | Apache-2.0 | 5.1 M downloads; whether Tona has a Gemma-4 family: not checked |
| Qwen3.5-9B | 5.68 GB | no (dense) | not read | fits 12 GB VRAM alone |
| Qwen3.5-4B | 2.74 GB | no | not read | smallest sensible option; 16 GB Mac |
| Qwen3.8-27B dense | IQ3_XXS 10.1 GB (research/21) | no | | listed as a medium coder tier already |

Speeds are the coder's rates scaled by bytes per token [E]: Qwen3.5-35B-A3B reads roughly 3B active parameters at Q4, about
1.7 GB per token, upper bound 315 / 1.7 = 185 passes/s on the reference before the 0.65 factor of the class files, and about
half of that again for experts outside VRAM at 22 GB of weights on a 14.6 GB budget (unmeasured). Quality: no general-purpose
benchmark was collected in this research; take the per-model cards ([49](49-moe-models-2026-11.md) covers coding).

## Recommendation

Rule of the product: per purpose and hardware class one recommendation and at most one alternative. Hardware groups used
below: **PC 12 to 16 GB VRAM** (16/32/64 GB RAM), **Mac 16 to 24 GB**, **Mac 32 GB and larger / PC 24 GB+**. All sizes Q4-class
files from the tables above; speeds are [E] from the compute/bandwidth arithmetic and unmeasured.

| Purpose | Route | PC 12 to 16 GB VRAM | Mac 16 to 24 GB | Mac 32 GB+ / PC 24 GB+ |
|---|---|---|---|---|
| Coding | (a) existing | the existing tiers (ISTA Coder on 16 GB + 64 GB; Ornith-35B-A3B / Groot-19B on smaller RAM) | Ornith-1.5-35B-A3B or Groot-19B per `mac-16-24gb.md` | Medium tier per `mac-32gb.md` |
| General | (a) | **Qwen3.5-35B-A3B** Q4_K_M 22 GB (16 GB VRAM, 32+ GB RAM); alt **Qwen3.5-9B** 5.7 GB for 12 GB VRAM or 16 GB RAM | **Qwen3.5-9B** 5.7 GB; alt Qwen3.5-4B 2.7 GB | **Qwen3.5-35B-A3B**; alt Gemma-4-26B-A4B 17 GB |
| Vision (screenshots, UI) | (a) coder's own encoder; fallback (b) | the coder with its vision tower (to check); alt Holo3.1-4B Q4_K_M 3.07 GB via `llama-server --mmproj` | Holo3.1-4B; alt Qwen3-VL-8B 5.0 GB + 0.75 GB | coder vision; alt Qwen3.5-35B-A3B + mmproj (22 + 1.8 GB) |
| Image generation | (b) stable-diffusion.cpp | **FLUX.2 klein 4B** Q4_K_M 2.6 GB (Apache-2.0, fast, edits); alt **Z-Image-Turbo** Q4_K_M 3.85 GB (Apache-2.0, 8 steps) | FLUX.2 klein 4B via stable-diffusion.cpp (Metal); alt the same via mflux (MLX) | Z-Image-Turbo; alt FLUX 3 Image Klein Q4_K_M 3.67 GB (custom licence) |
| Image editing | (b) | FLUX.2 klein 4B (edits in the same checkpoint) | same | alt Qwen-Image-Edit-2511 Q4_K_M 12.8 GB (Apache-2.0) |
| TTS | (b) | **Kokoro-82M** (0.33 GB, Apache-2.0); alt ZONOS3-Mini (Apache-2.0, 100M-active MoE) via zonos2.cpp (Vulkan, CPU) | Kokoro via MLX-Audio; alt ZONOS3-Mini (Metal) | Qwen3-TTS 1.7B Q4_K_M 1.22 GB; alt ZONOS2 Q4_K 4.9 GB |
| STT | (b) + adapter | **Parakeet-TDT-0.6B-v3** 0.74 GB (CC-BY-4.0, transcribe.cpp, Vulkan); alt **Whisper large-v3-turbo** Q5_K_M 0.62 GB (whisper.cpp ROCm) | Parakeet-TDT-0.6B-v3 (Metal); alt Whisper turbo (Metal / Core ML) | Qwen3-ASR-1.7B Q4_K_M 1.32 GB; alt Nemotron 3.5 Transcribe 30B-A4B only if a Nemotron-H backend exists |
| Transcription (recordings, video) | (b) + ffmpeg + adapter | **Parakeet-TDT-0.6B-v3** 0.74 GB, **`parakeet-primeline`** for German (FLEURS de WER 5.98 %); diarization **Sortformer 4spk** (max 4 speakers); alt Whisper large-v3-turbo Q5_K_M 0.62 GB (word timestamps experimental) | same on Metal; expect lower x RT (unmeasured) | Parakeet + Sortformer; alt Nemotron 3.5 Transcribe 30B-A4B (speaker-attributed built in, 1 h files) only with an engine; pyannote community-1 (CC-BY-4.0, Python) for more than 4 speakers |
| Embeddings + reranker | (b) | **Qwen3-Embedding-0.6B** Q8_0 0.64 GB + Qwen3-Reranker-0.6B; alt EmbeddingGemma 2 (MTEB code 78.68 [V]) | same | Qwen3-Embedding-4B/8B + Reranker-4B |
| FIM / next-edit | (b), test (a) | **Qwen2.5-Coder-1.5B** Q8_0 ~1.6 GB (Apache-2.0); alt **Sweep Next-Edit 1.5B** 1.54 GB | same | Qwen2.5-Coder-7B (Apache-2.0) or the coder's own FIM |
| Video | (c) | leave out; Wan 2.2 TI2V-5B only by the user's own tool | leave out | leave out |

Strengths and weaknesses for the installer text (all [V] from the cards or tables above unless marked):
FLUX.2 klein 4B: Apache-2.0, 4 steps, editing in one checkpoint; weaker than larger models at fine text [K]. Z-Image-Turbo:
Apache-2.0, 8 steps, quality claim by the authors only. Parakeet: very fast, English-centric long-form WER 13.37 against 10.98 for
the MoE Nemotron (NVIDIA's own table); Whisper: broad language coverage, no longer the accuracy leader. Kokoro: tiny, English
and a few languages, no voice cloning. ZONOS3-Mini: voice cloning, 100M active; vendor-only evaluation. Qwen3-Embedding: 32K
context, instruction-aware; vendor benchmark. FIM 1.5B: fast, weak on long context (8K for Sweep).

**The three extra purposes with the most value for a coding-agent user** (judgment from the tables, not a measurement; "general"
and "coding" are not counted because the same engine already serves them):

1. **Vision-language for screenshots and UI.** Agents that build or test UIs must see the result; Tona turns images into
   "[image omitted]" today. It is also the only extra purpose that needs **no switch**: the coder's own family has a vision
   encoder [V].
2. **Embeddings plus reranker for code search.** Two Apache-2.0 0.6 GB models decide which part of a large repository the
   agent reads. Under the one-at-a-time rule it is a batch job (index, then switch back) unless the maintainer allows a
   CPU-side exception (see "Switching", point 4).
3. **Speech to text** (dictation to the agent, and transcription of meetings and videos as its own installer entry). Tiny model, short exclusive job, engines exist on ROCm/Vulkan/Metal; only the
   OpenAI-style route needs an adapter. FIM autocomplete is the fourth: it needs a resident model while editing, which the
   rule forbids unless the same exception exists. Image generation is the purpose that the one-at-a-time model suits best
   technically (compute-bound, wants the whole GPU) but it adds the least to coding work.

## Open checks (nothing was run)

1. Does the coder's vision tower convert to an mmproj, and how large is it? (read `Qwen/Qwen3.8-Flash-Next` `vision_config`
   and the safetensors index; run a 20-screenshot test once the path exists.)
2. FIM through the coder: 200 infill prompts against the IQ1_M quant vs the base model (exact match, latency).
3. Switch time: measure stop, VRAM release and start for a 4 GB stable-diffusion.cpp model and for the coder on the reference
   box (plan: 5 rounds, expectation under 10 s for the small model from the [E] arithmetic, tolerance a factor of 2); check
   that `mem_info_vram_used` returns to the desktop's 1.4 GB after the child exits.
4. Transcription: measure x RT for Parakeet-TDT-0.6B-v3 and Whisper turbo on a 1 h German recording (Vulkan on the PC, Metal on the Mac, CPU), plus Sortformer and the WER of v3 vs `parakeet-primeline` on 20 min of German speech (expectation from [E]: 100 to 300x RT for Parakeet on the GPU, 5 to 20x on the CPU, tolerance a factor of 3).
5. Which image and transcription routes `sd-server` and the transcribe.cpp server really implement (start them with a model,
   list routes); decide the adapter set.
6. Seconds per image on gfx1200 for stable-diffusion.cpp with Z-Image-Turbo Q4_K_M and FLUX.2 klein 4B (expectation from
   the [E] numbers above: about 10 s for 8 steps at 1024 px, tolerance a factor of 3).
7. Web app login: test a hostile page in another tab, a page on another local port and a DNS-rebinding name against the Host/Origin/CSRF gate; check whether browsers keep the one-time URL in history after the 303; test prefetch/link-preview burning; check cpp-httplib v0.51.0 for streaming (SSE) and `Range`; OSC 8 click in GNOME Terminal, Konsole, tmux, Terminal.app; whether stable-diffusion.cpp and the transcription engines report progress.
8. Re-check in 3 months: stable-diffusion.cpp PR #2315 (Nucleus-Image), MoE image checkpoints at or below 20 GB, WeLM-Image
   weights (abstract only; no repository checked).

## Sources (all fetched 2026-11-15)

Hugging Face model cards and file trees: [HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0),
[Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image), [HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full),
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
[flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein), [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B),
[Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B),
[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
[ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini), [ZONOS2-GGUF](https://huggingface.co/Zyphra/ZONOS2-GGUF),
[Ming-3.0-tts-tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny), [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B),
[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), [embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2),
[sweep-next-edit-1.5B](https://huggingface.co/sweepai/sweep-next-edit-1.5B), [Mellum2.1-12B-A2.5B](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking),
[Holo-3.1-35B-A3B](https://huggingface.co/Hcompany/Holo-3.1-35B-A3B), [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
arXiv abstracts: [Nucleus-Image 2604.12163](https://arxiv.org/abs/2604.12163), [HiDream-I1 2505.22705](https://arxiv.org/abs/2505.22705),
[WeLM-Image 2611.07392](https://arxiv.org/abs/2611.07392).
Engines: [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (README, docs/build.md, releases API, issues #1421, #1927, PR #2315),
[ComfyUI](https://github.com/Comfy-Org/ComfyUI), [mflux](https://github.com/filipstrand/mflux), [whisper.cpp](https://github.com/ggml-org/whisper.cpp),
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), [zonos2.cpp](https://github.com/Zyphra/zonos2.cpp),
[llama.cpp multimodal](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) and
[server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [llama-swap](https://github.com/mostlygeek/llama-swap),
[llama.vscode](https://github.com/ggml-org/llama.vscode).
Secondary (marked [K]): aggregator and blog summaries returned by web search for image models, Wan/LTX/HunyuanVideo VRAM and licences,
the Open ASR snapshot ranking, MoE TTS releases (Zyphra, Ant Group) and local FIM guides (including the VS Code limitation quote).
Access design: [Jupyter Server security](https://jupyter-server.readthedocs.io/en/latest/operators/security.html) and its [login handler](https://github.com/jupyter-server/jupyter_server/blob/main/jupyter_server/auth/login.py), [RFC 6265 section 8.5](https://www.rfc-editor.org/rfc/rfc6265#section-8.5), [MDN Cookies](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Cookies), [OSC 8 specification](https://gist.github.com/egmontkob/eb114294efbcd5adb1944c9f3cb5feda), [OSC 8 adoption table](https://github.com/Alhadis/OSC8-Adoption), [MCP transports, security warning](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports), [Are We Sixel Yet](https://www.arewesixelyet.com/), [kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/), [iTerm2 images](https://iterm2.com/documentation-images.html).
Repository files: `docs/backends/reference-platform.md`, `docs/hardware/*.md`, `research/44-memory-priority.md`, `research/49-moe-models-2026-11.md`,
`engine/server/api.cpp`, `toolchain/models.toml`.
