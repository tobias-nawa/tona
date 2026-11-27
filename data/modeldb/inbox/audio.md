# Inbox from the audio area (emptied 2026-11-19)

All items folded into records (nemotron-3.5-transcribe-30b-a4b, sov-transcribe-1, voxtral-mini-4b-realtime-2602,
vibevoice-asr, parakeet-tdt-0.6b-v3). Nothing waits for a human decision.
- NVIDIA Nemotron 3.5 Lightning 30B-A3B (coding/general area): the decoder of `nemotron-3.5-transcribe-30b-a4b` is derived from it. A Nemotron-H backend for it would be reusable for that STT model (plus an audio encoder).
- Gemma 4 26B-A4B (general area): `sov-transcribe-1` (Salesforce) uses it as decoder behind a Whisper large-v3 encoder; a Gemma 4 MoE backend would serve it.
- Ministral 3 3B Base (general area): decoder of `voxtral-mini-4b-realtime-2602`.
- Qwen2 7B-class dense decoder: `vibevoice-asr` (Microsoft) reuses the config (28 layers, hidden 3584, 4 KV heads).
- Correction for research/52: transcribe.cpp now lists HIP/ROCm 6.1+ (README, checked 2026-11-19), not only Vulkan.

## From the coding area (gap pass 2026-11-19)
- `granite-switch-4.1` (IBM, Apache-2.0 by card): composed model with an embedded Granite Speech 5.0 TurboCTC encoder (470M, 7 languages) next to chat, decision and guardian adapters; the record has tasks general + audio-transcription; the audio area may refine the speech part.

## From the 2026-11-19 new-release sweep
- Qwen3-ASR architecture is now the backbone of three records: `qwen3-asr-1.7b`, `rumble-2-flash` (0.5B, German 5.52% on the Open ASR Leaderboard mix, builder-run) and `confucius4-r2t2` (1.7B streaming, stable-prefix output). One Qwen3-ASR implementation in Tona would cover all three.
- FastConformer-TDT now covers `parakeet-tdt-0.6b-v3`, `parakeet-primeline` and `parakeet-ultra` (FLEURS German 3.61% vs 4.13% for v3, vendor run; transcribe.cpp measures 4.58% on its own normalization). `parakeet-ultra` is the strongest small German ASR candidate found; needs our own German set.
- `fish-audio-drama-3`: 35.9B/3B-active TTS on a Qwen3.5-MoE backbone, license limits commercial use above USD 500k revenue and any hosted service; Qwen3.5-MoE backend work for the coding area would be reusable.
- `ming-3.0-tts-0.5b`: Qwen2.5-0.5B backbone plus DiT head, 3 GB, MIT, German WER 0.365% (builder). Cheap to port compared with the Ling-MoE `ming-3.0-tts-tiny`.
- `zonos3-mini` record: its codec ZAC-1 is published separately (`Zyphra/ZAC-1`, Apache-2.0, ggml decoder and encoder files, 44.1 kHz, 15 Hz x 10 codebooks); not given its own record, relevant if ZONOS3 is ported.
- `vibevoice-asr-streaming` (1.5B/7B, MIT) adds streaming speaker-attributed ASR to the offline `vibevoice-asr`.
- Not recorded: `moondream/parakeet-redux` (ternary 178 MB sibling of parakeet-ultra, card not read), `gradium/phonon-*` TTS (CC-BY-NC, card gated), `maya-research/maya2-native` (Indic languages only), `mistralai/Voxtral-Mini-4B-Realtime-Arabic` (Arabic fine-tune of the existing voxtral record, 17.7 GB, no license field on HF).

## From the 2026-11-19 gap pass 2
- `ming-3.0-audio` (inclusionAI, MIT): flash 126B-A6.5B (259 GB BF16, no class) and tiny 9.3B-A2.7B (single GGUF Q4_K 9.97 GB incl. encoder, Talker, VAE; audio.cpp family `ming3_audio`). Chinese and English only, so verdict watch; same KDA/MLA MoE backbone family as `ming-3.0-tts-tiny`, one backend would serve both.
- `liquid-codec` (Liquid AI, 74M, LFM Open License with US$10M revenue cap): speech codec behind LFM3-Audio-2B and LFM3-Omni-2B; those two models are not recorded yet (candidates for the next sweep; the codec card lists German among its languages, the models' own language support is not checked).
- Gap pass 2 (ASR/TTS/embeddings/translation/any-to-any worker) wrote: `phonon-2.5`, `orukeet`, `parakeet-redux`, `jais-asr`, `voxtral-mini-4b-realtime-arabic`, `haas-ja-v1.0`, `audio8-asr-infinite`, `edda-v0.2-large`, `gradium-phonon-tts` (31m-en Apache, all 40m models CC-BY-NC incl. German), `audio8-tts-realistic-air` (23 languages incl. German, Apache, llama.cpp-style runtime: strongest permissive small TTS found), `maya2-native`, `supertonic-3` (31 languages incl. German, OpenRAIL-M), `ema-lightning`, `tiny-voca`, `higgs-tts-3-4b` (100+ languages incl. German, non-commercial), `higgs-audio-v3-8b-stt-v2`, `yue2-3b`, `luciole-8b-audio-1.0`, `apriel-voice-27b-preview`, `sophon-realtime-v1`, `realtime-venus`.
- Phonon name clash: FermionResearch Phonon (ASR, Parakeet v3 compression) and Gradium Phonon (TTS, Pocket TTS/Mimi based) are unrelated; records use `phonon-2.5` and `gradium-phonon-tts`.
- Quant versus original on German: `phonon-2.5` and `parakeet-redux` lose to the uncompressed Parakeet v3 on German FLEURS (vendor tables); `orukeet` claims to beat it (pooled FLEURS 9.85 vs 11.01). A German-only check of Orukeet is the next audio test worth running.
- Not recorded (reasons): language-specific small ASR/TTS by individuals (Turkish, Persian, Uzbek, Indic, Thai, Arabic, Vietnamese `VieNeu-TTS-v3-Nano`, Piper voices, Paradee-8M, mithi, Inflect-Mini) as no German/EU relevance; `FireRedASR2` (vllm-project copy 33.5 GB Qwen2 LLM-ASR, Chinese/English); `mihuai/turkish-stt-66m-v2` (custom community licence); `yovoice-model` (only Kokoro-82M-1.1-zh GGUF packages).
