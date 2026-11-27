# Research notes

Working notes from the development of the engine: agent-assisted literature and project research, design studies and
the measurements behind decisions. Each file is a snapshot at the date it names. The notes are not maintained
after the fact; later code and [BENCHMARKS.md](../BENCHMARKS.md) take precedence where they differ. Sources (URLs,
issue and PR numbers, paper IDs) are given in the files. Agent research is marked as such and was not re-measured
unless a file says so; own measurements are kept apart in [measurements/](measurements/).

Notes and measurements from before 2026-11-04 use the old names (`ie-*`, `IE_*`); since then the names are `tona-*` and `TONA_*`.

## Files

| File | Content |
|---|---|
| [02-engines-compared.md](02-engines-compared.md) | Comparable projects and models: llama.cpp, vLLM, SGLang, ktransformers and others, plus smaller models |
| [03-feasibility.md](03-feasibility.md) | Feasibility of an own engine: hardware ceiling, effort estimate, options |
| [04-rocm.md](04-rocm.md) | ROCm 10: versions, changes, sources, setup from stable wheels |
| [06-qwen38-architecture.md](06-qwen38-architecture.md) | The target model in detail: layer order, tensor shapes, state, bytes and FLOPs per token |
| [07-harnesses.md](07-harnesses.md) | Coding-agent harnesses against `ie-server`: request shapes, prompt sizes, recordings |
| [08-models.md](08-models.md) | Other models besides Qwen3.8-Flash-Next and what they would cost on this platform |
| [09-gpu-reset.md](09-gpu-reset.md) | GPU reset and KFD eviction with the megakernel on gfx1200 |
| [10-prefix-cache-hybrid.md](10-prefix-cache-hybrid.md) | Prefix cache for hybrid models with several conversations |
| [11-suffix-decoding.md](11-suffix-decoding.md) | Model-free drafts for coding agents (SuffixDecoding and relatives) |
| [12-gpu-hangs.md](12-gpu-hangs.md) | Whole-system hangs under compute on RDNA4: diagnosis and countermeasures |
| [13-deltanet-chunked.md](13-deltanet-chunked.md) | Chunked Gated DeltaNet (WY form) for prefill on WMMA |
| [14-expert-prefetch.md](14-expert-prefetch.md) | Predicting and preloading experts, cache policy, especially with speculative decoding |
| [15-sparse-decode-attention.md](15-sparse-decode-attention.md) | Decode attention for sparse top-k attention (QSA/DSA) |
| [16-prefill-fusion.md](16-prefill-fusion.md) | Prefill: fusing element-wise kernels, hyper-connections, GDN, MoE, megakernel |
| [18-moe-prefill-streaming.md](18-moe-prefill-streaming.md) | MoE prefill with experts in RAM: streaming, layer-major order, VRAM split |
| [19-rust.md](19-rust.md) | Moving to Rust: state of the ecosystem, AMD route, fork-join measurement, proof-of-concept plan |
| [20-parallel-requests.md](20-parallel-requests.md) | Several requests at once: experts in a batch (OEA, XShare), draft budget across sequences, state slots |
| [21-backends.md](21-backends.md) | Further platforms as separate backends: how to cut them, Mac, Windows, other engines' platforms |
| [22-installer.md](22-installer.md) | Installer: TUI libraries, hardware detection, models for 32 GB, checksums |
| [23-waiting-in-kernel.md](23-waiting-in-kernel.md) | Long waits in the megakernel: kernel timeouts, pausing instead of aborting, restart cost |
| [24-running-open-source.md](24-running-open-source.md) | Running the project in the open: AGENTS.md, AI policies, platform tiers, issue templates, code gates |
| [25-security-scanners.md](25-security-scanners.md) | Static analysis, quality and security scanners, GitHub-native and supply-chain security for a public repo |
| [26-testing-coverage.md](26-testing-coverage.md) | Audit of the tests, coverage tooling and enforced thresholds in pull requests |
| [27-updates-signing.md](27-updates-signing.md) | Update process, release signing, keys and maintainers, reproducible builds (with own measurements) |
| [28-release-licenses.md](28-release-licenses.md) | Licences of the ROCm and LLVM libraries shipped in release assets, what is missing and where to get it |
| [29-long-context-prefill.md](29-long-context-prefill.md) | Prefill at 64K to 256K: indexer scoring, top-k and KV gather of the sparse attention, candidates and a measurement plan |
| [30-cross-layer-index-reuse.md](30-cross-layer-index-reuse.md) | Reusing the sparse-attention selection across layers, decode steps and draft tokens (IndexCache, Kascade, TidalDecode, SPIN and others) and what to try next |
| [31-nll-threshold-validation.md](31-nll-threshold-validation.md) | Is 0.01 nats per token a sensible quality limit? Literature on Δ-NLL/KL against task accuracy, a long-context code benchmark on one shared context, and a dose-response plan |
| [32-long-context-1m-yarn.md](32-long-context-1m-yarn.md) | 1M context with YaRN: what the model card says, static vs dynamic (re-rotating the cached keys), quality evidence, memory and time at 1M on the reference platform, product option |
| [33-context-extension-survey.md](33-context-extension-survey.md) | Using a finished LLM beyond its training length, 2023-2026: RoPE scaling, position remapping, sparse selection, linear-attention hybrids, temperature scaling, what 1M models do, benchmarks; five ranked candidates with test plans for 250K / 512K / 1M |
| [35-prompt-cache-prewarm.md](35-prompt-cache-prewarm.md) | Pre-warming the prompt cache: when coding agents pay the full prefill (TraceLab, Claude Code cache rules, llama.cpp reports), warm-up contracts (`max_tokens: 0`), template checkpoints on the SSD, position-independent reuse for DeltaNet hybrids (HYPIC, LinearKV); proposal, memory per prefix, risks |
| [34-prefill-attention-kernel.md](34-prefill-attention-kernel.md) | The prefill attention kernel `k_attn_wmma` at 64K-250K: cost per position from the rocprofv3 traces (26-27 % of GPU time, 4-6 % of WMMA peak), roofline and ceiling, sparse-attention kernels 2025-2026 (union, KV-outer, QUILT), RDNA4 facts, ranked ideas with bit-exactness and a measurement plan |
| [55-expert-coactivation.md](55-expert-coactivation.md) | Expert co-activation as a predictor (vertical chains over layers, horizontal partners inside a layer): papers and trackers 2025-2026, ideas adapted from market-basket analysis, collaborative filtering, hardware prefetchers, PPM and branch predictors, and the replay result on the Coder |
| [papers.md](papers.md) | Running overview of new papers by topic, with relevance and status |
| [measurements/](measurements/) | Own measurements, one file per run, date in the name |
