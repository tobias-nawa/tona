# Comparable projects and models

As of 2026-10-22. Pure internet research by an agent, **none of it measured by us**. All
tok/s values are community reports from individuals, one data point each. Model benchmark
values are vendor claims.

Target hardware: RX 9060 XT 16 GB (gfx1200), Ryzen 5 9600X, 64 GB DDR5-6000, desktop on
the same GPU.

## Engines

| Project | Approach | gfx1200 / RDNA4 | Numbers on comparable hardware | For agents | License / maturity |
|---|---|---|---|---|---|
| **llama.cpp** | `-ot`, `--n-cpu-moe` (PR #15077), `--fit` | gfx1200 native with ROCm 7.1.1 (docs PR #26745). Qwen3.8 on HIP: **#29149 open** (load livelock on gfx1201/gfx1151, Vulkan loads). #21376: ROCm does not offload the KV cache to RAM, Vulkan does | Qwen3.8-Flash-Next on RTX 5080 16 GB + 64 GB, UD-IQ3_XXS: **24.7 tok/s decode, only 15–20 tok/s prefill** ([HF thread](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3)). RDNA4 in general: Vulkan mostly faster in decode, ROCm partly in prefill | OpenAI API, Anthropic `/v1/messages` (PR #17570), tool calls, slots, prompt cache | MIT, very active. Qwen3.8 since PR #27742 (08-27); MTP PR #28243 closed, successor #29761 |
| **ik_llama.cpp** | Fork with faster CPU MoE kernels | **ROCm, Vulkan, Metal not supported**, issues about them unwelcome | – | OpenAI API, tool calls | MIT. **Not usable here** |
| **ktransformers** | Hot experts on GPU, cold on CPU (AMX/AVX-512), via SGLang | AMD since 2025 per README, route via SGLang on RDNA blocked | server numbers only | via sglang-kt | Apache-2.0. Qwen3.8 not listed |
| **vLLM** | GPU-centric, only `--cpu-offload-gb` | gfx1201 patches merged (#38086, #38455), gfx1200 hardly mentioned. MoE CPU offload RFC #33869 "not planned" | – | very good | Apache-2.0. **Unsuitable here** |
| **SGLang** | like vLLM | **#30599 open**: gfx1200/1201 blocked on `main` (sgl-kernel only gfx942/950, AITER only CDNA) | – | very good | Apache-2.0. **Not runnable on RDNA4** |
| **Ollama** | llama.cpp as engine since 0.30 | ROCm and Vulkan | No `--n-cpu-moe` (issue #11772, PR #12333 open). Automatic expert placement only without `num_gpu` set | OpenAI, Anthropic API | MIT |
| **LM Studio** | llama.cpp with MoE CPU slider (since 0.3.23) | ROCm and Vulkan | – | OpenAI API | proprietary |
| **Lemonade (AMD)** | Wrapper around llama.cpp with nightly ROCm 7 builds | gfx120X incl. 9060 XT listed | – | OpenAI API | open source |
| **ExLlamaV3** | Own quantization, CPU offload for MoE | CUDA only | – | via TabbyAPI | MIT |
| **PowerInfer, MLC-LLM** | Hot/cold neurons and compiler, respectively | ROCm only old (PowerInfer: gfx1100) | – | weak | little activity in 2026 |
| Research | HybriMoE, Apex (arXiv 2610.23006), exllamav3max PR #4 | all CUDA | – | – | prototypes |

## Qwen3.8-Flash-Next on 16 GB + 64 GB

- **Size:** Unsloth GGUFs 72.5 GB (IQ1_S), 78.9 GB (Q2_K_XL), 82 GB (IQ3_XXS)
  ([Unsloth](https://unsloth.ai/docs/models/qwen3.8-next)). Fits in 80 GB only if the
  n-gram table stays on the NVMe via mmap. Reports say this works.
- **The bottleneck is reading the prompt (prefill).** Decisive for agents with long prompts.
- llama.cpp reports about 15–20 tok/s prefill for this model on a 16 GB card (see the
  table above), so prefill speed is the open problem for this model on this class of hardware.

## Smaller models with more headroom

Benchmark values are **vendor claims**.

| Model | Size (total/active) | Q4 GGUF | Vendor claim |
|---|---|---|---|
| **Qwen3.6-35B-A3B** | 35B/3B, 262K context | ~20–22 GB (estimate) | SWE-bench Verified 73.4, Terminal-Bench 2.0 51.5 ([model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) |
| **Laguna XS 2.1** (poolside) | 33B/3B, 262K | similar | SWE-V 70.9, TB2 37.5 ([poolside](https://poolside.ai/blog/introducing-laguna-xs-2-1)) |
| **Qwen3-Coder-Next** | 80B/3B, 262K | 48.5 GB Q4_K_M, 36 GB Q3 | SWE-V ~70.7, third-party source only |
| **gpt-oss-120b** | 117B/5B | ~63 GB | SWE-Pro 16.2 per poolside table, weaker for agentic use |

- For Qwen3.6-35B-A3B and Laguna, attention and KV sit in VRAM, experts partly in RAM.
  That leaves room for several slots, i.e. several agents.
- Expect roughly 25–45 tok/s decode (**estimate**). On an RX 7800 XT under Vulkan the value
  dropped from 18 to 6.5 tok/s decode beyond ~50K context while the KV cache lived in RAM.

## The agent's assessment

This is the agent's opinion, not a decision:

1. **For gfx1200 with experts in RAM, only llama.cpp (HIP or Vulkan) remains in 2026.**
   vLLM, SGLang, ktransformers, ik_llama.cpp and ExLlamaV3 drop out. Ollama and Lemonade are
   wrappers around llama.cpp.
2. **Desktop on the same GPU** costs ~1.4 GB VRAM. Prefill hangs with a GPU reset are
   reported on RDNA4 when the desktop shares the card; whether the desktop aggravates them is
   not confirmed, but it affects us directly.
3. **For the worker in the coordinator pattern**, Qwen3.6-35B-A3B or Laguna XS 2.1 via
   `llama-server` would be the more robust choice: Anthropic API, parallel slots, lots of
   context. Qwen3.8-Flash-Next would be the attempt to get more quality, with a stability risk.
4. **Measure ourselves:** Vulkan versus ROCm via `llama-bench` on our own card, several
   runs. No third-party numbers exist for the 9060 XT with these models.

## Sources

- llama.cpp: [#27742](https://github.com/ggml-org/llama.cpp/pull/27742), [#28243](https://github.com/ggml-org/llama.cpp/pull/28243), [#29149](https://github.com/ggml-org/llama.cpp/issues/29149), [#21376](https://github.com/ggml-org/llama.cpp/issues/21376), [#26745](https://github.com/ggml-org/llama.cpp/pull/26745), [#17570](https://github.com/ggml-org/llama.cpp/pull/17570), [discussion #28588](https://github.com/ggml-org/llama.cpp/discussions/28588)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), [ktransformers](https://github.com/kvcache-ai/ktransformers)
- [SGLang #30599](https://github.com/sgl-project/sglang/issues/30599)
- vLLM: [#40081](https://github.com/vllm-project/vllm/issues/40081), [#33869](https://github.com/vllm-project/vllm/issues/33869)
- [Ollama MoE offload](https://aliteq.com/ollama-moe-offload-cpu), [LM Studio 0.3.23](https://lmstudio.ai/changelog/lmstudio-v0.3.23)
- [ROCm 7.14](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.14-blog/README.html)
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [exllamav3max PR #4](https://github.com/pstaniewska/exllamav3max/pull/4)
- [Qwen3-Coder-Next GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
- [Ollama Vulkan vs. ROCm on gfx1201](https://discuss.cachyos.org/t/comparison-ollama-vulkan-vs-ollama-rocm/21789)
