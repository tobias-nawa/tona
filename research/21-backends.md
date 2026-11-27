# Further platforms as backends of their own, first research 2026-10-26

Two agent research runs (Sonnet), 2026-10-26. [V] = page read, [S] = search hits only. Sections marked "knowledge"
come from the agent's training and are not documented. The second research run (section "Addendum") corrects
the first on Windows.

## Other engines

- ggml/llama.cpp [S]: backend registry, dynamic loading (`ggml_backend_load`), the scheduler distributes graph nodes to
  backends and copies in between; new an `optimize_graph` hook per backend (llama.cpp PR #15850). Limit: single
  graph operations, fusion only within one backend. A whole-model kernel does not fit this model.
- vLLM [S]: platform plugins via the entry point `vllm.platform_plugins`, class inherits `vllm.platforms.interface.Platform`
  (worker, attention backend); the model graph stays PyTorch (https://docs.vllm.ai/en/latest/design/plugin_system/).
- MLX [V] (github.com/ml-explore/mlx, docs "Custom Metal Kernels"): devices CPU and GPU (Metal), plus a CUDA backend
  on Linux (`mlx[cuda]`). `mx.fast.metal_kernel()` takes only the kernel body, atomics possible, nothing on
  synchronization across threadgroups or persistent kernels. A megakernel would only work with custom Metal.
- ktransformers [V] (README): hot experts on the GPU, cold ones on the CPU (AMX, AVX-512, AVX2), ROCm since 2025-03,
  Intel Arc, Ascend, native Windows; no Apple.
- PowerInfer [V] (README): hot neurons GPU, cold CPU; Apple CPU only, Metal "upcoming".
- SGLang: not researched.

## Megakernel on Apple Metal

- No counterpart to cooperative groups, no documented grid sync; no guarantee of simultaneous residency or
  progress of all threadgroups (Apple forum https://developer.apple.com/forums/thread/831017, /672532) [S].
- Sorensen et al., OOPSLA 2021, arXiv 2109.06132 [S]: Apple and ARM GPUs do not satisfy the occupancy model.
- Conclusion: our megakernel with ~1,100 grid barriers is not safe on Metal. Ways: dependency counters instead of
  global barriers (ForgeMegakernel, arXiv 2609.12379, CUDA), several dispatches or indirect command buffers
  (measure the cost per dispatch).
- MPK/Mirage (OSDI 2026, arXiv 2512.22219) [S]: megakernel compiler, CUDA only.

## Large MoE on 32 GB unified memory

- Qwen3-Next-80B-A3B on M1 Max 64 GB, MLX (community, self-reported) [S]: ~43 tok/s decode at 4K context,
  ~37 at 8K, peak ~43 GB (https://omlx.ai/benchmarks/v2gdcued). Does not fit in 32 GB.
- Flash-MoE (C/Metal, Qwen3.5-397B-A17B on M3 Max 48 GB) [S]: dense weights (~6 GB) in RAM, experts from the SSD
  via the page cache (~71 % hits): 4.36 tok/s (decode) with 4-bit experts, 5.74 with 2-bit (tool calls broken)
  (https://simonwillison.net/2026/mar/18/llm-in-a-flash).
- Conclusion: on the Mac dense weights and as many experts as possible in shared memory, the rest by mmap
  from the SSD; or a smaller model / stronger quantization (installer). Measure.

## Windows

- HIP SDK for Windows 7.2.0 [V] (rocm.docs.amd.com/projects/install-on-windows, system requirements): **RX 9060 XT
  and RX 9060 (gfx1200) supported** (runtime, HIP SDK, debugger). MIOpen and MIGraphX are missing on Windows [S];
  we do not need them.
- WSL2 [V] (compatibility matrix Radeon/Ryzen): ROCm 7.2.1 with Adrenalin 26.1.1, RX 9060 XT listed, Ubuntu
  24.04.2/22.04, WSL kernel 5.15. AMD states lower performance than native on Linux [S].
- llama.cpp on Windows [V] (docs/build.md): HIP build documented (`-DGGML_HIP=ON`), Vulkan likewise; no
  documented comparison of HIP against Vulkan on RDNA4 under Windows.
- VFIO passthrough [S]: RDNA4 reset bug (9070 XT), after the VM ends amdgpu hangs or the host reboots; only
  workarounds (Proxmox forum, TrueNAS forum). Risky for us: a GPU reset takes the desktop down with it.
- Conclusion: native Windows HIP first (gfx1200 officially), WSL2 as the second way; VM with passthrough not.
  Open: our toolchain (ROCm 10.1, HIP 7.16) against the Windows SDK 7.2; measure megakernel with `grid.sync()` and
  pinned host memory under Windows (WDDM).

## Proposal for the cut (knowledge, the agent's recommendation)

- Common: model description, GGUF loader, tokenizer, chat template, API, planner, conversations/slots,
  memory budget. Per backend: the whole forward pass with its own kernels and its own expert strategy.
- Interface roughly, one call per pass: load with budget, pass over slots → logits/drafts, save/load
  state. No single ops in the interface (otherwise the megakernel is lost).
- Selection at build time (CMake target per backend); later optionally `dlopen` with a C ABI table (one indirect call per
  pass costs nothing measurable).
- Tests: reference against itself bitwise; other backends against the reference or CPU with tolerance (deviation per
  layer, KL on a prompt set, top-1).

## Addendum: the Mac in detail (second research run)

- M1 Pro [V] (Wikipedia "Apple M1"): 14 or 16 GPU cores, 204.8 GB/s, at most 32 GB. llama.cpp discussion #4167
  [V], Llama 7B on M1 Pro 16 cores: Q4_0 decode 36.4 tok/s, prefill (PP512) 266 tok/s; from this roughly 130–180 GB/s
  effective (derived, not measured).
- GPU memory limit [S]: `recommendedMaxWorkingSetSize` is about 2/3–3/4 of RAM, i.e. ~21–24 GB for 32 GB;
  can be raised via `sysctl iogpu.wired_limit_mb`, not permanently, not supported by Apple.
- Model larger than RAM [V] (llama.cpp #19825, M4 Pro 24 GB, Qwen-Next 80B): as soon as the RAM is full, a "swap
  death spiral" up to kernel panic, even with `--n-cpu-moe` and a raised wired limit; ik_llama.cpp with `--cpu-moe`
  ~10 tok/s decode, prefill the bottleneck. Closed 2026-02-26 without a fix.
- "LLM in a flash" (arXiv 2312.11514, ACL 2024) [V, abstract only]: weights in flash, windowing (keep the most recently active
  neurons) and row-column bundling (larger contiguous reads); models up to 2× DRAM, 4–5×
  (CPU) or 20–25× (GPU) faster than naive loading. For MoE: experts instead of neurons, own reasoning.
- Qwen3-Next-80B-A3B 4-bit, MLX (oMLX community) [S]: M3 Max 40 cores ~60 tok/s decode, prefill ~830 tok/s,
  peak ~61.6 GB. No numbers found for M1/M2/M3 Pro.
- Metal dispatch [S]: no published µs measurement for dispatch, command buffer or ICB found; measure ourselves.
  Progress between threadgroups [V] (forum 831017, 2026-06, no reply from Apple): not guaranteed; counter
  dependencies only safe if all threadgroups are resident at the same time (size the grid to residency).

Conclusion for the Mac: the 58 GB model does not fit (the GPU sees ~21–24 GB, paging out ends in swap according to #19825).
Ways: (1) stream experts from the SSD with an own cache instead of mmap/swap (Flash-MoE: 4–6 tok/s at 397B), (2)
a smaller model for 32 GB Macs via the installer. First measure what an expert read from the Mac SSD
costs.

## Still open

SGLang (structure of the backends); MLX behavior with mmap; Metal cost per dispatch and ICB (measure);
M1 Pro bandwidth and FP16 performance measured; llama.cpp HIP on Windows with gfx1200.

## Proposal for the backend interface (2026-10-26, for decision)

The generator today calls 33 methods of `gpu::Mega` (forward, forward_multi, commit, prefill, mtp_fill/step,
draft, save/load_state, stash/unstash, use_slot, …). Exactly this surface becomes the interface of a backend: one
whole pass per call, no single ops.

- **Common:** `model/` (GGUF, tokenizer, quant formats), `server/` (API, chat template, generator, planner,
  prompt cache, sampling, speculation), `tools/ie_top`, scripts, benchmarks.
- **Per backend:** forward pass, prefill, drafts, state and slots, memory plan, expert strategy, own kernels.
  Reference = `engine/gpu` as today (HIP, gfx1200, megakernel, AVX-512 experts).
- **Selection at build time:** a C++ concept `ie::Backend` checks via `static_assert` that a backend satisfies the surface;
  the generator takes the type chosen at build time (`IE_BACKEND=hip-gfx1200|metal|cuda|…` in CMake), one
  binary per backend. The reference path stays line for line the same code, no virtual function.
- **Verification:** reference against itself bit-identical (bench set, same tokens, speed ± spread) for every change
  to the common part. Other backends against the CPU reference (`cpu/ref.cpp`) with tolerance: deviation per layer,
  KL and top-1 on a fixed prompt set; plus the 23 tasks from `bench-effort.py`.
- **For contributors:** `docs/backends/README.md` (surface, tests, measurement duty), directory `engine/backends/<name>/`.

Alternative: virtual interface, one binary, backends selectable at start time (later as a `dlopen` plugin).
Costs ~30 indirect calls per pass of 20–60 ms, i.e. nothing measurable, and the installer would need only one
package. Drawback: all backends bound to a fixed binary interface, and the reference code gets a base class.

**Decided 2026-10-26 (maintainer): the alternative**, virtual interface and selection at start. Implemented
(`engine/backend/backend.h`, `gpu::Mega` as reference), speed unchanged; guide `docs/backends/README.md`.

## Platforms of other engines (third agent research run, 2026-10-26)

[V] read (WebFetch, summary), [S] search hits, [L] local file.

| Engine | NVIDIA | AMD gfx1200 | Intel | Apple | Windows native | Structure |
|---|---|---|---|---|---|---|
| llama.cpp [V] | ✓ | HIP and Vulkan | SYCL, Vulkan | Metal | ✓ | ggml backends, op graph; `GGML_BACKEND_DL` loads at runtime |
| ik_llama.cpp [V] | ✓ from Turing | ✗ (ROCm/Vulkan unwanted) | ✗ | partly | build | CPU + CUDA |
| vLLM [V] | ✓ | gfx1200/1201 listed (FP8 gap [S] #28649) | XPU | vllm-metal (MLX plugin) | ✗ (WSL) | platform plugins |
| SGLang [V] | ✓ | ✗ (MI300, Strix Halo) | ? | ? | ? | |
| MLX [V] | CUDA (Linux) | ✗ | ✗ | ✓ | ✗ | wheel per accelerator |
| mistral.rs [V] | ✓ | ✗ | ✗ | ✓ | ? | |
| MLC-LLM [V] | ✓ | ROCm, Vulkan | Vulkan | ✓ | ✓ | TVM compiler generates code per target |
| Ollama [V] | ✓ | Linux ✓ (9060/9070), Windows only RX 7000 | Vulkan | ✓ | ✓ | ggml fork, Vulkan as a safety net |
| ExLlamaV3 [V] | ✓ | ✗ | ✗ | ✗ | ✓ | CUDA only |


Conclusions: op-graph interfaces (ggml, vLLM) do not fit a megakernel; our cut "whole pass per call" stays.

## Update 2026-11-11: engines, MLX, llama.cpp Metal, watchdog

Details and sources in [39-mac-engines-2026-11.md](39-mac-engines-2026-11.md) (engines) and
[38-moe-on-mac.md](38-moe-on-mac.md) (papers). Corrections to the sections above:

- **Streaming the reference model on an M1 Max 32 GB is done elsewhere:** ds4 reaches 8.25 to 8.44 tok/s decode with the
  512-expert model; it does not run the 256-expert Coder.
- **MLX** v0.32.4 (2026-10-14): no IQ formats (GGUF types are cast to fp16), no expert offload (mlx-lm #1438 open, PR
  #1588 closed unmerged 2026-08-21), qwen4exp support open (mlx-lm #1788). Not a basis for our quantized Coder.
- **llama.cpp Metal:** qwen4exp merged (#27742, 2026-08-27); IQ formats lack small-batch kernels (#27350 open), so
  speculation on IQ files gains little upstream; SSD streaming PR #25294 open.
- **Watchdog:** Apple documents no forward progress between threadgroups; community reports put the command buffer limit
  near 5 s with waiting counted, consistent with our measurement (2 s completes, 8 s times out,
  [measurements/2026-11-11-mac-event.md](measurements/2026-11-11-mac-event.md)); Apple's `accessRevoked` can take GPU
  access away after repeated timeouts.
- **Recommended working set** is 26.8 GB on this M1 Max 32 GB (measured), not the 21 to 24 GB assumed above.

## Model tiers (2026-11-11, #107, #46)

Requirement (maintainer, 2026-11-11): the user picks one of three tiers, **large** (high quality, slow), **medium**
(medium quality, faster), **small** (lower quality, very fast), on the Mac (M1 Max 32 GB) and on Linux PCs with 32 GB
RAM (#46). Agent research (Opus, 2026-11-11), update of [08-models.md](08-models.md) (2026-10-23); key claims
re-checked by hand (sizes and runtime of Gyro-Coder, Ornith card, llama.cpp PR states). [V] page or HF API read,
[S] search only, [L] repo file. **All benchmark numbers are vendor claims; no independent run exists for any
candidate** [S]: every SWE-bench and Terminal-Bench number below is **vendor, unchecked**. Nothing here is measured
by us except the platform limits. **Condition before a tier is fixed:** our own quality measurement of each candidate
against the large model: `scripts/bench-effort.py` (tasks), `bench/threshold/` (task benchmark) and Δ-NLL on code
against the large model ([backends/README.md](../docs/backends/README.md#checks)). In particular Ornith's "SWE-V 79" at 3 B active
equals the large model and needs this check.

**Budgets.** Mac: 26.8 GB the GPU may use, 334 GB/s read, SSD 13 GB/s for 2 MB blocks at depth 16
([measurements/2026-11-11-mac-bandwidth.md](measurements/2026-11-11-mac-bandwidth.md),
[-mac-ssd.md](measurements/2026-11-11-mac-ssd.md)). PC: ~14.6 GB VRAM free next to the desktop, 315 GB/s; RAM usable
by the engine at 32 GB assumed 16 to 20 GB (unchecked). The qwen4exp n-gram table (28.8 GB) stays on the SSD on both
(16 rows per token, ~0.2 ms on the Mac). Decode bound (estimate) = bandwidth / (active weight bytes + KV bytes read per
token), passes per second, no MTP multiplier.

KV at 128K (int8; from config.json): qwen4exp 1.61 GB + indexer 0.40 GB (QSA reads only the selection, ~25 MB per
pass); qwen35moe (35B-A3B, 19B-A3B) 1.34 GB, read in full every pass; Qwen3.8-27B dense 4.29 GB.

### Large: the reference model, streamed

ISTA Qwen3.8-Flash-Next GSQ-RCO Coder "IQ1_M" (qwen4exp, 256 experts, 6 B active): 58.41 GB = 29.6 GB trunk + 28.8 GB
n-gram table [V]; no smaller quantization offered [V]. SWE-bench Verified (vendor, unchecked): **75.6** for the file Tona pins and
measures with (revision of 2026-09-29, `toolchain/models.toml`), **79.2** for ISTA's rebuild (card section of
2026-10-17, first shard replaced 2026-10-19, now `main`) [V], [L 08]; BF16 base 82.8. No Terminal-Bench for the Coder
(base: TB 2.1 67.6). Moving to the rebuild is a separate decision with its own measurement. Mac: the trunk does not fit, experts stream from the SSD. Decode
estimate from the measurements: **~22 to 31 passes/s** with misses read on demand per layer, **up to ~50 to 65** with
prediction ahead of the pass (GPU bound 66 if all resident). PC at 32 GB: 28 to 33 tok/s decode measured
([22-installer.md](22-installer.md)).

**Counter-evidence to this estimate** ([39-mac-engines-2026-11.md](39-mac-engines-2026-11.md)): open-source engines
already stream Flash-Next experts on an **M1 Max 32 GB**: ds4 reaches **8.25 to 8.44 tok/s decode** (Q2), with a design of
per-layer `MTLSharedEvent` handshake, LRU arena and `F_NOCACHE`. It runs the 512-expert model with an arena of well under half the experts;
our estimate assumes the 256-expert Coder (larger resident share), reads at depth 16 and overbooked prediction. **The
estimate of 22 to 31 (up to 50 to 65) is unconfirmed until it is reconciled with these measurements**.

**Revised after M0 (2026-11-11, [41-mac-reconciliation.md](41-mac-reconciliation.md)):** with measured expert kernels
(10.2 ms per token, untuned) the resident pass costs 25 to 32 ms, not 15.3 ms; decode estimate **20 to 28 passes/s on
demand, 23 to 34 with prediction** (provisional hit rates). The gap to ds4's 8.3 tok/s (factor 2 to 3) is not explained
by the M0 measurements.

### Medium

| | **Gyro-Coder TQ2_0 noPLE** (recommended) | Qwen3.8-27B ISTA IQ3_XXS (runner-up) |
|---|---|---|
| Architecture | qwen4exp, 256 experts, ~2.1 bpw; **new quant format** (trellis, TQ2_0) | qwen35 dense: DeltaNet shapes as ours, full attention, no MoE/PLE |
| Size | 21.40 GB, n-gram table loaded separately (`--ple`) [V] | 10.09 GB (IQ3_S 11.77) [V] |
| Resident at 128K | ~23.5 GB | ~14.5 GB |
| Benchmarks (vendor, unchecked) | **none** (KLD/top-1 against Q8_0, own 16-task suite 13/16) [V] | TB 2.1 73.0 (Qwen card), SWE-Pro 61.7 [L 08] |
| License | qwen-community-1.0 [V] | Apache-2.0 [V] |
| GGUF / runtime | agentionai, 2026-10-21, mod. 11-05; **stock llama.cpp cannot load it**, agentionai fork only, marked experimental [V, checked] | ISTA, 2026-08-28; upstream `qwen35` |
| Fits | Mac yes (~3 GB headroom); PC VRAM + ~7 GB RAM | both |
| Decode bound (est.) | Mac 71 to 82, PC 67 to 78 (resident) | Mac 35 / 24 at 128K, PC 33 / 23 |
| Cost in Tona | loader unchanged; TQ dequant kernel in every backend, separate PLE file | new dense path in every backend |

Third option without the trellis format: requantize the 256-expert Coder ourselves with standard IQ2_XXS experts
(~19 to 20 GB, estimate); no such file exists. Before any kernel work: KLD/top-1 and our task benchmark of Gyro-Coder
against the ISTA Coder.

### Small

| | **Ornith-1.5-35B-A3B** (recommended) | Groot-19B-A3B (runner-up) |
|---|---|---|
| Architecture | qwen35moe: 30 DeltaNet + 10 gated attention, MoE 8 of 256 + shared, MTP; ~2.94 B active | qwen35moe, 8 of 128 experts, ~3 B active [V] |
| Sizes | Q4_K_M 21.71 to 21.86, IQ4_XS 19.28, IQ3_XXS 15.34, IQ2_M 12.54 GB; GSQ-RCO 3 bit 12.99 GB [V] | Q4_K_M 11.66 GB (GrootTeam), IQ4_XS 10.94 [V] |
| SWE-bench Verified (vendor, unchecked) | **79** (OpenHands, 5-run mean) [V, checked] | not reported [V] |
| Terminal-Bench 2.1 (vendor, unchecked) | **67.8** (Terminus-2), 68.5 (Claude Code) [V, checked] | 52.1 (Claude Code) [V] |
| Long agent tasks (vendor, unchecked) | weak: DeepSWE 22, Toolathlon 48.7 (Flash-Next 58.7 / 73.5) [V] | - |
| License | MIT [V, checked] | Apache-2.0 [V] |
| GGUF | ornith-ai 2026-08-18, bartowski 08-19, DFlash drafter 09-20 [V] | GrootTeam 10-23, bartowski 10-25 [V] |
| Fits | Mac Q4_K_M + KV 23.2 GB; PC entirely in VRAM only at IQ2_M / 3 bit (quality unmeasured) | Mac yes; **PC Q4_K_M + 128K KV 13.0 GB entirely in VRAM** |
| Decode bound (est.) | Mac 187 short / 107 at 128K; PC IQ2_M 306 / 133 | Mac 189 / 107; PC 178 / 101 |

**No qwen4exp model with ~3 B active exists** [V, HF search]: every qwen4exp variant reads 3.4 to 4.1 GB of dense
weights per token, so the small tier needs a second architecture. qwen35moe is a close relative: DeltaNet, gated
attention with 256-wide heads, MoE + shared expert, MTP and the tokenizer (vocab 248,320) are reusable; new are
pre-norm RMSNorm instead of Gated Residual, full attention instead of QSA, other widths, top-8, no PLE. One
implementation serves Ornith, Groot-19B, Qwen3.6-35B-A3B and their finetunes. Not recommended: Ornith-1.5-9B (dense,
~3× the bytes per token for lower scores), Laguna XS 2.1 (unrelated architecture, upstream PR #25165 unmerged),
Clover-35B-A3B (diffusion LM).

### Fit on both platforms

| Tier | Candidate | Mac 26.8 GB | PC 16 GB VRAM + 32 GB RAM |
|---|---|---|---|
| large | ISTA Coder IQ1_M | SSD streaming | VRAM + RAM + SSD (measured) |
| medium | Gyro-Coder TQ2_0 noPLE | yes | yes (VRAM + ~7 GB RAM) |
| medium | Qwen3.8-27B IQ3_XXS | yes | yes, at the VRAM limit with 128K |
| small | Ornith-1.5-35B-A3B | yes (Q4_K_M) | yes (IQ2_M in VRAM, or Q4_K_M VRAM + RAM) |
| small | Groot-19B-A3B | yes | yes (Q4_K_M in VRAM) |

### Changes since 2026-10-23 and open upstream work

- Qwen released no new language model [V].
- GrootTeam (2026-10-23): Groot-65B-A6B (qwen4exp, same shapes as our Coder; vendor SWE-V 56.1, smallest file 54.5 GB:
  unattractive) and Groot-19B-A3B [V].
- Gyro-Coder: `-noPLE` files and `--ple` in the agentionai fork since 2026-11-05; Gyro-Coder-L v1.1 (10-26), v1.2
  (10-29) [V].
- llama.cpp: #31469 (out-of-core MoE expert streaming, `--moe-direct-io`) **closed unmerged 2026-11-10** [V, checked];
  #32044 (NextN hidden width for Qwen4-Exp) open since 2026-11-10 [V, checked]; no PR for the trellis format found [V].

### Open before the tiers are fixed

- Gyro-Coder against the ISTA Coder: KLD/top-1 and our task benchmark; cost of the trellis dequant on gfx1200 and M1 Max.
- Ornith at IQ2_M / 3 bit against Q4_K_M on agent tasks (decides the PC file of the small tier).
- Usable RAM on a 32 GB PC next to desktop and agents.
- Whether the ISTA IQ4_NL n-gram shard can replace the Q8_0 table Gyro-Coder expects (same base table, untested).

## 16 GB Macs (2026-11-11, #108)

Research and estimates; nothing here is measured on a 16 GB Mac (we have none).

### Budget

| Item | Value | Source |
|---|---|---|
| `recommendedMaxWorkingSetSize`, 16 GB, macOS 14/15 | 11,453 MB (2/3 of RAM; M1, M1 Pro, M2, M4) | llama.cpp issues [#14199](https://github.com/ggml-org/llama.cpp/issues/14199), [#4482](https://github.com/ggml-org/llama.cpp/issues/4482), [#14527](https://github.com/ggml-org/llama.cpp/issues/14527), [#11083](https://github.com/ggml-org/llama.cpp/issues/11083) |
| same, macOS 26 and later | **12,713 MB** (~74 %; M2 Mac mini, M4, M1 Pro) | [#20704](https://github.com/ggml-org/llama.cpp/issues/20704), [PR #16559](https://github.com/ggml-org/llama.cpp/pull/16559), [nixpkgs #431464](https://github.com/NixOS/nixpkgs/issues/431464); our 32 GB M1 Max reports 26,800 MB, the same ratio |
| failure below the limit | 18 GB M3 Pro: 11.0 of 12.3 GB placed, run aborts with `kIOGPUCommandBufferCallbackErrorOutOfMemory` | [#19224](https://github.com/ggml-org/llama.cpp/issues/19224) |
| left after macOS, desktop, one browser | no measurement found; reports put the LLM budget at ~7 GB (unverified, secondary blogs) | — |

**Planning budget for model + KV + buffers: 8 GB** (estimate, unchecked; 7 to 9 GB). The working-set limit
(12.7 GB) is not the binding constraint; the rest of the desktop is, and memory pressure compresses or swaps the
desktop before Metal refuses anything. Raising the limit (`iogpu.wired_limit_mb`) is a system setting we do not use.

**Base chips** (all 16 GB Macs except M1 Pro / M2 Pro configurations): read bandwidth M1 68 GB/s (AnandTech, not
Apple), M2 and M3 100 GB/s, M4 120 GB/s ([Apple](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/));
8 to 10 GPU cores against our 32. **Internal SSD read** (Blackmagic, sequential): M2 256 GB (one NAND chip) **1.48
GB/s** ([Notebookcheck](https://www.notebookcheck.net/The-new-Apple-MacBook-Air-M2-has-arrived-Initial-impressions-and-benchmark-results-of-the-entry-level-SKU.635062.0.html),
[MacRumors](https://www.macrumors.com/2023/01/24/m2-mac-mini-256gb-slower-ssd/)); M1 256 GB 2.76; M3 and M4 256 GB
2.9; M4 512 GB 3.0 GB/s ([iMore](https://www.imore.com/mac/macbook/m3-macbook-air-got-a-massive-ssd-upgrade-over-m2-256gb-models-two-nand-chips-deliver-83-read-speed-boost),
[Gigazine](https://gigazine.net/gsc_news/en/20250317-apple-m4-macbook-air-benchmark)); our M1 Max reads 13 GB/s.

### Candidates

Sizes from the GGUF headers (range reads, 2026-11-11): Ornith-1.5-35B-A3B Q4_K_M dense 1.39 GB + token embedding
0.29 + experts 20.03 GB (41 blocks with MTP, 256 experts of 3 × 2048 × 512); Groot-19B-A3B-261023 Q4_K_M dense
1.35 + 0.29 + experts 10.01 GB (128 experts, same shapes). Experts at other types computed from 16.5 / 33.0 B expert
weights. KV (10 attention layers, 2 KV heads × 256) int8: 32K 0.34, 64K 0.67, 128K 1.34 GB; DeltaNet state 0.06 GB.

**Rate model** (estimate): time per token = dense bytes / (0.57 × bandwidth) + expert bytes / (0.29 × bandwidth) +
KV bytes / (0.57 × bandwidth); 0.57 and 0.29 are the shares of the read bandwidth our M1 Max kernels reach (dense
190 of 336 GB/s, experts 96 of 336, [densek](measurements/2026-11-11-mac-densek.md),
[expertk](measurements/2026-11-11-mac-expertk.md)). Unchecked: whether base chips with a quarter of the GPU cores
reach the same shares (the expert kernels are ALU-bound on the M1 Max, so probably less).

**(a) Resident**, decode at 32K context:

| Model, experts | Total + 32K / 64K / 128K | M1 / M2, M3 / M4 (tok/s) | 128K, M2 | Bound M2 |
|---|---|---|---|---|
| Groot IQ3_XXS | 8.7 / 9.0 / 9.7 GB (over budget) | 16 / 23 / 28 | 17 | 48 |
| **Groot IQ2_S** | 7.6 / 8.0 / 8.6 GB | 17 / 25 / 29 | 17 | 50 |
| Groot IQ2_XXS | 6.6 / 6.9 / 7.6 GB | 18 / 26 / 31 | 18 | 51 |
| Ornith, any type | ≥ 10.1 GB at IQ2_XXS | does not fit | | |

Groot at 2 to 2.5 bits per expert weight: quality unmeasured; the Coder variants on the PC (A/B/C, same expert-type
choice) will show what IQ2 costs on experts.

**(b) Streaming** with the M2 machinery: dense part, KV (64K) and buffers resident (~2.7 GB), the remaining ~5.3 GB as
expert cache, misses from the SSD. Hit rates are assumptions (unchecked; the only number is ~90 % with half the
experts cached, llama.cpp #29887 in [08](08-models.md)). Decode on an M2 / M3 at 100 GB/s; "overlapped" assumes the
prediction hides the reads, "serial" none of it:

| Model, experts | Cache share | Hits (assumed) | Miss bytes per token | 1.5 GB/s SSD | 2.9 GB/s SSD |
|---|---|---|---|---|---|
| **Ornith IQ3_XXS** | 42 % | 85 % | 58 MB | 11 to 21 tok/s | 15 to 21 |
| Ornith Q4_K_M | 26 % | 70 % | 183 MB | 6 to 8 | 8 to 16 |
| Groot Q4_K_M | 53 % | 90 % | 61 MB | 10 to 18 | 13 to 18 |

Streaming lets the stronger Ornith (vendor numbers, unchecked: SWE-bench Verified 79, Terminal-Bench 2.1 67.8) run on
16 GB at about the rate of a resident Groot at IQ2, and Groot at Q4 instead of IQ2. The SSD matters on the 256 GB M2
models only; elsewhere the compute part dominates.

**(c) The large model** (Coder variant A, dense part 4.45 GB): ~2.6 GB of expert cache (16 %), at an assumed 50 %
hit rate 310 MB of misses per token; 97 ms compute and 108 ms reads per token on an M2 / M3 with a 2.9 GB/s SSD,
**5 to 9 tok/s**. Does not carry: half the budget goes to the dense part, and the rate stays below every option above.

### Testing without a 16 GB Mac

On the 32 GB M1 Max:

- **Budget:** the backend gets a hard total budget (arena, KV, buffers) and refuses to exceed it; checked against
  Metal's `currentAllocatedSize` and the process footprint.
- **Memory pressure:** a ballast process that allocates and touches ~14 to 16 GB of random (incompressible) data
  leaves about what a 16 GB Mac has. It competes with whatever else runs on the machine, so only in short blocks and
  announced first.
- **SSD rate:** a rate limiter in our expert reader (1.5 / 2.9 GB/s) reproduces the bandwidth, not the latency of a
  one- or two-chip SSD.
- **Bandwidth and cores:** not reproducible. Kernel times scale by bandwidth only as an estimate.

Only a real 16 GB Mac shows: the shares the kernels reach on 8 to 10 GPU cores; how macOS compresses, swaps and
evicts under a desktop plus browser; whether Metal aborts with out-of-memory below the working-set limit
(#19224); the random-read behaviour of one- and two-chip SSDs; sustained clocks in a fanless Air.
