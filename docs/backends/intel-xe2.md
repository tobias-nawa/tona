# Intel Xe2 backend (Arc Battlemage: B580, B60, B70)

Candidate backend for Intel Arc Battlemage (Xe2-HPG), first the **Arc Pro B70** (32 GB GDDR6, 608 GB/s, 32 Xe cores, 256 XMX
engines, 367 INT8 TOPS dense, PCIe 5.0 x16, 160 to 290 W). Follow [blueprint.md](blueprint.md); evidence with
sources: [research/48-intel-xe2.md](../../research/48-intel-xe2.md). Nothing here has been run by us on Intel hardware. Marks: [V]
read, [S] search result only, [K] general knowledge, [D] derived (2026-11-14). Classes: [pc-12gb-vram.md](../hardware/pc-12gb-vram.md)
(B580), [pc-24gb-and-larger.md](../hardware/pc-24gb-and-larger.md) (B60 24 GB, B70 32 GB).

## Short answer

| Question | Answer |
|---|---|
| Route | native SYCL (DPC++) kernels, Level Zero on the host where SYCL's queue gets in the way. Vulkan ([vulkan.md](vulkan.md)) is the reach route: about 1.5 to 2x slower decode in the one engine with both, no persistent kernel |
| Make-or-break | `xe` does not kill long-running kernels (kernel and runtime source [V]); a spin on a CPU-written flag in host memory is **undocumented, atomics unsupported on discrete GPUs** [V]; grid-wide barrier only as experimental SYCL `root_group` with unstated co-residency [V]. With every expert resident on 32 GB only the barrier is needed. Phase 0 tests 1 to 3 settle it |
| Estimate, decode with MTP, large model resident, one B70 | **80 to 190 tok/s, central 110 to 150** ([method](../hardware/pc-24gb-and-larger.md#estimates)); prefill roughly 1,700 to 3,500 tok/s, unmeasured |
| Tier | 3 ([backends/README.md](README.md#tiers)), off by default, needs a named owner with the card |

## Route options

| | A. SYCL on Level Zero (recommended) | B. Level Zero + OpenCL C / SPIR-V | C. Vulkan |
|---|---|---|---|
| Kernels | SYCL, ESIMD (DPAS), `joint_matrix` for XMX | OpenCL C or SPIR-V from `ocloc`; SYCL kernels can be built ahead of time (`bmg`) | GLSL or Slang, `VK_KHR_cooperative_matrix` |
| Persistent kernel, barrier | `root_group` (experimental) or hand-built on device-USM atomics | same, plus `zeCommandListAppendLaunchCooperativeKernel` | **no** ([vulkan.md](vulkan.md#disadvantages)) |
| CPU flag in host memory | undocumented, test in phase 0 | same; `zeEventHostSignal` is the documented way | no |
| Matrix units | XMX (`joint_matrix`, int8 K=32, fp16 K=16 [V]) or ESIMD DPAS (llama.cpp PR 29864 [V]) | OpenCL subgroup matrix extensions [K] | cooperative matrix |
| Toolchain | oneAPI DPC++ 2025.3.3, `intel-compute-runtime` (26.35 in Fedora 44 [V]) | runtime and `ocloc` only | Mesa ANV |
| Public numbers (B70) | Qwen3.6-35B-A3B Q4_K_M decode 54.7 tok/s (16 % of bandwidth), 7B Q4_0 about 106 (about 67 %) | none | decode 1.25 to 2.2x lower than SYCL [V] |
| Cost | highest kernel work, best ceiling | highest host work | lowest effort and ceiling, also runs on NVIDIA and AMD |

Proposal: start with A; keep kernels compilable as SPIR-V so B is a host-side change; choose between persistent kernel, chained
short kernels in one command list and events per layer from the phase 0 numbers.

## What carries over from the reference

| Reference technique | On Xe2 |
|---|---|
| Shared host code (loader, tokenizer, quant formats, server, generator, speculation, `cpu/ref_*.cpp`, `budget.h`), pass interface, slots, MTP at widths 1 to 8, bit-identity rules | yes, unchanged |
| One launch per pass, grid barrier 0.75 us | maybe (test 3); fallback a few chained kernels |
| CPU to GPU flag in host memory (0.73 us) | **unknown**, not needed on 32 GB; `zeEventHostSignal` and device-side waits are the alternative |
| Expert pinning, slots, AVX-512 miss path | only 12 and 24 GB cards; on 32 GB all experts are resident |
| int8 mat-vec; f16 WMMA | rewrite (DP4A-class and XMX int8 [K]; XMX fp16 peak about 183 TFLOPS [D] against 93 measured on the reference) |
| fp8 prefill, int4 matrix path | **no counterpart known** (Xe2 XMX has no fp8 [K]) |
| HIP kernels | none carry over: other ISA, registers (128 or 256 per thread, sub-group 16 or 32 [V]), memory system |
| `TONA_DETERMINISTIC` | yes, fix the reduction order; sub-group size changes it if left to the compiler |

## Platform facts

| Fact | Value | Mark |
|---|---|---|
| Memory bandwidth | 608 GB/s spec; **no raw measurement found**; best public engine reaches 53 to 67 % in decode | [V], [D] |
| Usable VRAM | about 29.2 GiB on a headless B70 | [V] one report |
| Xe core | 8 vector engines, 8 XMX engines, 8 threads per vector engine; 128 KB SLM; 64 KB registers per vector engine; sub-group 16 or 32; work-group up to 1024 | [V] |
| Resident grid | 32 cores x 8 x 8 = 2,048 hardware threads; one work-group per core mirrors the reference's one block per CU | [D] |
| Job timeout on `xe` | 5 s default, 10 s maximum, **not applied to long-running (compute) VMs**; preemption timeout 640 ms, then an engine reset | [V] kernel source |
| Idle power | 38 to 51 W reported; one review says 12 to 15 W at the wall | [S] disagreeing |
| Driver | `xe`, kernel 6.12 or later; GuC firmware 70.60.0 may need a manual install; Resizable BAR required | [V], [K] |

## Watchdog and timeouts

Template: [the blueprint](blueprint.md#gpu-watchdogs-and-timeouts). On Linux `xe` compute queues run in long-running mode
with an unlimited GuC timeout [V]. The engine can reset on the 640 ms preemption timeout (a higher-priority context such as a
compositor becomes runnable while the kernel reaches no arbitration point) and on firmware or runtime bugs (compute-runtime 25.44
hotfix "for potential hang issues ... in very long (multi-day) running workloads" [V]). **Not measured:** whether a spinning kernel
is preempted cleanly, and what a reset does to a desktop on the same GPU. The B70 drives no display; next to the maintainers'
RX 9060 XT desktop it is a headless second GPU. Keep the reference rules anyway: wait cap 100 ms (`TONA_XE_WAIT_MS`), no blocking
runtime call while a kernel waits on the CPU, a recovery path that repeats the pass. Windows: TDR 2 s [V]; not the first target.

## Phases and gates

Follow [blueprint.md](blueprint.md). Specific here:

| Phase | Work |
|---|---|
| 0 (gate 0) | blueprint table plus: (1) raw bandwidth from SYCL and Level Zero, long and short buffers (the largest unknown); (2) **host flag** test: a kernel spins on a `malloc_host` flag the CPU writes, with plain loads, system-scope atomic loads where `usm_atomic_host_allocations` says yes, ESIMD loads with uncached hints; latency and whether the write is seen at all; (3) **long run and preemption**: 10 s and 60 s kernels headless, then with a second process drawing on the GPU, `xe` log and queue survival; (4) **grid barrier**: `max_num_work_groups_sync`, `root_group`, a hand-built barrier on device-USM atomics with 32, 64, 128 work-groups, cost in us, deadlock when over-subscribed; (5) dependent-kernel dispatch in one immediate command list with and without events, SYCL against raw Level Zero, host-signaled event wake-up; (6) XMX peaks (int8, fp16; reference f16 93, int8 183), fp8/int4 existence; (7) copy engine overlap, PCIe rates; (8) idle power, clocks after idle gaps, ReBAR and GuC version. Tests 2 and 3 are throwaway processes, warn the user first, wait cap 100 ms. Gate adds the pass-structure decision with its numbers |
| P | route A or B; pass structure; headless or shared card; model tier (the large model fits a headless B70 with about 0.5 GiB spare: shorter context or some experts off-card; medium and small fit easily); quant formats first (the model's K-quants and IQ types); `tona_backend_xe2`, `-DTONA_BACKEND_XE2=ON`; owner |
| M0 | expert and dense mat-vecs for the model's formats at B = 1 to 8; compare with llama.cpp SYCL on the same card (Q4_K_M reached 53 to 57 % of the bus, Q8_0 66 % after a fix, IQ4_NL 14 %; a fallback can be several times slower, issue 21517); gate: at least 70 % of measured bandwidth |
| M1 | resident, medium tier first; plus: recurrent (DeltaNet) state must not leak between prompts (a reported SYCL bug on hybrid models), `--check` passes at sub-group 16 and 32 if both are used |
| M2 | only 12 GB (B580) and 24 GB (B60); not planned for the first version |
| M3, M4 | MTP at widths 2 to 8 (compare llama.cpp's multi-column XMX kernels); prefill through XMX fp16 (int8 if the quality rule allows); other models as in the blueprint |

## Traps

1. Atomics on `malloc_host` are unsupported on discrete GPUs [V]: no wait on `atomic_ref` over it without the phase 0 test; no
   `malloc_shared` for concurrent host and device access.
2. `root_group` is experimental, its launch size limited by a query, forward progress not stated [V].
3. A failing int8 `joint_matrix` tile was traced to the IGC version [S]: pin compiler and runtime in the backend's `AGENTS.md`.
4. Assertions cost a B70 rig about half of its prefill speed in llama.cpp [V]: benchmarks build with `NDEBUG`, say so in every file.
5. Q8_0 and IQ4_NL decode were at 21 % and 14 % of the bus before a kernel existed [V]: measure every format.
6. An oversized KV pool dropped a B70 with error 20 [V]: keep a reserve and plan against real free VRAM (29.2 GiB, not 32).
7. SYCL exited with `UR_RESULT_ERROR_INVALID_MEM_OBJECT` after correct work [V]: destroy queues and free memory in a fixed order.
8. Idle power of 38 to 51 W per card [S] is a cost for an always-on service: measure it.
9. Missing GuC 70.60.0 or Resizable BAR cost speed or fail to probe [V, S]: check both before any measurement.
10. The runtime releases about every two weeks and llama.cpp kernels changed decode speed 3x in a month: pin versions per measurement.
11. IPEX-LLM is archived (2026-01) [S]; vLLM XPU lacked gated delta net kernels in April 2026 [V]: neither is a base or yardstick.

## The tier it would start at, and what hardware it needs

Tier 3: named owner, off by default (`-DTONA_BACKEND_XE2=ON`), no CI until the owner's machine runs it. Tier 2 needs three stable
months, public CI on a runner with the card and a second person who can build it ([backends/README.md](README.md#time-limits)). The
maintainers own no Intel hardware: the owner provides card, logs and measurements ([AGENTS.md](../../AGENTS.md#contributing)).

| Need | Why |
|---|---|
| **Arc Pro B70** (about 950 USD at launch) | the target; 32 GB makes the large model resident |
| a second GPU for the desktop (the RX 9060 XT works) | keeps the B70 headless; `amdgpu` and `xe` coexist [K] |
| free PCIe x16 slot, Resizable BAR on, about 230 W extra power | the card is x16 native |
| kernel 6.12 or later, `intel-compute-runtime` 26.35 or newer, GuC 70.60.0, oneAPI DPC++ 2025.3 or newer | the stack |
| 64 GB RAM or more | model file and page cache |

Stand-ins: **Arc B580 12 GB** (456 GB/s, PCIe 4.0 x8, same ISA) tests kernels, sub-group sizes and phase 0 tests 1 to 3, not the 32 GB
budget; **Arc Pro B60 24 GB** adds a larger model; Lunar Lake and Panther Lake laptops say little about discrete behavior.

## Open questions

[research/48-intel-xe2.md](../../research/48-intel-xe2.md#9-open-points-unchecked-in-this-run). First three: raw bandwidth, host-flag
visibility, barrier co-residency.

## First steps

1. `python3 tools/hw_report.py`; add `lspci -vv` (BAR size, link), `zeinfo` or `sycl-ls`, the `xe` firmware versions (sums, not dumps).
2. Phase 0 micro-benchmarks in `bench/xe2/`, smallest first.
3. Write the proposal with the pass-structure decision and the comparison to llama.cpp SYCL on the same card.
