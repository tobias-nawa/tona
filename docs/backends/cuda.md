# CUDA backend (NVIDIA)

Candidate backend for NVIDIA Ampere (sm_80/86), Ada (sm_89) and Blackwell (sm_100/120), typically 12 and 16 GB (RTX 3060 to
5070 Ti). Follow [blueprint.md](blueprint.md); this page is platform knowledge for phase 0 and the
proposal. Nothing here has been run on NVIDIA by us. Marks: [V] read, [S] search result only, [M] measured by us, [K]
general knowledge (2026-11-11).

Classes: [pc-12gb-vram.md](../hardware/pc-12gb-vram.md), [pc-16gb-vram.md](../hardware/pc-16gb-vram.md),
[pc-24gb-and-larger.md](../hardware/pc-24gb-and-larger.md).

## State of the art, 2026-11

| Topic | Finding | Mark |
|---|---|---|
| Persistent decode megakernel on NVIDIA | Cohere runs the decode loop of a 30B MoE as one persistent CUDA kernel, one block per SM, spin barriers; H100, BF16, batch 1 to 8; an RTX Blackwell version with FP8/FP4 announced; no license stated. <https://cohere.com/blog/megakernels> | [V] |
| Megakernel compiler for consumer cards | AutoMegaKernel (arXiv 2606.09682): Llama-family model as one persistent cooperative kernel for sm_80, sm_90, sm_120; int8 W8A16 beats CUDA-graphed cuBLAS bf16 at batch 1 on L4, L40S, A10G, RTX 5090, trails on A100 and H100 (cross-SM sync); largest model 1.1B, dense, position 0 only | [V] abstract |
| Numbers on the 3060 | 22.9 tok/s (`-ncmoe 25`), 35B-A3B IQ3_XXS; see [pc-12gb-vram.md](../hardware/pc-12gb-vram.md) | [V] |
| llama.cpp CUDA backend | the yardstick: CUDA graphs for decode, `mmq` integer kernels, flash attention, tensor-core paths. MIT: kernels may be adapted with an origin note | [K] |

## What transfers from the reference

| Reference technique | On CUDA | Notes |
|---|---|---|
| Megakernel, one launch per pass | yes, with cooperative launch | `cudaLaunchCooperativeKernel` guarantees residency, so a grid barrier is legal (`cudaDevAttrCooperativeLaunch`, [K]); grid limited to what fits at once (one block per SM). Measure the barrier on 46 to 84 SMs in phase 0 (reference: 0.75 us on 32 CUs) |
| CPU to GPU flag in host memory | possible (pinned memory, `volatile` reads) | forbids any blocking runtime call meanwhile; see watchdog |
| Expert pinning, slots in VRAM, AVX-512 CPU path | yes, shared logic in `engine/backend/budget.h` | PCIe 4.0 x8 on 4060 Ti and 5060 Ti 16 GB (about 14 GB/s usable [K]) makes streaming the bottleneck |
| WMMA f16 / bf16 | `nvcuda::wmma` or `mma.sync` (PTX), Ampere and later | |
| int8 matrix units | `mma.sync` s8, Ampere and later | candidate for 8-bit prefill |
| FP8 | Ada and later (e4m3, e5m2 `mma`), Hopper wgmma | the reference's fp8 GEMM maps to Ada and Blackwell; none on Ampere |
| FP4 / NVFP4 | Blackwell only | possible lossy requantization target |
| Int4 matrix path (reference: 341 TOPS on gfx1200) | not a general CUDA path | check in phase 0 |
| Launch cost 2.1 us per kernel | CUDA graphs cut it; measure | llama.cpp uses them for decode |

The reference's int8/fp8 choices (`k_wide`, int8 dense path) follow RDNA4 throughput ratios; derive yours from the phase 0
compute peaks.

## Watchdog and timeouts

- **Windows:** TDR, 2 s for a task the scheduler cannot preempt, bug-check after five hangs in a minute [V]
  ([Microsoft Learn](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery)).
  A persistent kernel on a display GPU is at risk: short launches there, or a GPU without a display.
- **Linux, display on the same GPU:** `cudaDevAttrKernelExecTimeout` reports the limit; old driver notes say a few seconds
  with a display and none without [S]. Whether Wayland triggers it with current drivers is **unchecked**; phase 0 queries
  the attribute and runs a throwaway long kernel only after warning the user.
- **Linux, headless or compute-only GPU:** no limit reported [S]; the engine's wait cap still applies.
- Design: a wait cap and pause like the reference (kernel ends voluntarily, host restarts from the step), or short launches
  in a CUDA graph; decide from the measured limit.

## Open questions

1. Grid barrier cost and launch overhead per SM count on an RTX 4070 and 5070.
2. Is a persistent kernel acceptable with a desktop on the card (attribute, TDR)? If not, a graph of short kernels.
3. Do IQ and K-quant matvecs reach 70 % of bandwidth on Ada with our block layout? Compare llama.cpp's `mmq`.
4. 12 GB and PCIe 4.0: is CPU compute of missing experts faster than streaming (DDR4 about 40 GB/s [K])?
5. Which int8 against f16 tensor-core ratio do the 4070 and 5070 give at sustained clock (reference: f16 93, int8 183 TOPS)?

## First steps

1. `python3 tools/hw_report.py`; open the class file for your VRAM and RAM.
2. Phase 0 micro-benchmarks in `bench/cuda/` (list of `research/measurements/2026-10-22-micro.md`): bandwidth, `mma` peaks
   for f16, int8, fp8 where present, launch, cooperative grid barrier, host flag latency, PCIe, kernel timeout attribute.
3. Write the proposal; the human decides between persistent kernel and graph.
4. M0: expert and dense matvec for your model's formats; compare with llama.cpp `mmq` on the same card (setup named).
5. M1 on the small tier that fits resident; the gates in the blueprint.
