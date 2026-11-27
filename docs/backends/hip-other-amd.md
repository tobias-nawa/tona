# HIP on other AMD GPUs (RDNA3, RDNA2)

Candidate backend for AMD cards with another ISA than the reference's gfx1200: RDNA3 (gfx1100, gfx1101, gfx1102) and RDNA2
(gfx1030 and relatives). RDNA4 gfx1201 is probably the reference code with a second build target, not a new backend
([pc-16gb-vram.md](../hardware/pc-16gb-vram.md#other-16-gb-cards)). Follow [blueprint.md](blueprint.md).
Nothing here has been run by us on these cards. Marks: [V] read, [S] search result only, [M] measured, [K] general
knowledge (2026-11-11).

Classes: [pc-12gb-vram.md](../hardware/pc-12gb-vram.md), [pc-16gb-vram.md](../hardware/pc-16gb-vram.md),
[pc-24gb-and-larger.md](../hardware/pc-24gb-and-larger.md).

## ROCm support, 2026-11

From the [ROCm support page](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html) [V] (ROCm 7.14.0 production):

| ISA | Listed on Linux (Ubuntu 24.04.4 and 22.04.5, RHEL 10.1 and 9.7) | Not listed |
|---|---|---|
| gfx1100 | RX 7900 XTX, 7900 XT, 7900 GRE; PRO W7900 and W7800 | |
| gfx1101 | RX 7800 XT, 7700 XT, 7700; PRO W7700, V710 (V710 only on Ubuntu) | |
| gfx1102 | none | RX 7600 and relatives |
| gfx1030 | PRO W6800 and V620 (V620 on Ubuntu only) | **every Radeon RX 6000 card** |

"Not listed" is unofficial: it may run (`HSA_OVERRIDE_GFX_VERSION`, [K]) without a support promise. The reference
toolchain is ROCm 10.1 under `work/rocm-10.1` ([reference-platform.md](reference-platform.md)); check which targets it
builds for before choosing a toolchain, and ask before a download above 5 GB.

## What differs from gfx1200

| Topic | RDNA4 gfx1200 (reference) | RDNA3 gfx110x | RDNA2 gfx103x |
|---|---|---|---|
| Matrix instructions | WMMA f16, bf16, fp8, int8, int4 (measured 93, 97, 186 TFLOPS, 183, 341 TOPS) | WMMA 16x16x16 f16, bf16, int8, int4, wave32 and wave64 [V] ([LLVM PR 164036](https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20251013/1756969.html)); **no FP8**; lower throughput per CU (not measured) | **none**; dot products through `v_dot4` / `sdot4` [K]; prefill on vector units |
| FP8 | yes | no | no |
| Infinity Cache | 32 MiB on the 9060 XT | 32 to 96 MB [K] | 32 to 128 MB [K] |
| PCIe | 5.0 x16 | 4.0 x16 (half the streaming rate) | 4.0 x16 |
| In-kernel sync | `grid.sync()` 0.75 us, host flag 0.73 us (measured) | to measure | to measure |

Consequence: kernels on the fp8 or int4 matrix path need a rewrite (the int8 path may map to RDNA3 WMMA); gfx1100 prefill is
slower per CU; gfx103x has no matrix units, so prefill is slower still while bandwidth-bound decode loses less.

## State of the art, 2026-11

rocWMMA covers gfx1100 to gfx1102 (ROCm 6.0 or later, <https://gpuopen.com/learn/wmma_on_rdna3/>); llama.cpp's rocWMMA flash attention
(`-DGGML_HIP_ROCWMMA_FATTN=ON`) shows little gain on gfx1100 for one user (issue 13565) [S]. Vulkan is often as fast for token
generation, ROCm leads prompt processing ([discussions 15021](https://github.com/ggml-org/llama.cpp/discussions/15021),
[10879](https://github.com/ggml-org/llama.cpp/discussions/10879)) [S].

## What transfers from the reference

| Technique | RDNA3 / RDNA2 |
|---|---|
| Megakernel with grid barrier and host flag | in principle yes (same HIP model); the risk is the reference's: MES and KFD queue handling and the desktop sharing the GPU. Read `research/09-gpu-reset.md` and `research/23-waiting-in-kernel.md` before the first CPU wait; they are gfx1200 findings, **unchecked on other ISAs**, and many gfx1101 and gfx1030 reset reports exist ([S], `research/09-gpu-reset.md`) |
| Expert pinning, slots, CPU AVX-512 path, budget rules | yes, shared |
| Pinned-buffer rule (allocate once, no re-registering) | yes; it fixed most aborts on the reference |
| int8 dense path | maps to RDNA3 WMMA int8; not to gfx103x |
| fp8 and int4 prefill | no |
| `isa-diff.py` | works per target; compile your target next to gfx1200 and compare nothing across ISAs |

## Watchdog and timeouts

Linux amdgpu: `lockup_timeout` default 2 s on the gfx ring (compositor and Vulkan jobs); the compute queue goes through KFD
with its own limits (queue preemption 9 s, MES fence 2.1 s on gfx12; gfx11 constants **unchecked**). A reset takes the
desktop with it; the reference mitigates with 100 ms pauses (`research/23-waiting-in-kernel.md`). Windows: TDR, 2 s [V].
Phase 0 reads your kernel's driver parameters and tests a wait in a throwaway process after warning the user.

## Open questions

1. Does the gfx1200 code build for gfx1100 once the fp8 parts are replaced, and with what loss? Compile both targets and
   compare kernel by kernel.
2. Is gfx11 MES/KFD behavior the same as gfx12 for a persistent kernel next to the desktop? Untried.
3. gfx103x: unofficial ROCm only; worth it, or is Vulkan the better route for RX 6000?
4. Prefill without matrix units on gfx103x: what rate; CPU or dot4 path?

## First steps

1. `rocminfo` to name your target; `python3 tools/hw_report.py`.
2. Add your gfx target to a scratch build of the reference (no commit) and list the kernels that fail to compile: the size
   of the port.
3. Phase 0 with `research/measurements/2026-10-22-micro.md`: bandwidth, WMMA throughput (f16, int8), `grid.sync`, host flag
   latency, PCIe.
4. Write the proposal: reuse the reference megakernel with a second ISA path (shared code, tier 3) or a separate backend
   directory; the human decides, because shared device code touches rule 1.
