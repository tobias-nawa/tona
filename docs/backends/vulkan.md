# Vulkan backend (cross-vendor)

A possible compute backend in Vulkan for NVIDIA, AMD (including RDNA2 and RDNA3 without a ROCm route) and Intel Arc: a real
option for reach, a weak one for the speed Tona is built for. Follow [blueprint.md](blueprint.md) if you
take it on. Nothing here has been run by us. Marks: [V] read, [S] search result only, [K] general knowledge (2026-11-11).

## Advantages

- **One code base for many cards**, including GPUs without a usable ROCm or CUDA story (RX 6000, Arc B580, older NVIDIA);
  see [pc-12gb-vram.md](../hardware/pc-12gb-vram.md).
- **Mature yardstick:** llama.cpp's Vulkan backend is competitive on AMD for token generation; ROCm leads prompt
  processing, MoE with CPU offload can reverse it (anecdotal: [discussions 10879](https://github.com/ggml-org/llama.cpp/discussions/10879),
  [15021](https://github.com/ggml-org/llama.cpp/discussions/15021), [Phoronix](https://www.phoronix.com/review/lemonade-vulkan-rocm/3)) [S].
  Its kernels are MIT and may be adapted with an origin note.
- **Cooperative matrix:** `VK_KHR_cooperative_matrix` exposes matrix units portably; NVIDIA also has
  `VK_NV_cooperative_matrix2` [S].
- No vendor toolkit beyond the driver and a shader compiler.

## Disadvantages

- **No forward-progress guarantee between workgroups:** a grid barrier or a spin on another workgroup's flag can deadlock
  when not all workgroups are resident (OOPSLA 2021, "Specifying and Testing GPU Workgroup Progress Models", 8 GPUs, 5
  vendors [S]). Grid barrier, megakernel and CPU flag in host memory cannot be assumed; the pass becomes many dispatches
  with a launch cost to measure (the reference's 2.1 us per kernel is a HIP figure).
- **Slower on the hot path by design:** about 1,000 dependent steps per token as dispatch boundaries; the Mac plan
  estimated 13 % of a resident pass at 2 us each (`research/40-mac-backend-proposal.md`).
- **Quant formats:** int8/fp8 dot paths and bit-exact dequant need vendor-dependent extensions
  (`VK_KHR_shader_integer_dot_product`, float8 [K]); check each.
- **Watchdogs:** Windows TDR 2 s [V]; Linux amdgpu `lockup_timeout` 2 s on gfx jobs (a Vulkan compute job is one and can
  reset the desktop, unlike an AMD KFD queue); NVIDIA with a display unchecked ([cuda.md](cuda.md#watchdog-and-timeouts)).
- **Determinism** (`TONA_DETERMINISTIC`): subgroup sizes differ per vendor and driver version and change reduction order;
  fix the order in the shader, not through subgroup operations.
- **Debuggability:** device-lost errors are opaque; no cross-vendor kernel profiler.

## Is it worth it?

Yes as a **reach** backend, no as first choice for a card with a vendor toolkit. Proposal for the human: NVIDIA gets CUDA
first, RDNA3 the HIP route; Vulkan covers what they leave out: RDNA2, Intel Arc (native SYCL route:
[intel-xe2.md](intel-xe2.md)), older NVIDIA. Tier 3. Vulkan earns more than it costs only as a **second route that an owner
of one card can test for everyone**, and only if the maintainers or a named owner can run the CI.

## Open questions

1. Dispatch and dependent-step cost per vendor driver: the phase 0 table of the blueprint, on at least two vendors.
2. Can the MoE expert matvec reach 70 % of bandwidth with IQ and K-quant formats in GLSL or Slang on each vendor? Compare
   llama.cpp's Vulkan shaders.
3. Which integer-dot and float8 extensions exist on the 12 and 16 GB class cards?
4. Does a compute job on the graphics queue next to the desktop stay under the 2 s limit with a 100 ms wait cap?

## First steps

1. `python3 tools/hw_report.py`, then `vulkaninfo --summary` (device, queue families, subgroup size, cooperative matrix and
   integer-dot properties); keep the sums, not the dump.
2. Phase 0 micro-benchmarks in `bench/vulkan/`: bandwidth, dependent dispatch, command buffer round trip, cooperative
   matrix throughput, host-visible flag latency.
3. Write the proposal with a comparison to llama.cpp Vulkan on the same card (setup named).
