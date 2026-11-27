# Intel Arc Pro B70: memory system, compute facts and other engines' Intel kernels (decode view for Tona)

Date of this run: 2026-11-17. Research only: source code read (shallow, sparse clones under
local shallow clones, never built or run), web pages and GitHub issues/PRs read. Nothing run on any GPU, no
build, no server, nobody contacted.

Marks: **[V]** code or page read by me in this run (path:line or URL); **[S]** search snippet only; **[K]** from
memory, unchecked; **[D]** derived by me (arithmetic shown); **[V2]** read by a helper agent in this run (web and
`gh`, URL and date reported to me), not re-opened by me. A number from a forum, blog, issue or PR is [V] only as
"that person wrote it", never our measurement.

Code revisions read (shallow clones, commit date):
compute-runtime `85bafcca` (2026-11-17), IGC `e48954ad` (2026-11-17), Linux master `666539ee` (2026-11-16, `drivers/gpu/drm/xe`),
llama.cpp `10152bc4` (2026-11-17, `ggml/src/ggml-sycl`), level-zero-tests `eaf05324` (2026-11-17),
vllm-xpu-kernels `93f275f6` (2026-11-13); single files fetched raw: intel/llvm `sycl` branch at `4a0e21d7`
(2026-11-17: `sycl_ext_oneapi_matrix.asciidoc`, `unified-runtime/source/adapters/level_zero/common/device.cpp`,
`.../level_zero/usm.cpp`, `sycl/test-e2e/Adapters/max_malloc.cpp`), compute-runtime master
`shared/offline_compiler/source/offline_compiler.cpp`, llama.cpp master `ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
OpenVINO master `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_int3.cl`.

## 1. Answer in short

| Question | Answer | Mark |
|---|---|---|
| Achievable DRAM read bandwidth on a B70 | **raw streaming read about 600 GB/s (98 to 99 % of 608)**; copy 264 to 273 GB/s payload = 528 to 546 GB/s read + write (87 to 90 %). Source: one lab's own Level Zero runtime, 16-byte loads, 32,768 work-items in groups of 128, SG32, 64 to 256 MiB buffers, 20 timed events each, sums checked (I recomputed the medians from the raw event stamps). **What real decode kernels reach on a B70 today:** 490 to 520 GB/s for whole-model dense decode with K-quants/Q8_0 in llama.cpp SYCL (80 to 86 %), 445 to 480 GB/s at verify width 8 with XMX, 283 GB/s for IQ3_S (47 %), 471 GB/s for an IQ4_NL kernel in an open PR, 50 to 68 GB/s for one lab's IQ2_S/IQ3_S expert GEMV; a persistent fp8 decode kernel about 408 GB/s per layer. **Plan with 450 to 520 GB/s effective for Tona's byte mix (dense streams 80 to 88 %, IQ experts 55 to 75 %), range 370 to 535.** Older figures (IDFS probe 443 GB/s, method unnamed) are superseded | [V] code/receipt/PR, [V2] IQ rows, [D] |
| Usable VRAM, one process, headless | **ECC first:** Intel's support article 000102907 (reviewed 2026-06-08): a B70 "is reporting 28 GB instead of the 32 GB", "ECC enabled by default", disable it in Intel Graphics Software (Windows); on Linux the lab's note uses `xpu-smi config --device N --memoryecc 0` plus a reboot (not checked by me). With ECC off: xe VRAM region 32,656 MiB = 31.89 GiB (32 GiB minus 112 MiB of flat-CCS and stolen memory). Level Zero reports `totalSize` 30.30 GiB with runtime 26.35 (= 0.95 x region); compute-runtime master (commit of 2026-10-20, **not** in 26.40.40196.8) raises the factor to 0.98 = 31.25 GiB. The factor is a reporting/limit value: the driver does **not** refuse a total beyond VRAM (overflow goes to system RAM, issue 1047) | [V] code + issue + Intel page, [D] |
| Tona fits? | 30.8 GB = 28.7 GiB of model. **With ECC on (the default) it does not fit** (about 28 GB visible, 0.95 x that = about 25 to 27 GiB reported). With ECC off it fits with 1.6 GiB spare under the 26.35 `totalSize`, 2.5 GiB under master, 3.2 GiB against the physical region; a desktop on the card (about 1.4 GB on the reference) leaves 0.2 to 1.1 GiB. Headless and ECC off are both preconditions. The earlier "29.2 GiB usable" is superseded | [V], [D] |
| Max single allocation | **No 4 GiB cap on Battlemage.** Xe2 forces stateless (64-bit) addressing, so `maxMemAllocSize` = `totalSize` (30.30 GiB reported on a B70). The 4 GiB cap (4 GiB minus 8 KiB) applies only to pre-Xe-HPC parts (Alchemist, Gen12). Above `maxMemAllocSize`: Level Zero `ze_relaxed_allocation_limits_exp_desc_t` with `ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE` (cap becomes the physical size), SYCL `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`, or debug key `NEOReadDebugKeys=1 AllowUnrestrictedSize=1` | [V] code |
| Resizable BAR | **Hard requirement for compute.** compute-runtime on `xe` refuses the device ("WARNING: Resizable BAR not detected for device", returns -1) when the CPU-visible VRAM is smaller than the region. `xe` tries to resize BAR2 at probe (32 GB supported on a B70) and says "Consider enabling 'Resizable BAR' support in your BIOS" if it cannot | [V] code |
| fp8 on XMX | **No** on Xe2 (IGC: `hasFP8Dpas()` only for Xe3P). XMX types: int8/uint8 (K=32), fp16, bf16 (K=16), tf32 (K=8, half rate), int4/int2 (ESIMD/vISA precisions, not in `joint_matrix`). dp4a: yes | [V] code |
| Decode estimate, Tona's large model, one B70, all resident, MTP 2.5 tokens/pass | **about 120 to 165 tok/s central (range 105 to 170)** at an effective 450 to 520 GB/s, method of `docs/hardware/README.md`; the memory-only bound is 220 to 255 tok/s. Inside research/48's range (80 to 190), now in its upper half, because B70 kernels already reach 80 to 86 % on dense K-quant streams | [D] |

Compute facts table (BMG-G31 / Arc Pro B70 unless stated):

| Item | Value | Source | Mark |
|---|---|---|---|
| Xe cores / XVE / XMX | 32 / 256 (8 per core) / 256 | spec; llama.cpp Vulkan device table `0xE223: // Pro B70 -> 32` (ggml-vulkan.cpp:17240-17241) | [V] |
| Clock | 2,800 MHz typical boost; TBP 230 W (Intel card), partners 160 to 290 W | PCGH spec table (2026-03-25) | [V] as written |
| Memory | GDDR6 19 Gbps, 256 bit = 608 GB/s, 32 GiB; ECC on by default, about 4 GB reserved when on (3.0) | PCGH; Intel article 000102907; [D] 19 x 256 / 8 = 608 | [V]/[D] |
| L2 (Intel calls it L3) | **24 MiB** on B70 ("confirmed by Intel" in PCGH's update note); 18 MB on B580 (Intel guide); runtime reads it from the hardware config, not hard-coded (`drm_neo.cpp:813`) | PCGH; Intel oneAPI GPU guide 2025.2 "Intel Xe GPU Architecture" table | [V] |
| L1 / SLM per Xe core | **256 KB shared L1/SLM block on Xe2-HPG (B580, BMG)**; 192 KB on Xe2-LPG (Lunar Lake). The IWOCL slide's "192 KB" is the Lunar Lake value. SLM carve-out encodings on BMG go up to **160 KB** per core (0/16/32/64/96/128/160 KB), Lunar Lake up to 128 KB. Chips and Cheese measured 96 KB of L1 on B580 (= 256 - 160) | Intel guide table; compute-runtime `release_helper_bmg_g31.cpp:20` -> `release_helper_preferred_slm_xe2_hpg_160k.inl:12-20` vs `release_helper_lnl.cpp` -> `..._xe2_lpg_128k.inl`; IGC `SystemThread.cpp:1825-1833` (SLM 128 or 160 configs); Chips and Cheese 2025-02-11 | [V], conflict resolved |
| SLM per work-group (what SYCL/L0 report) | **128 KB** (`maxProgrammableSlmSize = 128`, `localMemSize` = that) | `hw_info_bmg.cpp:50`, `device_caps.cpp:156` | [V] |
| Max work-group | 1,024 work-items (= 64 threads x SIMD16), hard limit field 2,048 | `device_caps.cpp:121-142`, `xe2_hpg_core/hw_cmds_base.h:38` | [V] |
| Sub-group sizes | 16 and 32 (no 8 on Xe2) | `xe2_hpg_core/hw_cmds_base.h:37`; Intel guide | [V] |
| Threads per XVE | 8 at 128 GRF, **4 at 256 GRF** (large-GRF mode); 2,048 or 1,024 resident threads on 32 cores | `hw_info_bmg.cpp:102`; `gfx_core_helper_base.inl:833-839`; `release_helper_base.inl:39-41` (`{4, 8}`) | [V], [D] |
| GRF | 64-byte (512-bit) registers; 128 or 256 per thread (Xe-HP..Xe2 "static register sharing with two configurations"); 8 KB or 16 KB per thread; 64 KB per XVE | IGC `Platform.hpp:1012` (`getGRFSize` 64 for Xe-HPC+), `Platform.hpp:1040-1043`; Chips and Cheese "64 KB register file" | [V], [D] 128 x 64 B x 8 |
| Largest LSC block message | 512 B per message (8 GRF x 64 B, D64) | IGC `Platform.hpp:772` | [V] |
| XMX shapes (`joint_matrix`, `intel_gpu_bmg_g31` listed) | u8/s8 -> s32: M <= 8, N = 16, K = 32; fp16/bf16 -> fp32 (or fp16/bf16 out): M <= 8, N = 16, K = 16 (plus composite 16x16x16, 1x64x16, 32x64x16 etc.); tf32: M <= 8, N = 16, K = 8. No fp8, no int4 in `joint_matrix` | intel/llvm `sycl_ext_oneapi_matrix.asciidoc` "Intel XMX Supported Combinations" (line 1107 ff.) | [V] |
| XMX in hardware | DPAS precisions in IGC: U8, S8, U4, S4, U2, S2, BF16, FP16, TF32; BF8/HF8/E2M1 exist in the enum but `hasFP8Dpas()`/`hasFP4DPAS()` are Xe3P only; exec size 16 DPAS on Xe-HPC and later | IGC `common/Types.hpp:32-47`, `CISABuilder.cpp:7117-7147`, `Platform.hpp:609-615` | [V] |
| XMX throughput | per Xe core FP16 2,048 ops/clk, INT8 4,096 ops/clk (IWOCL 2026 slide, earlier run); 32 x 4,096 x 2.8 GHz = 367 TOPS int8, 183 TFLOPS fp16; tf32 "half the K per instruction" | earlier report point 5; llama.cpp `fused-gemm.cpp:27-29` | [V], [D] |
| Vector engine | 512-bit XVE, 16 fp32 lanes: 32 x 8 x 16 x 2 x 2.8 GHz = 22.9 TFLOPS fp32 | [D] from Intel guide register width | [D] |
| dp4a | yes (`hasHWDp4AddSupport()` = Gen12 and later); OpenCL `cl_khr_integer_dot_product` advertised | IGC `Platform.hpp:368`; compute-runtime `compiler_product_helper.cpp:75` | [V] |
| Preemption | mid-thread preemption is the default mode on BMG | `hw_info_bmg.cpp:46` | [V] |
| Cooperative group count (what `zeKernelSuggestMaxCooperativeGroupCount` computes) | available threads (8 or 4 per XVE by GRF) / threads per work-group, capped by 32 barrier registers per core and by 128 KB SLM per core | `kernel_helpers.cpp:34-79`, `constants.h:97` | [V] |

## 2. Bandwidth (open point 1)

### 2.1 What is published

| Card | Tool / workload | Result | Share of spec | Source | Mark |
|---|---|---:|---:|---|---|
| B70 | home-built "memory-bandwidth probe via large allocations and reads", PyTorch 2.10 XPU in `intel/vllm:0.10.2-xpu`, kernel 6.17 | 443 GB/s | 73 % | IDFS blog (April 2026), method see 2.3 | [V] as written |
| 2x B70, TP2 | vLLM `--monokernel` (persistent per-layer kernels), Qwen3.6-35B-A3B fp8, decode, 1 request, MTP off, TPOT 5.31 ms | about 345 GB/s per card end to end | 57 % | vllm-project/vllm#65566 and vllm-xpu-kernels#685 (open, 2026-11-04), arithmetic in 2.2 | [V] as written, [D] |
| same | one GatedDeltaNet decoder layer, device time 83.7 us | about 408 GB/s per card | 67 % | same, 2.2 | [D] |
| same | MTP verify layers 1x2, 1x4, 1x8 tokens (135 / 198 / 315 us) | about 316 to 412 GB/s (depends on the unknown expert overlap) | 52 to 68 % | same, 2.2 | [D] |
| **B70** | **own Level Zero runtime, kernel `stream16_sum_u64_v1`: 32,768 work-items, groups of 128, SG32, one 16-byte `vec<uint32,4>` load per iteration, grid-stride, u64 sum per lane; pseudo-random data (`index * 1664525 + 1013904223 + salt`), every sum checked** | **256 MiB: 601.1 / 601.5 GB/s median of 20; 128 MiB: 598; 64 MiB: 589 to 593** | **98.9 %** | steveseguin/b70-optimization-lab (Unlicense), receipt `experiments/own-xpu-runtime/stage2/packet7-kopt2/receipts-card3-20261011/ceiling/native.json`, kernel `kernels.cpp:155-173`, data `reference.hpp:53-58`, driver 1.15.38308, ECC state of that run not recorded | [V] receipt and code, [D] medians recomputed by me |
| B70 | same receipt, SYCL `memcpy` device to device | 264 to 273 GB/s payload = 528 to 546 GB/s read + write | 87 to 90 % | same | [V], [D] |
| B70 | llama.cpp SYCL whole-model decode, Qwen3.8-27B (dense, qwen35 arch), `llama-bench -fa 1 -r 3`, Windows 10, oneAPI 2026.1 | Q6_K 22.4 GB x 22.8 t/s = 511 GB/s; Q8_0 29.0 GB x 18.0 = 522 GB/s (file size x rate; the file also holds the token-embedding table that decode does not stream, about 1 GB [K], so the streamed rate is about 4 % lower) | 80 to 86 % | llama.cpp PR 29864 (merged 2026-10-02) | [V] PR, [D] |
| B70 | same PR, verify-like batches with XMX (`pp8` = 8 tokens per step) | Q8_0 133 / 8 x 29.0 = 482 GB/s; Q6_K 159 / 8 x 22.4 = 445 GB/s; Q4_K_S 268 GB/s | 44 to 79 % | same | [V], [D] |
| B70 | llama.cpp `test-backend-ops perf` MUL_MAT 4096 x 14336 x 1 (33 to 40 MB, partly L2-resident) | Q5_K 583 to 594; Q4_K 596; IQ4_NL 105 -> 471 (open PR 31998); IQ4_XS 359 -> 433 | 17 to 98 % | PRs 29375, 31528, 31998 | [V2], [D] |
| B70 | llama.cpp whole-model decode IQ3_S 27B (12.12 GB x 23.39 t/s) after the IQ3 reorder | 283 GB/s | 47 % | PR 29107 comment 2026-10-02 | [V2], [D] |
| B70 | one lab's own GEMV, layer 0/3 weights of an IQ3_XXS MoE | F16 router 289 GB/s; Q6 attention 78 to 93; IQ2_S/IQ3_S experts 50 to 68 ("the per-weight decode (codebook lookups, sign bits, scales) is the bottleneck") | 8 to 48 % | lab `packet7-kopt2/README.md:318-341` | [V] as written |
| B580 | clpeak 1.1.2 "Global Memory Bandwidth", OpenCL, Phoronix | 417.1 GB/s (2024-12-10), 418.0 (2025-01-13) | 91.5 % | OpenBenchmarking results 2412128-PTS-GPUCOMPU05 and 2501131-PTS-GPUCOMPU20 | [V2] |
| B580 | ProjectPhysX OpenCL-Benchmark | coalesced write 453 to 474; "read" 520 to 586 is above the 456 spec, so not a DRAM figure | write 99 to 104 % | Phoronix 2412136, repo README | [V2], flagged |
| B580 / B70 | Intel's roofline constants in torch-xpu-ops | 410 / 532 GB/s (origin not stated) | 90 / 87.5 % | intel/torch-xpu-ops issue 3784 (2026-05-29) | [V2] |
| B70 | llama-2-7B Q4_0 decode (3.82 GB), Vulkan, Windows | 113.6 t/s -> 434 GB/s | 71 % | llama.cpp discussion 10879 (2026-10-25) | [V2], [D] |
| B60 | same model, SYCL `-fa 1`, compute-runtime 26.27, kernel 7.0 | 84.3 t/s -> 322 GB/s | 71 % of 456 | same discussion (2026-11-05) | [V2], [D] |

[V2] = read by a helper agent in this run (web and `gh`), reported to me with URL and date, not re-opened by me.
Not found by anyone: BabelStream, mixbench or ze_peak results on any Battlemage card; a clpeak number for a B70;
a raw bandwidth figure for a B60.

### 2.2 Arithmetic for the vLLM mono-decode rows

Model dimensions from `Qwen/Qwen3.6-35B-A3B/config.json` (hidden 2048, 40 layers, 30 GatedDeltaNet + 10 full
attention, 16 K / 32 V linear heads of 128, 16 Q / 2 KV heads of 256 with output gate, 256 experts, top 8,
expert and shared intermediate 512, vocab 248,320) [V]. Per rank (TP2), fp8 weights (1 byte), fp16 router and
lm_head (2 bytes), as the PR states [V]:

- expert: w13 2048 x 512 + w2 256 x 2048 = 1.57 MB; 8 routed + 1 shared = 14.2 MB
- GatedDeltaNet in_proj (q 1024, k 1024, v 2048, z 2048, b 16, a 16 rows) x 2048 = 12.6 MB, out_proj 2048 x 2048 =
  4.2 MB, router 256 x 2048 x 2 B = 1.05 MB, fp32 recurrent state read + write 2 x 16 x 128 x 128 x 4 B = 2.1 MB
- **GatedDeltaNet layer: 34.1 MB / 83.7 us = 408 GB/s** [D]; full attention at 1K context 29.9 MB / 98.2 us = 304
  GB/s, at 32K (KV 32 MB more) 62.4 MB / 152.2 us = 410 GB/s [D]
- per token per rank: 30 x 34.1 + 10 x 29.9 MB + lm_head 124,160 x 2048 x 2 B = 0.51 GB -> 1.83 GB; / 5.31 ms TPOT =
  **345 GB/s per card**, everything included (launch gaps between layers, sampling, two in-kernel all-reduces per
  layer over PCIe 5.0 x8) [D]
- verify 1 x T: dense part 19.9 MB + union of experts (between 0.8 x 8T and 8T experts, unknown) + per-token state
  copies: 1x2 42.7 to 47.8 MB / 135.4 us = 316 to 353 GB/s; 1x4 65 to 75 MB / 198 us = 328 to 379 GB/s; 1x8 109 to
  130 MB / 314.8 us = 348 to 412 GB/s [D]

What the PR shows beyond the number: 32 work-groups x 512 work-items (32 SIMD16 sub-groups) with 256 GRF fill the
32 Xe cores "exactly once"; stages hand off through arrival counters (release RMW after a work-group barrier),
7 or more grid-wide hand-offs per layer, no device-scope acquire fence "(it would invalidate L1 on every thread)";
weights are read as `[N][K]` rows with 16 fp8 per lane, software-pipelined, the next stage's weights preloaded
into registers while waiting; batched rows (2/4/8) and verify use DPAS with activations as the A matrix padded to
8 rows; plain stores into the peer card's memory over PCIe P2P work on that machine (2x B70, PCIe 5.0 x8, Xeon
w5-3435X, kernel 6.17, compute runtime 26.27.39122.11). [V] PR text, "that person wrote it"; it is an open PR,
not merged, AI-assisted per its own text.

### 2.3 Method notes

- **ze_peak** (Intel's own, level-zero-tests `perf_tests/ze_peak`): read-only kernels, each work-item sums 16
  elements of float, float2 ... float16, two variants (stride = local size or stride = global size), best of the
  two reported; buffer = min(maxMemAllocSize / 8, 2^29 floats) = 2 GiB on a B70; work-group = `maxGroupSizeX`
  (1,024); GB/s counts read bytes only (the 1-in-16 write is ignored) (`src/global_bw.cpp:24-29, 270-303`,
  `include/ze_peak.h:44,119`, `kernels/ze_global_bw.cl:16-26`, `src/ze_peak.cpp:704-720`) [V]. It is a pure read
  test at maximum work-group size; it does not measure copy.
- **The lab's 601 GB/s kernel** reaches the spec with modest occupancy: 32,768 work-items at SG32 = 1,024
  hardware threads, i.e. 32 per Xe core or 4 per XVE, one 16-byte load in flight per lane per iteration [V code,
  D]. Bandwidth rises with buffer size (589 -> 598 -> 601 GB/s for 64 -> 128 -> 256 MiB), the opposite of what L2
  help would give; a fit t = f + bytes / R on the 64 and 256 MiB medians gives R = 606 GB/s (99.6 %) and f = 3.2 us
  fixed per launch [D: (446.4 - 114.0) us for (268.4 - 67.1) MB]. Buffer compression (flat CCS) is unlikely to
  inflate it: the data are a pseudo-random affine sequence and BMG's capability table has
  `ftrRenderCompressedBuffers = false` (`hw_info_bmg.cpp:59`) [V], [D].
- **IDFS 443 GB/s** (re-opened by the helper): the only description is "a memory-bandwidth probe via large
  allocations and reads" in a PyTorch 2.10 XPU container; no code, no buffer size, no read/copy distinction [V2].
  It is 74 % of what a plain SYCL kernel reaches on the same card type, so it measured the framework, not the DRAM.
- **Access pattern that a decode kernel should copy ([D] from the code above):** 16-byte loads per lane,
  sub-group-contiguous (SG32 x 16 B = 512 B, SG16 x 16 B = 256 B per instruction; 2D block loads up to 512 B per
  message), 32 or more hardware threads per Xe core, grid-stride or one long row run per sub-group. The lab
  kernel hits 99 % with exactly that; the vLLM mono kernel (32 x 512 items, 256 GRF) and reordered llama.cpp
  kernels (256-item groups, one row per sub-group) are variants. Little's law: 608 GB/s x 300 ns (B580 unloaded VRAM
  latency, Chips and Cheese) = 182 KB in flight, 5.7 KB per Xe core; 32 threads x 32 lanes x 16 B = 16 KB per core
  in the lab kernel, enough even at twice the latency under load [D]. What costs bandwidth in practice is not the
  load pattern but the work per byte: codebook decode for IQ types, per-column re-unpacking at widths above 1
  (PR 29864: "about 5 lane operations per dp4a per extra column"), launches and host syncs per MoE step [V], [D].
- **B580/B60 to B70 scaling:** same 19 Gbps GDDR6 and the same Xe2 core; bus 192 bit -> 256 bit, so a B580/B60
  read result scales by 608 / 456 = 1.33 if the share of peak carries over [D]: the B580 clpeak 417 GB/s maps to
  556 GB/s on a B70, below the 601 the lab measured directly. The B70 has 24 MiB L2 against 18 MB, 32 cores
  against 20 [V].
- **Read versus copy:** read reaches 99 %, copy 87 to 90 % (read + write counted). A decode pass is almost pure
  read, so the read figure is the relevant ceiling [D].

## 3. Usable VRAM, allocation limits, ReBAR (open point 2)

### 3.0 ECC (decides whether the model fits)

- Intel support article 000102907 (applies to B50, B60, B65, B70; last reviewed 2026-06-08): a B70 "is reporting
  28 GB instead of the 32 GB"; "ECC reduces the total VRAM for stability"; configuration noted as "ECC enabled by
  default"; the fix described is Intel Graphics Software on Windows 11, setting persistent [V]
  (https://www.intel.com/content/www/us/en/support/articles/000102907/graphics.html).
- Linux: the lab's note `docs/b70-ecc-and-vram.md` gives `xpu-smi config --device N --memoryecc 0`, then reboot,
  check "Memory ECC Current/Pending"; their two B70 reported ECC disabled on 2026-08-30; Intel publishes no
  ECC-on/off bandwidth comparison [V] as written; the `xpu-smi` option itself not checked by me.
- compute-runtime's 98 % commit says the xe KMD reports the region "already net of the firmware and ECC
  reservations" [V], so with ECC on the region and everything derived from it shrink by about 4 GB.
- Consequence: 28 GB visible x 0.95 = about 24.8 GiB reported [D, assuming 28 GB means 28 GiB of region it is
  26.6 GiB]; either way below Tona's 28.7 GiB. **ECC off is a precondition**, and the installer must check it
  (phase 0 item). The issue-1047 card (region 32,656 MiB) evidently ran with ECC off [D].

### 3.1 How the number reaching the application is built (code)

1. `xe` sets the VRAM region to the address of the flat-CCS base (Xe2: CCS = tile size / 512 = 64 MiB for 32 GiB),
   below the GSM ("stolen") base: `tile_vram_size()` and `get_flat_ccs_offset()` in `drivers/gpu/drm/xe/xe_vram.c:64-191`
   [V]. The query returns `total_size` = TTM manager size (`xe_query.c:280-304`) [V]. On a B70 this is 32,656 MiB
   (issue 1047's driver log: `max=32656MB`) [V], i.e. 32,768 - 32,656 = 112 MiB below 32 GiB [D].
2. compute-runtime: `globalMemSize` = region size x `getPercentOfGlobalMemoryAvailable()` =
   `isLocalMemorySizeNetOfKmdReservations() ? 0.98 : 0.95` (`drm_memory_manager.cpp:2339-2343`); the xe helper
   returns true (`ioctl_helper_xe.h:159`) since commit a9797c6e3c of 2026-10-20 "use 98% of local memory as global
   memory size on xe" ("the xe KMD reports the local memory region size already net of the firmware and ECC
   reservations"). That commit is **not** in release 26.40.40196.8 (GitHub compare: diverged, release behind by 325)
   [V]. So released drivers give 0.95 x 31.89 = **30.30 GiB**, exactly what issue 1047 printed for a B70 on 26.35
   ("totalSize 30.30 GiB, maxMemAllocSize 30.30 GiB, sysman free 31.66 GiB") [V], [D]; master will give 31.25 GiB [D].
3. Level Zero `zeDeviceGetMemoryProperties`: `totalSize` = `globalMemSize`, `physicalSize` from the KMD
   (`level_zero/core/source/device/device.cpp:694, 725`) [V]. SYCL `global_mem_size` returns the **physical** size
   if available, else `totalSize` (UR `device.cpp:204-220`); `ext_intel_free_memory` = min(global, sysman free)
   (UR `device.cpp:986-992`) [V]. llama.cpp prints "32656 MiB, 32571 MiB free" for the same card (issue 1047) [V].

### 3.2 Single-allocation cap

- `maxMemAllocSize` = `getGlobalMemorySize(single sub-device)`, then reduced to 4 GiB - 8 KiB
  (`maxStatefulBufferSize`) **only if** `!isForceToStatelessRequired()` (`device_caps.cpp:69-93`, `constants.h:36-39`)
  [V]. BMG includes `compiler_product_helper_xe_hpc_and_later.inl` (`enable_compiler_product_helper_bmg.cpp:13`),
  whose `isForceToStatelessRequired()` returns true unless the debug key `DisableForceToStateless` is set
  (`compiler_product_helper_xe_hpc_and_later.inl:13-19`); Gen12/DG1/ADL use the "before Xe-HPC" variant that returns
  false [V]. So on BMG: **cap = `totalSize`** (30.30 GiB on 26.35) [V], [D].
- Kernels: the runtime adds `-ze-intel-greater-than-4GB-buffer-required` to every L0 module build when stateless is
  forced (`module_imp.cpp:207-212`), and so does `ocloc` for AOT builds (`offline_compiler.cpp:1608-1610`) [V].
  llama.cpp passes the same flag explicitly (`ggml-sycl/CMakeLists.txt:219-226`) [V].
- Above the cap: L0 `ze_relaxed_allocation_limits_exp_desc_t` in `pNext` of `zeMemAllocDevice` with
  `ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE` sets `relaxedSizeAllowed` (`context.cpp:116-129`); with it the
  limit becomes the physical size (`checkMemSizeLimit`, `context.cpp:263-287`) [V]. Debug key
  `AllowUnrestrictedSize` (with `NEOReadDebugKeys=1`) does the same (`debug_variables_base.inl:736`) [V]. SYCL:
  `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` makes UR report global size as `max_mem_alloc_size` and add the relaxed
  descriptor (UR `device.cpp:505-513, 2046-2054`, `usm.cpp:177-181, 223-225`; e2e test `max_malloc.cpp` runs with it
  and `SYCL_PROGRAM_COMPILE_OPTIONS=-ze-intel-greater-than-4GB-buffer-required`) [V]. llama.cpp allocates model
  buffers with `zeMemAllocDevice` plus the relaxed descriptor (`ggml-sycl/common.cpp:112-139`) [V].
- For Tona: one 28.7 GiB allocation is legal without any switch on BMG; a few large allocations (per tensor class)
  are the cleaner choice anyway [D].

### 3.3 Overcommit and deferred backing (new, matters for a 29 GiB model)

- compute-runtime issue 1047 (open, 2026-11-03, B70, runtime 26.35.39758.17 and 26.31, kernel 7.0): `zeMemAllocDevice`
  checks each allocation's size, not the total; "past 50% of VRAM the driver uses deferred backing (no VRAM reserved
  until first use)"; 36 x 1 GiB succeed on a 30.30 GiB device, sysman free does not see deferred allocations, the
  overflow goes to system RAM, a 45 GB model load page-faulted and once drained half the host RAM [V] as written. The
  driver log line comes from `DrmMemoryManager::isDeferBackingMemoryPressureReached` (`drm_memory_manager.cpp:2670-2697`);
  switch `EnableDeferBacking` (`debug_variables_base.inl:611`) [V]. Disabling it did not stop the overcommit (issue) [V].
- llama.cpp works around it with its own accounting, `GGML_SYCL_BMG_VRAM_CHECK=1` default (`common.cpp:152-185`) [V].
- Tona consequence: count our own bytes against `totalSize`, touch every weight page at load (upload does that), and
  never rely on an allocation failure as the out-of-memory signal [D].

### 3.4 ReBAR

- compute-runtime: `if (getMemoryInfo()->isSmallBarDetected()) { fprintf(stderr, "WARNING: Resizable BAR not detected for device %s\n", ...); if (!ioctlHelper->isSmallBarConfigAllowed()) return -1; }`
  (`drm_neo.cpp:690-695`), and `IoctlHelperXe::isSmallBarConfigAllowed()` returns false (`ioctl_helper_xe.h:158`);
  small BAR = any region with `cpuVisibleSize < probedSize` (`memory_info.cpp:37-39`) [V]. So without a full BAR the
  B70 is not usable for Level Zero/OpenCL/SYCL at all on `xe` (i915 allowed it; i915 does not drive BMG) [V], [D].
- `xe`: `xe_pci_rebar_resize()` resizes BAR2 to the largest supported size at probe, or to `xe.vram_bar_size=<MiB>`
  (negative disables) (`xe_pci_rebar.c:38-106`, `xe_module.c:47-49`); it needs a 64-bit root window above 4 GiB
  ("platform support is missing. Consider enabling 'Resizable BAR' support in your BIOS") [V]. A B70's `lspci`:
  "BAR 2: current size: 32GB, supported: 256MB ... 32GB" (issue 1047) [V].
- Next to the AMD desktop card the board must place two large BARs (32 GB + 16 GB) above 4 GiB; usual on AM5 with
  "Above 4G decoding" [K], unchecked on the maintainer's board.

## 4. Compute facts for kernel design (open point 3)

The table in section 1 is the answer. Design notes that follow from it [D]:

- **Occupancy versus registers.** 256 GRF halves the threads (4 per XVE, 1,024 on the card). For a bandwidth-bound
  mat-vec this is the better trade when the registers hold the in-flight weight loads (vLLM's mono kernel does
  exactly this); 128 GRF only helps latency hiding by thread switching. Cooperative launch counts follow from it:
  one 512-item SIMD16 group = 32 threads, 32 per core at 256 GRF -> one group per core -> 32 groups (matches the
  vLLM kernel) [V code, D].
- **SLM.** 128 KB per work-group is what the API gives; with one group per core the whole 128 KB is available for
  activations (a 2048 or 4096 hidden vector in fp16 is 4 to 8 KB, 8 MTP rows 32 to 64 KB) [D].
- **MTP widths 2 to 8 map onto DPAS M <= 8** (repeat count): one DPAS tile row per verify token; weights as the B
  matrix (N = 16 output columns, K = 32 int8 or 16 fp16). The vLLM kernels and llama.cpp's ESIMD paths use this
  shape [V code].
- **IQ formats** need codebook lookups per 8 weights; the B70 has about 11.5 T lane-ops/s (32 x 8 x 16 x 2.8 GHz)
  against 1.46 T weights/s at 608 GB/s and 3.33 bits per weight: about 8 lane-ops per weight at full bandwidth,
  enough at width 1, tight at width 8 if each column redoes the dot product on the vector engines; decoding once to
  int8 and using DPAS removes the width factor [D, rough].
- **No fp8:** the reference's fp8 prefill has no XMX counterpart; int8 DPAS at 367 TOPS is the replacement [V], [D].

## 5. Other engines on Xe2 (open point 4)

### 5.1 llama.cpp SYCL (code read at `10152bc4`)

| Item | Finding | Where | Mark |
|---|---|---|---|
| Sub-group | 16 on Intel (`GGML_SYCL_WARP_SIZE=16`) | `ggml-sycl/CMakeLists.txt:219-220` | [V] |
| MoE decode (1 token) | fused `mul_mat_vec_q_moe`: grid = experts used x rows, **one work-group of 16 work-items (one sub-group, one hardware thread) per output row**; covers Q4_0..Q8_0, Q2_0, K-quants, MXFP4, NVFP4 and **all IQ types incl. IQ1_M, IQ2_S, IQ3_S, IQ4_NL** | `mmvq.cpp:2630-2707` (kernel, launch with `GGML_SYCL_MMV_Y = 1`), `mmvq.cpp:2709-2737` (types) | [V] |
| MoE decode, reordered (SoA) layout | only Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4 | `mmvq.cpp:2739-2751` | [V] |
| MoE with more than one token (MTP verify) | `ne12 != 1` skips the fused path (`ggml-sycl.cpp:5586-5603`); then a device-scheduled grouped dequant + XMX GEMM (`fused-gemm.cpp:1387-1418`) for IQ4_NL, IQ3_S, IQ4_XS, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M, Q8_0, Q2_K..Q6_K (`fused-gemm.hpp:62-81`); **Q2_0 is not in that list** and falls back to the per-expert loop with a blocking `stream->wait()` and a host copy of the ids (`ggml-sycl.cpp:5791-5800`) | as given | [V] |
| Dense mat-vec | reordered (SoA) MMVQ for Q1_0, Q4_0, Q8_0, Q2_K..Q6_K, IQ3_XXS, IQ3_S (`ggml-sycl.cpp:4374-4390`): work-group 256 = 16 sub-groups, one row per sub-group (`mmvq.cpp:850-866`, kernel `mmvq.cpp:64-115`); other IQ types (IQ1_M, IQ2_S, IQ4_NL) unreordered, 16-item groups (`mmvq.cpp:1944-1964`) | as given | [V] |
| XMX multi-column (2 to 80 columns) | ESIMD DPAS only for Q2_K..Q6_K and Q8_0 on reordered weights (`mmvq.cpp:2087-2163`, default max 80 columns `common.hpp:247`); not for IQ types | as given | [V] |
| Tuning evidence in code | Q4_K two-rows-per-sub-group crossover at 6,144 to 6,176 rows, "Derived on Intel Arc Pro B70 with `test-backend-ops perf -o MUL_MAT`" (`mmvq.cpp:44-62`); tf32 DPAS is "half the K per instruction", "B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s" (`fused-gemm.cpp:27-29`); SYCL `max_compute_units` counts XVEs, i.e. a 20-core B60 reads as 10 "16-EU cores" (`topk-radix.cpp:215-222`) | as given | [V] |
| Memory | model buffers via `zeMemAllocDevice` (avoids "DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference"), relaxed size descriptor, own VRAM accounting on BMG | `common.cpp:108-139, 152-185` | [V] |

Reading for Tona: the IQ decode path that llama.cpp uses on Intel is one sub-group per row with tiny work-groups and
unreordered AoS blocks, which is where the low IQ bandwidth shares come from; the reordered SoA layout (quants and
scales in separate streams) is what lifted Q4_0/Q4_K on Intel. Tona's own layout should be SoA from the start [D].

### 5.2 llama.cpp Vulkan on Xe2 (ggml-vulkan.cpp master)

- Xe2 detected as `INTEL_XE2` ("Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8",
  line 84); cooperative matrix tile 8 x 16 x 16 with sub-group 16 (lines 1488-1497) [V].
- MMVQ (int8 dot) for single-token decode on Xe2 only for Q2_0, Q2_K, Q3_K, Q6_K; other types use the fp16
  dequant mat-vec unless k >= 2048 (lines 7125-7140) [V]. Intel collectives disabled (PR 14316 reference, line 4154) [V].

### 5.3 vLLM XPU (vllm-xpu-kernels) and the mono-decode PRs

- Fused MoE on Xe2 = sycl-tla (CUTLASS for SYCL) grouped GEMM with weights bf16/fp16, fp8 (per-tensor, MXFP8,
  block), int4, MXFP4, dequantized in registers; 6 k-tiles prefetched "to hide the packed-weight decode"
  (`csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp:63-65, 335-347`); decode uses an 8-row tile up to 64 routed rows,
  measured on BMG (`moe_shared_fused_interface.cpp:9-13`) [V].
- Mono decode (open PRs, 2.2): decode 188 tok/s (TPOT 5.31 ms) at 1 request MTP off, **249 tok/s (TPOT 4.01 ms)
  with MTP 2 tokens, acceptance 2.49**, on 2x B70 TP2, Qwen3.6-35B-A3B fp8, 1K prompt [V] as written. That is the
  fastest single-stream MoE decode on B70 hardware I found; it needs two cards.

### 5.4 OpenVINO GPU plugin

- MoE decode kernels in `src/plugins/intel_gpu/src/graph/impls/ocl_v2/`: `moe_3gemm_swiglu_opt`, a new
  `moe_3gemm_int3.cl` (batched GEMV for u3 weights on f16 activations, one work-group per (token, top-k slot), SIMD16,
  expert ids read on the GPU, no gather/scatter) [V]. PR 39366 (merged 2026-11-05): Qwen3.6-35B-A3B int4/int3 on
  "BMG" (card not named) decode 16.4 to 16.5 ms/token (about 61 tok/s), int4 16.5 to 16.6 [V] as written. Bytes per
  token for int4 (2.43 B non-lm_head active parameters x 4.5 bits = 1.37 GB, plus lm_head 0.5 to 1.0 GB) -> 115 to
  145 GB/s, 25 to 32 % of a B580's 456 GB/s or 19 to 24 % of a B70 [D, rough, quant overhead and lm_head type assumed].

### 5.5 MoE decode numbers and their share of bandwidth

Active bytes per token for Qwen3.6-35B-A3B [D from the config in 2.2]: 2.43 B active parameters outside the lm_head
(30 x 62.4 M + 10 x 56.1 M) plus a 508.6 M lm_head. Q4_K_M at about 4.8 bits plus a Q6_K lm_head: about 1.9 GB per
token; Q2_K about 1.3 GB; int4 (4.25 bits) with a bf16 lm_head about 2.3 GB [D, rough, quant mixes assumed].

| Engine, card | Model, quant | Decode | Bytes/token | Share | Source | Mark |
|---|---|---:|---:|---:|---|---|
| llama.cpp SYCL, 1 B70 (2026-04-21 build) | Qwen3.6-35B-A3B UD-Q4_K_M | 54.7 t/s, 114 W | 1.9 GB | 103 GB/s, 17 % | PMZFX `llm-benchmarks.md` | [V2], [D] |
| llama.cpp SYCL, 1 B60 (2026-11-05) | same model Q4_K_M | 83.4 t/s | 1.9 GB | 157 GB/s, 34 % of 456 | discussion 23313 | earlier [V], [D] |
| llama.cpp SYCL, 1 B70 (b10235, 2026-08) | same model Q2_K, depth 0 / 32K / 65K / 127K | 81.5 / 52.9 / 38.5 / 22.8 t/s | 1.3 GB | 108 GB/s, 18 % at depth 0 | issue 26581 | [V2], [D] |
| vLLM XPU, 1 B70, 275 W cap | same model GPTQ-Int4 | 76.2 t/s; MXFP4, GPTQ and AutoRound all 71 to 89 "despite an 85% spread in bytes read"; a fixed 5 to 6.6 ms per step on MoE | 2.3 GB | 176 GB/s, 29 % | vllm-xpu-kernels issue 509 (2026-08) | [V2], [D] |
| OpenVINO 2026.3, 1 B70 | same model | 87 t/s (vs 74 for vLLM in that comparison) | about 2.3 GB | about 200 GB/s, 33 % | same issue | [V2], [D] |
| OpenVINO GPU, "BMG" | same model int4 / int3 | 16.5 ms/token = 61 t/s | 1.9 to 2.4 GB | 19 to 32 % | OpenVINO PR 39366 | [V], [D] |
| llm-scaler vLLM, 1 B70 | same model sym_int4, MTP with a reduced draft vocabulary | 147 -> 190 t/s (speculative) | n/a | n/a | intel/llm-scaler issue 728 (2026-09-24) | [V2] |
| vLLM `--monokernel`, 2 B70 TP2 | same model fp8 | 188 t/s, 249 with MTP 2 | 1.83 GB per card | 345 GB/s per card, 57 % | 2.2 | [V], [D] |
| one lab's own runtime, 1 B70 | Qwen3.8-Flash-Next UD-IQ3_XXS (IQ3/IQ2_S experts) | 37.6 t/s (llama.cpp 15.9 on one card) | 4.951 GB (their figure) | 186 GB/s, 31 % | lab `SCOREBOARD.md` | [V2], [D] |
| one lab's patch to llama.cpp, 1 B70 | LFM2.5-24B-A2B Q8_0 | 41.7 -> 137.6 t/s, token-identical on 12 prompts | about 2.1 GB (assumed) | about 290 GB/s, 48 % | lab `repro/lfm25-24b-a2b-q8-b70` | [V2], [D] |

Reading: **single-card MoE decode on Intel engines sits at 17 to 35 % of bandwidth**, while dense whole-model
decode on the same card reaches 80 to 86 % (2.1). The gap is per-step structure (many small launches, host
synchronisation for expert ids, per-expert loops for more than one token; vLLM issue 509 measured a fixed 5 to 6.6
ms per step that the bytes do not explain) and IQ decode cost, not the memory system. The one persistent design
(vLLM mono kernel) reaches 57 % end to end on the same model. This is the gap Tona's one-launch pass targets [D].

Other llama.cpp numbers on B-series (helper agent, [V2], before -> after as stated by each author): fused MoE MMVQ
(PR 21920, B70) Qwen3.5-35B-A3B Q4_K_M at 16K depth 33.3 -> 46.0 t/s; MXFP4 MoE (PR 29809, B70) gpt-oss-20b 55.8 ->
106.6; Q8_0 reorder (PR 21527) 27B 4.9 -> 15.2; IQ3 reorder (PR 29107) B580 Qwen3.5-9B UD-IQ3_XXS 22.6 -> 52.2 and
B70 27B IQ3_S 13.7 -> 23.4, with draft MTP 14.4 -> 32.8; open issue 31195 "Improve IQ quantization performance on
Xe2/Battlemage" (no numbers). No IQ1 or IQ2 timing on Battlemage was found anywhere.

### 5.6 IPEX-LLM

Archived read-only 2026-01-28 (earlier note, [S]); no FlashMoE result on a B-series card found [V2 negative].

## 6. Power and clock under sustained memory-bound load (open point 5)

All [V2] (helper agent) unless marked; none is a controlled memory-bound measurement.

| Item | Value | Source |
|---|---|---|
| Board power | 230 W, configurable 160 to 290 W; graphics clock 2,280 MHz, maximum dynamic 2,800 MHz | StorageReview B70 review |
| Compute load (Igor's Lab) | average 236.2 W in compute (gaming 225.5 to 228.9 W); clock in compute "around 2,100 to 2,150 MHz" against about 2,500 MHz in games; attributed to the power/voltage point, not temperature; hotspot 75 to 76 C, VRAM 78 to 80 C; PEG slot at 5.5 A (its limit); peaks 296 W (10 to 20 ms) to 389 W (< 1 ms) | Igor's Lab B70 review, pages 7 and 8 |
| Idle | 47.8 W (FHD 60 Hz) and 51.4 W (UHD 120 Hz) with a display attached; another report: under 15 W and C6 within about 2 s without `-cb`, about 50 W with the clock pinned at 2,800 MHz with `-cb` (disputed by maintainers) | Igor's Lab; llama.cpp issue 24946 |
| LLM decode power | MoE Qwen3.6-35B-A3B Q4_K_M 114 W, Gemma 4 26B-A4B 102 W, dense 27B Q4_K_M 178 W (xe hwmon energy counters, 500 ms sampling) | PMZFX |
| Prefill, dense 27B Q8_0 | series peak 183 W stock, 223 W patched; fore/aft drift -0.01 % and +0.25 %, i.e. no throttling seen | llama.cpp PR 30701 |
| Readouts on Linux | `/sys/class/drm/cardN/device/tile0/gt0/freq0/act_freq`, `.../gtidle/idle_status`, `hwmon/*/energy*_input`, `sensors xe-pci-*` (pkg, vram, fan) | issue 24946, PMZFX |
| Memory clock | 19 Gbps GDDR6 on B70 [S]; no controlled memory-overclock bandwidth data on any Battlemage card | |

Reading [D]: decode on a B70 runs far below the board limit (100 to 180 W of 230 W), so a bandwidth-bound pass
should not hit power throttling; the compute clock of 2.1 GHz under heavy load matters for prefill (XMX peak at
2.1 GHz is 275 TOPS int8 / 138 TFLOPS fp16 rather than 367 / 183) and for IQ decode ALU budgets (8.6 instead of 11.5
T lane-ops/s), not for the memory system. Unmeasured: clock and memory clock during a long pure-bandwidth loop.

## 7. Decode estimate for Tona's large model on one B70 (estimate, not a measurement)

Inputs: bytes per pass 4.11 GB dense + 0.98 GB experts = 5.1 GB, 2.5 tokens per pass with MTP, all experts
resident (`docs/hardware/README.md`, research/06: experts gate/up IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS, down IQ4_NL/Q2_0,
dense part includes 1.28 GB BF16 gated residual) [V]. Method of `docs/hardware/README.md`: GPU part = bytes /
effective bandwidth; upper reading = 0.65 x (2.5 / GPU part), lower reading = 2.5 / (GPU part + 10 ms).

The bytes split into two classes with different evidence: 4.11 GB dense (81 %: BF16 gated residual, DeltaNet,
QSA, lm_head, router) and 0.98 GB IQ/Q2_0 experts (19 %). Dense-type streams on a B70 reach 80 to 86 % today
(PR 29864 at width 1, 73 to 79 % for Q8_0/Q6_K at width 8 with XMX); IQ kernels reach 47 % (IQ3_S) to 77 %
(IQ4_NL, open PR), and naive ones 8 to 11 %. Pass time = 4.11 / (608 x dense share) + 0.98 / (608 x IQ share) [D]:

| Case | Dense share | IQ share | GPU part | Effective | Memory bound | Upper (x 0.65) | Lower (+10 ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| pessimistic | 70 % (426 GB/s) | 40 % (243) | 13.7 ms | 372 GB/s | 183 tok/s | 119 | 106 |
| central low | 80 % (486) | 55 % (334) | 11.4 ms | 447 GB/s | 220 | 143 | 117 |
| central | 85 % (517) | 60 % (365) | 10.6 ms | 478 GB/s | 235 | 153 | 121 |
| central high | 88 % (535) | 75 % (456) | 9.8 ms | 518 GB/s | 254 | 165 | 126 |
| optimistic | 90 % (547) | 80 % (486) | 9.5 ms | 534 GB/s | 262 | 171 | 128 |

[D] arithmetic, central row: 4.11 / 517 + 0.98 / 365 = 7.95 + 2.68 = 10.6 ms; 2.5 / 10.6 ms = 235; x 0.65 = 153;
2.5 / 20.6 ms = 121.

Reading: **central about 120 to 165 tok/s, range 105 to 170**, against 89 measured on the reference. The 10 ms
"fixed" term of the lower reading is a fit on the reference; the B70 data point that speaks to it is vLLM's mono
decode: 40 layer kernels take about 3.5 ms of a 5.31 ms step, the lm_head about 1.3 ms at 400 GB/s, leaving about
0.5 ms of other fixed cost per step [D], so on a persistent design the upper reading is the more likely one.
Cross-check: at that engine's 345 GB/s end-to-end rate Tona's 5.1 GB pass gives 2.5 x 345 / 5.1 = 169 tok/s
before any factor [D].

What would move it: (1) dense streams, 81 % of the bytes, set the result; one point of dense share is worth about
1 to 1.5 tok/s; (2) the IQ expert kernels: dropping from 60 % to the 47 % of llama.cpp's IQ3_S costs about 0.75 ms per pass
(about 7 %) [D]; (3) MTP width 8 passes that touch more experts than the 0.98 GB average; (4) ECC on: the model
does not fit at all (3.0).

## 8. What remains unknown and how phase 0 settles it

- First: ECC state (`xpu-smi config`) and the Level Zero `totalSize` / `physicalSize` on the actual card and driver;
  ECC on means the model does not fit.
- A read/copy bandwidth number on our B70 from `ze_peak global_bw` (Intel's tool, 2 GiB buffer, 1,024-item groups)
  and from one kernel with Tona's access pattern (32 x 512 items, 256 GRF, 16 B per lane loads). Expectation from
  the lab receipt: read 590 to 606 GB/s, copy 528 to 546 GB/s read + write; below 560 GB/s read is a finding
  (ECC, clock, BAR or PCIe setup) [D].
- Kernel bandwidth for each of Tona's types (BF16, the dense quant types, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL,
  Q2_0) at widths 1, 2, 4, 8 against that ceiling; this replaces the dense and IQ shares in section 7 with
  numbers. No IQ2_S, IQ2_XXS or Q2_0 timing on Battlemage has been published.
- Whether 256 GRF + 32 resident groups also holds with our register use (compile report of the kernel).
- The cooperative group count `zeKernelSuggestMaxCooperativeGroupCount` returns for that kernel (code says 32 at
  512 items and 256 GRF) [V code, untested].
- Clock and power during a 10-minute decode loop (section 6).

## Sources

Code [V] (paths relative to the clone or file named above): compute-runtime `shared/source/device/device_caps.cpp`,
`shared/source/device/device.cpp`, `shared/source/helpers/constants.h`, `shared/source/helpers/compiler_product_helper_xe_hpc_and_later.inl`,
`shared/source/helpers/compiler_product_helper_before_xe_hpc.inl`, `shared/source/xe2_hpg_core/enable_compiler_product_helper_bmg.cpp`,
`shared/source/xe2_hpg_core/hw_info_bmg.cpp`, `shared/source/xe2_hpg_core/hw_cmds_base.h`,
`shared/source/release_helpers/release_helper/{release_helper_base.inl, release_helper_bmg_g31.cpp, release_helper_lnl.cpp, release_helper_preferred_slm_xe2_hpg_160k.inl}`,
`shared/source/helpers/{gfx_core_helper_base.inl, kernel_helpers.cpp, compiler_product_helper.cpp}`,
`shared/source/os_interface/linux/{drm_memory_manager.cpp, drm_neo.cpp, memory_info.cpp}`,
`shared/source/os_interface/linux/xe/{ioctl_helper_xe.h, ioctl_helper_xe.cpp}`, `shared/source/debug_settings/debug_variables_base.inl`,
`level_zero/core/source/{context/context.cpp, device/device.cpp, module/module_imp.cpp, kernel/kernel_imp.cpp, helpers/api_specific_config_l0.cpp}`,
`shared/offline_compiler/source/offline_compiler.cpp`; IGC `IGC/Compiler/CISACodeGen/{Platform.hpp, CISABuilder.cpp}`,
`IGC/common/{Types.hpp, SystemThread.cpp}`; Linux `drivers/gpu/drm/xe/{xe_vram.c, xe_query.c, xe_pci_rebar.c, xe_module.c}`;
intel/llvm `sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc`,
`unified-runtime/source/adapters/level_zero/{common/device.cpp, usm.cpp}`, `sycl/test-e2e/Adapters/max_malloc.cpp`;
level-zero-tests `perf_tests/ze_peak/{src/global_bw.cpp, src/ze_peak.cpp, include/ze_peak.h, kernels/ze_global_bw.cl}`;
llama.cpp `ggml/src/ggml-sycl/{mmvq.cpp, ggml-sycl.cpp, fused-gemm.cpp, fused-gemm.hpp, common.cpp, common.hpp, CMakeLists.txt, topk-radix.cpp}`,
`docs/backend/SYCL.md`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`; vllm-xpu-kernels `csrc/xpu/grouped_gemm/{moe_shared_fused_interface.cpp, xe_2/gemm_xe2.hpp}`;
OpenVINO `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_int3.cl`.

Pages and threads [V]:
- compute-runtime issue 1047 (open): https://github.com/intel/compute-runtime/issues/1047
- compute-runtime commit a9797c6e3c and compare with 26.40.40196.8 (GitHub API)
- vllm-xpu-kernels PR 685 (open): https://github.com/vllm-project/vllm-xpu-kernels/pull/685
- vLLM PR 65566 (open): https://github.com/vllm-project/vllm/pull/65566
- OpenVINO PR 39366 (merged 2026-11-05): https://github.com/openvinotoolkit/openvino/pull/39366
- Qwen3.6-35B-A3B config: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json
- Intel oneAPI GPU Optimization Guide 2025.2, Intel Xe GPU Architecture: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html
- Chips and Cheese, Intel's Battlemage Architecture (2025-02-11): https://chipsandcheese.com/p/intels-battlemage-architecture
- PCGH, Arc Pro B70 & B65 launch (2026-03-25, with L2 update): https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/Specials/Arc-Pro-B70-B65-Preis-Gaming-Specs-Release-1523369/
- Intel support article 000102907 (ECC, B-series VRAM, reviewed 2026-06-08): https://www.intel.com/content/www/us/en/support/articles/000102907/graphics.html
- steveseguin/b70-optimization-lab (Unlicense, pushed 2026-11-17): receipt `experiments/own-xpu-runtime/stage2/packet7-kopt2/receipts-card3-20261011/ceiling/native.json`,
  `packet7-kopt2/{kernels.cpp, reference.hpp, tests/ceiling.cpp, README.md}`, `docs/b70-ecc-and-vram.md`: https://github.com/steveseguin/b70-optimization-lab
- llama.cpp PR 29864 (merged 2026-10-02, B70 XMX multi-column table): https://github.com/ggml-org/llama.cpp/pull/29864

Read by the helper agent only [V2] (URLs as it reported them): OpenBenchmarking clpeak B580
https://mail.openbenchmarking.org/result/2412128-PTS-GPUCOMPU05 and .../2501131-PTS-GPUCOMPU20; OpenCL-Benchmark
https://mail.openbenchmarking.org/result/2412136-PTS-INTELARC59 and github.com/ProjectPhysX/OpenCL-Benchmark;
intel/torch-xpu-ops issue 3784; llama.cpp PRs 21527, 21700, 21845, 21920, 22066, 23142, 24452, 27062, 29107,
29375, 29809, 30156, 30374, 30701, 30775, 31528, 31993, 31998, issues 22413, 24946, 26581, 27373, 31195,
discussion 10879; vllm-xpu-kernels issue 509; intel/llm-scaler issue 728; https://vllm.ai/blog/2025-11-11-intel-arc-pro-b;
StorageReview and Igor's Lab B70 reviews; PMZFX/intel-arc-pro-b70-benchmarks `llm-benchmarks.md`; the lab's
`SCOREBOARD.md` and `repro/lfm25-24b-a2b-q8-b70/README.md`; IDFS https://idfs.ai/blog/six-days-with-the-intel-arc-pro-b70.

Search snippets only [S]: Intel B60 datasheet (20 Xe2 cores, 456 GB/s): https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b60-gpu.pdf
