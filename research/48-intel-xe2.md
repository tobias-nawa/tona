# Intel Xe2 (Arc Pro B70): what a backend would take and what it could reach

Research of 2026-11-14 (agent run, web only; no code, no build, no GPU job, nothing run on Intel hardware).
Question from the maintainer: the Arc Pro B70 (32 GB GDDR6, 256 bit, 608 GB/s, 32 Xe2-HPG cores, 256 XMX engines,
367 INT8 TOPS dense, PCIe 5.0 x16, 160 to 290 W, about 950 USD at launch) is interesting because Tona does not cover
Intel at all. The platform page that follows from this note is [docs/backends/intel-xe2.md](../docs/backends/intel-xe2.md).

**Update 2026-11-17 (four follow-up research runs, web and source code only, nothing run on Intel hardware):**
the three make-or-break questions of section 1 are answered on paper; one test on a B70 would confirm them.

- CPU-to-kernel flag: possible. Proven route in Intel's own production code (Intel SHMEM, oneCCL): the flag lives in
  device memory, the CPU writes it through the BAR mapping, the kernel polls with an L1-bypassing load; answers go to
  host memory with a system-release fence. A flag in host memory (our AMD route) is likely but untested; a plain or
  volatile load in a spin loop sees a stale L1 line. [host-flag](48-intel-xe2-host-flag.md)
- Grid barrier and request-long kernel: possible with cooperative launch and one work-group per Xe core (the
  occupancy query over-reports on BMG with SLM); no job timeout for long-running compute VMs; mid-thread preemption
  likely, card must be headless. The dispatch fallback costs about 0.5 to 3 us per boundary, so only 5 to 15 % below
  the megakernel, not 25 to 50 %. [barrier](48-intel-xe2-barrier.md)
- Memory and compute: a plain read kernel reaches about 600 of 608 GB/s; ECC is on by default and must be off for the
  model to fit (31.25 GiB reported by newer drivers, 28.7 GiB needed, no desktop on the card); no 4 GiB single
  allocation limit on Battlemage; ReBAR required; 256 KB L1/SLM per Xe core, 128 KB SLM per work-group, 24 MiB L2;
  XMX int8, fp16, bf16, tf32, no fp8. Estimated decode for Tona about 120 to 165 tok/s.
  [memory-compute](48-intel-xe2-memory-compute.md)
- Re-check of this note claim by claim: [recheck](48-intel-xe2-recheck.md)

Marks: [V] opened and read in this run (page, source file, API answer); [S] search result or snippet only, page not
opened; [K] from memory, unchecked; [D] derived by us from marked inputs (arithmetic shown); [L] local file of this
repository. A figure from a forum or blog is [V] only as "that person wrote it"; it is not our measurement.
Nothing in this note is [M] (measured by us).

## 1. Answer in short

- **Route:** native SYCL (DPC++) kernels on Level Zero, with the host side written against Level Zero where SYCL gets
  in the way, as one more backend with its own pass. Vulkan stays the reach route of
  [vulkan.md](../docs/backends/vulkan.md); on Intel it is about 1.5 to 2 times slower for decode in the one engine
  that has both (section 6), and it cannot express a persistent kernel.
- **Make-or-break question:** "can a kernel spin on host memory that the CPU writes, and can a persistent kernel run
  for the length of a request without the driver killing it?" Answer, split:
  1. **Long-running kernel:** on the `xe` driver the answer is yes by construction. The compute runtime creates every
     VM in long-running mode and the GuC submission code gives such queues an unlimited job timeout ([V], kernel and
     runtime source, section 3). What remains is the preemption timeout (640 ms default, then an engine reset).
  2. **CPU flag in host memory:** **not documented, and Intel's guide says the opposite for atomics.** The oneAPI GPU
     guide (2025.2) states for discrete GPUs: atomics on `malloc_host` memory are not supported and concurrent
     host/device access to `malloc_shared` is not supported ([V]). Plain loads of a host-pinned flag may well work;
     nobody has published a test. It has to be measured in phase 0, with cache-bypassing loads.
  3. **It matters less than on the reference:** on a 32 GB card the whole model is resident and the CPU has no
     expert to deliver. The mailbox exists for misses. What the megakernel then still needs is a **grid barrier**
     (about 1,100 per pass). SYCL has an experimental `root_group` barrier whose co-residency semantics the spec
     does not state ([V]); a hand-built barrier on device-USM atomics is the fallback. If both fail, the barrier becomes a
     dispatch boundary, and the estimate below drops by a quarter to a half.
- **Estimated decode with MTP at 2.5 tokens per pass, all experts resident, one B70:** about **110 to 190 tok/s**
  with in-kernel barriers, about **80 to 120 tok/s** if every barrier is a dispatch of 5 to 10 us ([D], section 8).
  Compare 89 tok/s measured on the reference. Without MTP divide by about 2.5.
- **Biggest risks:** (1) the memory system reaches only 55 to 67 % of 608 GB/s in the best public engine and nobody
  has run a raw bandwidth test on the B70; (2) host-flag and barrier semantics are undocumented; (3) 30.8 GB of model
  barely fits 29.2 GiB of usable VRAM, so the card must be headless and the context short or a few experts off-card;
  (4) the SYCL toolchain moves weekly and has Intel-specific bugs (compiler-version-dependent matrix code, recurrent
  state leaks, device-lost errors); (5) no maintainer owns the hardware (tier 3, named owner needed).

## 2. Linux compute stack for Battlemage, 2026-11

| Topic | Finding | Mark |
|---|---|---|
| Kernel driver | `xe` is the driver for Battlemage (needs kernel 6.12 or later); `i915` is old and not for BMG. A llama.cpp SYCL user found different results with both loaded and was told to keep one, `xe` preferred ([discussion 23313](https://github.com/ggml-org/llama.cpp/discussions/23313)). B70 benchmark rigs run `xe` on kernel 7.0 ([PMZFX hardware.md](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks), 2026-04) | [V] |
| Fedora | `intel-compute-runtime` 26.35.39758.21 in Fedora 44 and 45, 26.18.38308.4 in Fedora 43 ([Fedora packages](https://packages.fedoraproject.org/pkgs/intel-compute-runtime/intel-compute-runtime), read 2026-11-14); upstream's latest release is 26.40.40196.8 of 2026-11-03 (GitHub API). The reference PC runs Fedora 44 with kernel 7.2 [L]. A Fedora guide for the B60 says the AI setup was blocked by a kernel-level bug reported to Intel (no details found) | [V] versions, [S] guide |
| Level Zero, OpenCL | from `intel-compute-runtime` (Level Zero loader separate); IGC 2.30 and compute runtime 26.09 on the PMZFX rig | [V] |
| SYCL (DPC++) | oneAPI DPC++ 2025.3.3 on the PMZFX rig. Whether Fedora packages the toolkit, or one installs Intel's repository, is **unchecked** | [V] version, unchecked packaging |
| Vulkan | Mesa ANV, 25.2.8 on a B70 rig; `VK_KHR_cooperative_matrix` for Xe2 enabled in llama.cpp since 2025-06-05 ([PR 14001](https://github.com/ggml-org/llama.cpp/pull/14001), merged, Xe2 only because Alchemist regressed); the log of that PR shows subgroup size 32 and `int dot` 1 on Linux | [V] |
| IPEX-LLM | archived read-only 2026-01-28 ([Phoronix](https://phoronix.com/news/intel-ending-bigdl), search result); not a base for new work | [S] |
| vLLM | Intel's `llm-scaler-vllm` container runs FP8 and continuous batching on B60 and B70; mainline vLLM on XPU could not run Qwen3.5 (gated delta net) in April 2026 ([engine-comparison.md](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks)) and a custom build leaked recurrent state ([bentech](https://bentech.substack.com/p/running-local-llms-on-intel-arc-pro)) | [V] |
| OpenVINO | OpenVINO backend of llama.cpp crashed on Qwen3-Coder-Next in the SYCL discussion; OpenVINO GenAI not researched | [V] one line, rest unchecked |
| GuC firmware | one user needed GuC 70.60.0 (Ubuntu shipped 70.44.1), installed by hand to `/lib/firmware/xe/bmg_guc_70.bin`; "without it, performance is degraded" | [V] (his statement) |
| Resizable BAR | the large BAR is visible on a working rig (32 GB BAR per card). Intel treats ReBAR as required for Battlemage [K]; a Proxmox passthrough report shows a probe timeout and half the aperture with ReBAR on ([Intel community](https://community.intel.com/t5/Graphics/ARC-PRO-B70-Passthrough-Hell-on-PVE-Resizable-BAR-Problems/td-p/1748878)); a buyer reports about 20 tok/s turning into over 80 after fixing the setup including ReBAR (aiweekly summary) | [V] BAR, [S] rest |
| Idle power | 38 W (one llama.cpp run), 45 W (two forum posts), 48 to 51 W (Igor's Lab text as summarized), 12 to 15 W at the wall (one review) , i.e. the sources disagree; the NVIDIA cards in the same posts idle at 10 to 15 W. Mechanism (ASPM, memory clocks) not found | [S] |
| Hangs | compute-runtime 25.44.36015.8: "hotfix ... for Battlemage Pro customers only, addressing potential hang issues ... in very long (multi-day) running workloads" ([release](https://newreleases.io/project/github/intel/compute-runtime/release/25.44.36015.8)); B50 pair: "driver ends compute due to timeout" during inference, text of the log not visible ([thread](https://community.intel.com/t5/Intel-Arc-Discrete-Graphics/Driver-ends-compute-due-to-timeout-on-dual-Arc-Pro-B50/m-p/1743709)); a vLLM user hit `DEVICE_LOST` (error 20) when oversizing the KV pool, "a real XPU driver bug" (Level1Techs, 2026-10-24); SYCL exits with `UR_RESULT_ERROR_INVALID_MEM_OBJECT` at teardown after correct work | [V] each, as reports |

Maturity in one line: usable and improving weekly, the compute runtime releases about every two weeks, but the
documentation of the points that matter to a megakernel is thin.

## 3. The megakernel pattern on Intel (the make-or-break point)

### 3.1 How long may a kernel run?

- Default per-engine-class job timeout is 5000 ms, hard maximum 10000 ms, preemption timeout 640000 us
  ([V] `drivers/gpu/drm/xe/Kconfig.profile`, kernel master, read 2026-11-14). The preemption timeout is "how long the
  driver waits for the current context to reach an arbitration point and yield the GPU voluntarily when a
  higher-priority context becomes runnable. If the context does not yield before the timer expires, the HW is reset".
- **Compute queues are exempt from the job timeout:** `xe_guc_submit.c` (kernel master) sets the scheduler timeout to
  `MAX_SCHEDULE_TIMEOUT` when `xe_vm_in_lr_mode(q->vm)`; `IoctlHelperXe::getFlagsForVmCreate` in the compute runtime
  (master) starts from `DRM_XE_VM_CREATE_FLAG_LR_MODE` for every VM ([V], both read 2026-11-14). So a Level Zero or
  SYCL kernel is not killed by the 5 s job timeout. This is derived from source, not tested by us, and the B50 report
  above shows "timeout" messages exist in practice (cause unknown to us).
- **What can still reset:** (a) the preemption timeout when another context needs the engine (a compositor on the same
  GPU); whether a spinning kernel reaches an arbitration point (mid-thread preemption on Xe2) is [K] and must be tested;
  (b) firmware and runtime bugs (the multi-day hang hotfix). On the maintainer's machine the cleanest answer is that the
  B70 drives no display (section 8 of the platform page).
- Windows: TDR as everywhere ([V] Microsoft Learn, as in [cuda.md](../docs/backends/cuda.md)); not relevant for Linux
  first.
- Old i915 behavior (hang detection resets an engine after a few seconds, Codeplay advice "split into smaller
  launches") is documented for the previous driver ([S], Intel community threads); do not transfer it to `xe`.

### 3.2 A kernel spinning on host memory written by the CPU

- **Documented:** SYCL says system-scope `atomic_ref` on USM is atomic against host code *if the device has the aspect*
  `usm_atomic_host_allocations` ([S], SYCL reference). Intel's GPU guide (2025.2, page id 771772, [V]) says, under
  "on discrete GPU": "Atomic operations on host allocated USM (sycl::malloc_host) are not supported", "Concurrent
  accesses from host and device to shared USM location (sycl::malloc_shared) are not supported", and recommends
  `malloc_device` for atomics. An integrated-GPU forum thread says Iris Xe does not advertise either aspect ([S]).
- **Not documented:** whether a plain or volatile load of a `malloc_host` flag from a running kernel sees a CPU
  store after the store becomes visible to the PCIe root complex, what the cache path is (GPU L1/L2 may hold the line;
  host memory may be mapped write-back coherent or uncached depending on the kernel's PAT index), and the latency.
  No public benchmark exists that I could find (searches for spin-wait on a host flag, megakernel on Arc or XPU:
  nothing). ESIMD load instructions carry cache-control hints (uncached, streaming) [K]; that is the tool to try.
- **Alternative without in-kernel polling:** `zeCommandListAppendWaitOnEvents` on an event that the host signals with
  `zeEventHostSignal`; the GPU command streamer waits, no kernel spins. Counter-based events cannot be signaled by the
  host ([V] spec text in a search result). Latency unpublished; the Mac analog measured 55 us ([L] docs/backends/metal.md).
  Per-layer cost for 40 to 60 layers would be 2 to 4 ms if it were 55 us; at the reference's 0.73 us for a host flag it
  would be nothing. Measure.
- **Consequence for the design:** with all experts resident (32 GB card) the CPU never has to answer within a pass.
  Routing can run on the GPU (a variant of the reference's pass that does not wait), and no host polling is needed
  during decode at all. The host flag is then only a "start" doorbell per pass (one launch). The reference's wait
  pattern stays relevant only for the 16 and 24 GB cards of the Intel line (B580 12 GB, B60 24 GB), which are
  not this question.

### 3.3 Grid-wide barriers and co-residency

- SYCL: `sycl_ext_oneapi_root_group` (experimental, DPC++): kernel launched with `use_root_sync`, group count limited by
  `max_num_work_groups_sync`; the spec says the semantics "are not identical" across backends and never states
  forward progress outright; implementation notes propose `zeCommandListAppendLaunchCooperativeKernel` on Level
  Zero ([V], extension document in intel/llvm). A Codeplay thread reports a runtime exception on root-group barriers
  tied to the oneAPI install ([S]). The DPC++ compatibility tool says SYCL has no cross-group synchronization and
  suggests the root-group extension ([S]).
- Level Zero: `zeKernelSuggestMaxCooperativeGroupCount` and the cooperative launch exist and are used by LLVM's
  offload plugin ([S]); which Intel GPUs report support is **unchecked**. Unified Runtime says any device can claim
  support by limiting the count to 1 ([S]).
- Hand-built barrier: device-USM atomics (the recommended kind) plus a launch of as many work-groups as are resident.
  Works if the hardware scheduler gives every started work-group forward progress (occupancy-bound model). The 2021
  study tested one Intel GPU (HD 620, Mesa Vulkan) among eight GPUs and found the non-conforming models only on
  Apple and ARM ([V] paper table and abstract, text extracted from the PDF; Xe2 not tested by anyone I found).
  Intel's own earlier research prototype of "cooperative kernels" ran on Intel ([L] research/23-waiting-in-kernel.md).
- Atomic throughput: the Chips and Cheese Battlemage article derives about 884 GB/s from INT32 atomics on the B580's
  L2, "massively" above the A770 ([V]); a barrier with 32 to 64 arrivals is not an atomics-bandwidth problem.
- Resident grid: 32 Xe cores x 8 vector engines x 8 threads = 2,048 hardware threads on the B70 ([D] from the Intel
  guide table [V]); a work-group of up to 1024 items; one resident work-group per Xe core mirrors the reference's
  one block per CU (32 on the 9060 XT, [L]).

### 3.4 Compute and copy engines

Battlemage exposes a compute engine and a blitter (copy) engine to Level Zero; copies on the copy engine are meant to
overlap compute [K]. The SYCL runtime has switches to use the copy engine [K]. No published measurement of
overlap with a heavy compute kernel, none of PCIe 5.0 x16 host-to-device bandwidth on the B70 (the B580 runs x8 and
the Chips and Cheese OpenCL transfer tests fell "well short" of theoretical, no number, [V]). Phase 0 item.

## 4. Throughput facts

| Item | Value | Mark |
|---|---|---|
| Memory | 256 bit GDDR6, 19 Gbps, 608 GB/s, 32 GB; about 29.2 GiB usable on a B70 in one report | [V] spec (StorageReview, Igor's Lab), [V] usable (bentech) |
| **Measured bandwidth** | **no raw bandwidth test of a B70 or B580 found** (Chips and Cheese gives none; reviews quote only the spec). Indirect: llama.cpp SYCL on llama-2-7b Q4_0 gives tg128 about 106 to 111 tok/s on four B70 rigs (discussion 23313, [V]); 3.56 GiB model [K] gives about 405 GB/s = 67 % of 608 [D]. Qwen3.5-27B on a B70: Q4_0 57 %, Q4_K_S 56 %, Q4_K_M 53 %, Q6_K 48 %, Q8_0 66 % after a kernel fix, IQ4_NL 14 % (issue 21517, [V]) | [D] |
| Reference GPU for comparison | 315 GB/s measured of 320 spec = 98 % on the 9060 XT [L] | [L] |
| XMX | 8 per Xe core, 256 on the B70 ([V] Igor's Lab: 256 engines); 367 INT8 TOPS dense ([V] StorageReview, Igor's Lab). Lunar Lake: 4,096 int8 and 2,048 fp16 ops per clock per Xe core ([S] Jon Peddie): 32 cores x 4,096 x 2.8 GHz = 367 TOPS, so the B70 clock is about 2.8 GHz and fp16 peak about 183 TFLOPS dense | [D] |
| Reference compute (measured) | f16 WMMA 93, int8 183 TOPS, fp8 186, int4 341 [L]. On paper the B70 has about twice the matrix peak of the 9060 XT in f16 and int8, no fp8 or fp4 path on Xe2 [K] | |
| What llama.cpp SYCL reaches in prefill | llama-2-7b Q4_0 pp512 about 2,760 to 3,010 tok/s on B70 (discussion 23313) = about 38 TFLOPS [D] (2 x 6.7 G parameters x 2,830), about 21 % of the fp16 peak | [D] |
| XMX from SYCL | `joint_matrix` (shapes per Intel guide: int8 K=32, fp16 K=16 on DG2/PVC, Xe2 table not found); ESIMD DPAS used by llama.cpp's multi-column kernels ([PR 29864](https://github.com/ggml-org/llama.cpp/pull/29864), merged 2026-10-22: 2 to 80 columns, Q4_K..Q8_0; Q4_K_S pp16 50 to 308 tok/s on a B70). A third-party write-up says an int8 `joint_matrix` tile (8x16x32, sub-group 16) failed for some nodes depending on the IGC version ([S] yiliu30) | [V] PR, [S] rest |
| Sub-group sizes | 16 and 32 (Xe2 dropped 8) | [V] Intel guide, Chips and Cheese |
| Registers | 512-bit registers; 128 per thread (8 threads per vector engine) or 256 in large-register mode (4 threads) = 8 KB or 16 KB per thread; 64 KB register file per vector engine ([D] 128 x 64 B x 8) | [V] Intel guide, Chips and Cheese |
| SLM, L1 | 128 KB SLM per Xe core, also the maximum per work-group; 256 KB L1/SLM block per core, "a bit under 256 B/cycle" measured L1 bandwidth per core; max work-group 1024 | [V] |
| L2 | 18 MB on the B580; the B70's (BMG-G31) L2 size and the achieved L2 bandwidth are **unchecked** | [V] B580 only |
| VRAM latency | just over 300 ns (Vulkan, B580) | [V] |

## 5. Real LLM numbers (hardware and engine named, all third party)

All llama.cpp, 1 B70 unless stated; "tg" = decode, "pp" = prefill, both tokens per second.

| Setup | Model, quant | pp | tg | Source |
|---|---|---:|---:|---|
| SYCL, Ubuntu, kernel 7.0, 2x B70 rig, build 2026-04-21 | Qwen3.6-35B-A3B UD-Q4_K_M (MoE, 20.6 GiB) | 615 (pp512) | 54.7 | PMZFX README [V], 114 W |
| same | Gemma 4 26B-A4B Q4_K_M | 1,129 | 52.6 | same [V] |
| same, 2 cards layer split | Qwen3-Coder-Next 80B-A3B Q4_K_M (45 GiB) | 305 | 43.4 | same [V] |
| same | Qwen3.5-27B Q4_K_M (dense) | 718 | 20.4 | same [V] |
| same | Llama 3.3-70B Q4_K_M, 2 cards | 338 | 11.5 | same [V]; about 446 GB/s of 1,216 (37 %) |
| SYCL vs Vulkan, Level1Techs, 2026-08-18, builds unnamed | Qwen3.6-35B MXFP4 MoE | 929 vs 1,185* | 52.3 vs 33.5 | [V]; *flagged by him as anomaly |
| same | gemma-26B MoE | 1,502 vs 910 | 59.6 vs 47.7 | same [V] |
| same | Qwen3.6-27B Q4_K_S dense | 751 vs 483 | 25.3 vs 18.9 | same [V] |
| SYCL vs Vulkan, PMZFX | Qwen2.5-1.5B Q4_K_M | | 229 vs 102 | [V] "Vulkan about 2x slower than SYCL on decode at every size tested" |
| Vulkan, bentech | Qwen3.5-27B Q5_K_XL, B70, 150K context | ~18 | ~12 | [V] |
| SYCL, B580 12 GB (Ryzen, Ubuntu 25.10) | llama-2-7b Q4_0 | 2,064 | 73.8 | discussion 23313 [V] |
| SYCL, B60 24 GB (Fedora 43) | llama-2-7b Q4_0 | 1,869 | 80.0 | same [V] |
| SYCL, Arc 140T iGPU (Ultra 255H); Xe3 iGPU Panther Lake | gemma4 26B; llama-2-7b | 224; 652 | 12.7; 19.2 | same [V] (Lunar Lake class hint; B580 numbers are the closer yardstick) |
| vLLM XPU (llm-scaler), 4x B70, batch 32 | GPT-OSS 120B / Qwen3 Coder 30B / Llama 3.1 8B | | 6,870 / 6,643 / ~12,000 aggregate | StorageReview [V]; throughput, not single stream |
| vLLM XPU TP=2, 2x B70 | Qwen2.5-14B FP16 | 7,104 | 36.1 | PMZFX [V]; llama.cpp one card Q4_K_M 523 pp, 36.8 tg |

Readings that matter for Tona:

- **MoE decode is far from the bandwidth bound.** Qwen3.6-35B-A3B Q4_K_M reads about 1.8 GB per token ([D], 3 B active
  parameters at about 4.9 bits plus shared parts): 54.7 tok/s is about 100 GB/s, **16 % of 608 GB/s** (reference
  engine: 98 % of its bus on its own kernels). The gap is the engine, not the card; a hand-tuned pass is where the
  room is, and also where the risk is.
- **Software moves fast:** Q8_0 was 4.9 tok/s in April and 15.3 after one pull request (21527, 21638, merged 2026-04-07
  and later); XMX multi-column kernels for speculative verification arrived 2026-10-22. In one engine a
  draft-model path (MTP) on a B70 gave no gain after a fix (issue 23533, [L] research/papers.md, 2026-05-22).
- **SYCL has correctness holes on hybrid models:** recurrent state carried over between prompts on Qwen3.5 (bentech,
  [V]); our DeltaNet layers hit the same class. The K greater than 1 trap of research/papers.md applies.
- Tona's engine decodes the large model at 89 tok/s on the reference; no published Intel number is for that model.

## 6. SYCL against Vulkan in one place

Decode favors SYCL by 1.25 to 2.2 times on every pair above; prefill favors SYCL on dense models (1.5 to 2.5 times) and
is mixed on MoE (one anomaly in each direction). Vulkan works out of the box with Mesa and needs no toolkit, runs
hybrid models where SYCL leaks state, and is the only route on a machine without oneAPI. For Tona the decisive
difference is not the number: Vulkan has no co-residency guarantee between work-groups ([V]
[vulkan.md](../docs/backends/vulkan.md), OOPSLA 2021 study), so there is no persistent kernel and no in-kernel barrier.

## 7. Multi-card

- Intel's Project Battlematrix stack: `llm-scaler-vllm` supports tensor, pipeline and data parallelism and "CCL (P2P or
  USM)"; reference design up to 8 Arc Pro B60 ([S], Intel developer articles, August 2025 update, search results).
  Whether the B70 does PCIe peer-to-peer, and at what bandwidth, is **unconfirmed**; Intel's repo has a platform-evaluation
  mode that measures P2P and bandwidth ([S]).
- B70 pairs on consumer boards run x8/x8 (the card is x16 capable, one rig shows `LnkSta 32GT/s x8`) and the author
  states GPU-to-GPU traffic then goes through the CPU ([V] PMZFX hardware.md).
- llama.cpp SYCL layer split does not speed up models that fit one card (sequential by design, [V]); row split segfaults
  (mixed B70 and B60, [V]); dual B580 was slower than one on the same model ([V] discussion 23313). vLLM TP=2 on 14B
  FP16 decodes at 36.1 against 36.8 for llama.cpp on one card, i.e. no decode gain, large prefill gain ([V]).
- Four B70 under vLLM: 369 tok/s average with 50 concurrent requests on a 27B model, about equal to four RTX 3090
  (Level1Techs, via Hardware Corner, [S]).
- Relevance to Tona: none for one-card decode. Two B70 would only matter for models above 32 GB; the engine has no
  multi-GPU design today.

## 8. Estimate: Tona's large model, fully resident on one B70 (an estimate, not a measurement)

Method of [pc-24gb-and-larger.md](../docs/hardware/pc-24gb-and-larger.md#estimates): bytes per pass 4.11 GB dense +
0.98 GB experts = 5.1 GB; GPU part = bytes / effective bandwidth; upper reading divides by the calibration factor
0.65, lower reading adds a fixed 10 ms per pass; 2.5 tokens per pass (MTP). All experts resident, no CPU part.

| Effective bandwidth | Share of 608 | GPU part | Upper (/0.65) | Lower (+10 ms) |
|---:|---:|---:|---:|---:|
| 608 GB/s | 100 % (spec, not reachable) | 8.4 ms | 194 tok/s | 136 tok/s |
| 486 GB/s | 80 % (a tuned engine) | 10.5 ms | 154 | 122 |
| 395 GB/s | 65 % (what llama.cpp SYCL reaches on dense models today) | 12.9 ms | 126 | 109 |

If the grid barrier is not usable and every one of the 1,100 dependent steps becomes a dispatch of t microseconds
(Intel launch cost unmeasured; the reference's 2.1 us is a HIP figure), replace the reference's 0.8 ms of barriers:
extra time 1,100 t - 0.8 ms = +4.7 ms at 5 us, +10.2 ms at 10 us. In the middle row that gives about 120 / 99 tok/s
(5 us) and 95 / 81 tok/s (10 us) for upper / lower.

| Case | Decode with MTP |
|---|---:|
| in-kernel barriers, tuned engine | about 120 to 190 |
| in-kernel barriers, SYCL-like bandwidth | about 110 to 125 |
| dispatch per step (5 to 10 us), tuned engine | about 80 to 120 |

Stated range **80 to 190 tok/s, central 110 to 150**; the reference measures 89 tok/s on 36 % resident experts with
CPU misses. The spread is dominated by the two numbers nobody has: achieved bandwidth of our kernels and launch
or barrier cost. Notes: the table of pc-24gb-and-larger lists the R9700 upper reading as 260 tok/s; the stated
method gives 8.0 / 0.65 = 12.3 ms and about 200 tok/s, so that cell looks inconsistent with its own method
(checked arithmetic only, not changed here).

Capacity: the large model needs about 30.8 GB with all experts resident ([L] pc-24gb-and-larger: 5.6 GB fixed + 25.15
GB experts) = 28.7 GiB; one B70 report shows 29.2 GiB usable headless. That leaves about 0.5 GiB: no desktop on the
card, a short context or a few experts off-card with a small CPU path.

Prefill (estimate): compute-bound when resident. Peak fp16 about 183 TFLOPS [D] against 93 measured on the reference;
today's best public Intel engine reaches about 21 % of that peak on a 7B model. If a tuned XMX path reaches 40 to
60 %, 70 to 110 TFLOPS: **roughly 1,700 to 3,500 tok/s**, about the reference's 1,741 up to twice it, unmeasured.

## 9. Open points, unchecked in this run

- Raw GDDR6 read bandwidth of the B70 (any API); B70 L2 size and bandwidth.
- Host-flag visibility, cache path and latency from a running kernel; the PAT/caching attribute of `malloc_host` memory
  on `xe`; `usm_atomic_host_allocations` aspect value on a B70.
- Whether a spinning kernel is preempted cleanly (mid-thread preemption, 640 ms preempt timeout) when a compositor
  shares the GPU; behavior of the B70 as a headless second GPU next to the AMD desktop GPU.
- Cooperative launch support and `max_num_work_groups_sync` on Xe2; residency of a hand-built barrier on Xe2.
- Launch cost per dependent dispatch and per round trip through SYCL queues, immediate command lists and raw Level Zero;
  host-signaled event latency; copy-engine overlap with compute.
- B70 PCIe 5.0 x16 host-to-device bandwidth; peer-to-peer.
- Xe2 XMX shapes and sustained int8 and fp16 rates from `joint_matrix` and ESIMD; whether int8 x int4 exists; fp8.
- Fedora packaging of the oneAPI compiler; the Fedora B60 kernel bug.
- OpenVINO GenAI performance; Intel's own B70 vLLM numbers beyond the 4-card page; the Phoronix B70 Linux review (403).
- The text of the B50 "timeout" report and whether it is the xe job timeout.
- The 2026-10-24 and later posts of the Level1Techs thread beyond what I read.

## Sources

[V] opened: Intel oneAPI GPU optimization guide, Intel Xe GPU architecture and atomic operations pages (2025.2);
[Chips and Cheese, Intel's Battlemage Architecture](https://chipsandcheese.com/p/intels-battlemage-architecture);
kernel `xe_guc_submit.c` and `Kconfig.profile` (master); compute-runtime `ioctl_helper_xe.cpp` (master) and releases;
[PMZFX benchmarks](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks) (pushed 2026-04-22; README, hardware,
engine-comparison, FINDINGS); llama.cpp [discussion 23313](https://github.com/ggml-org/llama.cpp/discussions/23313),
issue 21517 (closed 2026-04-07), PR 14001 (merged 2025-06-05), PR 29864 (merged 2026-10-22);
[StorageReview B70 review](https://www.storagereview.com/review/intel-arc-pro-b70-review); Igor's Lab B70 review;
[bentech](https://bentech.substack.com/p/running-local-llms-on-intel-arc-pro); Level1Techs B70 thread page 8;
Hardware Corner (2026-03-26); intel/llvm `sycl_ext_oneapi_root_group`; Sorensen et al., OOPSLA 2021 (PDF text);
Fedora packages page. [S]: search results for Level Zero cooperative kernels, SYCL USM atomics, joint_matrix, Battlematrix,
IPEX-LLM archive, Intel community threads. [L]: this repository.
