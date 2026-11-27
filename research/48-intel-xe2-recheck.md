# Intel Arc Pro B70 for Tona: verification and extension of the 2026-11-14 research

Date of this run: 2026-11-17. Web research only. Nothing run, built or measured on any GPU. Nobody contacted.
Marks: [V] page opened and read in this run; [S] search snippet only; [K] from memory, unchecked; [D] derived (arithmetic shown).
A forum or blog figure is [V] only as "that person wrote it". Earlier note = `research/48-intel-xe2.md` and `docs/backends/intel-xe2.md` (2026-11-14).
Fetch problems in this run: the kernel file `xe_hw_engine_class_sysfs.c` and the compute-runtime issue page returned HTTP 429 / an expired certificate, so the sysfs knob details are not verified.

## 1. Short answer

**CAN (evidence in hand)**
- Run a kernel for the length of a request on the `xe` driver without the 5 s job timeout. Re-read of kernel master today: LR-mode queues get `MAX_SCHEDULE_TIMEOUT` [V]. Still not tested on a B70.
- Use XMX with fp16, bf16, int8, int4, int2 (Intel's own slide). No fp8 [V].
- Fit the model only if the card is headless and the context is short: 30.8 GB = 28.7 GiB against 29.2 GiB usable reported by one person [D, earlier claim, not re-verified].
- Reach a useful share of bandwidth with someone else's engine today: B70 gpt-oss-20b MXFP4 decode 144.8 tok/s on 2026-11-16, Qwen3.6-35B-A3B Q4_K_M on a B60 83.4 tok/s decode on 2026-11-05 (llama.cpp SYCL, discussion 23313) [V as "that person wrote it"].

**CANNOT (or officially not supported)**
- Rely on atomics on `malloc_host` memory or concurrent host/device access to `malloc_shared` on a discrete Intel GPU: Intel's guide says "not supported" [V, 2025.2].
- Use fp8 on XMX [V slide: INT2, INT4, INT8, FP16, BF16 only].
- Count on peer-to-peer between two B70: the only multi-card report (Puget) had to turn P2P off after bus errors and engine resets [V, "that person wrote it"].
- Count on P2P between the B70 and the AMD card: no mechanism found; assume host-staged copies [K, D].

**UNKNOWN (nobody published it; the decisive items)**
1. Does a running kernel see a CPU write to a `malloc_host` flag, and with which load (plain, atomic, uncached)? Intel's slide says host allocations on discrete GPUs are "L1 cached, CPU->GPU coherent" [V]; whether a spinning thread ever drops its stale L1 line is not stated anywhere I found.
2. Does `root_group` / cooperative launch work on Xe2, how many work-groups may be resident, and does a hand-built barrier on device USM atomics make progress with 32 work-groups?
3. Raw bandwidth of a B70 from a clean kernel (only one home-built probe: 443 GB/s, method unnamed).
4. Whether a spinning kernel survives preemption requests (compositor, memory eviction) on the `xe` LR path.
5. Max single allocation size (typical Level Zero limit is 4 GiB unless relaxed limits are requested [K]) for a 30 GB model.

**The one test that decides the biggest unknown** (headless B70, throwaway process, wait cap 100 ms, user warned): one small SYCL/Level Zero program with two kernels. (a) Query `usm_atomic_host_allocations` and the L0 host-allocation access capability, then a kernel that spins on a `malloc_host` flag (variants: plain load, `atomic_ref` relaxed load, ESIMD load with uncached hint, flag allocated with `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED`) while the CPU writes it after 50 ms; record seen/not seen and latency. (b) A kernel with 32 and 64 work-groups doing 1,000 hand-built barriers on device-USM atomics, then the same with `use_root_sync`; record deadlock or cost per barrier. Result (b) decides in-kernel megakernel versus dispatch chain (about 80 to 120 versus 110 to 190 tok/s in the earlier estimate); result (a) decides whether the reference's CPU hand-over pattern can exist at all. On a 32 GB card (b) matters more, because all experts are resident.

## 2. Verdict on earlier claims (research/48-intel-xe2.md)

| Earlier claim | Still holds? | Evidence today |
|---|---|---|
| Job timeout not applied to LR VMs (`MAX_SCHEDULE_TIMEOUT`) | Holds | `xe_guc_submit.c` master: `timeout = (q->vm && xe_vm_in_lr_mode(q->vm)) ? MAX_SCHEDULE_TIMEOUT : msecs_to_jiffies(q->sched_props.job_timeout_ms);` [V]. New detail: in `guc_exec_queue_timedout_job` an assert says "LR jobs can only get here if queue has been killed or hit an error" [V]. |
| Preemption timeout 640 ms default, then reset | Not re-verified | Kconfig page not re-read; search shows `CONFIG_DRM_XE_PREEMPT_TIMEOUT` exists, "preempt timeout (us, jiffy granularity)" [S]. New in code: `set_min_preemption_timeout()` lowers it to 1 when scheduling is disabled immediately, and suspend waits `vm->preempt.min_run_period_ms` [V]. |
| Intel guide: atomics on malloc_host and concurrent malloc_shared not supported on discrete | Holds, text re-read | see point 1 |
| `root_group`: experimental, no forward-progress statement | Holds | see point 3 |
| No raw bandwidth test for B70 | Mostly holds | one probe found: 443 GB/s [V, method unnamed], see point 4 |
| Xe2 has no fp8 | Now [V] (was [K]) | Intel slide, point 5 |
| "256 KB L1/SLM block per Xe core" | CONFLICT | Intel's Xe2 slide says "192 KB Shared L1$ / SLM" per XeCore [V]; earlier value came from Chips and Cheese (B580). Unresolved; query `local_mem_size` and the cache size at runtime. |
| compute-runtime latest 26.40.40196.8 of 2026-11-03 | Holds | GitHub API [V]; also 26.35.39758.10 (2026-09-17, IGC v2.41.5, Level Zero v1.32.0), 26.31.39395.13 (2026-08-21) [V]. |
| MoE decode on B70 only 16 % of bandwidth | Outdated | newer rows below (point 4/8). |
| Sub-group 16 and 32, 128 KB SLM max per work-group, 1024 work-group | Not re-verified | |
| IPEX-LLM archived, vLLM mainline GDN bug | Not re-verified; vLLM issue #38994 still only seen second-hand [S], status unknown | |

## 3. Tables per point

### Point 1. Flag in host memory written by the CPU while the kernel runs

| Item | CAN / CANNOT / UNKNOWN | Evidence | Mark |
|---|---|---|---|
| Allocate host-pinned memory that the device can read | CAN | Intel GPU guide 2025.2, "Unified Shared Memory Allocations" (id 771772, 7/10/2025): "Allocation can be accessed by the host and any other device in the context." "The data stays on the host all the time and is accessed via PCI from the devices." "offloading regions that read from or write to this memory do it remotely through either PCIe bus or fabric link." | [V] |
| Atomics on `malloc_host` | CANNOT (officially) | Intel guide 2025.2, "Atomic Operations", discrete GPU: "Atomic operations on host allocated USM (sycl::malloc_host) are not supported." | [V] |
| Concurrent host/device access to `malloc_shared` | CANNOT (officially) | same page: "Concurrent accesses from host and device to shared USM location (sycl::malloc_shared) are not supported." | [V] |
| What the runtime reports on a B70 (`aspect::usm_atomic_host_allocations`, L0 host access capabilities) | UNKNOWN | Level Zero spec "latest" Core Programming Guide, USM access table: host allocation "Buffer can be atomically accessed from hDevice"; "Concurrent atomic access is at the granularity of the whole allocation" [V]. That is the capability model; what a B70 advertises is not published. An 11th-gen Iris Xe forum post says that iGPU advertises neither aspect [S]. | [V]/[S] |
| Coherence model on discrete parts | UNKNOWN (partly answered) | Intel IWOCL 2026 slides, M. Mrozek, "Optimizing AI Workloads on Intel GPUs With OpenCL", slide 10, "Host Allocations": Discrete parts: "L1 cached", "CPU->GPU coherent", "GPU -> CPU requires L1 flush". Integrated: "L1/L2 cached ... GPU -> CPU requires L1+ L2 flush". Read from the PDF text layer. This says a CPU write becomes visible to the GPU, but a spinning thread's own L1 copy is a risk; the slide does not say whether the line is invalidated while the kernel runs. | [V] |
| Cache-bypass loads (uncached flag, atomic load, ESIMD uncached load) | UNKNOWN | compute-runtime has an `uncacheable` bit in `AllocationProperties::flags` / `AllocationData::flags` [V, `allocation_properties.h`]; how it is consumed (memory type, PAT) not read. `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` is a host-allocation hint flag [K]; its spec text was not found (the "latest" Programming Guide only says cacheability hints "may be provided via separate host and device allocation flags when memory is allocated" [V]). ESIMD LSC loads with uncached cache controls [K]. | [V]/[K] |
| Plain `volatile` load | UNKNOWN | nothing found; SYCL has no volatile-visibility guarantee across agents; do not rely on it [K]. | |
| A tested example of GPU polling host memory on Intel discrete | UNKNOWN (none found) | searched oneCCL, Intel MPI, llama.cpp SYCL, IPEX, vLLM XPU, OpenMP offload (LLVM offload L0 plugin): no code or test that spins on a host flag found. oneCCL scale-up default is the "topo" algorithm using kernels or copy engines, not a host proxy [S]. | [S] negative |
| Alternative: command streamer waits | UNKNOWN | Intel's ULLS slides show `SEMAPHORE` commands between `WALKER` dispatches in one batch buffer ("Submission model explained", ULLS flow) [V]; so the hardware command streamer can wait on memory between kernels. Level Zero API for it: `zeCommandListAppendWaitOnEvents` with a host-signalled event, or `zexCommandListAppendWaitOnMemory` [K, not read]. Latency unpublished. | [V]/[K] |
| Spec text on host-signalled events and counter-based events | Counter-based events: host allocation must be CPU-accessible USM [V spec text]; host signalling of counter-based events not allowed [earlier note, S]. | | |

Tona consequence: with all experts resident on the B70 the CPU never has to answer inside a pass. If decisions (accept or reject of MTP drafts) must come from the CPU mid-pass, a flag is still needed; otherwise a doorbell per launch is enough. Decide with test (a).

### Point 2. Long-running kernels on `xe`

| Item | Verdict | Evidence | Mark |
|---|---|---|---|
| Job timeout (5 s default, 10 s max) applies to compute VMs | CANNOT happen (derived) | `guc_exec_queue_init()`: LR-mode VM gets `MAX_SCHEDULE_TIMEOUT` [V today, kernel master]. The compute runtime creates VMs with `DRM_XE_VM_CREATE_FLAG_LR_MODE` [earlier note, V 2026-11-14, `ioctl_helper_xe.cpp`; not re-read]. Not tested on hardware. | [V] |
| Preemption timeout (640 ms) and what happens | UNKNOWN for a spinning kernel | Kernel doc text (earlier note, V): "If the context does not yield before the timer expires, the HW is reset". Whether a spinning kernel reaches an arbitration point is not published. Mid-thread preemption on Xe2 [K]. In LR mode the VM can also be suspended for memory eviction or userptr invalidation (`min_run_period_ms`, suspend path in `xe_guc_submit.c`) [V code, D reading]; with a full 29 GiB VRAM this path is a risk. | [V]/[D] |
| Raise the timeouts per queue without root | UNKNOWN (likely no) | per-engine-class sysfs `preempt_timeout_us`, `job_timeout_ms` exist (IGT test `xe_sysfs_preempt_timeout`) [S]; file permissions not read (429), root-only is the usual mode [K]. A per-queue override through exec queue properties was not checked. | [S]/[K] |
| Display on the same card | Matters only for preemption | a compositor is the main source of competing contexts [D]; on a headless B70 no compositor context competes. Plan stays: B70 headless, AMD card keeps the desktop. | [D] |
| Real hang reports | exist | B50 pair: "driver ends compute due to timeout" (Intel community thread 1743709) [S, text not visible]; compute-runtime 25.44.36015.8 "hotfix ... Battlemage Pro ... very long (multi-day) running workloads" [earlier V]; Arrow Lake Xe2 iGPU, sustained OpenCL compute from llama.cpp hangs, xe resets the engine (compute-runtime issue 939 [S], status unchecked). A Proxmox B580 user saw `xe_guc` failures only under heavy decode load [S]. | [S] |

### Point 3. Grid-wide barrier

| Item | Verdict | Evidence | Mark |
|---|---|---|---|
| `sycl_ext_oneapi_root_group` status | experimental | intel/llvm extension text: "experimental extension specification, intended to provide early access ... not finalized and may change incompatibly" [V, raw asciidoc on the `sycl` branch, 2026-11-17] | [V] |
| Barrier call | CAN (API exists) | `sycl::group_barrier(root_group)`; kernel must carry property `use_root_sync`; nd_range incompatible with it throws `errc::nd_range`; calling the barrier in a kernel without it is undefined [V] | [V] |
| Co-residency / forward progress guarantee | UNKNOWN (spec silent) | the spec "has no explicit forward progress or co-residency guarantee"; kernel "must be launched with a limited number of work-groups"; the query `max_num_work_groups_sync` returns the maximum allowed; a commit says the name was later shortened to `max_num_work_groups` [S, semanticdiff c99bd6fa], so check the current name in the installed headers [V] | [V]/[S] |
| Level Zero mapping | UNKNOWN on Xe2 | implementation note: Level Zero could use `zeCommandListAppendLaunchCooperativeKernel`; backends without cooperative launch may restrict to one work-group [V]. L0 spec "latest": "Cooperative kernels allow sharing of data and synchronization across all launched groups in a safe manner"; max from `zeKernelSuggestMaxCooperativeGroupCount`; "the total group count must be smaller than the queried maximum"; new path `zeCommandListAppendLaunchKernelWithParameters` with a cooperative descriptor [V]. This is the only wording close to a guarantee, and it is a guarantee only for the queried count. Which count a B70 returns: unpublished. | [V] |
| Hand-built barrier on device USM atomics | UNKNOWN | no Xe2 study; the 2021 OOPSLA study (earlier note) found non-conforming schedulers only on Apple and ARM and tested one Intel iGPU [earlier V]. XeCore has 64-bit atomics [V slide]. 32 work-groups = one per Xe core. | [V] |
| Compat tool | DPCT1087 maps CUDA grid sync to root_group only with `--use-experimental-features=root-group` [S] | | |

### Point 4. Bandwidth and capacity

| Item | Value | Evidence | Mark |
|---|---|---|---|
| Spec | 608 GB/s = 256 bit x 19 Gbps / 8 | StorageReview, Igor's Lab (earlier V) | [D] |
| Raw measured bandwidth | 443 GB/s = 73 % of 608 (443/608 = 0.729) | IDFS "Six days with the Intel Arc Pro B70": "a memory-bandwidth probe via large allocations and reads", run inside `intel/vllm:0.10.2-xpu`, PyTorch 2.10.0+xpu, kernel 6.17, early April 2026, API/script/buffer size not named; same script gives about 936 GB/s on an RTX 3090 [V, "that person wrote it"] | [V]/[D] |
| clpeak, BabelStream, mixbench on B70 | none found (two more searches) | | [S] |
| Best engine decode, bandwidth share (new rows) | B60 (456 GB/s [K]), Qwen3.6-35B-A3B Q4_K_M tg 83.4 tok/s (2026-11-05, build 3e4dfa8, fa 0): 83.4 x 1.8 GB = about 150 GB/s = 33 % [D, 1.8 GB/token from the earlier note]; B70 gpt-oss-20b MXFP4 tg 144.8 with fa 1 against 106.0 with fa 0 (2026-11-16, build b1b770d): 144.8 x about 1.9 GB (3.6 B active x 4.25 bit [K]) = about 275 GB/s = 45 % of 608 [D, rough]; B580 (456 GB/s [K]) Qwen3.8-27B IQ3_XXS tg 18.7 with fa 1 (2026-11-10): 18.7 x about 10.5 GB (file size [K]) = about 196 GB/s = 43 % [D, rough] | discussion 23313 table rows [V] | [V]/[D] |
| IQ-type quants | improved | issue 21517 had IQ4_NL at 14 % of bandwidth in April [earlier V]; a 27B IQ3_XXS row now gives about 43 % on a B580 [D]. IQ1_M, IQ2_S, Q2_0 on Intel: no data found. | [D] |
| Usable VRAM | 29.2 GiB reported on a B70 | bentech (earlier V); nothing new | [V] |
| Max single allocation | UNKNOWN | Level Zero/OpenCL devices usually cap one allocation (often 4 GiB) unless relaxed limits are requested [K]; the model would be allocated in chunks | [K] |
| Resizable BAR | required, treat as required | earlier note: [K] plus a Proxmox passthrough thread and a buyer report [S]; nothing new | [K]/[S] |
| PCIe 5.0 x16 host rate | not measured | boards often give x8 to a second card (PMZFX showed x8) [earlier V]; an AM5 board gives a second GPU x4 or x8 [K] | |

### Point 5. Compute features

| Item | Value | Evidence | Mark |
|---|---|---|---|
| XMX formats | INT2, INT4, INT8, FP16, BF16; no FP8 | Intel IWOCL 2026 slide 6, "Xe Matrix Extension Engines": "8 x 2048 bit XMX Engines ... INT2, INT4, INT8, FP16, BF16 ... FP16 2048 OPS/clock, INT8 4096 OPS/clock, 3-way co-issue FP + INT + XMX" | [V] |
| Peak | 32 cores x 4096 x 2.8 GHz = 367 TOPS int8; fp16 32 x 2048 x 2.8 G = 183 TFLOPS; the 2.8 GHz is back-computed from the 367 TOPS spec, not a published clock | | [D] |
| int8 x int4 mixed instruction | UNKNOWN | INT4 listed as a format; mixed-precision rule not stated | [V]/[K] |
| Vector engines | 8 x 512-bit per XeCore; 64-bit atomics; FP64; transcendentals | same deck slide 5 | [V] |
| L1/SLM per XeCore | 192 KB (Intel slide) vs 256 KB block (Chips and Cheese, earlier note); SLM max per work-group 128 KB [earlier V] | conflict, measure | [V] |
| Sub-groups 16/32, register file 128 or 256 GRF per thread, 64 KB per vector engine | earlier [V], not re-read | | |
| `joint_matrix` status, Xe2 shapes | UNKNOWN | llama.cpp uses ESIMD DPAS for multi-column kernels (PR 29864, merged 2026-10-22, earlier V); a third-party note about IGC-version-dependent failures [earlier S] | |
| int dot products | CAN (DP4A-class) [K] | not verified | [K] |

### Point 6. Software stack, 2026-11

| Item | State | Evidence | Mark |
|---|---|---|---|
| Kernel driver | `xe`, kernel 6.12 or later; B70 rigs on 6.17 (llm-scaler PV 1.4, Ubuntu 24.04 installer [S]) and 7.0 (PMZFX, Phoronix review); reference PC has 7.2 [L] | | [V]/[S] |
| Kernel news | Linux 7.1 gave the B70 "free performance" (no numbers, second-hand, 2026-06-10) [V, lavx.hu summary of Phoronix]; Linux 7.4 (stable about late December): Ultra Low Latency Scheduling on the migration queue with CPU binds, page-fault benchmarks "multiple GB/s" better, more power [S, Phoronix 2026-10-02]; vRAM health check with page offlining [S]. None of it targets LR kernels directly. | | [V]/[S] |
| compute-runtime | 26.40.40196.8 (2026-11-03), 26.35.39758.10 (2026-09-17), 26.31 (2026-08-21), 26.27 (2026-07-21), 26.22 (2026-06-16) [V, GitHub API]. B70 branding since 26.18.38308.1 (2026-05-13, Phoronix) [S]. Fedora 44 ships 26.35.39758.21 (earlier V). | | [V] |
| IGC / Level Zero | IGC v2.41.5 and Level Zero v1.32.0 in 26.35 [V] | | [V] |
| oneAPI DPC++ | a 2026.2 version of the Intel compiler guide exists (document dated 10/15/2026) [V]; the exact `icpx` build and Fedora packaging unchecked | | [V]/unchecked |
| Vulkan cooperative matrix on Xe2 | enabled in llama.cpp since PR 14001 (merged 2025-06-05, earlier V); no newer status found | | |
| Known bugs relevant to us | vLLM XPU: Qwen3.5 gated delta net kernel bug, issue #38994 [S, second-hand]; recurrent state leak in a custom build (bentech, earlier V); `DEVICE_LOST` when oversizing a KV pool (Level1Techs 2026-10-24, earlier V); Intel container needs `NEO_ReadDeviceBinaryBuiltins=0` and `VLLM_WORKER_MULTIPROC_METHOD=spawn` against ICD conflicts and fork crashes (Puget, [V]) | | |
| Intel vLLM recipes | `vllm-openai-xpu:latest` validated on B60 and B70 for Qwen3.5-35B-A3B; refers to vLLM PR #34571 for a CUDA graph and Mamba cache size error [S] | | [S] |

### Point 7. Two B70, or B70 plus the AMD card

| Item | Verdict | Evidence | Mark |
|---|---|---|---|
| P2P between B70 cards | UNKNOWN, leaning CANNOT today | Puget (2026-06-18, modified 2026-09-10), four B70: P2P off with `CCL_TOPO_P2P_ACCESS=0`; direct P2P gave "bus errors (RxErr)", engine resets and container deadlocks; after removing a riser they did not retest; traffic goes through host RAM (USM) [V, "that person wrote it"] | [V] |
| Multi-GPU speed-up | TP=4 on Llama 3.1 8B 35.4 to 70.3 tok/s (1.99x); Qwen3.6-35B-A3B TP=4 16.3 tok/s at 1 user [V, Puget] (low; vLLM, not our engine) | | [V] |
| Mixing vendors in one PC | CAN (separate devices and drivers `xe` + `amdgpu`), no direct GPU to GPU path | a second process or backend per card; copies through host memory [K, D]. Tona has no multi-GPU design today. Lane widths on the board (x8 or x4 for the second slot) and BAR space (32 GB BAR plus the AMD card) need a check [K]. | [K]/[D] |
| llama.cpp SYCL multi-GPU | PR 21597 (merged 2026-05-14) fixed system RAM exhaustion with multi-GPU by using Level Zero allocations, tested on two B70 [S] | | [S] |

### Point 8. New since 2026-11-14

Little. What I found, with dates:
- 2026-11-16: B70 gpt-oss-20b MXFP4 tg 144.8 tok/s (fa 1) vs 106.0 (fa 0), pp512 3425 vs 3007, Ubuntu 24.04, llama.cpp build b1b770d (discussion 23313 table) [V]. No bug reports in the post-11-01 rows.
- 2026-11-10: B580 Qwen3.8-27B IQ3_XXS tg 18.7; 2026-11-05: B60 Qwen3.6-35B-A3B Q4_K_M tg 83.4 (see point 4) [V].
- 2026-11-03: compute-runtime 26.40.40196.8 (already known to the earlier note) [V].
- Context before the note but not in it: llama.cpp PR 29809 (2026-10-10) said to double decode on Arc, 1.91x on one B70 gpt-oss-20b [S, blog runaihome.com, unverified against GitHub]; PR 25222 (merged 2026-07-15) XMX attention path [S]; Linux 7.4 xe pull (2026-10-02) [S].
- Nothing new found on: SYCL root_group on Xe2, host-flag polling, a raw B70 bandwidth test, SGLang XPU, papers.

## 4. What this changes for the design

1. Barrier and flag are two separate unknowns; one 100-line test settles both (see short answer).
2. Intel's slide text "L1 cached, CPU->GPU coherent" for host memory on discrete GPUs is the first vendor statement that is neither "supported" nor "not supported": test with a cache-bypass load first; plain loads may spin forever on a stale L1 line.
3. fp8 prefill from the reference has no Xe2 counterpart (now [V]); int8 and int4 XMX do exist.
4. The bandwidth floor moved: 443 GB/s probe (73 %) and 33 to 45 % in engines for MoE and IQ models, up from 16 %. The earlier estimate range (80 to 190 tok/s) is unchanged; the low edge is less pessimistic about the memory system, the barrier is still the larger risk.

## 5. Sources

Opened and read ([V]):
- Intel oneAPI GPU Optimization Guide 2025.2, Atomic Operations: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/atomic-operations.html
- Same guide, Unified Shared Memory Allocations (id 771772): https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/current/unified-shared-memory-allocations.html
- Intel IWOCL 2026 slides, Mrozek, "Optimizing AI Workloads on Intel GPUs With OpenCL": https://www.iwocl.org/wp-content/uploads/IWOCL-2026-Mrozek-Optimizing-AI.pdf (slides 5, 6, 10, ULLS slides)
- intel/llvm `sycl_ext_oneapi_root_group.asciidoc` (sycl branch): https://raw.githubusercontent.com/intel/llvm/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_root_group.asciidoc
- Level Zero Core Programming Guide, latest: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html
- Linux `drivers/gpu/drm/xe/xe_guc_submit.c`, master: https://raw.githubusercontent.com/torvalds/linux/master/drivers/gpu/drm/xe/xe_guc_submit.c
- compute-runtime `allocation_properties.h`: https://raw.githubusercontent.com/intel/compute-runtime/master/shared/source/memory_manager/allocation_properties.h
- compute-runtime releases (GitHub API): https://api.github.com/repos/intel/compute-runtime/releases
- llama.cpp discussion 23313 (open): https://github.com/ggml-org/llama.cpp/discussions/23313
- IDFS "Six days with the Intel Arc Pro B70": https://idfs.ai/blog/six-days-with-the-intel-arc-pro-b70
- Puget Systems, "Intel Arc Pro B70: Multi-GPU AI Inference Performance": https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/
- lavx.hu summary of Phoronix Linux 7.1 B70: https://news.lavx.hu/article/intel-arc-pro-b70-picks-up-free-performance-on-linux-7-1-s-battlemage-tuning

Search snippet only ([S]):
- Phoronix, Xe ULLS and CPU binds for Linux 7.4 (2026-10-02; page returned 403): https://phoronix.com/news/Intel-CPU-Binds-ULLS-Migration
- compute-runtime issue 939 (Arrow Lake iGPU hangs; page not readable): https://github.com/intel/compute-runtime/issues/939
- Intel community, B50 pair timeouts: https://community.intel.com/t5/Intel-Arc-Discrete-Graphics/Driver-ends-compute-due-to-timeout-on-dual-Arc-Pro-B50/m-p/1743709
- intel/llvm commit c99bd6fa (root group property doc): https://app.semanticdiff.com/gh/intel/llvm/commit/c99bd6faeaa42ccefeda4a1954231cc9b51d7540
- DPCT1087: https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/developer-guide-reference/2025-1/dpct1087.html
- vLLM recipes Qwen3.5-35B-A3B: https://recipes.vllm.ai/Qwen/Qwen3.5-35B-A3B.md
- llama.cpp PR 29809 via https://runaihome.com/blog/intel-arc-llama-cpp-2x-speed-optimization-2026/ (secondary)

Not found despite searching: any GPU-spins-on-host-flag code or test on Intel discrete; Level Zero text for `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED`; a clpeak/BabelStream B70 result; the xe sysfs permissions; root_group on Xe2.
