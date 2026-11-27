# Host flag on the Intel Arc Pro B70 (Xe2-HPG, BMG): can a running kernel see a CPU write?

Date: 2026-11-17. Method: reading source code of Intel's own libraries, compiler, runtime, memory-policy library and the
Linux `xe` driver (local shallow clones, single files via the GitHub API). Nothing was built,
run, or executed on any GPU. Nobody was contacted.

Marks: **[V]** code or page read by me in this run (repo path:line at the commit listed in "Sources");
**[S]** search snippet only; **[K]** from memory, unchecked; **[D]** derived by me from [V] inputs (reasoning stated).
Evidence classes: **(doc)** Intel documents it; **(prod)** shipped code relies on it; **(forum)** someone wrote it.

Context: `report.md` point 1 and `research/48-intel-xe2.md` section 3.2. Already known there: Intel's GPU guide 2025.2
says atomics on `malloc_host` and concurrent host/device access to `malloc_shared` are "not supported" on discrete GPUs;
Intel's IWOCL 2026 slides (Mrozek) say host allocations on discrete parts are "L1 cached, CPU->GPU coherent,
GPU->CPU requires L1 flush".

## 1. Short answer

| Question | Verdict | Evidence level |
|---|---|---|
| Kernel spins on a flag in **host memory** (`malloc_host` / `zeMemAllocHost`) and sees a CPU store | **UNKNOWN, likely CAN with an L1-bypassing load** | No (prod) user found. Not (doc)-supported: Level Zero on BMG does not report concurrent access for host allocations [V, D]. The mechanism in the code points to "works if L1 is bypassed": dGPU accesses to system memory are snooped (xe driver comment) [V], and the GPU L1 is read-cached by default [V] |
| Kernel spins on a flag in **device memory (VRAM)** that the CPU writes through a BAR mmap | **CAN (prod)** | Intel SHMEM (lists Arc B-Series as supported) and oneCCL 2022.2 (host-initiated persistent kernel) both do exactly this [V]. Nobody published it on a B70 |
| Plain or `volatile` load in a spin loop | **CANNOT be relied on** | Intel code says a spinning default load "keeps hitting the stale L1 copy" (torch-xpu-ops) [V]; the IGC barrier fix needs an acquire fence per iteration or an L1-uncached load [V] |
| `sycl::atomic_ref` load as the poll | Works on **device** memory (prod); **avoid** on host memory | IGC lowers an atomic load to an atomic OR with 0, a read-modify-write [V]. On host memory, Level Zero does not report concurrent atomics on BMG [V, D]. Many pollers on one address can starve the writer: IGC fixed its own device-barrier hang for exactly this reason (2026-09-03) [V] |

**Recipe A (best supported, (prod)): CPU-written mailbox in VRAM**
1. Allocation: `zeMemAllocDevice` with `ze_external_memory_export_desc_t{ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF}` (ishmem)
   or `sycl::aligned_alloc_device(64, ...)` (oneCCL). Then `zeMemGetIpcHandle` gives a dma-buf fd, `mmap(PROT_READ|PROT_WRITE,
   MAP_SHARED, fd)` gives the CPU pointer, and `zeMemPutIpcHandle` follows at once [V both]. The BO must sit in
   CPU-visible VRAM (`xe_ttm_resource_visible` check, xe_bo.c:751-756) [V]. With ReBAR, which the B70 needs anyway, all
   VRAM is visible [K]. Compression must be off: ishmem *requires* `RenderCompressedBuffersEnabled=0` on Arc B [V];
   current compute-runtime has buffer compression off for BMG anyway (`ftrRenderCompressedBuffers = false`) [V].
2. CPU write: put the payload and the sequence number into **one 64-byte slot** and write it with `_movdir64b` (a single
   64-byte PCIe write; both libraries do this) [V]. The Zen 5 reference CPU has `movdir64b` [V /proc/cpuinfo].
3. GPU poll, one leader work-item per work-group: oneCCL and ishmem use
   `sycl::atomic_ref<uint32_t, acq_rel, system, global_space>(slot.seq).load()` [V]. The cheaper alternative with
   (prod) use on BMG for peer-written VRAM is an inline vISA `lsc_load.ugm.uc.ca` (L1 uncached, L3 cached) [V torch-xpu-ops, oneCCL ARC-B].
   Then `group_barrier`, then read the payload.
4. Reply GPU to CPU: put it in host memory, using a plain store plus `lsc_fence.ugm.evict.sysrel`
   (`__builtin_IB_lsc_fence_global_untyped(LSC_FS_SYSTEM_RELEASE, LSC_FT_EVICT)`) [V ishmem, torch-xpu-ops].

**Recipe B (closest to the reference, untested): host-memory flag**
1. Allocation: `zeMemAllocHost` with `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED`. In the code this maps to GMM
   `OCL_BUFFER_CSR_UC`, which becomes Xe2 PAT index 3 (L3 UC, L4 UC) [D from V chain, section 3]. On BMG this does **not**
   make the whole kernel uncached, because MOCS defers to PAT [V]. Without the flag, host USM gets PAT 1 (L3 WB, 1-way
   coherent) [D].
2. CPU write: payload first, flag last. Use plain stores, which are ordered under x86 TSO. The CPU mapping is WB, and
   GPU reads snoop it [V xe].
3. GPU poll: `asm volatile("lsc_load.ugm.uc.uc (M1, 16) %0:d32 flat[%1]:a64" ...)` with the kernel pinned to
   `sub_group_size 16`. The alternative is `__builtin_IB_lsc_load_se_global_uint(p, 0, LSC_LDCC_L1UC_L3UC)`, which needs
   IGC >= v2.42.6 (compute-runtime >= 26.40.40196.8). The non-`_se_` builtin may be hoisted out of the loop [V commit].
   After the flag matches, call `sycl::atomic_fence(acquire, device)` (lowers to an LSC fence with L1 invalidate) before
   reading the payload [V IGC].
4. Never use `atomic_ref` on the host flag (RMW over PCIe, not advertised as concurrent-safe) [D].

What decides between A and B for Tona: with all experts resident on a 32 GB card, the mailbox only carries decisions
(small), so Recipe A's BAR writes cost almost nothing. B matters only if the GPU must read bulk data the CPU writes into
host memory mid-pass, which is the reference's expert-miss path.

## 2. Per project

### 2.1 Intel SHMEM (oneapi-src/ishmem, cdfdaeb, 2026-08-04): GPU-initiated ops with a host proxy thread
- Design comment: `src/proxy.cpp:20-23`: "the host has to be able to write the completion array ... so those things must
  be 64 byte aligned and mapped to host memory. **They must be in device memory so they can be cached**" [V] (prod).
- GPU to CPU requests go to host memory: `ishmemi_gpu_ring::sendbuf` "located in host memory" (`src/proxy_impl.h:259`),
  allocated with `zeMemAllocHost`, flags 0 (`src/proxy.cpp:128`, `src/accelerator.cpp:638-652`) [V].
- CPU to GPU completions go to device memory: `completions` "location in device memory" (`src/proxy_impl.h:261`).
  `ishmemi_gpu_info` comes from `zeMemAllocDevice` with a DMA_BUF export desc (`src/memory.cpp:102`,
  `src/accelerator.cpp:654-670`) and is CPU-mapped through `zeMemGetIpcHandle` + `mmap(MAP_SHARED, fd)`
  (`src/accelerator.h:94-121`, `src/memory.cpp:114-115`) [V].
- CPU write: `_movdir64b(&ishmemi_ring_host_completions[i], &comp)` (`src/proxy.cpp:86`, also 159, 165) [V].
- GPU poll: `sycl::atomic_ref<unsigned, acq_rel, system, global_space>` load in a bare `while` loop
  (`src/proxy_impl.h:91-105`, `198-204`, `241-246`). `ishmem_wait_until` uses a `seq_cst` system atomic load
  (`src/synchronization.cpp:21-35`) [V].
- GPU store to host memory: 7 plain stores, `__builtin_IB_lsc_fence_global_untyped(LSC_FS_SYSTEM_RELEASE, LSC_FT_EVICT)`,
  the 8th store (the sequence word), and the same fence again (`src/intrinsic.h:44-62, 71-95`). The commented-out
  `sycl::atomic_fence(seq_cst, system)` alternative sits on line 85 [V].
- Optional mode `ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP` (default **false**, `src/ishmem/env_defs.h:22`) puts the symmetric
  heap in host USM (`src/memory.cpp:49-61`). In that mode `ishmem_wait_until` would spin on host memory with an
  atomic_ref load. This is the only code I found where an Intel EU kernel would poll host memory, and it is off by default [V].
- Hardware: README lists "Intel Arc B-Series GPUs: `RenderCompressedBuffersEnabled=0` is **required**"
  (`README.md:247-248`). Default AOT targets are `xe-hpc,xe2` (`docs/source/compiling_and_running_programs.rst`) [V].

### 2.2 oneCCL (uxlfoundation/oneCCL 2022.2.0, 51ce639, 2026-11-03)
- Host-initiated persistent kernel: the CPU proxy produces commands and up to 32 work-groups consume them
  (`deps/libccl/src/proxy/host_initiated/`) [V] (prod).
- The ring is in device memory: "Command ring: device memory for optimal device polling + GDR mmap for host access"
  (`cmd_ring_context.hpp:52-53`, mmap at `65-69`). The mmap is `zeMemGetIpcHandle` + `mmap` + `zeMemPutIpcHandle`
  (`src/sched/entry/ze/ze_primitives.cpp:901-926`) [V].
- CPU write: `_movdir64b(dst, src)`, "single 64-byte PCIe write" (`cmd_ring_host.hpp:36-49`). The slot is exactly one
  cache line with `sequence` inside (`cmd_ring_types.hpp:75-89`) [V].
- GPU poll: leader-only, `sycl::atomic_ref<uint32_t, acq_rel, system, global_space>.load()` until the sequence matches
  (`cmd_ring_device.hpp:28-45`; call at `cmd_persistent_kernel.hpp:178`), then `group_barrier`. There is no timeout
  [V]; the ring-size comment calls the poll "no timeout" (`cmd_ring_types.hpp:100-108`).
- Host memory is used only for GPU-to-CPU traffic: completions and the kernel-started generation are written with a
  plain store plus `atomic_fence(acq_rel, system)` (`cmd_ring_device.hpp:60-70`, `cmd_persistent_kernel.hpp:229-239`).
  Comment: "accessed with plain load/store + system fence (**device atomics are not valid on USM**)"
  (`cmd_ring_types.hpp:37-41`) [V].
- Hint of history: "Submit exit command ... This eliminates the need for **polling lifecycle->finish in USM host
  memory**" (`cmd_ring_host.hpp:181-182`). An earlier version apparently had the GPU poll host memory and moved away.
  Why is not visible, because the repo is one squashed release commit [V, D].
- LL256 protocol, flag polling in VRAM: for ARC-B (non-ARCA) builds the poll is `lsc_load.ugm.uc.ca (M1,16) ... d32x4`;
  for ARC-A it is `uc.uc` (`src/coll/algorithms/utils/sycl_ll256.hpp:32-55, 131-146`; build switch in
  `deps/libccl/cmake/helpers.cmake:145-152`, "Enable ARC B-series LL256 support") [V].
- Device-API barrier (`include/oneapi/ccl/device_api/lsa_barrier.hpp:93-118`): relaxed device-scope atomic load in the
  spin, then a system acquire fence [V].

### 2.3 PyTorch XPU (intel/torch-xpu-ops, `src/xccl/Signal.hpp` at 0d1f9f5, 2026-10-12)
- Lines 29-32: "A default store leaves the flag in L1 write-back where the peer GPU never sees it, and **a default load
  in a spin loop keeps hitting the stale L1 copy**. Both accesses therefore use LSC messages with L1 bypassed (`.uc`)." [V] (prod)
- Lines 12-14: "(sycl::atomic_ref is not supported, use explicit fence instead)". This concerns IPC peer mappings [V].
- Poll: `asm volatile("lsc_load.ugm.uc.ca (M1, 16) %0:d32 flat[%1]:a64")`. Store: `lsc_store.ugm.uc.wb`. Release:
  `sycl::atomic_fence(acq_rel, system)` + `lsc_fence.ugm.evict.sysrel` (lines 52-90). The spin has no per-iteration fence
  (lines 100-108) [V].
- Lines 45-50: exec size and sub-group size must match. "pvc and bmg reject 8"; 16 is "Verified ... on dg2, pvc, bmg,
  lnl, ptl and xe3". PR #5310 (merged 2026-09-11) notes a 32-wide asm worked on bmg only because IGC happened to pick
  SIMD32 [V].
- Lines 104-105: "no timeout check: XPU has no device-side global timer yet, an IGC request is pending" [V].
- PR #4838 (merged 2026-08-24): moved to uncached LSC polling "to avoid peer-visibility issues due to L1 caching" [V].
- Scope: GPU-to-GPU signal pads, not CPU-written host memory.

### 2.4 IGC (intel/intel-graphics-compiler, e48954a, 2026-11-17)
- `atomic_ref.load()` on global memory becomes `__spirv_AtomicOr(p, scope, sem, 0)`: "Since we dont have specialized
  atomic load/store HDC message we're using atomic_or( a, 0x0 ) to emulate an atomic load"
  (`IGC/BiFModule/Implementation/atomics.cl:118-135`). Codegen emits a SIMD1 `LSC_AtomicRaw(EATOMIC_OR)` with default or
  metadata cache options (`IGC/Compiler/CISACodeGen/EmitVISAPass.cpp:15894-15975`, `23618-23630`) [V]. So an atomic poll
  is a read-modify-write, and on host memory that is a PCIe RMW [D].
- Acquire semantics add an LSC fence with **L1 invalidate**; release adds L1 evict
  (`IGC/BiFModule/Implementation/include/atomic_fence_impl.h:16-28`; `EmitVISAPass.cpp:16872-16886`) [V].
- **`memory_scope::system` is lowered to GPU scope**: "TODO: CrossDevice should be mapped to LSC_FS_SYSTEM_RELEASE or
  LSC_FS_SYSTEM_ACQUIRE" and `(scope) == CrossDevice ? LSC_FS_GPU` (`barrier.cl:20-34`) [V]. This is why ishmem,
  torch-xpu-ops and oneCCL add explicit `sysrel` LSC fences for GPU-to-host writes [D].
- `volatile`: I found no cache-control effect in `EmitVISAPass.cpp` or `LSCCacheHintsPass.cpp` (grep, no hit) [V negative].
  Intel's commit bb606c5cd194 (2026-09-03, "Fix device barrier hang with many workgroups") says a waiting loop needs "a
  plain volatile load behind an acquire fence, which supplies the cache invalidate the load needs". The same commit
  says the old atomic-RMW wait let "the arrivals ... stop making progress" with many waiting workgroups [V].
- Commit 313901101298 (2026-09-11, "Poll global barrier with L1-uncached LSC load") adds `__builtin_IB_lsc_load_se_global_*`:
  "LSCLoad is readonly, so LICM can hoist it out of a loop ...; the new builtin is never hoisted". The barrier now polls
  with `LSC_LDCC_L1UC_L3C` and the comment "Bypass L1 to observe updates in L3 without invalidating the whole L1 each
  spin" (`barrier.cl:332-376`, load at 363) [V]. Both commits are in IGC v2.42.6 and not on the v2.41.5 branch (GitHub
  compare: "behind" vs "diverged") [V]. compute-runtime 26.40.40196.8 ships IGC v2.42.6; 26.35.39758.10 ships v2.41.5 [V release notes].
- LSC cache-control enums for user builtins: `LSC_LDCC_L1UC_L3UC = 1`, `L1UC_L3C = 2`, `L1IAR_L3C = 7`, plus Xe3P
  three-level forms (`IGCBiF_Intrinsics_Lsc.cl:75-133`). User code can call `__builtin_IB_lsc_*` by name; the pass
  matches prefixes (`IGC/Compiler/Optimizer/OpenCLPasses/LSCFuncs/LSCFuncsResolution.cpp:220-241, 297-333`) and needs a
  DG2-or-later product (line 390) [V].
- SPV_INTEL_cache_controls (SYCL `annotated_ptr` + `read_hint<cache_control<cache_mode::uncached, cache_level::L1>>`):
  SPIR-V cache level 0 maps to L1 and level 1 to LSC "L3" (the Xe2 device L2)
  (`IGC/AdaptorOCL/Utils/CacheControlsHelper.h:318-333`) [V]. The API is in `sycl_ext_intel_cache_controls`
  (experimental) [V intel/llvm doc]. Hoisting risk: such a load is an ordinary non-volatile load [D].

### 2.5 Intel compute-runtime (85bafcc, 2026-11-17): how host USM is mapped on BMG with xe
- `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` sets `locallyUncachedResource = 1` (`level_zero/core/source/context/context.cpp:183-185`),
  which becomes `AllocationProperties::flags.uncacheable` (`shared/source/memory_manager/unified_memory_manager.cpp:565`) [V].
  `BIAS_CACHED` has no handling in the host path (grep) [V]. Note the spec text: "host should not cache allocation (UC)"
  (`ze_api.h:7629`) [V]. The implementation acts on the **GPU** side; on dGPU the CPU stays WB anyway (section 3) [D].
- GMM usage: host USM (`bufferHostMemory`) gets `GMM_RESOURCE_USAGE_OCL_SYSTEM_MEMORY_BUFFER`. If uncacheable, it gets
  `OCL_BUFFER_CSR_UC`, because Xe2-HPG has `newCoherencyModelSupported = true`
  (`shared/source/gmm_helper/cache_settings_helper.cpp:22-35, 117-130, 142-152`;
  `shared/source/release_helpers/caps/caps_xe2_hpg.h:52`) [V].
- Allocation path on xe dGPU: a GEM BO in system memory, mmapped WB (`drm_memory_manager.cpp:3702-3790`). Its gmm usage
  comes from the uncacheable flag (`3182-3202`), and the PAT comes from `Drm::getPatIndex(gmm, ...)` (`3436-3465`,
  `drm_neo.cpp:1667-1739`) [V]. CPU caching is WB if the PAT is coherent, otherwise WC
  (`xe/ioctl_helper_xe.cpp:724-737`); the coherent Xe2 PAT list is {1,2,4,5,7,22,23,26,27,30,31}
  (`shared/source/os_interface/product_helper_xe2_and_xe3.inl:15-21`) [V].
- MOCS: on Xe2 `deferMOCSToPatIndex()` returns true (`product_helper_xe2_and_later.inl:18-20`), so an uncacheable kernel
  argument does **not** switch the whole kernel to uncached MOCS (`level_zero/core/source/cmdlist/cmdlist_hw.inl:6032-6041`) [V].
- L1 default for stateless access on Xe-HPG to Xe3, including BMG: `L1_CACHE_CONTROL_WBP`
  (`shared/source/helpers/cache_policy_from_xe_hpg_to_xe3.inl:34-37`; Xe2 enum WBP=0, UC=1, WB=2, WT=3, WS=4 at
  `generated/xe2_hpg_core/hw_cmds_generated_xe2_hpg_core.inl:1622-1628`) [V]. Debug switches
  `OverrideL1CachePolicyInSurfaceStateAndStateless` and `ForceAllResourcesUncached` exist (`cache_policy_base.inl:16-24`) [V].
- Host USM capability on BMG: there is no BMG override, so the default `access | atomicAccess` applies
  (`shared/source/os_interface/product_helper.inl:871-873`). Concurrent bits are added only under the debug flag
  `EnableUsmConcurrentAccessSupport` (`product_helper.inl:181-215`). DG2 overrides to `access` only
  (`xe_hpg_core/dg2/os_agnostic_product_helper_dg2.inl:209-211`) [V]. L0 reports this as `hostAllocCapabilities`
  (`level_zero/core/source/device/device.cpp:743-744`) [V]. So on BMG: RW and ATOMIC yes, CONCURRENT and
  CONCURRENT_ATOMIC no [D].
- Production use of "a GPU agent polls system memory written by the CPU": ring buffer and semaphore buffer of direct
  submission (ULLS) are forced to system memory on every platform (`shared/source/memory_manager/memory_manager.cpp:691-712`) [V].
  `zexCommandListAppendWaitOnMemory` emits `MI_SEMAPHORE_WAIT` on any USM pointer, host allowed
  (`cmdlist_hw.inl:5297-5340`) [V]. That is the **command streamer**, not an EU load (prod).
- L0 cache reservation: the upstream Linux build returns false (`level_zero/core/source/cache/linux/cache_reservation_impl_upstream.cpp:16-18`) [V].
- Side facts for point 4 of report.md: `CapsBmg::defaultMidthreadPreemptionDelayTimer = 3`, comment "MTP_TIMER_VAL_150 -
  give threads about to exit 150 us to retire instead of saving their state" (`caps_xe2_hpg.h:62`). BMG has mid-thread
  preemption with state save [V, D].

### 2.6 gmmlib (intel/gmmlib, 733c91a, 2026-09-09): usage type to PAT on Xe2
- Usage table columns are `L3_CC, L3_CLOS, L1CC, L2CC, L4CC, Coherency, IgPAT`
  (`Source/GmmLib/CachePolicy/GmmXe2_LPGCachePolicy.h:78`). `OCL_SYSTEM_MEMORY_BUFFER` = `1,0,0,0,0,1` (L3 WB, L4 UC,
  1-way IA snoop) (line 224). `OCL_BUFFER_CSR_UC` = all 0 (L3 UC, L4 UC, no snoop) (line 215) [V].
- PAT lookup by (L4, coherency, L3, CLOS, no compression) (`GmmXe2_LPGCachePolicy.cpp:133-193`). The fixed table has PAT1
  = 1-way, L4 UC, L3 WB and PAT3 = non-coherent, L4 UC, L3 UC (`:659, :661`). Hence host USM default is PAT1 and
  BIAS_UNCACHED is PAT3 [D].
- Restriction list: "No 2-way coherency on dGPU" (`:121`); "On BMG, L4 is a pass through" (`:140`) [V].

### 2.7 Linux xe driver (torvalds/linux master 666539e, 2026-11-16): see section 3.

### 2.8 Others
- intel/llvm (SYCL, 4a0e21d, 2026-11-17): `aspect::usm_atomic_host_allocations` is defined as `UR_DEVICE_INFO_USM_HOST_SUPPORT
  & ATOMIC_CONCURRENT_ACCESS` (`sycl/source/detail/device_impl.hpp:1339-1342`). UR maps it from
  `ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC` (`unified-runtime/source/adapters/level_zero/common/device.cpp:878-899`) [V].
  So the aspect is false on BMG [D, from 2.5]. Device clock: `sycl_ext_oneapi_clock` says "Current Intel GPU devices
  support only the `ext_oneapi_clock_sub_group` aspect" and the frequency may change; IGC implements it with
  `__builtin_IB_read_cycle_counter` (`IGC/BiFModule/Implementation/clock.cl:10-17`) [V].
- llama.cpp SYCL: `malloc_host` only for staging (`ggml/src/ggml-sycl/common.cpp:42`); no system-scope atomic anywhere
  (code search total 0) [V/S].
- oneDNN: `malloc_host` only in `src/gpu/intel/ocl/usm_utils.*`; `memory_scope::system` 0 hits [S, code search].
- Kokkos SYCL: host USM space for allocation only (`core/src/SYCL/Kokkos_SYCL_Space.cpp:188-215`); no polling found [V/S].
- SYCLomatic: `cuda::thread_scope_system` becomes `sycl::memory_scope::system` 1:1
  (`clang/lib/DPCT/RuleInfra/MapNames.cpp:1513-1515`); `cudaHostAllocMapped` is in the host-alloc set
  (`clang/lib/DPCT/RulesLang/MapNamesLang.cpp:214-218`) [V]. A migrated CUDA host-flag poll therefore becomes an atomic
  OR on host memory with a GPU-scope fence, which nothing guarantees on discrete [D].
- LLVM offload Level Zero plugin: `OL_ALLOC_TYPE_HOST` is a plain `zeMemAllocHost` (`offload/plugins-nextgen/level_zero/src/L0Memory.cpp:671-674`).
  No signalling code in that file; the rest of the plugin was not read [V partial].
- IPEX (archived), RAJA, vLLM XPU kernels: not read in code for this question.

## 3. Xe2 PAT and coherence facts (BMG = GRAPHICS_VERx100 2001, "Xe2_HPG", `xe_pci.c:178`)

- PAT entry fields (`drivers/gpu/drm/xe/xe_pat.c:120-152`): no_promote, comp_en, l3clos, l3_policy (0=WB, 1=XD, 3=UC),
  l4_policy (0=WB, 1=WT, 3=UC), coh_mode (0=no snoop, 2=1-way, 3=2-way) [V]. The PAT has **no L1 field**; L1 policy comes
  from state (WBP default) or per-message LSC cache control [V 2.5, D].
- Xe2 table (`xe_pat.c:155-185`, entry: l3, l4, coh): 0 WB,UC,none; 1 WB,UC,1-way; 2 WB,UC,2-way; 3 UC,UC,none;
  4 UC,WB,1-way; 5 UC,UC,1-way; 7 UC,WB,2-way; 8 UC,WB,none; 9-12 and 14-15 compressed; 13 WB,WB,none; 20-31 CLOS 1-3
  copies. BMG: "Wa_16023588340 ... `n_entries = 28; /* Disable CLOS3 */`" (`:643-645`). Kernel-internal indices:
  NONE=3, WT=15, WB=2 (`:649-652`) [V].
- **dGPU system memory is snooped**: "DGFX system memory is always WB / ttm_cached, since other caching modes are only
  supported on x86. **DGFX GPU system memory accesses are always coherent with the CPU.**" (`xe_bo.c:604-608`) [V] (doc in code).
  uAPI: when mapping system memory on discrete GPUs, "the caching mode selected will then be overridden to
  DRM_XE_GEM_CPU_CACHING_WB, and coherency between GPU- and CPU is guaranteed" (`include/uapi/drm/xe_drm.h:893-900`) [V].
  This is coherence at the memory/snoop level. It says nothing about stale lines inside GPU L1/L2 during a running
  kernel [D]. Together with Mrozek's slide ("Discrete: L1 cached, CPU->GPU coherent") the only stale cache on the
  read side should be L1, but the default PAT1 for host USM says L3 WB. Whether BMG really keeps system-memory lines in
  its L2 is not stated anywhere I found. Recipe B therefore bypasses both (PAT3 and the `uc.uc` load) [D].
- What userspace may choose: any valid `pat_index` per VM bind (it is written into the PPGTT PTE). The KMD rejects
  `coh_mode == NONE` with CPU caching WB, and rejects NONE for userptr and imported dma-buf (`xe_vm.c:3842-3851,
  3969-3995`; uAPI `xe_drm.h:1090-1137`) [V]. VRAM objects must be CPU-WC (`xe_drm.h:826-828`). CPU mapping of VRAM needs
  the resource in the CPU-visible BAR (`xe_bo.c:751-770`, `xe_ttm_resource_visible`) [V]. compute-runtime exposes
  overrides behind `NEOReadDebugKeys=1`: `OverridePatIndexForSystemMemory`, `OverridePatIndexForDeviceMemory`,
  `OverridePatIndex` (`drm_neo.cpp:1667-1684`) [V].
- Atomic route: atomic loads are LSC atomics (2.4). On device memory they run at L3 (the Xe2 device L2) [K]. For host
  memory, Level Zero does not claim concurrent atomics on BMG (2.5). Treat host-memory atomics during CPU writes as
  undefined [D].

## 4. The minimal test that settles it on a B70

Preconditions (AGENTS.md rule 2 and GPU safety). Write the plan into the measurement file first: scope, expectation,
active tooling. The B70 must be headless; the AMD reference GPU is not touched. Use compute-runtime >= 26.40.40196.8
(IGC 2.42.6) so the `_se_` builtin exists. Run `quiet_check.py` before and after. Watch `dmesg -w` for
`xe ... GT reset`, `engine reset` and `job timedout`; one hit aborts the series. Each kernel's wait has a hard cap
(below), and the host never calls a blocking queue wait before it has written the flag.

Capability print (no kernel): `zeDeviceGetMemoryAccessProperties().hostAllocCapabilities` (expect 0x3 = RW|ATOMIC,
from 2.5 [D]) and `dev.has(aspect::usm_atomic_host_allocations)` (expect false).

Variants: allocation x load, each run 1,000 ping-pong rounds in one launch.
- Allocations: **H0** `zeMemAllocHost` flags 0 (PAT1); **H1** `zeMemAllocHost` + `BIAS_UNCACHED` (PAT3); **D0**
  `zeMemAllocDevice` + DMA_BUF export + `mmap` (Recipe A).
- Loads: **P** `volatile` load (negative control); **F** `volatile` load + `atomic_fence(acquire, device)` each
  iteration; **U** asm `lsc_load.ugm.uc.uc`; **UC** asm `lsc_load.ugm.uc.ca`; **S** `__builtin_IB_lsc_load_se_global_uint(...,
  LSC_LDCC_L1UC_L3UC)`; **A** `atomic_ref<acq_rel, system>.load()`. Run A on D0 only first, and on H0/H1 only as the very
  last item.

```cpp
// kernel: nd_range{16 * WG, 16}, [[sycl::reqd_sub_group_size(16)]]; WG = 1 first, then 32
struct Slot { uint32_t payload[15]; uint32_t seq; };              // 64 B, one cache line
// mbox: Slot in H0/H1/D0; ack: uint32_t[WG] in host memory (H0) for the GPU->CPU leg
k(Slot* mbox, uint32_t* ack, Result* out, uint64_t cap_cycles, uint32_t rounds) {
  if (sub-group 0 of each WG) {        // all 16 lanes run the (M1,16) asm, as torch-xpu-ops does; lane 0 decides
                                        // (other sub-groups wait at a group_barrier after the loop)
    for (r = 1; r <= rounds; ++r) {
      uint64_t t0 = clock<clock_scope::sub_group>(); uint32_t it = 0;
      while (LOAD(&mbox->seq) != r) {                              // variant-specific load
        if ((++it & 1023) == 0 && clock<sub_group>() - t0 > cap_cycles) { out->timeout_round = r; goto done; }
        if (it > (1u << 26)) { out->timeout_round = r; goto done; } // backstop if the clock misbehaves
      }
      atomic_fence(memory_order::acquire, memory_scope::device);   // L1 invalidate before the payload
      out->bad_payload += (mbox->payload[r % 15] != pattern(r));   // data written before the flag
      out->iters[r] = it;
      ack[wg] = r;                                                 // GPU -> CPU: store + sysrel evict
      asm volatile("lsc_fence.ugm.evict.sysrel" ::: "memory");
    }
  }
done: ;
}
// host: launch on an in-order immediate queue, do NOT wait.
// for r in 1..rounds: write payload (pattern(r)), then seq = r
//   (H*: plain stores, x86 TSO keeps order; D0: one _movdir64b of the whole Slot + _mm_sfence),
//   t_w = now(); spin on volatile ack[wg] == r with a 200 ms host timeout; rtt[r] = now() - t_w.
// after the last round (or a host timeout): q.wait(); read Result; check dmesg.
```

- Cap: `cap_cycles` equals 100 ms at the nominal clock (the `TONA_MEGA_YIELD_MS` analogue). The total kernel life stays
  well under the 640 ms preemption timeout, and LR mode means no job timeout [report.md point 2].
- Measure per variant: seen yes/no (timeout round), RTT median, p99 and spread over 1,000 rounds, polls per round,
  `bad_payload`, and the same with WG = 32 pollers on 32 slots and on one shared slot (atomic starvation per IGC
  bb606c5). One-way latency comes from RTT minus the GPU-to-CPU leg; measure that leg separately (GPU writes `ack`
  first, CPU timestamps).
- Expectation, to be written into the plan [K/D, wide tolerance because nothing is published]:
  - H0-P and H1-P: timeout in round 1 or 2 (stale L1).
  - H0/H1 with F, U, S: always seen. RTT of a few µs (two PCIe crossings, each poll a PCIe read). The reference AMD host
    flag measured 0.73 µs [L research/48 §3.2] and is the yardstick.
  - D0-UC and D0-A: seen, RTT around 1 to 3 µs.
  - A on H*: unknown; watch for lost CPU writes (seq never matches, or payload stale).
- What failure looks like:
  - Timeout with F/U/S on H1: CPU-to-GPU visibility from host memory is not usable; take Recipe A.
  - Seen but `bad_payload > 0`: the ordering or the acquire fence is wrong.
  - Seen only with WG = 1: polling starves.
  - A GPU reset in dmesg: stop and report.
- Active tooling to list in the plan: the NEO debug keys, if any (`NEOReadDebugKeys`, any PAT/L1 override; none in the
  baseline), the dmesg watch, and the compute-runtime/IGC versions. One optional extra run: H0-P with
  `OverrideL1CachePolicyInSurfaceStateAndStateless=1`. If it then succeeds, L1 was the stale cache.

## 5. What stays open
1. Whether BMG's L2 (LSC "L3") holds system-memory lines under the default PAT1, which would make `uc.ca` unsafe on
   host memory. Nothing in the code or docs I read says so; the test (H0-UC vs H0-U) answers it.
2. The latency of each path on a B70. Nothing is published.
3. Whether Intel will ever advertise CONCURRENT for host USM on BMG. Today it does not (code) and the guide says no (doc).
4. Behaviour of host-memory atomics under concurrent CPU stores: undefined, untested.
5. Side finding for the grid-barrier question (report.md point 3): IGC's own device-wide barrier hung with many
   work-groups until 2026-09-03 (fix in IGC 2.42.6). Fedora 44's compute-runtime 26.35 line ships IGC 2.41.x, which
   does not have it [V commits, D packaging].

## Sources (all [V] unless noted; commit = state read)
- Intel SHMEM: github.com/oneapi-src/ishmem @ cdfdaebed6f6 (2026-08-04)
- oneCCL: github.com/uxlfoundation/oneCCL @ 51ce63980991 (2026-11-03, "oneCCL 2022.2.0")
- compute-runtime: github.com/intel/compute-runtime @ 85bafcca003f (2026-11-17); releases 26.40.40196.8 and 26.35.39758.10 (GitHub API)
- IGC: github.com/intel/intel-graphics-compiler @ e48954ad5bf9 (2026-11-17); commits bb606c5cd194 and 313901101298; releases v2.42.6, v2.41.5
- gmmlib: github.com/intel/gmmlib @ 733c91a19baa (2026-09-09)
- Linux: github.com/torvalds/linux @ 666539eecdf4 (2026-11-16): `drivers/gpu/drm/xe/{xe_pat.c,xe_bo.c,xe_vm.c,xe_pci.c}`, `include/uapi/drm/xe_drm.h`
- torch-xpu-ops: `src/xccl/Signal.hpp` (last change 0d1f9f5d6e, 2026-10-12); PRs #4838 and #5310
- intel/llvm @ 4a0e21d79cf0 (2026-11-17): `sycl/source/detail/device_impl.hpp`, `unified-runtime/source/adapters/level_zero/common/device.cpp`, `sycl/doc/extensions/experimental/sycl_ext_intel_cache_controls.asciidoc`, `sycl_ext_oneapi_clock.asciidoc`
- Level Zero headers: github.com/oneapi-src/level-zero `include/ze_api.h` (main, API 1.21)
- SYCLomatic `clang/lib/DPCT/RuleInfra/MapNames.cpp`, `RulesLang/MapNamesLang.cpp`; LLVM `offload/plugins-nextgen/level_zero/src/L0Memory.cpp`; Kokkos `core/src/SYCL/Kokkos_SYCL_Space.cpp`; llama.cpp `ggml/src/ggml-sycl/common.cpp`
- Code search only [S]: oneDNN (`malloc_host`, `memory_scope::system`); llama.cpp (`atomic_ref memory_scope::system`, 0 hits)
- Local: `/proc/cpuinfo` of the reference PC (Ryzen 5 9600X: `movdir64b`, `movdiri`)
