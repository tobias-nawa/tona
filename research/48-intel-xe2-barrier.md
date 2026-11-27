# Grid-wide barriers and persistent kernels on the Intel Arc Pro B70 (Xe2-HPG, `xe` driver)

Date: 2026-11-17. Source reading only: nothing was built, run or measured on any GPU, and nobody was contacted.
Question: can Tona run one persistent kernel per pass, with about 1,100 grid-wide barriers per pass, for the length of a
request on a B70 (BMG-G31, 32 Xe cores), safely? And how?

Marks:
- [V] I read the code or page myself in this run (repo path:line or URL).
- [V*] A research sub-agent of this run read it; I did not re-read it. Spot checks that I re-read are marked [V].
- [S] Search snippet only.
- [K] From memory, unchecked.
- [D] Derived (the arithmetic is shown).

Where it matters, a claim is labelled as a documented guarantee (spec), production reliance (shipped code depends on it)
or a forum claim.

Source revisions read (local shallow clones):
- Linux master `666539eecdf4` (2026-11-16): `drivers/gpu/drm/xe`, `include/uapi/drm`.
- compute-runtime `85bafcca003f` (2026-11-17).
- IGC `e48954ad5bf9` (2026-11-17).
- intel/llvm `sycl` `4a0e21d79cf0` (2026-11-17): SYCL runtime, docs, Unified Runtime (UR) Level Zero adapter, e2e tests.
- oneDPL `3328cd254ef9` (2026-11-17).
- The sub-agents' clones (SYCLomatic, ishmem, oneCCL, torch-xpu-ops, kokkos, ginkgo, RAJA, llama.cpp, sycl-tla, xetla,
  vllm-xpu-kernels, sgl-kernel-xpu) are in the same folder, with commits named in section 4.

## 1. Short answer

| Question | Verdict | Basis |
|---|---|---|
| Co-resident grid barrier with 32 work-groups (one per Xe core) | **CAN** (high confidence; not yet run on a B70) | **Documented guarantee:** Level Zero cooperative launch "allow[s] sharing of data and synchronization across all launched groups in a safe manner" for a group count within `zeKernelSuggestMaxCooperativeGroupCount`. **Runtime:** DPC++ reports `concurrent` work-group progress at root-group scope for every Level Zero GPU and maps `use_root_sync` to `zeCommandListAppendLaunchCooperativeKernel`. **Driver:** compute-runtime enables cooperative dispatch on Xe2 and computes the maximum as full occupancy. **CI:** the intel/llvm root-group e2e test (`maxWGs` groups, two device barriers) is enabled on Level Zero, and precommit CI has a Battlemage runner. **Production reliance:** oneDPL's radix sort and oneCCL's ring kernels; vLLM PR 685 runs 32 x 512 persistent work-groups with grid hand-offs on 2x B70 (plain launch). |
| Known traps | 3 | (1) On BMG the maximum-group query can return too many groups when the kernel uses SLM ("causing a hang", oneDPL). (2) The driver checks the group count only for kernels that use IGC's device barrier; a hand-built barrier gets no check. (3) On Xe2, a relaxed atomic load in a spin loop can stall at low probability (oneDPL). Section 2 handles all three. |
| Persistent kernel for the length of a request | **CAN** (from source; not tested) | Every compute VM is created in long-running (LR) mode, and LR queues get `MAX_SCHEDULE_TIMEOUT`, so there is no job timeout. compute-runtime's own low-latency submission mode (ULLS) keeps a ring spinning on the compute engine by design. The only remaining killer is a preemption request that the kernel cannot honour (next row). |
| Preemption of a spinning kernel | **UNKNOWN for this card, expected CAN** | Mid-thread preemption is the default on Xe2: compute-runtime BMG default `MidThread`, and the kernel driver keeps the platform default. The driver forces thread-group-level preemption only if a fuse bit (`CFEG_WMTP_DISABLE`) says the hardware lacks it. Which SKUs have the bit set: not found. With mid-thread preemption, a spinning kernel is saved and restored and is not reset [K mechanism]. Without it, the work-groups never end, and the engine is reset after the 640 ms preemption timeout [V Kconfig text, D]. |
| Raise the preemption timeout without root | **CANNOT** | The sysfs knob `preempt_timeout_us` is mode 0644 (root writes), with a ceiling of 10 s from Kconfig. There is no per-queue preemption-timeout property in the uAPI. An unprivileged process may set its own queue's *timeslice* up to 10 s, but only through the raw `xe` ioctl; compute-runtime does not expose it. |
| Is an in-kernel barrier faster than a kernel boundary on Intel? | **Not clearly** (new finding) | PR 685 on 2x B70 measured 10.5-22.9 us of hand-offs per layer for 4-5 hand-offs. That is about 2-6 us each, including imbalance and the release fence. A dependent kernel in a replayed in-order command list costs 0.42-0.47 us on a B580 (tiny kernels). The research/48 fallback estimate (5-10 us per dispatch, giving 80-120 tok/s) is too pessimistic. Revised numbers are in section 5. |

## 2. Best recipe (evidence in sections 3-5)

1. **Card and process setup.**
   - Keep the B70 headless; the AMD card drives the desktop.
     - Why: render and compute share one reset domain [V `xe_gt.c:1180-1201`]. An engine reset caused by our kernel would also hit a compositor on the same card [D].
   - Run no other GPU process on the B70.
     - Why: any other context on the compute engine triggers GuC timeslicing (1 ms default [V `xe_hw_engine.c:553`]) and thus preemption requests.
   - Do not overcommit VRAM.
     - Why: BMG VMs are LR without page-fault mode, because `isPageFaultSupported` is false for BMG [V `product_helper.inl:368-370`, `ioctl_helper_xe.cpp:1346-1360`]. Any eviction of one of our buffers therefore suspends (preempts) our queue through a preempt fence [V `xe_vm.h:228-231`, `xe_preempt_fence.c:76`, `xe_guc_submit.c:1985-2010`].
2. **Launch.** Use a SYCL kernel with a cooperative launch, so that the documented guarantee applies:
   - `use_root_sync` if you use `group_barrier(root_group)`;
   - `work_group_progress<forward_progress_guarantee::concurrent, execution_scope::root_group>` for a hand-built barrier.

   Both set the cooperative flag [V `sycl/source/detail/kernel_data.hpp:248-260`], which becomes `UR_KERNEL_LAUNCH_FLAG_COOPERATIVE` [V `scheduler/commands.cpp:2563-2564`], which becomes `zeCommandListAppendLaunchCooperativeKernel` [V `unified-runtime/.../level_zero/v2/command_list_manager.cpp:262-268`].
   - Note: the root-group extension text changed in 2026 (`use_root_sync` became a launch property, and the query was renamed `max_num_work_groups_sync`), while the runtime still answers `kernel_queue_specific::max_num_work_groups` [V* agent, `kernel_impl.hpp:418-429`; V intel/llvm #21838]. Check the names in the installed headers.
3. **Grid size.** Use n = min(32, the query, your own estimate). Own estimate: one work-group per Xe core, 512 work-items, SIMD16, 256-GRF mode, which is 32 hardware threads per Xe core, the full core in large-GRF mode.
   - The query alone is not safe on BMG with SLM [V oneDPL `kt_utils.h:59-67`].
   - Round SLM up to the hardware granularity table, as oneDPL does [V `kt_utils.h:72-87`].
   - PR 685 uses the same shape: 32 x 512, 256 GRF, "one WG per Xe core, all resident" [V `mono_layers.hpp:25-26`, `:49`].
   - In 128-GRF mode the formula allows two 512-item groups per core [D, section 3.2]. Prefer one group per core anyway: it mirrors the reference's one block per CU.
4. **Barrier.** Start with `group_barrier(root_group)` and measure it (test T1). On BMG, IGC compiles it into a flag barrier (section 3.3), whose spins carry device-scope acquire fences (an L1 invalidate per iteration). If that is too slow, use a hand-built epoch barrier (oneCCL pattern [V `oneCCL/examples/sycl/sycl_lsa_allreduce.cpp:87-122`]; PR 685 uses monotonic per-stage counters [V `mono_layers.hpp:451-478`]):
   - Memory: one `uint32 cnt` and one `uint32 epoch` in `malloc_device`, zeroed once.
   - Arrive: `group_barrier(wg)`, then the leader does `fetch_add(1)` with `acq_rel` (or `release`) ordering and `memory_scope::device` scope.
   - Spin: the leader spins on an acquire load, or on a relaxed load plus an acquire fence every 256 iterations (oneDPL's Xe2 mitigation [V `sycl_radix_sort_kernels.h:653-668`]), or on an L1-uncached load. IGC's own atomic barrier uses `LSC_LDCC_L1UC_L3C` [V `IGC/BiFModule/Implementation/barrier.cl:361-365`]. In SYCL this is reachable through `sycl_ext_intel_cache_controls`, `read_hint<cache_control<cache_mode::uncached, cache_level::L1>>` [V `sycl_ext_intel_cache_controls.asciidoc:102-107`].
   - Leave: `group_barrier(wg)`.
   - With a hand-built barrier the driver does not check the group count (section 3.2), so step 3 is the only protection.
5. **Keep mid-thread preemption alive.**
   - IGC turns it off for a kernel that has no loops and few instructions, or that contains an fp64 division or fp64 sqrt [V `OpenCLKernelCodeGen.cpp:1319-1333`, `GenericShaderState.hpp:101-104`, `EmitVISAPass.cpp:1373-1381`]. A persistent kernel has loops, so only the fp64 rule matters: no fp64 `/` or `sqrt` in the megakernel.
   - Check the zebin flag `disable_mid_thread_preemption` of the built kernel [V `zeinfo_decoder.cpp:662-663, 755`].
6. **Bound every spin.**
   - Use an iteration cap plus `sycl_ext_oneapi_clock` sub-group clock deadlines (PR 685 gives up after about 12 s and sets a sticky health word [V `mono_layers.hpp:84-90, 274-284`]).
   - A stuck barrier must end the kernel with an error code, never hang it. That keeps a bug from turning into an engine reset.
7. **Fallback when a barrier fails or costs more than a boundary:**
   - Split the pass at the barrier into kernels.
   - Record the pass once as an in-order command list or SYCL graph and replay it per pass.
   - Keep MoE routing on the device so the graph never breaks (llama.cpp #30156 did exactly this "to keep GPU graphs unbroken" [V*]).
   - Section 5 gives the cost.

## 3. How it works underneath

### 3.1 DPC++ and Unified Runtime (intel/llvm `sycl` branch)

| Item | Finding | Mark |
|---|---|---|
| Progress claim | `getImmediateProgressGuarantee`: for `is_gpu() && backend::ext_oneapi_level_zero`, root-group and work-group scope return `forward_progress_guarantee::concurrent`; sub-group returns `weakly_parallel`. This is a hard-coded table, not a driver query | [V `sycl/source/detail/device_impl.cpp:399-425`] |
| Spec behind it | forward-progress extension: "it may only be possible to provide `concurrent` forward progress guarantees for backends that submit kernels eagerly and which support some notion of a 'cooperative' kernel launch"; the table maps work_group at root_group to `concurrent` on GPU & Level Zero; a kernel requesting more than the device can give must throw `errc::feature_not_supported` | [V `sycl_ext_oneapi_forward_progress.asciidoc:360-376, 455-495`] |
| Cooperative flag | set by `use_root_sync`, or by `work_group_progress` / `sub_group_progress` = `concurrent` | [V `kernel_data.hpp:248-260`] |
| Mapping | `UR_KERNEL_LAUNCH_FLAG_COOPERATIVE` → `zeCommandListAppendLaunchCooperativeKernel`, or `zeCommandListAppendLaunchKernelWithArguments` with `ze_command_list_append_launch_kernel_param_cooperative_desc_t` when the driver supports it | [V `commands.cpp:2563-2564`, `v2/command_list_manager.cpp:262-268, 1204-1214, 1313-1331`] |
| Max-group query | `queryMaxNumWorkGroups` → `urKernelSuggestMaxCooperativeGroupCount` → `zeKernelSetGroupSize` + `zeKernelSuggestMaxCooperativeGroupCount`, then capped by dynamic SLM per compute unit. If unsupported, the fallback is 0 or 1 | [V `kernel_impl.hpp:376-410`, `unified-runtime/.../v2/kernel.cpp:663-686`, `common/helpers/kernel_helpers.cpp:65-87`] |
| `group_barrier(root_group)` | one group: work-group barrier; otherwise `__spirv_ControlBarrier(Device, ...)` with `seq_cst` | [V `include/sycl/ext/oneapi/experimental/root_group.hpp:94-110`] |
| Status | root-group extension experimental: "Shipping software products should not rely on APIs defined in this specification" | [V* `sycl_ext_oneapi_root_group.asciidoc:51-56`; V in report.md] |
| Tests | `test-e2e/GroupAlgorithm/root_group.cpp`: XFAIL only for OpenCL non-CPU and native CPU, unsupported on CUDA; it launches `maxWGs * WorkGroupSize` with two root barriers and delays half of the groups. Precommit CI has "Intel / Battlemage Graphics", runner `bmg`, targets `level_zero_v1:gpu;level_zero_v2:gpu`. So the test runs on some BMG in CI [D; which SKU and pass history: not checked]. DG2 was temporarily disabled in 2024 (#13984, closed) | [V test lines 1-13, 92-103; V `.github/workflows/sycl-linux-precommit.yml:173-175`; V issue list] |

### 3.2 compute-runtime (Level Zero driver, BMG = Xe2-HPG)

**Maximum cooperative group count.** `KernelHelper::getMaxWorkGroupCount` [V `shared/source/helpers/kernel_helpers.cpp:34-82`]:
- `availableThreads = min(ThreadCount, threadsPerEU * EUCount)`, with threadsPerEU = 8, or 4 in large-GRF mode [V `gfx_core_helper_base.inl:833-840`].
- `threadsPerGroup = ceil(wgSize / simd)`.
- `max = availableThreads / threadsPerGroup`, clipped to whole groups per DSS (DSS = Xe core): `(threadsPerDss / threadsPerGroup) * dss` [V `:842-846`].
- If the kernel uses barriers: `min(max, dss * (32 / barrierCount))` [V `constants.h:97`].
- If it uses SLM: `min(max, dss * (128 KiB / usedSlm))` [V BMG `maxProgrammableSlmSize = 128`, `hw_info_bmg.cpp:50`].
- `adjustMaxWorkGroupCount`: cooperative dispatch is supported on Xe2 (`isCooperativeEngineSupported` true [V `os_agnostic_product_helper_xe2_hpg_core.inl:32-34`], render-compute engine not exposed [V `caps_xe2_hpg.h:57`, `engine_node_helper.cpp:272-279`]). It is divided by the number of compute engines (CCS), which is 1 on BMG [V `os_agnostic_product_helper_bmg.inl:34-36`, `gfx_core_helper_pvc_and_later.inl:37-76`]. So the result is plain full occupancy.
- B70 [D]: 32 DSS, 8 EUs x 8 threads = 64 threads per DSS (32 in 256-GRF mode).

  | Work-group, mode | Threads per group | Max groups |
  |---|---|---|
  | 1024 items, SIMD16, 128 GRF | 64 | 32 |
  | 512 items, 128 GRF | 32 | 64 |
  | 512 items, 256 GRF | 32 | 32 |
  | 1024 items, 256 GRF | 64 | 0 (does not fit) |

**What a cooperative launch does.**
- If the kernel uses IGC's sync buffer (its device barrier): the group count is checked against the max and fails with `ZE_RESULT_ERROR_INVALID_ARGUMENT`, and a sync buffer is allocated and patched [V `cmdlist_hw.inl:4388-4410`]. This happens only inside `if (kernel->usesSyncBuffer())` [V `cmdlist_hw_xehp_and_later.inl:262-273`]. **A hand-built barrier therefore gets no count check** [D].
- It marks the execution type `concurrent`, which sets `ComputeDispatchAllWalkerEnable` in COMPUTE_WALKER on Xe2 [V `command_encoder_xe2_hpg_core_and_later.inl:109-119`, `cmdlist_hw_immediate.inl:226`]. Hardware meaning of that bit: not documented in what I read [K: "dispatch all thread groups"].
- No separate cooperative engine on Xe2: only PVC creates `EngineUsage::cooperative` contexts [V `gfx_core_helper_xe_hpc_core.cpp:39` versus `gfx_core_helper_xe2_and_later.inl:32-87`].
- The compute queue group always advertises `ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS` [V `level_zero/core/source/device/device.cpp:388-391`].

**Residency guarantee.** There is no hardware reservation mechanism beyond "the count fits the occupancy formula". The guarantee is the Level Zero spec text plus this formula. Spec text [V Level Zero `ze_api.h:9785-9800` and report.md]:
- "Launch kernel cooperatively over one or more work groups".
- "only be used for a command list that are submitted to command queue with cooperative flag set".
- "Use zeKernelSuggestMaxCooperativeGroupCount to recommend max group count".

**Known bug.** oneDPL: "On BMG, zeKernelSuggestMaxCooperativeGroupCount suggests too many work-groups due to miscalculation of available SLM in certain cases", and the UR adapter dropped the dynamic SLM size: "This workaround is needed for: oneAPI versions <= 2026.2 and currently all compute-runtime versions" [V `oneDPL/include/oneapi/dpl/experimental/kt/internal/kt_utils.h:59-67`]. The current UR adapter clamps by dynamic SLM [V `kernel_helpers.cpp:65-87`]. The compute-runtime part is still open per that comment.

**Preemption defaults for BMG.**
- `defaultPreemptionMode = PreemptionMode::MidThread` [V `hw_info_bmg.cpp:46`].
- The interface descriptor gets `ThreadPreemption = true` in mid-thread mode [V `preemption_xe2_hpg_core.cpp:18-24`].
- The mode is downgraded per kernel only if the binary says `disable_mid_thread_preemption` [V `preemption.cpp:35-36, 84`].
- Feature flag `ftrWalkerMTP = true` [V `hw_info_bmg.cpp:79`].
- The Xe2 STATE_COMPUTE_MODE has mid-thread-preemption delay-timer and over-dispatch fields [V `hw_cmds_generated_xe2_hpg_core.inl:6275-6280, 6330-6340`].

**Long-running submission is the normal mode.**
- VMs are created with `DRM_XE_VM_CREATE_FLAG_LR_MODE`, plus fault mode only if supported, and it is not supported on BMG [V `ioctl_helper_xe.cpp:1346-1360`, `drm_neo.cpp:1418-1433`, `product_helper.inl:368-370`].
- Direct submission (ULLS: a ring that spins on memory on the compute engine) is supported on Xe2 [V `product_helper_xe2_and_later.inl:63-65`]. It is listed for CCS, CCS1 and BCS in the BMG table [V `hw_info_bmg.cpp:35-39`] and enabled when VM bind is available [V `os_context_linux.cpp:107-112`, `os_context.cpp:76-100`].

**Long-running hang hotfix.** compute-runtime 25.44.36015.8: "potential hang issues ... in very long (multi-day) running workloads" for Battlemage Pro [earlier V, research/48].

### 3.3 IGC: what `group_barrier(root_group)` becomes on BMG

- `__spirv_ControlBarrier(Execution == Device)` → `global_barrier()` [V `IGC/BiFModule/Implementation/barrier.cl:150-155`].
- `global_barrier()` uses the atomic version only for `PlatformType >= IGFX_NVL`; **BMG gets `__global_barrier_nonatomic`** [V `:430-436`]. Steps per barrier:
  1. Work-group barrier with a device-scope acq_rel fence.
  2. The first item of each work-group writes `syncBuffer[groupId] = 1`, then a release fence.
  3. The *last* work-group polls all flags. Each iteration runs an acquire fence, which is an L1 invalidate [V `:100-131`]. It then clears the flags, followed by a release fence.
  4. Every other work-group spins on its own flag, with an acquire fence each iteration.
  5. Work-group barrier.

  [V `:379-428`]
- The atomic version used on newer chips does `atomic_inc` on an alternating slot and spins with `__builtin_IB_lsc_load_se_global_uint(..., LSC_LDCC_L1UC_L3C)`. Comment: "Bypass L1 to observe updates in L3 without invalidating the whole L1 each spin" [V `:332-376`].
- The sync buffer is null unless the launch is cooperative, so `intel_is_device_barrier_valid()` / `__spirv_BuiltInDeviceBarrierValidINTEL` can test it at run time [V `:169-175`, `IBiF_intel_device_barrier.cl:9-14`].
- Fence scope mapping: Device and CrossDevice map to `LSC_FS_GPU` (a TODO says system scope should be system) [V `barrier.cl:20-35`].
- Cost [D, unmeasured]: the flag barrier on BMG has a two-step latency chain (arrive → last group sees it → release seen by all) and pays an L1 invalidate per spin iteration. Expect a few microseconds per barrier; measure it in T1.

### 3.4 `xe` kernel driver: timeouts, LR mode, preemption

| Item | Finding | Mark |
|---|---|---|
| Job timeout | `timeout = (q->vm && xe_vm_in_lr_mode(q->vm)) ? MAX_SCHEDULE_TIMEOUT : msecs_to_jiffies(job_timeout_ms)` | [V `xe_guc_submit.c:2144-2145`] |
| uAPI meaning of LR | "An LR, or Long Running VM accepts exec submissions to its exec_queues that don't have an upper time limit on the job execution time" | [V `include/uapi/drm/xe_drm.h:966-977`] |
| Defaults | job timeout 5000 ms (max 10000), timeslice 1000 us, preemption timeout 640000 us; preemption-timeout Kconfig text: "how long the driver waits for the current context to reach an arbitration point and yield ... If the context does not yield before the timer expires, the HW is reset" | [V `xe_hw_engine.c:548-557`, `Kconfig.profile:1-66`] |
| Who may change them | sysfs `preempt_timeout_us`, `timeslice_duration_us`, `job_timeout_ms` are mode 0644, so root only; range capped by Kconfig min/max (default 1 us to 10 s for preemption) | [V `xe_hw_engine_class_sysfs.c:191-192, 333-334, 370-401`] |
| Per-queue properties | only `PRIORITY` (above normal needs `CAP_SYS_NICE`), `TIMESLICE`, PXP, multi-group; **no preemption timeout** | [V `xe_drm.h:1364-1367, 1435-1441`; `xe_exec_queue.c:751-766, 1058-1059`] |
| Timeslice without root | `exec_queue_set_timeslice`: without `CAP_SYS_NICE` the value must lie within the class min/max, i.e. 1 us to 10 s by default | [V `xe_exec_queue.c:770-836`]. compute-runtime only sets priority 0 for low-priority contexts [V `ioctl_helper_xe.cpp:2238-2240`], so this is not reachable through Level Zero or SYCL [D] |
| Preemption level, Xe2 | rule `DISABLE_WMTP_ON_UNSUPPORTED_HW`: if `XEHP_FUSE4 (0x9114)` bit 20 `CFEG_WMTP_DISABLE` is set, CS_CHICKEN1 is set to thread-group level for every compute and render engine; otherwise the platform default (mid-thread) stays | [V `xe_hw_engine.c:337-349, 372-379`; `regs/xe_gt_regs.h:261-262`] |
| Debug knob | debugfs `gt*/gpgpu_preemption_level` (default, mid-thread, thread-group, command); applies to new **render** LRCs only, taints the kernel, refuses mid-thread with "MTP fused off in hardware" when fused | [V `xe_gt_debugfs.c:518-570`, `xe_lrc.c:1509-1537, 1642-1643`; commit 6a8789e948c1, 2026-08-27, V via GitHub API] |
| Old platforms | `WaDisableGPGPUMidThreadPreemption` only for graphics 12.00-12.10 (Tiger Lake class), not Xe2 | [V `xe_wa.c:643-648`] |
| Which SKUs fuse it off | not found; the 2024 patch series says "Xe2 supports mid thread preemption, which is enabled by default"; a proposal to expose WMTP status to userspace is not in the current uAPI | [S lists.freedesktop.org intel-xe 2024-01 and 2024-08; V uAPI grep negative] |
| Eviction | LR VM without fault mode = preempt-fence mode. Eviction or userptr invalidation → `suspend` → waits until the queue has run `min_run_period_ms` (5 ms default) since resume → schedule disable (normal preemption timeout). If the GuC does not answer within 5 s → "Suspend fence ... failed to respond, banning queue" | [V `xe_vm.h:228-231`, `xe_vm.c:1735-1739`, `xe_device.c:439`, `xe_preempt_fence.c:76-83`, `xe_guc_submit.c:1985-2010, 2486-2511, 2525-2560`] |
| Kill / teardown | deregister sets the queue's preemption timeout to **1 us** before schedule disable, so killing a process with a non-preemptible spinning kernel means an immediate engine reset | [V `xe_guc_submit.c:536-546, 1365-1372, 1552-1562`] |
| Reset | GuC engine reset → G2H → `xe_guc_exec_queue_reset_handler` logs "Engine reset: engine_class=..." and cleans up the queue | [V `xe_guc_submit.c:3483-3513`] |
| Reset domain | render and compute share one reset domain | [V `xe_gt.c:1180-1201`] |

### 3.5 What happens to a spinning persistent kernel

| Event on the B70 | With mid-thread preemption (expected default) | Without it (fuse set) |
|---|---|---|
| Nothing else uses the card | runs until done; no job timeout [V] | same |
| Another process submits compute work | GuC timeslices every 1 ms and preempts at thread level; state saved and restored, all work-groups come back together, so the barrier stays valid. Cost per switch about 2 x 20 MiB of state traffic [D: 2048 threads x 128 GRF x 64 B = 16 MiB + 32 x 128 KiB SLM = 4 MiB; at about 400 GB/s that is about 50 us each way] | context cannot yield, engine reset after 640 ms, queue lost; other render contexts in the reset domain reset too [D] |
| Our own buffer evicted (VRAM pressure) | suspended after at least 5 ms of run time, rebound, resumed; data may now live in system memory (speed cliff, not a correctness issue) [V path, D] | engine reset after 640 ms [D] |
| Process killed while spinning | preempted with a 1 us timeout, clean teardown [V, D] | engine reset [D] |
| Compositor on the same card | render and compute share the EUs [K]; a full-occupancy spinning kernel can starve render threads, the render job timeout fires (non-LR, 5 s), and the shared reset domain takes our queue too [D, unverified chain] | same, worse |

## 4. Per-project findings (production code)

| Project (commit) | Cross-work-group sync? | Code (path:line) | Memory, atomics | Launch, grid | Key comment | Mark |
|---|---|---|---|---|---|---|
| vLLM-XPU PR 685 `mono_decode` (open, 2026-11-04) | yes: grid hand-offs, 4-5 per layer | `csrc/xpu/mono_decode/xe_2/mono_layers.hpp:25-26, 49, 451-478, 1640` | device USM `uint32` counters; signal = release fence + group barrier + `fetch_add(release)`; wait = relaxed load spin, **no acquire fence** ("lowers to an L1 invalidate on every thread"; "L1 starts clean at every launch") | **plain launch**, 32 x 512, 256 GRF, "one WG per Xe core, all resident"; refuses fewer than 32 cores | measured on 2x B70: hand-offs cost 10.5 / 16.3 / 22.9 us per layer; "They include the release fence, which waits for outstanding weight loads"; with XPU graphs, base kernel gaps "about 0" | [V file and PR comment, "that person wrote it"] |
| oneDPL KT onesweep radix sort | yes: lookback spins | `sycl_radix_sort_kernels.h:653-668, 804-823`; `kt_utils.h:39-93` | relaxed `atomic_ref<device>` loads, acquire after 256 spins, release fence per tile | **cooperative** via `work_group_progress<concurrent, root_group>`; grid = min(query, tiles, own SLM-granular estimate) | "On Xe2, we have seen some low probability instances where the lookback gets stuck when using relaxed atomic loads"; the BMG query bug quoted in 3.2 | [V] |
| oneDPL single-pass scan | lookback, dynamic tile id | `single_pass_scan.h:160-178`, `cooperative_lookback.h:65-67, 159-174` | acq_rel device | plain; tile id by `fetch_add`, so earlier tiles are already running (no residency need) | n/a | [V*] |
| oneCCL ring / send / recv kernels | yes: `group_barrier(root_group)` | `sycl_barrier.hpp:70-82, 406-449`; `allreduce_large_sycl_ring.hpp:86,182`; `send_simple.hpp:68-71` | root barrier | **cooperative** (`use_root_sync`); grid min(count-derived, env, total_threads*4) | "use_root_sync is the launch property required by ext_oneapi_get_root_group(): without it the runtime rejects the kernel" | [V*] |
| oneCCL small kernels `kernel_barrier` | yes: counter spin | `deps/libccl/src/coll/algorithms/utils/sycl_barrier.hpp:120-137` | relaxed device `+=1`, spin on load | **plain**, capped at HW thread count, else split into several kernels | "use separate kernels if there are more sycl threads than hw threads to avoid deadlocking in sync barriers from single kernels" (`allreduce_small_sycl_impl.hpp:289`) | [V* ; quote V] |
| oneCCL LSA example | yes: grid barrier | `examples/sycl/sycl_lsa_allreduce.cpp:87-122, 229-250` | device USM `cnt` + `epoch`, acq_rel device | **plain**, default 16 x 512, check `global <= CU x max_wg` | "the intra-GPU grid barrier deadlocks if the whole grid is not simultaneously resident" | [V lines 229-244; rest V*] |
| SYCLomatic dpct `nd_range_barrier` | yes (deprecated) | `dpct/util.hpp:791-877` | one counter, seq_cst device, top-bit flip | **plain**, no check | "Please make sure that all the work items of all work groups within a SYCL kernel can be scheduled actively at the same time"; new rule maps `this_grid().sync()` to `group_barrier(root_group)` + `use_root_sync` (`AnalysisInfo.cpp:6500-6509`) | [V*] |
| sycl-tla stream-K GEMM, split-KV FMHA (BMG examples) | yes: spin waits | `include/cutlass/barrier.h:96-168`; `xe_tile_scheduler_streamk.hpp:377-393`; `xe_fmha_fwd_kernel.hpp:784-800` | acquire-load spin, device scope | **plain**, grid = Xe-core count (`kernel_hardware_info.h:62-71`) | `can_implement` requires batch*heads <= sm_count | [V*] |
| xetla (archived 2024-12) | stream-K fix-up spin | `stream_k_op_xe.hpp:172` | cmpxchg spin | plain, `avail_xecores` | n/a | [V*] |
| torch-xpu-ops | no grid barrier; last-block pattern; cross-GPU signal pad with 1 work-group | `Reduce.h:911-925`; `src/xccl/Signal.hpp:14, 31, 53-60, 104` | L1-uncached `lsc_load.ugm.uc.ca` | n/a | "a default load in a spin loop keeps hitting the stale L1 copy"; "no timeout check: XPU has no device-side global timer yet" | [V quotes; rest V*] |
| Intel SHMEM | only cross-GPU team sync, 1 thread per PE | `src/collectives/sync_impl.h:29-62` | seq_cst system | n/a | n/a | [V*] |
| Kokkos, Ginkgo, RAJA SYCL | none (last-team counters only) | Kokkos `Kokkos_SYCL_ParallelScan_Range.hpp:~228`; Ginkgo `cooperative_groups.dp.hpp:415-451` (`grid_group` has no `sync()`) | n/a | n/a | n/a | [V*] |
| llama.cpp SYCL (`10152bc`) | none; "the previous launch is the barrier" | `ggml-sycl/topk-radix.cpp:193-198, 273, 322-332` | n/a | SYCL graphs opt-in (`GGML_SYCL_ENABLE_GRAPH`, default 0) | PR #28725 (merged 2026-10-23): graph gain "on B70 it is negligible" | [V*] |
| vllm-xpu-kernels main, sgl-kernel-xpu, IPEX 2.8.10 | persistent work-stealing counters only, no waits; MoE shared-fused decode uses "six kernels" with boundaries as barriers | `grouped_gemm_xe2.hpp:220`; `moe_shared_fused_decode_xe2.cpp:1-23`; `GroupGemmW4A16Xe20.cpp:260` | n/a | grid = slices x subslices | vllm-xpu-kernels #457 (closed): counter reset without a device barrier gave duplicated tiles and `DEVICE_LOST` | [V*] |
| Papers on megakernels or persistent kernels for Intel GPUs | none found | n/a | n/a | n/a | n/a | [S negative] |

Reading:
- **Cooperative launch:** oneDPL onesweep, oneCCL ring kernels, and SYCLomatic's new translation.
- **Plain launch with the grid sized to the core count, just assuming residency:** PR 685 (on a B70), sycl-tla, oneCCL small kernels, the deprecated dpct barrier.
- Nobody publishes a residency guarantee beyond the Level Zero cooperative text.

## 5. Kernel-boundary cost for the fallback, and what it does to the estimate

Measured numbers:

| What | Value | Hardware, stack | Mark |
|---|---|---|---|
| Dependent tiny kernels (1 work-item) in a regular in-order command list, replayed, completion included | N=4: 7.35 us; N=10: 9.87 us; N=32: 20.43 us, so **0.42-0.47 us per extra kernel** [D: (20.43-7.35)/28, (9.87-7.35)/6]. Native graph extension: 8.53 / 11.11 / 21.83 us | Arc B580 (BMG-G21), compute-runtime 26.40.40196.8, Linux 6.17, Core Ultra 7 265; intel/llvm benchmark CI run `Baseline_BMG_L0`, 2026-11-17, `graph_api_benchmark_l0 --test=SubmitGraph --InOrderQueue=1 --MeasureCompletionTime=1 --KernelExecutionTime=1 --EmulateGraphs=1/0` | [V JSON local copy of the CI data.json; kernel `zeKernelSetGroupSize(1,1,1)` V `submit_graph_l0.cpp:74-86`] |
| Host cost of one replay | 1.03-1.11 us, flat in N | same | [V] |
| Immediate in-order command list, per kernel (host submit) | about 1.1-1.2 us (10 kernels: 11.59 us, 10.67 without events); SYCL in-order queue 1.3-1.4 us (L0 v2), 1.7 us (L0 v1) | B580, same run | [V*] |
| torch XPU graph, 200 small dependent ops | eager 4.3 us per launch, replay 0.9 us per op including execution | **B70**, torch 2.13+xpu; vllm-xpu-kernels issue #509, 2026-08-06 | [V*] |
| Base vLLM decoder kernels under XPU graphs | gaps "about 0" | **2x B70**, PR 685 comment | [V, "that person wrote it"] |
| In-kernel grid hand-off (PR 685) | 10.5-22.9 us per layer for 4-5 hand-offs, about 2.1-5.7 us each [D], including the critical path waiting on loads and imbalance | **2x B70**, oneAPI 2026.1, torch 2.15 | [V comment] |
| Reference for comparison | 1,100 barriers cost about 0.8 ms per pass, about 0.73 us each | 9060 XT, HIP | [L research/48 §8] |

Estimate [D]:
- A dependent full-device kernel boundary on a B70 in a replayed command list costs 0.5 us (tiny-kernel slope) plus ramp and tail for 32 x 512 work-groups. That is about 0.5-3 us; PR 685 shows 2-5 us of "launch ramp, last work-group end" for a whole-layer kernel, part of which is imbalance that an in-kernel barrier also pays.
- 1,100 boundaries: **0.55-3.3 ms per pass**. A host submit per kernel (no graph) adds 1.1-1.4 us per kernel on the CPU, hidden only if the CPU runs ahead; with replay the host cost is about 1 us per pass.
- In-kernel barrier on BMG: unmeasured as a pure barrier; PR 685's hand-off of 2-6 us is an upper bound that includes useful waiting. 1,100 x (1-6 us) = **1.1-6.6 ms per pass**.
- So on Intel the megakernel's gain comes from fusion and overlap (weights of the next stage already in flight, no intermediate tensors), not from cheaper synchronization. PR 685 says the same: "The gain here is the glue kernels and intermediate tensors, not launch gaps" [V].

Revised decode estimate (method of research/48 §8, middle row 395 GB/s, GPU part 12.9 ms, 2.5 tokens per pass) [D]:
- Replace the reference's 0.8 ms of barriers with 1,100 t.
- t = 0.5 us: -0.25 ms, unchanged at **126 / 109 tok/s** (upper / lower).
- t = 2 us: +1.4 ms, giving 2.5 / ((12.9+1.4)/0.65 ms) = **114** and 2.5 / (12.9+1.4+10 ms) = **103 tok/s**.
- t = 3 us: +2.5 ms, giving 106 / 99 tok/s.
- The research/48 rows "dispatch per step 5-10 us → 80-120 tok/s" assumed 5-10 us per dispatch. With command-list replay that is too pessimistic: expect the fallback 5-15 % below the in-kernel case, not 25-50 %.
- The in-kernel case itself is unproven on Intel: at 3-6 us per barrier it would be *slower* than the fallback.

## 6. Minimal B70 test

Rules: AGENTS.md rule 2. A measurement file under `research/measurements/` with the plan written first; `tools/hw_report.py`
and `tools/quiet_check.py` before and after; nothing built while measuring.

Hardware conditions:
- B70 headless, no display cable, no compositor on that card (check `/sys/class/drm/card*/device` and that the B70 card has no DRM master).
- The AMD card keeps the desktop. The user is warned beforehand.

Record:
- Kernel version, compute-runtime, IGC and oneAPI versions.
- `cat /sys/class/drm/cardN/device/tile0/gt0/engines/ccs/{preempt_timeout_us,timeslice_duration_us,job_timeout_ms}` (world-readable).
- `dmesg` baseline.
- Optional with root, read-only: `intel_reg read 0x9114` (IGT), bit 20 = mid-thread preemption fused off.

Decision questions (what each result decides):
1. Does a 32-group cooperative grid barrier work? Yes → in-kernel megakernel path open. No → command-list replay fallback.
2. Pure cost per barrier: root barrier versus hand-built versus a kernel boundary in a replayed command list. The cheapest wins per call site; at 1-3 us both paths are about equal, so choose by fusion benefit.
3. Does a spinning kernel survive another context on the card? Yes → safe for request-length runs. No (reset) → keep kernels short (one per layer, as PR 685 does) and keep the card exclusive.

Pseudo code (SYCL; one throwaway process; every spin bounded):

```
// T0 queries (no kernel run)
for wg in {256, 512, 1024} x grf in {128, 256} x slm in {0, 32K, 64K, 128K}:
    record max_num_work_groups(q, wg, slm)            // runtime query (cooperative)
    record own_estimate = 32 * (threads_per_dss(grf) / ceil(wg/16)) clipped by SLM granularity
    record zeKernelSuggestMaxCooperativeGroupCount directly (Level Zero interop)
record device: max_compute_units, local_mem_size, work_group_progress_capabilities<root_group>
record zebin flag disable_mid_thread_preemption of the test kernel (must be false)

// T1 root barrier: correctness + cost
kernel K1 [use_root_sync, grf 256], G groups x 512, K = 10000 iterations:
    for i in 0..K:
        if leader: data[g] = i                        // device USM
        group_barrier(root_group)
        if leader: err += (data[(g+1)%G] != i)        // everyone must see round i
        if (i % 1000 == 0) record sub_group clock delta (sycl_ext_oneapi_clock)
    G in {16, 32}; 3 runs each with event profiling -> us per barrier = kernel_time / K

// T2 hand-built epoch barrier: same checks; variants of the spin load
kernel K2 [work_group_progress<concurrent, root_group>], G = 32:
    arrive: group_barrier(wg); leader: a = cnt.fetch_add(1, acq_rel, device) + 1
            if a == G: cnt.store(0); epoch.fetch_add(1, release)
            else spin until epoch != e or spins > CAP:      // CAP -> err_timeout++, break
                 variant A: epoch.load(acquire)
                 variant B: epoch.load(relaxed), acquire fence every 256 spins
                 variant C: annotated_ptr read_hint<uncached, L1> load
    group_barrier(wg)
    also: same kernel with a PLAIN launch (no property)  -> does residency differ?

// T3 negative control (proves the timeout path, cannot hang)
K2 with G = own_estimate + 32 (more than resident) and CAP small:
    expect err_timeout > 0 on the first round, kernel exits, no dmesg reset

// T4 boundary cost (the fallback)
in-order command list (or SYCL graph) of 1100 kernels, each 32 x 512 doing
    the same tiny work as one barrier interval; replay 100 times:
    us per boundary = (t_total - t_work) / 1100; also host time per replay

// T5 duration + preemption (only after T1-T4 are clean)
K2 persistent for 1 s, then 10 s, then 60 s (iteration count from T2 cost),
    a) alone; b) while a second process submits a tiny kernel every 1 ms on the B70
    record: completes? errors? wall-time inflation; dmesg after each run
```

What to measure:
- Group counts from T0 versus the formula in section 3.2.
- Error counts: wrong round, timeouts.
- Microseconds per barrier: median and spread over 3 runs, root versus hand-built versus boundary.
- T5 inflation in percent and the second process's latency.
- dmesg lines.

Expectation, to write into the plan with a tolerance [D]:
- T0 gives 32 groups for 512 x 256 GRF.
- T1 and T2 give 0 errors at G = 32, at 1-5 us per barrier.
- T3 times out cleanly.
- T4 gives 0.5-3 us per boundary.
- T5(a) runs clean; T5(b) runs clean with about 5-15 % inflation if mid-thread preemption is active.

Failure signs:
- `err > 0`: barrier semantics or memory order wrong.
- `err_timeout > 0` at G = 32: not co-resident, or a stuck spin; try variant A or C.
- Kernel wall time far above K x cost: preemption or stalls.
- Level Zero `ZE_RESULT_ERROR_DEVICE_LOST`.
- In dmesg: `Engine reset: engine_class=ccs`, `GT reset`, `wedged`, `Suspend fence ... failed to respond, banning queue`, `Schedule disable failed to respond`.

Safety limits:
- Every device spin has an iteration cap and a sub-group clock deadline.
- T1-T4 kernels stay under 100 ms each (far below the 640 ms preemption timeout, so even a non-preemptible kernel cannot trigger a reset).
- T5 only after T1-T4 are clean, starting with 1 s.
- Host waits are polled with a host deadline of 2x the expected time. Never kill the process while a kernel spins (kill means a 1 us preemption timeout, which resets if not preemptible): let the caps end it.
- No VRAM pressure test in this round (eviction path), and no other job on the card except the one in T5(b).

## 7. Open points (not settled by reading)

- Whether any B70 (BMG-G31) has `CFEG_WMTP_DISABLE` set. Settle with a register read or T5(b).
- Hardware meaning of `ComputeDispatchAllWalkerEnable`, and whether it changes residency or dispatch order [K only].
- Pure barrier latency on BMG: L2 atomic round trip, cost of the flag barrier versus a counter barrier (T1, T2).
- Whether the compute-runtime SLM miscount in the cooperative query is fixed in a release newer than 26.40 (oneDPL says no as of its commit).
- Whether a full-occupancy compute kernel starves render work on the same card (irrelevant if headless).
- Cost of mid-thread context save and restore on the B70 (the 50 us each way is arithmetic, not a measurement).

## 8. Sources

Read by me ([V]):
- Linux master `drivers/gpu/drm/xe`: `xe_guc_submit.c`, `xe_exec_queue.c`, `xe_hw_engine.c`, `xe_hw_engine_class_sysfs.c`, `Kconfig.profile`, `xe_gt.c`, `xe_lrc.c`, `xe_gt_debugfs.c`, `xe_gt_types.h`, `xe_wa.c`, `xe_vm.c`, `xe_vm.h`, `xe_device.c`, `xe_preempt_fence.c`, `xe_pci.c`, `regs/xe_gt_regs.h`, `include/uapi/drm/xe_drm.h`. Commit 6a8789e948c1 message via the GitHub API.
- compute-runtime master:
  - `shared/source/helpers/kernel_helpers.cpp`, `gfx_core_helper_base.inl`, `gfx_core_helper_pvc_and_later.inl`, `gfx_core_helper_xe2_and_later.inl`, `engine_node_helper.cpp`, `constants.h`
  - `shared/source/xe2_hpg_core/hw_info_bmg.cpp`, `os_agnostic_product_helper_xe2_hpg_core.inl`, `bmg/os_agnostic_product_helper_bmg.inl`, `preemption_xe2_hpg_core.cpp`
  - `shared/source/command_stream/preemption.cpp`, `preemption_xe2_and_later.inl`
  - `shared/source/command_container/command_encoder_xe2_hpg_core_and_later.inl`
  - `shared/source/os_interface/linux/xe/ioctl_helper_xe.cpp`, `drm_neo.cpp`, `product_helper.inl`, `product_helper_xe2_and_later.inl`, `os_context.cpp`
  - `level_zero/core/source/cmdlist/cmdlist_hw.inl`, `cmdlist_hw_xehp_and_later.inl`, `kernel/kernel_imp.cpp`, `device/device.cpp`
  - generated `hw_cmds_generated_xe2_hpg_core.inl`
- IGC master: `IGC/BiFModule/Implementation/barrier.cl`, `IBiF_intel_device_barrier.cl`, `OpenCLKernelCodeGen.cpp`, `GenericShaderState.hpp`, `EmitVISAPass.cpp`.
- intel/llvm `sycl`:
  - `sycl/source/detail/{kernel_data.hpp, kernel_impl.hpp, device_impl.cpp, device_impl.hpp, scheduler/commands.cpp}`
  - `sycl/include/sycl/ext/oneapi/experimental/root_group.hpp`
  - `sycl/doc/extensions/experimental/{sycl_ext_oneapi_forward_progress, sycl_ext_intel_cache_controls}.asciidoc`
  - `unified-runtime/source/adapters/level_zero/{v2/kernel.cpp, v2/command_list_manager.cpp, common/helpers/kernel_helpers.cpp}`
  - `sycl/test-e2e/GroupAlgorithm/root_group.cpp`, `.github/workflows/sycl-linux-precommit.yml`
  - issues #13984, #14462, #21838
- Level Zero `include/ze_api.h` (master); Level Zero spec kernel page: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/kernel.html
- oneDPL `kt_utils.h`, `sycl_radix_sort_kernels.h`.
- Lines spot-checked in the sub-agents' clones: oneCCL `examples/sycl/sycl_lsa_allreduce.cpp:229-244`; `allreduce_small_sycl_impl.hpp:289`; torch-xpu-ops `src/xccl/Signal.hpp`.
- vLLM-XPU PR 685 file `mono_layers.hpp` (refs/pull/685/head) and its per-part breakdown comment: https://github.com/vllm-project/vllm-xpu-kernels/pull/685
- intel/llvm benchmark CI data (branch `sycl-benchmark-ci-results`, `data.json`, run `Baseline_BMG_L0` of 2026-11-17); compute-benchmarks `submit_graph_l0.cpp`.

Read by sub-agents ([V*]):
- SYCLomatic `dpct/util.hpp`, `APINamesCooperativeGroups.inc`, `AnalysisInfo.cpp`.
- ishmem `sync_impl.h`.
- oneCCL `sycl_barrier.hpp`, ring kernels, `cmd_persistent_kernel.cpp`.
- torch-xpu-ops `Reduce.h`.
- Kokkos, Ginkgo, RAJA SYCL backends.
- llama.cpp `ggml-sycl` and PRs #28725, #30156, issue #32322.
- sycl-tla stream-K and FMHA.
- xetla; vllm-xpu-kernels main, #457, #509, #524; sgl-kernel-xpu; IPEX 2.8.10.
- Intel immediate-command-list guide; SYCL graph article.

Search snippets only ([S]):
- intel-xe list, "drm/xe/xe2: synchronise CS_CHICKEN1 with WMTP support" (2024-01): https://lists.freedesktop.org/archives/intel-xe/2024-January/022354.html
- "Expose wmtp status as engine capability" (2024-08; page returned 403): https://lists.freedesktop.org/archives/intel-xe/2024-August/045016.html
- No paper on Intel megakernels found.
