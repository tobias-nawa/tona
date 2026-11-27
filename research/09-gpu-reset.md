# GPU reset and KFD eviction (gfx1200, megakernel)

Agent research, as of 2026-10-23. None of it measured by us, nothing reproduced on our machine.
Sources are mostly user reports and mailing-list patches. Statements about causes are **hypotheses** unless
noted otherwise. Search results were partly read only from summaries, not from the full texts.

Question: why does the engine abort with `ring gfx_0.0.0 timeout` / GPU reset, and what helps?

## Findings (own observations, carried over unchanged)

1. Barriers occasionally abort after 1 s, all 32 blocks at the same step: the whole queue stood still.
   `evicted_ms` ~700 ms in the first 15 s, keeps growing.
2. Two hard crashes: `ring gfx_0.0.0 timeout` (a terminal window) → ring reset fails → `GPU reset begin` →
   `Failed to evict queue 1` → `remove_all_kfd_queues_mes: Failed to remove queue 0` → MODE1 reset → VRAM lost.
3. With HSA_CU_MASK (28 CUs) one 45 s test without a crash, another run with the mask **nevertheless**
   `ring gfx_0.0.0 timeout` (a terminal window, ring reset only, "device wedged, but no recovery needed"), right at startup,
   while vkcube was running. CU reservation is therefore **not** the solution.

## How KFD eviction works (mechanism, documented)

- Each KFD process has an eviction fence. If TTM wants to free VRAM and hits a BO of the process,
  the fence's `enable_signaling` triggers a KFD eviction: **all** user queues of the process are preempted
  (MES/HWS: REMOVE/SUSPEND), only then is the fence signaled and the BO moved. A worker restores the queues
  later. Source: commit "drm/amdgpu: Add KFD eviction fence" (F. Kuehling),
  https://zircon-guest.googlesource.com/third_party/linux/+/d8d019ccffb838bb0dd98e583b5c25ccc0bc6ece
- Between KFD processes there is a per-GPU limit against VRAM overcommit so they do not evict each other.
  **The desktop (compositor and its clients) is not a KFD process** and is therefore not subject to this limit: it
  competes with us through normal TTM. (Derived from the commit description.)
- Preempting a queue's content while a wave is running means: CWSR saves the wave state (VGPRs, LDS) into the
  context-save area. With 32 blocks of 512 threads with a lot of LDS/VGPR this is a large amount of data to save
  (magnitude not documented, **estimate**: a few MB to tens of MB per preemption, in the ms range at VRAM bandwidth).
- Other eviction triggers: userptr invalidation by MMU notifier (see below, #12528), suspend/resume,
  GPUVM fault (`no_queue_eviction_on_vm_fault`), contiguous allocations (patch by P. Yang 2024, marked in the hits
  as **not merged**: https://lists.freedesktop.org/archives/amd-gfx/2024-April/106935.html).

## Known cases and patches

| Source | Content | Status |
|---|---|---|
| [ROCm/TheRock #9279](https://github.com/ROCm/TheRock/issues/9279) | gfx1201 (R9700), ROCm 7.14, kernel 7.0: llama.cpp generation hangs, `MES(0) failed to respond to msg=REMOVE_QUEUE`, MODE1, PSP resume -62. MES FW 0x8b, MEC 0xcda. **No effect:** `cwsr_enable=0`, `mes_log_enable=1`, newer nightlies. `uni_mes=0` lowered the rate to ~1 in 5, no fix. Vulkan runs stable. AMD could not reproduce | open, triage (as of late Aug./Oct.) |
| [ROCm/TheRock #9299](https://github.com/ROCm/TheRock/issues/9299) | gfx1201 (2x R9700), kernel 7.0.14-x: host freezes, `SMU: No response msg_reg: 12` → MES unrecoverable; `gpu_recovery` does not take effect because the host stalls. The reporter suspects clock change + a lot of SDMA/PCIe traffic (MoE expert offload, i.e. our pattern). Dense model stable | open, triage, no technical AMD answer |
| [ROCm/TheRock #9217](https://github.com/ROCm/TheRock/issues/9217), [#9253](https://github.com/ROCm/TheRock/issues/9253), [#9271](https://github.com/ROCm/TheRock/issues/9271) | same chain (MES does not answer REMOVE_QUEUE/SUSPEND, GPU reset fails, Wayland session crashes) on gfx1103, gfx1151 | only titles/hits seen, status not checked |
| [ROCm/TheRock #5793](https://github.com/ROCm/TheRock/issues/5793) | gfx1102, llama-server: `cp queue preemption failed`, `Failed to evict process queues`, MODE1 | status not checked |
| [CarouselAether/rocm_exl3 #2](https://github.com/CarouselAether/rocm_exl3/issues/2) | gfx1201: pp8192 prefill hangs in ~50% of runs, MES `REMOVE_QUEUE` not answered, with `HIP_LAUNCH_BLOCKING=1` 2/2 stable. The reporter suspects **cooperative launches** (GWS queue) against MES. Exit crash: ROCr commit e5a0f920d and CLR commit 71830204b (GWS queue release, xferQueue race), in `develop` and ROCm 10.0 | hypothesis, single report; the commits are confirmed only on gfx1100 |
| [ROCm/rocm-systems #12528](https://github.com/ROCm/rocm-systems/issues/12528) | **gfx1200 (RX 9060 XT), kernel 7.1.11, also ROCm 10.2:** clr pins pageable memory (`hsa_amd_memory_lock` → KFD userptr BO); on reclaim the MMU notifier invalidates, KFD evicts **all** queues until restore. Measured `evicted_ms`: ~1.0 s without pressure, up to 146 s of 195 s under RAM pressure. Remedies see below | open, reproduced by schung-amd; PR #12670 (staging instead of pinning) open |
| [amd-gfx, drm/amdkfd: mark queues as reset after a full GPU reset (v2, 2026-08-28)](https://ratatoskr.run/amd-gfx/2026/08/17438017/t) | After a full reset MES no longer knows the queues; teardown sends REMOVE_QUEUE, MES refuses, a second reset follows. Acked-by A. Deucher. Observed on Navi32 SR-IOV | in review/ack, merge status and kernel version unknown. Presumably explains the follow-up timeouts of other processes **after** a reset, not the cause |
| [LKML 2411.3 "gfx1103 queue evict/restore crash"](https://lkml.iu.edu/hypermail/linux/kernel/2411.3/04084.html) | `Failed to evict process queues` → reset on APU; C. Koenig rejected the workaround (skip eviction), the error chain is a real bug | discussion, no fix seen |
| [Fedora Discussion "RX 9060XT persistent ring timeout"](https://discussion.fedoraproject.org/t/amd-rx-9060xt-persistent-ring-timeout-gpu-reset-on-fedora-43/180131) | gfx1200, kernel 6.18: `ring gfx_0.0.0 timeout` also **without** ROCm, Wayland, multi-monitor. Mentioned: `amdgpu.sg_display=0`, `lockup_timeout=10000`. `mes=0` made it worse. SMU interface mismatch (driver 0x2e, FW 0x32) in the log | no confirmed fix. **Important:** `ring gfx timeout` on the 9060 XT also occurs without our program |
| Fedora thread, one user | `amdgpu.cwsr_enable=0` against "Memory access fault" on gfx1200 ([X post](https://x.com/hogehoge61/status/1998041028084076858)) | single report, different error type (page fault) |
| Launchpad [#2117463](https://bugs.launchpad.net/bugs/2117463) | gfx11.5.1: fixed by updated MES firmware | gfx11.5.1, not gfx12 |
| [ROCm #5724](https://github.com/ROCm/ROCm/issues/5724) | Strix Halo: MES FW 0x83 causes hang/memory access fault, maintainer: independent of CWSR | kernel fix announced, status not checked |

No hit found for: a gfx12-specific MES/CWSR fix in a named linux-firmware version,
and for a drm/amd GitLab issue that describes "ROCm compute on the display GPU → ring gfx timeout" on gfx1200.
That does not mean none exists (the search was superficial, GitLab search cannot be queried directly).

## Assessment of the hypothesis "VRAM pressure → TTM eviction → queue preemption → compositor waits"

The mechanism is **plausible and built into the kernel this way** (above). The evidence in detail:

- For: `evicted_ms` > 0 on our side proves that KFD actually evicted our queues. 14.5 GB + desktop
  (0.5–1.4 GB) + compositor buffers lie very close to 16,304 MiB. New windows/frames/vkcube reserve VRAM,
  TTM evicts our BOs (or moves desktop BOs to GTT, depending on location).
- Against/gap: ~700 ms of eviction in the first 15 s is typical **even in cases without VRAM pressure**: #12528 measures
  ~1.0 s without pressure from pinning/restore alone. Our number therefore **does not yet prove VRAM pressure**. Causes that
  also fill `evicted_ms`: userptr invalidation of our pageable/pinned host buffers (RAM pressure, 64 GB
  shared with several Claude sessions), page migration/compaction (THP), `hipHostRegister`/`hipMemcpy` from
  pageable memory. **To distinguish:** `debug_evictions=1` (see remedies) shows the cause of each eviction in dmesg.
- An eviction alone does not necessarily explain the ring timeout: the gfx ring is the compositor's; KFD preemption
  runs through MES. Possible coupling (**undocumented**): the compositor waits for a BO move/fence that sits behind the
  preemption of our megakernel. Preempting a kernel with 32 resident spinning blocks that wait on
  atomics may take long (CWSR has to save 16k threads); compositor jobs do not run until then
  and exceed `lockup_timeout` for gfx.
- The opposite possibility: the 9060 XT has `ring gfx timeout` even without ROCm (Fedora thread), the cause might be
  (partly) a display/SMU/firmware problem and only occur more often with us because of load (**hypothesis**).

## Persistent kernels, spin waits and CWSR

- No evidence found that spin waits are fundamentally incompatible with MES preemption. CWSR exists precisely to save
  waves mid-run. But: **no evidence that it is robust on gfx12.** The reports above
  (#9279, rocm_exl3) show hangs in MES queue operations under load on gfx1201, stable with Vulkan.
- A kernel that occupies all CUs and never ends can only be displaced through preemption. With all 32
  blocks resident (one block per CU) the compositor gets room **nowhere** unless the hardware scheduler
  preempts. With free CUs (HSA_CU_MASK) gfx waves can run alongside, but our run with the mask showed
  a timeout nevertheless. CU room alone is therefore not the bottleneck. **But** the mask applies only to our
  queue; compositor shaders have their own CUs free in this calculation, the preemption (eviction) is unaffected by it.
- Cooperative launches / GWS: rocm_exl3 suspects a race with MES. We use our own atomics instead of GWS and are
  presumably not affected (**unchecked**).
- How other engines deal with it: **no evidence found** (other engines, llama.cpp HIP, MLC, Mirage MPK,
  ThunderKittens on AMD were not evaluated specifically; no hits in the searches). llama.cpp HIP has the same
  reset reports (#5793, #9279) with short kernels, i.e. **short kernels do not protect** against the MES error chain.

## Module parameters (docs: https://dri.freedesktop.org/docs/drm/gpu/amdgpu/module-parameters.html)

| Parameter | Default per docs | Relevance |
|---|---|---|
| `lockup_timeout` | 2000 ms all queues (the page elsewhere says 10000 gfx / 60000 compute, contradictory: check kernel version); format `[gfx,compute,sdma,video]`, negative = infinite | A higher gfx timeout hides short evictions but does not prevent a real blockage |
| `cwsr_enable` | 1; docs say "non-HWS mode only" (unclear with MES) | `=0` without effect in #9279; one report for the 9060 XT against page faults. Without CWSR no wave preemption: eviction would have to wait for the kernel to end, with an endless kernel probably a timeout (**hypothesis**) |
| `uni_mes` | 0 per docs (on Fedora 44/kernel 7.2 probably on for gfx12, **not checked**) | `=0` lowered the rate in #9279, no fix |
| `mes_log_enable` | 0 | diagnosis |
| `debug_evictions` | false | **Shows the cause of every eviction in dmesg. First diagnostic step** |
| `queue_preemption_timeout_ms` | 9000 | time the HWS waits for preemption. HWS path only, unclear with MES |
| `no_queue_eviction_on_vm_fault` | 0 | debug only |
| `gpu_recovery` | -1 (auto) | in #9299 recovery does not take effect when the host itself stalls |
| `sched_hw_submission` | 2 | no evidence of relevance |
| `mcbp` | -1 | mid-command-buffer preemption (gfx ring), no evidence of relevance |
| `halt_if_hws_hang` | 0 | debug |

A parameter `amdgpu.mes` does not appear in the docs; `mes=0` made the timeouts worse in the Fedora thread.

## Firmware

- Reports name MES 0x8b / MEC 0xcda for gfx1201 (Ubuntu system, TheRock #9279). No fix version found for gfx1200.
  Fedora's linux-firmware 20260810 lists only a blanket "amdgpu: numerous firmware updates", 20260519
  "DMCUB updates" (source: Fedora advisories, summary), nothing on MES/gfx12.
- SMU interface mismatch (driver 0x2e vs. FW 0x32/0x33) is common on the 9060 XT/R9700 and lies in the **VBIOS**
  (#9299): no host fix via linux-firmware.
- Check the loaded version (read only): `sudo cat /sys/kernel/debug/dri/*/amdgpu_firmware_info` (MES, MES_KIQ, MEC,
  SMC, PSP), also `dmesg | grep -i -E 'smu|mes|firmware'` and `rpm -q linux-firmware`. Latest source:
  `git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git`, directory `amdgpu/gc_12_0_0_mes*.bin`
  or `gc_12_0_0_mec*.bin` (gfx1200 = GC 12.0.0 per the naming scheme, **not cross-checked**).

## Remedies, prioritized

### Immediate (application/configuration only)

1. **Increase the VRAM reserve and measure.** Budget from sysfs `mem_info_vram_used` (desktop included) instead of
   `hipMemGetInfo`. Test: engine at e.g. 13.0 GB instead of 14.5 GB and compare `evicted_ms` at startup (several
   runs, with vkcube and without). This checks the working hypothesis directly. Recommendation for the size: **no documented
   number found**; rule of thumb (estimate): desktop 1.4 GB + window gain/browser/vkcube + 1–2 GB buffer.
2. **Make the eviction cause visible:** kernel parameter `amdgpu.debug_evictions=1` (to be set only after approval by
   nothing changed here) and `dmesg -w` at startup. Separates VRAM TTM pressure from userptr invalidation.
3. **Defuse host pinning** (applies to our `hipHostMalloc`/expert buffers, #12528):
   - allocate all large host buffers **once at startup** and hold them with `mlock`-like behavior;
     no pageable `hipMemcpy` sources during operation, no `hipHostRegister` of pageable memory;
   - test `HSA_USERPTR_FOR_PAGED_MEM=0` (on ROCm 10.x additionally `HSA_SVM_FOR_PAGED_MEM=0`) and
     `GPU_PINNED_MIN_XFER_SIZE=100000` as environment variables (effect per #12528, only partial on gfx1200);
   - avoid RAM pressure: other GPU- and RAM-hungry sessions and services off during the test (`vm.compact_unevictable_allowed=0` per
     #12528 helps only partly).
4. **Make the barrier time limit eviction-proof:** read `evicted_ms` per step from `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms`
   and do not treat the 1 s limit as an error while the eviction time grows (prevents follow-up aborts, not the timeout).
5. **Pass granularity:** the megakernel runs 20–90 ms per pass. Between the passes there are preemption points; a
   short idle (e.g. putting the mailbox to sleep when there is no job) prevents the kernel from permanently
   holding resident waves. Effect **not documented**, but cheap to measure.
6. No continuous run next to the desktop without safeguards: until clarified, tests in a session without compositor load
   (or on `ROCR_VISIBLE_DEVICES=0` with the desktop on the iGPU, if a monitor can be connected there, unchecked).

### Experimental (parameters, the maintainer decides)

7. `amdgpu.lockup_timeout=10000,60000` (gfx, compute) against short stalls (Fedora thread names 10000, without proof of success).
8. `amdgpu.cwsr_enable=0` / `amdgpu.uni_mes=0` only as an A/B test; risk: without CWSR, preemption of a continuous kernel
   is less likely. Not effective in #9279.
9. `amdgpu.sg_display=0` (named against ring timeouts on the 9060 XT, anecdotal).

### Needs driver/firmware/upstream

10. Wait for current linux-firmware (gc_12_0_0 MES/MEC) and a kernel with the "mark queues as reset" patch, or check
    whether it is in 7.2.x stable. Follow: TheRock #9279, #9299, rocm-systems #12528 / PR #12670.
11. File an issue of our own with logs (incl. `debug_evictions`, `amdgpu_firmware_info`) at TheRock/drm-amd: so far
    no gfx1200 case with a persistent kernel is documented.

## Open questions

- How much VRAM reserve is needed for `evicted_ms` to stay ≈ 0 (measurement 1).
- Does our eviction come from TTM (VRAM) or userptr invalidation (host pinning)? Measurement 2.
- How long does a CWSR preemption of our megakernel take (32 blocks, 512 threads, LDS)? No measured value known.
- Is `ring gfx timeout` a consequence of the preemption or a 9060 XT problem independent of compute?
  Test: desktop load (vkcube, moving windows) without our engine over a longer time.
- Is `uni_mes` on on this system, and which MES/MEC versions are loaded?
- How do other engines/llama.cpp/Mirage MPK/ThunderKittens avoid the problem? Not found in research.

## Own measurement afterwards (2026-10-23, 18:50–19:19)

Implemented: immediate measure (3), all host buffers for per-pass copies pinned once via `hipHostMalloc`. Result
(measurements/2026-10-23-pinned-buffers): 12 of 12 teacher runs without abort and without gfx timeout, with vkcube
alongside, with 2000 MB and with the default reserve. Before, roughly every second run aborted. `evicted_ms` now rises only
once by ~720 ms at the end of loading, not at all during generation. This fits #12528 (userptr) better than
the VRAM hypothesis. The eviction during loading remains open.
