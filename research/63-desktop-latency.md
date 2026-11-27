# Why a desktop submission waits 14 to 15 ms while Tona decodes, and what could shorten it (#168)

Research only (CPU, 2026-11-18); no GPU was used. Finding that started it: in level 0 of the compositor survival test
([measurements/2026-11-18-vram-level0.md](measurements/2026-11-18-vram-level0.md)) a headless Vulkan canary (one 64 MiB
fill submitted every 100 ms) had **p50 14 to 15 ms, p99 about 54 ms, max 58 to 69 ms** in every phase and arm while Tona
decoded, against **p50 about 1 ms, max 1.4 ms** on the idle card (smoke test). A 60 Hz frame budget is 16.7 ms, so a
compositor frame that waits like the canary would miss its deadline about half the time. Tona decodes at 80 tok/s with
3.45 tokens per pass, so a pass takes about 43 ms (`Request: ... 400 tokens in 5.0 s = 80.5 tok/s, 3.45 tokens per
pass` in the level-0 server logs).

## 1. What is already known on this machine

| Fact | Source |
|---|---|
| The megakernel occupies all 32 CUs for a whole pass and waits on the CPU inside the kernel; ordinary kernels are used for prefill | `engine/gpu/mega_op_wait.h`, `research/23-waiting-in-kernel.md` |
| After every kernel of 1.0 ms or longer there is a stall of about 280 us; kernels below 1.0 ms have none. The threshold equals the MES gang quantum of 1 ms | `measurements/2026-11-16-prefill-idle-gaps.md` (trace and "Cause research") |
| 28 of 32 CUs (`HSA_CU_MASK=0:0-27`) cost 5 to 11 % decode; 30 CUs aborted at step 0. The mouse was judged "smoother" with 32 blocks, so the mask did not help the desktop in that test (one run, subjective) | `measurements/2026-10-23-cu-mask-desktop.md` |
| `TONA_FREE_CUS=n` sets `HSA_CU_MASK=0:0-<cus-n-1>` before the HIP runtime starts and launches fewer blocks (multiple of 4); 16 CUs measured in #164 | `engine/gpu/mega_init.cpp`, `gpu/cu_mask.h`, `measurements/2026-11-17-cu-scaling.md` |
| `TONA_MEGA_YIELD_MS` (default 100): when block 0 waits longer than this for the CPU at a router, all blocks end and the CPU restarts the kernel; it is a safety valve for waits, not a per-pass cut | `engine/gpu/mega_op_wait.h`, `mega_launch.cpp` (`resume_if_yielded`), `docs/use/environment.md` |
| All megakernel launches and copies use the null stream (stream 0); the other streams (prefetch, two idle "spacer" streams, prefill copy) are `hipStreamNonBlocking` without priority | `engine/gpu/mega_launch.cpp`, `mega_build_bufs.cpp` (`setup_prefetch`), `prefill_run.cpp` |
| Module parameters here: `sched_policy=0` (HWS), `uni_mes=1`, `cwsr_enable=1`, `mcbp=-1`, `queue_preemption_timeout_ms=9000`; kernel 7.2.15 at the time of that note | `measurements/2026-11-16-prefill-idle-gaps.md` |

## 2. How the hardware and driver decide who runs (what the sources say)

1. **MES time slices.** The MES firmware time-slices user queues when there are more queues than hardware queue slots
   ("the MES firmware will preempt other user queues to make sure each queues get a time slice"; kernel queues are
   "statically mapped to HQDs and never preempted"). The kernel's user-queue code sets a process quantum of 10 ms and a
   gang quantum of 1 ms as defaults; KFD does the same for compute queues (`KFD_MES_PROCESS_QUANTUM` 10 ms,
   `KFD_MES_GANG_QUANTUM` 1 ms; research in `measurements/2026-11-16-prefill-idle-gaps.md`).
   Sources: [MES documentation](https://kernel.org/doc/html/v7.2-rc1/_sources/gpu/amdgpu/gc/mes.rst.txt),
   [mes_userqueue.c](https://gbmc.googlesource.com/linux/+/0a70cac7896712a08e3cd22c16f44be976d40dbf/drivers/gpu/drm/amd/amdgpu/mes_userqueue.c).
   Whether a quantum expires only when another queue is waiting for the same hardware (the "oversubscription" case) was
   not found in the text read; the 1 ms threshold of our trace is consistent with a time slice, which is not proof.
2. **Priority has two layers and they do not meet for our case.**
   - KFD (HIP/ROCr) queues: a priority of 0 to 15 per queue, "relative within a process and only a hint" ("HW scheduling
     may override the priority"), translated to pipe/queue priority classes; MES gets it as `inprocess_gang_priority`
     while the **global priority level is fixed to NORMAL** (`add_queue_mes` in `kfd_device_queue_manager.c`, read in
     `measurements/2026-11-16-prefill-idle-gaps.md`). A priority stream in Tona would therefore order Tona's own queues
     against each other, not Tona against the compositor.
     Sources: [kfd_priv.h](https://codebrowser.dev/linux/linux/drivers/gpu/drm/amd/amdkfd/kfd_priv.h.html),
     [KFD priority commit (Oak Zeng)](https://lab.nexedi.cn/kirr/linux/-/commit/0ccbc7cdf54d855f48a023464c207e9d5ca2b6f1).
   - Graphics/compute contexts through amdgpu (what a compositor uses): `AMDGPU_CTX_PRIORITY_*`. **Only HIGH and
     VERY_HIGH reach the hardware pipe priority** (`amdgpu_ctx_prio_to_gfx_pipe_prio`); LOW and VERY_LOW only change the
     software DRM scheduler order, NORMAL and below are allowed for everyone; above NORMAL needs `CAP_SYS_NICE` or being
     the DRM master. Source: [amdgpu_ctx.c](https://codebrowser.dev/linux/linux/drivers/gpu/drm/amd/amdgpu/amdgpu_ctx.c.html)
     (`amdgpu_ctx_priority_permit`, lines 100 to 114).
   - MES knows five levels (LOW 0, NORMAL 1, MEDIUM 2, HIGH 3, REALTIME 4) and a grace period per band when it
     preempts other bands. Source: [amdgpu_mes.h](https://sources.debian.org/src/linux/6.16.3-1/drivers/gpu/drm/amd/amdgpu/amdgpu_mes.h).
   - The older (2016 to 2021) design: a high-priority compute queue needs **wave limiting on the graphics pipe**
     (`SPI_WCL_PIPE_PERCENT_GFX`) to work, and only one such queue is enabled.
     Sources: [RFC 2016](https://lists.freedesktop.org/archives/amd-gfx/2016-December/004070.html),
     [one high-prio compute queue, 2021](https://lists.freedesktop.org/archives/amd-gfx/2021-February/059008.html).
     Whether gfx12 uses this path under MES was not found.
3. **CU occupancy is the other limiter, independent of queue priority.** A persistent kernel with one block per CU keeps
   its waves resident for the whole pass; a second program's waves can only start where wave slots, VGPRs and LDS are
   left. This is the case the CU mask addresses: with `HSA_CU_MASK` the compute waves are confined to the lower CUs and
   the rest stay free for graphics. Source for the mechanism: the HIP API `hipExtStreamCreateWithCUMask` and ROCr's
   `HSA_CU_MASK` ([HIP stream management](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream.html));
   no source measuring compositor latency under a CU mask was found.
4. **HIP stream priority.** `hipStreamCreateWithPriority` exists with the range [greatest, least]; one documentation
   version says the call is "under development on AMD GPUs" and returns success, a newer one describes it normally
   ([HIP stream management](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream.html)). Which holds
   for ROCm 10.1 was not checked. In any case see 2: the effect would stay inside the process.

## 3. What compositors and other engines do

- **Compositors raise their own priority, the other side of the problem.** KWin requests a high-priority EGL context
  (`EGL_IMG_context_priority`) since 2018 and discusses `setcap CAP_SYS_NICE`; Mutter and Weston request one and fall back
  to the default if refused. On amdgpu this reaches the hardware pipe priority only for HIGH, which the DRM master is
  allowed. Sources: [KWin commit](https://commits.kde.org/kwin/2c26215d7ebc6493f52a502c92205383422215bc),
  [plasma-devel on setcap](https://mail.kde.org/pipermail/plasma-devel/2023-September/123134.html),
  [Weston patch](https://lists.freedesktop.org/archives/wayland-devel/2018-March/037272.html),
  [Mutter MR 2973](https://gitlab.gnome.org/GNOME/mutter/-/merge_requests/2973).
  So the real compositor probably already has HIGH pipe priority and our canary (NORMAL, no privilege) is a **worse**
  case than the compositor for the priority part. Whether the compositor on this machine gets it was not checked.
- **Other engines:** no engine found that limits CUs or lowers its priority for a desktop on the same GPU. Reports are of
  freezes and of GPU hangs under llama.cpp/Ollama on shared consumer GPUs, answered by moving the display to another GPU
  or by driver and environment workarounds (`GPU_MAX_HW_QUEUES=1` in a 2023 thread, unverified), not by a latency design.
  Sources: [CachyOS thread](https://discuss.cachyos.org/t/issue-with-amd-7900-xtx-gpu-and-cachyos-latest-ollama-and-freezing/17857),
  [Fedora flip_done report (RX 9070 XT)](https://discussion.fedoraproject.org/t/drm-flip-done-timed-out-recurring-freeze-artifacting-on-rx-9070-xt-gfx1201-500hz-hdr-over-displayport/198858).
- **Display on the iGPU** (card0 has the monitors' connectors too) is the structural answer and is a user decision
  (research/62 placement 2); it removes the latency problem entirely but needs a session change.

## 4. Hypotheses (each with what would refute it)

| # | Hypothesis | Predicts | Refuted by |
|---|---|---|---|
| H1 | The canary waits for the megakernel's pass or for a gap between passes: the megakernel holds all waves of all CUs, so graphics waves get in only when blocks leave. | latency spread of the order of one pass (about 43 ms): p50 about half of it, p99 near it (observed 15 / 54 ms); idle Tona about 1 ms; **freeing CUs removes it** | a CU mask that frees CUs without changing p50 |
| H2 | The 1 ms MES gang quantum bounds the wait. | latency capped at a few ms | the observed 15 to 69 ms; quanta apply to time-slicing, not to resident waves |
| H3 | Queue priority from the Tona side moves the latency. | a priority arm changes p50 | no change in a test; **expected no change** (section 2.2) |
| H4 | `TONA_MEGA_YIELD_MS` shortens it. | yield 10 ms cuts latency | no change: the yield fires only when block 0 waits for the CPU longer than the limit, not at every pass |
| H5 | During prefill (kernels of about 1 ms, gaps between them) the wait is short. | p50 in the low ms, high frame hit rate | prefill as bad as decode |

## 5. What the sweep can and cannot test (feeds the plan in measurements/2026-11-18-desktop-latency.md)

- **Testable now without engine change:** default, CU mask 4 and 8 (`TONA_FREE_CUS`), `TONA_MEGA_YIELD_MS` 10 and 2.
- **Tona-side queue priority: not testable without an engine change.** The megakernel and its copies run on the null
  stream; a priority stream would mean moving all launches, memsets and syncs off stream 0, in the code path where a
  blocking call while the kernel waits can reset the GPU (AGENTS.md "GPU safety"). By 2.2 it would not even reach the
  compositor's queue. It is not built.
- **Canary-side priority:** HIGH (what a compositor gets) is refused to an unprivileged client; the sweep records whether
  the driver grants it (one second probe). LOW is only a software scheduler order for graphics contexts (2.2), so it is not
  an arm.
- **Not testable:** `mes` quantum values (kernel parameters; the module has none for them found), process priority via
  `CAP_SYS_NICE` (needs root).

## 6. Open points

- Section 2.1: does the quantum expire without a competing queue? Not found.
- Section 2.4: is `hipStreamCreateWithPriority` a no-op in ROCm 10.1? Not checked.
- Which priority the desktop compositor on this machine really has. Not checked.

## 7. Measured outcome (2026-11-18, sweep in [measurements/2026-11-18-desktop-latency.md](measurements/2026-11-18-desktop-latency.md))

H1 supported (4 free CUs: median wait 19 ms -> 0.27 ms, hit rate of the 16.7 ms budget 41 % -> 98 % in decode, decode -4.4 %, prefill -6.0 %; 8 free CUs: 100 %, max 3 ms, decode
-13.3 %, prefill -13.6 %). H2 not supported (waits of 15 to 70 ms). H4 refuted (`TONA_MEGA_YIELD_MS` 10 and 2: no effect on the canary or on Tona). H3 not tested (needs the megakernel
off the null stream; by section 2.2 it would not reach the compositor). The driver refuses HIGH priority to an unprivileged client (`VK_ERROR_NOT_PERMITTED_KHR`), as section 2.2 predicted.
Section 6 stays open (quantum without a competing queue, stream priority in ROCm 10.1, the compositor's real priority).
