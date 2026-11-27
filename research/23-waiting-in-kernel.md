# Long waits in the megakernel (low RAM): abort, pause or preload?

Agent research of 2026-10-27, not checked by ourselves. [V] read (source code, paper), [S] search hits, [W] knowledge
of the agent. Complements 09-gpu-reset.md and 12-gpu-hangs.md.

Occasion: With `IE_RAM_GB=20` the CPU reads missing experts from the NVMe via page faults; a pass then takes over
1.5 s, the barrier aborts (HOST_WAIT_MS = 1000), the request fails with HTTP 500
(measurements/2026-10-27-ram-limit).

## Does `lockup_timeout` apply to our kernel?

According to the source code no (derived, no direct statement found):

- The timeout (default 2 s) hangs on the drm_sched per ring (`amdgpu_device_init_schedulers`, torvalds/linux master)
  [V]: it hits jobs via the CS ioctl, i.e. compositor, terminal, Vulkan. Our kernel runs in a KFD user queue,
  which the MES serves directly via the doorbell, without a drm_sched job [W, fits mes.rst v7.2-rc1: kernel queues fixed,
  user queues time-shared when oversubscribed].
- KFD has limits of its own: `queue_preemption_timeout_ms` (9000) when removing a queue, then `kfd_hws_hang`
  [V, kfd_device_queue_manager.c]; MES fence 2100 ms (`mes_v12_0.c`), then "MES failed to respond" → reset of the
  queues → `kfd_hws_hang` [V]. This is the start of the chain from 09.
- Consequence: `ring gfx_0.0.0 timeout` means a job of the desktop did not finish in 2 s, not that our kernel ran too
  long. Why it hangs (wave slots, MES busy with preemption, an own problem of the 9060 XT) is open.
- Our GPU-side 1 s limit measures wall clock: during a KFD preemption the queue stands still, the clock runs. It then
  measures preemption time, not CPU wait time, and can produce false alarms (the agent's conclusion).
- For "a long spin alone triggers the timeout" there is neither evidence nor counter-evidence; only passes of
  20–90 ms are measured (and the 6 s stall of 2026-10-24 that hit a terminal window, memory).

## How others do it

- Mirage Persistent Kernel (arXiv 2512.22219, OSDI '26) [V]: everything on the device, scheduler warps poll queues in
  device memory; no host I/O wait, no interruption.
- HazyResearch low-latency Llama (blog 2025-05-27) [V]: counters in global memory; no host waits.
- Cooperative Kernels (Sorensen et al., arXiv 1707.01989) [V]: workgroups end voluntarily at fixed points
  (`offer_kill`), the barrier shrinks with them; the kernel must be re-enterable there. Prototype Intel only. Closest
  known pattern for "kernel pauses voluntarily".
- Everywhere else (Windows TDR 2 s, NVIDIA under Linux with display): split the work into shorter launches [S].
- For AMD no statement on long-running kernels next to the desktop (as in 12).

## Restart costs

Own measurement (measurements/2026-10-22-micro): 2.1 µs per kernel, start+sync 17.7 µs. Not measurable against hundreds of ms of NVMe. In addition
warm-up, registers/LDS gone, caches cold (unmeasured).

## Host side

- `MADV_WILLNEED` reads ahead only a little for large ranges; `MADV_POPULATE_READ` fills synchronously [S, lkml 2401.3].
- Page faults on the compute path slow things down; reports of up to ~2× from an io_uring read path instead of mmap [S, Hypura,
  llama.cpp #19825], anecdotal.

## Recommendation of the agent

- **A (preferred): wait budget, then voluntary pause.** The kernel waits up to a budget (50–200 ms), then
  all blocks end at the same router barrier; the host restarts from step L as soon as the experts are there. Cost: µs per restart; effort: clean joint termination, resetting epochs and barriers. Makes the
  abort after 1000 ms unnecessary, time is measured on the CPU.
- **B (only after a test): wait long, CPU supervises with `evicted_ms`.** Beforehand a micro test (spin 5/10/30 s next to
  vkcube); risk: crash of the desktop session.
- In addition: read missing experts explicitly (io_uring/pread into a pinned buffer) and request them ahead of time
  instead of page faults in the compute thread.
