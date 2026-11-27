# Prefill idle gaps at 130K: what the existing kernel trace shows (#159, lever 5)

Diagnosis from the kernel trace `run5` of 2026-11-16 (130K prompt, 72.0 s window, 217,236 kernels, rocprofv3 kernel
trace, no GPU run for this note). Scripts: `/var/tmp/claude/w159/gaps.py`, `pattern.py`, `pairs.py`, `perlayer.py`
(not part of the repository).

## Plan

- Question: where do the 2.58 s of 50 to 500 us gaps come from, and which lever fits which cause?
- Decision: a cause that the trace identifies gets a lever with an estimate; a cause that needs more evidence gets the
  one run that separates the candidates.
- Scope: CPU analysis of the existing trace, no new measurement.

## Facts from the trace

1. **Size.** Gaps between consecutive kernels (running maximum of the end time, all queues): below 5 us 526 ms
   (168,598 gaps), 5 to 50 us 356 ms, 50 to 200 us 37 ms, **200 to 500 us 2,733 ms (9,658 gaps)**, 0.5 to 2 ms 109 ms,
   above 2 ms 1,119 ms (132 gaps: the group boundaries and other stalls, the earlier finding of lever 3). So the 50 to
   500 us class is almost entirely one class of gap: 9,658 gaps of 272 to 295 us (p10, p90; median 280 us),
   **3.8 % of the window**.
2. **The length is constant.** The gap does not depend on the kernel before it (after kernels of 30 us to 6 ms) or after
   it. A fixed 280 us points at a fixed latency (a wake-up or a handshake), not at work.
3. **Where.** Per layer-chunk (1,296 of them, counted from `k_route`): 10 to 11 gaps in two out of three layers (the
   delta-net layers), 3 to 5 in every third (the attention layers); expert-GEMM batches and the indexer/top-k loop
   have none (`k_gemm_q2` -> `k_gemm_q2` 0.6 %). The gaps sit in the chain of dense kernels around the MoE:
   `k_moe_sum` -> `k_combine` 69 % of the occurrences, `k_combine` -> `k_norm16` 78 %, `k_norm16`/`k_combine_norm16`
   -> `k_gemm2` 76 / 90 %, `k_gemm2` -> `k_gemm2` 58 %, `k_gemm8b<256>` -> `k_gemm2` / `k_gemm8b<256>` / `k_gemm8b<128>`
   92 / 91 / 97 %, `k_dn_wave` -> `k_dn_norm` 76 %, `k_dn_norm` -> `k_gemm2` 91 %, `k_attn_wmma` -> `k_gemm2` 97.5 %.
   Per pair it is mostly a high share, not 100 %, so it is not a property of the two kernels alone.
4. **Not the copies.** Of the 9,658 gaps 9,412 (2.63 s) have no host-to-device copy running at any time during
   the gap; only 153 end where a copy ends. The expert copies (143,044 host-to-device copies, 6.6 s of copy time in the
   trace) are not what the kernels wait for.
5. **Not the known syncs.** The per-layer `hipStreamSynchronize` in `moe_group` (counts to the host) costs about 24 us
   of gap in the trace (shared expert runs meanwhile): it is overlapped as designed. The per-batch event records and
   waits in `moe_experts` show no gaps. `fp8_stats` and `moe_check` syncs are diagnostic switches, off.
6. The gaps are spread evenly over the window (about 700 per 5 s, falling to 217 in the last partial bucket).

## What the trace does not show

The kernel trace has no API timeline (the `rocpd_region` table is empty: no HIP API trace was recorded), so it cannot
tell whether the host is inside a call or the command processor is waiting. Candidates, with what each predicts:

| Candidate | Prediction | Lever if true |
|---|---|---|
| Host launch starvation in the dense chain after a drained queue (fixed 280 us wake/launch latency) | HIP API trace shows the launch call of the next kernel starting after the previous kernel ended | keep the queue full: launch the dense chain of the layer ahead of the syncs, or one fused chain; up to 2.7 s (3.8 %) |
| A HIP runtime step between dependent launches (marker / barrier packet with a host-handled completion, e.g. `hipLaunchHostFunc` progress callback per layer, event record in layered mode) | API trace shows a call (`hipEventRecord`, `hipLaunchHostFunc`, `hipMemsetAsync`) spanning the gap; gaps cluster right after it | remove or batch that call (the progress callback only once per chunk, events only where used); up to 1 to 2.7 s |
| Command-processor or power-state latency after short bubbles | no API call in the gap, the host is already ahead; gap does not change when the launch order changes | none from our side; check `TONA_` spin / clock probes of #161 |

Evidence against pure launch starvation: the host is ahead of the GPU at the end of a layer (it has launched the
whole expert loop and the tail while the GPU still computes the batches), so a drained queue would have to be caused by
a blocking call, which the code only has once per layer (`moe_group`).

## Decision and the one run needed

Not decidable from the existing trace. One run separates the candidates: a rocprofv3 trace of the 30K prompt with
`--hip-runtime-trace --kernel-trace --memory-copy-trace` (about 1.5 min GPU, exclusive, kind `exclusive`), then ask for
each 280 us gap: which API call is active on the host in the interval, and is the next launch call issued before or
after the previous kernel ended. If a call spans the gaps: remove it (bit-exact, host code). If the host is idle and
ahead: command-processor latency, no lever. Not queued yet; wait for the coordinator.

## Plan of the API trace run (written before the run)

- Scope: `rocprofv3 --kernel-trace --memory-copy-trace --hip-runtime-trace` of the production server path (build
  `work/build/w159`: levers 2 and 3, `TONA_PF_DEBUG=1`), one warm-up prompt of 2,000 tokens and one of 30,000 tokens,
  one round, about 2 to 3 min of exclusive GPU time. `hw_report.py` and `quiet_check.py` before and after (files in
  `/var/tmp/claude/w159/gap30/`).
- Why this size: 30K is the shortest prompt that shows the 280 us gap class (it was in the 30K trace too: the same
  call counts per layer-chunk); a repeat or 130K adds nothing to the question "which API call is active in the gap".
- Expectation (source: the 130K trace of this file): about 4,000 gaps of 272 to 295 us in the 30K window (the 130K trace
  has 9,658 in 72 s; 30K has about 15.5 s, about 2,000 gaps). The HIP API trace slows the host by some per cent, so the
  count may differ; a gap class of a different length would be a finding.
- Decision: a HIP call (event record, host function, memset, copy) active on the host thread across at least 90 % of the
  gaps -> remove or batch it (bit-exact host fix, then isa-diff, goldens, `ab.py` at 30K and 130K). The host thread idle
  in the gap and the next launch call issued before the previous kernel ended -> command-processor latency, lever 5
  closed with this evidence. Launch call issued after the previous kernel ended and no other call active -> host
  starvation, look for the blocking point.
- Active tooling: rocprofv3 with the three traces, `TONA_PF_DEBUG=1`, `OMP_NUM_THREADS=6`, `ROCR_VISIBLE_DEVICES=0`,
  `TONA_EXPERT_PROFILE=work/profile/experts-coder.bin`.

## Result of the API trace (30K, `gap30`, 15.67 s window, 50,859 kernels, 171,855 HIP calls, one host thread)

- Gaps: 2,320 gaps of 200 to 500 us, 654 ms (4.2 % of the window), median 279 us (p10 273, p90 295): same class as at 130K.
  The count is about 15 % above the plan's 2,000 (the 130K rate of 9,658 per 72 s is 2,080 per 15.5 s; per layer-chunk the
  rate is equal, the 30K window has one more attention layer share); the length class is the one expected.
- **The host is not the cause.** In 2,314 of the 2,320 gaps the host thread is inside `hipStreamSynchronize`, which it
  entered 17 to 55 ms (p10 to p90) before the gap: the kernel after the gap was launched long before. No other HIP call
  is active in the gap (2,296 of 2,320 gaps: no call at all besides the `hipStreamQuery` poll, which runs in 2 of them).
  So the queue is full and the command processor waits 280 us before dispatching a kernel that is already queued.
  The three candidates of the plan reduce to the third one (GPU side); not host starvation, not a HIP call.
- **The trigger is the length of the previous kernel.** Share of kernels followed by a 200 to 500 us gap, by duration
  (130K trace): below 1,000 us: 0.0 to 0.2 % (n about 200,000, including `k_gemm_q2` up to 900 us); 1,000 us and
  longer: 78.5 % (1.0 to 1.1 ms), 62.5 %, 70 %, 81.6 %, 46.7 %, 81 % (4 ms and longer). Inside one kernel pair the split
  is clean: `k_gemm2` -> `k_gemm2` is followed by a gap in 900 of 1,559 cases, every one with the first kernel at 1,275 to
  1,296 us (grid 409,600) and none of the 659 no-gap cases with a first kernel above 454 us. `k_combine` (1.59 ms,
  98 % followed by a gap), `k_moe_sum` (2.08 ms), `k_attn_wmma` (18 ms, 97.5 %), `k_gemm8b<256>` (3.06 ms, 92 %).
  A cumulative busy time does not trigger it: the expert-GEMM chains run 12 ms of kernels below 1 ms without a gap.
- Reading: a stall of about 280 us after a kernel that ran at least about 1 ms (a time-slice or preemption quantum of the
  hardware scheduler or the profiler hook are the candidates; the profiler cannot account for it alone: the profiled
  window of the 130K run is 72.0 s against 71.1 s unprofiled, at most 0.9 s of the 2.7 s of gaps).
- Next step (queued, `slice`): a spin-kernel chain without any profiler (`/var/tmp/claude/w159/slice.hip`, kernels of
  200 to 3,000 us, 64 and 512 blocks) measures the per-kernel overhead against the kernel length. A step in the
  overhead at about 1 ms confirms the hardware (or driver) cause without the profiler; then the lever is to cut kernels
  below 1 ms (the long kernels listed above: `k_combine`, `k_moe_sum`, `k_norm16`, long `k_gemm2`, `k_gemm8b<256>`,
  `k_attn_wmma`) if the cut costs less than the 280 us it saves, bit-exactly because only the grid is split.

## Cause research: the hardware scheduler quantum (no GPU; skill research-before-building, change type `bugfix`)

### Prior art

- [kfd_device_queue_manager.c, `add_queue_mes`](https://codebrowser.dev/linux/linux/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c.html)
  (Linux mainline source browser, lines 219 to 226), status: merged code. The KFD hands every compute queue to the MES
  with `/* MES unit for quantum is 100ns */`, `process_quantum = KFD_MES_PROCESS_QUANTUM /* Equivalent to 10ms */` and
  `gang_quantum = KFD_MES_GANG_QUANTUM /* Equivalent to 1ms */` (macro values 100000 and 10000), `inprocess_gang_priority =
  q->properties.priority`, `gang_global_priority_level = AMDGPU_MES_PRIORITY_LEVEL_NORMAL`. **The 1 ms gang quantum is the
  threshold of our trace** (0 % of the kernels below 1.0 ms followed by a gap, 78 % from 1.0 ms on). Read from the page
  (first 100,000 characters); `hws_max_conc_proc` and the oversubscription logic were not in the part read.
- [amdgpu MES documentation (mes.rst)](https://kernel.org/doc/html/next/_sources/gpu/amdgpu/gc/mes.rst.txt), status:
  documentation. "the MES firmware will preempt other user queues to make sure each queues get a time slice" when there
  are more queues (MQDs) than hardware queue slots (HQDs); kernel queues are "statically mapped to HQDs and never
  preempted". The page says nothing about quantum values or priority levels.
- [drm/amdgpu/mes12: set enable_level_process_quantum_check](https://lists.freedesktop.org/archives/amd-gfx/2024-September/114120.html),
  status: merged (2024). Title read via a search result; the page itself returned 403, so the claim that the flag is needed to
  enable process-quantum-based scheduling on MES 12 (gfx12) comes from the search summary, not from the patch text. Not verified.
- [mes_v12_api_def.h for gfx12](https://lists.freedesktop.org/archives/amd-gfx/2024-April/107622.html), status: merged (2024):
  the v12 queue descriptor has `queue_quantum_scale : 2` and `queue_quantum_duration : 8` bitfields; the mapping to time was
  not found.
- own research: research/12-gpu-hangs.md ("Priorities: no documented effect with MES found") and research/23-waiting-in-kernel.md
  (KFD queue preemption stops our queue while the wall clock keeps running; MES REMOVE_QUEUE/CWSR on preemption).
- On this machine (`/sys/module/amdgpu/parameters`, read only): `sched_policy=0` (HWS with oversubscription), `uni_mes=1`,
  `cwsr_enable=1`, `hws_max_conc_proc=-1` (default), `queue_preemption_timeout_ms=9000`, `mcbp=-1`, `compute_multipipe=-1`;
  kernel 7.2.15-200.fc44; KFD node fw_version 3490. Kernel command line has no amdgpu options.

Search: tools/papers/scan.py topics v1 (none fit; searched by hand with `gh search issues` and web search); terms: "MES time
slice quantum", "gang quantum", "KFD CWSR preemption latency", "compute kernel gap after long kernel"; window 2026-06-01..2026-11-16
(3 months plus) for issues, kernel sources and mailing-list pages without a window. `gh search issues` found nothing
on a 250 to 300 us stall after long kernels (hits were unrelated or about MES hangs and resets, for example
[warpfront/hipfire #1027](https://github.com/warpfront/hipfire/issues/1027));
no AMD statement on the quantum stall was found.

### Reading

- **Plausible chain (inference, not shown):** the MES gives a gang (our compute queue) a 1 ms quantum; a kernel that is still
  running at the end of the quantum is preempted (CWSR saves the waves) and rescheduled, which costs the 272 to 295 us before
  the next kernel starts. It matches the sharp 1 ms threshold, the constant length, the host being idle in a sync, and no
  dependence on the next kernel. What it does not prove: whether the stall is the CWSR save/restore of the running waves or
  the MES scheduling round trip, and why the queue is switched out when no other gang seems to want the slot (the desktop
  compositor works on kernel queues, which the documentation says are never preempted; the number of other user queues of
  this process is small). A kernel shorter than the quantum that happens to straddle a quantum boundary would also be hit,
  but the trace shows none below 1 ms: so the quantum seems to be counted per kernel or per dispatch, not as a free-running
  clock.
- **User-space levers:** `gang_global_priority_level` is fixed to NORMAL by the KFD, `inprocess_gang_priority` only orders
  the queues of one process: a stream priority (`hipStreamCreateWithPriority`) would change the second, not the quantum. No
  per-queue quantum is exposed by KFD or HIP in the sources read. CU masking does not change the scheduling quantum.
  Decision: no test of stream priority (the sources predict no effect and research/12 found none); a test costs a build and
  an `ab.py` and could only confirm that.
- **System-level levers (root, reboot, affect the desktop and the hang risk: for the maintainer to decide, not tried):** `amdgpu.sched_policy=1`
  (no oversubscription: queues are mapped once, no time slicing), `amdgpu.cwsr_enable=0` (no wave save: preemption would wait
  for the waves; the 9 s preemption timeout then bounds a kernel that does not end: dangerous for the megakernel that waits for
  the CPU, see AGENTS.md GPU safety), `hws_max_conc_proc`. `sched_policy=1` is the one that would remove the quantum switch by
  design; its effect on the desktop and on our two queues is unknown and the hang history (research/12, research/09) argues for
  a separate, planned test.
- **What stays in our hands:** shorten the kernels that run 1 ms or longer where it streams (done for four row-wise kernels,
  +1.1 %), and do not split compute-bound ones (the GEMMs lost 0.9 %).

### Proposal for the paper list (maintainers decide; abstract or source page only, not full text)

| Date | Item | Key statement | Relevance | Status |
|---|---|---|---|---|
| 2026-11-16 | [kfd_device_queue_manager.c add_queue_mes](https://codebrowser.dev/linux/linux/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c.html) | KFD compute queues: MES gang quantum 1 ms, process quantum 10 ms (100 ns units), global priority level fixed NORMAL | explains the 1 ms threshold of the 280 us stall | new |
| 2026-11-16 | [amdgpu mes.rst](https://kernel.org/doc/html/next/_sources/gpu/amdgpu/gc/mes.rst.txt) | user queues are time-sliced when MQDs outnumber HQDs; kernel queues are never preempted | context for the stall | new |
