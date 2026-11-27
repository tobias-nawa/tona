# Pass deadline and queue evictions (#85)

Date 2026-11-02. Platform: RX 9060 XT, Ryzen 5 9600X, `OMP_NUM_THREADS=6`, build `work/build/w85` = main 98dc089 (no source change
yet), `--mtp --ctx 140000`, port 8097, `systemd-run --scope -p MemoryMax=54G -p MemorySwapMax=0`, GPU lock. Raw data
(driver `drive.py`, logs, jsonl): `/var/tmp/claude/w85/`, not kept. **Every rate in this file is a decode rate unless it says prefill.**

## Source check (step 1): what resets the GPU

- `amdgpu_device_get_job_timeout_settings` (drivers/gpu/drm/amd/amdgpu/amdgpu_device.c, master): "By default timeout for all queues is 2 sec"
  (`gfx_timeout = compute_timeout = sdma_timeout = video_timeout = msecs_to_jiffies(2000)`); `lockup_timeout` is empty here.
  So the 2 s are the DRM-scheduler job timeout (`amdgpu_job_timedout`, amdgpu_job.c) of the amdgpu rings, in practice the
  gfx job of the compositor that waits behind our resident kernel. A queue that is evicted does not occupy the GPU.
- KFD user queues are not DRM-scheduler jobs. The eviction (`kfd_process_evict_queues`, kfd_process.c; `evict_process_queues_cpsch` ->
  `unmap_queues_cpsch`, kfd_device_queue_manager.c) waits for the preemption fence at most `queue_preemption_timeout_ms` (9000 here),
  then `kfd_hws_hang`. That timer measures the preemption, not the time the queue stays evicted.
- `evicted_ms` (kfd_process.c `kfd_procfs_stats_show`) reads `evict_duration_counter`, which the restore adds
  (`get_jiffies_64() - last_evict_timestamp`, kfd_device_queue_manager.c): the counter has grown when the waves run again.

## Plan: reproduce the abort on main (step 2)

1. Scope: driver of the #86 pressure series (30K and 130K prompts alternating, 200 tokens, greedy; every second pair with a reader
   `cat` of 55 GB of unrelated files to /dev/null), 20 pairs = 40 requests, about 55 min per arm, fresh server per arm.
   - Arm A: `IE_PIN_OVERFLOW=cap` (old way: the pin budget is lowered to the GTT, unpinned experts come from the page cache).
   - Arm B: default (`register`).
   Why this size: #85 appeared in a 130K decode under reclaim; the 130K requests carry the long passes. Both arms get the same request count.
2. Expectation (source: #85, `2026-11-01-slow-decode.md`): arm A may show 0-2 aborts in 20 pressured requests (rate unknown: #85 saw one in an
   unknown number of requests; I cannot bound it better than "not zero by construction"). Arm B: 0 aborts, evicted time per pressured
   request below 1 s (series `s86f`: 0.36 s). An abort in A is the reproduction; 0 in A means the series is too short or the trigger is gone,
   and the next step is a simulated eviction switch, not more luck.
3. Tooling: `IE_DIAG=1`, `IE_DECODE_DIAG=1`, the 10 ms `evicted_ms` sampler of the driver, core limit unlimited. No test switch.

## Result: reproduction on main (step 2), arm A `IE_PIN_OVERFLOW=cap` under pressure

Run `capA` (20 pairs, 40 requests: 20 with the reader, 20 without), 55 min, driver `/var/tmp/claude/w85/drive.py`, `IE_DIAG=1`.

| | requests | aborts (500) | slow decodes (< 55 tok/s) | `evicted_ms` in the requests | longest episode |
|---|---:|---:|---:|---:|---:|
| with reader | 20 | **0** | 0 | 19.9 s (max 4.7 s in one request) | 360 ms |
| without reader | 20 | 0 | 0 | 0.11 s | 40 ms |

Decode 58-74 tok/s (30K: 60-74, 130K: 58-65). Against the expectation (0-2 aborts): inside it, but the abort did not come: the evictions
under pressure are large in sum (19.9 s, as in the old default behavior) but come in episodes of at most 360 ms, below the 300 ms grace
plus the 1.5 s budget. #85 needed a pass whose deadline was overshot by 4.4 s. Arm B (default) was not run: it evicts less than A (0.36 s
per pressured request in `s86f`) and A already gave 0 aborts, so B cannot show more. The abort is not reproducible by luck at this size;
the proof of the fix needs a simulated eviction (see below).

## Plan: speed and bits of the eviction credit (step 3, R4 and R5)

Base `w85-base` = main 98dc089; B `w85` = commit 134035d (the credit). `isa-diff` (A `w85-base`, B `w85`): `k_mega<1>` spills 61 -> 62 VGPR and
107 -> 105 SGPR, `k_mega<4>` 49 -> 54 VGPR and 107 -> 111 SGPR, `k_mega<8>` 149 -> 180 VGPR; `k_chain_phase`, `k_chain_persistent` (test kernels,
share the barrier) and `op_qsa_prep` differ in code. So this is class C5, not C4: the kernel changed on purpose and the speed must be measured.
1. Scope
   - R4: `ab.py --env IE_DETERMINISTIC=1 --workload mid,long`, 1 round; SHAs equal to `bench/sha.toml`.
   - R5: `ab.py --a w85-base --b w85 --workload short,mid --primary decode`, rounds decided by `ab.py` (noise.toml). Decode is the rate that the
     barrier code touches; `mid` adds prefill (the 30K prompt runs through `k_mega<8>`, the kernel with the largest spill change).
2. Expectation: decode within -0.5..+0.5 % (ticket), prefill within -0.5..+0.5 %. Source for the risk: `2026-10-28-mega-spills.md` (a shape change
   of the loop costs 0.5-3.4 %); here the change sits in the cold branch of the wait only (a noinline call), so the loop shape is the same
   and the extra spills come from the call. A decode loss over 0.5 % is a finding; up to 2 % may be accepted as a stability fix with a reason.
3. Tooling: none beyond `ab.py` (its own plan block); the credit is on in B (counter found) and has no cost without a deadline hit except
   one `pread` of the sysfs counter per launch (about 3 us).

## Result: speed and bits (R5, `2026-11-02-evict-deadline-ab-speed.md`)

6 rounds, `short,mid`, base `w85-base` against `w85` (134035d): prefill@2000 -0.15 %, prefill@30000 +0.03 %, decode@2000 +0.88 %, decode@30000 +0.73 %.
Decode is **outside the expectation (-0.5..+0.5 %), on the good side**: B is faster. The reason is not known; the spill counts of `k_mega<1>`
fell by 2 SGPR and rose by 1 VGPR, so a shift in code generation is the likely source (the kernel is sensitive to its shape,
`2026-10-28-mega-spills.md`). Nothing got worse. Expert cache counts and SHAs of A and B are identical for all three requests.

## Result: simulated credit on the GPU (diag build `w85-diag`, `-DIE_WAIT_DIAG=ON`)

Written to the plan agreed with the coordinator before the run. Server `--mtp --ctx 140000`, one request of 48 tokens per fresh server,
`IE_MEGA_BUDGET_MS=300 IE_TEST_FREEZE_ALL=1 IE_TEST_FREEZE_MS=650 IE_TEST_FREEZE_GEN=300 IE_TEST_FREEZE_LAUNCH=20`, `IE_MEGA_PROF=1`
`IE_MEGA_STALL_MS=200`, `IE_DIAG=1`. The freeze fires at the first launch from number 20 on whose program reaches barrier 300 (launch 23;
the drafter programs are shorter, so the first attempt with `launches == 20` froze nothing). 60 s pause between starts.
Expectation: arm 1 0 aborts and a pass of about 0.67 s; arm 2 5 of 5 aborts at 0.65-0.70 s; the kernel never runs longer than 1.25 s.

| Arm | Credit | Starts | Aborts | Pass length (kernel) |
|---|---|---:|---:|---|
| 1 | simulated: `IE_TEST_EVICT_MS=650` (frist 300 + 650) | 5 | **0** (HTTP 200 x5) | 691.2-691.8 ms |
| 2 | real KFD counter (`stats_7773/evicted_ms`, grew by 0 ms: the waves ran, nothing was evicted) | 5 | **5** (HTTP 500 x5) | 664.8-665.2 ms up to the abort |

Both as expected. Arm 2 is the case the coordinator named as dangerous (a credit for waves that really ran): the host answers 0, the wave
aborts at 665 ms, far below 2 s. The start log of arm 2 names the counter file. `dmesg` is not readable for this user ("Operation not
permitted") and `journalctl -k` returns no entries, so amdgpu messages could not be checked; the desktop did not report a stall.

## Result: acceptance series with the credit (R8), `IE_PIN_OVERFLOW=cap`, build `w85`

Run `capB`: the series of `capA` again (20 pairs, 40 requests, 20 with the reader, `IE_DIAG=1`) on the build with the credit; the start log names
`stats_7773/evicted_ms` (the dGPU). 0 aborts in 40 requests, 0 slow decodes (< 55 tok/s), 0 credit requests (no evicted stretch moved a pass
past its deadline). `evicted_ms` in the pressured requests 15.2 s in total (max 1.5 s per request, longest episode 308 ms), without reader
0.67 s. Decode 58-72 tok/s. The same request count as `capA` (0 aborts, 19.9 s evicted). So the natural trigger of #85 did not show in 80
requests on either build; the evidence that the credit works is the simulated freeze above, and that it cannot be abused is arm 2 there.

## Result: final build (after the clang-tidy fixes, kernel shape changed again: `k_mega<4>` 117 SGPR spills, `k_mega<8>` 181 VGPR / 247 SGPR)

The tidy fixes moved the credit branch into a helper, so every measurement was repeated on the final build `w85` (f97596c plus formatting).
- Speed (`2026-11-02-evict-deadline-ab-speed-final.md`, 6 rounds? see file): prefill@2000 +0.33 %, prefill@30000 -0.18 %, decode@2000 +0.95 %,
  decode@30000 +0.93 % (again outside the expectation of -0.5..+0.5 % on the good side; reason unknown, likely code generation).
- Bits (R4, `IE_DETERMINISTIC=1`, 2 rounds): SHAs `08f0bd31e0c0277f` (30K) and `e9545d70f2400b87` (130K), equal to A and to `bench/sha.toml`.
- Goldens (`gpu-status.sh --keep-going`): gpu, cpu plan, prefill, prefill dflash identical; server identical except `ssd-1` and `ssd-2`.
  **Those two differ on main 98dc089 as well** (same run with `work/build/w85-base`, `GOLDEN_CASES=ssd`: both builds give the same, different
  text, so the reference in `work/golden/server/ssd-*.json` is stale or the SSD case drifted; not caused by this change, to be handled in its own issue).
- Simulated credit again (3 starts per arm, same values): arm 1 (credit 650 ms simulated) 0 of 3 aborts, arm 2 (real counter, credit refused) 3 of 3 aborts.
- Pressure series (R8) was not repeated on the final build: the first build showed 0 credit requests, so the series does not reach the changed branch.
- Gates: `just check` green (clang-tidy included), 39 of 39 unit tests, 319 pytest.
