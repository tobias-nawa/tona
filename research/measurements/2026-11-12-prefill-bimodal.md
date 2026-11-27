# Prefill@2000 bimodal per server start

## Plan (written before the run)

- Scope: 8 server starts of build `tona129` (commit 85c93b4) in production mode (no TONA_DETERMINISTIC), port 8099, ctx 140000, MTP drafter, expert profile as ab.py.
  Per start: prefill of the 2000-token "sum" prompt (20 decode tokens), then the 2000-token "write" prompt. Log of every start kept (TONA_PF_DEBUG=1, TONA_DECODE_DIAG=1).
  Why this size: the effect is per start, so starts are the unit; one prompt pair each is the smallest workload that shows it. About 65 s per start, 10 min total.
- Expectation: sum prefill about 1507 or about 1350 tok/s per start (source: verify-pass.md around line 235), write prefill about 1610 tok/s (source: mtp-dynamic-head-ab-off-identity-r2.md). Tolerance 1 %.
  If both modes show: compare the counters (expert hits/misses, CPU experts, phase times) of a fast and a slow start.
- Active tooling: TONA_PF_DEBUG=1, TONA_DECODE_DIAG=1 (both only log); OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=work/profile/experts-coder.bin.

Test conditions: `tools/hw_report.py` and `tools/quiet_check.py` before (load average 4.5, other workers on the machine; 1.34 GiB VRAM held by others; swap 1.7 GiB in use).

Test conditions (`tools/hw_report.py` before the first run; `tools/quiet_check.py` before and after):

```text
### Test conditions

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.34 GiB held by others before the run, 1 % busy before the run
- RAM: 60.5 GiB, 55.9 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 4.91, 3.9, 2.87
- CPU use by other programs over 0.5 s: 0.2 % of all threads, swap in use 1.72 GiB
- sandbox: no
- Commit: `85c93b4` (clean tree)
- Quiet check (2026-11-12 22:35:17): warning
  - warning: load average 4.52 is high (it lags a minute)
  - warning: 1.34 GiB of 15.9 GiB VRAM are held by others
- Quiet check (2026-11-13 00:42:48): warning
  - warning: 1.9 GiB of 15.9 GiB VRAM are held by others
```

The machine was shared with other workers (load average 1 to 4.5, builds queued through the GPU line); every run below held the GPU lock.
Several later runs added to the plan: they are listed under "Runs added to the plan" with the question each answers.

## Result 1: the bimodality did not appear today

The 2000-token "sum" prompt as the first request of a start, production mode, about 75 starts of build 85c93b4 (and 6 of the older `merged` build):
1291 to 1371 tok/s **prefill**, never 1507 (about 75 starts in all, `work/ab` and `/var/tmp/claude/tona129`); A/A with `ab.py` (6 rounds, same build both arms): 1359.8 to 1365.1 and 1359.7 to 1367.4,
"write" 1605 to 1613: no second mode. The deterministic counters of every start are equal (7641 hits / 1262 misses for the first request, 5071 / 1569 for the second; 133160 / 19689 over 400 decoded tokens),
so the expert cache and the profile are not the difference. Loaded: 82 slots, 27.4 GB pinned in all starts; the sessions of the fast mode (`work/ab/ab-20261112-153712-logs`) had 83 slots and 0.7 GB of the pinned experts registered as userptr
(of the 327 server logs in `work/ab` with the 1934-token prompt, the 10 that report 1.3 s all have that configuration, so does 14 + 7 that report 1.4 and 1.5 s; none of the 82-slot starts reports 1.3 s; `TONA_EXPERT_SLOTS=83` and a holder process that forced 0.7 and 1.8 GB of userptr did not bring the fast mode back). Checked and without effect on the rate:
delay between ready and the request (0, 5, 20, 60 s), GPU clock levels (`pp_dpm_sclk/mclk/fclk/pcie`: same in every start), PCIe link (32 GT/s x16 in all), THP, buddy info, thread placement.
**The fast mode of 2026-11-12 (about 1507 tok/s) is therefore not reproduced; ungeprueft (unchecked): what it was.**

## Result 2: the first request of a start is cold (GPU clock ramp)

Same prompt twice in one start (5 starts): request 1 1366 +- 4, request 2 1611 +- 3 tok/s **prefill** (+18 %).
`rocprofv3` kernel trace (same kernels, 5128 per request): kernel time of the first request 1234 ms, of the second 1039 ms (every GEMM family +30 to +40 %), idle gaps between kernels equal.
Clock of the GPU (hwmon `freq1_input`, 10 ms): after 0.5 s of idle the first request runs about 550 ms at 1.8 GHz and 80 W, ramps to 3.1 GHz by about 950 ms; the second request starts at 3.1 GHz and 160 W.
The decay is fast: request after a pause of 0.5 / 1 / 2 / 4 / 8 / 16 / 30 / 60 s: 1339 / 1305 / 1359 / 1351 / 1359 / 1349 / 1356 / 1351 tok/s **prefill**, directly after the previous one: 1614 / 1590.
So every prefill after 0.5 s of idle pays about 190 ms of GPU time in this state (firmware DPM, not the engine), also in production, not only after a start.
The ramp is firmware work: `power_dpm_force_performance_level` and `pp_power_profile_mode` are root files. A standalone one-block spin kernel (`burn`, 32 threads) ramps the clock to 3.1 GHz in 0.5 s (from 100 ms on),
but a spin kernel started with the prefill (tried: `ClockWarm`, 1 block on a stream of its own, condition: clock below 2400 MHz) did not move the ramp of the first request (still 550 ms plateau), so it was dropped (principle 8, below).
Suggestion for the owner of the machine (needs root, not tried): `pp_power_profile_mode` COMPUTE (5) or `power_dpm_force_performance_level=high` by a udev rule; ungeprueft how large the gain is.

## Result 3: the number of HIP queues moves the prefill by 7 to 9 % (found, cause unknown)

While looking for a way to ramp the clock, creating one more HIP stream (not blocking or blocking, never used) in the first prefill made it faster from request 1 on, a stream before the prefill's own copy stream was needed;
an event did nothing, a hipMalloc or hipHostMalloc did nothing, `hipStreamCreateWithPriority` of the copy stream (high or low) did nothing.

| Configuration | request 1 (sum, 1934 tokens) | request 2 | request 3 | note |
|---|---:|---:|---:|---|
| default | 1370 | 1612 | 1570 | queues at the first prefill: null, prefetch, prefill copy (3, `AMD_LOG_LEVEL=3`); the stash stream comes after request 1 |
| 1 extra stream created in the first prefill (before the copy stream) | 1463 | 1760 | 1710 | |
| 2, 3 or 4 extra streams there | 1460-1466 | 1759-1764 | 1707-1713 | the same |
| 1 extra stream at the end of the first prefill | 1370 | 1607 | 1710 | fast from request 3 |
| `GPU_MAX_HW_QUEUES=2` | 1469 | 1764 | 1709 | |
| `GPU_MAX_HW_QUEUES=3` with the extra stream | 1464 | 1759 | 1708 | |
| 1 spacer stream at load (`TONA_QUEUE_SPACERS=1`) | 1369 | 1610 | 1714 | fast from request 3 |
| 2 spacer streams at load (`TONA_QUEUE_SPACERS=2`) | 1468 | 1762 | 1712 | |
| `GPU_MAX_HW_QUEUES=1` | 283 | 518 | 512 | not usable (all streams on one queue) |

(**Prefill** tok/s, one row = 2 or 3 starts that agreed to 0.5 %.) With `rocprofv3` (it changes the queues) kernel times were equal and the idle gaps 20 to 60 ms shorter, so the effect sits in the dispatch, not in the kernels.
Known on RDNA4: the MES firmware treats several queues differently (llama.cpp #24237: `GPU_MAX_HW_QUEUES=1` against a clock that stays at maximum while idle; #3929, #21430). **The mechanism of the +7 to 9 % is not known (ungeprueft: queue to pipe or SDMA engine assignment).**
Idle cost checked: with 2 spacers the GPU idles like without (busy 2 %, 22 to 23 W, 800 MHz clock; no server 18 W, 27 MHz; the server's own idle cost is the same in both).

## Result 4: A/B of the fix (`TONA_QUEUE_SPACERS=0` against the default 2, production mode)

## Result

A: `tona129` env {'TONA_QUEUE_SPACERS': '0'}; B: `tona129` env {'TONA_QUEUE_SPACERS': '2'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1359.4 / 1362.5 / 1362.6 / 1364.9 / 1360.3 / 1362.2 | 1453.9 / 1455.2 / 1457.2 / 1454.9 / 1456.0 / 1457.9 | 1362.4 (1359.4-1364.9) | 1455.6 (1453.9-1457.9) | +6.89 % [+6.69, +7.10] | **B faster** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 75.8 / 75.6 / 75.6 / 75.7 / 75.5 / 75.5 | 75.6 / 75.7 / 75.6 / 75.8 / 75.5 / 75.6 | 75.6 (75.5-75.8) | 75.6 (75.5-75.8) | +0.02 % [-0.25, +0.29] | **equal** after round 3 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1609.7 / 1607.0 / 1608.3 / 1608.6 / 1611.2 / 1605.3 | 1739.4 / 1738.2 / 1739.1 / 1739.1 / 1739.5 / 1739.4 | 1608.4 (1605.3-1611.2) | 1739.2 (1738.2-1739.5) | +8.13 % [+7.92, +8.35] | **B faster** after round 1 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 124.6 / 123.7 / 124.0 / 123.0 / 123.6 / 123.8 | 123.8 / 123.2 / 124.5 / 124.3 / 123.3 / 124.9 | 123.7 (123.0-124.6) | 124.0 (123.2-124.9) | +0.17 % [-0.80, +1.15] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 94.1 / 94.1 / 94.1 / 94.3 / 94.3 / 94.4 | 94.1 / 94.2 / 94.2 / 94.2 / 94.1 / 94.1 | 94.2 (94.1-94.4) | 94.2 (94.1-94.2) | -0.07 % [-0.60, +0.46] | **equal** after round 4 |
| decode-text: Decode tok/s (bench-set text) | 68.1 / 68.1 / 68.1 / 68.1 / 68.1 / 68.0 | 68.1 / 68.1 / 68.0 / 68.0 / 68.0 / 68.1 | 68.1 (68.0-68.1) | 68.0 (68.0-68.1) | -0.05 % [-0.56, +0.46] | **equal** after round 4 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | +6.95 % | -0.17 % | +8.06 % | -0.70 % (off plan) | +0.00 % | +0.00 % | 206 |
| 2 | BA | +6.81 % | +0.14 % | +8.16 % | -0.35 % | +0.11 % | +0.00 % | 208 |
| 3 | AB | +6.94 % | -0.07 % | +8.13 % | +0.40 % | +0.11 % | -0.15 % | 207 |
| 4 | BA | +6.59 % | +0.13 % | +8.11 % | +1.04 % | -0.11 % | -0.15 % | 207 |
| 5 | AB | +7.04 % | -0.05 % | +7.96 % | -0.19 % | -0.21 % | -0.15 % | 207 |
| 6 | BA | +7.02 % | +0.15 % | +8.35 % | +0.85 % | -0.32 % | +0.15 % | 207 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +6.89 % | +4 to +12 % | within the expectation |
| decode@2000 | +0.02 % | -0.5 to +0.5 % | within the expectation |
| prefill@2000:write | +8.13 % | +4 to +12 % | within the expectation |
| decode@2000:write | +0.17 % | -0.5 to +0.5 % | within the expectation |
| decode-code | -0.07 % | -0.5 to +0.5 % | within the expectation |
| decode-text | -0.05 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 130767, 22082 / 130767, 22082 / 130767, 22082 / 130767, 22082 / 130767, 22082 / 130767, 22082 | 130767, 22082 / 130767, 22082 / 130767, 22082 / 130767, 22082 / 130767, 22082 / 130767, 22082 | yes |
| 2000:write | 83981, 26296 / 83981, 26296 / 83981, 26296 / 83981, 26296 / 83981, 26296 / 83981, 26296 | 83981, 26296 / 83981, 26296 / 83981, 26296 / 83981, 26296 / 83981, 26296 / 83981, 26296 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 22.5 min for 6 rounds (3.4 min per round); a fixed 3 rounds would take 10.3 min (-10.3 min saved).

Prefill **+6.89 % [+6.69, +7.10]** (sum, 2000 tokens) and **+8.13 % [+7.92, +8.35]** (write), decided after round 1, all 6 rounds listed; decode (**decode** tok/s) equal in all four metrics; hits and misses and the SHAs of the answers equal in all rounds.
Goldens of build `tona129` (bit identity, `TONA_DETERMINISTIC=1`): decode (csv, rust, explain), CPU plan, prefill, prefill with dflash, server (36 cases plus): all identical (44 `identical` lines, none different).
Slot count (82) and loaded memory are unchanged by the two streams; they hold no memory beyond the queue.

## Runs added to the plan

Each of these is a start or a few starts of the repro script (`a scratch folder (repro*.py)`, not kept in the repo): 2 requests with the same text (cold against warm), delay before the first request, hwmon sampling of clock and power (10 ms),
`rocprofv3 --kernel-trace --memory-copy-trace` on 4 + 2 + 1 starts (kernel sum and gaps per request), the idle-decay series, the queue-count variants of Result 3, a holder process for the GTT, `TONA_EXPERT_SLOTS=83`, priorities of the copy stream, `HSA_ENABLE_SDMA=0` (355 tok/s **prefill**: the copies need the SDMA engines), and the idle cost.
Per-start logs: `/var/tmp/claude/tona129/` (outside the repo).

## Principle 8: what was dropped

- Spin kernel for the clock ramp (`ClockWarm`): measured +6.5 % first request, but the same gain came with the stream alone and no spin kernel (`TONA_CLOCK_WARM_MHZ=1`: the kernel never started), and the clock curve of the first request was unchanged with it. Reason: no effect on the ramp. Search 2026-11-13 (llama.cpp issues on HIP/RDNA4 idle and clock, MES queues: #24237, #3929, #21430; `research/papers.md` grep DPM/queues): nothing on a user-space way to ramp the clock faster without root. Attempt to remove the cost: spin from the start of the prefill, 1 block, 32 threads (above).
- Priority of the copy stream: no effect, not kept.

## Prior art

- [llama.cpp #24237](https://github.com/ggml-org/llama.cpp/pull/24237), status: closed (body read). `GPU_MAX_HW_QUEUES=1` as default on HIP against an RDNA4 MES behaviour that keeps the GPU at maximum clock while idle. Not adopted: it goes the other way (fewer queues; with 1 queue our prefill falls to 283 tok/s); it confirms that the number of queues changes the clock and dispatch behaviour of the MES.
- [llama.cpp #21430](https://github.com/ggml-org/llama.cpp/issues/21430), status: closed (body read). RX 9070 XT: a second queue user keeps the clock at ~3400 MHz and GFX at 100 % after inference. Same remark; we measured an idle GPU with the 2 extra streams (22 to 23 W).
- [llama.cpp #3929](https://github.com/ggml-org/llama.cpp/issues/3929), status: closed (body read). Streams on ROCm at constant 100 % load (RDNA3). Not adopted.
Search: tools/papers/scan.py topics v1; topics: none (the ledger of research/papers.md was grepped for DPM, queue, MES, clock); terms: GPU_MAX_HW_QUEUES, hardware queue, MES, clock ramp; window: 2025-01..2026-11 (research/papers.md is complete from 2025-01; `scan.py` not run for the last weeks: nothing on a clock ramp in a user-space program was expected there)
