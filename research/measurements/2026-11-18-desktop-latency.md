# Desktop submission latency next to Tona: sweep of CU mask, yield and canary priority (#168)

Plan, written before any run (2026-11-18). Background and sources: [research/63](../63-desktop-latency.md). Results are added below the line
`## Results`. **No default is changed by this work**; it feeds the work/server priority setting (#116).

## Question and decision

Question: a headless Vulkan canary (stand-in for a compositor frame: a 64 MiB fill submitted every 20 ms) waits **p50 14 to 15 ms, p99 about 54 ms** while Tona
decodes, and about 1 ms on the idle card (level 0, `2026-11-18-vram-level0.md`). Which lever shortens that wait, and what does it cost Tona?

Output: per arm the share of canary submissions within the 16.7 ms frame budget (60 Hz) against Tona's decode and prefill rate, as a trade-off table.
Decision per outcome (for the coordinator; nothing is changed here): an arm with frame hit rate of at least 95 % in the decode phase at a decode cost
of at most 10 % is a candidate for the "work" priority mode of #116; a lever with no effect on the canary is dropped from that setting; if no arm helps, the answer
for a desktop on the same GPU stays "display on the iGPU" (research/62 placement 2, a user decision).

## Scope (every run is one exclusive queue job, one server start; same build `b164t` and context 140000 in all arms; the desktop guard stays on)

| Arm | Variables | Why |
|---|---|---|
| default | none | reference |
| cus4 | `TONA_FREE_CUS=4` | 4 CUs (2 WGPs) free for graphics waves; H1 |
| cus8 | `TONA_FREE_CUS=8` | 8 CUs free; H1 |
| yield10 | `TONA_MEGA_YIELD_MS=10` | a shorter yield (default 100 ms); H4 |
| yield2 | `TONA_MEGA_YIELD_MS=2` | the extreme of H4; restarts may cost decode |
| prio-high | canary created with global priority HIGH (what a compositor gets as DRM master) | tests whether an unprivileged client is granted it; if the driver refuses, the arm is recorded as "not granted" and no server starts |

Order: default, cus4, cus8, yield10, yield2, prio-high, then the same arms in reverse order (default last): 11 server runs. The two defaults show drift
over the sweep. A job ends after: idle phase 20 s (server up, no request), decode phase 40 s (400-token requests, Tona's own timings), prefill phase 40 s
(2000-token prompts, a different head each time so no prompt cache is hit; prefill rate = prompt tokens / prompt time). The canary runs through all
phases at one submission per 20 ms. Tona-side queue priority is **not** an arm: it would need the megakernel moved off the null stream (research/63 section 5).

Why this size: the question is whether a lever changes the p50 by a factor, not by a few percent, so one 40 s phase per arm and pass is enough
(2000 canary samples per phase, p99 from about 20 samples; the level-0 canary phases had 250 to 510 samples and gave stable p50 within 3 % between arms). The
decode cost needs about 1 % resolution: within-phase decode means differed by 0.1 to 0.2 % in level 0 (6 to 13 requests of 400 tokens), and the two defaults give the
drift. Two passes (forward and reverse) are the minimum that shows an order effect. A repeat of a surprising arm is a follow-up job, not part of this plan.

## Expectation (with source; an outcome outside is a finding)

| Quantity | Expectation | Source |
|---|---|---|
| default, idle phase | p50 about 1 ms (0.5 to 2), hit rate 100 % | smoke test on the idle card: p50 1.0 ms, max 1.4 ms (no Tona; Tona idle adds nothing known) |
| default, decode phase | p50 14 to 16 ms, p99 50 to 60 ms, hit rate 50 to 65 %; Tona decode 80 tok/s (+-1) | level 0, four arms: p50 14.4 to 15.5, p99 52.8 to 55.5 ms; hit rate computed from p50 of 15 ms about 55 % (not measured) |
| default, prefill phase | p50 under 5 ms, hit rate 80 % or more; prefill about 1,450 tok/s at 2000 tokens | H5 (research/63), no earlier measurement of the canary during prefill; prefill rate from `2026-11-18-vram-chunks-gtt-r3.md` |
| cus4, decode phase | p50 3 ms or less, hit rate 95 % or more; decode -5 to -11 % | H1 (research/63); cost from `2026-10-23-cu-mask-desktop.md` (28 CUs: +4.8 % ms/pass short run, -11 % tok/s long run) |
| cus8, decode phase | p50 3 ms or less, hit rate 95 % or more; decode -10 to -20 % | extrapolation of the same two points to 24 CUs; the 16-CU data of `2026-11-17-cu-scaling.md` was not used |
| yield10 | no change of the canary (p50 within 10 % of default); decode within -1 % | H4: the yield fires only on a wait for the CPU longer than the limit |
| yield2 | no change of the canary; decode -5 % to 0 | H4; the restart cost per yield is not known |
| prio-high | not granted (VK_ERROR_NOT_PERMITTED) | `amdgpu_ctx_priority_permit`: above NORMAL needs `CAP_SYS_NICE` or DRM master (research/63 section 2.2) |

## Active tooling

Only the arm's `TONA_*` variable. `tools/desktop_latency.py` follows the kernel log (`journalctl -k`) for GPU reset, ring timeout, failed command submission and
stops at the first. No `TONA_DIAG`, no profiler, no VRAM balloon, no VRAM pressure (the canary holds 64 MiB; the server uses its default reserve of 1 GiB).
Test conditions: `tools/hw_report.py` and `tools/quiet_check.py` before and after every job (`/var/tmp/claude/168/<job>.tc`).

## Stop rules

STOP file in the run directory, a kernel abort line, a canary error, a failed Tona request: the job ends and is reported. Jobs are not started after 19:20 so that
nothing runs into the 20:00 night run (#165); what has not run waits for tomorrow.

## Results (2026-11-18 09:34 to 10:00; build `b164t`, tools at the commit of this file; 12 queue jobs, exclusive, desktop guard on, no requeue)

All 12 jobs ran as planned, in the order default, cus4, cus8, yield10, yield2, prio-high and the reverse. The prio-high probe was refused by the driver twice
(`vkCreateDevice` returned `VK_ERROR_NOT_PERMITTED_KHR`, -1000174001), so those two jobs ran no server, as predicted. 10 server runs, 2 per arm. Summaries (sums and times only):
`2026-11-18-desktop-latency-runs/<job>-summary.json`, test conditions `<job>.tc.txt`. Quiet check before and after every job: a warning for the 1.9 GiB of VRAM the
desktop holds and, in 7 of 12 jobs, swapping of 1 to 8 pages in one second; nothing else. No stop rule fired, no kernel abort line, no canary error, no failed request.

**Frame hit rate** = share of canary submissions that finished within 16.7 ms (one 60 Hz frame). The canary waits for each submission before it starts the next one, so a
run with long waits submits fewer frames (default decode: 968 submissions in 40 s, cus8: 1969); the hit rate is per submission and flatters the slow arms, the real share of
missed frames in time is higher than `1 - hit rate`. Rates are Tona's own timings: decode = generated tokens per s of 400-token requests after a 35-token prompt,
prefill = prompt tokens per s of 2000-token prompts (a different head each request, no cache hit). Each cell is the mean of the two runs of the arm (forward and reverse
order; the two agree within 0.5 % on Tona's rates and within 6 % on p50 and p99, so there is no order or drift effect worth correcting).

| Arm | decode phase: canary p50 / p99 / max (ms) | frame hit rate | decode tok/s | vs default | prefill phase: p50 / p99 (ms) | frame hit rate | prefill tok/s | vs default |
|---|---|---:|---:|---:|---|---:|---:|---:|
| default | 18.8 / 47.2 / 54.0 to 74.2 | 40.7 % | 82.84 | - | 0.25 / 30.5 | 93.9 % | 1709 | - |
| `TONA_FREE_CUS=4` | 0.27 / 31.2 to 36.4 / 66.5 to 63.0 | 98.3 % | 79.17 | -4.4 % | 0.23 / 0.76 | 99.9 % | 1607 | -6.0 % |
| `TONA_FREE_CUS=8` | 0.24 / 0.34 to 0.37 / 2.99 to 3.19 | 100 % | 71.86 | -13.3 % | 0.23 / 0.62 to 0.78 | 100 % | 1477 | -13.6 % |
| `TONA_MEGA_YIELD_MS=10` | 18.9 / 46.7 to 48.3 / 53.7 to 59.4 | 41.0 % | 82.95 | +0.1 % | 0.25 / 30.2 to 30.4 | 93.8 % | 1713 | +0.2 % |
| `TONA_MEGA_YIELD_MS=2` | 18.9 / 46.7 to 48.4 / 53.8 to 66.8 | 40.7 % | 82.91 | +0.1 % | 0.25 / 30.3 to 30.6 | 93.8 % | 1712 | +0.2 % |
| canary priority HIGH | refused (`VK_ERROR_NOT_PERMITTED_KHR`) | - | - | - | - | - | - | - |

Idle phase (server up, no request), all arms: p50 0.79 to 0.85 ms, p99 1.15 to 1.21 ms, hit rate 100 %.
Where two values are given ("a to b"), they are the two runs of the arm; a single value is the mean of two runs that differ by less than 1 %.

Against the plan's expectation:
- default idle: inside (p50 0.8 ms). **default decode: outside, worse**: p50 18.6 to 19.0 ms (expected 14 to 16), hit rate 40 % (expected 50 to 65 %); level 0 had a 100 ms canary
  period and 15 ms, the 20 ms period of the sweep samples more of the slow stretches. default prefill: inside (p50 0.25 ms, hit rate 94 %), but the p99 is 30 ms: about 6 % of the
  submissions wait about 30 ms or more even in prefill.
- cus4: p50 and hit rate inside (0.27 ms, 98.3 % against 3 ms and 95 %); **decode cost -4.4 % is better than the expected -5 to -11 %**. The tail is not gone: p99 31 to 36 ms and a
  maximum of 63 to 67 ms in the decode phase (1.6 % of the submissions miss the budget).
- cus8: inside (hit rate 100 %, max 3 ms); decode -13.3 % inside the expected -10 to -20 %.
- yield10 and yield2: **no effect on the canary or on Tona** (H4 not supported), as expected; the shorter yield costs nothing measurable in this decode loop.
- prio-high: refused, as expected.

## Reading (hypotheses of research/63)

- **H1 supported**: the wait of 19 ms median in decode disappears when CUs are free (0.27 and 0.24 ms), while the idle server and the prefill phase (kernels of about 1 ms with gaps)
  are fast. The canary's wait is about half a pass (43 ms); this fits waves that cannot start while the megakernel holds every CU. It was not shown which resource (wave slots,
  VGPRs, LDS) is the limit, and the CU mask is the only lever tested that works.
- **H2** (the 1 ms MES quantum bounds the wait): not supported, waits are 15 to 70 ms.
- **H3** (Tona-side queue priority): not tested, see research/63 section 5. The refusal of HIGH for an unprivileged client confirms that the compositor's HIGH priority cannot be
  reproduced by this canary; the real compositor may therefore do better than the canary on the default arm. That is not measured.
- **H4** (yield): refuted for this workload.
- **H5** (prefill fast): partly: p50 0.25 ms but a 30 ms tail in 6 % of the submissions; with 4 free CUs the prefill tail is 0.76 ms.

## Trade-off table for the priority setting (#116); no default is changed

| Lever | frame hit rate, decode / prefill | decode cost | prefill cost | verdict against the plan's rule (hit rate 95 % or more at decode cost 10 % or less) |
|---|---|---:|---:|---|
| default (32 CUs) | 40.7 % / 93.9 % | 0 | 0 | misses the budget on 6 of 10 submissions during decode |
| 4 CUs free | 98.3 % / 99.9 % | -4.4 % | -6.0 % | **qualifies**; 1.6 % of submissions still miss in decode (tail 31 to 36 ms p99, max 66 ms) |
| 8 CUs free | 100 % / 100 % | -13.3 % | -13.6 % | **does not qualify on cost**; no tail at all (max 3 ms) |
| shorter yield (10, 2 ms) | no change | none | none | no effect, drop from the setting |
| Tona queue priority | not tested | | | by the sources no effect on the compositor; needs an engine change on stream 0 |

The levers act on the whole process: the CU mask needs a server restart (it is an environment variable read before the HIP runtime starts), so it is a start-time setting of the
"work" mode, not a switch while a conversation runs. A finer step between 4 and 8 CUs (`TONA_FREE_CUS` takes multiples of 4) and the tail at 4 CUs were not examined.

## Limits

One canary client, 64 MiB, 20 ms period, NORMAL priority, no window and no swap chain: a stand-in, not the compositor. Two runs per arm, 40 s per phase. Context 140000, one prompt
for decode and one for prefill; no draft-heavy or long-context run. The earlier subjective result of 2026-10-23 (mouse not smoother with a CU mask) is one run of 45 s without a
measurement of the submission latency; this sweep does not contradict it, it measures something else.
