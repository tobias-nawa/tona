# 60 - GPU clock ramp after idle (issue #161)

Research 2026-11-16 plus one small measurement (`research/measurements/2026-11-16-clock-ramp.md`, probe `bench/micro/clock_ramp.hip`).
Marking: **[own]** our measurement, **[cited]** source read this session, **[ungeprueft]** not checked, with the check that would settle it.
Rates: no token rates here; the load is a synthetic matrix-core kernel (kernel times in ms).

## 0. Verdict

1. **The ramp is real and reproduced [own]:** after 2.5 s idle, five matrix-core kernels (hot 89 ms each) take 682 ms instead of 445 ms: **+237 ms**, kernel 1 at 1.75 GHz against 3.0 GHz hot
   (the prefill file measured about 190 ms). True idle is 13 W (22.9 W in the first table was a lagging reading, see section 5).
2. **Without root only one thing works: keep the engine busy for at least about 1 s before the request [own].** A one-block spin kernel on its own stream, started 1 s before the work,
   leaves kernel 1 at +10 % (98 ms) and removes 91 % of the penalty; with `s_sleep` in the loop +7 %. 600 ms removes 75 %, 300 ms 34 %, 100 ms nothing.
   A request cannot give that lead (the HTTP body arrives milliseconds before the prefill), so the 2026-11-12 attempt could not work. A warm-up needs a **session-scoped hold**: keep the clock up from the end of one
   response until the next request, while an agent session is active. **Cost: +17 W** (40 W at 3.15 GHz against 22.9 W idle) for as long as it is held [own].
3. **No effect [own]:** a one-thread kernel every 50, 200 or 500 ms for 4 s (idle power stays 23 W, kernel 1 +60 % as without), and 2 or 4 extra HIP streams touched once. The firmware reacts to the busy fraction, not to the presence of work or queues.
4. **No API gives a user the workload profile [cited]:** AMD SMI's performance-level setter needs root and writes `power_dpm_force_performance_level`; those sysfs files are `root:root 0644` here [own, `ls -l`]. I found no HIP/HSA/KFD call (no `hipExtSetPerfMode`-like function) that a normal process can use. `/dev/kfd` and `renderD*` are world-writable, but the profile is not an ioctl there [ungeprueft: a search of the KFD uapi header would settle it].
5. **Root test done (section 5): no installer root step.** COMPUTE does nothing; `high` removes the ramp but costs about 4 % sustained clock and +10 W idle.

## 1. Non-root mechanisms

| Mechanism | Effect on the ramp | Idle cost | Source |
|---|---|---|---|
| Spin kernel, lead >= 1 s, one block | kernel 1 +10 % (from +60 %), 91 % of the penalty gone | 29.5 W mean over the 1 s lead, about +27 W steady over true idle (40 W) | [own] |
| Spin with `s_sleep(20)` | the same or a little better (kernel 1 +7 %, hot 85 ms instead of 89 ms) | the same: 36 W after 3 s | [own] |
| Spin with lead 0.1 to 0.3 s | nothing / 34 % | | [own] |
| Tiny kernel every 50 to 500 ms | none | 0 | [own] |
| Extra HIP streams (more HW queues) | none | 0 | [own] |
| Stream priority | not tried; the 2026-11-12 file shows no effect of priority on the copy stream; the firmware ramp does not look at priority | | [own, prior file] |
| Compute-bound against memory-bound warm-up | not tried; the one-block spin already reaches 3.15 GHz, so busy fraction is enough | | [own] |
| `hipExtSetPerfMode`, rocm-smi / AMD SMI calls | root only [cited: AMD SMI docs, performance-control section]; no non-root HIP call found | | [cited] |
| sysfs `pp_*`, `power_dpm_*` | read-only for us: `root:root 0644` on `card1` (the 9060 XT, device 0x7590) | | [own] |

How long before the request: at least 1 s, at best 3 s (clock steady). The server knows a request is coming only at the HTTP body; for an agent session the better trigger is **the end of the previous response**:
hold until `now - last_request_end > T` with T from the tool-call pause distribution. The pause pattern of agent tool calls is not in our data [ungeprueft: the request log of `tona claude` sessions would give it];
the bimodal file shows that the clock decays within 0.5 s, so any gap longer than that pays the full penalty today.

Safety (AGENTS.md, GPU safety): a spin kernel must not be a long-lived kernel on a stream that something synchronises with. A `hipDeviceSynchronize` or an event on the device would wait for it,
and `k_mega` waits on the CPU inside the kernel. Use short chunks (about 250 ms, well under the 2 s driver limit) that a helper thread relaunches on its own stream, never a device-wide synchronisation, and do not run it while `k_mega` runs
(the clock is up then anyway). The 2026-11-12 file also showed that the number of HW queues moves prefill by 7 to 9 %; an extra stream for the spin changes the queue count, so every hold needs an A/B on prefill [ungeprueft].
Energy: holding 17 W over a 30 s pause is 510 J to save about 0.2 s at about 160 W (30 J). It is a latency trade, not an energy gain; scope it to active sessions only.

## 2. Mechanisms that need root (set once by `tona-install`)

State on this machine [own]: `power_dpm_force_performance_level` = `auto`; `pp_power_profile_mode` lists BOOTUP_DEFAULT (active), 3D_FULL_SCREEN, POWER_SAVING, VIDEO, VR, COMPUTE, CUSTOM, WINDOW_3D.
COMPUTE differs from BOOTUP_DEFAULT in the GFXCLK row (MinActiveFreqType 6 against 1, no 800 MHz booster) [own, file contents]; what that does to the ramp is not documented in the files I could read.

| Setting | What it does [cited: kernel amdgpu docs] | Idle cost | Side effects |
|---|---|---|---|
| `pp_power_profile_mode` = 5 (COMPUTE) | heuristics for compute; needs `power_dpm_force_performance_level` = `manual` first, per the kernel doc | **[ungeprueft]**; measure power1_average with the probe | `manual` takes over the clock choice from the auto policy; the kernel's own switching of workload profiles on submit [cited: amd-gfx Jan 2025 patch series, COMPUTE/FULLSCREEN3D with about 1 s idle timeout] no longer applies; desktop is on the same card |
| `power_dpm_force_performance_level` = `high` | forces the highest clocks | high: sclk pinned, idle power well above 22.9 W (the 3.15 GHz spin was 40 W with a tiny load, a held `high` with a display on this card will be higher) [ungeprueft] | the desktop (2 x 4K on the same card) runs at full clock all day; heat and fan noise |
| `profile_peak` | same family, disables clock and power gating | highest | the same, plus gating off |
| group-writable sysfs via udev (`chgrp`/`chmod` of the two files at `add`) | not a clock policy itself: lets `tona-server` (unprivileged) set COMPUTE only while a session is active and restore `auto` after | zero when idle | users of the group may force clocks (fan, heat, no data risk); a write of `manual` stops auto switching for everyone on the card until restored, so a crash must restore it (unit `ExecStopPost`) |

I did not use root and did not write to `/sys`.

Recommended order for the root experiment (a few minutes, with the probe): run `clock_ramp none 0 0 6 <hwmon> 60000` three times, as root once with
`echo manual > power_dpm_force_performance_level; echo 5 > pp_power_profile_mode`, and once with `echo high > power_dpm_force_performance_level`, then `echo auto` back. Compare kernel 1 and the idle power (`hold_w` is empty for `none`; read `power1_average` after 30 s idle).
Only a setting that gets kernel 1 within about 10 % at an acceptable idle power is worth an installer step.

## 3. Prior art

- **llama.cpp #24237** (PR, July 2026) [cited]: RX 9070 (gfx1201), ROCm 7.2.1: with `--mmproj` the idle server holds 3.2 GHz and 100 % utilisation. The PR sets `GPU_MAX_HW_QUEUES=1` by default on HIP builds, blaming the MES multi-queue behaviour; it was closed by its author on 2026-07-15 (merge not shown). The reporter's bug #23965 closed as fixed. Same symptom family as ours, opposite sign: there the clock does not drop, here it drops too eagerly.
- **llama.cpp #3929** [cited]: RDNA3, ROCm, 8 HIP streams per GPU keep the GPU at 100 % and high power while idle; workarounds `amdgpu sched_policy=2` (idle power low, console lags) and `GPU_MAX_HW_QUEUES=1`; closed as stale April 2024; also reported to freedesktop.
- **llama.cpp #21430** [cited]: RX 9070 XT (0x7550) with `--mmproj` stays at 100 % and 3400 MHz, 106 W (R9700, same gfx1201, idles at 15 W); maintainer: "kernel/GPU firmware bug"; closed 2026-04-05. Kernel 6.19.10, ROCm 7.2.1.
- **ROCm/TheRock #9297** [cited via #24237]: on gfx1201 a pending `hipStreamWaitEvent` adds about 17 us to small kernels; gone with `GPU_MAX_HW_QUEUES` 2 or less.
- **Linux 7.0 amdgpu: idle-power fix for RDNA4 after compute workloads** [cited: Phoronix headline, 2026-03-13; article not readable, details unchecked].
- **amd-gfx, Jan 2025 (ring helpers for the workload profile)** [cited: search summary; mail archive not readable]: the kernel switches to COMPUTE or FULLSCREEN3D when work is submitted, back after about 1 s idle. That is consistent with a ramp that needs time after the first submission, but whether it is the cause of the 550 ms plateau is **[ungeprueft]**.
- **amd-gfx, Feb 2025 (SMU 14.0.2/3 gfxoff delay for compute)** [cited]: proposed, objected to; gfxoff is disabled by the KFD compute-idle path already.
- vLLM and ollama reports for RDNA3/RDNA4: I found none about first-request clock ramp or workload profiles. The search results were benchmarks. **No report of our symptom exists in the trackers I could reach**, so this is not a known bug with a known fix.

Link to our result 3 (queue count moves prefill by 7 to 9 %): the llama.cpp issues say that MES treats several queues differently; our extra-queue test here changed neither the ramp nor idle power (22.9 W), so the two effects are separate: the queue effect sits in the dispatch, the ramp in the clock policy.

## 4. Recommendation for `tona-install`

No change in the installer on today's evidence. Order of work:

1. **the maintainer's decision, root, 2 minutes:** run the experiment of section 2 (COMPUTE after `manual`, and `high`) with the probe. Outcome A: COMPUTE removes the ramp at an idle cost he accepts: `tona-install` offers a udev rule that makes the two files writable for the `render` group, and `tona-server` sets COMPUTE while a session is active and restores `auto` (ExecStopPost). No permanent system-wide change, no idle cost between sessions. Outcome B: only `high` helps: not recommended as a default (permanent idle cost, desktop on the same card); offer as an opt-in "performance mode".
2. **Engine ticket (no root, only if +17 W is wanted while a session is active):** a session-scoped hold: after a response ends, a helper thread relaunches a 250 ms one-block `s_sleep` spin on its own stream until `last_request_end + T` (start with T = 20 s, from the pause data), stops at the next request start (the clock is then up by itself). Gate: A/B on prefill, with TONA_QUEUE_SPACERS interplay, power logged; expected gain about 190 ms of the first prefill after each pause (about 10 % of a 2000-token prefill at 1.35k tok/s) at +17 W only inside sessions.
3. Do not spend time on ping kernels or extra streams for the ramp: no effect [own].

Open, in order of value: (a) the root experiment; (b) the distribution of pauses between requests in real agent sessions; (c) whether the hold survives a prefill A/B without moving the queue effect.

## 5. Root test ( probe `none`, 6 trials each; other GPU jobs ran just before)

| Setting | kernel 1 | sustained kernel | clock | idle (30 s) |
|---|---:|---:|---:|---:|
| `auto` | 141 ms | 89 ms at 3.0 GHz | 1.77 GHz in kernel 1 | 13 W |
| `manual` + COMPUTE (5) | 141 to 177 ms | same as auto | 1.78 GHz | 13 W |
| `high` | 93 ms | 93 ms | 2.6 to 2.75 GHz steady | 23 W (+10 W) |

COMPUTE does not touch the ramp. `high` has no ramp but is about 4 % slower on long prefill (the clock does not reach 3.0 to 3.1 GHz) and costs +10 W all day with the desktop on the same card.
**Conclusion: no root step in `tona-install`.** This supersedes the udev recommendation of section 4, item 1. Correction: 22.9 W "idle" earlier was a lagging `power1_average`; the spin hold costs about +27 W over the true 13 W idle. Details: `research/measurements/2026-11-16-clock-ramp.md`.

## 6. The server hold, measured (2026-11-16/17)

Built: `TONA_CLOCK_HOLD` (off by default), rule in `engine/server/hold_policy.h` (hold = the largest observed pause whose hold-seconds per saved ramp are at most `TONA_CLOCK_HOLD_S_PER_SAVE`, default 20, over the last 32 pauses; none with fewer than 4 pauses). Details and tables: `research/measurements/2026-11-16-clock-ramp.md`, part 2.

| Question | Result |
|---|---|
| First answer after a pause (replay of 24 real agent turns, 9600X) | time to first token -103 ms mean, -121 ms median over the 12 turns after the rule had data; -71 / -57 ms over all 20 comparable turns |
| Energy over the 216 s replay | +2.7 kJ (+21 %), about +33 W while the hold runs (probe: 40 W against 13 W idle) |
| Does it keep the 7 to 9 % queue effect? (ab.py, 9950X, mode 1, spin on the spare queue) | yes: prefill@2000 -0.11 % [-0.68, +0.46], equal; decode and prefill-write within -0.5..+0.5 % |
| Own stream (mode 2, 9600X, disturbed by another job's build) | prefill@2000 +1.92 % [+0.65, +3.20]; not yet measured on the 9950X |

Decision: the end user decides, not a default. The hold stays off; `tona-install` asks "Keep the GPU warm between requests" with the cost (about +27 W while held) and the gain (about 0.1 s per answer after a pause), default no; the engine switch is the mechanism. The recommendation of section 4 (a root step) is withdrawn by the root test (section 5).
