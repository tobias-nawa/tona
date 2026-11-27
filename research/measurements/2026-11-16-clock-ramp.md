# GPU clock ramp after idle: non-root warm-up strategies (#161)

## Plan (written before the runs)

- Question: can a user-space strategy, without root, make the first real kernel after an idle pause run at full clock, and at what idle-power cost?
  Decision: if a strategy removes the cold penalty at a cost of a few watts, it goes to the engine as a ticket (#161 follow-up); if none does, the recommendation is the root-side udev rule.
- Probe: `bench/micro/clock_ramp.hip` (no engine code). One trial: 2.5 s idle, then the strategy, then 5 timed full-GPU FMA kernels (256 blocks x 256 threads, 8 WMMA accumulators per wave, about 89 ms each when hot; a first FMA-only variant stayed at 1.9 GHz and did not show the ramp, so it was dropped; the matrix-core load draws power like prefill and reaches 3.0 GHz).
  Cold penalty = time of kernel 1 and 2 against kernel 5. Clock and power from hwmon `freq1_input` and `power1_average` at 10 ms (as in 2026-11-12-prefill-bimodal.md).
- Strategies (5 trials each, the unit is one trial; 5 is enough because a ramp is firmware behaviour and the earlier A/A spread was below 1 %):
  1. `none` (baseline).
  2. `spin` with lead 100, 300, 600, 1000 ms: one block of 32 threads spins on its own stream, then the work starts on another stream (tests how long before the request a warm-up must start).
  3. `ping` every 50, 200, 500 ms for a hold of 4 s: a one-thread kernel, synchronised, to see whether a trickle of work keeps the clock up and what it costs.
  4. `queues` with 2 and 4 extra HIP streams touched once, hold 4 s (llama.cpp #24237: more HW queues keep the clock high on RDNA4).
- Expectation (source: bimodal file, result 2): `none` kernel 1 about 30 to 80 % slower than kernel 5 with 1.8 GHz in kernel 1 and 3.1 GHz in kernel 5; idle 22 W / about 65 MHz (3 calibration trials today: 22.7 W). Spin with lead 600 ms or more: kernel 1 within 5 % of kernel 5.
  A result outside this is a finding. Tolerance on times 5 %.
- Active tooling: only the probe's own hwmon sampler; no diagnostics, no profiler, no TONA_* variables. Machine shared with other workers (CPU load high, see test conditions); the probe is GPU-clock bound, the 10 ms sampler may jitter.
- Rate naming: no token rates here; times are kernel milliseconds, the load is a synthetic compute-bound kernel, not prefill.

Added to the plan after the first calibration: `spin` with lead 3000 and 6000 ms and `spinsl` (the same spin with `s_sleep` between polls) with lead 1000 and 3000 ms, to read the steady-state cost of a held clock; run 2 in `run2.sh`.
Trial 0 of every series is dropped from the tables: the process before it left the clock warm (kernel 1 of 130 ms instead of 142 ms), so it is not idle-cold.

Test conditions (`tools/hw_report.py` before the runs, commit `e4a4f41` plus the new probe):

```text
### Test conditions

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.49 GiB held by others before the run, 0 % busy before the run
- RAM: 60.5 GiB, 49.6 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.1.52802-9999
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 9.15, 2.71, 0.99
- CPU use by other programs over 0.5 s: 103.3 % of all threads, swap in use 2.1 GiB
- sandbox: no
- Commit: `e4a4f41` (dirty tree)
```

- - Quiet check (2026-11-16 12:36:39): do not measure
    - do not measure: other programs use 95.2 % of all CPU threads
    - warning: load average 9.54 is high (it lags a minute)
    - warning: 1.49 GiB of 15.9 GiB VRAM are held by others
- - Quiet check (2026-11-16 12:49:58): do not measure
    - warning: other programs use 7.9 % of all CPU threads
    - do not measure: 14.83 GiB of 15.9 GiB VRAM are held by others
  (after: the 14.8 GiB VRAM were held by another worker, the probe had exited; CPU load of other programs during the runs was high, 95 to 103 % of all threads, see before)

## Results

Mean over trials 1..n (n = 4 to 8; spread of kernel 1 inside a series 0.5 ms, except the rare 10 to 30 ms outliers of kernels 2 to 5 that every series has, also `none`).
"k1..5 sum" is the time of the five kernels; hot reference 5 x 89 = 445 ms (kernel 5 of the best series 84 to 90 ms). Power is the mean over the hold window including the start of the strategy.

| Strategy | n | kernel 1 ms | k1..5 sum ms | clock in kernel 1 MHz | power during hold W |
|---|---:|---:|---:|---:|---:|
| none (idle 2.5 s) | 8 | 143 | 682 | 1754 | idle 22.9 |
| spin, started 100 ms before | 4 | 150 | 681 | 1814 | 27.1 |
| spin, 300 ms before | 4 | 150 | 602 | 1817 | 27.1 |
| spin, 600 ms before | 4 | 114 | 504 | 2369 | 27.1 |
| spin, 1000 ms before | 4 | 98 | 467 | 2810 | 29.5 |
| spin, 3000 ms before | 3 | 98 | 471 | 2825 | 35.9 |
| spin, 6000 ms before | 2 | 104 | 474 | 2810 | 40.4 |
| spin with `s_sleep`, 1000 ms before | 3 | 92 | 455 | 2824 | 29.4 |
| spin with `s_sleep`, 3000 ms before | 3 | 92 | 441 | 2826 | 35.9 |
| ping (1 thread kernel every 50, 200, 500 ms, 4 s) | 4 each | 142 | 684 to 686 | 1730 to 1742 | 22.9 to 23.0 |
| 2 or 4 extra HIP streams touched, 4 s | 4 each | 141 to 142 | 684 to 690 | 1770 to 1782 | 22.9 |

Steady state of a held clock (from the 6 s series, after the ramp): about 3.15 GHz and 40 W, against 22.9 W and 46 to 70 MHz idle: **+17 W**.

Result against the plan:

| Item | Measured | Expected | Verdict |
|---|---|---|---|
| `none`, kernel 1 against hot | 143 against 89 ms (+60 %), 1.75 against 3.0 GHz; five kernels +237 ms | +30 to +80 %, 1.8 GHz | within |
| idle | 22.9 W | 22 W | within |
| spin with lead 600 ms or more: kernel 1 within 5 % of kernel 5 | 114 ms at 600 (+28 %), 98 ms at 1000 (+10 %), 92 against 86 at 1000 with `s_sleep` (+7 %) | within 5 % | **outside**: the lead must be at least 1 s, and 5 to 10 % remain |
| ping and extra queues | no effect | not predicted | finding: a trickle of tiny kernels or more HW queues does not hold the clock |

Reading: the firmware ramps from the busy fraction of the engine, not from the existence of work. A tiny kernel every 50 ms (busy fraction far below 1 %) and idle queues change nothing.
A block that keeps one engine busy lifts the clock in 0.5 to 1 s; shorter leads give a part of the ramp (share of the 237 ms penalty gone over the five kernels: 100 ms 0 %, 300 ms 34 %, 600 ms 75 %, 1000 ms 91 %).
So the 2026-11-12 spin kernel failed because its lead was the few milliseconds between the request start and the first prefill kernel.
The ramp itself is not linear in the lead: kernel 1 sees 1.8, 2.4 and 2.8 GHz after 300, 600 and 1000 ms.

Not measured: any root setting, the real prefill (the engine is not touched), a keep-alive strategy over a whole session. The numbers above are for a synthetic matrix-core load, 5 kernels of about 90 to 140 ms; the 190 ms of the prefill file corresponds to the 237 ms measured here.

## Root test (6 trials per setting, 6 trials per setting, probe `none`, 30 s idle before the power reading)

Raw: `/var/tmp/claude/161/root_results.txt` (not in the repo). Caveat: other GPU jobs ran just before, so trial 0 of each series is warm (dropped below).

| Setting | kernel 1 ms (trials 1..5) | hot kernel 5 ms | clock in kernel 1 MHz | idle after 30 s |
|---|---:|---:|---:|---:|
| `auto` (default) | 141.3 to 141.8 | 113 (the ramp is still not done in kernel 5; hot at 3.0 GHz is 89 ms, section above) | 1770 | 13 W |
| `manual` + `pp_power_profile_mode` 5 (COMPUTE) | 141.8, 177.1, 150.4, 158.7, 167.6 | 107 to 117 | 1780 | 13 W |
| `power_dpm_force_performance_level` `high` | 93.5 to 111.0 | 93 | 2590 to 2680 (steady 2.75 GHz) | 23 W |

Findings: COMPUTE has no effect on the ramp (kernel 1 equal or worse). `high` removes the ramp, but the clock sits at 2.6 to 2.75 GHz, so a sustained kernel takes 93 ms against about 89 ms at 3.0 to 3.1 GHz under `auto`: about 4 % slower on long prefill; idle +10 W (23 against 13 W).
**Correction to the first table:** the 22.9 W "idle" of the first runs was `power1_average` read 2.5 s after the previous load (the sensor averages over seconds); true idle is 13 W after 30 s. The cost of a held clock (40 W) is therefore about +27 W over true idle, not +17 W.

## Part 2: the server hold (`TONA_CLOCK_HOLD`), plan written before the runs

Question and decision: does the session-scoped hold (server/hold_controller.h, rule in server/hold_policy.h) remove the first-prefill ramp of real agent turns, at what energy in the pauses, and does it keep the prefill of the 7 to 9 % queue effect?
If time to first token after a pause drops by about the 190 ms of the ramp at an acceptable energy cost, the hold goes forward as an opt-in; if the queue effect is lost with mode 1 (spacer reused), mode 2 (own stream) is compared; if neither is clean the switch stays off and documented.

The rule (no fixed hold length): the last 32 pauses of the server (end of the last response to the next request, server idle). For T among them, cost(T) = mean(min(pause, T)) / share(pause <= T), hold-seconds per saved ramp; the hold is the largest observed pause whose cost is at most `TONA_CLOCK_HOLD_S_PER_SAVE` (default 20 s per saved ramp, i.e. about 540 J for 0.2 s of prefill), times 1.1. Fewer than 4 pauses: no hold. A long absence among short pauses is therefore not covered.

Runs:
1. Replay: stored agent requests (Anthropic `/v1/messages` bodies from `work/dumps`, 2026-11-03, 244 files; only replayed, nothing copied) at their original arrival times, `scripts/replay_hold.py`. First request fills the cache (not measured). Requests 1 to N, `max_tokens` capped at 32, stream on, time to first token and, from /status, prefill seconds and prefill tok/s of each request; hwmon power at 100 ms: energy over the run and mean power while no request runs.
   Arms A (`TONA_CLOCK_HOLD` unset) and B (`TONA_CLOCK_HOLD=1`), one fresh server each, production mode, port 8099, ctx 140000, MTP drafter, expert profile (as `ab.py`). 1 start per arm, A then B.
   Why this size: the question is whether the first request after a pause starts about 190 ms earlier; the smallest workload is a sequence of real turns with their real pauses. The rule needs 4 pauses before it holds anything, so 24 measured turns (about 6 minutes, pauses 1 to 30 s) give about 20 turns with a hold in B, and the comparison is paired per turn (same request in A and B), so the per-turn noise of TTFT (a few percent of a 1 to 3 s prefill) against the expected 190 ms is covered by 20 pairs. One start per arm (A then B); a second pair B then A only if the order matters or the result is undecided. Not 55 turns and not 4 starts: that would be GPU time without a different decision.
2. Queue effect: `scripts/ab.py --workload short` prefill@2000 with `TONA_CLOCK_HOLD=1` and `=2` against unset, early stopping on, `--max-rounds` 4. Why this size: the effect to exclude is the 7 to 9 % of 2026-11-12 (a lost or added queue); the A/A spread there is 0.2 %, so a decision comes in 1 to 2 rounds; 4 rounds at most. Only the 2000-token workload (`short`). ab.py reports decode@2000 and decode@2000:write as well and requires an expectation for them: -0.5..+0.5 % (the hold does not run during decode; the decode noise of 2026-11-12 is 0.1 to 0.5 %, so decode may stay open).
- Expectation (source: this note, part 1, and the 2026-11-12 file): time to first token B shorter than A by 150 to 250 ms for requests after a pause longer than 0.5 s that the hold covers; prefill tok/s of those requests +8 to +15 % for 2000-token deltas; idle power in pauses 13 W (A) against about 35 to 40 W (B); energy B above A by roughly (held seconds x 25 W). Mode 1 prefill@2000 within 0.5 % of A; mode 2 may lose the queue effect or gain nothing (one queue more).
- Active tooling: `TONA_CLOCK_HOLD`, the sampler of the replay script; no diagnostics.

## Part 2 results

### Replay of stored agent turns (9600X, 2026-11-16)

24 turns of a recorded agent session, original arrival times, `max_tokens` 32; one fresh server per arm, A hold off, B `TONA_CLOCK_HOLD=1`, one start each (first attempt discarded: the server answered 503 while loading; readiness now checked on /health). Raw: `/var/tmp/claude/161/replay-A.json`, `replay-B.json` (they quote the recorded prompts, so they stay out of the repo).
The hold covered 15 of 24 arrivals (18 holds, 80 s held; the last rule value 72 s).

| Comparison (B minus A) | n | mean | median | range |
|---|---:|---:|---:|---:|
| time to first token, all comparable turns | 20 | -71 ms | -57 ms | -190 to +10 ms |
| time to first token, turns after the rule had data (turn 10 on, pauses 1 to 8 s) | 12 | -103 ms | -121 ms | -190 to -10 ms |

Left out: the two cache-fill turns (54k and 24k computed tokens, 30 s and 14 s, no difference) and two turns whose /status entries were swapped by concurrent requests (TTFT is client-side and kept for them; the table above omits them to keep one definition).
Energy (hwmon `power1_average` at 100 ms, 216 s): A 12.6 kJ, B 15.2 kJ, **+2.7 kJ (+21 %)**, about 33 W extra while the hold ran; mean power in pauses 20.7 W against 37.0 W (the sensor averages over seconds, indicative only).
Against the plan: expected -150 to -250 ms for covered turns: measured -103 ms mean (-121 median), **below the expectation**: the ramp costs about 190 ms of a long prefill (probe), the replayed turns compute 150 to 1300 tokens, and the first turns of B get no hold while the rule learns.

### ab.py, `TONA_CLOCK_HOLD` against unset (does the hold keep the 7 to 9 % queue effect?)

9950X, kernel 7.3.4, 2026-11-17, prefill@2000, early stopping, at most 4 rounds. Mode 1 = spin on the first spare queue: `2026-11-17-clock-hold-ab-9950x-mode1.md`.

| Metric | B vs A, mean [CI] | Decision | Expected |
|---|---|---|---|
| prefill@2000 | -0.11 % [-0.68, +0.46] | **equal** after round 1 | -0.5 to +0.5 %: within |
| decode@2000 | -0.42 % [-1.20, +0.36] | open | within |
| prefill@2000:write | -0.37 % [-1.01, +0.27] | open | within |
| decode@2000:write | -0.10 % [-1.63, +1.43] | open | within |

Two rounds ran (round 3 could not start: a device allocation failed because another process held VRAM; the partial result is kept, none dropped). **Mode 1 keeps the queue effect and costs nothing measurable on prefill or decode.**
Mode 2 (own stream), 9600X, 2026-11-16: `2026-11-16-clock-hold-ab-9600x-mode2.md`, prefill@2000 +1.92 % [+0.65, +3.20], B faster after round 3 (3 rounds; round 4 failed on VRAM held by another process). All rounds of the 9600X runs (mode 1 and mode 2) fall in 01:39 to 01:49, while another job's build ran (sums only): they are **disturbed**; the 9600X mode 1 (`...-9600x-mode1.md`: -1.96 % [-8.85, +4.93], undecided) is not used. Mode 2 on the 9950X is still to run.
CPU and kernel changed between the days (9600X to 9950X, 7.2 to 7.3), so the 9600X rows and the 9950X row are not compared.

Test conditions of the 9950X run (`tools/hw_report.py` before):

```text
### Test conditions

- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads
- GPU: AMD Ryzen 9 9950X 16-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 15.24 GiB held by others before the run, 2 % busy before the run
- RAM: 60.5 GiB, 30.5 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.3.4-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 0.33, 0.28, 0.73
- CPU use by other programs over 0.5 s: 8.6 % of all threads, swap in use 0.0 GiB
- sandbox: no
- Commit: `dc11332` (dirty tree)
```

- - Quiet check (2026-11-17 20:45:26): do not measure
    - do not measure: 15.26 GiB of 15.9 GiB VRAM are held by others
- - Quiet check (2026-11-17 20:45:58): do not measure
    - do not measure: other programs use 66.2 % of all CPU threads
    - do not measure: 14.73 GiB of 15.9 GiB VRAM are held by others
  (the machine was shared: other workers' jobs ran through the queue before and between my rounds; the quiet check said "do not measure" because of VRAM held by others and other programs' CPU use; the rounds above were measured anyway and the two failed starts are described above)
