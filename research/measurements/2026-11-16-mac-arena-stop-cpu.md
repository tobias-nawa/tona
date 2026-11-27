# Mac: where the arena's CPU time goes, and the stop wait (2026-11-16, M1 Max 32 GB)

The second DM-T arena A/B (2026-11-16-mac-dmt-arena-ab.md) showed our processes at 73 to 84 % of a core with the
arena, 6 to 7 % without it.

## Profile (no plan needed: a profile, no comparison)

tona-server, DM-T + MTP-164, arena auto with the profile `dmt-12.bin`, `TONA_METAL_ARENA_GOVERN=1`; four requests of
250 tokens, `sample` over 8 s of the third.

| | |
|---|---|
| CPU per request | 92.3, 83.0, 83.2 % of a core; idle 0.2 % |
| engine thread, share of the pass time in the spin `wait_flag` (runner_cold.mm) | **97.6 %** (2632 of 2698 samples) |
| of that in `steady_clock::now()` of the deadline check | 95 % |
| `waitUntilCompleted` (blocks) | 2.4 % |
| `read_cold`, `prefetch` on the engine thread | about 0 |
| read workers (`pread`) | in the kernel waiting for the SSD |

The CPU time is the spin at the stops, not the cold reads nor the lookahead: throttling the lookahead would not lower
it. Per stop the GPU needs 1.5 to 2.1 ms to the next stop (arena log), the reads 0.06 to 0.22 ms.

The sampled request failed: "the GPU did not reach an arena stop within 4 s" (a wall-clock deadline; `sample` holds
the process). The same error ended a check under memory pressure (2026-11-16-mac-arena-fill-steps.md).

## The stop wait (built)

- The wait gives up only as `stop_wait.h` decides (unit test `metal_host`): on the command buffer's error (macOS's
  watchdog, about 5 s), on a buffer that completed without the stop, or after 10 s in which the waiting thread itself
  ran. A wait over 1 s logs the thread's CPU time, the page-ins and the flag.
- The clock and the buffer status are read every 1024 rounds of the spin instead of every round.
- Checks: tona-check DM-T (`TONA_DETERMINISTIC=1`, "def quicksort(items):", -n 64, arena with profile and governor)
  identical to the run without the arena, 69 of 69 steps, the process held three times for 6 s during it. Server held
  for 6 s 3 s into a 250-token request: the request completed (17.9 s against 9.3), no error.
- Not reproduced: the case under memory pressure; it waits for the maintainer's consent to displacing runs. The log
  line of a long wait will tell whether it was paging.

## A/B: sleeping before the spin (plan)

- Scope: `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, DM-T + MTP-164 in both arms, same build;
  both arms with the arena (auto, profile `dmt-12.bin`, `TONA_METAL_ARENA_GOVERN=1`, so the fill takes no other
  program's memory); B adds `TONA_METAL_STOP_SLEEP=0.5`. Rounds 2 to 6, alpha 0.05, margin 0.5 %, load log on,
  backup wait and repeat as in ab.py, compressor watcher on, nothing closed. About 35 to 106 min.
- Expectation (B against A): decode-code, decode-text, decode@8000 -1 to +1 % (a sleep of half the expected wait,
  approaching the GPU's arrival from below); prefill@8000 -1 to +1 % (prefill stops wait in waitUntilCompleted, not
  in the spin). CPU ours: A 73 to 84 % of a core as in the second DM-T A/B, B 35 to 55 % (half of each stop asleep).
- Decision: the switch goes on by default if decode stays within -1 % and the CPU falls; outside, a finding.

Test conditions before:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.61, 3.63, 3.61
- CPU use by other programs over 0.5 s: 5.6 % of all threads
- sandbox: no
- Commit: `b4a72c4` (dirty tree)
```

```
- Quiet check (2026-11-16 05:33:01): warning
  - warning: load average 2.61 is high (it lags a minute)
```

## Result of the sleep A/B

Raw data and tables: [2026-11-16-mac-stop-sleep-server-ab.md](2026-11-16-mac-stop-sleep-server-ab.md) (`.json`).
6 rounds (4 more repeated: system backups), 144.6 min; answers identical in every start (SHA 727ff658).

| Metric | A median | B median | B vs A, mean [CI] | Rounds 2 to 5 | Plan |
|---|---|---|---|---|---|
| decode-code | 31.1 | 30.8 | -0.39 % [-4.96, +4.18] | -1.3 to -4.2 % | -1 to +1 % |
| decode-text | 20.0 | 20.0 | +0.05 % [-5.02, +5.12] | -1.5 to -2.9 % | -1 to +1 % |
| decode@8000 | 19.4 | 19.1 | -0.70 % [-4.62, +3.22] | -2.3 to -3.0 % | -1 to +1 % |
| prefill@8000 | 162.1 | 161.7 | +0.15 % [-1.14, +1.45] | -0.1 to -0.6 % | -1 to +1 % |
| CPU ours, % of a core | 76 to 81 (median 78) | 42 to 48 (median 44) | -44 % | | B 35 to 55 |

- **CPU as planned:** sleeping half of each stop's expected wait takes 34 points of a core off our processes.
- **Decode is undecided, and the means hide a loss.** Rounds 2 to 5 put B 1.3 to 4.2 % behind A in every decode
  metric; round 6 alone (+6 to +9 %) pulls the means to zero, and its A start was disturbed (857 MiB swapped in,
  memory level 8, the lowest of the run). Without round 6 the loss is about 2.5 %, outside the plan: a finding.
  Likely cause: the sleep overshoots the GPU's arrival at some stops (the expected wait is a mean, and a sleep
  wakes late by the scheduler's granularity), and every late wake delays the GPU's next layer.
- **Decision:** the switch stays off by default. A 2.5 % decode loss counts; the CPU saving is the trade, and that is
  a maintainer decision. A smaller factor (0.25) or a sleep capped below the shortest recent wait would trade less;
  not measured.

Test conditions after:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.82, 2.31, 2.16
- CPU use by other programs over 0.5 s: 3.3 % of all threads
- sandbox: no
- Commit: `77b2b62` (dirty tree)
```

```
- Quiet check (2026-11-16 08:02:30): warning
  - warning: load average 2.89 is high (it lags a minute)
```

## Result of the quarter-sleep A/B

Raw data and tables: [2026-11-16-mac-stop-sleep25-server-ab.md](2026-11-16-mac-stop-sleep25-server-ab.md) (`.json`).
Build `da7793b` (flags tested at-or-past, sleep from the pace of the stop waits). B sleeps a quarter of each stop's
expected wait (`TONA_METAL_STOP_SLEEP=0.25`). 6 rounds (3 more repeated: system backups), 168.7 min; answers
identical in every start (SHA 727ff658).

| Metric | A median | B median | B vs A, mean [CI] | Plan |
|---|---|---|---|---|
| decode-code | 31.5 | 31.9 | +2.13 % [-1.85, +6.12] | -1 to +1 % |
| decode-text | 20.4 | 20.8 | +1.60 % [-3.62, +6.83] | -1 to +1 % |
| decode@8000 | 19.8 | 20.0 | +0.91 % [-1.82, +3.63] | -1 to +1 % |
| prefill@8000 | 162.4 | 162.3 | -0.28 % [-1.51, +0.96] | -1 to +1 % |
| CPU ours, % of a core | 74 to 79 (median 76.6) | 56 to 62 (median 57.6) | -19 points | B 55 to 65 |

- **CPU as planned:** a quarter of each wait asleep takes 19 points of a core off our processes.
- **No decode loss.** Every decode mean is on the plus side, outside the plan on the good side but undecided (each
  CI holds zero). B's rounds spread less than A's (decode-code 31.3 to 32.4 against 29.5 to 32.4). Unlike the
  half-sleep run there is no streak of rounds with B behind: rounds 5 and 6 are -1.5 to 0 %, rounds 1, 3 and 4 plus.
- **The sleep does not overshoot the GPU:** the sleep counters of B's last start (70000 sleeps) plan 0.39 ms and
  sleep 0.59 ms (the scheduler wakes 0.2 ms late), yet the flag was there on waking at only 0.2 % of the stops.
- **Decision:** none here; the default is a maintainer decision (CPU saving at no measured decode cost).

Test conditions after:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.45, 2.25, 2.1
- CPU use by other programs over 0.5 s: 3.9 % of all threads
- sandbox: no
- Commit: `da7793b` (dirty tree)
```

```
- Quiet check (2026-11-16 11:20:53): warning
  - warning: other programs use 8.9 % of all CPU threads
```
