# Mac: TONA_PRIORITY as built, a check against the reserve run (2026-11-25)

## Plan (written before the run)

- **Question**: does the built setting reproduce the reserve run of 2026-11-17 (`2026-11-17-mac-iq1m-reserve.md`) in
  the developer start, and what does it do in the installer start (`TONA_RAM_GB` set), where the reserve without the
  user's 2 GB is 3.0 GB with `work` and 1.5 GB with `tona`? The 1.5 GB case was never measured.
- **Arms** (Coder IQ1_M, its profile `toolchain/profiles/coder-iq1_m.bin`, MTP head `mtp-256`, `--ctx 32768`, default
  arena, governor off; build `work/build/engine` of this branch):

  | Arm | Environment | Reserve |
  |---|---|---|
  | D-bal | none (developer start, `TONA_PRIORITY` unset = balanced) | 5.0 GB |
  | D-tona | `TONA_PRIORITY=tona` | 3.5 GB |
  | I-work | `TONA_RAM_GB=15 TONA_PRIORITY=work` (the installer's split of 32 GB for work) | 3.0 GB |
  | I-tona | `TONA_RAM_GB=24 TONA_PRIORITY=tona` (the installer's split for tona) | 1.5 GB |

- **Scope**: one round per arm in this order, the phases of the reserve run (45 s from launch, 60 s idle, 60 s of
  200-token requests back to back), 20 s between arms; about 16 min. A control, not an A/B: one round each.
- **Expectation** (from the reserve run; decode tolerance 5 %, the server code changed since but not the memory path):
  - D-bal: budget 8.65 to 8.85 GB (plus or minus 0.3), other programs' compression during requests under 50 MB,
    decode 15.9 to 17.6 tok/s.
  - D-tona: budget 10.03 to 10.27 GB (plus or minus 0.3), 28 to 525 MB compressed, decode 18.1 to 21.0 tok/s.
  - I-work: the start budget 0.5 GB above D-tona (10.5 to 10.8 GB) unless the limit (15 GB less 1.5 GB less what
    the process holds) is lower; compression like D-tona or more (28 to 700 MB); decode 18 to 21 tok/s.
  - I-tona: about 12.0 GB, compression above D-tona, decode 19 to 22 tok/s.
- **Decision rule**: the setting is right for the developer start if D-bal and D-tona fall inside their ranges. In the
  installer start `tona` keeps its 1.5 GB reserve only if I-tona compresses no more than twice D-tona's top
  (1050 MB per minute of requests) and swaps nothing out; otherwise the installer start keeps at least 3.0 GB and
  `tona` gives nothing there.
- **Guard**: a shell loop every 5 s stops the server when the data disk falls under 8 GB free or swap rises 2 GB
  above its level at the start.
- **Active tooling**: the sampler (one Python process at 20 Hz, machine-wide compressions and swap-outs from
  `host_statistics64`), the guard, stop sleep and keepalive defaults; no other `TONA_*` variable.

## Result (2026-11-26, 00:31 to 00:45)

Quiet check before (00:31:00) and after (00:45:29): **do not measure**, other programs used 29.3 and 21.6 % of all
CPU threads (load 8.2 and 6.9): the first night after a restart, with the system's idle maintenance running. Before
the run a backup was stopped, an idle virtual machine closed and the display turned off; the maintainer then said to
measure at once rather than wait. The guard never stopped the run (free disk 172 to 170 GB; swap 0 to 0.7 GB, all of
it in arm I-tona).

| Arm | Reserve (`kept`) | Budget (at start) | Compressed machine-wide: start 45 s / idle 60 s / requests 60 s | Swapped out in requests | Decode median (range, requests) |
|---|---|---|---|---|---|
| D-bal | 5.0 GB | 10.77 GB (10.77) | 207 / 1864 / 7081 MB | 0 MB | 21.5 tok/s (20.2 to 21.6, 7) |
| D-tona | 3.5 GB | 12.29 GB (12.29) | 1973 / 2251 / 2324 MB | 0 MB | 25.7 tok/s (24.5 to 25.9, 9) |
| I-work | 3.0 GB | **9.53 GB** (12.85) | 556 / 0 / 2764 MB | 0 MB | 17.9 tok/s (17.3 to 18.6, 6) |
| I-tona | 1.5 GB | 13.54 GB (13.54) | 3986 / 1120 / 3376 MB | **742 MB** | 28.7 tok/s (27.2 to 28.9, 10) |

Read against the plan:

- **The setting does what it was built for.** The reserve is 5.0, 3.5, 3.0 and 1.5 GB as designed; in the developer
  start `tona` gives the arena 1.52 GB more and decodes 20 % faster (25.7 against 21.5 tok/s; 17 % on 2026-11-17).
- **Outside the expectation (a finding): the budgets.** D-bal got 10.77 GB, not 8.65 to 8.85: after the restart the
  other programs held 12.7 GB of 34.4 GB at the server's start, less than on 2026-11-17, and the budget is taken from
  what is free at start. Decode follows the budget (21.5 against 16.7 tok/s then).
- **Outside the expectation: compression.** Even idle arms compressed 1.1 to 2.3 GB per minute with the server doing
  nothing (D-bal, D-tona, I-tona; on 2026-11-17 idle was 0 MB in all six). The machine-wide counter was dominated by
  the background load and by programs growing after the arena had taken its budget at start; the compression numbers
  of this run cannot tell the arms apart and do not replace the 0 against 28 to 525 MB of 2026-11-17.
- **I-work: the installer's work start is bounded by its RAM limit, not by the 3.0 GB reserve.** `TONA_RAM_GB=15`
  cut the budget from 12.85 to 9.53 GB, below D-bal's: a release install with work keeps more room for other programs
  than a developer start, whatever the reserve says. Decode 17.9 tok/s.
- **I-tona swapped out 742 MB during requests**, the only arm to swap. Decision by the plan's rule: in the installer
  start `tona` gives nothing; the reserve stays at least 3.0 GB there. With the RAM limit of 24 GB the installer's
  tona split already gives the engine 9 GB more than work, so the 1.5 GB are not needed there.
- **A control run, not a measurement of speed**: quiet check "do not measure", one round per arm. The 20 % are a
  direction, no number for the README, the installer or BENCHMARKS comparisons; the speed numbers stay those of
  2026-11-17 (three rounds).

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
- Load average (1, 5, 15 min): 8.21, 6.12, 7.55
- CPU use by other programs over 0.5 s: 25.6 % of all threads
- sandbox: no
- Commit: `c5bccb8b` (dirty tree)
```

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
- Load average (1, 5, 15 min): 6.93, 6.89, 7.1
- CPU use by other programs over 0.5 s: 19.0 % of all threads
- sandbox: no
- Commit: `c5bccb8b` (dirty tree)
```
