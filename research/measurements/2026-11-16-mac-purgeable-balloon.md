# Mac: balloon against the purgeable arena during an active session (2026-11-16)

## Plan (written before the run)

- **Question** (one decision): during a session the arena's blocks stay non-volatile for the keepalive window
  (research/53-mac-purgeable-arena.md, "First checks"). Does a memory pressure event make the unused ones volatile
  fast enough that macOS takes them before it compresses other programs' memory? If other programs' memory is
  compressed, the non-volatile window is wrong and the blocks must go volatile sooner.
- **Scope**: one tona-server (DM-T + MTP-164, `--ctx 32768`, arena auto, profile `dmt-12.bin`, governor,
  `TONA_METAL_ARENA_PURGEABLE=1`, build after `f904f86` with the pressure log lines) answers one 200-token coding
  request in a loop (`temperature` 0). 120 s without balloon, then `tools/mem_balloon.py --rates 1 --targets 6,10
  --phase-s 240 --hold-s 30 --rest-s 30 --limit-s 600 --swap-brake-gb 2` (grows at 1 GB/s to 6 and 10 GB
  alternately, holds 30 s, releases), then 120 s without. About 10 min, one run.
- **Safety**: the balloon ends itself after 600 s, on critical pressure or on swap +2 GB; every /health wait has a
  10 min deadline. Maintainer's consent for the balloon: 2026-11-16. Commit and push before the run.
- **Logged**: per request start, end, decode, answer hash, error; per second the machine's compressor (stored and
  occupied bytes), free pages, swap, and the server's own compressed bytes (`top` CMPRS); the server's pressure events
  and the arena's pressure episodes with the wall clock; the balloon's own per-second log.
- **Read out**: other programs' compression = rise of the compressor's stored bytes minus the rise of ours, from the
  balloon's start to its end; time from the first warn event to the arena's "volatile at once" line; requests failed;
  answer hashes equal; decode during the balloon against before.
- **Expectation** (source: the probe of 2026-11-16, `2026-11-16-mac-purgeable-probe.md`: a 10 GB balloon took 1.5 GB of
  volatile blocks within ~1 s, compressed 0.00 GB):
  - a warn event while the balloon grows; the arena marks its unused blocks volatile within one pass or 250 ms;
  - other programs' compression 0 to 0.5 GB over the run (the machine compresses a little at all times);
  - no failed request, all answers equal; decode during the balloon lower (re-read blocks), recovered after.
  - **A rise of other programs' compression above 0.5 GB is the finding** that decides against the window.
- **Active tooling**: `TONA_METAL_ARENA_GB=auto`, `TONA_EXPERT_PROFILE`, `TONA_METAL_ARENA_GOVERN=1`,
  `TONA_METAL_ARENA_PURGEABLE=1`; stop sleep default (0.25); keepalive default (500 ms, 180 s window); `top` sampling
  the server once a second.

Test conditions before: see after (unchanged hardware); quiet check 21:12:49: warning, other programs 15.8 % of all
CPU threads, load average 5.21. Swap in use 20.0 of 20.5 GB, the VM volume 13 GiB free (a watcher ended the balloon
below two swap files' worth, 2 GiB; it never fired).

## Result

Logs: [-balloon.log](2026-11-16-mac-purgeable-balloon-balloon.log) (the balloon's), [-server.log](2026-11-16-mac-purgeable-balloon-server.log)
(the arena's lines). Build `edeadc6`.

**The balloon braked after 1.5 s, at 1.1 GB:** critical memory pressure, swap +0.81 GB in that second. The pressure
level was already warn (2) at the balloon's first sample. The question of the plan (other programs' compression
during the balloon) has no answer from 1.5 s; what decides is what happened before.

| Phase | Arena holds | Free pages (vm_stat) | Swap in use | Compressor, stored | Ours compressed |
|---|---|---|---|---|---|
| server ready, 21:12:54 | 12.94 GB | 0.07 GB | 20.96 GB | 6.52 GB | 0.00 GB |
| 120 s of requests, balloon start | 17.05 GB | 0.07 GB | 22.64 GB | 4.79 GB | 3.07 GB |
| balloon end (+1.5 s) | 17.05 GB | 1.07 GB | 23.62 GB | 2.66 GB | 3.07 GB |
| 120 s after | 17.05 GB | 0.07 GB | 23.32 GB | 2.75 GB | 3.06 GB |

- **The governor grew the arena into the machine's last memory.** With no reserve (purgeable: R = 0) it read 4.1 GB
  of blocks in eight steps of 520 MB while its free-RAM figure fell from 2.81 to 0.78 GB, the machine's free pages
  stayed at their floor (0.07 GB) and the swap grew by 1.68 GB; the compressor's stored bytes fell by 1.73 GB
  (compressed pages written out to swap). Other programs' load ran at the same time (sums above), so the swap growth
  is not ours alone, but it coincides with our growth step for step.
- **The blocks of an active session are not purgeable.** Every block a request routes to is non-volatile for the
  keepalive window (180 s); with requests back to back that is all of them. So the grown 4.1 GB were memory macOS
  could not take back, and it swapped instead: the deviation from "volatile at once" fails.
- **No pressure event reached the arena** (no "memory pressure event" line): the dispatch source reports changes of
  level, the machine was at warn before the server's governor started, and the critical second ended with the
  balloon's own release. Reacting to events alone is not enough on a machine that sits at warn.
- Requests: 39, none failed, one answer (identical); decode median 32.6 before, 34.0 after.
- What it means for the design: no growth into free RAM with non-volatile blocks (R = 0 and a 180 s window together
  hold memory proactively), and non-volatile only for what the current pass reads. Between the two windows measured
  (1 s: macOS purged 97 % of the blocks before their next use; 180 s: nothing purgeable during a session) the
  trade-off is the maintainer's.

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
- Load average (1, 5, 15 min): 2.95, 4.32, 4.13
- CPU use by other programs over 0.5 s: 15.9 % of all threads
- sandbox: no
- Commit: `edeadc6` (dirty tree)
```

```
- Quiet check (2026-11-16 21:17:01): warning
  - warning: other programs use 12.8 % of all CPU threads
  - warning: load average 3.8 is high (it lags a minute)
```

## Plan of the second run, purgeable v2 (written before it)

- **Question**: as above, for v2 (build `f936774`): blocks non-volatile only from their stop to a hold of about 5
  passes after it; the pressure level read every 250 ms, so a machine already at warn makes every block volatile at
  each pass's end; the governor grows only with volatile blocks.
- **Scope, safety, logging**: as in the first run (same driver, balloon `--rates 1 --targets 6,10 --phase-s 240
  --hold-s 30 --rest-s 30 --limit-s 600 --swap-brake-gb 2`, the disk watcher ending the balloon below two swap files
  of free space). At the start: swap 21.2 of 22.5 GB in use, the VM volume 13 GiB free, pressure level normal.
- **Expectation**: while requests stream about 406 of 514 blocks (13 GB) are non-volatile; when the balloon pushes
  the level to warn, all go volatile within one pass (about 0.17 s) plus 250 ms, and macOS takes them: the balloon
  reaches 6 and 10 GB without the brake; other programs' compression rises by at most 0.5 GB over the run; swap at
  most +0.5 GB; no failed request, one answer; decode lower while the balloon holds (re-reads), back after.
  **A brake, swap growth over 0.5 GB or other programs' compression over 0.5 GB is the finding.**

## Result of the second run (v2)

Logs: [-balloon2-balloon.log](2026-11-16-mac-purgeable-balloon2-balloon.log),
[-balloon2-server.log](2026-11-16-mac-purgeable-balloon2-server.log). Quiet check before (22:23:25) and after
(22:27:40): ok. The disk watcher did not fire.

**The balloon braked again, after 2.1 s at 1.1 GB** (critical pressure), and other programs' memory was compressed:
a finding against the design.

| Balloon time | Free pages | Compressor stored | Ours compressed | Swap | Arena |
|---|---|---|---|---|---|
| -0.3 s | 0.07 GB | 3.08 GB | 3.04 GB | 22.24 GB | about 410 of 514 blocks non-volatile (requests streaming) |
| +0.7 s | 0.04 GB | 3.39 GB | 3.04 GB | 22.24 GB | |
| +1.1 s | | | | | level warn (the balloon's sample) |
| +1.4 s | | | | | 410 blocks (13.76 GB) volatile at once (the arena's line) |
| +1.7 s | 0.05 GB | 4.98 GB | 3.05 GB | 22.32 GB | |
| +2.1 s | | | | | critical: the balloon releases |
| +2.7 s | 1.96 GB | 2.55 GB | 3.06 GB | 22.80 GB | 1501 experts read again by the end of the episode |

- **Other programs' memory was compressed before the level said warn**: +1.6 GB stored in the compressor between
  +0.7 and +1.7 s, ours unchanged; the arena released its blocks 0.3 s after the warn sample, 0.7 s after the
  compression began. Swap +0.56 GB.
- **Why so much was non-volatile**: a decode pass routes to about 480 experts over all 48 layers, and the experts
  of one 32 MB block are spread over the ranking, so a pass touches most blocks; with the hold of 5 passes about 410
  of 514 blocks (13.8 GB) stay non-volatile while requests stream. Volatile memory itself lasts no time: the
  machine's purgeable pages read about 0 during requests (`vm.page_purgeable_count`, sampled every second), and
  the purgeable wired pages 0 to 0.36 GB, so wiring is not why the blocks were not taken.
- Requests: 41, none failed, one answer; decode median 34.3 before and after.
- Next, to shrink what stays non-volatile: one buffer per expert instead of 32 MB blocks (a pass then holds only
  the experts it routes to, not their whole blocks), and a signal faster than the level: other programs'
  compression itself.

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
- Load average (1, 5, 15 min): 1.4, 1.55, 1.72
- CPU use by other programs over 0.5 s: 4.8 % of all threads
- sandbox: no
- Commit: `a483b88` (dirty tree)
```
