# Mac: the arena's governor under a memory balloon

Issue #108. The test of `docs/architecture/expert-tiers.md` section 5 on the Mac: another program's memory grows and
shrinks next to the Coder; the governor (`TONA_METAL_ARENA_GOVERN=1`, `backend/ram_governor.h`) must free arena
blocks in time, read them again when the memory is back, abort no pass and change no token. Maintainer's consent
2026-11-15 evening (it displaces other programs' memory on purpose; the session itself may stop).

## Plan (written before the run)

**Run.** One server: the Coder + MTP-256, defaults (auto budget, 32 cache slots per layer, lookahead 2), the
governor on, `TONA_DETERMINISTIC=1`, `--ctx 32768`. A driver sends the same greedy request (a short code prompt, 128
tokens) in a loop: 3 before the balloon (the reference answer), then for as long as the balloon runs, then 4 min
after (regrowth). The balloon (`tools/mem_balloon.py`, incompressible anonymous memory in 64 MB mappings) runs three
phases of 10 min, at 0.1, 1 and 3 GB/s; in each phase cycles to 10 GB and 20 GB alternately, holding 20 s, releasing,
resting 30 s. About 35 to 40 min.

**Abort rules (inside the balloon, so they hold if this session stops).** Hard end at 45 min (and an alarm signal
60 s later); release everything and end at once at critical memory pressure (`kern.memorystatus_vm_pressure_level`
4) or when the swap in use grows 2 GB above its level at the start. The driver ends at 75 min whatever happens.
Logs written as they happen: `-balloon.log` (a line per second), `-driver.log` (a line per request), `-server.log`
(a line per governor action).

**Expectation.** Memory: about 34.4 GB; besides the arena (about 12 GB with the governor's 6 GB reserve) the system
and our other buffers hold about 16 GB (wired 13 GB during passes, other programs' anonymous memory about 4 GB).

| What | Expected | Tolerance / reading |
|---|---|---|
| 10 GB cycles | the governor frees 4 to 8 GB per cycle, the balloon completes, no brake | brake at 10 GB is a finding (reaction too slow) |
| 20 GB cycles | the arena cannot make room for all of it (16 GB + 20 GB > 34.4 GB): the governor frees nearly all blocks; at 1 and 3 GB/s the brake (critical pressure or swap) is likely | brake at 0.1 GB/s is a finding |
| reaction | first release within 0.5 s of free RAM < 4 GB (250 ms sampling, between passes) | from the server log against the balloon log |
| decode | about 20 tok/s without balloon, 8 to 15 with the arena shrunk, back to about 20 within 1 to 2 min after (calm 30 s, doubled per repeated tight event) | |
| aborts and failed requests | 0 | any is a finding |
| answers | identical to the reference in every request | any difference is a finding (placement must not change numbers) |
| oscillation | per balloon cycle one shrink episode and one growth episode | more is a finding |

**Active tooling.** `TONA_METAL_ARENA_GOVERN=1`, `TONA_DETERMINISTIC=1`, `TONA_EXPERT_PROFILE`; the balloon; nothing
else. No build and no other measurement meanwhile.

## Test conditions before the run

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 3.17, 2.76, 2.12
- CPU use by other programs over 0.5 s: 6.9 % of all threads
- sandbox: no
- Commit: `58da351` (dirty tree)
```

```
- Quiet check (2026-11-15 22:43:56): warning
  - warning: other programs use 10.8 % of all CPU threads
  - warning: load average 3.17 is high (it lags a minute)
```

Memory before (vm_stat, GB): free 3.1, wired 3.4, file-backed 23.9, anonymous 2.2, compressor 0.9; swap total = 21504.00M  used = 20324.06M  free = 1179.94M  (encrypted)

## Result of the balloon run (log: 2026-11-15-mac-arena-balloon.log)

| What | Expected | Measured |
|---|---|---|
| answers | identical | **30 of 30 identical** to the reference, 0 failed requests, no pass aborted |
| 10 GB cycle at 0.1 GB/s | frees 4 to 8 GB | the governor freed every block (11.75 to 0.70 GB, the staging slots) within the cycle, 40 steps |
| free RAM while freeing | rises with each step | **stayed at 3.9 to 4.0 GB**: finding 1 |
| after the balloon | regrowth within 1 to 2 min | **none in 4 min**, free 2.9 GB, decode stayed at 5.7 to 6.1 tok/s |
| 20 GB cycle at 0.1 GB/s | completes | **brake** at 14.3 GB held (swap +2 GB, pressure "warn" from 12.8 GB): finding 2 |
| decode | about 20, 8 to 15 during | 16.8 to 18.2 before (deterministic), down to 5.5 to 6.5 |

The phases at 1 and 3 GB/s did not run (the brake ends the balloon).

**Finding 1: freed blocks stayed in memory.** The governor frees blocks between passes on the engine thread and,
while the engine is idle, on its monitor thread. That thread had no autorelease pool, so a released buffer was not
freed until the thread ended. The same release on the engine thread (churn, inside a pool) returned the memory at
once: in the churn run free RAM rose 3.1 GB and wired memory fell 3.1 GB at each step. Fix: `shrink` and `grow` run in
a pool of their own.

**Check of the fix without a balloon** (`TONA_METAL_ARENA_GOVERN_TEST_GB=6`, `_S=120`: free RAM reads 6 GB less for
2 minutes, nothing displaced): the governor freed 5.2 GB in 10 steps (to 5.92 GB), the measured free RAM rose with
them (9.5 to 9.8 GB real), 30 of 30 answers identical, 0 failed.

**Finding 3: the PC's thresholds do not fit 32 GB.** After the bias ended free RAM was 9.5 to 9.8 GB, in the dead
band (4 to 10 GB): the governor never grows the arena again, which stays at 5.9 GB (decode 8 tok/s deterministic
against about 17 before). With 34.4 GB of RAM, of which the Coder's server holds about 16 GB besides the arena, a
growth threshold of 10 GB free means the arena never gets back what it gave up. The thresholds must follow the
machine (the auto budget leaves 6 GB free; growth must start below the free RAM of an undisturbed start).

**Finding 2: 20 GB do not fit next to the Coder on this machine.** With the arena at its minimum the server holds
about 16 GB (wired weights 5.9, buffers 3.9, the staging slots, the system); 20 GB more exceed the 34.4 GB, so the
system swaps whatever the governor does. The expectation (brake only at 1 and 3 GB/s) was wrong: the brake at 0.1
GB/s is the machine's limit, not the governor's reaction.
