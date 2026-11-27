# Mac: a purgeable expert arena instead of a reserve

Status: design, 2026-11-16. Maintainer rules this answers: memory that may never be used is not kept free; when
another program needs memory it must be there at once; quantities that depend on the user's machine (free RAM,
budgets, reserves, floors) come from the machine at run time, not from constants that fit one Mac.

## What the measurements say

- **macOS purges volatile Metal buffers before it compresses anything**: next to 8.6 GB of volatile 32 MB blocks, a
  10 GB balloon took 1.5 GB of them within about 1 s and compressed 0.00 GB, at normal pressure
  (`measurements/2026-11-16-mac-purgeable-probe.md`).
- **The purge order is the order of the volatile marks** (xnu's FIFO token queue, `vm_purgeable.c`): the 45 blocks
  marked first went. A residency set does not protect a volatile block; it wires its allocations only at first.
- **Calls are cheap**: 0.3 us per state change, 1.0 us per residency add or remove with commit.
- **Filling without that protection displaces other programs**: an arena planned up to the working set and filled in
  growth steps compressed 15.6 GB in one start; the brake (more than half a step compressed) acts only after a step
  (`measurements/2026-11-16-mac-arena-stop-cpu.md`, the screening of 08:20). Filling to a start budget that keeps a
  reserve compressed nothing, but the reserve is what the maintainer rejects.
- Metal's header leaves a GPU read of a volatile or empty buffer undefined, and Metal refuses to make a buffer
  volatile while a command buffer uses it (Apple developer forum thread 698161). Apple advises against toggling per
  frame; the measured cost makes it affordable for the blocks a pass actually routes to, not for all.

## Design

Three states per 32 MB block (the block stays the purge unit: xnu purges a VM object whole):

| State | Purgeable | When |
|---|---|---|
| in use | non-volatile | a command buffer may read it: from the stop that routes to it until its pass completed |
| warm | volatile | not used for `warm_s`; marked in the order it fell idle, so the least recently used goes first |
| empty | purged by macOS | seen as `MTLPurgeableStateEmpty` when it is made non-volatile again: its experts are read from the SSD into it, as a cold read |

1. **Fill**: the arena is planned up to what the GPU's working set leaves (`arena_ceiling`); every block is filled
   and marked volatile at once, in profile order from the least used, so macOS takes the least used first. A fill
   step that makes the OS compress stops the fill, as now, but the blocks already in are volatile: the next request
   for memory purges them instead of compressing. The fill no longer needs a reserve.
2. **Decode stop**: the CPU knows the routed experts of the layer. For each routed expert in a volatile block: make
   the block non-volatile (0.3 us); if it was empty, read the block's experts from the SSD (the cold path, with the
   staging slots while it reads); remember it for the end of the pass. Hot blocks of the current window stay
   non-volatile and cost nothing.
3. **After the pass**, nothing is marked at once (Apple's advice and the cost): a block falls back to volatile when
   no pass used it for `warm_s`, at the next safe point (between passes, or by the monitor while idle). While
   requests stream, the blocks in use stay non-volatile; while the engine is idle, all blocks become volatile, so
   an idle Tona holds no memory macOS cannot take at once.
4. **Stops**: a layer whose experts are all in non-volatile blocks at encode time needs no stop. States change only
   between passes, so the test at encode time holds for the whole pass. With all experts resident and in use, the
   stops of the arena path disappear, the likely rest of the decode loss against no arena.
5. **What stays non-purgeable**: the weights outside the arena (mapped file pages, the OS reclaims them as file
   cache), the KV cache and DeltaNet state, the staging slots, the scratch buffers, the shared event and flag. For
   these Tona reacts only to pressure events (the dispatch memory-pressure source): no free-RAM threshold, no
   reserve.

## Governor: R becomes a step, not a distance

With the arena purgeable, the governor's free-RAM thresholds (`RamGovernorConfig::from_reserve`) have nothing left to
guard on the Mac: macOS takes volatile blocks itself. What remains:

- On a warn or critical pressure event: make the least recently used non-volatile blocks volatile, a step at a time
  (they are not in use between passes); on a repeated event, the next step. The step is a mechanism size (a growth
  step of 512 MB, the PC's 8 and 32 pieces) and may stay a measured constant.
- Grow again (re-read empty blocks) only when a pass routes to them, never ahead into free RAM: no proactive hold.
- The PC keeps `from_reserve` unchanged (it has no purgeable memory; Linux unchanged without a measurement).

## Constants in the Mac arena and the governor

Machine-dependent quantities (budgets, reserves, floors, thresholds) must be derived; mechanism sizes may stay as
measured constants with their source.

| Constant | Where | Kind | Plan |
|---|---|---|---|
| `RamGovernorConfig` defaults low 4, target 6, high 10, grow_floor 8, emergency 2 GB | `ram_governor.h` | threshold | PC starting values; the Mac derives them from R (`from_reserve`); with the purgeable arena the Mac uses none (pressure events) |
| reserve R = slack 0.5 + prompt reserve + page-cache floor 1.5 + `TONA_METAL_ARENA_FREE_GB` 2 | `arena.mm` `reserve_gb()` | reserve | drop on the Mac: the arena is purgeable, the prompt stash is its own allocation; the developer's 2 GB goes |
| `RAM_SLACK_GB` 0.5 | `budget.h`, `arena.mm` (working-set headroom) | reserve | keep for the PC; on the Mac the working-set headroom becomes the measured size of what Metal allocates besides our buffers (rest item) |
| `TIERED_CACHE_MIN_GB` 1.5 (page-cache floor) | `budget.h` | floor | PC (sweep 3.0 / 1.5 / 0.5); not used by the purgeable arena |
| `PROMPT_MIN_GB` 1, prompt default 12 GB, `STASH_RESERVE_GB` 8 | `budget.h` | budget / reserve | shared with the PC, out of scope here; rest item for the Mac (stash RAM should be purgeable too) |
| calibration clamp 3 GB | `arena_govern.mm` | threshold | goes with the free-RAM thresholds |
| fill brake: more than half a step compressed | `ram_governor.h` `fill_step` | threshold, relative to the step | keep relative; measure why one step showed 15.6 GB (counter or lag) |
| grow step 512 MB, release 260 MB, emergency 1,040 MB | `ram_governor.h` | mechanism | keep (maintainer: reaction steps are mechanism sizes) |
| block 32 MB | `arena.mm` | mechanism | keep: the PC's piece size, the purge unit |
| LRU 32 slots per layer | `arena.mm` | mechanism | keep (measured, `2026-11-15-mac-coder-arena.md`); not used when all experts fit |
| monitor 250 ms, window, calm 30 s, repeat 60 s, calm max 600 s, grow every 5 s | `arena_govern.mm`, `ram_governor.h` | time | keep; `warm_s` new, to be measured |
| stop limit 10 s | `stop_wait.h` | time | keep: above macOS's 5 s GPU watchdog |

## First checks of v1 (2026-11-16)

`tona-check` on DM-T (`TONA_DETERMINISTIC=1`, arena auto, profile, governor, `TONA_METAL_ARENA_PURGEABLE=1`, 64
tokens): identical to the run without the arena in both (69 steps, worst relative deviation 1.83e-06).

| `warm_s` | Fill | Compressed during the fill (ours) | Blocks woken / found purged | Wait per stop | Wall |
|---|---|---|---|---|---|
| 1 s | 16.07 of 17.22 GB, stopped by compression | 4.79 GB (+4.67 GB of our process) | 18798 / 18214 | 13.2 ms | - |
| 180 s | 17.22 of 17.22 GB | 0.00 GB | 514 / 193 | 10.5 ms | 191.7 s |

- Correction (2026-11-16, `measurements/2026-11-16-mac-purgeable-hold.md`): tona-check idles about 2.6 s before every
  step (its CPU reference), so with a 1 s window every block was volatile for over a second before each step; the
  97 % below are purges during those pauses, not between passes.
- **macOS takes volatile blocks first, not only under demand.** With 20 GB free (raw), a block volatile for a moment
  was gone at its next use in 97 % of the wakes: the system purges volatile memory as its first source of free
  pages, before its own file cache. A short `warm_s` therefore empties the arena on every pause between steps; the
  window is now the keepalive's after a request (`TONA_METAL_KEEPALIVE_S`, 180 s; one setting for both) (blocks of an active session stay non-volatile, an
  idle server holds only volatile blocks). Pressure events still mark every unused block volatile at once.
- **The fill's compression is our own.** The process's own compressed bytes (`TASK_VM_INFO`) rose by 4.67 of the
  4.79 GB the machine compressed during the fill: macOS compressed our other buffers (weights, caches untouched
  during the fill), not other programs' memory. The first stop then waited 3.4 s on page-ins. The 15.6 GB of the
  earlier screening start was most likely the same (that build could not tell).
- **Between the fill and the first pass** 193 blocks (6.2 GB) were purged at 180 s too: every block is volatile from
  its fill until a pass uses it. Acceptable for a server that starts idle; the first request reads them again.

## Cost A/B (2026-11-16)

Raw data and tables: [2026-11-16-mac-purgeable-cost-server-ab.md](measurements/2026-11-16-mac-purgeable-cost-server-ab.md)
(`.json`). DM-T + MTP-164, profile, governor; A plans the arena up to the working set (`TONA_METAL_ARENA_CEILING=1`),
B is the purgeable arena (build `f904f86`, 180 s window). 6 rounds (3 more repeated: system backups), 165.2 min;
answers identical in every start (SHA 727ff658). Plan: every metric within 1 %.

| Metric | A median | B median | Rounds 1 to 4, B vs A | Rounds 5 and 6 |
|---|---|---|---|---|
| decode-code | 34.8 | 34.7 | +0.00 % each | -2.0 %, +49.6 % |
| decode-text | 23.4 | 23.4 | -0.43 to +0.86 % | +0.4 %, +85.1 % |
| decode@8000 | 19.6 | 19.6 | -0.85 to +1.70 % | -12.3 %, +51.7 % |
| prefill@8000 | 154.1 | 148.5 | -5.41, -1.81, +0.35, -1.58 % | -5.4 %, -14.3 % |
| CPU ours, % of a core | 51 to 54 | 52 to 55 | | |

- **Decode costs nothing** where the machine was calm (rounds 1 to 4).
- **Prefill is a finding:** B behind in 3 of the 4 calm rounds, -1.6 to -5.4 %. Not explained yet; the likely cause
  is the wake at every stop of a prefill chunk (a chunk routes to almost every expert, so each stop walks and marks
  far more blocks than a decode stop). To measure before the default.
- **Rounds 5 and 6 are not comparable:** the machine compressed 21 to 28 GB per start (memory level 4 to 5, other
  programs' load) in both arms; A's round-6 start fell to 12 to 22 tok/s decode, which drives the means (+7 to
  +14 %) the A/B report prints.

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
- Load average (1, 5, 15 min): 7.64, 5.86, 4.27
- CPU use by other programs over 0.5 s: 17.8 % of all threads
- sandbox: no
- Commit: `f904f86` (dirty tree)
```

```
- Quiet check (2026-11-16, after the run): warning
  - warning: other programs use 18.5 % of all CPU threads
  - warning: load average 7.64 is high (it lags a minute)
```

## v2 after the balloon (2026-11-16)

Non-volatile for a whole session held memory macOS could not take (the balloon, `measurements/2026-11-16-mac-purgeable-balloon.md`).
Now: a block is non-volatile from the stop that routes to it until the end of its pass, plus a hold of a few passes
(the 0.99 quantile of the blocks' reuse gaps, 5 passes while requests stream:
`measurements/2026-11-16-mac-purgeable-hold.md`); under memory pressure and after 1 s idle every block is volatile.
A block found purged reads again only the experts a stop routes to, not all its pins. The pressure level is read
every 250 ms (the dispatch source reports only changes). The governor may grow into free RAM: its new blocks are
volatile until a stop routes to them.

## Prefill A/Bs of v2 (2026-11-16)

Raw data: [prefill](measurements/2026-11-16-mac-purgeable-prefill-server-ab.md) (v2) and
[prefill2](measurements/2026-11-16-mac-purgeable-prefill2-server-ab.md) (v2 with the idle time derived from the pass
cycle and the arena counters of B every 500 stops). Workload `server`, 8000:16, prefill@8000 primary, 2 rounds each
(early stop); A the arena of today planned to the working set, B the purgeable arena; answers identical.

| Run | prefill@8000 B vs A | decode@8000 (16 tokens after the prompt) |
|---|---|---|
| prefill | -2.50 % [-4.11, -0.89], B slower | -4.01 % [-10.67, +2.64] |
| a diagnosis start (2 rounds, no counters printed) | -0.71 % [-2.02, +0.59] | +0.09 % |
| prefill2 | -1.68 % [-2.97, -0.39], B slower | -2.31 % [-3.43, -1.19], B slower |

- **The cost is purges during the prefill.** Per 500 stops of B: 145 blocks found purged, 2195 experts read again,
  0.86 s of wakes; hold 5 passes (reuse gaps median 1, p99 4, max 6), about 406 of 514 blocks non-volatile after a
  pass's end. macOS took blocks volatile for a few passes although free RAM read 24 GB (raw; purgeable pages count
  as free): volatile memory goes before the file cache.
- The loss is the price of the blocks being available at once (maintainer's "work first"); a longer hold would cut
  it and hold memory. Not decided here.
- Pass cycle median 169 ms, gap between passes median 5.6 ms, p99 and max 54.9 ms: the source of the monitor's 1 s.

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
- Load average (1, 5, 15 min): 1.27, 1.25, 1.79
- CPU use by other programs over 0.5 s: 5.6 % of all threads
- sandbox: no
- Commit: `3bcdae3` (dirty tree)
```

```
- Quiet check (2026-11-16 22:22:32): ok
```

## Where the displacement comes from (2026-11-16, for the maintainer)

Holding memory between passes compresses nothing; running passes does, because Metal wires what a pass binds
(`measurements/2026-11-16-mac-purgeable-hold.md`, last section; `measurements/2026-11-16-mac-arena-default-wired.md`):

| DM-T server, 60 s each | Compressed | Wired |
|---|---|---|
| arena planned to the working set, idle | 0 MB | 4.6 GB |
| the same, 200-token requests back to back | 6515 MB | 29.6 GB |
| the 30 s after | 560 MB | 8.7 GB |
| no arena (file mapping), requests | 123 MB | 28.7 GB |
| default arena budget (13.65 GB), requests | 0 MB | 26.2 GB |

- Volatile marks cannot help against this (a block in use is wired and non-volatile anyway); the purgeable arena and
  its give-way stay opt-in. What sizes the arena is wired during passes (arena + weights + caches) against what
  other programs use, measured while passes run.
- The arena is not proposed as the default for models that fit the working set: without it DM-T decodes fastest
  (34.4 against 29.6 and 30.5 tok/s) and its file pages are dropped, not compressed.
- The PC has the analogue: the GPU's pinned host memory (GTT) for the pinned experts during passes, with its own
  headroom (`docs/backends/reference-platform.md`, the GTT half of the RAM; `docs/architecture/expert-tiers.md`, the warm tier).

## Measurements before it becomes the default

1. Cost: decode with the purgeable arena against the arena of today, both arms at full residency where it fits
   (DM-T), no pressure: within 1 %.
2. Decode against DM-T without the arena at full memory: the question of 2026-11-16 (does the loss go once all
   experts are resident and the stops are skipped).
3. Pressure (balloon 6 and 10 GB, maintainer's consent of 2026-11-16): compression per step, purged bytes, time to
   free, identical answers, no failed request.

## Removed (2026-11-17)

The hold of passes (`TONA_METAL_ARENA_HOLD_Q`, `measurements/2026-11-16-mac-purgeable-hold.md`) and the arena without a
reserve (`TONA_METAL_ARENA_REACT`, `measurements/2026-11-16-mac-arena-react.md`) failed their measured runs and were
removed from the code; both last lived in b9640fe. The purgeable arena with its give-way stays opt-in.
