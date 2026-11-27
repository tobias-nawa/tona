# Mac: the arena's constants against "no proactive reserve, no magic number" (2026-11-16)

The maintainer's rule: an amount that depends on the user's machine or use (free memory, budgets, reserves, floors)
is derived from a live measurement or removed; a mechanism granularity (step, chunk) may stay fixed with its measured
source. The PC's values in `engine/backend/budget.h` stay as they are (rule 1); only the Mac's use of them changes.

The arena is on by default for a model whose weights exceed the Metal working set (`arena_request`); the governor
(`TONA_METAL_ARENA_GOVERN`) is opt-in.

## The shipping default per model class (today)

| Class | Example here | Default | Reserve used | Best arm measured |
|---|---|---|---|---|
| GPU weights fit the working set (26.8 GB) | DM-T, 20.5 GB | no arena: the experts are file pages of the mapping | none | no arena: 34.1 tok/s, 156 MB of other programs compressed per minute of requests (`measurements/2026-11-16-mac-arena-react.md`, run 2) |
| GPU weights exceed it | Coder IQ1_M, 28 GB | the arena (`arena_request`), budget with the 5.0 GB reserve, governor off | 5.0 GB | the page cache decodes 41 to 55 % faster but re-reads 23 to 43 GB per pass and pushes other programs to the compressor (`measurements/2026-11-15-mac-coder-baseline.md`, `2026-11-16-mac-coder-arena-server-ab.md`) |

So the reserve is only in play above the working set. On DM-T every arena arm is judged against no arena (the best
arm, and the default); the reserve's fate is decided on the Coder IQ1_M, where the arena is the default.

| # | Constant | Value | Where (Mac) | What it guards | Decision | Decided by |
|---|---|---|---|---|---|---|
| 1 | `TONA_METAL_ARENA_FREE_GB` | 2.0 GB | `Arena::reserve_gb` | room for the user's programs | remove: the arena gives back what other programs lose | `measurements/2026-11-16-mac-arena-react.md` |
| 2 | `TIERED_CACHE_MIN_GB` | 1.5 GB | `Arena::reserve_gb` | page cache next to partial pins (the PC's floor sweep) | remove on the Mac: cold reads use `F_NOCACHE`, the mapped weights are counted on their own (`MemNow::mapped`) | same run |
| 3 | `PROMPT_MIN_GB` | 1.0 GB | `Arena::reserve_gb`, `TONA_RAM_GB` cap | stashed sequences the engine allocates later | remove from the reserve: the engine's own growth that compresses other programs is seen the same way; a run with stashed long prompts checks it | open: run with `TONA_SEQS` > 1 and long prompts |
| 4 | `RAM_SLACK_GB` | 0.5 GB | reserve, working-set slack, `arena_request` | the rest of the process | derive: the process's growth after the arena's start, measured (held bytes at start against during requests) | open |
| 5 | calibration clamp | 3 GB | `Arena::sample_free` | kernel memory the page counts miss | goes with the free-RAM governor: the react mode reads no free RAM | with 1 |
| 6 | free-RAM thresholds R/4, R/2, 3R/4, R + step | from R | `RamGovernorConfig::from_reserve` | when to free and grow | goes with the free-RAM governor on the Mac (the PC keeps it) | with 1 |
| 7 | governor times 5 s, 30 s, 600 s, 60 s; idle 1 s; window `WINDOW_S` | fixed | `ram_governor.h`, `monitor` | pacing of growth | react mode: in samples, judged over the median gap between demand signals; the idle 1 s keeps its source (18 times the longest gap between passes) | with 1 |
| 8 | fill brake | compressed > half a step | `fill_step` | the start fill displacing others | derive: any compression of other programs in the step (idle the machine compresses nothing, 0 of 600 samples) | open |
| 9 | `grow_step` 512 MB, `release_step` 260 MB, `emergency_step` 1040 MB | fixed | `ram_governor.h` | granularity | keep with source (8 and 32 pieces of the PC; 512 MB read in about 0.1 s at the safe point) | mechanism |
| 10 | blocks of 32 MB | fixed | `Arena::plan` | Metal buffer count | keep with source (one buffer per expert cost nothing and held no less, `measurements/2026-11-16-mac-purgeable-hold.md`) | mechanism |
| 11 | demand sample 50 ms | fixed | `monitor` | reaction time | keep with source (a 1 GB/s balloon compressed 1.6 GB in 0.7 s, `measurements/2026-11-16-mac-purgeable-balloon.md`) | mechanism |

Order: 1, 2, 5, 6, 7 fall together with the react mode (one run); then 8, 4, 3.

State (2026-11-17): the react mode as built did not meet its rule (4.9 GB of other programs' memory compressed per
minute of requests against 0.16 GB without an arena and 0 with the reserve; `measurements/2026-11-16-mac-arena-react.md`,
run 2). The learned ceiling (run 3) failed as well: 3.7 GB compressed per minute (median), one run fell to 10.1 tok/s.
The react mode was removed (its code last lived in b9640fe). On DM-T the default (no arena) is the best arm and uses no reserve;
for the class above the working set the reserve is the open item, to be judged there against the page cache and the
default arena.

## Ready to run: the reserve on the Coder IQ1_M, 5.0 against 3.5 GB

Not run on 2026-11-17: 12 GB free on the data disk and swap at 21.1 of 22.5 GB; a smaller reserve means more memory
pressure and more swap. Start only with at least 50 GB free on the disk and swap well under its size.

- **Question**: does the reserve's page-cache floor (`TIERED_CACHE_MIN_GB`, 1.5 GB) serve anything on the Mac? Cold
  reads use `F_NOCACHE` and the mapped weights are counted on their own, so the arena could take the 1.5 GB.
- **Scope**: Coder IQ1_M with its profile (`toolchain/profiles/coder-iq1_m.bin`), the default arena (governor off);
  arms 5.0 GB (today) and 3.5 GB (the floor left out; a test switch or a build without it), the `wired_arm` phases of
  `measurements/2026-11-16-mac-arena-default-wired.md` (45 s start, 60 s idle, 60 s of 200-token requests), three
  rounds alternating, about 3 min per arm, about 20 min in all. `diskwatch.sh` aborts the run under 8 GB free;
  `mem_balloon.py` is not used.
- **Expectation**: 3.5 GB gives the arena 1.5 GB more (about 650 more experts); decode up by 2 to 6 % (the cold reads
  per stop fall); other programs' compression during requests 0 to 500 MB in both arms (B on DM-T compressed 0 MB at
  13.6 GB).
- **Decision rule**: the floor goes on the Mac if the 3.5 GB arm compresses no more than the 5.0 GB arm plus the
  spread between its own rounds, and decodes no worse.
- **Active tooling**: the compression sampler (one Python process at 20 Hz), stop sleep and keepalive defaults.

## What carries over to the PC (notes, not measured there)

- **Passes, not holding, displace.** On the Mac a pass wires what it binds (4.6 GB wired idle, 29.6 GB during
  requests), and that is when other programs' memory is compressed. The PC's counterpart is the pinned host memory
  (GTT) of the warm tier during passes (`docs/backends/reference-platform.md`, `docs/architecture/expert-tiers.md`): a check
  of other programs' swap-outs while passes run, against idle, would tell whether the same holds there.
- **Other programs' compression is a clean signal; free RAM is not.** Idle the Mac compressed nothing in 600 samples;
  free RAM (with the file cache) swings by more than a growth step between passes and lifted a learned ceiling 1 to 4
  times per run. On Linux the analogues are PSI `memory some` and the machine's `pswpout` minus our own.
- **Reacting to it did not beat a fixed budget.** Giving back what others lost and regrowing in steps compressed 3.7
  to 4.9 GB per minute of requests (median) where the reserve's budget compressed nothing: the regrowth retakes the
  memory faster than the signal returns it. A governor on the PC that grows on calm free RAM should be checked for
  the same cycle.

## Which Coder shard the Mac's numbers use (checked 2026-11-17)

The PC found that stored references of 2026-11-10 used shard 1 of 2026-09-29 while `toolchain/models.toml` pins the
shard of 2026-10-19 since b6dd58c (`research/measurements/2026-11-17-prefill-fp8-experts.md`). On the Mac:
`work/models/coder-iq1_m` shard 1 is the pinned one (sha256 bfa3644d, checked by its `.sha256ok`); DM and DM-T were
cut from the shard of 2026-09-29 (`research/45-open-source-hygiene.md`). Every Mac quality check compares within one
file (tona-check against its CPU reference, MTP against the PC's goldens of the same DM file,
`measurements/2026-11-11-mac-mtp.md`), so none mixes shards.

## Result of the reserve run (2026-11-17, `measurements/2026-11-17-mac-iq1m-reserve.md`)

Constant 2 (`TIERED_CACHE_MIN_GB` in the Mac's reserve) stays: without it the Coder IQ1_M decodes 17 % faster (19.6
against 16.7 tok/s, three rounds) but its passes compress 28 to 525 MB of other programs' memory per minute of
requests, with it none. On the Mac the 1.5 GB act as room for what passes wire, not as a page-cache floor; the comment
in `Arena::reserve_gb` should say so, and a measured size of that room would replace it (constant 4).

Constant 4, first look (`measurements/2026-11-17-mac-pass-room.md`): during passes 18.6 GB move into wired and the
file cache falls from 8.5 to 10 GB to 2.6 to 3.5 GB, below the kernel's `vm_page_filecache_min` (7.7 GB) in both
reserve arms, so every page the machine needs then is compressed from someone's anonymous memory. No level to derive
the room from was found; open.

## For the maintainer (morning list, 2026-11-17)

1. Free disk space on the Mac: 12 GB free of 926 GB (99 %), swap 21.1 of 22.5 GB. No memory-pressure run until there
   are at least 50 GB free.
2. The purgeable arena stays opt-in (the hold and the react mode were removed, code in b9640fe); the numbers are in `research/53-mac-purgeable-arena.md`
   (where the displacement comes from) and in the table above.
3. Then the run above decides the first constant (`TIERED_CACHE_MIN_GB` on the Mac).

## State at the end of the night (2026-11-17)

Data disk 15 GB free, swap 18.5 of 19.5 GB: no run with memory pressure was started. Done without one: the hold and
the react mode removed (code in b9640fe), tona-check identical without the arena and with the default arena budget;
the Mac's unit build fixed (the KFD test only off macOS, 64 of 64).

Needs at least 50 GB free on the data disk to continue, in this order:
1. The reserve run on the Coder IQ1_M above (5.0 against 3.5 GB, about 20 min).
2. A tona-check of the purgeable path after the removal (it plans the arena up to the working set; about 3 min).
3. Then constants 8, 4 and 3 of the table, each with its own small run.

## End state (2026-11-17, evening)

- Constant 2 (`TIERED_CACHE_MIN_GB` in the Mac's reserve) stays for "work first": measured, the 5.0 GB reserve is its
  price. Correction (2026-11-18, the maintainer): work first or server first is a user setting, not a fixed rule. With
  "server first" the Mac runs without the 1.5 GB floor (19.6 against 16.7 tok/s decode, others compressed 28 to 525 MB
  per minute of requests). Next: plan the setting (an installer question and a config key named as on the PC, agreed
  with the PC side), then measure "server first" on the Coder IQ1_M with the method of the reserve run.
- Constant 4 stays open for "work first": no level derives the room (`measurements/2026-11-17-mac-pass-room.md`). An idea that fits
  #116, not built: grow the room only while the user is away, with the Mac desktop guard of #166 part 3 (input idle
  time from `HIDIdleTime` and the display state).
- Constants 5 to 8 belong to the governor, which is opt-in; in every run of 2026-11-16 and 2026-11-17 the step fill met
  no compression, so the fill brake (constant 8) never acted and a run cannot tell its threshold from zero. They wait
  until the governor is a default candidate again.
- Constants 9 to 11 stay as mechanisms with their sources. Constant 3 (`PROMPT_MIN_GB`) waits for constant 4.

## "Server first" on the Coder IQ1_M: no new run needed (2026-11-25)

The question "what does the Mac gain and cost without the 1.5 GB floor" was measured on 2026-11-17
(`measurements/2026-11-17-mac-iq1m-reserve.md`, arm B = server first, arm A = work first, three alternating rounds):

| Setting | Arena budget | Other programs compressed per minute of requests | Decode (200-token requests) |
|---|---|---|---|
| work first (5.0 GB reserve) | 8.65 to 8.85 GB | 0 MB in all rounds | 16.7 tok/s (16.6 to 16.9) |
| server first (3.5 GB) | 10.03 to 10.27 GB | 28 to 525 MB, median 397 | 19.6 tok/s (19.0 to 20.0) |

The code that decides both numbers has not changed since that run (commit `e0ee0cd` of the old history against
`feat/one-tona`): `engine/backend/metal/` differs only in the comment of `Arena::reserve_gb`; `backend/budget.h`,
`ram_governor.h` and `mem_macos.h` are identical. The server and model changes since (MTP verify, sampling, new models,
tokenizer) act on both settings alike and do not touch the budget, so a new run would repeat the same question on the
same memory path; the result is reused.

Decision that follows: "server first" is worth offering on the Mac, as a setting that drops `TIERED_CACHE_MIN_GB` from
the Mac's reserve: 17 % more decode (the ranges do not overlap), paid with 28 to 525 MB of other programs' memory
compressed per minute of requests. "Work first" keeps the 5.0 GB. Next: the setting itself (installer question and
config key with the PC's name), then one check run of the built setting against this table (same method, one round
per setting) to show it reproduces arm B.
