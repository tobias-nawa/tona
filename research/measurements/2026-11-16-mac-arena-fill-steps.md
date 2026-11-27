# Mac: the arena's first fill in growth steps, and the experts' page cache (2026-11-16, M1 Max 32 GB)

## Plan (written down after the first two runs, which were meant as correctness checks)

- Why: DM-T's first start with the arena after a start without it compressed 28.9 to 31.2 GB of other programs'
  memory (2026-11-16-mac-dmt-arena-ab.md, second A/B), the starts after it 0 to 4.2 GB. Built: with
  `TONA_METAL_ARENA_GOVERN=1` the arena allocates the staging slots only and fills the rest in growth steps
  (`backend::fill_step`, 512 MB, each leaving the growth floor R - 512 MB free, stop when a step made the OS compress
  more than 256 MB).
- Scope: `TONA_DETERMINISTIC=1 tona-check` on DM-T, `"def quicksort(items):" --ctx 1024 -n 64`, profile
  `dmt-12.bin`, one run per case, no server, no foreign pressure. Expectation: tokens and logit rows equal to the run
  without the arena (`work/dmt-noarena.err`, 69 of 69 steps); the auto budget fills to the end without compression.
- Active tooling: `TONA_METAL_ARENA_GOVERN=1`, `TONA_METAL_ARENA_GB` as below; nothing else.

## Results

| Case | Filled | Stopped by | Compressed during the fill | Steps against no arena |
|---|---|---|---|---|
| auto budget | 13.64 of 13.64 GB in 1.8 s | the budget | 0.00 GB | identical |
| fixed 20 GB (test of a partial fill) | 14.51 of 20.00 GB in 3.5 s | the OS compressed | **13.14 GB**, with 10.78 GB free RAM (free plus file cache) | not reached: "the GPU did not reach an arena stop within 4 s" |
| auto, 2.2 GB of the model file cached before | 15.85 of 15.85 GB | the budget | 0.00 GB | identical |
| auto, 8.7 GB of the model file cached, cache dropped first | 16.63 of 16.63 GB in 2.3 s | the budget | 0.00 GB; 4.28 GB of the experts' page cache dropped | identical |

Findings:

- **The fixed 20 GB case compressed 13 GB of other programs' memory although free RAM as counted (free plus
  file-backed pages) stayed at 10.8 GB.** macOS takes the room for new pages from its inactive queue, file pages and
  other programs' anonymous pages alike, and compresses the latter. Counting the file cache as free does not keep a
  fill from displacing others. This run displaced memory; it is not repeated without the maintainer's consent.
- Most of the file cache was our own model: 9.83 GB of DM-T's first file of 10.8 GB file-backed pages (mincore).
  A server without the arena reads all experts through its mapping and leaves them cached; a fill after it competes
  with those pages. `msync(MS_INVALIDATE)` over a range of a read-only shared mapping drops exactly that range's clean
  pages from the cache, without root and without touching the pages first (1.10 GB cached, half the range
  invalidated, 0.56 GB left; free pages rose by 9.8 GB for the whole file). `MS_KILLPAGES` did nothing on pages not
  mapped in the caller. The arena now drops its experts' file ranges before the step fill (it reads them with
  F_NOCACHE only); the edge pages of each tensor stay.
- The 4 s stop error of the 20 GB case appeared once before with the old fill, without a profile, at 00:29 (DM-T, auto
  budget). Both were first passes under memory pressure; unchecked whether the stop waited on paging.
- tona-check's CPU reference reads every weight through the mapping (8.7 GB of the file cached after a run), so it
  cannot show what a server start leaves in the cache.

Not measured: a server start after a start without the arena, the case of the A/B. Such a start itself compresses 11
to 27 GB (the arm without the arena), so it waits for the maintainer's consent to measurements that displace memory.

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
- Load average (1, 5, 15 min): 3.93, 5.01, 3.99
- CPU use by other programs over 0.5 s: 4.0 % of all threads
- sandbox: no
- Commit: `001ee20` (dirty tree)
```

```
- Quiet check (2026-11-16 05:22:37): warning
  - warning: load average 3.93 is high (it lags a minute)
```
