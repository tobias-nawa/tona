# Mac: what the passes wire beyond the idle page counts (2026-11-17)

## Plan (written before the run)

- **Question** (constant 4 of `research/54-mac-constants.md`): the reserve run showed that the 1.5 GB of
  `TIERED_CACHE_MIN_GB` act as room for what passes take beyond what the budget counts at start (without them the
  passes compressed 28 to 525 MB per minute). Which page class grows during passes, and by how much, so the room can
  be measured at the first passes instead of fixed?
- **Scope**: one server, Coder IQ1_M with the default arena (reserve 5.0 GB, arm A of
  `2026-11-17-mac-iq1m-reserve.md`, which compressed nothing); the machine's page classes (free, active, inactive,
  speculative, wired, file-backed, anonymous, compressor, purgeable, compressions) every 100 ms through 30 s idle
  after ready, 60 s of 200-token requests back to back and 30 s idle after; the arena's start line for its counts.
  About 4 min.
- **Expectation**: idle to requests, wired rises by about 18 GB (3.5 to 22 GB in the reserve run), most of it our
  mapped weights and buffers moving from file-backed and anonymous into wired, which the budget already counts; free
  plus file-backed falls by 1 to 2 GB more than our own pages that become wired (the room); compression 0 MB.
- **Use**: the room measured this way becomes a calibration at the first passes (the free-RAM window of the governor
  already keeps the least value while passes run, `Arena::sample_free`); no decision is taken from this run alone.
- **Active tooling**: the page-class logger (one Python process at 10 Hz), the disk and swap guard.

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
- Load average (1, 5, 15 min): 8.83, 6.89, 4.53
- CPU use by other programs over 0.5 s: 16.5 % of all threads
- sandbox: no
- Commit: `3d40fbd` (dirty tree)
```

```
- Quiet check (2026-11-17 21:54:55): warning
  - warning: other programs use 14.9 % of all CPU threads
  - warning: load average 8.83 is high (it lags a minute)
  - warning: the GPU is 7 % busy
```

## Result, arm A (5.0 GB reserve; 21:58 to 22:02)

Quiet check before: other programs 14.1 % of all CPU threads (one program near a full core), load average 3.44;
measured anyway (the user away, page classes do not depend on CPU load). Guard never fired. 5 requests.

| Phase | free | active | inactive | wired | file-backed | anonymous | compressor | free + file-backed | compressed |
|---|---|---|---|---|---|---|---|---|---|
| idle before (30 s) | 0.16 | 16.89 | 8.57 | 4.17 | 9.81 | 15.73 | 3.65 | 9.97 | 0 MB |
| requests (50 s) | 0.08 | 3.42 | 3.59 | 22.78 | 3.47 | 3.46 | 3.65 | 3.54 (least 2.95) | 36 MB |
| idle after (30 s) | 0.09 | 16.90 | 3.90 | 9.39 | 2.88 | 17.92 | 3.16 | 2.97 | 0 MB |

GB, medians of 100 ms samples. Budget 9.46 GB (2642 experts).

- Wired rises by 18.6 GB during passes; anonymous and file-backed pages fall by 12.3 and 6.3 GB as theirs become
  wired. What the machine can still give without compressing is free plus file-backed pages: 9.97 GB idle, at least
  2.95 GB during the passes, and the file cache does not come back after them.
- The kernel keeps a floor for file pages: `vm.vm_page_filecache_min` 471694 pages (1.93 GB in 4 KiB units, 7.7 GB in
  the 16 KiB of `vm_page_free_target`); below it the pageout scan reclaims anonymous memory, that is, compresses
  (XNU `vm_pageout`). Arm B of the reserve run had 1.4 GB less of that margin (about 1.5 GB) and compressed. Hypothesis:
  the room is "free + file-backed during passes" against that floor. Arm B is logged next to locate where compression
  starts.

## Result, arm B (3.5 GB reserve; 22:03 to 22:07)

Budget 10.21 GB, 6 requests, guard never fired.

| Phase | free | active | inactive | wired | file-backed | anonymous | compressor | free + file-backed | compressed |
|---|---|---|---|---|---|---|---|---|---|
| idle before (30 s) | 0.09 | 14.59 | 11.65 | 3.93 | 8.50 | 17.77 | 3.16 | 8.59 | 112 MB |
| requests (50 s) | 0.08 | 3.30 | 3.28 | 23.52 | 2.56 | 4.03 | 3.23 | 2.64 (least 2.61) | 114 MB |
| idle after (30 s) | 0.09 | 17.13 | 3.61 | 9.38 | 2.57 | 18.16 | 3.23 | 2.66 | 96 MB |

- The 17 samples with compression came at a median of 2.65 GB free + file-backed (file-backed 2.57 GB); A held 3.47
  GB file-backed during requests and compressed 36 MB in its 50 s, B 2.56 GB and 114 MB.
- **What the kernel does** (XNU `osfmk/vm/vm_pageout.c`, `vps_choose_victim_page`): below
  `vm_page_filecache_min` pageable file-backed pages the pageout scan takes anonymous pages, that is, compresses,
  and reactivates the file pages it finds; the floor is recomputed from the memory not in the compressor
  (`AVAILABLE_NON_COMPRESSED_MEMORY * 10 / divisor`). It is a page count: 471694 kernel pages, 7.7 GB at 16 KiB.
- **The hypothesis does not hold as stated**: both arms ran below that floor during passes (3.47 and 2.56 GB), so the
  floor alone does not tell where the cost begins; it only says that every page the machine needs during passes
  then comes from someone's anonymous memory. The measurable quantity is therefore not a level but the demand for
  pages while passes run (A 36 MB, B 114 MB per 50 s here, 0 and 28 to 525 MB per minute in the reserve run), and
  it varies with the other programs as much as with the arena.
- Open: no live rule for the room follows from these two runs; constant 4 stays open (the 5.0 GB reserve stays).

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
- Load average (1, 5, 15 min): 3.32, 4.07, 4.02
- CPU use by other programs over 0.5 s: 17.1 % of all threads
- sandbox: no
- Commit: `3d40fbd` (dirty tree)
```

```
- Quiet check (2026-11-17 22:03:48): warning
  - warning: other programs use 15.6 % of all CPU threads
  - warning: load average 3.29 is high (it lags a minute)
  - warning: the GPU is 6 % busy
```
