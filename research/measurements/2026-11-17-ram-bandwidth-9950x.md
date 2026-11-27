# CPU RAM bandwidth on the Ryzen 9 9950X (#164)

Plan, written before the run. Results below the line `## Results`.

## Question and decision

`docs/backends/reference-platform.md`, `docs/hardware/*.md` and the planner estimates use "58 GB/s read, 31 GB/s write", measured on the Ryzen 5 9600X (one CCD: its link to the I/O die limits
the read, one thread already reached 57.6 GB/s, `2026-10-22-micro.md`). The 9950X has two CCDs, so every CCD has its own link and the DRAM (DDR5-6000, dual channel, 96 GB/s theoretical)
can become the limit. Which numbers replace the 58 and 31, and from how many threads? The answer replaces the CPU row of the platform doc and the "CPU path 58 GB/s" of the hardware estimates;
it also says whether the 16 threads of the CPU experts (`2026-11-17-omp-threads.md`) can use more than one CCD's worth of bandwidth.

## Scope

`bench/micro/rambench.cpp` (extended in this branch: thread counts 1, 2, 3, 4, 6, 8, 12, 16, 24, 32; the 9600X list was 1, 2, 3, 6, 12; and a pin mode), buffers of 2048 MiB (two of them, 4 GiB of RAM; the 9600X run used the same
size), 7 repetitions per point, read / write (non-temporal stores) / copy. Pin `compact` (thread i on CPU i: on this machine CPUs 0 to 7 are CCD 0, 8 to 15 CCD 1, 16 to 31 their SMT siblings, from
`/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`: 8 threads fill one CCD, 16 threads one per core on both) and pin `spread` (threads alternate between the CCDs: 2 threads are one per CCD). Both pin modes twice
(compact, spread, compact, spread; the order shows drift). One exclusive job through the queue (RAM bandwidth of this job would disturb anything that runs beside it), about 3 minutes including the compile of the one file;
`hw_report` and `quiet_check` before and after.

## Why this size

The 2 GiB buffer is far above the 64 MiB of L3, so DRAM is measured, as on the 9600X, which keeps the two results comparable. 7 repetitions: the 9600X run had sigma of 0.1 to 0.35 GB/s at that count, far below the effect to read (58 against 80+).
Two runs per pin mode give the run-to-run spread. No larger size or more repetitions would change the decision (which numbers go into the doc, and whether more than 8 threads help).

## Expectation (with tolerance and source)

| Measurement | Expectation | Source |
|---|---|---|
| read, compact, 1 to 8 threads (one CCD) | 55 to 62 GB/s, flat from 1 thread | 9600X: 57.6 GB/s at 1 thread, 58 at 6 (one CCD, `2026-10-22-micro.md`); the same CCD design |
| read, compact, 16 threads (both CCDs) | 75 to 90 GB/s | DDR5-6000 dual channel 96 GB/s theoretical; 80 to 94 % read efficiency is usual for a streaming read |
| read, spread, 2 threads (one per CCD) | 75 to 90 GB/s (one thread per CCD already saturates its link, as on the 9600X) | same |
| write (streaming), 16 threads | 45 to 65 GB/s (9600X 31 GB/s on one CCD; two links) | est. |
| copy (read + write counted), 16 threads | 70 to 90 GB/s | 9600X 55 GB/s |
| 32 threads (SMT) against 16 | within 5 % (the DRAM limit, not the cores, binds) | 9600X: 12 against 6 within 1 % |

A value outside a range is a finding. Doc rows are updated with the medians and ranges measured here, and say which pin mode and thread count they hold for.

## Active tooling

None besides the tool itself (no profiler, no TONA_* variable); `OMP` does not enter (std::thread, own pinning). `hw_report` before and after.

## Test conditions

Job `bw.sh` through the queue (exclusive, desktop guard on). It ran twice: 2026-11-18 01:38 and, after the one-file tool was split into functions for the code gates (same measurement, no change in what it does), again at 02:2x on the committed code; the tables below are the second (committed) run's
files, `2026-11-17-ram-bandwidth-9950x-logs/` (`rambench.log`, `tc.log`); **the first run agrees within 0.7 GB/s at every point** (e.g. spread read 2 threads 71.8 and 72.1, compact read 16 threads 68.3 and 68.3, 32 threads 73.0 and 74.1, write 16 threads 57.8 and 57.7).
hw_report: Ryzen 9 9950X (16 cores / 32 threads), kernel 7.3.4, 60.5 GiB RAM, 56.4 GiB available; load average 3.2 / 7.2 / 5.6 (the end of the queue's own build jobs just before; "CPU use by other programs" 0.1 %); quiet check before and after: warning only for the
desktop's 1.97 GiB VRAM. No other job ran (exclusive lock).

## Results

GB/s, median of 14 repetitions (two runs of 7 per pin mode); min to max over all 14 in brackets; the two runs' medians differ by at most 0.5 GB/s except where noted.

| Threads | read compact | read spread | write (stream) compact | write spread | copy compact | copy spread |
|---:|---|---|---|---|---|---|
| 1 | 55.3 (54.5-55.8) | 55.2 | 30.9 | 31.0 | 53.4 | 53.4 |
| 2 | 56.6 | **71.8** (66.6-72.0) | 30.3 | **57.1** | 53.4 | 66.6 |
| 4 | 56.7 | 71.7 | 29.9 | 55.4 | 52.9 | 65.9 |
| 6 | 56.4 | 71.5 | 31.9 | 54.9 | 52.3 | 65.5 |
| 8 | 56.4 (53.2-56.6) | 70.6 | 31.8 | 54.5 | 52.6 | 65.3 |
| 12 | 65.3 | 69.2 | 45.8 | 58.2 | 60.5 | 64.2 |
| 16 | **68.3** (68.0-68.4) | 68.2 | **57.8** | 57.7 | 63.6 | 63.5 |
| 24 | 64.2 | 64.2 (63.9-70.1) | 45.5 | 45.5 | 59.3 | 59.2 |
| 32 | 73.0 (67.2-74.6; runs 73.8 / 70.6) | 74.1 (67.8-74.4) | 61.8 | 61.8 | 63.3 | 63.4 |

(compact: thread i on CPU i, so up to 8 threads fill CCD 0, 9 to 16 add CCD 1, 17 to 32 add the SMT siblings in the same order; spread: threads alternate between the CCDs.)

Against the expectation:
- **One CCD alone: 55 to 57 GB/s read, 30 to 32 GB/s write, from one thread on, flat to 8 threads**: as predicted (55 to 62), and the same as the 9600X (58 / 31): one CCD's link is the limit, not the cores.
- **Both CCDs: read 68 to 72 GB/s, below the predicted 75 to 90.** The best points: 2 to 8 spread threads 71.8 / 71.7 / 71.5 / 70.6; 16 compact threads (one per core, what the engine runs) 68.3; 32 threads 73 to 74 (with a spread of 67 to 75). That is 71 to 77 % of the 96 GB/s theoretical, **+24 % over one CCD**, not the +55 % that two independent links promised: the DRAM (or the memory controller path) binds well below the datasheet figure.
- **Write (stream) 55 to 58 GB/s with both CCDs** (predicted 45 to 65; 1.8x the 9600X's 31). **Copy** (read and write counted) 63 to 67 GB/s (predicted 70 to 90, below; 55 on the 9600X).
- 32 threads against 16: read +7 % (73 against 68 compact; the spread of the 32-thread runs is large, 67 to 75), write +7 %, copy 0 %: just above the predicted 5 %. The server's CPU experts did not gain from the SMT threads (`2026-11-17-omp-threads.md`, run 3: -0.1 to -1.1 %).
- 12 and 24 threads in `compact` are uneven teams (8 + 4 on the CCDs, 16 + 8 on the cores) and read 65 and 64: below the even points 16 and 32, as expected from the uneven split.
- Run-to-run: the two runs of each pin mode agree within 0.5 GB/s everywhere except 32 threads (read 73.8 and 70.6) and the 2-thread spread read (min 66.6).

Verdict for the docs: **CPU read 72 GB/s (both CCDs; 55 to 57 on one), write 57 GB/s**. The platform doc, BENCHMARKS hardware table, backend blueprint and hardware docs are changed (rows updated; the estimate tables of the hardware docs keep the 58 GB/s of a one-CCD desktop CPU and say so,
because they describe other machines, and their calibration anchor was measured on the 9600X). The engine's CPU experts read at up to 69 GB/s with 6 threads on the 9600X (`2026-11-15-ornith-o1-o2-cpu.md`: 44 to 69 GB/s effective): the effective rate of the expert kernels on the 9950X at 16 threads is not measured here.
