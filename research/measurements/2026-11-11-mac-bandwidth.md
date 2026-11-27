# 2026-11-11: memory bandwidth of the Mac (GPU and CPU), M1 Max 32 GB (#107)

Phase 0 of the Mac backend, point 3: what the GPU and the CPU can read from unified memory. Decode of a MoE model is
bandwidth-bound, so these two numbers bound every decode rate of a Mac backend. Speed numbers on this machine are
provisional (a second project of the owner runs on it; see the load below).

## Plan (written before the run)

**Scope.**

| Block | What runs | Rounds | Duration |
|---|---|---|---|
| A inventory | `bench/mac/gpubw` without a measurement: device name, GPU families, `recommendedMaxWorkingSetSize`, `maxBufferLength`, unified memory | 1 | < 1 s |
| B GPU | `bench/mac/gpubw`: compute kernels read (sum of `uint4`), write, copy, plus a blit copy; buffers `storageModeShared` (the case of mmap-ed weights) and `storageModePrivate`; 1 GiB per buffer; grids of 24 × 4 and 24 × 16 threadgroups of 256 threads, the better one counts; each test 10 passes after 2 warm-up passes, time from the command buffer's GPU timestamps | 5 runs of the whole set | ~5 s per run |
| C CPU | `bench/mac/cpubw`: NEON read (sum of 4 × 128 bit per step), write (`stp` of zero pairs), copy (`memcpy`); 1 GiB per buffer; 1, 2, 4, 8, 10 threads; each point 5 repetitions | 3 runs | ~15 s per run |

Total load at most ~1 min of full GPU or CPU, in blocks of < 20 s, pauses in between.

**Expectation.**

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| Peak (datasheet) | 409.6 GB/s | - | LPDDR5-6400, 512 bit (Apple: "400 GB/s", M1 Max product page) |
| GPU read | 330 GB/s | 280 to 390 | 70 to 95 % of the peak, typical for the GPU path of Apple chips (knowledge, unchecked) |
| GPU copy (read + write counted) | 330 GB/s | 280 to 390 | as read (knowledge, unchecked) |
| CPU read, 1 thread | 100 GB/s | 80 to 120 | AnandTech M1 Max review 2021: ~100 GB/s from one P-core (knowledge, unchecked) |
| CPU read, 8 to 10 threads | 220 GB/s | 180 to 260 | same review: ~240 GB/s from the CPU clusters, well below the peak (knowledge, unchecked) |

A result outside the tolerance is a finding: it gets a line in the evaluation, not a rerun until it fits.

**Active tooling and conditions.**

- Build: `swiftc -O` (Swift 6.4.2, Xcode 27.2); Metal shaders compiled at run time from source (`MTLDevice.makeLibrary(source:)`), default options. `clang++ -O3 -mcpu=apple-m1` (Apple clang 21).
- No profiler, no Metal validation layer, no `TONA_*` variables. No sudo, no `sysctl` changes (wired limit stays default).
- Threads are not pinned (macOS has no affinity API; QoS `USER_INTERACTIVE` puts them on the P-cores first).
- **Load at plan time (12:08):** load average 5.2 / 4.2 / 5.8 on 10 cores; background load: the three busiest processes
  together 89 % CPU (one process at 160 % at 11:35, no longer among them); GPU 0 % (ioreg Device Utilization); memory 76 % free.
  The load is captured again right before each block.
- **Power: AC since 12:09** (battery 37 %, charging; on battery at 12:08 when the plan was started), low power mode
  off (`pmset powermode 0`). Recorded with every block; a block on battery would be marked.

**Addendum to the plan, 12:12, after block C and before block D.** Block C read only ~57 GB/s with 1 thread and
~115 with 8 to 10 (expected 80 to 120 and 180 to 260), while `memcpy` moved ~90 GB/s with 1 thread. Suspicion: `op_read`
is limited by its four add chains, not by the memory. Block D: `read8` (8 independent sums, `ld1` of four q registers,
128 bytes per step), same threads and repetitions, 3 runs (~10 s each). Expectation: if the suspicion holds, read8 lies
in the original tolerance (1 thread 80 to 120, 8 to 10 threads 180 to 260); if it stays near block C, the CPU path is
the limit and the AnandTech figure does not apply to this machine or load.

## Results

Raw data, one JSON line per test and run header (time, load, power):
[gpu](2026-11-11-mac-bandwidth-gpu.jsonl), [cpu](2026-11-11-mac-bandwidth-cpu.jsonl),
[cpu read8](2026-11-11-mac-bandwidth-cpu-read8.jsonl). Tools: `bench/mac/gpubw.swift`, `bench/mac/cpubw.cpp`.

**Conditions during the runs (12:10 to 12:13).** AC power throughout. Load average 7.2 to 8.8 on 10 cores;
background load: the two busiest processes together up to ~90 % CPU; GPU 0 % before block B.

### A: inventory

| | |
|---|---|
| Device | Apple M1 Max, architecture `applegpu_g13s`, families apple7 and metal3 (not apple8/apple9) |
| `recommendedMaxWorkingSetSize` | 26,800,603,136 B = 26.8 GB (78 % of the 34.4 GB RAM) |
| `maxBufferLength` | 20,100,448,256 B = 20.1 GB |
| Unified memory | yes |
| macOS | 27.0.1 (26A434) |

The working set of 26.8 GB is above the ~21 to 24 GB that `research/21-backends.md` assumed for 32 GB.

### B: GPU (5 runs; per run the median of 10 passes, GB/s)

| Test | shared, grid 24×4×256 | private, grid 24×4×256 | shared, grid 24×16×256 |
|---|---|---|---|
| read | 328 / 335 / 335 / 334 / 328 | 331 / 333 / 332 / 333 / 332 | 309 to 319 |
| write | 332 / 341 / 347 / 340 / 330 | 328 to 345 | 307 to 313 |
| copy (read + write) | 318 to 324 | 323 to 325 | 265 to 285 |
| blit copy (read + write) | 363 to 373 | 364 to 373 | - |

- **GPU read: 334 GB/s median (328 to 335), 82 % of the 409.6 GB/s peak.** Within the expectation (280 to 390).
- Shared and private buffers read the same. Weights mapped from the file (shared) lose nothing against private copies.
- The smaller grid (24 × 4 threadgroups) is faster by 4 to 6 %. Whether an even smaller grid gains more is not measured.
- The blit engine copies at ~370 GB/s, faster than the copy kernel (~322).

### C and D: CPU (3 runs; per run the median of 5 repetitions, GB/s)

| Threads | read | read8 (block D) | write | copy (read + write) |
|---|---|---|---|---|
| 1 | 55 / 58 / 57 | 56 / 59 / 60 | 60 / 60 / 59 | 87 / 91 / 92 |
| 2 | 76 / 98 / 94 | 108 / 98 / 96 | 91 / 105 / 89 | 144 / 123 / 105 |
| 4 | 103 / 105 / 101 | 104 / 90 / 104 | 106 / 99 / 107 | 144 / 136 / 160 |
| 8 | 109 / 115 / 117 | 116 / 111 / 116 | 118 / 120 / 121 | 176 / 165 / 163 |
| 10 | 100 / 118 / 119 | 120 / 112 / 117 | 120 / 121 / 121 | 170 / 187 / 165 |

- **CPU read: ~58 GB/s with 1 thread, 112 to 120 GB/s with 8 to 10 threads. Both below the expectation** (80 to 120 and
  180 to 260). Finding.
- Block D rules out the read kernel as the cause: 8 independent sums and 64-byte loads give the same numbers.
- Copy reaches 165 to 187 GB/s (read and write counted) with 8 to 10 threads. So the CPU clusters move more when reads
  and writes mix than with reads alone.
- unchecked: whether the background load (load 7 to 9) or the missing pinning (threads may land on the two
  E-cores) explains the gap to the AnandTech figures. A rerun on an idle machine would decide.

## Evaluation

- **The decode bound of a Mac backend is the GPU at ~334 GB/s read**, 2.9× what the CPU reads with all cores busy
  (~115 GB/s). The CPU can at most supplement the GPU. A design like the
  reference (CPU computes the missing experts) gains little here, because GPU and CPU share the same memory and the GPU
  reads it 3× faster.
- **Memory the GPU may use: 26.8 GB** by default (no wired-limit change). The reference model (55 GB) does not fit;
  the SSD measurement (next file) decides how much of it streaming can recover.
- Rates per token follow in the architecture proposal, from these numbers and the active bytes per token of the model.
