# Apple silicon generations M1 to M6

Data for the Mac class files ([mac-16-24gb.md](mac-16-24gb.md), [mac-32gb.md](mac-32gb.md),
[mac-64gb-and-larger.md](mac-64gb-and-larger.md)). Method and source marks: [README.md](README.md#how-the-files-estimate).
Unified memory: "VRAM" is the share of RAM the GPU may wire.

## Memory bandwidth, GPU cores, maximum RAM

Peak bandwidth is the datasheet figure, not what a kernel reads. Sources: Wikipedia Apple M1 to M4 [V, 2026-11-11]; M5 Pro
and M5 Max: Apple newsroom 2026-03-03 [V]; M5: Apple newsroom 2025-10-15 via outlets [S]; 32-core M5 Max (460 GB/s) [S].

| Chip | Peak bandwidth | GPU cores | Max RAM | Notes |
|---|---:|---|---:|---|
| M1 | 68.3 GB/s | 7 or 8 | 16 GB | |
| M1 Pro | 204.8 GB/s | 14 or 16 | 32 GB | |
| M1 Max | 409.6 GB/s | 24 or 32 | 64 GB | **maintainers' machine (32 GB, 32-core)**; 334 GB/s GPU read measured [M] |
| M1 Ultra | 819.2 GB/s | 48 or 64 | 128 GB | two M1 Max dies |
| M2 | 102.4 GB/s | 8 to 10 | 24 GB | |
| M2 Pro | 204.8 GB/s | 16 to 19 | 32 GB | |
| M2 Max | 409.6 GB/s | 30 or 38 | 96 GB | |
| M2 Ultra | 819.2 GB/s | 60 to 76 | 192 GB | |
| M3 | 102.4 GB/s | 8 to 10 | 24 GB | |
| M3 Pro | 153.6 GB/s | 14 or 18 | 36 GB | less bandwidth than M2 Pro |
| M3 Max | 307.2 GB/s (30 cores), 409.6 GB/s (40 cores) | 30 or 40 | 96 / 128 GB | |
| M3 Ultra | 819.3 GB/s | 60 or 80 | 256 / 512 GB | |
| M4 | 120 GB/s | 8 to 10 | 32 GB | |
| M4 Pro | 273 GB/s | 16 or 20 | 64 GB | |
| M4 Max | 410 GB/s (32 cores), 546 GB/s (40 cores) | 32 or 40 | 128 GB | |
| M5 | 153 GB/s | 10, one neural accelerator per core | 32 GB | [S] |
| M5 Pro | 307 GB/s | up to 20 | 64 GB | [V] |
| M5 Max | 460 GB/s (32 cores) [S], 614 GB/s (40 cores) [V] | up to 40 | 128 GB | [V] |
| M6 | **unchecked**: reports of about 200 GB/s, up to 12 GPU cores | unchecked | unchecked | not released; Bloomberg (June 2026) via MacRumors and 9to5Mac: no Pro, Max or Ultra of M6, higher chips in an M7 generation in 2027 [S] |

M6 sources are secondary and unopened (<https://www.macrumors.com/2026/06/25/m6-macbook-pro-2026/>,
<https://9to5mac.com/2026/06/25/apple-will-skip-m6-pro-and-m6-max-chips-new-report-says/>); treat M6 as unchecked.

Shipped configurations are not every combination of the table: M3 Pro 18 and 36 GB; M4 Pro 24, 48, 64 GB; M4 Max 36, 48,
64, 128 GB; M1 Max 32 and 64 GB (from memory [K], check before relying on a size).

## Bandwidth the GPU actually reads

**334 GB/s on the M1 Max, 82 % of its 409.6 GB/s peak**, shared and private buffers alike, median of 5 runs ([M]
research/measurements/2026-11-11-mac-bandwidth.md). The CPU reads 115 GB/s with 8 to 10 threads, 58 GB/s with one. Other
chips are scaled with the same 82 %; unchecked.

## SSD read throughput

Matters only for the large tier (experts streamed) and loading. Throughput depends on capacity (fewer NAND packages on 256
and 512 GB models) and read pattern.

| Chip / machine | Read | Source |
|---|---:|---|
| M1 Max MacBook Pro 1 TB (ours), 16 MiB sequential | 7.0 to 8.1 GB/s | [M] 2026-11-11-mac-ssd.md |
| same, 2.046 MB random reads at depth 16 | **13.0 GB/s**; 0.68 MB blocks 7 GB/s; QD 1 3.5 GB/s | [M] same |
| M1 Max Mac Studio 512 GB / 1 TB, Blackmagic | 5.2 / 5.5 GB/s (user posts) | [S] Apple Community |
| M1 Pro, M4 Pro 512 GB, Blackmagic | 5.0 / 5.1 GB/s (one user's comparison) | [S] Apple Community |
| M4 MacBook Pro base, M5 MacBook Pro base 512 GB, Blackmagic | 2.0 / 6.3 GB/s | [S] Max Tech via Tom's Hardware, 9to5Mac (2025-10) |
| M5 Max MacBook Pro 4 TB | about 12.8 to 13 GB/s read | [S] review |
| M2, M3 generations | **unchecked** | no source found |

Blackmagic figures are sequential; our 13 GB/s for scattered 2 MB blocks beats every published figure for the same SSD.
For other chips, measure with `bench/mac/ssdbench.cpp`.

## GPU memory limit

`recommendedMaxWorkingSetSize` on our 32 GB M1 Max reads **26.8 GB, 78 % of RAM** ([M] mac-bandwidth.md). Other sizes: about
75 %, 8 to 16 GB left to macOS ([S], sources disagree). `sudo sysctl iogpu.wired_limit_mb=<MB>` raises it until reboot;
unsupported, macOS may swap or panic (llama.cpp issue 19825, M4 Pro 24 GB, research/21-backends.md). The class files
assume the default limit.

## Which engine facts matter per generation

- GPU matrix units: **M5 and later** (neural accelerators, Metal 4 tensor API); BaseRT reports prefill up to 6.4x over
  llama.cpp on M5 (arXiv 2607.19438 [S], research/38). M1 to M4 prefill runs on ALUs.
- CPU matrix extensions: M1 NEON with dotprod, no i8mm; i8mm from M2, SME from M4 (research/39).
- Registers per thread: **M1 and M2** reserve a SIMD group's peak register count for its whole run; **M3 and later**
  ("Dynamic Caching", Apple tech talks 111375 and 111431 [S], unopened) allocate per code section and lower occupancy
  rather than spill. So the Metal backend runs dense f32 mat-vecs with **at most 5 tokens per dispatch** and expert
  mat-vecs with at most 4 slots per weight pass: 8 tokens per threadgroup spilled on the M1 Max (pass W8 +31 ms), and 5-
  and 8-slot expert kernels cost 1.8 and 5 ms at W4 ([M] research/measurements/2026-11-13-mac-mv8.md). **Unchecked:** on
  M3 and later wider dispatches may not spill; measure `tona-bench -b 5..8` with 8 tokens per threadgroup, and make the
  limit a per-family value.
- The kernel watchdog aborts a command buffer after about 5 s on the M1 Max ([M] 2026-11-11-mac-timeout.md); other
  generations unchecked.
