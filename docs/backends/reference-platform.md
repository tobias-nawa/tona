# Reference platform

The machine the reference backend exploits. "Measured" numbers come from `research/measurements/` (file in the source
column).

| | | Source |
|---|---|---|
| GPU | AMD Radeon RX 9060 XT, RDNA4, gfx1200, 32 CUs | `rocminfo` |
| VRAM | 16 GB GDDR6 (16,304 MiB usable per driver), 128 bit × 20 Gbps; **measured 315 GB/s** read | 2026-10-22-micro |
| Caches | L2 4 MiB (~3.6–4 TB/s), Infinity Cache 32 MiB (**~1.18 TB/s**), measured | 2026-10-22-micro |
| Compute | measured: WMMA f16 93, bf16 97, fp8 186 TFLOPS, int8 183, int4 341 TOPS; FMA f32 24.3 TFLOPS (~3.1 GHz, 174 W). Under sustained load at the power limit (182 W, fixed): 2.6–2.8 GHz | 2026-10-22-micro, 2026-10-24-gemm-v2 |
| PCIe | 5.0 x16, measured 55.7 GB/s per direction, 60.8 GB/s both directions together | 2026-10-22-micro |
| CPU | Ryzen 9 9950X, Zen 5, 16C/32T, two CCDs (CPUs 0-7 and 8-15, SMT siblings 16-31), AVX-512; a Ryzen 5 9600X (6C/12T, one CCD) until 2026-11-17 | `lscpu` |
| RAM | 64 GB DDR5-6000 CL40 dual channel | |
| RAM bandwidth | theoretical 96 GB/s; measured **72 GB/s read** (threads on both CCDs, 2 to 8 spread threads 71.8; 16 threads on cores 0 to 15 68; 32 threads 73 to 74), **57 GB/s streaming write**, 64 to 67 GB/s copy; **one CCD alone 55 to 57 GB/s read, 31 write** (the link of the CCD; the 9600X had one CCD: 58 / 31) | 2026-11-17-ram-bandwidth-9950x, 2026-10-22-micro |
| Kernel launch | 2.1 µs per kernel back to back, HIP graph the same (2.08), launch + sync 17.7 µs | 2026-10-22-micro |
| In-kernel sync | `grid.sync()` 0.75 µs, block→block 0.28 µs, CPU↔GPU via host memory 0.73 µs | 2026-10-22-micro-persist |
| Storage | 2× NVMe (Samsung 990 PRO), LUKS; measured 2.65 GB/s sequential, 457K random 4 KiB reads/s | installer `--print` |
| OS | Fedora 44, kernel 7.3.4 (7.2 until 2026-11-17); toolchain ROCm 10.1 (HIP 7.16, LLVM 24) under `work/rocm-10.1` | `scripts/setup.sh` |

## What runs next to the engine

- **The desktop uses the same GPU:** two 4K monitors, ~1.4 GB VRAM permanently. `hipMemGetInfo` does not see it;
  the VRAM budget comes from `/sys/class/drm/card*/device/mem_info_vram_used`. A GPU reset takes the desktop with it.
- Several agent sessions and the desktop use RAM; the engine runs under `systemd-run -p MemoryMax=` (`TONA_RAM_GB`).
- **Transparent huge pages must be `madvise` or `always`** (`/sys/kernel/mm/transparent_hugepage/enabled`; `defrag` the
  same). The reference PC has `enabled=madvise`, `defrag=madvise` (read 2026-11-11 under kernel 7.2, not read again under
  7.3.4). The pinned experts that do not fit the GTT (32.5 GB, half of the RAM; about 4 GB at the service default
  `--ctx 262144`) are registered memory backed with huge pages; with `never`, or without free 2 MB pages, page-cache
  pressure stalls the GPU queues for seconds per request (idle server, 4.1 GB: 29.6 s of evicted time per 120 s of
  pressure with 4 KB pages, 1.8 s with huge pages, `research/measurements/2026-11-11-evict.md`). The start log names the
  mode and the share on huge pages and warns below 90 %. Other distributions: not checked.

## Ceiling and yardsticks

| | Decode | Prefill | Source |
|---|---:|---:|---|
| This engine, bench-set code, 64 GB RAM | 89.2 tok/s | 1,741 tok/s (13K prompt) | 2026-10-27-ram-limit |
| Ceiling of this platform | ~70 passes/s (3.52 GB dense + 0.98 GB experts per pass at 315 GB/s; the dense bytes are the GGUF's 4.11 GB with the hyper-connection down/up weights as Q8_0 instead of BF16, `tools/gguf_info.py --tensors` and `engine/gpu/arch/qwen4exp/residual.cpp`), with MTP (~2.5 tokens/pass) ~175 tok/s | 93 TFLOPS f16 / 186 fp8 | research/06, 2026-10-22-micro |

The binding limit for decode is VRAM bandwidth (315 GB/s), not RAM or PCIe. Prefill is compute-bound.
