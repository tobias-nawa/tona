# 2026-10-22: ROCm 7.1.1 (Fedora) against ROCm 10.1 (wheels)

Evaluation of two microbenchmarks run directly one after the other, 7 repetitions each:
[2026-10-22-micro-rocm-7.1.1.md](2026-10-22-micro-rocm-7.1.1.md) (HIP 7.1.52802, clang 20) and
[2026-10-22-micro-rocm-10.1.md](2026-10-22-micro-rocm-10.1.md) (HIP 7.16.26385, clang 24).
Same source `bench/micro/gpubench.hip`, `bench/micro/run.py --rocm system|venv`.

"Separated" = the min–max ranges do not overlap.

| Measurement | 7.1.1 | 10.1 | 10.1 / 7.1.1 | Separated |
|---|---:|---:|---:|---|
| VRAM read, 1 GiB | 315.5 GB/s | 315.6 GB/s | 1.00 | |
| Infinity Cache, 16 MiB | 1,183 GB/s | 1,187 GB/s | 1.00 | |
| PCIe H2D, 1 GiB | 55.73 GB/s | 55.91 GB/s | 1.003 | yes |
| PCIe both directions | 62.2 GB/s | 60.4 GB/s | 0.97 | yes |
| Kernels back to back | 2.19 µs | 2.17 µs | 0.99 | |
| HIP Graph | 2.11 µs | 2.09 µs | 0.99 | yes |
| `grid.sync()`, 16 blocks | 0.54 µs | 0.55 µs | 1.02 | |
| `grid.sync()`, 128 blocks | **1.09 µs** | **0.55 µs** | **0.50** | yes |
| own barrier, 128 blocks | 0.95 µs | 0.80 µs | 0.84 | yes |
| CPU ↔ GPU via host memory | 0.69 µs | 0.65 µs | 0.95 | |
| WMMA f16 → f32 | 92.2 TFLOPS | 95.5 TFLOPS | 1.04 | |
| WMMA fp8 → f32 | 185.6 TFLOPS | 181.4 TFLOPS | 0.98 | |
| WMMA int8 | 179.3 TOPS | 182.2 TOPS | 1.02 | |
| FMA f32 | 24.0 TFLOPS | 24.3 TFLOPS | 1.01 | |

**Result:**

- Hardware-near values (bandwidth, WMMA, kernel launch) are equal: the driver and runtime layer
  does not slow things down on 7.1.1.
- **10.1 is clearly better at the grid synchronization:** `grid.sync()` stays at 0.55 µs up to 128 blocks,
  7.1.1 doubles to 1.09 µs. The compiler (clang 24) makes the own barrier 16–20 % faster.
  Both matter for the megakernel.
- The synchronization values vary between runs (0.55 here against 0.75 µs in the run
  [2026-10-22-micro-persist.md](2026-10-22-micro-persist.md)). The clock in this part was now at
  2.97 GHz (10.1) against 2.32 GHz (7.1.1): light load lets the card clock down. For
  latency-critical decode the DPM state is a lever of its own (`power_dpm_force_performance_level`),
  not yet measured.
- GEMM library: measured with 10.1 only ([2026-10-22-micro-gemm-rocm-10.1.md](2026-10-22-micro-gemm-rocm-10.1.md)),
  `hipblas-devel` is missing for 7.1.1.

**Decision stands:** ROCm 10.1 as the engine's toolchain.
