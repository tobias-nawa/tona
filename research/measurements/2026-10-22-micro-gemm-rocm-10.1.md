# 2026-10-22: Microbenchmarks gfx1200

Generated automatically by `bench/micro/run.py`. Raw data: [2026-10-22-micro-gemm-rocm-10.1.json](2026-10-22-micro-gemm-rocm-10.1.json).

## Environment

| | |
|---|---|
| GPU | AMD Radeon RX 9060 XT (gfx1200), 16 CUs, clock according to driver 2780 MHz, memory bus 128 bit, L2 4 MiB |
| HIP | runtime 71626385, header 71626385 |
| Repo | b0e7892 (modified) |
| Kernel / CPU | 7.2.11-200.fc44.x86_64 / AMD Ryzen 5 9600X 6-Core Processor, governor performance |
| RAM | 30.5 GiB |
| GPU before start | 1824 MiB VRAM in use (desktop), DPM auto |
| GPU during `gemm` | clock mean 2,622 MHz, power mean 138 W, temp. max 77.0 °C, load 91.8 % |

## Warnings

- Repository has uncommitted changes

Median (min–max) and σ over the repetitions. Bandwidth in GB/s is decimal (10⁹).

## GEMM via hipBLAS (A, B f16/bf16, compute f32)

| Measurement |  | Median | Min–Max | σ | Unit |
|---|---:|---:|---:|---:|---|
| f16 8192x4096x4096 (square 4096) |  | 49.6 | 44.6–52.6 | 2.93 | TFLOPS |
| f16 8192x2560x2560 (projection 2560) |  | 56.0 | 53.9–58.2 | 1.44 | TFLOPS |
| f16 8192x1280x2560 (expert gate/up 2560→1280) |  | 27.7 | 27.4–27.9 | 0.17 | TFLOPS |
| f16 8192x2560x640 (expert down 640→2560) |  | 48.0 | 47.6–50.1 | 0.98 | TFLOPS |
| f16 512x640x2560 (expert, 512 tokens) |  | 49.3 | 48.9–49.4 | 0.15 | TFLOPS |
| f16 8192x8192x8192 (large 8192) |  | 44.7 | 44.5–45.1 | 0.19 | TFLOPS |
| bf16 8192x4096x4096 (square 4096) |  | 42.5 | 42.4–43.3 | 0.37 | TFLOPS |
| bf16 8192x2560x2560 (projection 2560) |  | 50.0 | 43.7–54.5 | 3.59 | TFLOPS |
| bf16 8192x1280x2560 (expert gate/up 2560→1280) |  | 28.1 | 27.5–28.6 | 0.40 | TFLOPS |
| bf16 8192x2560x640 (expert down 640→2560) |  | 45.4 | 45.0–48.9 | 1.70 | TFLOPS |
| bf16 512x640x2560 (expert, 512 tokens) |  | 49.1 | 47.5–49.3 | 0.76 | TFLOPS |
| bf16 8192x8192x8192 (large 8192) |  | 44.8 | 44.5–47.2 | 0.97 | TFLOPS |

## Assessment (by hand, 2026-10-22)

- hipBLAS → hipBLASLt (ROCm 10.1, hipBLASLt 1.4.1) reaches **45–56 TFLOPS** at f16, i.e. **~50–60 %**
  of the measured WMMA peak (93 TFLOPS). The shape 8192×1280×2560 drops to **28 TFLOPS** (~30 %):
  a tuned kernel apparently does not exist for this shape.
- Control `ROCBLAS_USE_HIPBLASLT=0` (rocBLAS/Tensile without hipBLASLt): **only ~7 TFLOPS** for all
  shapes. Without the switch, ROCm 10.1 already uses hipBLASLt (same values as `=1`).
- Clock during GEMM only ~2.6 GHz at 138 W, against ~3.1 GHz in the pure WMMA test. Part of the
  gap is clock, not only kernel quality.
- **Consequence for prefill:** Own WMMA kernels have up to ~2× headroom over the library, up to ~3×
  for expert shapes, and another 2× with fp8/int8.
- Comparison with Fedora's ROCm 7.1.1 open: `hipblas-devel` is missing for it.
