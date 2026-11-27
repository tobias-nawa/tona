# 2026-10-22: Microbenchmarks gfx1200

Generated automatically by `bench/micro/run.py`. Raw data: [2026-10-22-micro-persist.json](2026-10-22-micro-persist.json).

## Environment

| | |
|---|---|
| GPU | AMD Radeon RX 9060 XT (gfx1200), 16 CUs, clock according to driver 2780 MHz, memory bus 128 bit, L2 4 MiB |
| HIP | runtime 71626385, header 71626385 |
| Repo | 6502bfd |
| Kernel / CPU | 7.2.11-200.fc44.x86_64 / AMD Ryzen 5 9600X 6-Core Processor, governor performance |
| RAM | 30.5 GiB |
| GPU before start | 1647 MiB VRAM in use (desktop), DPM auto |
| GPU during `persist` | clock mean 2,315 MHz, power mean 28.0 W, temp. max 38.0 °C, load 100 % |

Median (min–max) and σ over the repetitions. Bandwidth in GB/s is decimal (10⁹).

## Synchronization inside a running kernel

| Measurement | blocks | Median | Min–Max | σ | Unit |
|---|---:|---:|---:|---:|---|
| grid.sync() (cooperative groups) | 16.0 | 0.78 | 0.78–0.79 | 0.00 | us/barrier |
| own barrier (atomics) | 16.0 | 1.10 | 1.09–1.11 | 0.01 | us/barrier |
| grid.sync() (cooperative groups) | 32.0 | 0.77 | 0.76–0.78 | 0.01 | us/barrier |
| own barrier (atomics) | 32.0 | 1.07 | 1.06–1.09 | 0.01 | us/barrier |
| grid.sync() (cooperative groups) | 64.0 | 0.74 | 0.74–0.75 | 0.00 | us/barrier |
| own barrier (atomics) | 64.0 | 1.04 | 1.04–1.10 | 0.02 | us/barrier |
| grid.sync() (cooperative groups) | 128 | 0.75 | 0.75–0.75 | 0.00 | us/barrier |
| own barrier (atomics) | 128 | 1.07 | 1.07–1.08 | 0.01 | us/barrier |
| block to block (half round trip) | 2.00 | 0.28 | 0.25–0.29 | 0.01 | us |
| CPU <-> GPU via host memory (half round trip) | 1.00 | 0.73 | 0.73–0.74 | 0.00 | us |

## Assessment (by hand, 2026-10-22)

Own measurement, 7 repetitions. Comparison with the kernel launch from `2026-10-22-micro.md`.

| Synchronization | Cost | against kernel launch (2.1 µs) or launch+sync (17.7 µs) |
|---|---:|---|
| `grid.sync()` over up to 128 blocks | **0.75 µs** | 2.8× cheaper than a kernel launch, independent of the block count |
| own atomic barrier | 1.07 µs | slower than `grid.sync()` (loop with `s_sleep`), so not needed |
| block signals block | **0.28 µs** | 7.5× cheaper than a kernel launch |
| CPU signals GPU or GPU signals CPU, running kernel, host memory | **0.73 µs** | **24× cheaper** than launch+sync |

**Consequences for the engine:**

- **The megakernel pays off.** Instead of ~500 kernel launches per decode pass (~1 ms), ~500
  dependencies between blocks cost ~0.14 ms (one signal per dependency) or ~0.4 ms (global barrier).
  Data flow with signals per task (like Mirage MPK) is cheaper than global barriers.
- **CPU experts without a kernel boundary.** The CPU can report results into a running kernel:
  48 layers × 2 signals × 0.73 µs ≈ 0.07 ms per pass instead of ~0.85 ms with launch+sync.
- **Limits of the measurement:** One block pair and one CPU thread, card otherwise empty. Under load
  (memory traffic, many waiters) the latencies rise; only the real kernel measures that.
  The own barrier was only checked for completion, not for correctness; the two ping-pongs
  check themselves (wrong value = abort).
