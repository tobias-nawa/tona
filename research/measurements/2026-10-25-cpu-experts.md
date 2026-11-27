# CPU experts with 512-bit kernels, 2026-10-25, own measurement

Ryzen 5 9600X (Zen 5, full 512-bit units), Coder IQ1_M, 80 expert slots (desktop used 2.1 GB VRAM).

## Upper bound first (throughput test B = 4, `IE_CPU_FAKE=1`: CPU experts not computed, output meaningless)

| CPU experts | ms per pass | GPU direct / CPU per pass |
|---|---:|---:|
| real, planner 80 µs | 39.42 | 62 / 115 |
| free, planner 80 µs | 39.20 | 60 / 117 |
| free, planner 40 µs | 37.25 | 25 / 161 |
| free, planner 20 µs | 36.43 | 10 / 174 |

Waiting for the CPU costs only 0.2 ms; there is only something to gain if the CPU computes faster and thereby takes
over more misses that would otherwise come over PCIe.

## Kernels (1 thread, one matrix, 1 token; `test_cpu_experts`)

Two sub-blocks (64 values) per zmm instead of one per ymm; signs of IQ3_XXS from the parity (pdep + folding) instead
of tables. Research (agent, ik_llama.cpp `iqk_gemm_iquants.cpp`, PR #2633, #2676): there too, single table lookups,
gain from integer scales, sign masks, row repack; up to 1.35× on Zen 4.

| Format (matrix) | before µs | now µs | Variants |
|---|---:|---:|---|
| IQ3_XXS gate 640 × 2560 | 118 | 81 | gather 105 |
| IQ2_S gate | 102 | 84 | scalar index per value 195, gather 91, vector index via the stack 84 |
| IQ3_S gate | 136 | 116 | gather 122, via the stack 126 |
| IQ4_XS gate | 58 | 47 | |
| IQ4_NL down 2560 × 640 | 48 | 38 | |
| Q2_0 down | 40 | 21 | |

First version without `always_inline`: IQ3_XXS 117 µs, IQ2_S 265 µs (decoding not inlined, `Dec` via memory).
Deviation from the reference in double ≤ 6e-8 per matrix, whole experts ≤ 2.2e-7.

Whole experts (all threads, 4 tokens, µs per expert): layer 0 71 / 77 → 50 / 54, layer 1 62 / 65 → 43 / 47,
layer 17 73 / 78 → 61 / 69, layer 47 44 / 50 → 32 / 37 (6 / 12 experts each).

## In the pass (throughput test B = 4)

| Planner CPU µs | ms per pass | GPU direct / CPU per pass, CPU time |
|---|---:|---|
| 80 | 39.15 | 65 / 118, 9.0 ms |
| 60 | 39.08 | 46 / 136, 10.2 ms |
| 50 | 39.32 | 35 / 147, 10.9 ms |
| 40 | 39.89 | 26 / 158, 11.7 ms |

Before 39.42 ms (80 µs, 10.8 ms CPU for 115 experts). In the pass the CPU computes only ~1.2× faster: CPU experts
and GPU-direct both read from RAM (~28 GB/s for the CPU alone, measured 58 GB/s reading).

## Server (median of 3, 2 runs each; with drafter calibration)

| Decode tok/s | old kernel (80) | new, 80 | new, 60 |
|---|---:|---:|---:|
| csv | 89.2 / 89.2 | 88.9 / 88.6 | 90.0 / 89.9 |
| rust | 77.8 / 78.1 | 76.2 / 75.4 | 76.0 / 75.9 |
| explain | 56.2 / 56.4 | 59.3 / 59.1 | 60.1 / 59.9 |
| after 13K | 53.8 / 52.9 | 55.1 / 53.7 | 52.3 / 54.6 |
| mean | 69.2 | 69.5 | 70.0 |

The CPU rounds differently than before, so the generated text changes and with it the acceptance of the drafts (rust −2,
explain +3.5). Teacher NLL (`--slots 80`, 1,500 tokens) 2.186663 → 2.189690 (80) / 2.192799 (60), within the
fluctuation caused by the CPU/GPU split. Planner default 80 → 60 µs.
