# PC with 24 or 32 GB VRAM, 96 or 128 GB RAM

Status: **concept only**; the reference machine cannot emulate a larger card. Method and source marks:
[README.md](README.md#how-the-files-estimate).

## Typical cards

| Card | VRAM | Arch / target | Bandwidth | Backend | Mark |
|---|---:|---|---:|---|---|
| RX 7900 XTX | 24 GB | RDNA3 gfx1100 | 960 GB/s | new HIP backend; gfx1100 is on ROCm's list | [V] list, [K] bandwidth |
| RX 7900 XT | 20 GB | RDNA3 gfx1100 | 800 GB/s | as above | [V] list, [K] |
| RTX 3090 / 4090 | 24 GB | Ampere / Ada | 936 / 1,008 GB/s | CUDA backend | [K] |
| RTX 5090 | 32 GB | Blackwell | 1,792 GB/s | CUDA backend | [K] |
| Radeon AI PRO R9700 | 32 GB | RDNA4 gfx1201 | 640 GB/s | the reference code with a second target, unverified; the card is on ROCm's PRO list | [V] list, [K] bandwidth |
| Arc Pro B70 | 32 GB | Intel Xe2 (Battlemage) | 608 GB/s | new SYCL backend, see [intel-xe2.md](../backends/intel-xe2.md); nothing run by us | [V] spec, [S] benchmarks |

## Budget

| | 24 GB card | 32 GB card |
|---|---:|---:|
| VRAM usable (desktop 1.4 GB, reserve 1.0 GB) | about 22 GB | about 30 GB (31.4 GB headless) |
| Large model fixed (dense 3.52 GB, KV and buffers 1.5 GB, estimate) | 5.0 GB | 5.0 GB |
| VRAM left for experts | about 17.0 GB = 68 % of 25.15 GB | all 25.15 GB fit, about 0 to 1.3 GB spare |
| RAM for the rest | 64 GB is enough (all experts pinned as backup); 96 or 128 GB lets the page cache keep the n-gram table (28.8 GB) | same |

The large model's trunk is 29.6 GB (research/21); 128K context adds 1.6 GB KV (int8). A 32 GB card with the desktop on it
does not hold both; headless or with the desktop on the iGPU it does.

## Model tier

| Class | Large | Medium | Small | Bigger models |
|---|---|---|---|---|
| 24 GB VRAM, 64 to 128 GB RAM | yes, 65 % of experts in VRAM | Gyro-Coder 23.5 GB at 128K does not fit with the desktop, fits at 32K; Qwen3.8-27B IQ3_XXS with KV fits | all in VRAM | Swift 1.5 IQ2_XS (68 GB file) with 96 GB RAM |
| 32 GB VRAM, 96 to 128 GB RAM | **entirely in VRAM** | fits | fits | Swift 1.5; Laguna S 2.1 (118B-A8B, only a 96 GB Q4_K_M exists, own quantization needed) with 128 GB RAM |

Bigger models (research/08-models.md) were out of reach on the reference; quality is a vendor claim. Swift is under "Swift
Open License 1.0" on a Qwen Community License base (research/42-model-licenses.md).

## Expert strategy

- **24 GB:** 65 % of experts resident in VRAM, the rest pinned in RAM; the CPU path handles about 7 % of the expert
  reads per token (0.07 GB).
- **32 GB:** all experts resident; no CPU expert path, no pinning, no PCIe streaming in prefill.
- **96/128 GB RAM:** pins the large tier whole; the page cache holds the n-gram table and a second model.

## Estimates

Same method as the other PC files; hit rate at 68 % cached is 0.94. The gap between method (16.2 ms per pass) and the
reference measurement (2.5 tokens in 28 ms, 89 tok/s) reads two ways: a factor 0.58 on everything (upper), or a fixed 12 ms
per pass that does not shrink with a faster card (lower). Truth lies between; the 12 ms is a fit to one measurement.

| Card | GPU part | CPU misses | Upper (0.58 factor) | Lower (+12 ms fixed) |
|---|---:|---:|---:|---:|
| RX 7900 XTX, 24 GB | (3.52 + 0.91) / 960 = 4.6 ms | 1.2 ms | 250 tok/s | **about 140 tok/s** |
| RTX 4090, 24 GB | 4.5 ms | 1.0 ms | 265 | about 145 |
| R9700, 32 GB, all resident | 7.0 ms | 0 | 205 | **about 135** ([CU scaling test](../../research/measurements/2026-11-17-cu-scaling.md): about 145 for code) |
| RTX 5090, 32 GB, all resident | 2.5 ms | 0 | 580 | about 175 |

Decode only, MTP at 2.5 tokens per pass. The rate is bandwidth-bound only until the fixed per-pass cost dominates. Grid
syncs cost 0.75 µs ([M] 2026-10-22-micro-persist), about 0.8 ms per pass, so the 12 ms fit is mostly something else
(attention, indexer, CPU interplay); open question.

Prefill (estimate): compute-bound when all experts are resident (32 GB class). The reference measures 93 TFLOPS f16 WMMA
and 1,741 tok/s with a third of the experts resident; the 5090 and 4090 have several times the tensor throughput [K], so
**several thousand tok/s** is plausible, unmeasured. On the 24 GB class streaming the remaining 35 % of the experts costs
8.8 GB / 25 GB/s (4.0 x16) = 0.35 s per chunk.

Third-party: llama.cpp on an RX 7900 XT 20 GB, Ryzen 9 5900X, 32 GB DDR4-3200, gpt-oss-20b: no MoE offload about 94 tok/s,
`-ncmoe 4` about 60, `-ncmoe 16` about 25 to 27, all on the CPU about 20 (discussion 15396, [V]): decode falls steeply per
layer of experts moved to DDR4 CPU.

Strix Halo machines (Ryzen AI Max, 128 GB shared memory, gfx1151) are a different class: LPDDR5X about 256 GB/s [K], no PCIe,
no CPU path; closer to the Mac files; unchecked.

## What we can reproduce, what only real hardware shows

| Reproducible | Only real hardware shows |
|---|---|
| nothing of the 24/32 GB budget (the reference cannot get more VRAM) | everything: bandwidth, the 65 % and 100 % resident cases, prefill without streaming |
| the CPU path at 7 % misses: `TONA_VRAM_GB=16` | the shape of the curve above a third cached |

The hit rate at 50 and 65 % of slots can be measured on the reference with routing traces (counters); replace the anchor
with it.


## First steps for a contributor

1. 24 GB gfx1100: build the micro-benchmarks, then the reference kernels with `--offload-arch=gfx1100`; the first failures
   name missing instructions (FP8, int4 WMMA). Port in the order of [backends/README.md](../backends/README.md#checks).
2. NVIDIA: new CUDA backend. Start with the pass interface in `engine/backend/backend.h`, run the CPU cross-check first.
3. 32 GB cards: set `TONA_EXPERT_SLOTS` to all experts and measure decode without the CPU path (the simplest
   configuration).
