# PC with 12 GB VRAM

12 GB cards with 16, 32 and 64 GB of RAM. Status: **budget emulated** on the reference machine (`TONA_VRAM_GB=12`), not
bandwidth or ISA; **concept only** for every card other than the RX 9060 XT. Method and source marks:
[README.md](README.md#how-the-files-estimate).

## Typical cards

| Card | Arch / target | Bandwidth | PCIe | Backend | Mark |
|---|---|---:|---|---|---|
| RTX 3060 12 GB | Ampere | 360 GB/s | 4.0 x16 | CUDA backend | [K] |
| RTX 4070 / 4070 Super | Ada | 504 GB/s | 4.0 x16 | CUDA backend | [K] |
| RTX 5070 | Blackwell | 672 GB/s | 5.0 x16 | CUDA backend | [K] |
| RX 7700 XT | RDNA3 gfx1101 | 432 GB/s | 4.0 x16 | new HIP backend; gfx1101 is on ROCm's support list | [V] list, [K] bandwidth |
| RX 6700 XT | RDNA2 gfx1031 | 384 GB/s | 4.0 x16 | not on ROCm's consumer list; unofficial only | [V] list, [K] bandwidth |
| Arc B580 | Xe2 | 456 GB/s | 4.0 x8 | SYCL or Vulkan backend, no ROCm | [K] |

ROCm support list: <https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html> [V], 2026-11-11.

## Budget

| | Value | Source |
|---|---|---|
| VRAM usable with a desktop on the card | about 9.6 GB (12 GB minus 1.4 GB desktop, minus 1.0 GB `TONA_VRAM_RESERVE_MB`) | derived from reference-platform.md |
| VRAM usable on a headless or iGPU-driven machine | about 11.3 GB | derived |
| Large model: fixed VRAM | 3.52 GB dense (4.11 GB in the GGUF, hyper-connection weights as Q8_0), 1.5 GB KV and buffers (estimate) | [M] gguf_info; KV estimate |
| Large model: VRAM left for experts | about 4.6 GB = 18 % of 25.15 GB | derived |
| RAM | as in [pc-16gb-vram.md](pc-16gb-vram.md#budget): 1.93 GB fixed, 25.15 GB if all experts are pinned | [M] ram-limit |

## Model tier

| RAM | Large (58.4 GB) | Medium | Small |
|---|---|---|---|
| 64 GB | yes, all pinned, 18 % of experts in VRAM | not needed | not needed |
| 32 GB | page cache plus NVMe, slow | Gyro-Coder (21.4 GB): 10 GB VRAM plus 12 GB RAM; 27B IQ3_XXS (10.1 GB) only at 32K context or less (KV 1.07 GB) | **Groot-19B IQ4_XS** (10.9 GB) with 2 GB in RAM; Ornith IQ3_XXS (15.3 GB) with 5 GB in RAM |
| 16 GB | not recommended | Gyro-Coder possible with 10 GB in RAM | **recommended**: Groot IQ4_XS; Ornith IQ3_XXS |

Sizes: research/21, "Model tiers" [V]; quality is unmeasured by us. The small tier needs the qwen35moe path, Gyro-Coder
the trellis dequant kernel.

## Expert strategy

- **Large, 64 GB:** all experts pinned in RAM, 18 % in VRAM slots. Almost one expert in three is computed by the CPU.
- **Large, 16/32 GB:** page cache and NVMe; hit rates below 33 % of experts are not covered by our curve.
- **Small and medium:** shared experts and dense weights in VRAM, routed experts split between VRAM and pinned RAM with
  the same slot and miss machinery (llama.cpp's `--n-cpu-moe` is the static version).

## Estimates

Method: hit rate at 18 % of experts cached extrapolated to about 70 % (curve anchored at 33 %, unchecked below), CPU path
58 GB/s, calibration 0.58 and 2.5 tokens per pass ([README](README.md#how-the-files-estimate)).

| Case | GPU part | CPU part | Passes/s | Decode (estimate) |
|---|---|---|---:|---:|
| RX 9060 XT with 12 GB limit (emulation), 64 GB RAM | (3.52 + 0.69) / 315 = 13.4 ms | 0.29 GB / 58 = 5.1 ms | 54 | **about 78 tok/s** |
| RTX 3060, 64 GB DDR5 | 11.7 ms | 5.1 ms | 60 | about 86 |
| RTX 3060, 64 GB DDR4-3200 (about 40 GB/s [K]) | 11.7 ms | 7.4 ms | 52 | about 76 |
| RTX 4070 | 8.4 ms | 5.1 ms | 74 | about 108, upper bound |
| 32 GB RAM, 9060 XT limit | 13.4 ms | CPU 5 ms plus SSD about 23 ms | 24 | **about 30 to 40** |
| 16 GB RAM, 9060 XT limit | 13.4 ms | CPU 5 ms plus SSD about 66 ms | 12 | **about 12 to 20** |

SSD terms: at 32 GB RAM about 20 of 25 GB of experts stay cached, so 20 % of the 0.29 GB missed per token (0.06 GB) is
read at 2.65 GB/s; at 16 GB about 10 GB stay cached and 60 % (0.17 GB) is read. The reference measured 39.7 and 50.3 tok/s
at 20 and 26 GB limits with 16 GB VRAM ([M] 2026-10-27-ram-limit), which anchors the 32 GB row; the 16 GB row is below
every measurement.

Prefill (estimate): a chunk of a few thousand tokens touches all 25 GB, which pass through the VRAM window over PCIe: 25 GB
at 55 GB/s (5.0 x16) is 0.46 s, at 25 GB/s (4.0 x16) 1.0 s per chunk. With 64 GB RAM expect **about 1,000 to 1,500 tok/s**
on the 9060 XT (the reference with a third of the experts in VRAM measured 1,741), lower on PCIe 4.0. With 32 or 16 GB RAM
the NVMe term of 3 to 9 s per chunk dominates: **about 200 to 700 tok/s**.

Third-party numbers (llama.cpp, their setups): RTX 3060 12 GB, Xeon E5-2650 v4, 31 GB DDR4, Qwen3.6-35B-A3B IQ3_XXS
(13.2 GB), 64K context: 22.9 tok/s at `-ncmoe 25`, 19.7 at `-ncmoe 35`, prefill mixed in ([V], jeanfbrito.github.io,
2026-05-09). RTX 3060 12 GB, Ryzen 7 5700X, gpt-oss-20b MXFP4, `-ncmoe 2`, 16K: about 64 tok/s (llama.cpp discussion
15396, [V]); llama-bench on a 3060 with tensor override: pp2048 2,230, tg128 30.6. A 3B-active model is the small tier; our
6B-active large and medium models are slower per token.

## What we can reproduce, what only real hardware shows

| Reproducible | Only real hardware shows |
|---|---|
| VRAM budget of 12 GB with `TONA_VRAM_GB=12` on the 9060 XT; slot count, pin rule, miss path | bandwidth of the card (315 against 360 to 672 GB/s) |
| RAM 16/32 GB with `MemoryMax` | PCIe 4.0 or x8 behavior of prefill streaming |
| CPU-side path with AVX-512 on Zen 5 | AVX2-only CPUs, DDR4, Intel hybrid cores |
| | ISA-specific kernels (gfx110x WMMA, Ampere/Ada tensor cores) |


## First steps for a contributor

1. Emulation: run `bench-set.py` on a 16 GB card with `TONA_VRAM_GB=12`, RAM 64 GB, then 32 GB and 16 GB (`MemoryMax`);
   three rounds, plan first. Fill the table above.
2. Own a 12 GB card: run the micro-benchmarks of `research/measurements/2026-10-22-micro.md` (bandwidth, WMMA throughput,
   PCIe, in-kernel sync), then follow [backends/README.md](../backends/README.md) for a tier 3 backend.
3. 12 GB with 16 GB RAM: start with the small tier (qwen35moe path), the only tier without SSD in the decode loop.
