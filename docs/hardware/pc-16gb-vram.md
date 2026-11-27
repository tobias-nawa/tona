# PC with 16 GB VRAM

16 GB cards with 16, 32 and 64 GB of RAM; the 64 GB row is the reference platform
([reference-platform.md](../backends/reference-platform.md)). Method and source marks: [README.md](README.md#how-the-files-estimate).

## Typical cards

| Card | Arch / target | Memory bandwidth | PCIe | Backend | Mark |
|---|---|---:|---|---|---|
| RX 9060 XT 16 GB | RDNA4 gfx1200 | 315 GB/s measured (vendor 322) | 5.0 x16 | **reference** | [M] |
| RX 9070 / 9070 XT | RDNA4 gfx1201 | 640 GB/s | 5.0 x16 | probably the reference code with a second target; unverified | [K] |
| RX 7800 XT, RX 7900 GRE | RDNA3 gfx1101 / gfx1100 | 624 / 576 GB/s | 4.0 x16 | new backend (WMMA shapes differ, no FP8) | [K] |
| RX 6800 / 6800 XT / 6900 XT | RDNA2 gfx1030 | 512 GB/s | 4.0 x16 | new backend; only the PRO W6800 and V620 are in ROCm's support list | [V] list, [K] bandwidth |
| RTX 4060 Ti 16 GB / 5060 Ti 16 GB | Ada / Blackwell | 288 / 448 GB/s | 4.0 x8 / 5.0 x8 | CUDA backend | [K] |
| RTX 4070 Ti Super, 4080, 5070 Ti, 5080 | Ada / Blackwell | 672 / 717 / 896 / 960 GB/s | x16 | CUDA backend | [K] |

ROCm supports gfx1200, gfx1201, gfx1100 and gfx1101 consumer cards on Linux; RDNA2 consumer cards are absent ([V],
<https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html>, 2026-11-11). [K] figures are vendor datasheet values
from memory.

## Budget

| | Value | Source |
|---|---|---|
| VRAM for weights, KV and buffers | about 13.6 GB (16 GB minus 1.4 GB desktop minus the 1024 MB `TONA_VRAM_RESERVE_MB`; the driver reports 16,304 MiB usable) | reference-platform.md |
| Fixed VRAM use | 3.52 GB dense weights (4.11 GB in the GGUF, hyper-connection weights as Q8_0), 1.5 GB KV and buffers (estimate) | [M] gguf_info of shard 1, `qwen4exp/residual.cpp` |
| VRAM left for expert slots | about 8.4 GB = 86 of 256 slots per layer = 34 % of the 25.15 GB of experts (the 2026-10-27 measurement quotes about 36 %) | derived; [slot size](../architecture/vram-giveback.md) |
| RAM held by the engine | 1.93 GB fixed, plus 25.15 GB if all experts are pinned | [M] 2026-10-27-ram-limit |
| Reserve for OS, desktop, agents | about 8 GB on a 32 GB machine (the pin rule leaves it) | [M] backend/budget.h, ram-limit |

## Model tier

| RAM | Large (ISTA Coder IQ1_M, 58.4 GB file) | Medium (Gyro-Coder 21.4 GB, Qwen3.8-27B IQ3_XXS 10.1 GB) | Small (Groot-19B Q4_K_M, Ornith-1.5-35B-A3B) |
|---|---|---|---|
| 64 GB | **yes, all experts pinned** | not needed | fits; Ornith is the offered alternative for coding (Rust) |
| 32 GB | yes, experts via page cache | yes, entirely in VRAM plus 7 GB RAM | yes |
| 16 GB | possible, slow (SSD-bound) | recommended | Groot Q4_K_M with 128K KV fits in VRAM alone (13.0 GB) |

Medium and small rows follow research/21 ("Fit on both platforms"); their quality is unmeasured by us. What the installer
offers is in [data/modeldb/offer.json](../../data/modeldb/offer.json) ([how the file works](../../data/modeldb/README.md#offer-data-what-the-installer-and-the-client-show)), not this table.

## 64 GB RAM

**Status: reference.** All 25.15 GB of experts are pinned in RAM; VRAM holds the dense weights and about a third of the
experts as slots; the CPU computes missing experts with AVX-512; PCIe 5.0 x16 moves experts for prefill.

- Decode: **89.2 tok/s** (bench-set code, with MTP), 64.6 on text, 62.9 after a 13K prompt [M]
  (research/measurements/2026-10-27-ram-limit.md).
- Prefill: **1,741 tok/s** (13K prompt) [M].
- Ceiling: about 70 passes/s at 315 GB/s (3.52 GB dense + 0.98 GB experts), about 175 tok/s with MTP.
- Estimate check: hit rate 86 %, 13.8 ms GPU (3.52 GB dense plus 0.84 GB experts at 315 GB/s) plus 2.4 ms CPU misses,
  62 passes/s, times 2.5 tokens times 0.58 = 89 tok/s. This is the calibration, so it agrees by construction.

## 32 GB RAM

**Status: emulated on the reference** with `systemd-run -p MemoryMax=` / `TONA_RAM_GB`; measured.

- Usable RAM next to desktop and agents is about 26 to 29 GB. `engine/backend/budget.h` pins all experts if they fit; with
  a usage profile and at least 70 % fitting it pins the most used (`TONA_TIERS`, on by default: 26 GB class +40 to +100 %
  decode, +70 to +90 % prefill, 2026-11-15-expert-tiers); below 70 % nothing is pinned (pins and page cache displace each
  other). The 26 GB figures below predate this.
- Measured at 16 GB VRAM (decode code / text, prefill): limit 26 GB **50.3** / 35.9 / 659; 29 GB 53.1 / 36.2 / 770; 32 GB
  (all pinned) 89.2 / 64.6 / 1,741 [M] (2026-10-27-ram-limit). The step from 53 to 89 lies between 29 and 32 GB, where all
  experts fit.
- A real 32 GB machine has less usable RAM than a 32 GB cgroup if the desktop is heavy.

## 16 GB RAM

**Status: emulated, not measured** (`MemoryMax=12G`, `TONA_RAM_GB=12`).

- Usable RAM about 10 to 12 GB (16 minus OS and desktop 3 to 4, minus 1.93 fixed); the 25 GB of experts come from the
  page cache and the NVMe.
- Decode estimate: the reference at a 20 GB limit with nothing pinned measured 39.7 (code) / 35.5 (text) [M]. At 10 to 12 GB
  of page cache about half of the 14 % VRAM misses (0.137 GB per token) hit it; the other 0.07 GB per token at 2.65 GB/s
  sequential NVMe is 26 ms, so a pass costs 13.8 + 26 + CPU time = about 42 ms, 24 passes/s, times 2.5 times 0.58 =
  **about 35 tok/s upper end; 20 to 30 expected**. Estimate.
- Prefill: bound by reading the experts a chunk touches (all of them for a few thousand tokens): 25 GB at 2.65 GB/s is
  9.5 s per chunk, so **about 200 to 600 tok/s** with VRAM and cache holding part. Estimate.
- The medium tier is the better product here: Gyro-Coder (21.4 GB) fits in VRAM plus 7 GB RAM and reads nothing from the
  SSD per token (decode bound 67 to 78 passes/s, research/21), but needs the trellis dequant kernel.

## Expert strategy

| RAM | Experts | Why |
|---|---|---|
| 64 GB | about a third resident in VRAM, all pinned in RAM | measured best |
| 32 GB | all pinned if 31.6 GB limit, else page cache | rule "all or none" |
| 16 GB | page cache plus NVMe, VRAM slots with the best reuse | no pinning possible |

## Other 16 GB cards

- **gfx1201 (9070 series):** same RDNA4 ISA family, most kernels may carry over with a second build target; 640 GB/s
  doubles the GPU part. Estimate (hit rate 86 %): 6.8 ms GPU plus 2.4 ms CPU misses = 109 passes/s, 157 tok/s with the
  0.58 factor (about 120 tok/s with a fixed 12 ms per pass instead, see [pc-24gb-and-larger.md](pc-24gb-and-larger.md#estimates)).
  Fixed per-pass costs (about 1,100 grid barriers) do not halve: read as an upper bound.
  [2026-11-17-cu-scaling.md](../../research/measurements/2026-11-17-cu-scaling.md) puts code decode at about 145 tok/s on
  the R9700 (gfx1201, 64 CUs); a 9070-class card has fewer CUs.
- **RDNA3 (gfx1100/1101):** WMMA exists (16x16x16 f16/bf16/int8), FP8 and the RDNA4 int4 path do not; FP8 kernels need a
  rewrite. PCIe 4.0 x16 (about 25 GB/s usable [K]) halves expert streaming in prefill.
- **NVIDIA 16 GB:** needs a CUDA backend. The 4060 Ti and 5060 Ti sit on x8 links (PCIe 4.0 x8 about 14 GB/s usable [K]):
  expert streaming for prefill binds, 25 GB takes 1.8 s per window.
- **CPU side:** missing experts are computed with AVX-512 (Zen 4/5; not Zen 3 or Intel 12th to 14th generation, which need
  an AVX2 expert kernel). DDR4-3200 dual channel (about 40 GB/s [K]) makes the CPU part 1.5x slower than 58 GB/s.

## What we can reproduce, what only real hardware shows

| Reproducible on the reference | Only real hardware shows |
|---|---|
| RAM limit 16, 32 GB (`MemoryMax`, `TONA_RAM_GB`), pin rule, page cache | real desktop and agent RAM use, swap, DDR4 speed |
| VRAM budget up to 16 GB (`TONA_VRAM_GB`) | bandwidth of other cards, PCIe 4.0 or x8 links |
| Slower SSD (cgroup `io.max`, [K]) | other NVMe drives, SATA SSDs |


## First steps for a contributor

1. 16 GB RAM: run `scripts/bench-set.py` under `systemd-run -p MemoryMax=12G` with `TONA_RAM_GB=12` and no pinning
   (`TONA_PIN_GB`, [environment.md](../use/environment.md)), page cache cleared before each run, three rounds, plan first
   (docs/contribute/measurement.md). Fill the 16 GB row.
2. gfx1201: build with the second target only after `rocminfo` shows it; run `tona-gpu-gen --check`
   ([backends/README.md](../backends/README.md#checks)), then the bench set.
3. NVIDIA / RDNA3: start from [backends/README.md](../backends/README.md); measure the micro-benchmarks of
   `research/measurements/2026-10-22-micro.md` on your card (bandwidth, WMMA throughput, grid sync) and write the table
   of [reference-platform.md](../backends/reference-platform.md) for it.
