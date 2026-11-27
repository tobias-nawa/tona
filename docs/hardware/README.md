# Hardware classes

Tona is optimized for one reference platform ([reference-platform.md](../backends/reference-platform.md)). Every other class below
is a **concept** for a contributor who owns that hardware: its rates are estimates with the derivation shown, none has
been run on the target.

## Classes and status

**reference** = maintainers own it, measured, release-blocking. **emulated** = a limit of the reference machine
reproduces the memory budget (not the bandwidth). **concept only** = needs other hardware or another backend.

| Class | File | Status | Backend needed (new: [docs/backends/](../backends/cuda.md)) |
|---|---|---|---|
| PC, 16 GB VRAM, 64 GB RAM (RX 9060 XT) | [pc-16gb-vram.md](pc-16gb-vram.md#64-gb-ram) | **reference** | `hip-gfx1200` (exists) |
| PC, 16 GB VRAM, 32 GB RAM | [pc-16gb-vram.md](pc-16gb-vram.md#32-gb-ram) | emulated (`MemoryMax`), measured on the reference | `hip-gfx1200` |
| PC, 16 GB VRAM, 16 GB RAM | [pc-16gb-vram.md](pc-16gb-vram.md#16-gb-ram) | emulated, not yet measured | `hip-gfx1200` |
| PC, 16 GB VRAM, other cards (RDNA3/RDNA4 gfx1201, NVIDIA) | [pc-16gb-vram.md](pc-16gb-vram.md#other-16-gb-cards) | concept only | gfx1201 probably the same code; gfx110x, CUDA new |
| PC, 12 GB VRAM, 16 / 32 / 64 GB RAM | [pc-12gb-vram.md](pc-12gb-vram.md) | budget emulated (`TONA_VRAM_GB=12`), bandwidth not | gfx1101 / CUDA / Arc: new |
| PC, 24 / 32 GB VRAM, 96 / 128 GB RAM | [pc-24gb-and-larger.md](pc-24gb-and-larger.md) | concept only (24 GB: budget partly emulated) | gfx1100 / gfx1201 / CUDA |
| Apple silicon, M1 to M6 | [mac-chips.md](mac-chips.md) | reference data for the Mac files | |
| Mac, 32 GB (M1 Max) | [mac-32gb.md](mac-32gb.md) | **emulated**: maintainers own an M1 Max 32 GB | `metal` (in development) |
| Mac, 16 / 24 GB | [mac-16-24gb.md](mac-16-24gb.md) | emulated by a hard budget on the M1 Max, bandwidth not | `metal` |
| Mac, 64 GB and larger | [mac-64gb-and-larger.md](mac-64gb-and-larger.md) | concept only | `metal` |

A class that needs a new backend: start with [blueprint.md](../backends/blueprint.md) and the page of your
platform ([CUDA](../backends/cuda.md), [other AMD](../backends/hip-other-amd.md), [Metal](../backends/metal.md),
[Vulkan](../backends/vulkan.md)).

Model tiers and candidates: [research/21-backends.md](../../research/21-backends.md) (section "Model tiers"),
[research/08-models.md](../../research/08-models.md); licences [research/42-model-licenses.md](../../research/42-model-licenses.md).
Backend rules and support tiers: [backends/README.md](../backends/README.md).

## How the files estimate

Same method in every file. All speeds are decode unless "prefill" stands next to them; every estimate is marked
**estimate**.

1. **Budget** (weights plus KV) = capacity minus what the desktop or OS keeps. PC: VRAM minus about 1.4 GB desktop and
   `TONA_VRAM_RESERVE_MB` (1024 MB); RAM minus OS, desktop and agent sessions (`TONA_RAM_GB`). Mac: the GPU working set
   (`recommendedMaxWorkingSetSize`), measured 26.8 GB on 32 GB.
2. **Bytes per pass** = dense weights + used experts + KV read. The large model reads 3.52 GB dense (4.11 GB in the GGUF;
   hyper-connection weights are Q8_0 at load) plus 0.98 GB of experts (480 experts of 2.046 MB), 4.5 GB
   ([reference-platform.md](../backends/reference-platform.md)). KV and tier sizes: research/21, "Model tiers".
3. **Decode bound** (passes/s) = effective bandwidth / bytes per pass. Measured bandwidth where we have it (315 GB/s on
   the RX 9060 XT, 334 GB/s GPU read on the M1 Max = 82 % of peak), else vendor peak (PC) or peak times 82 % (Mac), marked.
4. **Misses**: experts not in VRAM (PC) or the arena (Mac) are computed on the CPU from pinned RAM (58 GB/s measured on
   the one-CCD 9600X, 55 to 57 on one CCD of the 9950X; the reference's two CCDs read 72 GB/s, the estimates keep 58) or
   read from the SSD (13 GB/s for 2 MB blocks at depth 16 on the M1 Max). Hit rate by cache fraction: 33 % cached gives
   about 85 %, 67 % about 94 % (llama.cpp issue 29887 on this model, via research/03-feasibility.md); interpolated,
   unchecked below 33 %.
5. **Calibration**: the reference measures 89.2 tok/s decode (bench-set code, 64 GB RAM, MTP drafts). The method gives 62
   passes/s (3.52 GB dense plus 0.84 GB experts from VRAM = 13.8 ms, plus 2.4 ms CPU misses at 86 % hits); at 2.5 tokens
   per pass that is 154 tok/s, so every PC estimate carries a factor **0.58** (launch, attention, acceptance). Fixed
   costs per pass do not shrink on a faster card, so estimates for cards above ~500 GB/s are upper bounds.
6. **Prefill** is compute-bound while the experts sit in memory and PCIe- or SSD-bound when the whole expert set passes
   through a window; estimates name which binds.

## Source marks

[V] page read in the relevant part on 2026-11-11, [S] search result only, [M] measured by us (file under
`research/measurements/`), [K] general knowledge, not re-checked. Numbers of other projects are theirs.

## Common sources

| Source | Mark | Used for |
|---|---|---|
| <https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html> (2026-11-11) | [V] | ROCm GPU support and gfx targets |
| <https://en.wikipedia.org/wiki/Apple_M1>, `Apple_M2`, `Apple_M3`, `Apple_M4` (2026-11-11) | [V] | bandwidth, GPU cores, maximum RAM |
| <https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/> (2026-03-03) | [V] | M5 Pro and M5 Max |
| <https://github.com/ggml-org/llama.cpp/discussions/15396> (read 2026-11-11) | [V] | llama.cpp MoE offload measurements, RX 7900 XT and RTX 3060 |
| <https://jeanfbrito.github.io/posts/running-qwen3.6-35b-moe-rtx3060-ncmoe/> (2026-05-09, updated 2026-09-14) | [V] | RTX 3060 12 GB, 35B-A3B |

## Contributing a class

1. Read [backends/README.md](../backends/README.md) and the class file for your hardware.
2. Run the "first steps" of that file; open an issue with the log before you write a backend.
3. Measurements go to `research/measurements/` with the plan first ([measurement.md](../contribute/measurement.md)), compared to the
   class estimate. A result outside the estimate is a finding; correct the file in the same pull request.
