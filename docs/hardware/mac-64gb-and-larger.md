# Mac with 64 GB and larger

**Status: concept only** (the maintainers' Mac has 32 GB). Reproducible: the all-resident pass time, since the Phase 0
kernel benchmarks run on synthetic weights of the real shapes. Method: [README.md](README.md#how-the-files-estimate);
chips: [mac-chips.md](mac-chips.md). Backend: `metal`, in development.

Typical machines: 64 GB: M1 Max, M4 Pro, M4 Max, M5 Pro, M1 Ultra; 96 GB: M2 Max, M3 Max; 128 GB: M4 Max, M5 Max, M1 Ultra;
larger: M2 Ultra 192 GB, M3 Ultra 256 or 512 GB (maximum RAM [V] Wikipedia; sold sizes [K]).

## Budget

| | 64 GB (68.7 GB) | 128 GB (137 GB) |
|---|---:|---:|
| GPU working set, default (estimate: 75 %, the measured 78 % on 32 GB is the upper end) | 51 to 53 GB | 103 to 107 GB |
| Left for macOS and apps | 15 GB | 30 GB |
| Large tier: trunk 29.6 GB plus KV 1.6 GB at 128K plus buffers 1.5 GB | 32.7 GB, **fits with 18 GB to spare** | fits |
| n-gram table, 28.8 GB | read from the SSD or the page cache (16 rows of 90 bytes per token) | fits in the page cache |

The working-set figure for these sizes is unchecked.

## Model tier

| RAM | Large (ISTA Coder IQ1_M) | Medium | Small | Bigger models |
|---|---|---|---|---|
| 64 GB | **entirely resident** | yes | yes | Swift 1.5 IQ2_XS (68 GB file) does not fit the working set; with a raised wired limit only |
| 96 GB | resident, the n-gram table in the page cache | yes | yes | Swift 1.5 IQ2_XS (68 GB, Terminal-Bench 2.1 69.7 vendor) fits |
| 128 GB and more | resident with room for 2 conversations of 128K | yes | yes | Swift 1.5, REAP-288 (66 to 116 GB); Laguna S 2.1 (118B-A8B, only a 96 GB Q4_K_M exists, own quantization) |

Bigger models: research/08-models.md (quality figures are vendor claims); licences: research/42-model-licenses.md.

## Expert strategy

All experts resident in the shared buffer; no SSD in the decode loop (the SSD loads the model only). The 32 GB design
(arena, events, direct reads) costs nothing when the arena holds all experts.

## Estimates (decode, passes per second)

The large model reads 5.1 GB per pass. Bound = 0.82 x peak bandwidth / 5.1 GB (82 % measured on the M1 Max only). Anchor
(M1 Max kernels, resident): dense 21.5 ms + experts 10.2 ms + DeltaNet 0.9 ms + small ops about 3 ms = about 36 ms =
**28 passes/s** (research/measurements/2026-11-11-mac-densek.md, -expertk.md; from the parts, no end-to-end run). The kernels
are latency- and ALU-bound as much as bandwidth-bound, so "scaled" multiplies the anchor by the ratio of GPU cores times
bandwidth to the M1 Max, capped at the bound; a guess.

| Chip (RAM) | Peak bandwidth | Bound (passes/s) | Scaled from the anchor |
|---|---:|---:|---:|
| M1 Max (64 GB) | 409.6 GB/s | 66 | **28** (anchor) |
| M4 Pro (64 GB) | 273 GB/s | 44 | about 20 (16 or 20 cores, newer cores faster; unchecked) |
| M5 Pro (64 GB) | 307 GB/s | 49 | about 25 |
| M4 Max (128 GB, 40 cores) | 546 GB/s | 88 | about 35 to 45 |
| M5 Max (128 GB, 40 cores) | 614 GB/s | 99 | about 40 to 50 |
| M1 Ultra (128 GB) | 819 GB/s | 132 | about 40 to 55; two dies, scaling unchecked |

The MTP multiplier on the Mac is not estimated. Tuned expert kernels (about 5 ms instead of 10.2) would gain 14 % at the
anchor (research/41).

External yardsticks: Qwen3-Next-80B-A3B 4-bit, MLX, M1 Max 64 GB: about 43 tok/s decode at 4K, 37 at 8K (oMLX [S]); M3 Max
40 cores: about 60 decode, about 830 prefill (oMLX [S]); Flash-MoE Anemll fork, M5 Max 128 GB, Qwen3.5-397B-A17B Q3 experts
through the page cache: 12.9 tok/s, 4-bit MLX 9.5, 2-bit 14.5 (research/39 [V]). Those models have 3 to 17 B active
parameters; ours 6 B.

Prefill (estimate): compute-bound with all experts resident. M1 generation: about 10 TFLOPS fp16 [K] at 30 to 50 % use is
about 400 tok/s upper end for 12 GFLOP per token, unmeasured. M5 Pro and Max have neural accelerators (Apple: over 4x the
peak GPU compute of M4 [S]); BaseRT reports prefill up to 6.4x over llama.cpp on M5 (arXiv 2607.19438 [S]). No M5 prefill
number of ours.

## What we can reproduce, what only real hardware shows

| Reproducible on the M1 Max 32 GB | Only real hardware shows |
|---|---|
| kernel times of the resident pass with synthetic weights (dense, experts, DeltaNet, dispatches) | the real end-to-end pass with 33 GB resident and the working-set limit of 64 GB or more |
| correctness against the CPU reference on shapes of the real model | other chips' kernel efficiency, dispatch cost, GPU clock |
| | two-die scaling of the Ultra chips, thermal behavior of a Mac Studio, M5 matrix units |


## First steps for a contributor

1. Print `recommendedMaxWorkingSetSize` and `maxBufferLength` (`bench/mac/gpubw`), then run `bench/mac/densek` and
   `bench/mac/expertk`, three runs, plan first ([measurement.md](../contribute/measurement.md)): the anchor row for your chip.
2. Once the backend runs, load the large tier whole with `memory_pressure` open; record decode at 4K and 32K.
3. M5 owners: add a tensor-API prefill path behind the same backend interface ([backends/README.md](../backends/README.md)); one call per
   pass.
