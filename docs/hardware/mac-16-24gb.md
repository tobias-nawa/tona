# Mac with 16 or 24 GB

**Status: budget emulated** on the maintainers' M1 Max 32 GB (hard cap on weights and arena, plus ballast memory), **concept
only for bandwidth and SSD**. Method: [README.md](README.md#how-the-files-estimate); chips: [mac-chips.md](mac-chips.md).
Backend: `metal`, in development.

Typical machines (from memory [K]): 16 GB: M1, M2, M3, M4, M5 base, M1 Pro, M2 Pro; 24 GB: M2, M3, M4 base, M4 Pro, M5 Pro.

## Budget

The working set was measured on the 32 GB machine only (26.8 GB, 78 %, [M]); the table uses reported two thirds to three
quarters of RAM ([S], sources disagree).

| | 16 GB (17.2 GB) | 24 GB (25.8 GB) |
|---|---:|---:|
| GPU working set, default (estimate) | 11 to 13 GB | 17 to 19 GB |
| After buffers (about 1.5 GB) | 9.5 to 11.5 GB | 15.5 to 17.5 GB |
| Left for macOS and apps | 4 to 6 GB | 6.5 to 8.5 GB |
| With `iogpu.wired_limit_mb` raised (unsupported, reset at reboot) | about 12 to 13 GB, leaves macOS 4 GB | about 20 to 21 GB |

Do not size the cache to the limit: on an M5 Pro 24 GB decode fell from 13.9 tok/s at a 35 % expert cache to 2.3 at 50 %
when macOS began compressing and paging (MawForge, arXiv 2607.09686, research/38 [V]). llama.cpp issue 19825 (M4 Pro 24 GB,
Qwen3-Next 80B): a full RAM degrades into swap up to a kernel panic, even with `--n-cpu-moe` and a raised wired limit [V].

## Model tier

| Tier | 24 GB | 16 GB |
|---|---|---|
| Large (ISTA Coder IQ1_M, 58.4 GB) | **not recommended**: arena 7 to 9 GB = 28 to 36 % of the experts | no |
| Medium: Gyro-Coder 21.4 GB (23.5 GB at 128K) | no; with a raised limit at short context only | no |
| Medium: Qwen3.8-27B IQ3_XXS 10.1 GB, KV 1.1 GB at 32K, 4.3 GB at 128K | yes (14.5 GB at 128K, tight) | 32K context only (11.2 GB), at the limit |
| Small: Ornith-1.5-35B-A3B IQ3_XXS 15.3 GB plus KV | yes, 16.7 GB at 128K | no (Q2-class 12.5 GB plus KV exceeds the budget) |
| Small: Groot-19B-A3B Q4_K_M 11.7 GB, IQ4_XS 10.9 GB, plus KV 1.3 GB at 128K | yes | IQ4_XS at 32K context (11.3 GB), at the limit |

Sizes: research/21-backends.md "Model tiers" [V]; fit derived against the budget. The 16 GB class is the small tier at
reduced context; no quality figures from us.

## Expert strategy

- **Small, 24 GB:** all resident; Ornith IQ3_XXS and Groot read about 1.8 GB per token.
- **Small, 16 GB:** resident at short context; otherwise stream only cold experts from the SSD with the arena design of
  [mac-32gb.md](mac-32gb.md). TurboFieldfare (16-slot LFU per layer, Gemma 4 26B-A4B 4-bit): 5.1 to 6.3 tok/s on an M2 Air
  8 GB, 31 to 35 on an M5 Pro 24 GB (research/39 [V]).
- **Large, 24 GB (concept only):** 4.1 GB dense plus a 7 to 9 GB arena, hit rate about 80 to 85 %, so 72 to 96 expert
  reads of 2 MB = 150 to 200 MB per token. At the 5 GB/s of M4 Pro and M1 Pro SSDs [S] that is 30 to 40 ms on top of the
  resident pass: about **12 to 15 passes/s** at best (our M1 Max pass time; optimistic for chips with fewer GPU cores).
  External: ik_llama.cpp `--cpu-moe` about 10 tok/s on an M4 Pro 24 GB, Qwen3-Next 80B (research/21 [V]); DeepInfer-Mac,
  M4 Air 24 GB, a 748B model: 0.21 to 0.72 tok/s (arXiv 2610.36564, research/38).

## Estimates (decode, passes per second, memory bound)

Bound = 0.82 x peak bandwidth / bytes per pass (82 % measured on the M1 Max, unchecked elsewhere); bytes per pass
(research/21): Groot or Ornith 1.8 GB plus KV read (0.34 GB at 32K), 27B dense 9.6 GB plus KV. Real rates are lower by the
kernel efficiency (29 to 57 % on the M1 Max, untuned); drafts can give more than one token per pass.

| Chip | Bandwidth | Small, 32K context | 27B IQ3_XXS, 32K |
|---|---:|---:|---:|
| M1 | 68 GB/s | 26 | 5 |
| M2, M3 | 102 GB/s | 39 | 8 |
| M4 | 120 GB/s | 46 | 9 |
| M5 | 153 GB/s | 59 | 12 |
| M1 Pro, M2 Pro | 205 GB/s | 79 | 16 |
| M4 Pro | 273 GB/s | 105 | 21 |
| M5 Pro | 307 GB/s | 118 | 23 |

The 27B column is no recommendation on the base chips (5 to 9 passes/s is slow for an agent). Prefill: ALU-bound on M1 to
M4 (tens to a few hundred tok/s for a 3 B-active model on base chips); M5 has neural accelerators and Apple claims over 4x
the peak GPU compute of M4 [S]. No measurement of ours, so no prefill number.

## What we can reproduce, what only real hardware shows

| Reproducible on the M1 Max 32 GB | Only real hardware shows |
|---|---|
| the budget: cap weights plus arena at 11 to 13 GB or 17 to 19 GB; 20 GB ballast leaves macOS 6 to 8 GB | the real working-set limit of a 16 or 24 GB machine |
| model fit, context limits, behavior under memory pressure | bandwidth of the base chips (68 to 153 GB/s), their SSD speed |
| kernels and dispatch cost on the M1 generation | fanless thermals (Air), M5 matrix units |


## First steps for a contributor

1. Run `sysctl hw.memsize iogpu.wired_limit_mb` and print `recommendedMaxWorkingSetSize` (`bench/mac/gpubw` without a
   measurement prints it); post the line in an issue.
2. Run the three Phase 0 benchmarks (bandwidth, SSD, dispatch), plan first, and add the row to [mac-chips.md](mac-chips.md).
3. With the backend running, measure the small tier at 16K, 32K and 64K context under `memory_pressure`.
