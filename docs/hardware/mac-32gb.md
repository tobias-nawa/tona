# Mac with 32 GB

**Status: emulated.** The maintainers own a MacBook Pro with an M1 Max, 32 GB, 32 GPU cores; every measured number comes
from it. Method: [README.md](README.md#how-the-files-estimate); chip data: [mac-chips.md](mac-chips.md). The backend
(`metal`) is in development; [M] rates are of its test kernels, not of a running server.

Other 32 GB chips (M1 Pro, M2 Pro, M2 Max, M4 and M5 base) have the same budget; rates scale as in "Other chips with 32 GB".

## Budget

| | Value | Source |
|---|---|---|
| RAM | 34.4 GB (32 GiB) | [M] |
| GPU working set (`recommendedMaxWorkingSetSize`) | **26.8 GB** (78 %) | [M] mac-bandwidth.md |
| Largest single buffer | 20.1 GB | [M] same |
| Left for macOS and apps | 7.6 GB | derived |
| Usable for weights plus KV after buffers | about 25 GB | estimate: 26.8 GB minus about 1.5 GB of buffers |
| GPU read | 334 GB/s (CPU: 115 GB/s) | [M] |
| SSD, 2 MB blocks at depth 16 | 13 GB/s | [M] mac-ssd.md |

A cache sized to the working-set limit can collapse decode once macOS compresses memory (MawForge, M5 Pro 24 GB: 13.9
tok/s at a 35 % expert cache, 2.3 at 50 %; arXiv 2607.09686 [V via research/38]). Size the arena under measured memory
pressure, not to the limit.

## Model tier

| Tier | Candidate | Fits | Notes |
|---|---|---|---|
| Large | ISTA Coder IQ1_M, 58.4 GB (29.6 GB trunk plus 28.8 GB n-gram table) | no: experts **stream from the SSD** | trunk does not fit beside KV and arena |
| Medium | Gyro-Coder TQ2_0 noPLE, 21.4 GB (23.5 GB resident at 128K) | yes, about 3 GB headroom | needs the trellis dequant kernel; no quality figures by us |
| Medium | Qwen3.8-27B IQ3_XXS, 10.1 GB (14.5 GB at 128K) | yes | dense, 9.6 GB read per token |
| Small | Ornith-1.5-35B-A3B Q4_K_M (21.7 GB, 23.2 GB at 128K) or Groot-19B Q4_K_M (11.7 GB) | yes | needs the qwen35moe path in the Metal backend (today only the Coder's architecture) |

Sizes: research/21-backends.md, "Model tiers" (quality numbers are vendor claims).

## Expert strategy

- **Large:** dense weights resident (4.11 GB: Metal maps the BF16 hyper-connection weights as they are, the PC backend
  reads them as Q8_0, 3.52 GB), experts in an arena of 14 to 18 GB (56 to 72 % of the experts), the rest read from the SSD
  with direct reads of one packed expert (2.046 MB) at depth 16, 13 GB/s; the GPU waits per layer on an event. The n-gram
  table stays on the SSD (16 rows of 90 bytes per token, about 0.2 ms). 0.68 MB blocks (the GGUF layout) reach only 7 GB/s
  and buffered reads lose 40 % ([M] mac-ssd.md).
- **Medium and small:** everything resident, no SSD in the decode loop.

## Estimates (decode, one token per pass, no MTP multiplier)

| Case | Rate | Basis |
|---|---:|---|
| Large, upper bound, all resident | 66 passes/s | 5.1 GB at 334 GB/s (bound) |
| Large, resident pass with current kernels | **about 28 passes/s** | measured parts: dense 21.5 ms, experts 10.2 ms, DeltaNet 0.9 ms, small ops about 3 ms ([M] 2026-11-11-mac-densek.md, -expertk.md) |
| Large, streamed, on demand | **20 to 28 passes/s** (planning range, before the dense kernels improved) | research/41-mac-reconciliation.md |
| Large, streamed, with prediction | 23 to 34 passes/s | same |
| External yardstick, large model | **8.3 tok/s** | ds4, M1 Max 32 GiB, Qwen3.8 Q2, 512 experts, prefill about 10 tok/s (research/39) |
| Medium, bound | Gyro-Coder 71 to 82, 27B 35 (24 at 128K) passes/s | bandwidth bound, not kernel-limited |
| Small, bound | Ornith Q4_K_M 187 short, 107 at 128K; Groot 189 / 107 | 334 GB/s over 1.8 GB (3.1 GB with the KV read in full) |

The bounds ignore kernel efficiency; the expert kernels reach 29 % of the bound untuned, the dense part 57 % after tuning.
The factor 2 to 3 between the planning range and ds4's 8.3 tok/s is unexplained (research/41, "The gap to ds4").

Prefill (estimate): the M1 Max has no matrix units. About 10 TFLOPS fp16 [K] at 30 to 50 % use gives 12 GFLOP per token
(6 B active), roughly 250 to 450 tok/s in 512-token chunks, with 7 to 11 GB of non-resident experts per chunk read at 13
GB/s (research/40, "Expected rates"). The only external figure is ds4's 10 tok/s; treat the range as unconfirmed.

Third-party, neighboring hardware: Qwen3-Next-80B-A3B 4-bit in MLX, M1 Max 64 GB: about 43 tok/s decode at 4K (oMLX [S]);
Flash-MoE, Qwen3.5-397B-A17B, M3 Max 48 GB: 4.36 tok/s with 4-bit experts through the page cache (research/21 [S]);
llama.cpp PR 23324, Qwen3-30B-A3B Q6_K, M3 Pro 36 GB: 12 to 30 tok/s (research/39 [V]).

## Other chips with 32 GB

The bandwidth-bound part scales with the GPU read rate, the arena size stays, the SSD part scales with the SSD. M1 Pro and
M2 Pro (204.8 GB/s peak): resident large bound about 33 passes/s, Gyro-Coder 36. M4 base (120) and M5 base (153 GB/s):
large bound 19 and 25 passes/s; there the large tier is slower than the SSD path bound and the medium tier is the default.
M5 base prefill has matrix units and should be several times faster; unchecked.

## What we can reproduce, what only real hardware shows

| Reproducible on the M1 Max | Only real hardware shows |
|---|---|
| the 26.8 GB working set, SSD 13 GB/s, the arena and miss path, event waits, timeouts | other chips' bandwidth, SSD, GPU clock and dispatch cost |
| a hard budget (a smaller arena) to emulate 24 and 16 GB: see the next file | M5 matrix units; i8mm, SME on the CPU |
| power state effects (AC, battery, low power) | long-run thermal behavior in laptops other than ours |


## First steps for a contributor

1. Read `research/40-mac-backend-proposal.md` and `research/41-mac-reconciliation.md`; run `bench/mac/gpubw` and
   `bench/mac/ssdbench` and compare to the table above.
2. M2 to M5 owners: repeat the three Phase 0 benchmarks (bandwidth, SSD, dispatch) with the plan first, file
   `research/measurements/<date>-mac-<chip>-*.md`, and add a row to [mac-chips.md](mac-chips.md).
3. M5 owners: the neural-accelerator path (Metal 4 tensor API) for prefill is open.
