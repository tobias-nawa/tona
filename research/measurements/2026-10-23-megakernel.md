# 2026-10-23: Megakernel attempt, matvec chain of one token

Own measurement. Tool `ie-gpu-chain` (`engine/tools/ie_gpu_chain.hip`, `gpu::MatvecChain` in
`engine/gpu/matvec.hip`). RAM 32 GB, DPM `auto`, desktop runs on the same GPU.

## Setup

All matvecs of one token with real weights: 48 layers, 10 experts per layer (0–9), plus head.
2,211 matvecs, 4.989 GB of weights; lower bound at 315 GB/s: **15.84 ms**. The small steps (norms,
DeltaNet recurrence, attention, ~1,750 kernels) are missing. The phases follow the dependencies in the real
forward pass (e.g. experts gate+up of all 10 in one phase, down plus shared expert in the next):
435 phases.

- **A**: one kernel per matvec (like `gpu::Forward` today)
- **B**: one kernel per phase, all rows of the phase distributed evenly over the waves
- **C**: one persistent kernel for the whole token, grid barrier between the phases

All variants use the same row functions and are bit-identical (checked). 9 runs of 10 tokens, median.

## Result (3 program runs, register cap 8 waves/SIMD)

| Variant | Kernels per token | ms per token | GB/s | Aborts |
|---|---:|---:|---:|---:|
| A | 2,211 | 26.8–27.3 | 183–186 | – |
| B | 435 | 38.3 | 130 | – |
| C, 64 blocks (occupancy limit according to HIP) | 1 | 21.3 when it ran through | – | **7–8 of ~130** |
| **C, 48 blocks** | 1 | **22.2–22.3** | **224** | 0 |
| C, 32 blocks | 1 | 25.3 | 197 | 0 |

**C with 48 blocks saves 4.6–5.1 ms per token (−17 %) compared with A**, only through fewer launches and no
gaps between the kernels.

## What stood out

1. **`grid.sync` (cooperative launch) hung**: once 54 s for one token, once over 10 minutes at 100 % GPU load,
   until aborted with `kill` (no GPU reset). Replaced by an own barrier with a 100 ms time limit.
   If the limit expires, all blocks abort and `diag()` reports it.
2. **HIP overestimates the occupancy**: with the 64 blocks that `hipOccupancyMaxActiveBlocksPerMultiprocessor`
   allows, in ~6 % of the launches not all blocks run at the same time, and the barrier waits in vain.
   Presumably the desktop (two 4K monitors) occupies part of the compute units. With 48 blocks (75 %) there was
   no abort in ~390 launches, launch skew at most 36 µs.
3. **Registers**: with all 11 formats in one kernel the compiler used 218 VGPRs (6 waves/SIMD).
   Choosing the format per task instead of per row brought 207, a cap via `amdgpu_waves_per_eu`:

   | Cap | VGPRs | Spill | C, best block count |
   |---:|---:|---:|---:|
   | 8 | 158 | 0 | 20.7 ms (grid.sync, 64 blocks, one run) |
   | 12 | 119 | 0 | 26.1 ms |
   | 16 | 96 | 6 | not measured (run aborted) |

   Fewer registers make C slower: with more registers each wave keeps more reads in flight at once.
   Here that is worth more than additional waves. Hence cap 8 (CMake `IE_CHAIN_WAVES`).
4. **B is slower than A**: there each wave computes exactly one row and ends, with the high register count
   of the shared kernel. C distributes many rows over few long-lived waves.

## Extrapolation to the whole token

Measurement mode without reloading (2026-10-22-gpu-forward-v1): 36 ms wall, of which ~30 ms GPU. Matvecs as
a megakernel save ~5 ms. The ~1,750 small kernels and 48 synchronizations (together ~6 ms gaps) would
become phases of the same kernel (~1 µs per barrier). Expectation **~25 ms per token without reloading, ~40 tok/s (decode)**.
That has to be measured before it counts.

The lever is larger in the matvec kernels themselves: C reaches 224 of 315 GB/s, the IQ formats
individually sit at 70–110 GB/s.

## Risks

- A persistent kernel needs the certainty that all blocks run at the same time. On this card with
  the desktop that holds only with distance to the occupancy limit. Time limit plus fallback to the single kernels for the
  affected token is mandatory.
- Expert reloading and top-k run on the CPU today. In the megakernel the GPU would have to write the selection to
  host memory and wait for the copy (CPU↔GPU via host memory 0.73 µs, measured
  2026-10-22-micro-persist).
