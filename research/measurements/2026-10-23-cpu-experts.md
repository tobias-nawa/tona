# CPU experts (AVX-512) in the megakernel, 32 GB RAM

Own measurement, 2026-10-23.

- Machine: RX 9060 XT, Ryzen 5 9600X, 32 GB RAM
- Model: Coder IQ1_M, drafter DFlash2 Q8_0
- Prompt: CSV function in Python, 120 tokens
- Run: `systemd-run -p MemoryMax=18G`, `OMP_NUM_THREADS=6`

## Core alone (`test_cpu_experts`, weights in cache)

All 6 formats equal to the reference (dequantize_row, double): worst relative deviation ≤ 1.3e-7.

One matrix, 1 thread, 1 token (µs):

| Format | first version | 4 rows at once | with gather |
|---|---:|---:|---:|
| IQ4_NL (down) | 92 | 49 | 48 |
| IQ3_XXS (gate/up) | 180 | 117 | 118 (gather slower: 153) |
| Q2_0 (down) | 101 | 40 | 40 |
| IQ2_S (gate/up) | 323 | 323 | **102** |
| IQ3_S (gate/up) | 268 | 205 | **135** |
| IQ4_XS (gate/up) | 120 | 58 | 57 |

Whole expert (gate, up, down), 4 tokens, 1–2 tokens per expert:

| | 6 threads | 12 threads |
|---|---:|---:|
| 6 experts of a layer | 76–129 µs per expert | 49–77 µs per expert |

## In the megakernel

The CPU computes the misses, the GPU waits only before `moe.sum`. Per layer, up to 2 misses are additionally brought into the VRAM cache: read after computing, inserted in the next pass.

| | tok/s (decode) | from disk per run |
|---|---:|---:|
| old way (read misses and upload them), same page cache state | 26.7 | 3.4 GB |
| CPU mode, dense weights still in the page cache | 25.0–25.7 | 7.8–9.9 GB |
| CPU mode, dense weights dropped after upload, runs 1–6 | 27.5 / 30.4 / 32.9 / 33.3 / 31.7 / 28.0 | 4.6–9.1 GB |

- Hit rate in VRAM: 81.0 % in both modes, although the CPU mode reloads only 89 instead of 280 experts per pass.
- Per pass: 224 experts on the CPU.
- The text is valid code, but not bit-identical to the old way: different summation order and int8 rounding of the activation on the CPU.
- `--check 6` against the CPU reference: top-1 equal in 6 of 6, worst deviation 8.6e-2 (before 7.8e-2).

**With 32 GB the disk is the bottleneck.** The 25 GB of experts do not quite fit in the page cache. Computed from RAM, the CPU manages 224 experts in 28–30 ms per pass (measurement mode `IE_CPU_EXPERT_MOD=48`, see below). From the NVMe it is 70–100 ms.

## Measurement mode is only good for the CPU time

`IE_CPU_EXPERT_MOD=48` computes with the weights of expert e mod 48 so that everything fits in the cache. The text turns to nonsense, the engine then picks other experts (97 % hits, 2.3 tokens per pass). It is not usable for tok/s, only for the time per expert: ~125 µs with 6 threads.

## Conclusion

- Extrapolated to 64 GB (all experts in RAM): pass ≈ 41 ms GPU + ~20 ms waiting for the CPU. That would be ~55–60 tok/s (decode).
- Next step: a pinned tier where the GPU reads missing experts directly from RAM, and a planner that distributes the misses between GPU and CPU.

## Addendum: profile, pinned tier, planner

Same day, same machine (32 GB). Runs with `MemoryMax=20G`.

### Profile

- 6 coding prompts of 250 tokens each, without the measurement prompt: `work/profile/experts-coder.bin` (48 × 256 counts).
- The most frequent experts per layer cover this share of the calls:

  | Top per layer | Share |
  |---:|---:|
  | 64 | 64.6 % |
  | 96 | 77.6 % |
  | 128 | 86.5 % |

- Pre-filling the VRAM cache from the profile: 81.0 → 82.3 % hits, the CPU computes 209 instead of 224 experts per pass. Startup takes 4 s longer. The tok/s are too noisy because of the NVMe to say anything.

### GPU reads pinned experts directly

- The buffer is laid out so that place j gets slot number `slot0 + j`. The matvec kernels stay unchanged.
- `--check 6`: top-1 in 6 of 6, deviation 8.2e-2.
- Marginal cost in the megakernel: ~13 µs per expert. This comes from `IE_MEGA_PROF`: moe.gate_up and moe.down together +0.55 ms at ~42 experts per pass.
  - Reason: reading over PCIe runs alongside reading from VRAM. Only from ~480 experts per pass would PCIe be full.
  - For comparison: CPU ~125 µs per expert, 6 threads, from RAM.

`--batch-bench 400`, B = 4, 12 GB pinned:

| Planner | ms per pass | Misses per pass (GPU direct / CPU) | GPU waits for CPU |
|---|---:|---:|---:|
| automatic | 51.8 | 16.9 / 9.2 | 11.9 ms |
| everything to the GPU | 49.8 | 18.4 / 11.0 | 9.4 ms |

- The misses given to the CPU are here the unpinned experts. They come from the NVMe via page faults, over 1 ms per expert.
- With 64 GB everything is pinned, then this waiting disappears.
- Extrapolated: pass B = 4 ≈ 49.8 − 9.4 ≈ 40 ms.

### Race when refilling (fixed)

- Sequence: if the CPU had nothing to compute in a layer, it already wrote the next experts into the refill buffer. At the same time the GPU was still copying the previous ones from it.
- Consequence: deviation up to 0.21 instead of 0.08, top-1 stayed equal.
- Fix: two halves per layer, alternating. Afterwards deviation 7.8e-2 to 8.2e-2, identical from run to run.
