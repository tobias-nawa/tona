# 2026-10-22: GPU forward pass version 1

Own measurement. Tools: `ie-gpu-gen` (`engine/gpu/forward.hip`, `ops.hip`, `matvec.hip`), profile with
`rocprofv3 --kernel-trace --memory-copy-trace`, analysis with `tools/prof_tokens.py`. RAM 32 GB, DPM `auto`.

## Setup

- One kernel per step, activations F32, decode token by token (the prompt too).
- Dense weights fixed in VRAM (4.1 GB), experts in a per-layer LRU cache. Missing experts are loaded
  synchronously with `hipMemcpy` from the mapped file (pageable).
- Top-k of the experts and, over 2,051 tokens, the QSA selection on the CPU: 48 synchronizations per token.
- Prompt: chat format, "parse CSV, column means, code only", 33 tokens, then 100 or 200 tokens greedy.

## Correctness

Against the CPU reference (`--check 8`, "The capital of France is"): the same top-1 in all 8 steps,
largest relative deviation of the logits **9.9e-7**. Output " Paris. The capital of Germany".

## Speed

| Expert slots per layer | Share | Hits | Reloaded per token | tok/s (decode) |
|---:|---:|---:|---:|---:|
| 48 | 19 % | 75.7 % | 238 MB | 5.6 |
| 96 | 38 % | 89.8 % | 100 MB | **18.8** |
| 96, experts not loaded (measurement mode, wrong result) | – | – | 0 | 27.7 (3 runs: 27.4–27.7) |

VRAM at 96 slots: 4.1 GB dense + 9.4 GB experts + 0.4 GB KV (8K context) ≈ 14 GB.

Profile, 80 tokens (the profiler slows things down: 13.8 instead of 18.8 tok/s):

| | real | without reloading |
|---|---:|---:|
| Wall per token | 72.4 ms | 45.8 ms |
| GPU busy | 37.3 ms | 29.9 ms |
| Copies to the GPU | 117 MB, 3.0 ms DMA | 0 |
| Kernels per token | 3,964 | 3,964 |

Largest kernels (real): Q6_K 7.4 ms, BF16 6.5 ms (472 calls), IQ4_NL 3.6, IQ2_S 3.0, IQ3_XXS 2.8,
Q5_K 2.5, DeltaNet recurrence 2.2 ms. Small kernels cost ~1.7 µs per call (e.g. `silu_mul` 528× = 0.9 ms).

## Interpretation

Without the profiler, 18.8 tok/s (decode) = 53 ms per token, roughly split as:

- ~30 ms GPU compute time, of which ~25 ms matvec
- ~6 ms gaps from kernel launches and the 48 synchronizations (measurement mode: 36 ms wall against 30 ms GPU)
- ~17 ms reloading missing experts: the CPU copies from the page cache, synchronously; the DMA itself is
  small at 3 ms

Next steps in this order:
1. Asynchronous reloading: compute the hits first, missing experts in parallel through a pinned buffer.
2. Fewer kernels: gate+up together, merge small steps; then the megakernel attempt.
3. Matvec of the IQ formats.

With 64 GB the experts can be pinned in RAM. Then the DMA copies directly and the CPU no longer does.

## Addendum: asynchronous reloading and `pread`

Same setup, 96 slots, 200 tokens, 3 runs each:

| Variant | tok/s (decode) | Page faults (major / minor) |
|---|---:|---:|
| synchronous, `hipMemcpy` from the mapping | 18.8 | – |
| hits first, CPU copies with `memcpy` from the mapping into a pinned buffer, separate copy stream | 15.1 / 17.6 / 16.8 | 5,725–12,536 / 1.33 M |
| as above, but `pread` instead of `memcpy` | 22.4 / **24.4** / **24.4** | 0–98 / 1.05 M* |

\* The remaining small page faults come from uploading the dense weights at startup (4.1 GB), not
from generation.

Interpretation: the bottleneck was not the copy but reading through the mapping. With 32 GB RAM the
25 GB of experts do not quite fit in the page cache, hence the NVMe accesses. `pread` reads past the page cache
into the buffer, without page tables and page faults. Gap to the measurement mode without reloading (27.7 tok/s):
about 5 ms per token.
