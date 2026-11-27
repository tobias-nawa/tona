# 2026-10-23: DFlash2 drafter as CPU reference, acceptance rate

Own measurement. Drafter `dflash-Qwen3.8-Flash-Next-Q8_0.gguf` (ggml-org/Qwen3.8-Flash-Next-GGUF, 603M), CPU reference
`engine/cpu/dflash.cpp` after llama.cpp d9ee007 (`src/models/dflash.cpp`, `common/speculative.cpp`). Target: megakernel
int8, verify pass with 4 tokens (last token + 3 drafts), greedy. RAM 32 GB, real experts (64 slots).

## Drafter

- Metadata: 5 layers, d = 2560, 32 heads / 8 KV of 128, FFN 7680, window 2048 in all layers,
  M-RoPE with sections [64, 0, 0, 0] (= NeoX over 128 dimensions, base 10000), block 8, conv kernel 2 in
  groups of 16, selector rank 256 and top-16, target layers [6, 16, 26, 36, 46], non-causal.
- Context: input of the target layers (before PLE, mean over the 4 streams, llama.cpp #30927) → fc (12800 → 2560)
  → RMSNorm → K/V of each layer in the drafter's cache. New: op `feat` in the megakernel writes this per token.
- Draft: [last token, mask 248077 × 3] at pos … pos + 3, embedding and head from the target model, dynamic
  folding before and after attention and FFN, then selector (top-16 per position, transition scores) greedy.

## Result

Output with speculation **byte-identical** to generation without (CSV prompt, 120 tokens).

| Prompt (160 tokens) | Tokens per pass | Drafts accepted 0 / 1 / 2 / 3 |
|---|---:|---|
| Python: CSV means | 3.72 | 0 / 3 / 3 / 26 |
| Rust: merge two sorted vectors, with tests | 3.79 | 1 / 1 / 4 / 36 |
| English: explain hash map collisions | 2.41 | 17 / 20 / 14 / 15 |
| German: short story | 1.76 | 49 / 23 / 11 / 8 |

Code is close to the ceiling of 4 tokens per pass (block 4); longer blocks pay off there (trained is block 8).
For prose a pass with 4 tokens (41.7 ms) is more expensive than the gain: 1.76 tokens in 41.7 ms = 42 tok/s against 44
without speculation. The draft length has to adapt.

Time: drafter on the CPU ~350 ms per pass (oracle only); target with reloading at 32 GB 210–275 ms per pass
(~750 MB of experts per pass from the NVMe). Measured without reloading: 41.7 ms per pass with 4 tokens
(2026-10-23-verify-pass); with the drafter on the GPU (estimated 3–5 ms) that would be ~80 tok/s (decode) at 3.7 tokens per pass.

## Addendum: drafter on the GPU

Second program in the megakernel (same kernel `k_mega<4>`, own step list): feeding the m accepted
tokens (features stay on the GPU) and block in one launch; the target model's head is shared. Selector on
the CPU (logits of the 3 mask positions back).

- Drafts GPU equal to CPU reference: 51 of 51; output with speculation still byte-identical (CSV prompt, 120 tokens).
- Time per pass: kernel 5.7–5.9 ms (feeding + block + head over 4 rows), selector 0.45 ms (2 runs).
  About 1.0 GB is read (drafter 0.5 GB, head 0.52 GB) → ceiling ~3.3 ms.
- Without reloading (target 41.8 ms + drafter 6.3 ms per pass): at 3.72 tokens per pass **~77 tok/s (decode)** (without speculation 44).
- With reloading at 32 GB: 212 ms per pass for the target, almost all of it experts from the NVMe (750 MB per pass);
  17.0 tok/s (decode) against 15.8 without speculation. Here RAM is now the bottleneck.

## Addendum: calibrating the selector, draft length per pass

Softmax of the selector scores at the chosen candidate against "accepted" (only as long as all drafts before it matched),
5 prompts × 200 tokens:

| p | Position 1 | Position 2 | Position 3 |
|---|---:|---:|---:|
| < 0.3 | 0.07 (14) | 0.00 (13) | 0.00 (3) |
| 0.3–0.5 | 0.38 (34) | 0.12 (16) | 0.29 (17) |
| 0.5–0.7 | 0.51 (47) | 0.54 (24) | 0.59 (22) |
| 0.7–0.85 | 0.65 (31) | 0.68 (31) | 0.56 (16) |
| 0.85–0.95 | 0.85 (27) | 0.80 (25) | 0.77 (13) |
| ≥ 0.95 | 0.98 (154) | 0.96 (123) | 0.95 (103) |

p is a usable estimator (slightly too high). Rule per pass: k drafts with the largest
(1 + Σ_j Π_{i≤j} p_i) / (pass time(k + 1) + drafter); pass times measured on a sliding basis, start with the values without
reloading. Projection with fixed pass times (22.9 / 31.3 / 37.0 / 41.7 ms + 6.3 ms drafter), real acceptance:

| Prompt (160 tokens) | always 3 drafts | adapted | Widths 1 / 2 / 3 / 4 |
|---|---:|---:|---|
| Python CSV | 78.9 tok/s | 79.3 | 0 / 1 / 0 / 41 |
| Explain hash map | 50.5 | 53.1 | 0 / 11 / 9 / 46 |
| Short story (German) | 36.6 | 40.0 | 8 / 15 / 34 / 36 |

The story stays below 44 tok/s without speculation: the drafter (6.3 ms) runs even if the pass ends up checking only
1 token.
