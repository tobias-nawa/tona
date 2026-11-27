# 2026-10-22: Expert routing on code, CPU reference

Own measurement. Tool: `ie-ref --teacher` (CPU reference), analysis with `tools/expert_stats.py`.
Repo state f1ee189 plus teacher mode in `engine/tools/ie_ref.cpp`.

## Setup

- Model Coder IQ1_M (256 experts per layer, top-10), 48 layers
- Text: 6,000 bytes of Python from `bench/corpus/python-stdlib-3.14.txt` (asyncio), first **512 tokens**,
  teacher forcing (every token is computed, the next one comes from the text)
- 5 threads, 2.50 s per token, memory limited to 8 GB

## Results

**Top-1 hits** of the model on the text: **82.4 %**. Plausibility check for the reference; the
stdlib is probably in the training data.

Union of experts over b consecutive tokens, mean over all windows and layers:

| b | Experts per layer | × one token | % of disjoint |
|---:|---:|---:|---:|
| 1 | 10.0 | 1.00 | 100 |
| 2 | 15.7 | 1.57 | 78.6 |
| 3 | 20.6 | 2.06 | 68.7 |
| 4 | 25.0 | 2.50 | 62.5 |
| 6 | 32.8 | 3.28 | 54.7 |
| 8 | 39.5 | 3.95 | 49.3 |
| 12 | 50.5 | 5.05 | 42.1 |
| 16 | 59.4 | 5.94 | 37.1 |

Expert cache with 97 of 256 experts per layer (38 %, the VRAM left at 64K context according to `ie-model`):

| | Hits |
|---|---:|
| ideal static (the 97 most frequent, chosen in hindsight) | 88.3 % |
| LRU, cold start | 86.5 % |

On average each layer uses 207 of the 256 experts in 512 tokens.

## Interpretation

- Neighboring tokens share experts, but few: a verify pass of 4 tokens loads 2.5× the
  experts of one token, not 1× and not 4×. Speculation therefore costs noticeable
  bandwidth with this model; long drafts only pay off with a high acceptance rate (see 2026-10-22-suffix-sim.md).
- The usage is skewed enough for a cache: 38 % of the experts cover ~87 % of the calls. LRU is
  close to the static ideal.
- With 86.5 % hits and misses over PCIe (55 GB/s), the bandwidth model gives **~55 passes/s**
  instead of 62 with everything in VRAM.

## Limits

- One text, 512 tokens, Python only. German, tool calls and long sessions are missing. LRU hits on
  longer runs are probably higher (warm cache), lower on topic changes.
- Teacher forcing on foreign text, not on the model's own output.
- Repeat with llama.cpp routing over long agent traces once 64 GB are available.
