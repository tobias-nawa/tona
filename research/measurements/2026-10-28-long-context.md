# Prefill and decode over the prompt length (2026-10-28)

Question: how fast is the server at long context, up to 256K tokens? One run per point (a first look, not a series).

Setup: `ie-server` at 3a269e8 (`work/build/engine`), MTP drafter, `--ctx 262144`, expert profile
`work/profile/experts-coder.bin`, `OMP_NUM_THREADS=6`, no `IE_PROMPT_CACHE`, `MemoryMax` = RAM - 6 GB. Model Coder
IQ1_M, 64 GB RAM, desktop on the same GPU. At load: KV int8 3.62 GB in RAM, VRAM 14.47 GB, 80 expert slots per layer,
29.2 GB pinned.

Prompt: "Summarize what the following C++ and HIP code does, as a bullet list." followed by the engine sources at
3a269e8 (all `.cpp`, `.h`, `.hip` under `engine/` except `third_party`, sorted by path), cut to the target length with
`/v1/messages/count_tokens`; a random first line so no prompt cache applies. OpenAI API, thinking off, greedy, 200
tokens. Rates from the server's `timings`. The points ran in the order 32K, 64K, 128K, 250K, 8K, 16K against the same
server.

| Prompt (tokens) | Prefill (s) | Prefill tok/s | Decode tok/s after it |
|---:|---:|---:|---:|
| 8,105 | 5.6 | 1,459 | 66.6 |
| 16,277 | 11.2 | 1,457 | 64.4 |
| 32,357 | 21.6 | 1,499 | 58.7 |
| 65,254 | 47.2 | 1,382 | 57.9 |
| 130,788 | 161.8 | 808 | 57.1 |
| 249,732 | 458.4 | 545 | 44.6 |

## Result

- Prefill holds about 1,450–1,500 tok/s up to 32K and falls from 64K on: 808 tok/s at 128K, 545 tok/s at 250K
  (7:38 min before the first token). Where the time goes is not measured yet; the QSA indexer scores all earlier
  blocks for every prompt token, which grows with the square of the length (research/06-qwen38-architecture.md).
  Research on faster long-context prefill: research/29-long-context-prefill.md.
- Decode stays at 57–59 tok/s from 32K to 128K and drops to 44.6 tok/s at 250K. One run per point: decode after a
  long prompt scatters by about ±3 tok/s between runs (2026-10-28-mega-spills).
- The 8K and 16K prefill rates are below the 13K benchmark (1,750 tok/s, `scripts/bench-long-code.txt`); other text,
  other measurement, not compared further.

## Control: why prefill is below the first published numbers (2026-10-29)

The first published prefill rate (1,741 tok/s at 13K, 2026-10-27) ran with about 90 expert slots per layer. The
slots fill the free VRAM at start, so they depend on how much the desktop uses at that moment. Same build, same
server settings, one run each, `scripts/bench-openai.py URL x 1 long` (13K) plus the 8K and 16K prompts from above:

| `--ctx` | Expert slots per layer | Prefill 8K | Prefill 16K | Prefill 13K (`bench-long-code.txt`) |
|---:|---:|---:|---:|---:|
| 32,768 (KV in VRAM) | 68 | 1,670 | 1,688 | 1,627 |
| 262,144 (KV in RAM) | 71 | 1,493 | 1,618 | 1,624 |

- The curve above ran with 80 slots; this control with 68 to 71, about 0.9 GB less VRAM for the engine than the
  day before (desktop use at start; not measured which application). The 13K rate falls with the slots as in
  2026-10-25-slots (70 → 91 slots: 1,619 → 1,746 tok/s).
- The 256K setting itself costs little at 13K (1,624 vs 1,627); at 8K and 16K the single runs differ by 4 to 11 %,
  which one run per point cannot separate from noise.
- So the long-context curve is honest for a desktop with the card shared, and below the best case.

## Repeat on an idle desktop: 3 rounds (2026-10-29, night)

The first curve ran while the desktop was in active use (interactive use) on the same GPU; 128K and 250K could not be
repeated (808 → 1,047 tok/s at 128K, in every request order tried, also 64K before 128K as on the first day). Same
build, same server settings (`--ctx 262144`, 81 expert slots per layer), one server start per round, prompts
ascending. Raw: `/var/tmp/claude/ctx-curve/curve-idle.log` (not in the repository).

| Prompt (tokens) | Prefill tok/s, rounds 1 / 2 / 3 | Prefill (s) | Decode tok/s after it, rounds 1 / 2 / 3 |
|---:|---|---:|---|
| 8,105 | 1,631 / 1,630 / 1,629 | 5.0 | 67.9 / 67.0 / 68.1 |
| 16,277 | 1,616 / 1,615 / 1,615 | 10.1 | 66.2 / 66.1 / 67.9 |
| 32,357 | 1,538 / 1,537 / 1,538 | 21.0 | 62.8 / 64.4 / 66.7 |
| 65,254 | 1,394 / 1,394 / 1,394 | 46.8 | 55.4 / 56.7 / 54.6 |
| 130,788 | 1,048 / 1,047 / 1,047 | 124.9 | 55.5 / 53.8 / 56.0 |
| 249,732 | 597 / 597 / 595 | 418.5 | 45.7 / 44.8 / 44.2 |

Prefill spreads by 0.3 % or less, so a busy desktop explains the lower first curve at 128K and 250K (and 8K, 16K).

### Where the time goes at 128K (rocprofv3 kernel trace)

One 130,788-token prefill under `rocprofv3 --kernel-trace` (1,042 tok/s, profiler cost under 1 %), GPU busy 121.9 s
of 129.1 s:

| Kernel | Time (s) | Calls | ms per call |
|---|---:|---:|---:|
| `k_topk` (block selection) | 30.5 | 48,288 | 0.632 |
| `k_attn_wmma` (attention over the selection) | 24.0 | 264 | 91.0 |
| `k_gemm2<256, false>` | 8.3 | 1,870 | 4.43 |
| `k_gemm2<128, false>` | 7.1 | 11,325 | 0.63 |
| `k_dn_wave` (DeltaNet) | 6.4 | 792 | 8.07 |
| `k_idx_score_wmma` (indexer scoring) | 4.2 | 48,288 | 0.086 |

In the 16K + 64K trace together `k_topk` took 4.0 s and `k_attn_wmma` 11.1 s. The top-k grows with the square of the
length and is the largest single item at 128K; the scoring itself is small, as research/29 estimated. Attention per
call grows from about 26 ms (first chunk) to about 100 ms near 64K although every token attends to at most 2,048
cells, so the gather gets slower as the selected cells spread out.

KV cache in VRAM instead of pinned RAM (`IE_KV_HOST=0`, `--ctx 131072`, one run each): 128K prefill 1,222 tok/s with
48 expert slots against 1,119 to 1,136 tok/s in RAM with 84 to 88 slots: the host path costs about 10 %.

A first run with `--ctx 131072` and KV in RAM aborted with a GPU memory fault in `k_moe_sum` (128K prompt after a
64K one); not reproduced in two repeats.
