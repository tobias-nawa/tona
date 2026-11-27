# Predicting the experts of the next layer, 2026-10-24, own measurement

## Prediction hits (`IE_PREDICT_STATS=1`)

Router of layer l+1 (bf16, 256 × 2,560) applied to the int8 x of layer l (MoE input after the hyper-connections,
which the GPU sends for the CPU experts anyway), without norm correction. Teacher text Claude Code, 300 tokens.

| Candidates per token | 10 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|
| true top-10 hit | 65.4 % | 79.6 % | 87.8 % | 91.9 % |

Of 32,214 misses of the union (not in VRAM), **85.5 %** were in the top-24 of the prediction. This is in
the range of the literature (~90 % recall from layer 3 on Qwen3-30B, arXiv 2603.19289; research/14-expert-prefetch.md).

## What misses cost today (free generation, 500 tokens, LRU prompt, 2 runs)

| | Run 1 | Run 2 |
|---|---:|---:|
| tok/s (decode) with drafter | 59.9 | 59.7 |
| Extrapolation without reloading (pass times from 2026-10-23-verify-pass) | 66.1 | 66.1 |
| Misses per pass: GPU direct / CPU / refilled | 181 / 137 / 86 | equal |
| Expert hits | 74.8 % | 74.8 % |

So the upper bound for preloading is ~+10 %. With 85 % coverage and competition for PCIe (the GPU already reads misses
directly over PCIe), estimated ~+5 %. Deferred behind prefill DeltaNet; the measurement stays as a switch.

## Implemented: preloading during the pass (`IE_PREFETCH`, default 8)

The router of l+1 runs on the GPU in the same matvec as the router of l and lands in the mailbox (`pred`). The CPU
takes per token the top-16 of the prediction that are neither in VRAM nor currently arriving, sorted by number of tokens
and logit, and copies up to `IE_PREFETCH` of them by DMA on a separate stream into LRU or free slots of l+1.
Before the CPU picks the experts of l+1, it waits for these copies (event).

Decode, free generation, 500 tokens, LRU prompt (42 tokens), with drafter, warm, 1–2 runs each:

| `IE_PREFETCH` | Decode tok/s | VRAM hits | Preloaded ones used | Wait per pass |
|---|---:|---:|---:|---:|
| 0 | 60.5 | 74.9 % | – | – |
| 4 | 62.0 | 84.7 % | 54.7 % | 0.02 ms |
| **8** | **65.4–65.7** | 88.5 % | 49 % | 0.05 ms |
| 12 | 64.0 | 89.2 % | 45.8 % | 0.95 ms |
| 16 | 63.3 | 89.7 % | 44.9 % | 2.10 ms |

+8 % decode, more than the estimate (+5 %), because preloaded experts remain hits in later passes too.
From 12 on, competition for PCIe slows things down (waiting for the copies, more eviction).

Check `IE_PREFETCH_CHECK=1`: 5,617 preloaded slots compared byte by byte against the pinned RAM, all equal. Server smoke test
(`tools/server_smoke.py`, both APIs, with stream) ok; prefill after decode flushes open copies first (`prefetch_drain`).

Open: with `IE_PLAN=gpu` teacher NLL 0.959921 with against 0.960648 without preloading. The slots are correct, the difference
presumably comes from the summation order (hits in VRAM against directly from RAM), same class as the nondeterminism in 2026-10-24-planner.md.

## GPU reset caused by the check path (2026-10-24)

The first version of the check read with `hipMemcpy` while `k_mega` was waiting for the CPU: mutual waiting, after
~20 s the desktop's gfx ring hit its time limit, ring reset failed, MODE1 reset, session gone. Afterwards:

- Copy on the non-blocking stream `pre_st_`.
- amdgpu aborts jobs after **2 s** (`lockup_timeout`, default 2000 ms according to `modinfo -p amdgpu`). As long as
  `k_mega` waits, the desktop's jobs do not run. A CPU stall of 6 s (`IE_TEST_STALL_MS=6000`) with the
  limit of 4 s at that time made a terminal window crash (ring reset succeeded).
- Now the kernel waits at most `HOST_WAIT_MS` = 1 s, with barrier 1.2 s. Test with a 1.5 s stall: the pass aborts cleanly after
  1 s ("no message from router 11"), nothing in the driver log, desktop untouched. With a 2 s stall (old limit)
  it ran without error, but right at the limit.
- Follow-up: stability work.
