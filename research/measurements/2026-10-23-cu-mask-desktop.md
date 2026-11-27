# Keeping CUs free for the desktop (HSA_CU_MASK), 2026-10-23

Question: does the mouse pointer stutter less if the megakernel does not occupy all 32 CUs? Reason: while
`just hang-repro` was running, the desktop (compositor, same GPU) felt sluggish, CPU and RAM under 50 %.

Setup: `ie-gpu-gen --mega --batch-bench`, IE_FAKE_EXPERTS=1 (verify pass only, no reloading), 64 GB RAM.
Mask via `HSA_CU_MASK=0:0-N` (ROCr, applies to all queues of the process), blocks matched to it.

## Short run (200 tokens, 2 runs each, spread < 0.2 %)

| Blocks | Mask | B = 1 ms/pass | B = 4 ms/pass |
|---:|---|---:|---:|
| 32 | – | 23.1 | 35.8 |
| 30 | 0-29 | abort at step 0 (barrier) | abort |
| 28 | 0-27 | 23.5 (+1.5 %) | 37.5 (+4.8 %) |

30 blocks on 30 CUs do not fit: presumably the mask has to release whole WGPs or shader arrays (not checked).

## Long run with mouse movement (5000 tokens, B = 4, 1 run each, ~45 s)

| | ms/pass | Tokens/s (decode) | Mouse |
|---|---:|---:|---|
| A: 32 blocks | 53.4 | 74.9 | smoother |
| B: 28 blocks, mask 0-27 | 60.0 | 66.7 | |

Result: the mask does not help the desktop and costs 5–11 %. Stays at 32 blocks.
Open: both long runs are 50 % above the short run (53 instead of 36 ms). Cause unresolved; candidates are clock/temperature
over 45 s and the desktop itself needing GPU time because of the mouse movement. Next attempt against the stutter:
queue with low priority (hipStreamCreateWithPriority), so that MES prefers the desktop.
