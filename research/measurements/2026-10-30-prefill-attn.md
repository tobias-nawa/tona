# Prefill attention `k_attn_wmma` at long context: what limits it, gather pipeline (#78, 2026-10-30)

Question (research/34): `k_attn_wmma` is 26 % of the GPU time of a 128K prefill (24.0 s of 92.7 s) and runs at 4-6 % of
the WMMA peak. research/34 explained this with a latency chain (cell index -> K row -> V row, three dependent global
round trips per iteration of 32 cells, nothing prefetched) and expected -30 % (V early) to -55 % (full pipeline).

**Result: the latency theory is refuted for the default case.** With the KV cache in pinned RAM (the default above 1 GiB)
the kernel is limited by PCIe bandwidth and by how much of the gathered data the L2 already holds. A pipelined gather is
bit-identical, but only helps with the KV cache in VRAM (-7 % kernel time, 0.4 s of 92 s prefill at 128K). In RAM it was
5.9 % slower in the server, so the engine keeps the old gather there.

## What was built

- `engine/gpu/prefill_attn.hip`: `k_attn_wmma<Pipe>`. `Pipe = false` is the old code (ISA of `k_attn_wmma<false>`
  identical to main's `k_attn_wmma`, checked with `scripts/isa-diff.py`-style comparison of code and metadata, 104 VGPR).
  `Pipe = true` requests K, V and scales of the next 32 cells (registers, 119 VGPR, no spills) before the current ones
  compute, and the cell indices two iterations ahead. Same arithmetic, same order, same syncs.
  `AttnWmmaArgs::kv_host` (from `Mega::kv_host()`) selects: pinned RAM -> direct, VRAM -> pipelined.
- `engine/tests/test_gpu_attn_wmma.hip` + frozen copy of the old kernel `engine/tests/attn_wmma_ref.h`: output of the
  frozen kernel against both gathers, byte for byte. `--bench` times both.

First version of the pipeline: 197 VGPR and 1.31-1.36x *slower* than the old kernel. The compiler split the `uint4` loads
inside a branch into odd pieces (`global_load_b128 ... offset:269`, `d16_i8`) and waited (`s_wait_loadcnt 0`) right after
them, which defeats the prefetch. Unconditional loads of words, bytes cut out of the words by shifts: 119 VGPR, no wait
before the use.

## Exactness

- `test_gpu_attn_wmma`: 12 selections x KV in VRAM and in pinned RAM x 3 kernels (frozen, direct, pipelined): all
  byte-identical. Selections: dense path below the top-k (positions 0, 100, 2,000), random, drifting like neighbouring
  tokens, repeated cells, ragged lengths 1 ... 4,096 with repeated cells, positions 2,051 ... 262,000, 40-130 tokens.
  First run old against old (the frozen copy against itself in the first build): green.
- `scripts/golden.sh compare`, `GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash scripts/golden.sh compare`,
  `scripts/golden-server.sh compare` on `work/build/w78`: all identical (these runs use KV in VRAM, so the pipelined
  gather).
- 129,780-token prompt, `IE_DETERMINISTIC=1`, greedy, 200 tokens, `--ctx 140000`: sha256 prefix `40662553902d2bfb` for
  main and w78, with `IE_KV_HOST=1` and `IE_KV_HOST=0`.
- `scripts/isa-diff.py work/build/main work/build/w78`: only `k_attn_wmma` (now `<false>` and `<true>`) and the test
  kernels (`k_attn_wmma_ref`, `k_fill_kv`).

## What limits the kernel (micro-benchmark, `test_gpu_attn_wmma --bench`)

1,024 tokens per launch, synthetic cache and selections, median of 5 launches, us per token-layer (one token in one QSA
layer). `drift`: 90 % of the blocks of the neighbour token, base set renewed every 32 tokens; `churn`: renewed every
4 tokens, 30 % dropped; `random`: no sharing. Old kernel at position 128,000:

| Selection | KV in VRAM | KV in pinned RAM |
|---|---:|---:|
| drift | 6.7 | 10.0 |
| churn | 6.9 | 26.5 |
| random | 7.2 | 41.9 |

- In VRAM the time hardly depends on the sharing (6.7-7.3 us); `random` is 2.36 MB / 7.2 us = 330 GB/s, i.e. at the VRAM
  bandwidth (315 GB/s measured), the shared cases are not bandwidth-bound and sit at ~6.2-6.7 us: that is the cost of
  the kernel itself (syncs, dequantization, transposed V stores), far above the 2.4 us compute estimate of research/34.
- In RAM the time follows the *unique* bytes: `random` 2.36 MB / 41.9 us = 56 GB/s = the PCIe limit (55.7 GB/s). The real
  15-20 us per token-layer lie between `drift` and `churn`: the real sharing between neighbouring tokens decides, and it
  falls with the position (research/34 table), not a latency chain.
- Pipelined gather in the same benchmark (all positions 4,000 ... 250,000): VRAM 0.91-0.93x (drift/churn/random; 0.96 for
  random at 250K); RAM 0.97x (drift) to 1.05x (random).
- Grid order with the blocks of 16 / 64 / 256 tokens of one KV head side by side (more reuse in L2?): RAM 1.22-1.29x
  *slower* (churn, random), drift unchanged. Both heads of a token adjacent (the existing order) keep the 1,152 B
  rows together; the order was not changed. Bit-identical in all cases.
- Fewer blocks in flight (dummy LDS of 20-36 KB per block to push towards one block per CU) changed nothing measurable;
  not verified that the occupancy really dropped, not pursued.

## Server: `k_attn_wmma` per prefill, rocprofv3 kernel trace, 130,788-token prompt, `--ctx 140000`

`rocprofv3 --kernel-trace`, kernel time per prefill window summed from the trace; 3 rounds alternating main and w78, one prefill each.

| KV cache | build | `k_attn_wmma` total (3 rounds) | ms per call | prefill tok/s (3 rounds) |
|---|---|---|---|---|
| VRAM (42 expert slots) | main | 6.19 / 6.19 / 6.20 s | 8.06-8.07 | 1,423 / 1,419 / 1,419 |
| VRAM | w78 (pipelined) | 5.76 / 5.76 / 5.76 s | 7.50 | 1,425 / 1,417 / 1,426 |
| pinned RAM (84 slots), 1 round | main | 24.05 s | 91.1 | 1,397 |
| pinned RAM, 1 round | w78 (direct) | 24.06 s | 91.1 | 1,397 |

Kernel time in VRAM -7.0 % (spread <0.01 s), in RAM unchanged. The whole prefill in VRAM does not move beyond noise
(0.4 s of 92 s). Before the fix, the first pipelined build in RAM (single run): 25.41 s (+5.9 %), 1,377 tok/s.

## KV placement trade-off (M2): KV in VRAM against pinned RAM, whole prefill

Same build (w78), `IE_KV_HOST=0` against `1`, 3 rounds (spread of prefill ≤ 1 tok/s):

| ctx / prompt | KV VRAM (slots) | KV RAM (slots) | prefill VRAM | prefill RAM |
|---|---|---|---:|---:|
| 70,000 / 16K | 0.97 GB (65) | 0.97 GB (87) | 1,738-1,760 tok/s | 1,702-1,703 |
| 70,000 / 32K | | | 1,758-1,759 | 1,600 |
| 70,000 / 65K | | | 1,738-1,739 | 1,507-1,508 |
| 140,000 / 130K, main | 1.94 GB (42) | 1.94 GB (84) | 1,419-1,423 | 1,395-1,397 |

The kernel is 4x faster with the KV cache in VRAM at 128K (6.19 s against 24.05 s) and 15 % of the whole prefill at 64K
(1,738 against 1,508 tok/s), although the slots drop from 87 to 65. At 128K the 42 slots eat most of it (+2 %).
Decode after the prefill (single values, noisy): 128K VRAM 58-62 tok/s against RAM 64-68, because of the slots. No
automatic `IE_KV_HOST` policy was built (own ticket); these numbers are its input.

## Prefill curve in RAM (16K-250K): not run

The RAM path runs the old gather: the ISA of `k_attn_wmma<false>` equals main's `k_attn_wmma` (`isa-diff`), and the
128K `k_attn_wmma` time is 24.05 s against 24.06 s (3 profiled rounds were run for VRAM, one for RAM).

## bench-set and lang prefill, ctx 32,768 (KV in VRAM, so the pipelined gather), 3 rounds alternating

`bench-set` (11 code, 3 text prompts, MTP) and the 13K-token `long` prompt of `bench-openai.py`, main against w78, one
server start per run:

| Round | decode code, tok/s (main / w78) | decode text (main / w78) | prefill 13K, tok/s (main / w78) |
|---|---|---|---|
| 1 | 87.5 / 87.6 | 63.9 / 64.0 | 1,763 / 1,821 |
| 2 | 87.6 / 88.6 | 64.1 / 63.5 | 1,813 / 1,824 |
| 3 | 88.8 / 88.4 | 63.9 / 64.6 | 1,814 / 1,822 |
| mean | 88.0 / 88.2 | 64.0 / 64.0 | 1,797 / 1,822 (+1.4 %) |

Decode is not touched by this kernel (equal within the usual scatter). The 13K prefill is +1.4 % (round 1 of main is an
outlier; without it +0.5 %): `k_attn_wmma` is a small part of a short prefill.

## Open points

- The kernel stays at 6.2 us per token-layer in VRAM even with perfect sharing; research/34 ideas 5 (fewer syncs, no
  `part` buffer) and 6 (INT8 QK) address that part. They would help only with KV in VRAM (≤ ~77K tokens or
  `IE_KV_HOST=0`) until the RAM path gets less PCIe traffic.
- RAM path: only fewer bytes per token help (token union, numeric; needs u(T) of the real selections, not measured:
  no cell dump was made), or more of the cache in VRAM.
- The real sharing of neighbouring selections is unknown; the three synthetic patterns only bracket it.
