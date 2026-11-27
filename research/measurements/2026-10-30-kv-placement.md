# KV placement and KV staging (#80)

Reference platform, model Qwen3.8-Flash-Next Coder IQ1_M, drafter MTP. Server on port 8094, `OMP_NUM_THREADS=6`, one
server start per run, code corpus of `research/measurements/2026-10-30-prefill-attn.md` (curve.py, thinking off, greedy).
Every value is the mean of 3 rounds alternating between the builds; the range is in brackets. Raw logs:
`/var/tmp/claude/w80/` (`s1-*` step 1, `s5-*` final, `bx*` and `s6-*` bit-exact), scripts `srv.sh`, `s1.sh`, `s5.sh`.
**Prefill** = prompt tokens computed per second of the prefill; **decode** = milliseconds per pass, taken from the log
line `Request:` (seconds / (tokens / tokens per pass)), 460-600 tokens per request in step 1 and 200 in the final runs.

## Step 1: what the slots cost and what the KV place brings (main c25f9b2)

"RAM, few slots": KV in RAM and `IE_VRAM_GB` lowered so that the expert slots equal the VRAM case (10.4 at `--ctx 140000`:
42 slots, 12.6 at `--ctx 70000`: 67 slots; `IE_VRAM_GB` is the limit that `size_slots()` subtracts the buffers from).
Slots: ctx 140000 RAM 82, RAM few 42, VRAM 40; ctx 70000 RAM 85, RAM few 67, VRAM 63.

Prefill, tok/s (spread at most 3 tok/s, 128K RAM few 1,191-1,199):

| ctx | prompt | RAM | RAM, few slots | VRAM |
|---|---|---:|---:|---:|
| 140000 | 16K | 1,661 | 1,415 | 1,484 |
| 140000 | 32K | 1,602 | 1,356 | 1,484 |
| 140000 | 64K | 1,508 | 1,279 | 1,466 |
| 140000 | 128K | 1,404 | 1,195 | 1,435 |
| 70000 | 16K | 1,696 | 1,644 | 1,769 |
| 70000 | 32K | 1,594 | 1,563 | 1,759 |
| 70000 | 64K | 1,504 | 1,470 | 1,742 |

Decode, ms per pass (mean, range about +-1.5):

| ctx | RAM | RAM, few slots | VRAM |
|---|---:|---:|---:|
| 140000 | 42.5-43.9 | 46.1-47.9 | 45.8-47.1 |
| 70000 | 43.4-43.8 | 44.0-45.1 | 43.8-45.0 |

- The slots cost a lot: 82 to 42 slots is -15 % prefill at 128K and +3-4 ms per pass decode (+8 %). 85 to 67 slots is
  -3 % prefill and +1 ms (within the spread).
- The KV place VRAM alone (same slots) brings +20 % prefill at 128K, +15 % at 64K, +8 % at 16K. Decode: no gain (RAM few =
  VRAM).
- The old rule (VRAM below 1 GiB of KV) is at the edge: ctx 70000 (0.97 GB) in VRAM wins prefill by 5-16 % at the same
  decode; ctx 140000 in RAM wins everywhere except 128K (+-2 %).

## KV staging (variant c)

During a layer-by-layer prefill the KV rows of one QSA layer (layer >= 2) lie in the borrowed expert slots, the new rows go
to the pinned cache after the layer, the old rows come from there before it (`PrefillRun::stage_in/stage_out`). Decode,
slots, checkpoints and stash see the pinned cache as before. The staged plan may lower the chunk (250K: 5120 to 4096) and
the group (`choose_plan`).

### Prefill curve, `--ctx 262144` (KV 3.62 GB in RAM), cold prompts

| prompt | main, tok/s | w80, tok/s | prefill | decode ms/pass main / w80 |
|---|---:|---:|---:|---|
| 8K | 1,677 (1,654-1,689) | 1,724 | +2.8 % | 44.5 / 44.7 |
| 16K | 1,644 | 1,750 | +6.4 % | 43.7 / 44.4 |
| 32K | 1,576 (1,571-1,578) | 1,778 | +12.8 % | 44.3 / 43.1 |
| 64K | 1,502 (1,500-1,504) | 1,773 | +18.0 % | 43.8 / 43.7 |
| 128K | 1,395 (1,394-1,396) | 1,728 | +23.9 % | 42.9 / 43.1 |
| 250K | 1,289 (1,279-1,295) | 1,622 (1,622-1,623) | +25.8 % | 43.8 / 43.5 |

The w80 spread is at most 6 tok/s (8K-128K) and 1 tok/s (250K). Decode is equal within the spread (range 41.9-45.8).
With the first version (plan unchanged, staging only where it fitted) 250K reached 1,413 tok/s: the borrowed range was
full at chunk 5120/group 15360, so the staging got only the first 8 groups. The plan now reserves the staging
(`Mega::pf_layered_need(..., stage)`).

### Follow-up prefill (prompt cache reuse, new tokens appended)

| old + new | main | w80 | prefill |
|---|---|---|---|
| 67K + 6.2K | 5.6 s (1,113 tok/s) | 4.4 s (1,409 tok/s) | +26.6 % |
| 67K + 30.5K | 27.8 s (1,098) | 22.1 s (1,388) | +26.4 % |
| 208K + 7.9K | 10.6 s (one run) | 6.7 s, 9.6 s | not slower |

Prompts of up to `IE_PF_LCHUNK` (6,144) new tokens go chunk by chunk and are unchanged (no staging). The cost of staging is
the host-to-device copy of the old rows once per layer and group: at 67K old about 0.9 GB, a few hundred ms against
several seconds of attention saved, so no break-even rule is needed; the layered path starts at 6,144 new tokens
and wins there already (+26.6 %). The 208K run has a large spread (6.7 s and 9.6 s for the same prompt), main has one value (a
second main run aborted, see below).

### ctx 70000: variant (a), the 1 GiB threshold (KV 0.97 GB)

Prefill, tok/s; decode in ms/pass:

| prompt | main = w80, KV VRAM (63 slots) | main, KV RAM (85 slots) | w80, KV RAM + staging (85 slots) | staging against VRAM |
|---|---:|---:|---:|---:|
| 16K | 1,763 / 1,769 | 1,696 | 1,829 | +3.4 % |
| 32K | 1,768 / 1,765 | 1,594 | 1,791 | +1.5 % |
| 64K | 1,743 / 1,745 | 1,501 | 1,778 | +1.9 % |
| decode | 43.9-44.9 / 44.2-45.3 | 42.6-43.0 | 42.7-44.8 | equal |

With staging the pinned KV beats the VRAM KV at ctx 70000 on prefill by 1.5-3.4 % with 22 more expert slots, and decode
is equal within the spread. The threshold can go down; where exactly (decode with KV in RAM at ctx below 32K) is not measured.

### bench-set and bench-parallel, `--ctx 262144`, 3 rounds alternating (decode)

| | main | w80 |
|---|---|---|
| bench-set code, tok/s | 88.0, 87.8, 88.3 | 87.9, 87.7, 87.9 |
| bench-set text, tok/s | 64.2, 63.8, 64.1 | 64.2, 63.3, 64.3 |
| bench-parallel 2 clients, total tok/s | 94.0, 92.6, 93.5 | 93.4, 93.8, 92.7 |

(bench-parallel token counts differ from run to run on both builds, with the order in which the two clients are served.)

## Bit-exact (`IE_DETERMINISTIC=1`, 200 tokens, sha256 of the text)

`ctx 140000`, prompts 30K, 130K (follow-up of the 30K conversation), 64K (follow-up), 72K (follow-up of 64K): the four hashes
are equal for main (default, `IE_KV_HOST=0`, `IE_KV_HOST=1`, `IE_SEQS=1`) and w80 (default, `IE_KV_HOST=0/1`,
`IE_KV_STAGE=0`, `IE_SEQS=1`; the default is `IE_SEQS=2`). Hashes: fdd7f919b4727583, 97c824605134c74e, c1f6d32975183ef0,
036d297dadad74d7. `IE_KV_STAGE=0` (final build) and main, prompts 30K, 64K, 72K: fdd7f919b4727583, eecc53cc7014e4ea, fbd54e70f3ef8681. `ctx 262144`, prompt 250K: main and w80 both 2a8eac8ea5595017. Prompts 200K and 208K and the 8K follow-up:
f270fcfb7b884a94, c7217bd1f626ad97, 816f70875bbab743 for main and w80.
`scripts/golden.sh compare`, `GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash scripts/golden.sh compare` and
`scripts/golden-server.sh compare`: identical, with the default and with `IE_KV_HOST=1` (the golden contexts are small and
so use the VRAM cache by default).

## Open points

- Barrier abort on **main** (c25f9b2, not on w80), `--ctx 262144`, third request of the run (prompt of 208K tokens after a
  200K conversation, `IE_DETERMINISTIC=1`): `Pass failed: gpu::Mega: barrier aborted (time limit) at position 207780 (step
  673: matvec gr.up, longest step 19 moe.down 0.5 ms, pass up to there 33.0 ms, blocks at step 673, all equal)`. Log
  `/var/tmp/claude/w80/s6-r1-main-big.log` (#75).
- Sporadic slow decode right after a follow-up prefill (33-40 tok/s instead of 60-64): 1 of 48 requests on w80 in the final runs,
  2 of 4 reuse runs on w80 and main once (130K, 39.7 tok/s) in the bit-exact runs. Not tied to the staging (it appears on main);
  not investigated.
- Staging is off for the chunk-by-chunk path (prompts up to one chunk of 6,144 tokens) and for layers 0 and 1 (DeltaNet).
- Not measured: the staging with a second sequence prefilling while the first one decodes (`IE_SEQS=2` with two clients
  only ran in bench-parallel, with short prompts).
