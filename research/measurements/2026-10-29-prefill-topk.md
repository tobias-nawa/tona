# Prefill: faster block selection in k_topk (#71, 2026-10-29)

Question: `k_topk` (selection of the 512 best pool blocks per token) was the largest single item of the 128K prefill
(30.5 s of 121.9 s GPU time, research/29-long-context-prefill.md). How much does a multi-bit radix selection with
an identical result bring?

Change (`engine/gpu/prefill_topk.hip`, commit 52f3598): three radix passes (11, 11, 10 bits of the 32-bit score) with
an LDS histogram instead of 32 one-bit passes; suffix sum of the histogram by wave shuffles instead of thread 0;
counting and output per wave stripe with ballot and popcount, cells ascending as before. Threshold (the n_top-th
largest value as an unsigned integer) and tie rule (earlier blocks first) are unchanged.

## Exactness

- `ctest -R gpu_topk` / `work/build/w71/test_gpu_topk` (`engine/tests/test_gpu_topk.hip`): `cells` and `ncells` byte for
  byte against a CPU reference of the rule, 120 cases (6 score patterns: random, many ties, all equal, zeros, two
  levels, descending; 10 start positions from 2,051 to 262,111; batches of 32 and 7 tokens, `t0` 0 and 5). Old and new
  kernel: 120 of 120 green. Control: a deliberately wrong tie rule makes 81 of 120 red.
- `scripts/golden.sh compare`, `GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash scripts/golden.sh compare`,
  `GOLDEN_MODE=prefill scripts/golden.sh compare`, `scripts/golden-server.sh compare`: all identical.
- 130,775-token prompt, `IE_DETERMINISTIC=1`, greedy, 200 tokens, `engine` (main) and `w71`: identical text
  (sha256 prefix c509a0296420288e).
- `scripts/isa-diff.py work/build/engine work/build/w71`: only `k_topk` differs in code (VGPRs 46 -> 32).

## Kernel time (rocprofv3, 130,788-token prompt, `--ctx 262144`)

| | `k_topk` total | calls | per call | GPU time of the prefill window |
|---|---:|---:|---:|---:|
| main | 30.53 s | 48,288 | 0.632 ms | 121.9 s |
| w71 | 1.27 s | 48,288 | 0.026 ms | 92.7 s |

Micro-benchmark `test_gpu_topk --bench` (32 tokens, random scores, wall time per launch incl. launch overhead): 64K
0.81 -> 0.092 ms, 128K 2.45 -> 0.095 ms, 262K 3.89 -> 0.161 ms.

## Prefill over the prompt length (server, MTP, `--ctx 262144`, 3 rounds, builds alternating)

Setup as in 2026-10-28-long-context (engine sources as prompt, random first line, greedy, 200 tokens). Each round
starts a fresh server per build and runs 16K, 64K, 128K, 250K in this order. Raw output:
`/var/tmp/claude/w71/curve-ab.out` (not in the repository).

| Prompt | main (3 runs), tok/s prefill | w71 (3 runs), tok/s prefill | Change |
|---:|---|---|---:|
| 16K | 1,579 / 1,612 / 1,583 (mean 1,591) | 1,656 / 1,664 / 1,633 (mean 1,651) | +3.8 % |
| 64K | 1,396 / 1,396 / 1,387 (mean 1,393) | 1,512 / 1,511 / 1,497 (mean 1,507) | +8.2 % |
| 128K | 1,041 / 1,048 / 1,109 (mean 1,066) | 1,401 / 1,402 / 1,388 (mean 1,397) | +31 % |
| 250K | 599 / 599 / 706 (mean 635) | 1,286 / 1,288 / 1,285 (mean 1,286) | +103 % |

The spread of main at 128K and 250K is large because of the third round (1,109 and 706); the first two rounds agree
within 1 %. 250K: 194 s instead of 417 s to the first token.

## Decode must not get worse

`scripts/bench-set.py` and `scripts/bench-openai.py URL x 1 long`, `--ctx 32768`, 3 interleaved rounds. Decode does not
run `k_topk` (the megakernel has its own `op_topk`), so equal numbers are expected.

| | main | w71 |
|---|---|---|
| decode code, tok/s | 87.2, 88.3, 88.3 | 88.3, 88.2, 88.2 |
| decode text, tok/s | 63.6, 64.3, 64.3 | 64.3, 64.3, 64.3 |
| prefill 13K, tok/s | 1,701, 1,718, 1,719 | 1,778, 1,779, 1,779 |
| decode after 13K | 62.2, 62.0, 57.7 | 64.4, 65.9, 60.5 |

Decode after the 250K prompt (200 tokens, server log per request):

| | tok/s decode | tokens per pass | passes | time per pass |
|---|---|---|---|---|
| main | 46.4 / 45.4 / 46.4 | 3.02 / 2.78 / 2.86 | 66 / 72 / 70 | 65.2 / 61.1 / 61.4 ms |
| w71 | 42.8 / 44.0 / 43.5 | 2.65 / 2.84 / 2.84 | 75 / 70 / 70 | 62.7 / 64.3 / 65.7 ms |

Most of the gap is the MTP acceptance of the different texts (each run has its own random first line); the time per
pass overlaps (61.1 to 65.2 against 62.7 to 65.7 ms; the log rounds the time to 0.1 s, about 2 %). Decode code is
identical in both builds (isa-diff) and does not run `k_topk`. A small thermal effect after a prefill that keeps the
GPU busier is possible and not measured.

## Full curve on main after the merge (ded7e00, 3 rounds)

Same setup as the idle repeat in 2026-10-28-long-context (fresh server per round, 8K to 250K ascending, idle
desktop). Raw: `/var/tmp/claude/ctx-curve/curve-after.log` (not in the repository). The desktop held more VRAM this
time: 75 instead of 81 expert slots per layer (VRAM 13.98 instead of 14.57 GB), which costs the "after" side a little.

| Prompt (tokens) | Prefill tok/s, rounds 1 / 2 / 3 | Before (median) | Time to first token, before -> after | Decode tok/s after it |
|---:|---|---:|---|---|
| 8,105 | 1,666 / 1,666 / 1,666 | 1,630 | 5.0 -> 4.9 s | 70.0 / 64.7 / 65.0 |
| 16,277 | 1,638 / 1,636 / 1,637 | 1,615 | 10.1 -> 9.9 s | 63.0 / 65.8 / 65.4 |
| 32,357 | 1,586 / 1,583 / 1,585 | 1,538 | 21.0 -> 20.4 s | 59.8 / 69.2 / 61.4 |
| 65,254 | 1,497 / 1,495 / 1,495 | 1,394 | 46.8 -> 43.6 s | 56.8 / 55.9 / 57.2 |
| 130,788 | 1,380 / 1,387 / 1,386 | 1,047 | 124.9 -> 94.3 s | 56.2 / 56.6 / 55.3 |
| 249,732 | 1,267 / 1,284 / 1,285 | 597 | 418.5 -> 194.4 s | 43.5 / 44.5 / 44.0 |

Decode after the prompt is unchanged within the spread of the MTP acceptance (before, medians: 67.9 / 66.2 / 64.4 /
55.4 / 55.5 / 44.8 tok/s).

## Where the time goes now (128K, GPU time 92.7 s)

`k_attn_wmma` 24.0 s (26 %), `k_gemm2<256>` 8.3 s, `k_gemm2<128>` 7.1 s, `k_dn_wave` 6.4 s, `k_idx_score_wmma` 4.2 s,
`k_topk` 1.3 s. Next at long context: `k_attn_wmma` (91 ms per call).
