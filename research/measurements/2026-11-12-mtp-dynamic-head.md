# Dynamic rows of the MTP draft head, 2026-11-12

Status: **plan written before the runs; results are appended below it.**

## Question

Can the truncated draft LM head get the acceptance of the full head at (nearly) the cost of the truncated one by
choosing extra rows per draft step with a low-rank scorer (vLLM 59740, llama.cpp 30380)? The target verifies every token,
so only tokens per pass and the MTP step time can change, never the output text.

## Offline check before building more (real inputs of the head)

4,555 inputs `x` of the head (float, the vector the head matvec reads) were dumped from a bench-set run (11 code + 3
text prompts, 400 tokens each, adaptive depth, default build) and the full logits `W x` computed offline with the Q6_K
head dequantized (`scripts/build-mtp-scorer.py` holds the dequantization, tested against the engine's loop in
`tools/tests/test_build_mtp_scorer.py`). Share of steps whose full-head argmax lies in the static list, or in the top K
rows of the scorer outside the list (first 2,500 steps; the context rows of the engine are not simulated, so this is a
lower bound):

| Static list | static only | rank 64, K 512 | rank 128, K 256 | rank 128, K 1024 | rank 128, K 2048 |
|---|---:|---:|---:|---:|---:|
| current (41,501 ids) | 97.68 % | 99.40 % | 99.60 % | 99.80 % | 99.84 % |
| own ranking, 24,000 ids | 94.20 % | 98.08 % | 98.80 % | 99.28 % | 99.52 % |
| own ranking, 16,000 ids | 90.96 % | 96.68 % | 97.96 % | 98.68 % | 99.12 % |
| own ranking, 8,000 ids | 85.76 % | 94.40 % | 96.40 % | 97.68 % | 98.36 % |

Rank 32 is clearly worse (98.9 % at 41K, K 512). The bytes read per step follow: static rows 2,100 B (Q6_K), scorer
rank + 4 B per row outside the list, 2 x 2,100 B per picked row (copy, then read). Current list + context rows: 104 MB;
24K + rank 128 + K 1024: 84 MB; 41.5K + rank 64 + K 512: 120 MB.

## Plan

### Scope

Same build (`work/build/tona125`), model `work/models/coder-iq1_m/`, drafter `work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`,
`--ctx 32768`, greedy, `scripts/bench-set.py` (11 code, 3 text, up to 400 tokens, thinking off). One server start per
arm under `scripts/gpu_queue.py`. Tokens per pass and the times come from `TONA_DRAFT_LOG` (per verify pass: drafts,
accepted, verify ms, MTP ms, MTP steps); tokens per pass = (passes + accepted) / passes.

1. **Fixed depth** (`TONA_MTP_FIXED=1`, 7 drafts, same text in every arm): tokens per pass on code and text, MTP ms per
   pass. One run per arm; the first arm twice (repeat check). Arms:
   F0 today (list + 8,192 context rows) | F1 `TONA_MTP_HEAD=iq4` | F2 scorer rank 64, 512 rows | F3 rank 128, 1,024 rows |
   F4 rank 128, 1,024 rows on the 24K list | F5 rank 128, 2,048 rows on the 16K list | F6, F7 the best two with iq4 |
   F8 full head (`TONA_MTP_VOCAB=full`) | F9 old shard (`coder-iq1_m-0929`, profile `experts-coder-0929.bin`) like F0 |
   F10 list with Markdown weight 3 (41.5K) like F0.
2. **MTP step time** (`TONA_MTP_PROF=1` on the finalists, steps with one token).
3. **Decode tok/s, adaptive depth** (`scripts/ab.py`, workload `set`, base build against this build with the switch):
   the finalists against F0.
4. **Bit identity with the switch off**: `TONA_DETERMINISTIC=1` SHAs of the target (`ab.py` `short`) and the golden
   checks `tools/select_checks.py` names.

### Expectation

- Tokens per pass: F8 (full head) is the ceiling. From the earlier measurement on the old shard (2026-10-26-mtp):
  full head Q2_0 experts 5.41, list + context 5.27 on code (+2.6 %), text 3.00 against 3.01. The scorer arms should
  land between F0 and F8 (+0 to +2.5 % on code), text within +-1 %.
- MTP ms per pass: F0 about 8 ms (7 steps); scorer arms with the 41K list +10 to +20 % (the scorer adds 17 to 34 MB
  per step), arms with the 24K or 16K list -10 % to -20 % (fewer exact rows); iq4 -10 % on the head part.
- Source: the offline table above and the byte counts; tolerance +-1 % on tokens per pass (fixed depth should repeat
  exactly: a spread there is a finding), +-5 % on MTP ms.
- F9: coordinator observed 4.24 tokens per pass (adaptive, bench-set) on 2026-11-03 and about 3.2 on the current shard.

### Active tooling

`TONA_DRAFT_LOG` per arm; `TONA_MTP_PROF=1` only in step 2; no `TONA_DIAG`. The test conditions
(`tools/hw_report.py`) and `tools/quiet_check.py` before and after every arm are saved next to the logs
(`work/` runs of this file); the GPU is shared with other workers in a queue, so the CPU and desktop load is whatever
`quiet_check.py` reports.

## Results

Run order and raw files: arms under `/var/tmp/claude/tona125/runs/` (draft logs, server logs, quiet checks), `ab.py`
reports `work/ab/125-scorer-*.md` (logs next to them). Coordinator note (docs/contribute/testing.md): speed A/Bs of drafting run
without `TONA_DETERMINISTIC`, tokens per pass are pooled (tokens / passes over the requests).

### Fixed depth (7 drafts), one run per arm, tokens per pass code | text, MTP ms per pass (code)

| Arm | Configuration | Tokens/pass | MTP ms |
|---|---|---|---|
| F0 (repeat P0) | today: 41.5K list + 8,192 context rows | 5.199 \| 3.153 (repeat identical) | 8.09 (8.16) |
| F1 | `TONA_MTP_HEAD=iq4` | 5.167 \| 3.163 | 7.64 |
| F3 | 41.5K list, rank 128, K 1,024 | 5.269 \| 3.173 | 9.34 |
| G1 | 41.5K list, rank 128, K 256 | 5.278 \| 3.173 | 8.68 |
| F8 | full head | 5.295 \| 3.183 | 19.78 |
| F9 | old shard 0929, like F0 | 5.322 \| 2.994 | 8.13 |
| F10 | list with Markdown weight 3 (41.5K) | 5.199 \| 3.134 | 8.10 |
| F11, F12 | 24K / 16K list, no scorer | 4.964 \| 3.053 / 4.843 \| 2.882 | 7.04 / 6.62 |
| G2 | 24K, rank 128, K 512, 4,096 context rows | 5.244 \| 3.134 | 7.58 |
| G3 | 16K, rank 128, K 1,024, 4,096 context rows | 5.221 \| 3.134 | 7.32 |
| G4 / G5 | 12K / 8K, K 1,024, 2,048 context rows | 5.207 \| 3.106 / 5.159 \| 3.060 | 6.93 / 6.71 |
| G6 | G2 with the scorer made by the engine | 5.210 \| 3.134 | 7.55 |
| G7 / G8 | G2 with 0 / 1,024 context rows | 5.244 \| 3.115 / 5.244 \| 3.134 | 7.39 / 7.44 |
| K0 (= F4, final code) | 24K built-in list, rank 128, K 512, 1,024 context rows | 5.244 \| 3.134 | 7.37 (7.28) |
| K1 / K2 | K0 with K 256 / rank 64 | 5.179 \| 3.087 / 5.123 \| 3.060 | 7.32 / 7.21 |
| H1 / H2 / H3 | iq4 head with G2 / G3 / G4 | 5.195 \| 3.125 / 5.170 \| 3.106 / 5.156 \| 3.096 | 7.25 / 7.11 / 6.94 |

Spread: fixed depth repeats exactly for the same configuration (F0 and P0 equal); the differences between near-identical
arms (G2, G7, G8, K0: same tokens on code, text 3.115 to 3.134) show about 0.5 % in tokens per pass from text drift of
the arms that differ in the rows. Verify ms per pass 58 to 59.9 in all arms (the verify pass is not touched). Offline
recall (table above) is the reason the 24K list with the scorer holds the acceptance: 99.1 % to 99.7 % of the
full-head argmaxes against 97.7 % for today's list.

Modeled tokens per ms (verify 58.8 ms + MTP): F0 0.07806 per ms, K0 0.07933 (+1.6 %), G3 about +1.8 %, G4 about +1.9 %
on code; text -0.4 % to +0.4 %.

### MTP step time (`TONA_MTP_PROF=1`, 5,000 steps)

| Profile | per step | mtp.proj | mtp.dyn | mtp.gather | mtp.head |
|---|---|---|---|---|---|
| P0 default | 1000.6 us | - | - | - | 343.9 us |
| P2 (24K, K 512, 4,096 context rows, gather as a separate op) | 950 us | 14.7 | 64.4 | 6.8 | 206.8 |
| P3 (final code before the histogram copies) | 922.3 us | 7.4 | 70.4 | in dyn | 184.8 |
| P4 (final code) | 912.5 us | 7.4 | 61.2 | in dyn | 183.2 |

Ideas tried to cut the cost of the dynamic part: gather fused into the selection step (separate op 6.8 us gone), float4
loads of p in the projection (14.7 to 7.4 us), 8 histogram copies and a parallel boundary search in the radix passes
(70.4 to 61.2 us), a candidate id window below 98,304 (no loss: 99.72 % against 99.68 % over all 206,819 ids).

### Adaptive depth, `ab.py`, workload `set`, A = base build f9b1629, B = this build with the switch (no `TONA_DETERMINISTIC`)

| Arm B | decode code, B vs A, mean [CI] | decode text | tokens/pass code \| text (A: 4.454 \| 2.714, all rounds equal) | file |
|---|---|---|---|---|
| 24K list, rank 128, K 512 (final defaults) | +1.42 % [-4.38, +7.22] (round 1 is a cold outlier +11.95 %, rounds 2 to 6 -0.7 % on average: undecided) | +3.41 % [-5.88, +12.70] (round 1 +20.6 %, rounds 2 to 6 -0.03 %) | 4.468 \| 2.663 | 125-scorer-adaptive |
| 16K list (`mtp-vocab-125-n16k.txt`), K 1,024, 4,096 context rows | +0.83 % [+0.01, +1.65], 4 rounds, B faster | -0.97 % [-1.97, +0.03] | 4.511 \| 2.664 | 125-scorer-adaptive-16k |
| 24K list with Markdown weight 3, K 1,024, 4,096 context rows | +0.52 % [-0.40, +1.43], undecided after 6 rounds | +0.03 % [-1.01, +1.07] | 4.572 \| 2.711 | 125-scorer-adaptive-md3-24k |

Result against the plan (expected +0.5 to +3 % on decode): within for code in the three arms, no decided gain beyond
+0.8 %; the tokens per pass gain (up to +2.6 % on code with the md3 list) shows up as +0.5 % in decode. Tokens per pass
are identical in every round of an arm (greedy, same text), so the spread is in the rate only (A: 87.0 to 88.5 tok/s
on code, 67.1 to 67.9 on text).

Bit identity with the switch off (`TONA_DETERMINISTIC=1`, `ab.py` `short`, 6 rounds): SHA 11eacf1ee10017b4 (2000) and
a6f07fd6fb86e448 (2000:write) equal in A and B, expert cache counts identical; decode and prefill within +-1 % (the
prefill@2000 mean +1.4 % [-1.7, +4.6] is undecided, outliers in both arms). Golden checks with the switch off: see the
end of this file.

### Test conditions

`tools/hw_report.py` (taken during the queue, the GPU busy with a job of another worker; the arms run alone behind the
lock): see `hw_report.txt` block below. `tools/quiet_check.py` before every arm: "warning: 1.19 GiB of 15.9 GiB VRAM
are held by others" (desktop). The "after" check of `arm.sh` runs while the arm's server still holds its VRAM (15.3 GiB)
and therefore says "do not measure"; it is not a state of the machine during the run.

```
### Test conditions

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 15.23 GiB held by others before the run, 8 % busy before the run
- RAM: 60.5 GiB, 54.6 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.74, 1.76, 1.89
- CPU use by other programs over 0.5 s: 6.7 % of all threads, swap in use 1.74 GiB
- sandbox: no
- Commit: `adb0ca3` (dirty tree)
```

## Round 2 on main 1f17a93 (the verify-pass rework landed: verify 58.8 -> 55 ms per pass)

Base A = main 1f17a93 (`tona125base`), B = this branch rebased, `ab.py` workload `set`, no `TONA_DETERMINISTIC`, same
size as round 1. Tokens per pass of A (all rounds equal): code 4.672, text 2.726.

| Arm B | decode code B vs A [CI] | decode text | tokens/pass code \| text | file |
|---|---|---|---|---|
| 16K list file, rank 128, K 1,024, 4,096 context rows | +0.96 % [+0.13, +1.79], B faster after round 2 | +1.00 % [+0.19, +1.82] | 4.704 \| 2.790 | `-ab-r2-16k` |
| 24K Markdown-weighted list, same | +1.06 % [+0.05, +2.07] | **-4.60 % [-5.82, -3.37]** | 4.698 \| 2.548 | `-ab-r2-md3-24k` |
| **new defaults** (built-in 16K list, rank 128, K 1,024, 4,096 context rows) | **+1.51 % [+0.31, +2.71]** | **+1.67 % [+0.78, +2.56]** | 4.704 \| 2.790 | `-ab-scorer-default` |

The Markdown-weighted list loses on text (-4.6 %) and is dropped. The 16K list wins on both classes and is the default
(rule "Fokus Coding": a decided code gain justifies a small text loss; here there is none).

### Constants of the width policy (`t_step_`, `t_fill_`)

Fixed depth 3 and 7 (`TONA_DRAFT_N`, `TONA_MTP_FIXED=1`), MTP ms per pass of the draft log, step = (ms7 - ms3) / 4,
fill = ms3 - 3 x step:

| Build | ms at 3 / 7 drafts (code, text) | step | fill |
|---|---|---|---|
| base main | 3.49 / 7.88, 3.36 / 7.69 | 1.09 ms | 0.15 ms |
| scorer (24K list, K 512 at the time; profile 911.7 us per step against 1004.1 us for the base) | 3.11 / 7.20, 3.06 / 7.05 | 1.01 ms | 0.05 ms |

Constants in the code: 1.15 / 1.3 ms. Test of the measured base values in the policy (`ab.py`, A = base, B = base with
`TONA_MTP_STEP_MS=1.09 TONA_MTP_FILL_MS=0.15`, 6 rounds): decode code **-1.27 % [-2.08, -0.46]**, text -0.25 %. The
logged MTP ms are not the quantities the policy needs (it also pays for host work around the chain), so the constants
stay at 1.15 / 1.3 ms for both builds (the scorer changes the step by less than 0.1 ms).

## Reading and decision

The scorer works as designed: 98.9 % (16K list, K 1,024) to 99.7 % (41.5K list, K 1,024) of the full-head argmaxes
against 97.7 % for the old list. In round 1 (before the verify-pass rework) the decode gain was at most +0.8 % on code
with the 16K list and -1.0 % on text, undecided for 24K, and the switch stayed off. On the faster verify pass of round 2
the head is a larger share of the pass: the 16K arm wins both classes by +1.0 %, the defaults by +1.5 % and +1.7 %, so
**`TONA_MTP_SCORER=128` is now the default**. `TONA_MTP_SCORER=0` returns to the old head, bit-identical to the base:
SHAs equal in `short` deterministic (`-ab-off-identity-r2`), all 36 server goldens identical.

Effect on the goldens with the new default (build `tona125`): `gpu`, `cpu plan`, `prefill dflash`, `prefill` identical;
server golden: 31 of 36 cases identical (all greedy ones); the four `sampled-*` cases (temperature 0.7, seed 42) and
`reuse-long-2` differ. Cause shown: with `TONA_MTP_SCORER=0` all 36 are identical again (including `reuse-long-2`
against its re-recorded reference). Seeded sampling draws through the draft's rejection step, so another draft head
changes the sample for the same seed (valid output, other text); `reuse-long-2` differs only in the cache counters
(`cache_read_input_tokens` 2,190 -> 2,139, `input_tokens` 25 -> 76), the same kind of deviation as the verify-pass
rework. The references are not re-recorded here (shared `work/golden/`).

### Principle 8 (docs/contribute/measurement.md): arms that are not made default

1. **Markdown-weighted 24K list:** -4.6 % on text decode (tokens per pass 2.548 against 2.726). The 16K frequency list
   needs no such weighting.
2. **Larger lists (24K, 41.5K) with the scorer, iq4 head, rank 64, K 256:** not better in tokens per ms in the
   fixed-depth table; the iq4 combinations (H1 to H3) were not decided against the plain scorer.
3. **Fresh research on a cheaper way** (2026-11-12, `tools/papers/scan.py`, arXiv topic speculative 2025-11-12 to
   2026-11-09, GitHub 2026-10-13 to 2026-11-09, plus targeted reading): vLLM PR 59740 (open; Gram eigenvectors top 256,
   FP8 scorer rows, topk 16K per draft token, exact gather), llama.cpp PR 30380 (closed, split), vLLM 59973 (quantized
   draft head, merged: our `TONA_MTP_HEAD=iq4`), vLLM 60226, SlimSpec 2605.10453 (low-rank draft head, needs training),
   NanoSpec 2605.26444 (in-context vocabulary, like our context rows), SpecVocab 2602.13836, EvoSpec 2605.27390,
   FlashHead 2603.14591, CSV-Decode 2511.21702, vocabulary trimming 2603.05210. Nothing cheaper per step without
   training a head.
4. **Active attempts to remove the cost:** gather fused into the selection (6.8 us), float4 projection (-7 us),
   histogram copies and parallel boundary search (-9 us), candidate id window, lists 24K to 8K, K 256 to 1,024, rank 64
   and 128, context rows 0 to 4,096, iq4 combined, Markdown weight, constants of the width policy (worse, above). Not
   tried: a scorer in FP8 rows (needs WMMA kernel work).

## Golden checks, round 1 (switch off, before the rebase)

`scripts/golden.sh compare` gpu (28 s), cpu plan (27 s), prefill dflash (27 s), prefill (154 s), and
`scripts/golden-server.sh compare` (193 s, run on port 8099 through a copy with the port changed): all identical to the
recorded references (exit 0). Unit tests `test_gpu_mtp_dyn` and `test_gpu_mtp_nan`: ok. `just check`: all hooks and
clang-tidy green.
