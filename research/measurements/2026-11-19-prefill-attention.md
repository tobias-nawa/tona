# Prefill flash attention: one tile per wave (2026-11-19, #175)

Kernel: `k_fattn_prefill` (`engine/gpu/prefill_fattn.hip`), used by Ornith and Bolt (the Coder prefills with the QSA kernel, not this one;
its `is_qsa` layers never reach it). Starting point: `2026-11-19-ornith-opt2.md` item 3 (attention is 81 % of Ornith prefill at 64K at 9 %
of the WMMA peak) and `2026-11-19-bolt-opt2.md` P1 (27 % of Bolt kernel time at 16K, about 7 TFLOPS).

## Test conditions

Output of `python3 tools/hw_report.py` before the first run (21:50): RX 9060 XT gfx1200, 15.9 GiB VRAM, 1.1 GiB held by the desktop, 0 % busy;
Ryzen 9 9950X, 55.6 GiB RAM available, load average 1.0 / 2.6 / 1.9, other programs 0.2 % of the threads, swap 1.5 GiB in use, no `TONA_*` set,
commit `47af3ee` plus this branch. Quiet check 21:50: only warning "1.1 GiB of 15.9 GiB VRAM held by others" (the desktop). Microbenchmarks
below are kernel timings in one process (median of 5 launches after a warm-up), no server.

## Prior art (searched 2026-11-19)

- [llama.cpp #18481 HIP: add fattn-mma-f16 for RDNA4](https://github.com/ggml-org/llama.cpp/pull/18481), merged 2026-01-13: the MMA flash
  attention of llama.cpp on gfx12; [#28102 Flash Attention tuning (gfx1201)](https://github.com/ggml-org/llama.cpp/pull/28102), merged
  2026-09-11; [#22880 RDNA3 mma FA, faster AMD transpose](https://github.com/ggml-org/llama.cpp/pull/22880), merged 2026-05-14;
  [#26419 MMA FlashAttention for head dim 256 on AMD RDNA](https://github.com/ggml-org/llama.cpp/pull/26419), **open** (updated 2026-09-15).
  All tile 32 to 64+ query rows per block with a Q tile per warp and keep S and P in registers. Taken over: one Q tile per wave, scores and
  probabilities in registers. Not taken: their summation order (ours must stay as `fattn_tile.h` for bit-identity).
- [AMD GPUOpen, WMMA guide for RDNA 4, part 1](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/): the operand swap (compute S^T = K Q^T)
  so that the accumulator layout of one WMMA is the B-operand layout of the next; the guide says llama.cpp's RDNA 4 flash attention uses it.
  Adopted: this is the core of the new kernel (no LDS round trip for P, one row per lane).
- Composable Kernel has a gfx12 WMMA header (`wmma_gfx12.hpp`) but no gfx12 FMHA forward pipeline that I found; Modular's Mojo
  `mha_rdna` is an RDNA wave32 prefill/decode path on the same 16x16x16 WMMA. Both: same tile shape, nothing beyond llama.cpp to take.
- FlashAttention-2 (work partitioning: split queries over warps, not the head dimension, no cross-warp reduction of S): the idea the new
  kernel applies. The old kernel splits the head dimension over 8 waves and sums 8 partial score tiles through LDS (8 LDS reads per
  probability, 16 of 32 lanes busy in the softmax). FA-3/4 (warp specialization, async copies, ping-pong softmax) are Hopper/Blackwell
  features without a counterpart in RDNA4 (no TMA, no warpgroup MMA): not taken.

Search: tools/papers/scan.py not run (web and `gh pr view` by hand, no topic for RDNA attention); topics: flash attention RDNA4 WMMA, CK FMHA
gfx12; window 2025-11-19..2026-11-19 (12 months); sources: web search, GitHub PRs.

## Ceiling

Causal attention FLOP per layer for n tokens from position 0: 4 * hd * nh * n(n+1)/2. Ornith (hd 256, 16 heads): 16K tokens 4.4e13
(0.47 ms at the 93 TFLOPS peak per... see the table of results); the exact numbers come from the bench below (`--prefill-bench` prints
TFLOPS and percent of peak per kernel).

## Plan 1: bit-identity and kernel microbench of the wave kernel (written before the run)

Scope: `test_gpu_fattn --prefill-cmp` (wave vs multi-tile vs single-tile, all group sizes 1..16, hd 128 and 256, ragged token counts, bytes
compared) and `--prefill-bench` (Ornith 16/2 hd 256 and Bolt 32/2 hd 128 at 2K, 16K, 32K tokens, one call, median of 5). One `gpu` job for the
comparison, one `exclusive` job for the bench. Expectation: bits equal everywhere (the summation order is kept by design: 8 partial sums added
in turn, 16 cells per softmax step, the wave_sum tree). Speed (source: old kernel at 9 % of peak and 7 TFLOPS; FA-2-style kernels of llama.cpp
reach 25 to 40 % of the WMMA peak on RDNA): wave kernel 2 to 4 times faster than the multi-tile kernel at 16K for hd 128, hd 256 limited by
VGPRs (256 allocated, spills possible). Decision: if bits differ the kernel is wrong; if it is not faster at 16K the change goes out.
Active tooling: none; `-Rpass-analysis=kernel-resource-usage` for registers (compile time).

## Result 1: bits and kernel microbench (jobs `pf-cmp`, `pf-bench`, build `pf1` = `1479756`; quiet check before: only the desktop's 1.1 GiB VRAM)

`test_gpu_fattn --prefill-cmp`: wave kernel = multi-tile = single-tile **bit for bit** in all 60 cases (hd 128 and 256, groups 1, 2, 4, 8, 16
from 2 to 32 q heads per 2 KV heads, 1 to 4097 tokens, pos0 0 to 5000). The other GPU tests (`test_gpu_fattn` incl. the oracle checks,
`test_gpu_mamba`, `test_gpu_moe_latent`, `test_gpu_bolt_quant`) pass.

Microbench `--prefill-bench` (one call from position 0, median of 5), first version W = 8 waves per block, chunk 32 cells, then the final
W = 16 (hd 128 chunk 64, hd 256 chunk 32; hd 256 at 64 cells needs 70 KB LDS > 64 KB):

| Kernel | hd 256, 16 q heads, 16K | 32K | hd 128, 32 q heads, 16K | 32K |
|---|---|---|---|---|
| multi-tile (old) | 12.78 us/token, 10.5 TFLOPS (11 % of peak) | 23.9, 11.2 | 21.8, 6.2 (7 %) | 40.2, 6.7 |
| wave W = 8 | 3.60, 37.3 TFLOPS (40 %) | 7.15, 37.6 | 3.29, 40.8 (44 %) | 6.50, 41.3 |
| wave W = 4 (chunk 32) | 6.56 (22 %) | | 3.93 (37 %) | |
| **wave W = 16 (final)** | **3.24, 41.5 TFLOPS (45 %)** | **6.49, 41.4** | **3.14, 42.8 (46 %)** | **6.20, 43.3** |

Per token and layer; "peak" = 93 TFLOPS f16 WMMA, FLOP = 4 * hd * q heads * causal pairs. Speed-up over the old kernel at 16K: **3.9x (hd 256), 6.9x
(hd 128)**; at 2K 3.4x and 6.7x. Registers (`-Rpass-analysis=kernel-resource-usage`): hd 128 187 VGPRs no spill, occupancy 8; hd 256 256 VGPRs, 11
spilled (48 bytes scratch per lane), occupancy 5. The old kernel: 59 to 177 VGPRs without spill.

Why it is faster (ISA-level reading, not a profile): the old tile summed 8 partial score tiles from LDS (8 LDS reads per probability), ran the
softmax with half the lanes idle, wrote P back through LDS and used 5 barriers per 16 cells; the wave kernel has 2 barriers per 32 or 64 cells.
The remaining gap to the peak is the softmax (exp, 8 sequential adds of partial sums to keep the summation order) and the staging with 16 waves.
Open levers: the hd 256 spills (Q in LDS), double-buffered staging, fewer VALU adds if the order may change (it must not).

## Plan 2: server speed with `ab.py` (written before the runs; three `exclusive` jobs, build `pf1` at `1479756` in both arms)

A = `TONA_FATTN_PF_OLD=1` (the multi-tile kernel of the previous round), B = default (wave kernel), alternating AB/BA, 2 rounds (no noise entry
for these workloads; stop when decided), one discarded warm-up start, no `--drafter` (MTP head from the file for Ornith as in the previous round,
`--drafter none` for Bolt as in its B0), `--no-expert-profile` for Bolt as in B0. Jobs: **Ornith** workload `server` targets `2000:100 16000:100
64000:50`, `--ctx 70000`; **Bolt** workloads `prose,longdoc` (2K and 16K at ctx 32768, 64K at ctx 98304); **Coder** (reference) workload `server`
targets `2000:200 16000:100 64000:50`, `--ctx 70000`. Why this size: the question is the shape of the prefill curve at 2K, 16K, 64K; decode tokens
are few because decode is not touched (it is checked, not targeted). Expectation (B against A; source: Result 1 microbench per token and layer
applied to the per-token costs of the previous round, Ornith 10 attention layers, Bolt 8, attention at 64K taken from the 16K and 32K points
doubled):

| Metric | Ornith | Bolt | Coder |
|---|---|---|---|
| prefill@2000 | +2 to +8 % | +2 to +8 % | -0.5 to +0.5 % |
| prefill@16000 | +30 to +55 % | +25 to +45 % | -0.5 to +0.5 % |
| prefill@64000 | +70 to +140 % | +70 to +120 % | -0.5 to +0.5 % |
| decode (named per length) | -1 to +1 % | -1 to +1 % | -0.5 to +0.5 % |

The Coder prefills with the QSA kernel, so its arms run the same code: the expectation is the noise. Output SHAs of A and B must be equal (bit
identical). Active tooling: none beyond `TONA_FATTN_PF_OLD=1` in A; no `TONA_DIAG` change; the quiet check is run before and after each job.
Decision: Ornith or Bolt slower at any length, or a SHA difference, takes the kernel out of the default; the Coder outside 0.5 % is investigated
(the shared path is only the launcher).

## Result 2: Ornith (job `pf-ab-ornith`, `work/ab/pf-ornith.{md,json}`; quiet check before 22:00: only the desktop's 1.1 GiB VRAM)

`ab.py` 4 rounds AB/BA, A = `TONA_FATTN_PF_OLD=1` (multi-tile kernel), B = wave kernel, same build `pf1` at `1479756`, workload `server` 2000:100,
16000:100, 64000:50 at `--ctx 70000`, `--drafter none` (MTP head from the model file), 1.4 min per round. Rounds agree to 0.1 % in prefill.

| Rate | A (tok/s, 4 rounds) | B (tok/s, 4 rounds) | B vs A [95 % CI] | Plan | Verdict |
|---|---|---|---|---|---|
| **prefill** @2000 | 2226.6 (2224.7-2227.4) | 2272.6 (2270.6-2273.4) | +2.07 % [+1.38, +2.76] | +2 to +8 % | as planned |
| **prefill** @16000 | 3018.5 (3017.5-3020.5) | 4051.4 (4050.7-4053.8) | **+34.22 % [+33.54, +34.90]** | +30 to +55 % | as planned |
| **prefill** @64000 | 1297.0 (1296.1-1297.9) | 2829.1 (2828.4-2831.9) | **+118.17 % [+117.45, +118.89]** | +70 to +140 % | as planned |
| **decode** after 2K / 16K | 104.9 / 95.8 | 104.6 / 95.6 | -0.28 % / -0.28 % [about -1.0, +0.5] | -1 to +1 % | as planned (undecided: inside noise) |
| **decode** after 64K | 35.5 (33.3-35.7) | 35.4 (35.3-35.5) | +1.14 % [-2.71, +4.99] | -1 to +1 % | outside only through one slow round of A (33.3; its other three 35.5-35.7); B's four rounds agree to 0.3 % |

Output SHAs of A and B equal at 2000, 16000 and 64000 prompt tokens (bit-identical), expert counters and tokens per pass identical. At 64K the
prefill rate is now 2829 tok/s; the attention share falls from about 81 % to about 57 % (new per-token total 0.353 ms against 0.771 ms; the
rest of the pass, about 0.15 ms per token, is the MoE and the DeltaNet layers).

## Result 3: Coder, the reference (job `pf-ab-coder`, `work/ab/pf-coder.{md,json}`)

Same arms and build, workload `server` 2000:200, 16000:100, 64000:50 at `--ctx 70000`, MTP head and expert profile as the default, 4 rounds
(1.7 min each). The Coder prefills with the QSA kernel, so A and B run the same code in every pass: this is the guard that the launcher change
costs nothing. **Output SHAs equal in A and B at all three lengths** (stable in every round).

| Rate | A median (tok/s) | B median (tok/s) | B vs A [95 % CI] | Rounds 2 to 4, B vs A |
|---|---|---|---|---|
| **prefill** @2000 | 1468.9 | 1469.1 | -0.54 % [-2.11, +1.04] | +0.05, +0.01, +0.14 % |
| **prefill** @16000 | 1909.6 | 1905.7 | -0.43 % [-1.44, +0.59] | -0.32, -0.04, 0.00 % |
| **prefill** @64000 | 1956.3 | 1956.3 | -0.35 % [-1.44, +0.74] | -0.01, +0.05, -0.02 % |
| **decode** after 2K | 73.8 | 73.5 | -0.70 % [-1.90, +0.50] | -0.95, 0.00, -0.09 % |
| **decode** after 16K | 70.2 | 70.1 | -0.23 % [-1.10, +0.64] | +0.04, -0.16, +0.10 % |
| **decode** after 64K | 76.1 | 76.1 | -0.10 % [-0.90, +0.69] | 0.00, +0.20, -0.03 % |

Two rates are formally outside the plan's +-0.5 % (prefill@2000 -0.54 %, decode@2000 -0.70 %): both are round 1 alone (-2.34 % and -1.77 %), the
first measured round after the warm-up start, in which the expert cache counters of arm A also differ from all later rounds (A hits/misses at
2000 tokens 71203/9497 against 69670/11030 in every other start of both arms): a cold page-cache effect of the first round, not of the arms
(rounds 2 to 4 agree to 0.2 %, except decode@2000 round 2 -0.95 %). No Coder code path differs between the arms; the Coder is unchanged.

## Result 4: Bolt (job `pf-ab-bolt`, `work/ab/pf-bolt.{md,json}`; first attempt stopped at the argument check (missing `--expect` per workload) and the prose corpus was rebuilt with `scripts/make_prose_corpus.py`, sha256 78f05cb0...)

`ab.py` workloads `prose` (2K and 16K prompts, 400 decode tokens, ctx 32768) and `longdoc` (65,536 tokens, 200 decode tokens, ctx 98304),
`--no-expert-profile --drafter none`, 4 rounds AB/BA (3.4 min each), A = `TONA_FATTN_PF_OLD=1`. Quiet check before 22:14 (only the desktop's 1.79 GiB
VRAM) and after (22:32, same). Each server: 134 expert slots per layer, 13.3 GB VRAM.

| Rate | A median (tok/s) | B median (tok/s) | B vs A [95 % CI] | Plan | Verdict |
|---|---|---|---|---|---|
| **prefill** @2000 (prose) | 1314.7 | 1351.6 | +2.83 % [+2.08, +3.58] | +2 to +8 % | as planned |
| **prefill** @16000 (prose) | 1457.4 | 1838.8 | **+26.15 % [+25.44, +26.86]** | +25 to +45 % | as planned (low end) |
| **prefill** @65536 (longdoc) | 884.2 | 1726.5 | **+95.27 % [+94.58, +95.96]** | +70 to +120 % | as planned |
| **decode** after 2K / 16K / 64K | 42.1 / 42.7 / 30.8 | 42.1 / 42.7 / 30.7 | -0.12 / -0.02 / -0.26 % | -1 to +1 % | as planned (inside noise) |

Output SHAs (normal, not deterministic mode): 16000 and 65536 equal in every round of both arms. The 2000-prose SHA of A is the same in all four rounds;
B gave it in rounds 2 to 4 and a different one in round 1 (the first measured start of B, after the discarded warm-up of A), so ab.py marks B
"not stable" there. The kernel is bit-identical by the comparison test, so this is the run-to-run difference of the non-deterministic mode (a wide
pass rounding differently, which `docs/backends/README.md` allows without `TONA_DETERMINISTIC=1`); the deterministic check follows below.

Deterministic check (job `pf-ab-bolt-det`, `work/ab/pf-bolt-det.md`, `TONA_DETERMINISTIC=1`, workload `prose`, 2 rounds, fixed; quiet check after 22:35, only
the desktop): **SHAs of A and B equal and stable at 2000 and 16000 tokens** (and equal to A's SHAs of the run above); prefill +3.03 % at 2K and
+26.24 % at 16K, decode +0.02 / -0.16 %, expert counters identical. So the 2000-prose difference of round 1 above was the non-deterministic mode, not the kernel.

## Summary, recommendation, open

- **The wave kernel is bit-identical to the old tile kernels, so no number-changing variant and no quality run (KL, top-1, tasks) is needed**: the
  rule "bit-exact or measured" is met by the byte comparison (60 cases) and by equal output SHAs at 2K/16K/64K on Ornith, 2K/16K/64K on Bolt and
  on the Coder. Recommendation: **on by default** (done: `FattnPrefillKernel::automatic` is the wave kernel); `TONA_FATTN_PF_OLD=1` restores the
  previous multi-tile kernel, `TONA_FATTN_PF_MT1=1` the single tile.
- Prefill (server, `ab.py`): Ornith 2K +2.1 %, 16K +34.2 %, **64K +118 % (1297 to 2829 tok/s)**; Bolt 2K +2.8 %, 16K +26.2 %, **64K +95 % (884 to 1727
  tok/s)**; decode unchanged (all |delta| below 0.3 % except one slow round of A at Ornith 64K); Coder unchanged (same code path, SHAs equal).
- Kernel: 40 to 46 % of the 93 TFLOPS WMMA peak (from 7 to 12 %), 3.9x (hd 256) and 6.9x (hd 128) at 16K.
- Open: (1) hd 256 uses 256 VGPRs with 11 spilled and occupancy 5: Q in LDS or a two-pass split of the head dimension could remove the spills; (2)
  double-buffered K/V staging (the staging is still synchronous, 2 barriers per chunk) and a coalesced K/V global load (one cell per 16 threads
  today); (3) the softmax keeps the old 8-partial summation order for bit-identity: a different order (one chain over the full head dimension) would save
  the 8 VALU adds per score and maybe 10 % more, but changes numbers (quality check, maintainer); not built; (4) the remaining Ornith 64K prefill is now
  about 57 % attention and 43 % MoE and DeltaNet layers, so the next prefill lever is elsewhere; (5) the Coder's QSA prefill kernel was not touched.
- Gates: `just check` (hooks, openmp, clang-tidy 5 files) green; `just test-cpu` 69 unit tests pass; `just test-py` 678 passed and 2 failed in
  `tools/tests/test_gpu_queue.py` (they start queue callers themselves and fail inside a queue job, as in the previous round; `gpu_queue.py` is not
  touched); `test_gpu_fattn`, `test_gpu_mamba`, `test_gpu_moe_latent`, `test_gpu_bolt_quant` pass; Coder goldens (`golden.sh compare` decode, cpu plan,
  prefill with the dflash drafter, `golden-server.sh compare`) exit 0.

# Round 2: registers, staging, softmax bookkeeping (2026-11-19, #175; bit-exact only)

Scope: the wave kernel stays byte-equal to the tile kernels (`test_gpu_fattn --prefill-cmp`, the summation order is not touched). The number-changing
lever (one chain over the full head dimension instead of the 8 partial sums) is out of scope and not measured.

## Test conditions (round 2)

`python3 tools/hw_report.py` at 22:54: RX 9060 XT gfx1200, 15.9 GiB VRAM, 1.84 GiB held by the desktop, 0 % busy; Ryzen 9 9950X, 54.0 GiB RAM
available, load average 4.3 / 8.2 / 6.1 (compiles of other jobs), other programs 3.2 % of the threads, swap 2.09 GiB in use, no `TONA_*` set, commit
`5034f45` plus this branch. Quiet check 22:54: only warning "1.84 GiB of 15.9 GiB VRAM held by others" (the desktop). Each job below repeats the quiet check
before and after; deviations are written at the job.

## Prior art (searched 2026-11-19, delta to round 1)

Round 1's search still holds (llama.cpp #18481 merged, #28102 merged, #26419 open, GPUOpen WMMA guide, CK has only the gfx12 WMMA header). New search
today (web, topics: llama.cpp RDNA4 flash attention double buffer, CK FMHA gfx12): nothing new; llama.cpp's RDNA4 MMA path reports `ldmatrix_trans` as its
main extra cost versus the older WMMA path (#18481 discussion), which matches our transposed V^T staging cost. No PR with an async or double-buffered K/V
copy for gfx12 found (RDNA4 has no async global-to-LDS copy; the prefetch is a register-staged load issued before the compute of the previous chunk).

## Plan 3: kernel microbench of round 2 (written before the runs)

Jobs: `test_gpu_fattn --prefill-bench` (exclusive; Ornith 16/2 hd 256 and Bolt 32/2 hd 128 at 2K, 16K, 32K, wave and multi-tile, median of 5), once for the
round-1 binary (baseline, build of `5034f45`) and once per step below; `--prefill-cmp` (gpu job) for every step before its bench. Registers and spills
from `-Rpass-analysis=kernel-resource-usage` (deterministic, first). Expectation (baseline from Result 1: hd 256 3.24 us per token at 16K, 41.5 TFLOPS;
hd 128 3.14, 42.8 TFLOPS; tolerance +-3 % run to run): (a) skipping the accumulator rescale when no lane's maximum moved (`al == 1`, `x * 1.0f == x`
exactly): 128 (hd 256) / 64 (hd 128) VALU multiplies per 16 cells, 0 to 10 % faster; (b) removing the hd 256 spills (11 VGPRs): 0 to 8 % on hd 256; (c)
prefetching the next chunk into registers and the coalesced global load: 0 to 10 %. Decision per step: kept if bytes equal and not slower than the baseline
by more than the tolerance; each step's gain is reported separately. Active tooling: none.

## Result 5: microbench of round 2 (jobs `pf2-bench-*`, `test_gpu_fattn --prefill-bench`, median of 5, per token and layer at 16K; quiet check before each: only the desktop's VRAM)

| Step (byte-equal in `--prefill-cmp`, 60 cases, every row) | hd 256, 16 q heads | hd 128, 32 q heads |
|---|---|---|
| baseline (`5034f45`, round 1) | 3.24 us, 41.4 TFLOPS (44.5 %) | 3.14, 42.7 (45.9 %) |
| (a) accumulator rescale skipped when no row maximum moved (`f2bad8b`) | 3.00, 44.7 (48.1 %) | 2.85, 47.0 (50.6 %) |
| (b) permlanex16 instead of ds_bpermute for the half-wave exchange (`c263977`) | **2.93, 45.8 (49.3 %)** | **2.81, 47.8 (51.5 %)** |

The numbers at 32K agree (5.87 / 5.61 us per token); at 2K the kernel is launch-bound and unchanged (0.86 / 0.59 us). Total round 2 so far:
hd 256 **-9.6 %**, hd 128 **-10.5 %** kernel time. Registers (`-Rpass-analysis=kernel-resource-usage`): hd 128 182 VGPRs, no spill, occupancy 8 (as before);
hd 256 256 VGPRs, **3 spilled (16 bytes scratch)** after (a), 11 before: the spills of round 1 were not the cost (see below).

Where the rest is (ablations on the step-(a) or (b) build, not committed; timings only, the output is wrong in these variants):

| Ablation | hd 256 | hd 128 | Reading |
|---|---|---|---|
| no global load, convert or LDS write after the first chunks (barriers kept) | 2.73 (-9 %) | 2.63 (-8 %) | upper bound of everything the staging could gain |
| every lane reads the same LDS address for K and V^T (no bank traffic) | 2.98 (-1.6 %) | 2.68 (-7 %) | LDS read bandwidth is not the limit for hd 256, a little for hd 128 |
| softmax and rescale removed (scores go straight to P) | 2.70 (-8 %) | 2.43 (-14 %) | upper bound of the VALU work of the softmax; even then 54 / 60 % of the peak |

Tried and not kept (all byte-equal, timings per token at 16K, hd 256 / hd 128):

- **Double-buffered K/V staging** (two LDS buffers, global loads of chunk i + 2 issued before the compute of chunk i, one barrier per chunk, chunk 16 cells at hd 256
  and 32 at hd 128 so that two buffers fit 64 KB): 3.03 / 2.89 against 3.00 / 2.85 without, i.e. **no gain** (the ablation above bounds the whole staging at 8 to
  9 %, and the global-load latency is already covered by the 4 waves per SIMD). A first version spilled 130 VGPRs at hd 256: a wave-uniform `continue` for the
  steps a wave does not see, inside the chunk loop, made the compiler copy all 128 accumulator registers every iteration; two loops (compute while the wave has rows
  to see, then only stage and wait) removed the copies (0 spills in the loop). Not kept because it does not pay.
- **Coalesced K/V global loads** (4 consecutive threads per cell instead of one cell per thread): 3.07 / 2.98, slightly worse (more bank conflicts in the transposed
  V^T writes: with a row stride of a multiple of 8 halves, rows 8 apart fall on one bank, whatever the padding). The 8-byte loads per cell are not the cost.
- **FRAGS_AHEAD pipeline** (three LDS fragments in flight per WMMA chain, scheduling barrier): hd 128 165 VGPRs instead of 181, no speed change; not kept.
- **32 waves per block at hd 128** (staging shared by twice the rows): 3.04, worse (fewer blocks, 8 waves per SIMD).
- hd 256 head split or Q in the LDS: not built. The spill is 3 VGPRs, the ablations bound the VALU and LDS shares at 8 and 2 %, and Q in the LDS would add
  K-sized LDS traffic per WMMA. Nothing to remove.

Number-changing lever, not measured (as ordered): one chain over the full head dimension instead of the 8 partial sums saves the 56 VALU adds per 16 cells.

## Plan 4: server speed of round 2 with `ab.py` (written before the runs; three `exclusive` jobs)

A = build `pf2base` (source tree of `5034f45`, the round-1 wave kernel), B = build `pf2full` (this branch at `684af98`: rescale skip and permlanex16). Both
arms: default environment, no `TONA_*` set by hand; the only difference is the kernel code. Same sizes as round 1 (Plan 2): **Ornith** (`--model
work/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf`) workload `server` 2000:100 16000:100 64000:50, `--ctx 70000`, `--drafter none` (MTP head from the
file); **Bolt** (`--model work/models/nemotron-bolt-74b-a9b-24gb/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`) workloads `prose,longdoc`, `--no-expert-profile
--drafter none`. Rounds: 4 AB/BA at most, `ab.py` stops when every rate is decided (margin 0.5 %), one discarded warm-up start. Why this size: the question is
the prefill curve at 2K, 16K, 64K for the two models that use the kernel; decode tokens are few because decode is not touched (checked, not targeted).
Expectation (B against A; source: Result 5 kernel time per token and layer, -9.6 % hd 256 and -10.5 % hd 128 at 16K, applied to the attention share of the
previous round: Ornith 10 attention layers, 64K prefill 0.353 ms per token of which about 57 % attention; Bolt 8 layers): Ornith prefill@2000 0 to +2 %,
@16000 +2 to +6 %, @64000 +4 to +8 %; Bolt prefill@2000 0 to +2 %, @16000 +2 to +6 %, @65536 +4 to +8 %; decode -1 to +1 % for both. Output SHAs of A and
B must be equal (the kernel is byte-equal). Coder (reference): its path is not touched, so no A/B run; the goldens (`golden.sh compare`, `golden-server.sh
compare`) must pass unchanged. Active tooling: none; quiet check before and after each job. Decision: any rate slower than A, or a SHA difference, takes
the change out.

## Result 6: Ornith (job `pf2-ab-ornith`, `work/ab/pf2-ornith.{md,json}`; quiet check before 23:46 and after 23:50: only the desktop's 1.83 GiB VRAM)

`ab.py` 4 rounds AB/BA (64 s each), A = `pf2base` (round-1 kernel), B = `pf2full` (rescale skip and permlanex16), workload `server` 2000:100, 16000:100, 64000:50 at
`--ctx 70000`, `--drafter none` (MTP head from the file), default environment in both arms.

| Rate | A median (tok/s) | B median (tok/s) | B vs A [95 % CI] | Plan | Verdict |
|---|---|---|---|---|---|
| **prefill** @2000 | 2267.9 (2250.7-2273.2) | 2272.9 (2268.5-2275.3) | +0.33 % [-0.45, +1.12] | 0 to +2 % | as planned (undecided: inside noise) |
| **prefill** @16000 | 4044.6 (4044.0-4049.5) | 4101.4 (4098.1-4105.5) | **+1.38 % [+0.70, +2.07]** | +2 to +6 % | **below the plan** (round to round 1.37 to 1.44 %) |
| **prefill** @64000 | 2828.7 (2828.7-2829.3) | 2964.7 (2963.3-2966.4) | **+4.80 % [+4.12, +5.49]** | +4 to +8 % | as planned |
| **decode** after 2K / 16K / 64K | 104.8 / 95.3 / 35.3 | 105.0 / 95.2 / 35.4 | +0.15 / -0.10 / +0.16 % | -1 to +1 % | as planned (inside noise) |

Output SHAs of A and B equal at all three lengths (bit-identical), expert counters and tokens per pass identical. The 16K plan was too high: it assumed a larger attention share of the 16K pass than there is. The measurement implies a share of about 14 % (1.38 / 9.6; derived, not profiled). The Ornith 64K prefill is now 2965 tok/s (2829 before).

## Result 7: Bolt (job `pf2-ab-bolt`, `work/ab/pf2-bolt.{md,json}`; quiet check before 23:51 and after 00:04: only the desktop's 1.83 GiB VRAM; a first attempt stopped at the argument check, missing `:prose` and `:longdoc` suffixes in `--expect`)

`ab.py` workloads `prose` (2K and 16K prompts, 400 decode tokens, ctx 32768) and `longdoc` (65,536 tokens, 200 decode tokens, ctx 98304), `--no-expert-profile --drafter none`,
4 rounds AB/BA (2.9 min each), A = `pf2base`, B = `pf2full`.

| Rate | A median (tok/s) | B median (tok/s) | B vs A [95 % CI] | Plan | Verdict |
|---|---|---|---|---|---|
| **prefill** @2000 (prose) | 1352.3 | 1354.5 | +0.11 % [-0.71, +0.93] | 0 to +2 % | as planned (inside noise) |
| **prefill** @16000 (prose) | 1835.6 | 1845.9 | +0.53 % [-0.18, +1.24] | +2 to +6 % | **below the plan** (rounds +0.32 to +0.69 %, undecided) |
| **prefill** @65536 (longdoc) | 1727.4 | 1757.2 | **+1.77 % [+1.07, +2.47]** | +4 to +8 % | **below the plan** (rounds 1.68 to 1.95 %) |
| **decode** after 2K / 16K / 64K | 42.1 / 42.6 / 30.7 | 42.1 / 42.6 / 30.7 | -0.04 / +0.02 / +0.06 % | -1 to +1 % | as planned |

Output SHAs equal at all three lengths and in every round (bit-identical), expert counters identical. The plan assumed a larger attention share of the Bolt pass than
there is: the measurement implies about 17 % of the 64K pass (1.77 / 10.5; derived, not profiled).

## Summary of round 2 (so far; gates below)

- Two byte-equal changes in `fattn_wave.h`: the accumulator rescale is skipped when no row maximum moved, and the half-wave exchange uses permlanex16. Kernel time per token and
  layer at 16K: hd 256 3.24 to 2.93 us (-9.6 %, 44.5 to 49.3 % of the WMMA peak), hd 128 3.14 to 2.81 us (-10.5 %, 45.9 to 51.5 %).
- Server prefill (`ab.py`, 4 rounds, SHAs equal): Ornith +0.3 % at 2K, +1.4 % at 16K, **+4.8 % at 64K (2829 to 2965 tok/s)**; Bolt +0.1 %, +0.5 %, **+1.8 % at 64K (1727 to 1757 tok/s)**;
  decode unchanged. The gain follows the attention share of the pass, which is why the server numbers are smaller than the kernel's: attention is now about 52 % of the
  Ornith 64K pass (57 % before, derived) and about 17 % of the Bolt 64K pass (derived).
- The staging (double buffer, coalesced loads), the hd 256 spills (3 VGPRs left, not the cost) and 32-wave blocks gave nothing; the ablations bound what is left in the
  bit-exact design: staging 8 to 9 %, LDS reads 2 % (hd 256) to 7 % (hd 128), the whole softmax 8 % (hd 256) to 14 % (hd 128) of the kernel time.
- Open: (1) the 40 % between 54 to 60 % of the peak (everything removed) and the peak is the WMMA issue itself plus the dependency chains; a second row tile per wave at hd 128
  (halves the LDS reads per WMMA, 192 VGPRs of 256) is the one structural bit-exact idea left, bounded at about 7 % of the hd 128 kernel; (2) a pre-converted f16 K/V in
  VRAM would remove the conversion VALU (part of the 8 to 9 %) at twice the global bytes and a scratch buffer: not taken, searched 2026-11-19 (web: llama.cpp RDNA4 FA PRs,
  CK FMHA gfx12), nothing found that does it; (3) number-changing, not measured: one chain over the full head dimension (saves the 56 adds per 16 cells, at most the
  VALU share above).

Gates of round 2 (builds `pf2full` at `4aaae17`'s kernel, own build directories): `just check` (hooks, openmp, clang-tidy 1 file) green; `just test-cpu` 69 of 69 pass; `just test-py` 678
passed and 2 failed in `tools/tests/test_gpu_queue.py` (they start queue callers inside a queue job, as in round 1; `gpu_queue.py` is not touched); `test_gpu_fattn` (incl.
`--prefill-cmp`, all bits equal), `test_gpu_mamba`, `test_gpu_moe_latent`, `test_gpu_bolt_quant` pass; Coder goldens identical: decode (csv, rust, explain), cpu plan, prefill with
the dflash drafter (long), server (`golden-server.sh`). The Coder path is not touched, so no Coder A/B run.
