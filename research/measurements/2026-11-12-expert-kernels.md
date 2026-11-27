# Expert kernels at wide passes (#128): three bit-exact variants measured, none faster

RX 9060 XT (gfx1200), Zen 5, DDR5, Coder IQ1_M shard 1 of 2026-10-19. Branch `128-expert-kernels` from main 126e0c7.
Result in one line: **no kept code change**; three candidates for lever 1 and 2 of #126 were built, all bit-exact (logits hash W1..8
equal to main), none faster. Levers 3, 4, 5 were not built (reasons and numbers below). The barrier profile at W = 1, 4, 8
and the probe variants are new data for the follow-up.

## Plan (written before the first run)

**Question.** Which cheaper form of the integer expert kernels (moe.gate_up: IQ3_XXS 5.5 GB, IQ2_S 5.4 GB, IQ3_S 3.6 GB,
IQ4_XS 0.45 GB over the layers; moe.down: IQ4_NL 9.2 GB, Q2_0 1.1 GB; `tona-gguf`) lowers the verify pass at W = 2..8
without changing one bit?

**Scope.** (1) Probe `tona-wide-probe --exps --types ... --runs 30` (large matrices, as k_mega: x in LDS, 32 blocks x
16 waves, L = 8) for the shape of the limit; variants compiled as probe-only macros. (2) Engine: `kernel-regs.py`
(VGPR and spill counts of k_mega<1,4,8>), batch bench `TONA_BENCH_WIDTHS=1..8 TONA_BENCH_REPEAT=2 TONA_BENCH_HASH=1`
(base = main build, same session), per candidate. (3) `TONA_BAR_PROF` at W = 1, 4, 8 (barrier latency and imbalance per
step kind). Expected duration per engine run about 6 min of GPU after the load.

**Expectation (with tolerance).** From the #126 profile: gate/up types at nb = 4 limited by dot and unpacking (no-dot probe
+20..33 %): a cheaper dot or sign form would take 0.3 to 1.0 ms off the W = 4 pass (moe.gate_up 6.6 ms); a result inside
+-0.15 ms of the base curve (round-to-round spread of the batch bench, see the two rounds below) counts as no effect.

**Active tooling.** `TONA_BENCH_HASH=1`, `TONA_MEGA_PROF` not set in the curve runs (the profile perturbs), `TONA_BAR_PROF`
only in the barrier runs. No diagnostics otherwise. The other helper (#127) shares the GPU through `scripts/gpu_queue.py`; every
run held the lock alone. Quiet check before and after each run in the logs (`work/v128/*.log`): the same warning every time (1.4 GiB
VRAM held by others; some runs also a high load average from the other helper's CPU work, listed per log).

## Test conditions

```
### Test conditions

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.36 GiB held by others before the run, 1 % busy before the run
- RAM: 60.5 GiB, 55.2 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.72, 1.64, 1.73
- CPU use by other programs over 0.5 s: 17.0 % of all threads, swap in use 1.77 GiB
- sandbox: no
- Commit: `126e0c7` (clean tree)
```

Quiet check at the end of the work (2026-11-12 19:59): warning, 1.36 GiB of 15.9 GiB VRAM held by others; 16.8 % of all CPU threads
used by other programs.

## Results

### Base curve (main build, 2 rounds interleaved, `work/v128/base-curve.log`)

Verify pass ms W1..W8 = 24.01 / 29.36 / 33.12 / 36.73 / 42.86 / 47.53 / 51.25 / 54.42 (both rounds within 0.1 ms except
W5 0.09 and W1 0.5). Higher than the 23.74 ... 52.61 of #126 by 0.3 to 1.8 ms: the machine state of this session (the other
helper's CPU work, higher load average); all variants below are compared with this curve of the same session, never with the #126 numbers.
Logits hash per width (identical across all builds below): W1..5 9754b76b2c08ed53, W6 and W8 5df41736bcdba681, W7 8105935297755027.

### Variant A: two rows per lane share the x reads (bit-exact by construction, `units_loop_q8_x2`)

Idea: the NOX probe (x words from registers, no LDS read of x) gains +5 to +25 % bandwidth at nb = 4 in IQ3_XXS, IQ3_S,
IQ4_NL, Q2_0 (probe-NOX: IQ3_S 116 -> 145 GB/s, IQ4_NL 117..143 -> 150, IQ3_XXS 108 -> 135, IQ2_S 94 -> 97). A wave took two
units (rows r and r + 4) of one expert, read each x sub-block once for both rows, each row summed in the old order and reduced the same way.
Hash equal at W1..8. Curve (ms) W1..W8 = 24.42 / 30.42 / 34.35 / 38.09 / 43.80 / 48.34 / 51.84 / 55.02: **+0.4 / +1.1 /
+1.2 / +1.4 / +0.9 / +0.8 / +0.6 / +0.6 ms against the base**. kernel-regs: k_mega<4> VGPR spills 54 -> 90, k_mega<8> 109 -> 237, k_mega<1> 62 -> 62.
Slower at every width, W = 1 included (one vector: x reads are only 3 LDS reads per sub-block), so the cost is the pair structure
(8 live words more, ten one-sub iterations instead of five two-sub rounds, half the units), not the saved reads. Dropped.

### Variant B: raw-load pipeline for IQ2_S, IQ3_S and IQ3_XXS, one sub-block ahead (G = 1 instead of #126's G = 2)

The #126 attempt (two sub-blocks ahead, IQ2_S and IQ3_S) lost 0.40 to 0.48 ms. Here `Raw<22>`, `Raw<21>` and a new `Raw<18>` with one sub-block
in flight ahead (cur and nxt of 1 instead of 2 each), NB <= 4 as before. Hash equal. kernel-regs: VGPR spills unchanged
(k_mega<1> 62, <4> 54, <8> 109), SGPR spills <4> 121 -> 129. Curve W1..W8 = 24.16 / 29.34 / 33.17 / 36.71 / 42.82 / 47.50 / 51.17 / 54.75:
**+0.15 / -0.02 / +0.05 / -0.02 / -0.04 / -0.03 / -0.08 / +0.33 ms, all inside the spread: no effect.** The register cost of the
old attempt is gone, the gain is not there either. Dropped.

### Variant C: scalar prefetch (`s_prefetch_data`, 31 lines) of the next unit's rows into L2, probe only

The instruction exists on gfx1200 (the compiler emits it for `__builtin_amdgcn_s_prefetch_data`) and costs no VGPR and no load
counter. In the probe (`--exps`, types 18, 20, 21, 22, 23, two runs each against two base runs): **no change** (base 108 / 133..137 / 108..112 / 97..99 / 172..175 GB/s
at nb = 4, L = 8, IQ3_XXS / IQ4_NL / IQ3_S / IQ2_S / IQ4_XS, prefetch 108..109 / 111..137 / 90..113 / 70..100 / 140..172, the lower values of
the first prefetch run came with a noisy machine and did not reproduce). Not built into the engine.

### What the probes say about the limit

- Same bandwidth at 4 MB (L2) as at 256 MB (`--cap 4`: IQ2_S 89..101 against 94..97 GB/s at nb = 4): the expert phase is not limited by the
  latency or bandwidth of DRAM, so more bytes in flight (variants B, C) cannot help; it is limited inside the CU.
- Instruction count of the row loop (ISA of `row_q8<T, 4>`, G = 2, per loop body of two sub-blocks): VALU 243 (IQ3_XXS), 266
  (IQ2_S), 307 (IQ3_S), 315 (IQ4_XS), 257 (IQ4_NL), of them 64 `v_dot4`; LDS 48 / 40 / 48 / 24 / 24 (24 are the x and meta reads). The rate in
  sub-blocks per second is nearly the same for all types (7.6 to 10.6 G sub-blocks/s) although the VALU count differs by 30 %.
  Estimated VALU utilisation 20 to 35 % at 8 waves per SIMD (ungeprueft: no hardware counters were read), so the loop is
  limited by dependency chains and waits (LDS table lookups feeding the sign step, then the dot chain), not by issue slots.
- At nb = 1 (16 dots per sub-block) IQ3_XXS runs at 150 GB/s and IQ2_S at 130, the lighter-decode IQ4_XS at 240: the decode of the grid types
  (table reads, 3 integer ops per word for the sign) already bounds the narrow pass; at nb = 4 the dots and x reads add about 20 to 25 %
  each (probe-NODOT, probe-NOX; NODOT showed no gain for IQ3_XXS and a loss for IQ2_S in this session, noisy: the
  base probe itself moves by 10 to 15 % between sessions, so only the repeated pairs above are used).

### Barrier profile of the base build (`TONA_BAR_PROF`, `work/v128/barprof{1,4,8}.bin`, 130 to 580 passes)

| W | pass ms | barrier latency per pass | imbalance (wait at barriers) per pass | moe.gate_up work / wait us | moe.down work / wait us |
|---|---|---|---|---|---|
| 1 | 23.4 | 0.96 ms | 2.11 ms | 2768 / 210 | 1956 / 127 |
| 4 | 35.5 | 1.00 ms | 3.40 ms | 6388 / 317 | 4969 / 222 |
| 8 | 51.2 | 0.52 ms | 6.00 ms | 10450 / 677 | 8296 / 359 |

Lever 3 (fuse moe.gate_up and moe.down): the tool reports a total barrier latency of 0.96 / 1.00 / 0.52 ms per pass over about 1,000 steps (about 1 us per step), so one barrier less per MoE layer
is worth at most about 0.05 ms per pass (48 layers); what could be won is the imbalance wait of the two steps (0.5 / 0.5 / 1.0 ms
at W = 1 / 4 / 8) by letting a block start the down rows of an expert as soon as that expert's activations are ready, and the intermediate round
trip. That needs unit-level flags between blocks (a dependency per 32-value activation block) in the persistent kernel: not built here, the upper bound is
about 0.3 to 0.5 ms at W = 4 (1 to 1.4 %). Imbalance at W = 8 by step (us): host.y 1072 (host side, #127), gr.down 823, moe.gate_up 677, moe.shared
551, gr.up 440, moe.down 359, dn.in 363, dn.conv 349, dn.recur 320. These are the numbers for levers 4 and 5.

## Not done (and why)

| Item | Status | Reason |
|---|---|---|
| Lever 4: gr.up split (640 units on 512 waves) | not built | the wait of gr.up is 199 us (W4) and 440 us (W8) per pass: at most 0.2 to 0.4 ms; a split changes the reduction tree unless done by rows (rows are already independent), to be tried next |
| Lever 5: dn.recur imbalance, wave quantization | not built | dn.recur wait 211 / 320 us; not bit-exact fixable by reordering (#126) |
| WMMA int8 (`v_wmma_i32_16x16x16_iu8`) for the expert matvec | analysed, not built | removes the 64 dots and the x reads per loop body (the two costs of 20 to 25 % each), keeps the decode work; the epilogue needs per-row scales in the C layout and the L = 8 reduction tree needs 8 float accumulators per (row, vector) = 64 per lane to stay bit-exact: needs a design of its own |
| Cheaper sign (table with m and m & 0x01010101 as one 16-byte entry, 2 ops instead of 3 per word) | not built | at most 8 of ~120 VALU per sub-block (7 %) at nb = 1, LDS tables +3 KB |

## Principle 8: items not taken

| Item | Gain seen | Reason not taken | Research (this date, 2025-11 .. 2026-11, `research/papers.md`, `research/papers-screened.tsv`) | Attempt to remove the cost |
|---|---|---|---|---|
| Two rows per lane sharing x (A) | none (+0.4..1.4 ms) | VGPR spills of k_mega<4> 54 -> 90, <8> 109 -> 237 and half the units | warp-count tuning per type in llama.cpp MMVQ (#19478, #24386, #23349/#23528; titles and abstracts only, ungeprueft in detail); nothing on row blocking with LDS x found in the book | restricted to NB = 4 (the spill growth is already at <4>): not tried separately, the W1 loss shows the cost is not spill-only |
| One-ahead raw pipeline IQ2_S/IQ3_S/IQ3_XXS (B) | none | no bandwidth gain: the phase is not limited by DRAM latency (`--cap 4` equals `--cap 256`) | nothing found on pipelined IQ GEMV (as in #126) | G = 2 (#126: +0.4 ms) and G = 1 (here: 0) both measured |
| `s_prefetch_data` (C) | none in the probe | same reason as B | no software-prefetch change found in the book for MMVQ (ungeprueft in the sources) | length 31 lines only; the instruction is accepted by gfx1200 and executed, the probe shows no effect |
| Fusing moe.gate_up and moe.down | not measured as a change | upper bound 0.3 to 0.5 ms at W = 4 | llama.cpp #18740, #19139, #16512 (gate and up as one matmul, fused GEMV) concern CUDA kernels with one grid launch per op; the persistent kernel has the barrier already at ~1 us | the barrier cost is measured (barprof), the gain would have to come from the imbalance wait |
| Speculative decoding meets quantization (2505.22179) | n/a | confirms that dequantization cost grows with the verify width (the effect in this engine) | read abstract (book row of 2025-05-28) | none needed |

Fresh search: `tools/papers/scan.py` coverage book is current to 2026-11-11 (research/papers.md lists no paper on integer grid-quant
(IQ2/IQ3) GEMV with row blocking or WMMA int8 for MoE verify); only the book (grep of research/papers.md) was searched; `scan.py` itself was not run again in this session (ungeprueft for the last days).

## Prior art

- llama.cpp #19478, #24386, #23349, #23528 (merged / open; read): RDNA4 and RDNA3 MMVQ warp-count tables; adopted only the idea of a per-type sweep (L per type was already swept in this engine, `TONA_MEGA_LANES_EXP`); no row blocking.
- llama.cpp #16512, #18740, #19139 (read): gate and up as one matmul, fused GEMV; the engine already runs gate and up as one group (moe.gate_up); fusing with down was bounded by the barrier measurement above.
- vLLM #62799 (read): RDNA4 skinny GEMM, 420 to 470 GB/s; the dense wide path (k_wide) is the counterpart in this engine; not used for the grid types.
- arXiv 2505.22179 (read, abstract): verify cost on quantized targets grows with the width because of dequantization; consistent with the probe result, no technique to adopt.
- Own prior work: research/measurements/2026-11-12-verify-pass.md (#126), 2026-10-25-hc-matvec.md (G = 2 loads).

Search: tools/papers/scan.py coverage; topics: expert GEMV, IQ-quant decode, verify pass; terms: MMVQ, row blocking, WMMA int8, gate up fused, prefetch; window: 2025-11..2026-11 (12 months)

# Part 2: expert matvec on matrix cores (int8 WMMA), probe first (#128, second round)

Question (coordinator decision after part 1): does `v_wmma_i32_16x16x16_iu8` remove the dot and x-read cost (20 to 25 % each in
the part 1 probes) of the grid and IQ4_NL expert kernels, bit-exactly, and does the pass get shorter? Branch `128-expert-kernels`, main 0a6b53f.

## Plan (before the runs)

**Scope.** (1) Prior art and search (below). (2) Probe in `tona-wide-probe` (new `--mma` mode, patch kept in `work/v128/patches/`): the
decode of row_q8 kept as is, the products by WMMA, outputs compared bit for bit with row_q8 (L = 8) on the real expert matrices (`--phase N`:
N experts of one matrix per phase, N = 10 and 40), nb = 2, 3, 4, 5, 8, IQ4_NL (moe.down), IQ3_XXS, IQ3_S, IQ2_S (moe.gate_up);
row_q8 and WMMA in the same process, same session. (3) Only if clear: engine, curve W1..8 with hash, barrier profile of the step.
**Expectation.** Dots (8 per sub-block and token) and x reads (3 LDS reads per token) go from nb-proportional to constant; at nb = 4 to 8
the instruction count of the loop falls by an estimated 11 % (nb = 4) to 36 % (nb = 8) for the grid types (computed from the ISA counts of
part 1: decode stays, the epilogue runs for 4 tokens per lane half), more for IQ4_NL whose decode is light; at nb <= 2 no gain. Tolerance: a probe
difference below 5 % counts as none. **Active tooling:** none besides the probe; engine runs as in part 1 (`TONA_BENCH_HASH=1`).

## Prior art and search (principle 8; window 2025-11 .. 2026-11)

`tools/papers/scan.py` run for the open day (arXiv topics moe-offload, speculative, linear-attention, kv-cache, quantization, megakernel, amd-rocm,
apple-silicon, long-context, memory-pressure, tool-calling-quality: 0 hits on 2026-11-11, "younger than 3 days, the API may not list it yet"; GitHub
llama.cpp, vLLM, SGLang, MLX, mlx-lm, ROCm repos with the topic terms RDNA4, WMMA, gfx12, gfx1200, MoE offload, n-cpu-moe: issues 0 hits, two
hits each in vLLM and SGLang for `n-cpu-moe`, titles not related). Ledger: arXiv covered to 2026-11-10, GitHub to 2026-11-11 (`scan.py coverage`). Then the book
(`research/papers.md`, grep wmma/mfma/int8/mmq/gfx12) and `gh search prs`:

| Item | Status | What it says for this change |
|---|---|---|
| GPUOpen, WMMA guide for RDNA4, part 2 (book 2025/26) | read (page) | Double-K for FP8/INT8: two WMMAs fused, 128-bit loads, bit-identical; layout of the fragments as used here (confirmed by the probe: 0 differing values) |
| vLLM #45559 (open, body read in the book) | open | SWMMAC skinny GEMM for N = 5..8 on GFX12 for low-concurrency decode: a matrix-core kernel for 5 to 8 tokens against dense weights (not experts); same lesson: the matrix path pays from about 4 tokens per weight |
| vLLM #57286 (open) | open | interleaved two-M4-group BF16 head for M = 8 on R9700 (8.06 -> 4.06 ms against two M4 calls): one weight read for 8 tokens |
| arXiv 2611.00486 Integer-Exact Microscaling (abstract) | read abstract | INT8 matrix units for 4-bit expert weights with INT8 activations (prefill); the same idea for the decode of our IQ types would need the int sums exact: they are (int32) |
| llama.cpp #14624 (merged 2025-08, "Enable Matrix cores for MMQ kernels", CDNA) and the book's #17495 (closed: RDNA3 WMMA in MMF), #23227 (MFMA MMQ for speculative-verify batches, CDNA) | merged / closed | MMQ on matrix cores exists for CDNA (MFMA); RDNA4 MMQ stays on dp4a-style paths in the versions in the book; no gfx12 int8 WMMA MMQ for mul_mat_id found |
| hipfire (book 2026-10), pat883 w4a8-fp8-wmma (HF) | open / unmeasured | gfx1200 WMMA kernels for prefill (FP8, MoE), no decode numbers |

Result of the search: no published int8 WMMA kernel for IQ-grid (codebook) formats on RDNA4 or for MoE decode with per-expert token groups;
the nearest are the dense skinny-GEMM kernels above. (Ungeprueft in the sources: only titles and abstracts of the items were read.)

## Design (bit-exact)

Matrix instruction: A = x (16 x 16: rows 0..3 tokens 0..3, rows 8..11 tokens 4..7), B = weights (16 x 16: column = row of the expert matrix `l % 16`, K = 8 values per
lane half), D lane `l`: matrix row `l % 16`, tokens `4 * (l / 16) + i`. The int32 sums lo (values 0..15) and hi (16..31) of a sub-block are exact, equal to the four `dot4` of row_q8
each. The float part per (row, token) keeps the sequence of `apply_q8`: `r = fma(s_hi, hi, s_lo * lo)`, `acc_j = fma(d_x, r, acc_j)` over the sub-blocks j, j + 8, ... of
virtual lane j (L = 8), and the butterfly of `reduce_lanes` becomes a local tree `((a0 + a4) + (a2 + a6)) + ((a1 + a5) + (a3 + a7))` (float addition is commutative, so every lane of the
butterfly gets the same value). 8 float accumulators per (row, token) are never alive at once: P of the 8 virtual lanes run side by side (P = 2 in the engine try, 4 and 8 in the probe); the 4 tokens
per lane half are 32 accumulators. A lane decodes only the half sub-block it needs (2 of 4 table lookups of the grid types, 2 of 4 words of IQ4_NL), so the decode work per row and sub-block is the same as in row_q8.

## Probe results (same process, row_q8 L = 8 against WMMA, `differing values` = 0 in every row below and in all runs)

IQ4_NL (moe.down, 9.2 of 10.3 GB), x in LDS, phase of 40 experts (25,600 rows, about the shape of W = 4..6), pipelined loads, P = 8 (`work/v128/mma6-ph40.log`):

| nb | row_q8 us | WMMA us | change |
|---|---:|---:|---:|
| 2 | 258.5 | 196.0 | -24 % |
| 3 | 217.0 | 167.3 | -23 % |
| 4 | 197.0 | 151.6 | -23 % |
| 5 | 245.7 | 151.6 | -38 % |
| 8 | 292.9 | 156.6 | -47 % |

Grid types, same run: IQ3_XXS +30 % (nb 2) / +38 % (4) / +14 % (5) / **-16 % (8)**, IQ3_S +44 % / +40 % / +16 % / -10 %, IQ2_S +31 % / +34 % / +12 % / -14 %: slower below
8 tokens, faster at 8. With the 10-expert phase (6,400 rows) the grid types are slower even at 8 (IQ3_XXS 91 against 80 us): one wave runs a whole 16-row group alone,
and for the formats with separate qs / sign / scale arrays (IQ2_S, IQ3_S, IQ3_XXS) each load instruction touches 16 rows (16 cache lines) against 4 rows in row_q8. Nothing of this was
fixed (decode kept as is, as asked). With x in device memory (as the down experts read it) the IQ4_NL gain is the same or larger (nb 4: -22 %, nb 8: -49 %; `mma7-gx1.log`).
At nb = 1 WMMA is 10 to 20 % slower than row_q8 (`mma1.log`).

## Engine: matrix path for moe.down (IQ4_NL), whole expert group when nb >= 2

Built with the same code the probe runs (`units_loop_mma`, 16-row units, `upe` and `units` recomputed per pass in `run_phase`, patch `work/v128/patches/mma-engine-and-probe.diff`).
Logits hash W1..8 equal to main in every variant (bit-exact). Verify pass ms W1..W8, same session as the base curve 24.01 / 29.36 / 33.12 / 36.73 / 42.86 / 47.53 / 51.25 / 54.42:

| Variant | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | k_mega<4> / <8> VGPR spills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all 8 virtual lanes at once, raw loads pipelined | 23.83 | 45.34 | 50.37 | 54.84 | 60.07 | 60.48 | 64.41 | 67.81 | 3427 / 3431 |
| P = 2, raw loads pipelined | 23.79 | 30.63 | 34.91 | 38.94 | 44.21 | 48.73 | 52.38 | 55.62 | 54 / 110 |
| P = 2, raw and x loads pipelined | 23.91 | 30.25 | 34.26 | 38.05 | 43.36 | 47.93 | 51.59 | 54.85 | 53 / 109 |

Slower than main by +0.9 / +0.9 / +1.3 / +0.5 / +0.4 / +0.3 / +0.4 ms (W2..W8) in the best variant. Barrier profile of the best variant against the base (work per pass, us): moe.down at W4 5772 against 4969
(+16 %), at W8 8410 against 8296 (+1 %); moe.gate_up (unchanged) 6418 against 6388 and 10411 against 10450: the machine state was the same, the down step is slower.

## Why the probe gain does not arrive in the pass

The probe computes every token for every expert (its cost is independent of nb from 2 to 8). In a verify pass most experts are used by one or two tokens: the megakernel passes a token mask per
expert and row_q8 skips the vectors without a bit. The work per GPU expert of moe.down is nearly flat in the width: step time per pass / (hits + GPU-direct experts per pass)
= 1956 / 467 = 4.19 us (W1), 4969 / 1126 = 4.41 us (W4), 8296 / 1672 = 4.96 us (W8), i.e. +5 % and +18 % while the number of tokens grows 4 and 8 times (expert counts from
the batch bench logs `work/v128/bar{1,4,8}.log`; per-expert token counts themselves were not logged: ungeprueft). The matrix path costs the same for 1 token as for 8, so it loses where
most experts have 1 to 2 tokens (nb = 1: -10..-20 % in the probe) and wins only for experts with 3 or more tokens (-23 % at 3, -38 % at 5, -47 % at 8). An expert-by-expert choice
(mask popcount >= 3 on the matrix path, others row_q8; two unit spaces per task) is the only form that can gain; its upper bound is the share of expert-slots with 3 or more tokens times 25..40 %:
at the measured mean of about 1.6 tokens per expert (W4) to 2.1 (W8) (slots = tokens x hits per token / experts per pass, ungeprueft) that is about 0.1 to 0.3 ms at W4 and 0.5 to 0.9 ms at W8 (1 to 2 % of the pass), against a second unit
space, a per-pass classification of the experts in the step and +code in k_mega.

## Principle 8 (part 2)

| Item | Gain seen | Reason not taken | Research (above) | Attempt to remove the cost |
|---|---|---|---|---|
| Matrix path for all experts of moe.down | probe -23 % (nb 2..4), -47 % (nb 8) | in the pass +0.3..1.3 ms: per-expert token counts are 1 to 2, the matrix cost is constant | no kernel for codebook formats on RDNA4 found | probe with x in device memory (no change), pipelining of raw and x loads (3 variants), P = 2 / 4 / 8 (registers: 8 at once spills 3.4 k) |
| Matrix path for gate_up (IQ3_XXS, IQ3_S, IQ2_S) | probe +14..44 % slower for nb <= 5, -10..-16 % at nb = 8 | slower where the passes live (W <= 6); 16 cache lines per load instruction for the split-array layouts | as above | P = 2, 4, 8 tried, phase of 10 and 40 experts; a layout change of the expert files (rows interleaved so that a lane group reads 4 B of 16 rows contiguously) would be needed: an IQ-format repack as in 2026-11-12-mac-moe-grouped.md (Mac type 1018), about 5 GiB for IQ3_XXS, not tried |
| Expert-level class split (>= 3 tokens on the matrix path) | upper bound 0.1..0.3 ms (W4), 0.5..0.9 ms (W8), ungeprueft | complexity (two unit spaces, classification step) for 1 to 2 % at the widest passes only | as above | not built; first a log of tokens per expert from the engine would size it |

## Recommendation

Stop the matrix-core direction for the experts here: bit-exact is possible (0 differing values at all widths) but only a token-count-dependent hybrid gains, and the bound is about 1 % at the usual widths. The kernel-level
levers left (the dot is the cheap part: the decode of the grid types is the cost at nb = 1) are a repack of the grid formats with a layout that the matrix path and row_q8 both read with fewer cache lines (needs a file
variant, a decision) and the host side (#127).

# Part 3: tokens per expert, and the hybrid (matrix path only for experts with enough tokens) (#128, third round)

## Plan (before the runs)

**Scope.** (1) Counter `TONA_TOKHIST=1` (new, diagnostic only: in `ExpertSelect::count_union`, histogram [pass width][tokens using the expert] over the union experts of every layer of
single-sequence passes; printed every 500 passes): production server (MTP drafter, adaptive width, `--ctx 32768`), `scripts/bench-set.py` (11 code and 3 text prompts, up to 400 tokens each), one run, no speed claim
(the run is a counter). (2) Expected hybrid gain from the histogram with the probe numbers, k = 2, 3, 4. (3) Only if at least 0.5 % of the pass at W4 or wider: build it (bit-exact), width curve with hash against a base of the same session.
**Expectation:** mean tokens per expert 1.3 to 2.1 (computed from the batch bench counts in part 2); the gain 0.1 to 0.9 ms. **Active tooling:** `TONA_TOKHIST=1` only; `TONA_EXPERT_PROFILE` as in production.
Conditions: quiet check warnings as before (1.4 GiB VRAM held by others; the other helper's CPU work; logs in `work/v128/`).

## Result 1: histogram (all layers, all single-sequence passes of the run; `work/v128/tokhist.txt`)

Experts of the union by number of tokens that use them (counts):

| W | experts | 1 token | 2 | 3 | 4 | 5 | 6 | 7 | 8 | mean tokens per expert |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 45,704 | 32,848 | 12,856 | | | | | | | 1.28 |
| 3 | 151,190 | 100,005 | 31,800 | 19,385 | | | | | | 1.47 |
| 4 | 369,904 | 230,223 | 77,867 | 36,093 | 25,721 | | | | | 1.61 |
| 5 | 68,461 | 40,436 | 14,400 | 6,696 | 3,969 | 2,960 | | | | 1.75 |
| 6 | 85,443 | 48,510 | 17,711 | 8,324 | 4,809 | 3,394 | 2,695 | | | 1.89 |
| 7 | 74,653 | 42,860 | 14,544 | 7,257 | 4,180 | 2,424 | 1,795 | 1,593 | | 1.94 |
| 8 | 502,185 | 281,887 | 103,765 | 49,153 | 26,959 | 16,271 | 10,509 | 7,529 | 6,112 | 1.97 |

(All union experts count, also those the CPU computes on a miss, about 5 to 9 %; the run contained the prefill passes of the short prompts of the set, so W8 is over-represented.)
56 to 62 % of the experts have one token, 35 to 40 % two or fewer than three; at W4 only 7 % have four.

## Result 2: expected hybrid gain (probe numbers)

Model: IQ4_NL moe.down only (39 of 48 layers), cost per expert from the probe at 40 experts (`mma6-ph40.log`: row_q8 6.46 / 5.43 / 4.93 / 6.14 / 7.32 us for 2 / 3 / 4 / 5 / 8 tokens, 5.14 at 1 from the 10-expert phase; WMMA 3.8 to 4.9 us),
moe.down step per pass 4.97 ms (W4) and 8.30 ms (W8) from the barrier profile, the others interpolated (ungeprueft):

| W | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|
| 4 | 0.39 ms (1.1 %) | 0.15 ms (0.4 %) | 0.06 ms (0.2 %) |
| 5 | 0.55 ms (1.3 %) | 0.26 ms (0.6 %) | 0.15 ms (0.3 %) |
| 6 | 0.70 ms (1.5 %) | 0.38 ms (0.8 %) | 0.25 ms (0.5 %) |
| 8 | 0.89 ms (1.6 %) | 0.49 ms (0.9 %) | 0.35 ms (0.6 %) |

By the rule of the ticket (at least 0.5 % at W4 or wider) the hybrid qualified, so it was built. Calibration warning that the part 2 engine run already gave: the probe's WMMA cost is about 25 % below what the engine measured
(5.1 us per expert in the engine against 3.8 to 4.9 in the probe), which moves the k = 2 gain to a loss; only k >= 3 stays positive (about 0.2 to 0.4 ms at W8).

## Result 3: the hybrid in the engine (bit-exact; patch `work/v128/patches/hybrid-and-tokhist.diff`)

Per moe.down step, thread 0 of each block sorts the union experts into two lists by `popcount(mask)` (>= k: matrix path, rest row_q8) in LDS, one `__syncthreads`; `resolve_unit` maps the unit of the k-th expert of a class to its expert through the list;
the flat unit sequence is first the row_q8 units (4 rows) of the few-token experts, then the matrix units (16 rows) of the others. Hash W1..8 equal to main. Same-session base curve (`base2-curve.log`): 23.84 / 29.39 / 33.08 / 36.64 / 42.75 / 47.44 / 51.08 / 54.42 ms.

| Variant | W2 | W3 | W4 | W5 | W8 | k_mega<4> / <8> VGPR spills |
|---|---:|---:|---:|---:|---:|---|
| k = 3 | 29.91 (+0.52) | 33.91 (+0.83) | 37.65 (+1.01) | 45.03 (+2.28) | 55.40 (+0.98) | 51 / 1366 (base 54 / 109) |
| k = 9 (no expert on the matrix path: lists, remapping and code only) | 30.30 (+0.91) | 33.97 (+0.89) | 37.61 (+0.97) | 45.41 (+2.66) | 55.88 (+1.46) | same code |

The machinery alone costs +0.9 to +1.0 ms (W2..4) and +2.7 ms (W5, k_mega<8> spills 109 -> 1366) and +1.5 ms (W8); the matrix path itself brings back only 0.4 (W2), 0.1 (W3), 0 (W4), 0.4 (W5), 0.5 ms (W8), which is the
calibrated estimate (about 0.2 to 0.5 ms at the widest passes, nothing at W4). Even with zero machinery cost the gain would be 1 % at W5 to W8 and 0 at W4.

## Conclusion and principle 8 (part 3)

| Item | Gain seen | Reason not taken | Research | Attempt to remove the cost |
|---|---|---|---|---|
| Hybrid matrix path, experts with k >= 3 tokens | 0.4 to 0.5 ms at W5 and W8 (best case, without the machinery) | the machinery (lists in LDS with a block barrier per down step, expert remap in `resolve_unit`, two loops in k_mega) costs +1 to +2.7 ms; k_mega<8> spills 1366 VGPRs | none newer than part 2 (vLLM #45559, #57286 are dense skinny GEMMs); no expert-level token-count dispatch found in llama.cpp (`mul_mat_id` groups tokens per expert on the CPU side, ids sorted before the kernel: book rows of #18958, #16512) | list build by thread 0 and one barrier measured against an empty variant (k = 9): the machinery, not the barrier alone, is the cost; a host-side class list (the host knows the masks at selection) would remove the barrier but not the code size and spills; not built: the remaining upside is 0.4 to 0.5 ms at W5 and wider (about 1 %), none at W4 |

Open (if someone wants the last percent): class lists produced by the host in the mailbox (no device sort, no barrier) together with the 16-row loop in its own kernel path with fewer live registers (`k_mega<8>`), after the grid-type repack, which is a file format decision.
The counter `TONA_TOKHIST` stays in the tree as a diagnostic (a static flag read once; no cost without it).
