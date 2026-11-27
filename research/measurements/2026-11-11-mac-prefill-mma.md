# Mac M1 Max: prefill, GEMM tile layout, DeltaNet scan and narrow matrices (prefill)

Three changes after step 1b, measured together:

- f7b6e6f: GEMM tile rows padded to 36 floats and the token tile stored [k][token] (bit-identical);
- 367e841: the DeltaNet scan takes the 8 state rows of a simdgroup through each token together instead of one row
  after the other (bit-identical, prefill and decode);
- a711625: matrices under 16 rows (the gated reads' inject, 4 rows) through a simdgroup per row instead of a 64-row
  GEMM tile (prefill numbers move by rounding, decode unchanged).

Raw rounds: [2026-11-11-mac-prefill-mma.jsonl](2026-11-11-mac-prefill-mma.jsonl).

## Plan (written before the run)

**Model.** `work/models/synth/synth-a-e256.gguf`, as in [2026-11-11-mac-prefill-gemm64.md](2026-11-11-mac-prefill-gemm64.md).

**Arms.** Release builds, kernels without fast math, both `TONA_PREFILL_CHUNK=2048` (the default):

| Arm | Build | Commit |
|---|---|---|
| A | `work/build/mtl-s1` (worktree) | 19f57ce, step 1b |
| B | `work/build/mtl` | a711625 |

**Runs.** `bench/mac/prefill_ab.py --arms mtl-s1:2048,mtl:2048 --procs 3 --prompt 2048 --rounds 1`: six processes
A B A B A B, 30 s pause, each one warmup and one counted round (prefill 2048, decode 16). ~6 min. No build and no
other GPU work during the run.

**Expectation.**

- A: 190.4 tok/s ±3 % (step 1b measurement, same commit).
- B: 219 ±4 %. Source: smoke runs of 367e841 216.4 and 216.3, and the profile of a711625 (gated reads −0.26 s of
  19.7 s GPU time, ~+1.3 %).
- Decode after the prompt: B 0 to +4 % over A (the scan change applies to decode too; smoke 13.9 against 13.6).

An arm outside its band is a finding.

**Where the time went (profiles, `TONA_METAL_PROFILE=1`, prefill of 2048 tokens, warmup + round; not speed
results).** 19f57ce: experts 7.08 s, DeltaNet projections 4.76 s, scan 3.19 s, gated reads 2.24 s, total 22.6 s.
a711625: scan 1.22 s, gated reads 1.98 s, total 19.7 s. Diagnostic builds of the GEMM (not committed): without the
weight decode the experts took 5.32 s, without the matrix products 2.37 s; the products bound the GEMM, not the
decode. Tried and dropped: two K buffers of 16 (no gain), 32×32 per simdgroup with 16 accumulators (six times
slower with 4 and with 8 simdgroups; cause not found).

**Active tooling.** No profiler. `TONA_SEQS=1`, `TONA_PREFILL_CHUNK=2048`, `LC_ALL=C`.

**Correctness before the run** (a711625): `tona-check --prefill` (chunks 64 and 96) on the mini model and on
`synth-a-l4-e16.gguf`: cut invariance byte-equal in every round; prefill against decode nll 3.3e-5 and 5.3e-5 (137
tokens), 4.16e-2 (404 tokens, the router tie of [2026-11-11-mac-prefill-attn.md](2026-11-11-mac-prefill-attn.md)'s
addendum); top-1 206/206 against `cpu/ref.cpp`; widths 0/423, `TONA_SEQS=2` 0/786; unit 45/45.

## Test conditions

Before the run (`tools/hw_report.py`, 18:54): Apple M1 Max, 10 cores, 32.0 GiB; macOS 27.0.1 (arm64); no `TONA_*`
variable in the shell; load average 3.07, 4.22, 4.29 (my builds before; the run waited until the quiet check no
longer warned about it); other programs 7.3 % of all CPU threads. Swap 15.37 GB; free + inactive memory 23.5 to 24.0
GB before each process; no swap-ins during the run.

Quiet check before (18:54:51): **ok**. After (18:59:42, right after the last process): "do not measure: the GPU is
98 % busy", the tail of the last process; 10 s later (18:59:53): warning, other programs 5.9 % of all CPU threads.
`pmset -g therm`: no warning before or after any process. No build or other GPU work ran during the run.

## Results

Prefill of 2048 tokens, tok/s, one counted round per process (warmup in brackets, not counted):

| Process | Arm | Prefill | Decode after it |
|---|---|---|---|
| 1 | A (19f57ce) | 190.46 (174.89) | 13.77 |
| 2 | B (a711625) | 219.29 (212.22) | 13.78 |
| 3 | A | 190.46 (184.62) | 13.73 |
| 4 | B | 219.59 (211.46) | 13.83 |
| 5 | A | 190.47 (185.11) | 13.84 |
| 6 | B | 219.51 (212.16) | 13.75 |

- A: 190.5 tok/s (190.46 to 190.47), inside the expectation (190.4 ±3 %), equal to the step-1b measurement.
- **B: 219.5 tok/s** (219.29 to 219.59), inside the expectation (219 ±4 %): **+15.2 % over A**.
- Decode: A 13.78 (13.73 to 13.84), B 13.79 (13.75 to 13.83); equal within the spread, at the lower edge of the
  expected 0 to +4 %. The scan is ~2 ms of a ~72 ms decode pass at this position, too little to show.

At 2048 tokens the prefill went 45.5 (decode passes) → 116.0 (step 3) → 190.4 (1b) → 219.5 tok/s. Target ≥ 250:
another ~12 % of GPU time. Left by the a711625 profile: routed experts 6.8 s and DeltaNet projections 4.5 s of
19.7 s, both on the GEMM at ~3 TFLOPS, bound by the matrix products.
