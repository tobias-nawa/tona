# 2026-11-11: expert mat-vec time for the Coder's expert types, B = 1 to 4, M1 Max (#108)

Step M0 of the Mac backend proposal ([../40-mac-backend-proposal.md](../40-mac-backend-proposal.md)). Question: are
the expert kernels of the Coder bandwidth-bound on the M1 Max GPU at 1 to 4 tokens per pass, or latency/ALU-bound? The
answer decides how far the external ~8 tok/s decode is a kernel limit. Speed numbers on this machine are provisional.

## Plan (written before the run)

**Scope.** Tool `bench/mac/expertk.mm` (host, Objective-C++) with `bench/mac/expertk.metal` (kernels). First-cut
kernels, not tuned: one simdgroup per output row, lanes stride over groups of 8 weights, byte loads, tables from
`engine/model/quant_tables.h` in constant memory, gate and up fused with SiLU, down separate; B tokens share each
decoded group. Shapes of the Coder: gate/up 640 × 2560, down 2560 × 640, 10 selected experts per layer. Weights are
synthetic blocks in the real formats (random bytes, block scale 0.002), a pool of 64 experts per matrix and type
(~0.75 GB in all), 10 distinct experts drawn per layer, so successive layers read different memory. Type mix per
layer from the header of the pinned Coder file (HTTP range read of 16 MB, `work/gguf-header/`): gate/up IQ2_S ×20,
IQ3_XXS ×17, IQ3_S ×10, IQ4_XS ×1; down IQ4_NL ×39, Q2_0 ×9.

| Block | What runs | Rounds | Duration |
|---|---|---|---|
| check | gate_up and down of 2 experts for 6 type pairs and B = 1 to 4 against `tona::dequantize_row` (double sums) | once per run | < 1 s |
| type | per kernel type and B: 48 layers × 10 experts in one command buffer; µs per layer | warm-up pass, then 10 command buffers per point | ~2 s |
| token | 48 layers with the Coder's types, gate_up + down per layer, one command buffer per token | 20 per B | < 1 s |

3 runs, 30 s pause.

**Expectation.**

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| check, relative error | < 1e-5 | < 1e-3 (abort above) | float against double sums |
| type, effective GB/s at B = 1 | 120 GB/s | 80 to 170 (25 to 50 % of 334) | untuned kernel |
| type, time at B = 4 against B = 1 | 1.5× | 1.1 to 2.5× | decoding shared, FMAs ×4 (estimate) |
| token, experts at B = 1 | 8 ms | 5.8 to 12 ms | 0.98 GB of experts at 80 to 170 GB/s |

Consequence to test: with the dense part at ~14 ms (unmeasured: 4.1 GB at ~300 GB/s), the resident pass would take
~20 to 26 ms, i.e. ≤ 50 passes/s even before any SSD read.

**Active tooling and conditions.** clang++ -O2 (Apple clang 21), Metal source compiled at run time, no validation,
no capture. AC power. Load recorded per run.

### Addendum to the plan (written after the first three runs, before the second set)

The first set (raw: [2026-11-11-mac-expertk-r1only.jsonl](2026-11-11-mac-expertk-r1only.jsonl), kernel with one row
per simdgroup) gave 12.3 ms per token at B = 1 and 2.8× that at B = 4, outside the expected 1.1 to 2.5×. Hypothesis:
every row reloads its inputs (B × 8 floats per group from cache), so time grows with B. Variant r4: each simdgroup
computes 4 consecutive rows and loads the inputs of a group once for all 4. The tool now runs r1 (same layout, the
inner sum written with `dot`) and r4 in one process; 3 runs, 30 s pause.

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| r1 | as the first set | ±3 % | same layout |
| r4, token at B = 1 | 8 ms | 5 to 11 ms | input loads per row / 4 (estimate) |
| r4, B = 4 against B = 1 | 1.6× | 1.1 to 2.5× | as the plan above |

## Results

Raw data: [2026-11-11-mac-expertk.jsonl](2026-11-11-mac-expertk.jsonl) (r1 and r4) and
[2026-11-11-mac-expertk-r1only.jsonl](2026-11-11-mac-expertk-r1only.jsonl) (first set). Tools:
`bench/mac/expertk.mm`, `bench/mac/expertk.metal`.

**Conditions (13:01 to 13:06).** AC power. Load average 2.2 to 2.8 (1 minute) on 10 cores. Every check passed:
relative error at most 2.4e-7 (288 checks over both sets).

**Token, experts of 48 layers** (ms, median of 20 command buffers; three runs; effective GB/s of the 0.98 GB of
expert weights in brackets):

| B | r1, first set | r1 | r4 | Expected r4 |
|---|---|---|---|---|
| 1 | 12.38 / 12.36 / 12.31 (79) | 12.90 / 12.92 / 12.90 (76) | **10.24 / 10.21 / 10.38 (96)** | 5 to 11 |
| 2 | 18.83 / 18.80 / 18.82 (52) | 18.81 / 18.81 / 18.80 (52) | 12.78 / 12.83 / 12.92 (77) | |
| 3 | 26.19 / 26.07 / 26.18 (38) | 26.27 / 26.28 / 26.27 (37) | 17.39 / 17.34 / 17.42 (57) | |
| 4 | 34.64 / 34.60 / 34.66 (28) | 34.51 / 34.54 / 34.55 (28) | 24.10 / 24.05 / 24.12 (41) | B4/B1 1.1 to 2.5× (got 2.35×) |

**Per kernel, one layer of 10 experts** (µs, median of the three run medians; spread between runs ≤ 2 %; GB/s):

| Kernel | r1 B = 1 | r1 B = 4 | r4 B = 1 | r4 B = 4 |
|---|---|---|---|---|
| gate_up IQ2_S | 219 (48) | 588 (18) | 155 (68) | 294 (36) |
| gate_up IQ3_XXS | 159 (79) | 467 (27) | 147 (85) | 284 (44) |
| gate_up IQ3_S | 176 (80) | 503 (28) | 141 (99) | 299 (47) |
| gate_up IQ4_XS | 148 (118) | 443 (39) | 143 (122) | 325 (54) |
| down IQ4_NL | 82 (112) | 204 (45) | 67 (139) | 222 (41) |
| down Q2_0 | 78 (59) | 161 (29) | 54 (86) | 128 (36) |

- **Untuned kernels already put one token's experts at ~10 ms (B = 1).** That is 29 % of the bandwidth bound
  (2.9 ms); the typical layer (gate_up IQ3_XXS + down IQ4_NL) takes ~210 µs. With the dense part at ~14 ms (unmeasured), a resident pass would take ~24 ms
  (~40 passes/s) with these kernels.
- **The kernels are latency/ALU-bound, not bandwidth-bound:** IQ2_S, the smallest format, is the slowest per byte
  (48 to 68 GB/s), because time follows the number of groups of 8 decoded, not the bytes. Loading the inputs once per
  4 rows (r4) gained 21 % at B = 1 and 30 % at B = 4; the time still grows by ~4.6 ms per extra token, so B ≥ 2 needs
  more reuse (inputs in threadgroup memory, more rows per simdgroup, decode into registers once per group).
- **For the reconciliation with the external ~8 tok/s:** 8 tok/s is ~125 ms per token; the expert kernels explain
  ~10 ms of it and the dense part ~14 ms (estimate). The remaining ~100 ms must come from SSD reads, waits and
  handshakes, so the kernels are not what caps the external engines (their per-layer kernel time matches ours).
- r1 in the second set is 4 % slower at B = 1 than in the first set (12.9 against 12.3 ms), just outside the ±3 %
  expected; the only code change in r1 is the inner sum written with `dot` (cause not checked further).
