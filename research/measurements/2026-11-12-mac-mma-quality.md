# Mac: quality of the half MMA mat-vec (TONA_METAL_MMA) against the f32 path (#108)

Maintainer decision 2026-11-12: half precision for passes of W ≥ 4 on Metal is allowed after a quality check;
W = 1 stays f32, the switch brings the f32 path back, default on only if the check passes.

## Plan (written before the run)

- Scope: `tona-width-nll` (new, `engine/tools/tona_width_nll.cpp`): each text teacher-forced through two Metal
  backends in lockstep on DM; A in passes of 1 token (f32 kernels, the exact path), B created with
  `TONA_METAL_MMA=1` in passes of 4 tokens (the half MMA kernel for dense matrices from 2048 rows below the
  vocabulary; everything else as A). Prefill of 8 tokens in both, then 10,000 predicted tokens per text. Texts
  (`work/quality/`, built from the repository at commit `acc2e63`): code = `engine/backend/metal/*.mm` and
  `engine/server/*.cpp`; shell = `justfile` and `scripts/*.sh` (as docs/backends/README.md names code and shell); text =
  `docs/*.md` (prose, asked for by the coordinator). Duration about 40 minutes. Evaluation with
  `scripts/nll-compare.py` (block bootstrap, blocks of 64 tokens, 4,000 draws, fixed seed); "alone" is A, "together"
  is B in its output.
- Thresholds (docs/backends/README.md#checks item 2, fixed before the run), per text, on the 95 % interval of the mean
  Δ(−log p) = B − A: upper end below 0.01 nats: pass; mean below 0.01 with the upper end at or above: undecided
  (measure more tokens or run the task benchmark); mean at or above 0.01: fail. KL and top-1 reported.
- Expectation: the 40-token smoke test gave +0.0065 nats (KL 0.0032, top-1 97.5 %). Half weights carry about 11 bits
  of mantissa against the decoded 6-bit codes and fp16 scales, so the weights are exact in half up to the product
  `d * scale`; the inputs (activations) lose precision. Expected mean Δ 0.000 to 0.008 nats per text; a text above
  0.01 would be a finding.
- Active tooling: `TONA_METAL_MMA=1` in B only (set by the tool before creating B).

## Prior art (researched 2026-11-12, window 2026-08 to 2026-11)

| Source | What it does | What Tona takes |
|---|---|---|
| [MLX #4712](https://github.com/ml-explore/mlx/pull/4712) (`qmv_mma`, `qmv_nax`, merged 2026-10-22) | 8x8 simdgroup MMA for quantized mat-vecs of 6 to 16 rows, named for speculative verify; affine 4-bit formats | the 8x8 fragment and half fragments with an f32 accumulator; Tona starts at 4 rows and K-quants |
| llama.cpp `mul_mv_mma` (ggml-metal, local checkout `d9ee007`) | half fragments for small batches from 2 to 3 rows | the lane layout of the fragment (also the order of `TONA_Q6_K_T`, docs/use/tona-types.md) |
| [skinny-mlx](https://github.com/rachittshah/skinny-mlx) (seen 2026-11-12) | bf16 MMA for 2 to 16 tokens, weights repacked into fragment lane order at twice the memory; ~2.5× at 8 tokens on an M4 Max | the fragment-order data layout, offline as a file variant (one copy, no runtime memory) |

All three use 4-bit affine or bf16 weights, whose decode is a shift and a multiply; the Coder's dense matrices are
Q6_K, BF16, Q5_K and Q4_K with super-block scales, so the decode, not the matrix units, sets the cost
(2026-11-12-mac-dense-mma-probes.md). The half path's quality on the Coder is measured below.

## Run (2026-11-12, 17:34 to 20:39)

Conditions: commit `f3dc902` (clean tree; kernels as `acc2e63`), AC power. Quiet check before (17:34): GPU 91 % busy
(the checks just finished), load 19.9, other programs 11.3 % of all CPU threads; `tools/hw_report.py` at the start:
load 18.96 / 15.39 / 12.14, other programs 21.2 %. After (20:40): ok. The check compares numbers only, so the load
lengthened the run (3 h 5 min against ~40 min planned) without touching the result. Raw output: 30,000 lines,
`work/quality/mma-dm.txt` (not in the repository, 2.0 MB); text 2 evaluated alone because the script names only two
conversations.

| Text | Tokens | Δ(−log p) B − A (95 % interval) | KL (95 % interval) | KL p99 | Top-1 equal | Verdict |
|---|---|---|---|---|---|---|
| code | 10,000 | +0.0018 [−0.0002, +0.0038] | 0.0040 [0.0036, 0.0044] | 0.046 | 98.02 % | pass |
| shell | 10,000 | +0.0010 [−0.0014, +0.0037] | 0.0044 [0.0040, 0.0050] | 0.046 | 97.54 % | pass |
| text | 10,000 | +0.0010 [−0.0012, +0.0033] | 0.0053 [0.0050, 0.0056] | 0.038 | 96.71 % | pass |

Reading: within the expectation (0.000 to 0.008 nats). Every upper end is below 0.01 nats, and every interval
contains 0: the half path's loss is not distinguishable from zero at 10,000 tokens per text. The spread per token of
the difference is 0.11 to 0.12 nats, so the intervals are about ±0.002. KL is about 1.3× the smoke test's 0.0032.
Result: the switch passes the check of docs/backends/README.md#checks item 2; default on is decided after run 5 (speed) and the
server A/B.
