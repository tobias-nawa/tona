# 2026-11-11: dense part of one Coder pass on the M1 Max GPU, B = 1 to 4 (#108)

Step M1 of the Mac backend proposal ([../40-mac-backend-proposal.md](../40-mac-backend-proposal.md)). The
reconciliation ([../41-mac-reconciliation.md](../41-mac-reconciliation.md)) assumed 14 to 21 ms for the dense part of
a pass; it is the largest open number. This measures its two main costs with first-cut kernels: the mat-vecs of
every non-expert weight matrix and the DeltaNet state step. Speed numbers on this machine are provisional.

## Plan (written before the run)

**Scope.** Tool `bench/mac/densek.mm` with `bench/mac/densek.metal` and the shared decoders `bench/mac/decode.metal`
(the layout of [2026-11-11-mac-expertk.md](2026-11-11-mac-expertk.md): one simdgroup per 4 rows (1 where the row
count is not a multiple of 16), lanes over groups of 8 weights, B tokens share each decoded group). Tensor list:
the 737 two-dimensional non-expert matrices of the pinned Coder file in file order with their real types and shapes
(`work/gguf-header/coder-dense.tsv`, from the header read for the expertk run): Q6_K 129, BF16 436, Q4_K 47,
Q5_K 35, IQ4_XS 42, IQ4_NL 47, Q8_0 1; 4.11 GB, the LM head (Q6_K, 2560 × 248,320) included, the token embedding and
the n-gram table excluded (one row each per token). Weights synthetic. DeltaNet step: 36 layers, 48 V heads × 128 ×
128 float state (113 MB in all), the gated delta rule of `cpu/ref.cpp` for B consecutive tokens, state kept in
registers across them.

Not measured (counted separately in 41): conv, norms, gates, SiLU, attention over the KV cache and the indexer,
the n-gram lookup, and the dispatches they need (~2 µs each).

| Block | What runs | Rounds | Duration |
|---|---|---|---|
| check | each type's first tensor, B = 1 to 4, sampled rows against `tona::dequantize_row` (double sums); DeltaNet layer 0, B = 1 to 4, output and new state against the ref.cpp loops | once per run | ~2 s |
| role | per role (e.g. `attn_qkv`, 36 matrices) at B = 1, one command buffer, median of 10; doubles as warm-up | once per run | ~1 s |
| token | all 737 mat-vecs in one command buffer; the 36 DeltaNet steps in another; B = 1 to 4 | 20 each per B | ~3 s |

3 runs, 30 s pause.

**Expectation.**

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| check, relative error | < 1e-5 | < 1e-3 (abort above) | float against double |
| dense mat-vecs at B = 1 | 19 ms (215 GB/s) | 12 to 30 ms | wide rows and BF16 (no decode) should beat the experts' 29 % of the bound; estimate in 41: 14 to 21 ms |
| dense mat-vecs, B = 4 against B = 1 | 1.6× | 1.2 to 2.5× | as for the experts (2.35× there), wider rows help |
| DeltaNet steps at B = 1 | 1.0 ms | 0.6 to 3 ms | 227 MB of state read and written at ~250 GB/s |
| DeltaNet steps, B = 4 against B = 1 | 1.2× | 1.0 to 2× | state stays in registers; only the dot products repeat |

**Active tooling and conditions.** clang++ -O2, Metal source compiled at run time, no validation, no capture.
AC power. The tool allocates ~4.3 GB of shared buffers. Load recorded per run.

## Results

Raw data: the `2026-11-11-mac-densek-*.jsonl` files next to this one. Tools: `bench/mac/densek.mm`,
`bench/mac/densek.metal`, `bench/mac/decode.metal`.

**Conditions (13:21 to 13:24).** AC power. Load average 1.6 to 2.7 (1 minute); GPU utilization 0 to 1 % before
each run. All 96 checks passed, relative error at most 1.4e-6.

**Token** (ms, median of 20 command buffers; three runs; min to max over all 60):

| B | dense mat-vecs (GB/s) | DeltaNet steps | Expected |
|---|---|---|---|
| 1 | **34.93 / 35.27 / 34.70** (118), 33.9 to 35.4 | **0.90 / 0.91 / 0.89** | mat-vecs 12 to 30; DeltaNet 0.6 to 3 |
| 2 | 39.44 / 39.45 / 38.56 (105) | 1.00 / 1.00 / 0.98 | |
| 3 | 50.22 / 50.76 / 49.61 (82) | 1.19 / 1.18 / 1.14 | |
| 4 | 56.77 / 56.75 / 56.11 (72) | 1.38 / 1.37 / 1.33 | B4/B1: mat-vecs 1.2 to 2.5 (got 1.62), DeltaNet 1.0 to 2 (1.52) |

**Per role at B = 1** (run 2; µs for all matrices of the role; GB/s):

| Role | Count | Shape K → N | µs | GB/s |
|---|---|---|---|---|
| attn_qkv | 36 | 2560 → 10240 | 5,191 | 137 |
| ssm_out | 36 | 6144 → 2560 | 3,555 | 125 |
| output (LM head, Q6_K) | 1 | 2560 → 248,320 | 3,193 | 163 |
| hc_attn_down + hc_ffn_down (BF16) | 96 | 10240 → 320 | 6,090 | 103 |
| attn_gate | 36 | 2560 → 6144 | 3,028 | 125 |
| hc_attn_up + hc_ffn_up (BF16) | 96 | 320 → 10240 | 2,299 | 274 |
| hc_attn_inject + hc_ffn_inject (BF16) | 96 | 10240 → 4 | 1,965 | 4 |
| attn_q | 12 | 2560 → 12288 | 1,912 | 137 |
| ffn_gate_shexp + ffn_up_shexp | 96 | 2560 → 640 | 2,483 | 40 |
| ssm_alpha + ssm_beta (BF16) | 72 | 2560 → 48 | 1,163 | 15 |
| attn_output | 12 | 6144 → 2560 | 1,127 | 134 |
| ffn_gate_inp (router, BF16) | 48 | 2560 → 256 | 937 | 67 |
| ffn_down_shexp | 48 | 640 → 2560 | 449 | 100 |
| attn_k, attn_v, indexer.k_proj, indexer.q_proj | 48 | 2560 → 512 / 128 | 893 | 42 to 139 |
| ple_key, ple_value, output_hc_* | 4 | | 233 | 134 to 437 |

- **Finding: the dense mat-vecs take 35 ms at B = 1, outside the expected 12 to 30 ms** and above the 14 to 21 ms
  assumed in 41. The large matrices with many rows run at 125 to 165 GB/s (≈ 20 ms for ~3.1 GB). The rest is
  **matrices with few rows**: one simdgroup works through a whole row, so a matrix with 4, 48, 256, 320 or 640 rows
  fills 1 to 40 threadgroups on a 24-core GPU. These 600 small dispatches cost ~14 ms for ~0.9 GB; the 96 inject
  matrices (10240 → 4, 3.9 MB in all) alone take ~2 ms, ~20 µs each.
- Remedies, not built yet: split the inner dimension across simdgroups or threadgroups for matrices with few rows
  (several simdgroups per row, partial sums combined), and fuse the small projections that read the same input
  (alpha + beta; hc_*_down + hc_*_inject; router + shared-expert gate + up). At the large matrices' 125 to 165 GB/s
  for all of the 4.11 GB, the dense part would take ~25 to 33 ms; at 250 GB/s ~16 ms (target, not measured).
- **DeltaNet steps: 0.9 ms at B = 1**, as expected (~250 GB/s for the 227 MB of state); 1.4 ms at B = 4.
- **Resident pass with these first-cut kernels** (estimate from the parts): dense 35 + experts 10.2 (expertk) +
  DeltaNet 0.9 + unmeasured small ops and dispatches ~3 = **~49 ms (~20 passes/s)**. The reconciliation's 25 to 32 ms
  holds only after the dense kernels reach ~150 to 250 GB/s; 41 is to be updated once they do.

### Addendum to the plan (written after the first three runs, before the split-K set)

The first set (raw: [2026-11-11-mac-densek-first.jsonl](2026-11-11-mac-densek-first.jsonl)) traced ~14 ms to matrices
with few rows. Variant split-K (`mvs`): for matrices with fewer than 2048 rows, a threadgroup of 8 simdgroups shares
4 rows (1 row for row counts not divisible by 4) and splits the groups of 8 among them; partial sums meet in
threadgroup memory. Larger matrices keep the first-cut kernel. One process now runs both layouts (`split_below` 0 and
2048), checks included; 3 runs, 30 s pause.

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| first-cut layout | as the first set | ±3 % | same code |
| split-K, dense mat-vecs at B = 1 | 25 ms | 20 to 30 ms | the narrow matrices at ~100 GB/s instead of 4 to 67 (~9 ms instead of ~14); large ones unchanged (~20 ms) |

### Results of the split-K set

Raw data: [2026-11-11-mac-densek-splitk.jsonl](2026-11-11-mac-densek-splitk.jsonl). 13:27 to 13:30, load average 2.3 to 2.5, GPU
idle before each run. All 180 checks passed (≤ 1.4e-6).

| B | first-cut layout (ms, three runs) | split-K below 2048 rows (ms, three runs; GB/s) | Expected split-K |
|---|---|---|---|
| 1 | 34.62 / 34.91 / 34.81 | **28.16 / 27.99 / 28.15 (146)** | 20 to 30 |
| 2 | 39.03 / 39.01 / 39.27 | 32.82 / 32.67 / 32.84 (125) | |
| 3 | 50.29 / 50.29 / 50.08 | 42.45 / 42.62 / 42.53 (96) | |
| 4 | 56.45 / 56.44 / 56.40 | 49.99 / 50.11 / 50.13 (82) | |

The first-cut layout repeats the first set within 1 %. Split-K roles at B = 1 (run 2, µs, GB/s): hc_*_down
1,400 + 1,393 (225), router 418 (151), shared-expert gate/up 781 + 811 (63), ssm_alpha/beta 199 + 221 (40 to 44),
inject 503 + 511 (8), attn_k/v 151 + 167 (75 to 83), indexer 78 + 139.

- **Split-K saves 6.7 ms at B = 1 (−19 %)**: 28.1 ms, inside the expectation. The rest of the gap to a ~16 ms target
  lies in (a) the large matrices at 125 to 165 GB/s (~20 ms, the decode loop itself), (b) ~1 ms of inject matrices
  that are too small even for split-K and belong fused into the hc_*_down kernel, and (c) the shared expert and
  alpha/beta at 40 to 63 GB/s, candidates for fusion with the router and with each other.
- Resident pass with split-K (estimate from the parts): 28.1 + 10.2 + 0.9 + ~3 = **~42 ms (~24 passes/s)**.

### Addendum to the plan: steps with concurrent dispatches (written before the runs)

The reference puts all mat-vecs that read the same input into one step of its persistent kernel and lets all waves
work through the units of all of them (`engine/gpu/mega_build_layer.cpp`, `matvec_phase.h`): alpha + beta + qkv +
gate, hc down + inject, router + shared-expert gate and up. The Metal counterpart without a persistent kernel: a
concurrent compute encoder (`MTLDispatchTypeConcurrent`), the matrices of one step dispatched back to back with
their own output ranges, a buffer barrier between steps. Steps per layer as in the reference (hc_attn down + inject;
hc_attn up; attention or DeltaNet inputs; output projection; hc_ffn down + inject; hc_ffn up; router + shared
gate/up; shared down), 388 steps per token instead of 737 serial dispatches. Kernels as in the split-K set
(split below 2048 rows) and, for comparison, the first-cut layout. Same process adds `phased` rows; 3 runs, 30 s
pause; raw data in 2026-11-11-mac-densek-steps.jsonl.

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| serial rows | as the split-K set | ±3 % | same code |
| steps, split-K, B = 1 | 24 ms | 18 to 28 ms | the ~8 ms of narrow matrices partly hidden behind the wide ones of the same step; the wide ones (~20 ms) stay |
| steps, B = 4 | 45 ms | 38 to 50 ms | same share of the 50.1 ms serial time |

### Results of the step set

Raw data: [2026-11-11-mac-densek-steps.jsonl](2026-11-11-mac-densek-steps.jsonl). 13:51 to 13:54, load average 2.3 to 3.5, GPU
idle before each run. All 180 checks passed (≤ 1.4e-6). ms, median of 20 per run, three runs:

| B | serial, split-K | **steps, split-K** (GB/s) | steps, first-cut layout | Expected (steps, split-K) |
|---|---|---|---|---|
| 1 | 28.17 / 28.28 / 28.00 | **25.27 / 25.45 / 25.50 (161)** | 28.71 / 28.80 / 28.79 | 18 to 28 |
| 2 | 32.71 / 32.77 / 32.59 | 29.99 / 29.91 / 29.79 | 33.00 / 33.09 / 32.92 | |
| 3 | 42.35 / 42.48 / 42.52 | 39.28 / 39.45 / 39.01 | 42.98 / 42.78 / 42.94 | |
| 4 | 49.99 / 50.00 / 50.18 | **46.37 / 46.33 / 46.37** | 48.98 / 49.11 / 49.02 | 38 to 50 |

- Serial rows repeat the split-K set within 1 %. **Steps with concurrent dispatches save another 2.7 to 3.6 ms**
  (25.4 ms at B = 1, 46.4 ms at B = 4), inside the expectation; on the first-cut layout they save 6 ms, about as
  much as split-K alone, and both together give 9.4 ms against 34.8 ms.
- What is left is the decode loop of the wide matrices (~20 ms at 125 to 165 GB/s against 336 GB/s read bandwidth,
  [2026-11-11-mac-overlap.md](2026-11-11-mac-overlap.md)): next step. Resident pass estimate now 25.4 + 10.2 + 0.9 +
  ~3 = **~40 ms (~25 passes/s)**.

### Addendum to the plan: vector loads in the decoders (written before the A/B runs)

The decoders read every weight byte with its own load and, for Q6_K, the scale per element. New: one 8-byte load per
group (`uint2`, or four `ushort` for the 2-byte aligned Q6_K, IQ4_NL and Q8_0 blocks), nibbles and high bits
extracted four bytes at a time, one scale per group; BF16 one 16-byte load. Same results (the checks against
`quant.cpp` decide). A/B: old binary (commit 2bc77e8) and new one alternating, 3 runs each, 30 s pause; one smoke
run of the new one before this plan gave 21.5 ms (steps, split-K, B = 1) and is not counted.

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| old binary, steps + split-K, B = 1 | 25.4 ms | ±3 % | step set |
| new binary, steps + split-K, B = 1 | 21.5 ms | 20 to 23 ms | smoke run |
| new binary, B = 4 | 40 ms | 37 to 43 ms | smoke run |

### Results of the vector-load A/B

A first A/B ([2026-11-11-mac-densek-vec-same.jsonl](2026-11-11-mac-densek-vec-same.jsonl), 14:01 to 14:05) was
invalid: densek compiles `decode.metal` at run time from the source tree, so the old binary ran the new decoders
too (both arms 21.0 to 21.6 ms). It counts as six runs of the new code, not as a comparison. The repeat takes the old
sources from a worktree of 2bc77e8 (raw: [2026-11-11-mac-densek-vec.jsonl](2026-11-11-mac-densek-vec.jsonl), 14:06
to 14:10, load average 2.1 to 4.8, GPU idle before each run). All 360 checks passed (≤ 1.4e-6); expertk's 48
checks pass with the new decoders (≤ 2.4e-7).

| B | steps + split-K, old decoders (ms, three runs) | **new decoders** (ms; GB/s) | Expected (new) |
|---|---|---|---|
| 1 | 25.44 / 25.48 / 25.48 | **21.54 / 21.51 / 21.55 (190)** | 20 to 23 |
| 2 | 29.92 / 29.92 / 29.98 | 25.72 / 25.86 / 25.90 | |
| 3 | 38.89 / 39.06 / 39.17 | 33.78 / 33.58 / 33.58 | |
| 4 | 46.42 / 46.10 / 46.27 | **39.85 / 39.80 / 39.94** | 37 to 43 |

- **Vector loads save 3.9 ms (−15 %) at B = 1** and 6.4 ms at B = 4, inside the expectation; the old arm repeats the
  step set within 0.3 %. Per role (smoke run): BF16 hc up 305 GB/s, hc down 250, LM head (Q6_K) 209; the Q4_K,
  Q5_K and Q6_K projections with K = 2560 to 6144 stay at 150 to 170 GB/s, the next target.
- Dense part since the first cut: 34.9 → 21.5 ms at B = 1, 56.5 → 39.9 ms at B = 4. Resident pass estimate
  21.5 + 10.2 + 0.9 + ~3 = **~36 ms (~28 passes/s)**.

### Explored, not adopted (smoke runs, one each, steps + split-K, B = 1; not speed results)

| Change | ms (21.5 with the committed kernel) | Wide K-quant roles |
|---|---|---|
| 2 rows per simdgroup (`ROWS=2`) | 21.7 | unchanged |
| 8 rows per simdgroup (`ROWS=8`) | 25.6 | ssm_out 95 GB/s |
| two groups per lane and step, decoded before the products | 27.7 | 103 to 125 GB/s (register pressure, unchecked) |
| 2 / 8 simdgroups per threadgroup (`NSG=`) | 21.5 / 23.1 | |

Row count, unrolling and threadgroup size are not what holds the Q4_K, Q5_K and Q6_K projections at 150 to 170 GB/s.
Next hypothesis (unchecked): the input loads, 4 bytes of float x per weight against 0.56 to 0.82 bytes of weights.
| x staged in threadgroup memory where B × K ≤ 8192 (`XTG=1`, bit-identical products) | 25.4 | ssm_out (K = 6144) 84 GB/s: the 24 KB copy per threadgroup costs more than it saves |

Diagnostics (smoke runs, wrong results by design, `Q6LOADS=`): Q6_K with its loads but without its arithmetic, and
the same with every load aligned down to 8 bytes. The only all-Q6_K role, the LM head, goes 209 → 296 GB/s (loads
only) → 261 GB/s (aligned loads only); the projection roles mix Q4_K, Q5_K and Q6_K per layer (attn_qkv: 22 of 36
Q6_K), so their 170 → 190 to 200 GB/s is not a clean per-type number. Reading of these runs: the arithmetic costs a
part, alignment does not explain the rest, and the input loads are not the limit. Next: one shape per type
(2560 → 10240) timed alone for a clean ceiling per decoder before any further change.
