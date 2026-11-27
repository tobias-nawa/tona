# Ornith O1 and O2: model module, CPU oracle, CPU expert kernels Q4_K and Q6_K (2026-11-15)

## Plan (written before the runs)

- Scope: one load of the real file (`tona-model`), the oracle teacher-forced over 64 tokens (one prose text, one code
  text), the mini Ornith over 32 tokens, `test_cpu_experts` and `tona-expert-check` on Ornith and on the Coder.
  One run each; the checks are deterministic (finite or not, bit-exact or not), the speed figure is a by-product.
- Expectation: all 753 tensors bound, 40 layers (30 DeltaNet, 10 gated attention at 3, 7 ... 39) plus MTP `blk.40`;
  no non-finite logit; expert kernels bit-exact against the GPU rows (`TONA_Q8_LANES=8`) and within 1e-5 per matvec of
  the dequantization oracle; speed within 2x of the measured 58 GB/s RAM read bound (N3), source: 2026-11-15 reference
  platform table in `docs/backends/reference-platform.md`.
- Active tooling: none of the diagnostic switches; `OMP_NUM_THREADS=6` for the oracle.
- Machine state: not quiet (quiet_check: load average 3.3, 1.2 GiB VRAM held by others, the 148 workers share the GPU
  queue); the speed figures are therefore upper bounds on the cost, the correctness checks are not affected.

## Test conditions

`tools/hw_report.py` (after the runs): Ryzen 5 9600X, RX 9060 XT gfx1200 (1.21 GiB VRAM held by others), 60.5 GiB RAM,
Fedora 44, kernel 7.2.15, hip 7.16.26385, no `TONA_*` variables. quiet_check before and after: warning, load average
3.3 / 3.4 and 1.21 GiB VRAM held by others (no program names recorded).

## Results

- O1: `tona-model` on the Ornith file: "40 layers (10 gated attention) + 1 MTP, d 2048, 256 experts (top 8), vocab
  248320, all shapes checked". Oracle, 64 teacher-forced steps: prose 0 non-finite logits (top-1 49.2 % against the
  teacher text), code 0 non-finite (top-1 87.3 %); 0.44 to 0.55 s per step. Mini Ornith: 32 steps, 0 non-finite.
  Coder mini model: top-5 lines of `tona-ref` identical before and after the shared-block refactor.
- O2 (N5): Ornith, per matvec worst relative deviation 3.6e-08 (Q6_K down) and 1.6e-08 (Q4_K gate); per layer 1.4e-07
  to 3.5e-04 (layer 1; the Coder's layer 17 shows 2.1e-04 in the same test, a cancellation effect of the test's
  random input, ungeprueft). `tona-expert-check`: bit-exact (0 of 1,048,576 silu, 0 of 2048 and 512 rows differ).
  The Coder: unchanged, bit-exact on all types.
- O2 speed (decode-shape, 6 to 12 experts, 4 tokens, 6 threads): 44 to 69 GB/s effective against the 58 GB/s bound
  (N3 holds: within 2x; layer 17 is the slowest at 43.9 to 48.8 GB/s). One thread, one matrix: Q6_K down 55 GB/s, Q4_K
  gate 13.6 GB/s (the Q4_K path is the slower one single-threaded; ungeprueft whether it matters with 6 threads, the
  layer figures say no).
- C1 for O2: no file under `engine/gpu/` differs from main c363c80 (`git diff c363c80 -- engine/gpu` is empty), so the
  device code is identical by construction. The isa-diff against the older `work/build/base` (R0) reports the F1
  full-attention kernels as added and `op_qsa_prep` as differing; both come from wave 2 on main, not from this
  branch (the cause of the `op_qsa_prep` difference is ungeprueft, a diff against a main build would settle it).
- C3 (`GOLDEN_PLAN=cpu`): queued, result in the hand-back.

## Prior art

- llama.cpp `src/models/qwen35moe.cpp` (b11822): layer pattern, silu readout gate of the DeltaNet norm, MoE with
  shared expert and sigmoid gate, MTP layer layout; read, our oracle follows it and is checked against the file.
- llama.cpp `ggml-cpu` Q4_K / Q6_K dot kernels: the format (super-blocks, 6-bit scales and mins); we do not use its
  float accumulation order but the GPU's int8 order, so CPU and GPU experts stay bit-identical.
- arXiv scan (linear attention, 2025-11-15 to 2026-11-14) was started; the book is not updated (`--no-book`), no
  result is used in this change.

## O3: Ornith on the HIP reference backend (2026-11-15)

Plan (written before the runs): cross-check the GPU path against the CPU oracle (real file, 64 teacher-forced ids
per prompt, widths 1 and 4) and against llama.cpp (N7: code and prose, 192 ids, last 64; one 32K prompt, last 64).
Expectation: KL against the oracle at the level of the Coder (mean below 0.03), top-1 equal except near-ties,
width 4 equal to width 1. A random-weight mini model is no oracle for this: its logits deviate by KL 0.65 (Coder
mini 0.02) while the real file agrees, so the mini only serves load and shape tests. Guards against the Coder:
C1 per object, C2, C3 (gpu and server goldens), C4 (batch bench W = 1, 4, 8, three interleaved runs).

Results (single machine, quiet check not repeated per run; no speed claim, so no BENCHMARKS row):
- GPU vs oracle, real Ornith Q4_K_M, 64 ids: prose top-1 93.75 %, mean KL 0.0100 (max 0.044); code top-1 98.4 %,
  mean KL 0.0050 (max 0.067). Width 4 bit-identical to width 1.
- N7 vs llama.cpp (llama's own CPU and HIP-with-CPU-experts runs agree exactly, so the baseline spread is 0):
  code top-1 100 %, mean KL 0.0087; prose top-1 92.2 %, mean KL 0.021 (max 0.27); 32K prompt top-1 92.2 %,
  mean KL 0.020 (max 0.15). The same level as GPU vs our own oracle, i.e. int8 activation quantisation.
- C1: every Coder object identical to main (6299ed6) with one rename rule (op_dn_norm becomes op_dn_norm<kSigmoid>);
  Ornith adds k_mega<NB, Qwen35MoeProfile<...>> and op_dn_norm<kSilu>. C2 identical (three dumps). C3 gpu goldens
  (csv, rust, explain) and all server goldens identical. C4: 22.3 / 33.7 / 48.2 ms per pass at W = 1 / 4 / 8,
  main and this branch within 0.1 ms, equal logits hash.
- N1: all k_mega variants at 192 VGPRs.

## O3 follow-up: per-op localization of the mini-model deviation (2026-11-16)

Question: the mini Ornith (random weights) differs from the oracle by about 1 % on the residual after one layer (final
KL 0.65, Coder mini 0.02). Is an op wrong? Plan: tap every intermediate of layer 0 on the GPU (temporary hooks, not
committed) and in the oracle, one token, same ids; compare relative L2 error per op. Expectation: matvecs at the level
of int8 activation quantisation (0.2 to 0.5 %); a wrong op would show a jump of tens of percent at one op. Models: the
mini with one layer (random weights, d 256) and a one-layer model cut from the real file (real weights of block 0,
d 2048, 256 experts).

| Op (layer 0, GPU vs oracle, relative L2) | mini, random weights | real weights, block 0 |
|---|---|---|
| DeltaNet in-projections qkv / z / beta / alpha | 0.39 / 0.37 / 0.17 / 0.44 % | 0.19 / 0.27 / 0.15 / 0.19 % |
| conv + silu + L2 norm of q, k | 0.32 % | 0.06 % |
| recurrence + head norm + silu(z) gate (readout) | 3.4 % | 0.12 % |
| router logits | 0.39 % (same top-8 set) | 0.29 % (same top-8 set, order) |
| shared-expert gate scalar | 0.50 % | 0.08 % |
| shared expert output | 1.3 % | 2.2 % (absolute 0.012 on a norm of 3.6) |
| MoE output (routed + shared) | 1.0 % | 0.95 % |

Finding: no wrong op. The one outlier, the DeltaNet readout of the mini (3.4 %), is conditioning of the random
weights: at the first token the state is zero, so the raw output is v * beta * (k . q) / sqrt(128); the random q and k are
nearly orthogonal (cosine -0.008 to -0.017), a cancellation that magnifies the 0.3 % input error tenfold, and the raw
mean-square (5e-8) is far below the norm eps (1e-6), so the norm does not renormalise. Shown, not assumed: a float64
numpy replay of the recurrence from the GPU's own inputs reproduces the GPU's raw output to 4e-7, and the implied
norm weight (output / (raw * rsqrt(ms + eps) * silu(z))) is identical on GPU and oracle to 7e-7, so the eps, norm
and silu gate are exactly the formula; replayed from the oracle's inputs the raw output differs by 3.2 % from the
replay on the GPU's inputs, which is the whole deviation. On the real block 0 the same op is 0.12 %. The
shared-expert output (three chained int8 activation quantisations, shared with the Coder through the same toolbox
ops) is the largest real-weight number and tiny in absolute terms. Deeper layers of the real model (taps of layer 38 and
39, accumulated through 38 layers): 2 % on the projections, 0.4 % on the router; this is the accumulated residual error, matching the
end-to-end KL of 0.005 to 0.01. Not tapped: the attention block (covered end to end only).
