# 36 — Draft length for MoE speculation: two published cost models fitted to our counters (issue #94)

Desk analysis, 2026-11-02. Nothing was measured, built or started; only numbers already in `research/measurements/` and
in the code were used. Marking: **[read]** paper HTML read (first ~100K characters; appendices not read), **[own]** our
measurement file, **[code]** read in our source, **[estimate]** own reasoning or fit on the numbers below. Rates named:
**decode** tok/s, **tokens per pass** = tokens emitted per verify pass (1 + accepted drafts).
The analysis script is short (numpy only); its inputs and results are in the tables below.

## 0. Summary

1. **Unique experts (2610.38981):** our measured union per layer (10.0 / 15.7 / 20.6 / 25.0 / 39.5 for 1 / 2 / 3 / 4 / 8
   tokens, [own] 2026-10-22-expert-routing) is 34-43 % below the uniform formula (25.0 against 37.7 at 4 tokens) and the
   consecutive-token overlap it implies (0.43 at the second token) is the paper's Qwen3-30B / Qwen3-235B value
   (0.431 / 0.447). The paper's effect exists on our model. The engine's own counter agrees (about 26 experts per layer
   in a 4-token pass, 2026-10-24-misses). The IPF fit itself needs pairwise overlap statistics per layer that we do not
   have; we use the measured union directly instead, which is what IPF would approximate to under 4 %.
2. **But unique experts are not what limits our verify pass.** Pass time rises ~4.9 ms per token (linear fit, 24.2 to
   58.1 ms over 1-8 tokens), while the expert bytes of the union grow by only 1.2-1.9 ms per token at 315 GB/s. A linear
   fit in the union is worse than a linear fit in the token count (rmse 1.5 against 0.8 ms). Bytes over bandwidth
   (2610.34440) therefore under-predicts our step time by a factor 2-3 and cannot be used as the cost model; the
   measured table `t_mpass_` is the better one (and is what the engine uses).
3. **Dedup of expert reads (2610.34440):** our kernels already read the union (gate_up + down read 2.6 GB for a
   4-token pass, which is 26 experts x 48 layers x 2.04 MB). A per-slot kernel would add 1.3 / 4.7 / 8.5 / 12.6 ms of
   bytes at 2 / 4 / 6 / 8 tokens; the paper's break-even `u < r` would need a per-slot kernel that is >1.6x as
   fast per requested byte as ours at 4 tokens (u = 0.625; ours reaches 76 % of VRAM peak). Not credible on
   GDDR6 with a 32 MiB Infinity Cache that holds ~16 experts. **Dedup stays; nothing to build.**
4. **Best draft length per acceptance band** (geometric acceptance, our measured pass table + MTP step costs): 1 draft
   at a <= 0.45, 2 at 0.5-0.62, 3 at 0.65-0.82, 4-6 at 0.85-0.90, 6-7 at 0.9-0.95; above 0.93 the model wants 9-12 drafts
   but our cap is 7 (width 8 = `BMAX`). The code bench is a ~ 0.88, text a ~ 0.68 (tokens per pass 5.27 / 3.01 at 7 drafts).
5. **Compared with what we run:** the engine already evaluates `E[tokens] / (verify pass(k+1) + catch-up + k steps)` per
   pass with calibrated per-position probabilities (`engine/server/draft.cpp`, `generator.h`) - the same objective as
   the paper's `T_a(k) = G_a(k)/V(k)`, with a measured `V`, but per pass instead of per band. A static staircase by
   running acceptance cannot beat that. The predicted decode gain from any change of the width rule is **0 to +1.5 %**
   (section 6), below the 2 % bar. **No GPU run is justified by this analysis.**
6. **Missing data** (section 7): per-width pass times with spread from the production MTP path (the table is one adaptive
   run), unions measured on verify passes with real drafts (not teacher forcing), and per-position MTP acceptance on
   `set`. All three come from one counters-only server run (recipe in section 7).

## 1. The two models (formulas)

### 2610.38981, Ranjan, Teru, Venkitesh, Locatelli - "Why Speculative Decoding Works Better Than Predicted on Sparse MoE
Models", NeurIPS 2026 [read, HTML v1]

- Target efficiency `TE = T_AR / T_SD` (one plain step over one verify pass of K+1 positions); speedup ~ acceptance
  length x TE. Prior bytes model: `TE ~ (C_dense + k C_expert) / (C_dense + unique_sd C_expert)`.
- Overlap statistic `overlap_j`: fraction of the top-k experts shared by tokens j positions apart. Baselines: uniform
  `E[U] = N (1 - (1 - k/N)^n)`, independent with skewed `p_e`: `sum_e (1 - (1 - p_e)^n)` (derivations by the summarizer,
  appendix G not read), empirical.
- Maximum-entropy IPF: for each expert the activations of B consecutive tokens are binary variables; fit the joint over
  2^B states subject to the marginal `p_e` and the pairwise constraint `P(X_t = 1, X_{t+j} = 1) = overlap_j p_e`;
  `E[U] = sum_e (1 - P(all X = 0))`. Skewed IPF within 4 % of measured unique counts on all 10 models (max +3.7 %,
  Qwen3-Next-80B, 512 experts); uniform IPF within 5 %.
- Numbers: unique experts for 4 tokens (K = 3): Qwen3-30B (8 of 128) uniform 29.1, independent 26.9, empirical 19.5.
  Step-1 overlap Qwen3-30B 0.431, Qwen3-235B 0.447, Kimi-K2.5 0.296.
- GPU finding (H100, small experts): MoE kernels use 45-49 % of HBM bandwidth against 22-25 % for attention at batch 1, and
  repeated co-routed experts hit the 50 MB L2 (5.7 % to 27.8 % hit rate), so TE is higher than the bytes model says.
  The paper says itself that this is kernel- and hardware-dependent (B200 differs).
- Limits: pairwise overlap only, no position dependence, transfer across model families not shown, mostly K = 3, MoE
  drafter analysis synthetic.

### 2610.34440, Lan - "Speculative Decoding for MoE: Expert Bytes, Kernel Bandwidth and the Best Draft Length" [read, HTML v1]

- Reuse `rho_l(n)`: share of token n's experts already in the n-1 preceding tokens. Distinct experts
  `U_l(n) = m sum_{j=1..n} (1 - rho_l(j))`. Byte share `u(n) = E_dedup / E_slot`.
- Verify time `tau_p(n) = t0 + D(n)/beta_D + E_p(n)/beta_p(n) + phi_p(n)`; step `V(k) = tau(k+1) + h(k)`.
- Dedup beats per-slot reads iff `u(n) < r(n)`, `r = beta_dedup / beta_slot` (measured r = 0.32-0.75, so dedup wins only
  from 5-8 tokens per pass, CUDA 4-6, Metal 6-8, MLX never).
- Geometric acceptance a: `G_a(k) = 1 + a + ... + a^k`, `T_a(k) = G_a(k)/V(k)`, `k*(a)` = smallest maximizer, a staircase,
  non-decreasing in a. One draft pays iff `a > V(1)/V(0) - 1`. For convex V: `k* >= j` iff `a^j V(j-1) > G_a(j-1) c_j`,
  `c_j = V(j) - V(j-1)`.
- Error bound: relative step-time error delta costs at most `2 delta / (1 + delta)` throughput; best draft length is the
  least sensitive prediction. Calibration: benchmark the expert op of one layer with chosen-id passes (under 1 h/model).
- Limits: batch 1, no draft cost in the main bound, stationary routing (repeating text breaks it), kernel bandwidths must
  be measured per model (fits from other models miss by 9-206 %), 4 small models, CUDA and Apple only.

## 2. Our inputs (all existing)

| Quantity | Value | Source |
|---|---|---|
| Model | 256 experts per layer, top-10, 48 layers, ~2.04 MB per expert (IQ1/IQ2), 0.98 GB experts + 4.11 GB dense per token | reference-platform.md, 2026-10-24-misses |
| Union of experts per layer, b tokens | 10.0 / 15.7 / 20.6 / 25.0 / 32.8 / 39.5 / 50.5 / 59.4 for b = 1 / 2 / 3 / 4 / 6 / 8 / 12 / 16 | [own] 2026-10-22-expert-routing (CPU reference, teacher forcing, 512 tokens of asyncio code, one text) |
| Union in a real 4-token pass | ~26 per layer (1,185 hits + 32 + 87 per pass, 60 refilled) | [own] 2026-10-24-misses |
| Expert op bandwidth, 4-token pass, all in VRAM | gate_up 255 GB/s (81 %), down 215 GB/s (69 %), together 2.6 GB in 10.9 ms | [own] 2026-10-24-misses |
| Pass time all in VRAM / real, B = 4 | 34.4-34.6 / 42.1-42.3 ms (misses 7.7 ms, hits 90.9 %) | [own] 2026-10-24-misses |
| Verify pass with MTP, width 1..8 | 24.2 / 29.5 / 33.1 / 36.7 / 43.7 / 49.3 / 52.7 / 58.1 ms (one adaptive run) | [own] 2026-10-26-mtp, [code] `t_mpass_` |
| MTP costs | 1.15 ms per step, 1.3 ms catch-up (shortened head, 41K rows) | [code] `t_step_`, `t_fill_` |
| Tokens per pass, 7 drafts fixed | code 5.27 (MTP) / 5.26 (DFlash2), text 3.01 / 2.88 | [own] 2026-10-26-mtp, -drafter-mtp |
| Tokens per pass, 3 drafts fixed | code 3.47 (MTP, llama.cpp) / 3.36 (DFlash2, ours), text 2.67 / 2.53 | [own] 2026-10-26-drafter-mtp |
| Server, adaptive depth, MTP | code ~90.6, text ~65.4 decode tok/s; `set` 2.6-3.0 tokens per pass on 2K prompts | [own] 2026-10-26-mtp, 2026-11-02-86-ab-ctx logs |
| DFlash2 (now `--dflash`, not default) | 3 drafts + large block, 6.3 ms drafter; pass 31.3 / 37.0 / 41.7 ms at width 2 / 3 / 4 | [own] 2026-10-25-drafter-calibration |
| IE_SPEC_CUT (ECOSPEC-style trimming) | 12 of 26 layers cut: pass 40.1 to 36.9 ms but tokens per pass 3.21 to 2.35 (decode 70.8 to 55.9 tok/s); one token ~14 ms, its experts ~3 ms | [own] 2026-10-24-misses, section "Shortening the verify pass" (file date 10-24; the issue says 10-26) |

## 3. Model 1 on our data: how many unique experts

Reuse from the measured union, `rho(n) = 1 - (U(n) - U(n-1)) / 10` (formula of 2610.34440), and byte share `u(n) = U(n)/(10 n)`:

| n | U measured | U uniform (N = 256, m = 10) | rho(n) | u(n) |
|---:|---:|---:|---:|---:|
| 2 | 15.7 | 19.6 | 0.43 | 0.785 |
| 3 | 20.6 | 28.8 | 0.51 | 0.687 |
| 4 | 25.0 | 37.7 | 0.56 | 0.625 |
| 6 | 32.8 | 54.4 | 0.61 | 0.547 |
| 8 | 39.5 | 69.9 | 0.67 | 0.494 |
| 12 | 50.5 | 97.3 | 0.73 | 0.421 |
| 16 | 59.4 | 120.7 | 0.78 | 0.371 |

(rho for n = 5, 7, 9 is interpolated from the table, which has no rows there.) rho(2) = 0.43 is paper 1's `overlap_1`; the
paper's values are 0.431 (Qwen3-30B) and 0.447 (Qwen3-235B) and its 4-token empirical/uniform ratio is 0.67 (19.5/29.1)
against ours 0.66. u(8) = 0.49 is inside the 0.43-0.50 range paper 2 measures for Qwen3 and OLMoE. So the routing
statistics of Qwen3.8-Flash-Next match the published fine-grained models, and the claim "speculation is cheaper in
experts than independence predicts" holds here (it is also why the 4-token pass is 2.5x, not 4x, one token in experts).

Caveat: the table is teacher-forced, one 512-token Python text, CPU reference (see its own "Limits"). Draft tokens
rejected by the verifier route differently from accepted text, and the production counter shows 26 against 25.0, so the
bias is small at n = 4 but unknown at n = 8.

## 4. Model 2 on our data: bytes over bandwidth

Expert bytes of the union at 315 GB/s: 3.1 / 4.9 / 6.4 / 7.8 / 9.0 / 10.2 / 11.2 / 12.3 ms for n = 1..8. Observed pass
time minus these: 21.1 / 24.6 / 26.7 / 28.9 / 34.7 / 39.1 / 41.5 / 45.8 ms. The non-expert remainder is not constant: it
grows from 21 to 46 ms. Fits to the MTP pass table (24.2 to 58.1 ms):

| Model | Fit | rmse | max error |
|---|---|---:|---:|
| linear in union `U(n)` (paper 2 with fixed overhead) | 10.7 + 1.16 ms per expert-per-layer | 1.5 ms | 8.1 % |
| linear in tokens `n` (per-slot / per-token cost) | 19.0 + 4.87 ms per token | 0.8 ms | 4.8 % |

The per-token term dominates: dense compute at width n (DeltaNet and attention state per token, LM head, WMMA tiles),
expert misses (7.7 ms per 4-token pass, which also grow with the union) and the width-5 kernel switch (+7.0 ms from 4 to
5 tokens against +3.6 for 3 to 4; `k_wide`/int8 path, 2026-10-25-width-8). Only about 1.2-1.9 ms per token is expert
traffic, consistent with the IE_SPEC_CUT result: a token's own experts ~3 ms of its ~14 ms. Consequences:

- Dedup lowers the per-token slope by about 30 % at most; the remaining 3.4+ ms per token is where speed would come from,
  and it is not an expert-bytes problem. Speculation is a loss for the pass only through that per-token term and the
  misses, not through the experts the drafts add.
- The paper's statement that the best draft length is the least sensitive output (loss <= 2 delta/(1+delta)) is what
  we can use; the byte side of its model we cannot, because our time is not bytes-proportional.
- IE_SPEC_CUT removes only experts and so saves ~3 of ~14 ms per cut token while losing the token: measured and
  explained by this split (also why DraftExpert/ECOSPEC gains do not transfer, research/papers.md).

## 5. Dedup threshold (2610.34440 section on r)

Dedup pays iff `u(n) < r(n)`. Ours is dedup by construction. For a hypothetical per-slot kernel to win at n = 4 it would
have to move the requested 3.9 GB (not 2.55 GB) in under 10.9 ms: 358 GB/s of requested bytes, above the 315 GB/s DRAM
peak, so only reuse from the 32 MiB Infinity Cache (~16 experts; every layer's repeats are consumed in a different
wave) could do it. The paper itself found r < 1 in every pair on H100/Metal because small-expert per-slot kernels get
their cache reuse; here the dedup kernel already reaches 69-81 % of peak, and a per-slot kernel wins only if r(4) < u(4) = 0.625, i.e. if it is >1.6x as fast per requested byte. The
per-slot extra bytes (1.3 / 4.7 / 8.5 / 12.6 ms at 2 / 4 / 6 / 8 tokens) are an upper bound on what dedup gives us. No
experiment is proposed; the argument would change only if a per-slot kernel measured above 330 GB/s effective.

## 6. Best draft length per acceptance band

`V(k) = t_mpass[k+1] + 1.3 + 1.15 k` (k >= 1; `V(0) = 24.2`), G geometric. Pass table beyond width 8 extrapolated with
4.0 ms per token (union slope) or 4.9 (fit); widths above 8 do not exist in the engine (`BMAX = 8`).

| Acceptance a | k* (table) | k* (U fit) | k* (n fit) | tokens per pass at k* | predicted decode (model) | loss of fixed k = 3 / k = 7 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0 | 0 | 0 | 1.00 | 41 tok/s (no draft) | -17 % / -49 % |
| 0.50 | 2 | 1 | 1 | 1.75 | 48 (x1.15) | -5 % / -38 % |
| 0.60 | 2 | 2 | 2 | 1.96 | 53 (x1.29) | -2 % / -32 % |
| 0.70 | 3 | 2 | 2 | 2.53 | 61 (x1.48) | 0 / -24 % |
| 0.80 | 3 | 4 | 3 | 2.95 | 71 (x1.72) | 0 / -13 % |
| 0.85 | 3 | 6 | 4 | 3.19 | 77 (x1.86) | 0 / -7 % |
| 0.90 | 6 | 9 | 6 | 5.22 | 86 (x2.07) | -3 % / -1 % |
| 0.93 | 9 | 11 | 7 | 7.37 | 95 (x2.29) | -8 % / -2 % |
| 0.95 | 11 | 11 | 9 | 9.19 | 104 (x2.53) | -14 % / -4 % |

(The last column uses the table V; "predicted decode" is the model's tok/s for one stream with the drafter cost but without
sampler, host and per-request overhead, so about 25 % above the server numbers; read ratios, not levels.) Fitting a from
our fixed-depth data (G_a(7) = 5.27 code, 3.01 text): code a = 0.876 (0.906 from the 3-draft number), text a = 0.684
(0.734). Acceptance is not geometric - the 3-draft numbers sit above the geometric prediction (3.47 against 3.32; 2.67
against 2.47) - so the effective a per position is higher at the start of the chain and lower at its end.

Reading against what we run:

- **Code (a ~ 0.88):** model says 5-6 drafts; the engine, adaptive, ends the chain by its own rate rule. 7 is the cap and the
  fixed-7 numbers (5.27 tokens per pass) are at the staircase. Nothing to gain.
- **Text (a ~ 0.68):** model says 2-3 drafts; the engine's table-based rule picks 2-3 (MTP text 2.6-3.0 tokens per pass in
  the logs). Fixed 7 would lose 24 %; fixed 3 loses 0.
- **High acceptance (a >= 0.93, repetitive output, csv-like):** the model wants 9-12 drafts, our cap is 7. Gain of an uncapped
  chain with the pass table extrapolated at 4.0 ms per token: +1.5 % at a = 0.93, +4.6 % at 0.95; with 4.9 ms per
  token +0 % and +1.1 %. This is the only band where the model says "longer than we allow", and it is the band the suffix
  path (`IE_SUFFIX_MIN`) and the history drafter already serve. Weighted over a realistic mix (a >= 0.93 in at most a
  few percent of passes) the gain is well below 0.5 % and would need kernels wider than 8 tokens.
- **Sensitivity to the cost table:** choosing k from the n-fit or U-fit instead of the measured table costs 0-4.8 %
  of throughput at single bands (e.g. U fit at a = 0.80: k = 4, -4.8 %), i.e. the measured table is better than either
  model. If the table is wrong by 5 % per entry, independent per width, the expected loss of the chosen k is 1.1-2.5 % at
  a = 0.5-0.93 (Monte Carlo, 4,000 draws); 3 % gives 0.5-0.9 %. That is the upper bound on what a re-measured, better
  table could give back; since the table was taken from one adaptive run (the 7.0 ms step at width 5 vs 3.4-3.6
  elsewhere is suspect), a re-measurement is plausible to move the choice at a = 0.68-0.88 by one draft and gain
  0.5-1.5 %. The paper's worst-case bound (delta = 5 % gives up to 9.5 %) is not reached on a flat optimum.

Prediction: expected decode gain from any change in how we choose the draft length: **+0.5 % (range 0 to +1.5 %)** on
`set`, up to +4 % only on rare very-high-acceptance passes that need width above 8. Below the 2 % bar of the issue.

## 7. What is missing and the smallest run that would deliver it

Missing (each by name):

1. **Per-width verify-pass times of the production MTP path with sample counts and spread.** `t_mpass_` is one adaptive run on
   a different build (2026-10-26); since then misses, slots (84 per layer at `--ctx 140000`) and kernels changed.
2. **Unions on real verify passes** (accepted plus rejected drafts) per width, to replace the teacher-forced table and to fit
   IPF properly (needs pairwise overlap per layer).
3. **Per-position MTP acceptance and chosen width histogram on `set`** (the current numbers are tokens per pass only), to
   test the geometric assumption and the share of passes with a >= 0.93.
4. **Pure expert-op bandwidth at 1..8 tokens** (paper 2 calibration), to measure r. Not needed: section 5 settles the
   question from bytes.

Smallest run (counters only, no new code: `IE_DRAFT_LOG` and `IE_ROUTE_TRACE` exist, docs/use/environment.md): one server
start on the big model, workload `set` (14 prompts, MTP, `--ctx 32768`), one arm, `IE_DRAFT_LOG=<file>` and
`IE_ROUTE_TRACE=<file>` on, `IE_DIAG=1`. Per docs/contribute/measurement.md that is principle 3 (deterministic counters first, one
run per arm) and a `set` run is about 5 min of GPU. Plan to write first: scope (one server, 14 prompts, up to 400
tokens), expectation (tokens per pass code ~5.3 / text ~3.0 at fixed 7 drafts, adaptive 2.6-3.0 on 2K prompts; pass times
within +-10 % of `t_mpass_`; U(4) = 25 +- 3 per layer), active tooling (the two logs, `IE_DIAG`). Do it with the next scheduled
scale run rather than as a separate GPU block (item 1 of the scale-run recipe already runs `set`). Then refit `t_mpass_`
from the log (hundreds of passes per width) and run the Monte-Carlo above with the real spread. A planned `ab.py` run on
the width rule is justified only if the refit moves some `t_mpass_` entry by more than ~5 % **and** changes the chosen
width at an acceptance band that holds more than 20 % of the passes.

## 8. Other candidate changes the models suggest (not predicted above 2 %)

- Smoothing the width-5 step (+7.0 ms): a hardware tile boundary, an engineering question about `k_wide`, not a
  draft-length question; its value is largest for a ~ 0.85-0.9 (4 to 6 drafts).
- Per-token term (3.4+ ms per extra token): the lever; any reduction raises k* and tokens per pass together (each
  millisecond per token at k ~ 3 is worth ~3-4 % decode). Candidates are in 2026-10-25-width-8 and the miss work, not here.
- Draft widths above 8 for repetitive output: needs a wider verify kernel; at best +1.5-4.6 % on rare passes.

## Sources

- Ranjan, Teru, Venkitesh, Locatelli, "Why Speculative Decoding Works Better Than Predicted on Sparse MoE Models", arXiv 2610.38981
  (NeurIPS 2026): https://arxiv.org/abs/2610.38981 [HTML v1 read to ~100K characters; appendices G and Qwen3-Next table not read]
- Lan, "Speculative Decoding for MoE: Expert Bytes, Kernel Bandwidth and the Best Draft Length", arXiv 2610.34440:
  https://arxiv.org/abs/2610.34440 [HTML read to ~100K characters]
- Related and open, not read: vLLM #64947 (MoE routes in adaptive verification cost profiling), llama.cpp #31497.
- ECOSPEC arXiv 2607.12696, DraftExpert arXiv 2607.24434 (research/papers.md): same trimming idea as `IE_SPEC_CUT`.
- Own: `research/measurements/2026-10-22-expert-routing.md`, `2026-10-24-misses.md`, `2026-10-25-drafter-calibration.md`,
  `2026-10-26-drafter-mtp.md`, `2026-10-26-mtp.md`; `engine/server/draft.cpp`, `engine/server/generator.h`; `docs/backends/reference-platform.md`.
