# 59 - Is a better drafter than the model's own MTP head worth building? (issue #158)

Desk analysis, 2026-11-16. Nothing was measured, built or started on the GPU. Marking: **[own]** our measurement file or
code, **[abs]** abstract or model card read this session, **[known]** row already in `research/papers.md` (read in an
earlier week), **[calc]** own arithmetic with `tools/self_spec_model.py` (reused, not rewritten; the driver is the
10-line loop in section 3). Rates named: **decode** tok/s, **tokens per pass** = tokens emitted per verify pass.

## 0. Verdict

**Close the drafter-architecture question (EAGLE-3, HASS, GRIFFIN, PARD, Falcon, DFlash/DSpark-class). Do not build.** One
narrow follow-up stays open as a maintainer decision, not as a build: self-distilling the MTP head (same 1.15 ms step).

1. **A new drafter pays only at a step cost near the MTP head's.** From our curve: at 2 ms per step survival must rise by
   +0.08 at every depth just to break even with today's MTP; at 4 ms by +0.26 (section 3). Every drafter family with a
   published gain over a native MTP head steps for several ms on our class (DFlash2 6.3 ms [own]), and the one measured
   on this model ties MTP (5.26 vs 5.27 tokens per pass code, 2.88 vs 3.01 text [own]).
2. **Our head is already strong.** Conditional acceptance per depth is 0.85 to 0.93 [own]. The published self-distilled
   MTP head (FastMTP) reaches 0.80 / 0.70 / 0.64 at depth 1 / 2 / 3 after training, ours is 0.851 / 0.845 / 0.866
   untouched. The loss at depth comes from compounding, not from a weak head: +0.01 conditional at every depth is worth
   only +3.4 % in the model, and the cap of 8 positions in the engine binds (width 8, research/37).
3. **No published method reports a gain over a native MTP head on a Qwen-class MoE that we could transfer**: EAGLE-3 on
   Qwen3.5-397B has the lower acceptance length than native MTP on every benchmark of its card, DSpark on our model is
   below MTP (3.70 vs 4.24 agent sessions, 6.76 vs 6.96 HumanEval) [known], DFlash2 ties [own]. The method papers
   report gains over EAGLE-2/EAGLE-3/DFlash (+8 to +28 %), not over a strong native head.
4. **Training does not fit this PC for the interesting variants.** A head with its own MoE/QSA block (the MTP GGUF is
   4.14 GB at Q8 [known]) cannot be trained in 16 GB VRAM; the target's full-precision weights are far above 64 GB RAM.
   A rented multi-GPU run would be needed, with data self-distilled from the 1-3 bit model we ship (the head must match
   the quantised trunk's errors), and the expected real gain is not above 0 to +3 % (section 5).

Rule check (#158): the *bound* is above noise (+6.7 % at +0.05, noise sd 0.2 to 0.6 %, `bench/noise.toml`), but the bound
is not what decides: the realistic uplift of a retrained head over a head that is already at 0.85 to 0.93 conditional is
a fraction of +0.05, and every architecture other than the MTP shape loses on step cost. Code workloads were the
reference throughout (92.2 decode tok/s code, 67.0 text).

## 1. Prior art (2025 to 2026)

Papers screened with `tools/papers/scan.py` (arXiv 2026-11-11..14, see section 6 for the
coverage state; 2 hits tagged `speculative` in arXiv, neither a drafter: 2611.16714 memory-based speculative decoding
for green text generation, 2611.18459 Callo, cache-miss requests under PD disaggregation) plus the rows of
`research/papers.md` weeks 43 to 46 and web search today. "vs MTP" = against the model's native MTP head, "vs none" = against no
speculation. Gains are the authors' numbers on their hardware, not ours.

| Method | Source | Reported gain | vs native MTP on Qwen-class | Step cost | Training data / compute |
|---|---|---|---|---|---|
| EAGLE-3 | [arXiv 2503.01840](https://arxiv.org/abs/2503.01840) [abs] | up to 6.5x vs none, 1.4x vs EAGLE-2; SGLang 1.38x throughput at batch 64 | Qwen3.5-397B-A17B community EAGLE-3 ([card](https://huggingface.co/AQ-MedAI/Qwen3.5-397B-A17B-eagle3)) [abs]: acceptance length below native MTP on every listed benchmark (MT-Bench 3.03 vs 3.28, CEval 1.77 vs 2.34), throughput ahead on GSM8K/Math500/HumanEval, close elsewhere; author blames limited training scale. One agent-workload blog claims 2.2-2.3x MTP acceptance, model not named [abs] | one transformer layer on 3 fused feature levels plus a full LM head per step; no number on our hardware | training-time test, scales with data (no size in abstract) |
| HASS | [arXiv 2408.15766](https://arxiv.org/abs/2408.15766) [abs] | +8 to +20 % over EAGLE-2, 2.81-4.05x vs none (four LLaMA models) | none | none over EAGLE | distillation plus context alignment; extra training passes, no inference cost |
| GRIFFIN | [arXiv 2502.11018](https://arxiv.org/abs/2502.11018) [abs] | acceptance length +8 %, speedup +7 % over the best prior (LLaMA, Vicuna, Qwen, Mixtral) | none | draft head takes input tokens, small | masks misaligned tokens in the loss |
| PARD | [arXiv 2504.18583](https://arxiv.org/abs/2504.18583) [abs] | adapts a small AR model into a parallel drafter, up to 3x training efficiency; vs AR drafters, not vs EAGLE-3 | none | one pass for all drafts (TensorRT-LLM supports it) | light adaptation of a small model |
| Falcon | [arXiv 2412.12639](https://arxiv.org/abs/2412.12639) [abs] | 2.91-3.51x on Vicuna/LLaMA2-Chat, beats EAGLE, Medusa, Lookahead | none | semi-autoregressive | not in abstract |
| FastMTP | [arXiv 2509.18362](https://arxiv.org/abs/2509.18362) [abs + HTML] | **re-training of the native MTP head**: 2.03x vs none at K=3, vanilla MTP 1.21x; acceptance k=1/2/3 70/10/0 -> 80/56/36 % | this is the MTP-vs-tuned-MTP case, on a 7B dense model whose head was not trained for recursion | same as MTP head (one head applied recursively) | 389.4K self-distilled samples, 3 epochs, MiMo-7B-RL, head 210.8M parameters, one H20 server, under 1 day |
| MTP-D | [arXiv 2603.23911](https://arxiv.org/abs/2603.23911) [abs] | self-distillation of MTP heads, +7.5 % head acceptance, loop extension for more heads | the closest to "re-train our head"; model sizes not on the abstract page | same as head | "minimal extra training cost", no figure given |
| DSpark / HF card | [known] | DeepSeek-V4 +60-85 % vs MTP-1 (a 1-token baseline) | on our model: 3.70 vs MTP-7 4.24 (agent), 6.76 vs 6.96 (HumanEval): **below** MTP | 5 layers, 665M | trained by third parties |
| DFlash2 | [known] + [own] | model card 2.94 vs 2.52 acceptance (HumanEval, vs MTP-3 in SGLang) | **ours: ties MTP at 7 drafts** (5.26 vs 5.27 code, 2.88 vs 3.01 text) | 6.3 ms | third-party sidecar, 0.65 GB |
| ReaDy, DLite, DStill, ChainFlow, D3D, D3Tree | [known] | +8.5 % acceptance vs DSpark; up to +70 % vs DSpark; +28 % vs DFlash; +39 % vs EAGLE-3 | none measured against MTP | diffusion / flow, several ms | all need training |
| SpecFit, DraftExpert, SP-MoE, AcceptMoE | [known] | gains by cost-aware depth or expert prefetch, not by better drafts | n/a | n/a | n/a |

Tracker status (PR numbers as named in `research/papers.md` or found today; states not re-checked except where noted):

- llama.cpp [#29761](https://github.com/ggml-org/llama.cpp/pull/29761) MTP for qwen4exp **merged** [known]; [#30927](https://github.com/ggml-org/llama.cpp/pull/30927) DFlash on qwen4exp **merged**, acceptance 2.37 on M5 Max [known]; [#27210](https://github.com/ggml-org/llama.cpp/pull/27210) adaptive MTP depth, open; [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) / [#31037](https://github.com/ggml-org/llama.cpp/pull/31037) trimmed draft LM head, open; [#24593](https://github.com/ggml-org/llama.cpp/pull/24593) and [#21437](https://github.com/ggml-org/llama.cpp/pull/21437) EAGLE3 for Qwen3.5/3.6 (title only, state not checked); [#19493](https://github.com/ggml-org/llama.cpp/pull/19493): a community write-up reports llama.cpp speculative modes not beating baseline on Qwen3.6-35B-A3B (RTX 3090), single user, unverified.
- SGLang: EAGLE3 for Qwen3.5 MoE needs Spec V2 and [#25408](https://github.com/sgl-project/sglang/pull/25408) (from the card, state not checked); SpecForge/SpecBundle lists community EAGLE3 drafts for Qwen3-Next-80B ([cookbook](https://docs.sglang.io/cookbook/specbundle/supported_models)); a third-party EAGLE3 draft for Qwen3-Coder-Next needs a custom fork.
- vLLM: native MTP for Qwen3-Next exists (`qwen3_next_mtp`); no EAGLE3 PR for Qwen3-Next found by search (not exhaustive); the blog and PRs #52228 / #47808 on adaptive verification are already in papers.md.

Reading across: (a) for Qwen-class models the vendor ships the MTP head precisely because it is the strongest cheap
drafter; third-party EAGLE-3 heads trail it in acceptance and win, where they win, on step cost in a framework where the
MTP path is slow. In our engine the MTP step is 1.15 ms already. (b) The only evidence of an MTP-head retrain winning is
FastMTP, and its vanilla head was not trained for recursive use (10 % at depth 2); Qwen's head is (0.845 conditional at
depth 2 [own]). (c) Our Qwen 4 class target has a hybrid DeltaNet trunk; a component-aware self-spec paper found that
mixed hybrids need the full-attention layers for a good draft (2605.01106 [known]), which is why a trunk-only draft
(layer skip, #155) was negative.

## 2. Inputs, all from existing files

| Quantity | Value | Source |
|---|---|---|
| MTP step / fill | 1.15 ms / 1.3 ms | `tools/self_spec_model.py`, generator.h |
| Survival over depth 1..7 | 0.851, 0.719, 0.622, 0.582, 0.53, 0.475, 0.424 | research/measurements/2026-11-15-self-spec-layer-skip.md, stage 4 (adaptive depth, deeper values carry selection) |
| Conditional acceptance per depth | 0.851, 0.845, 0.866, 0.935, 0.91, 0.897, 0.892 | same |
| Tokens per pass, verify pass, decode | 3.98, 42.1 ms mean, code 92.2 / text 67.0 tok/s | same |
| Verify pass table, width 1..8 | 23.7 / 28.9 / 32.5 / 35.9 / 41.7 / 46.1 / 49.6 / 52.6 ms | `PASS_MS` in the tool (research/37 has the older 24.2-58.1 ms run) |
| Width cap | 8 positions (`BMAX`) = 7 drafts | research/37 |
| Noise | `decode-code` sd 0.065 % (df 2, not used), `decode@2000` 0.23 %, `decode@2000:write` 0.61 % | `bench/noise.toml` (the ticket says scripts/noise.toml, the file is bench/noise.toml) |

## 3. Upper bound with the cost model

Method: `tools/self_spec_model.py` (`best()` with no layer-skip extension) with `S_j' = min(1, S_j + d)` kept
non-increasing, k chosen by the model (1..7), `MTP_STEP_MS` replaced by the step cost under test; the fill (1.3 ms) and the
verify table are kept. The model's absolute level is 84.0 tok/s against a measured 92.2 (code) and 67.0 (text), and its
tokens per pass at k = 7 is 5.2 against the measured 3.98 because the survival curve was recorded with adaptive depth and
the model applies it at fixed k = 7 for the bound: **read ratios, not levels.** The gain column is against today's
MTP at 1.15 ms (model 84.0).

| Survival rise d at every depth | step 1.15 ms | step 2 ms | step 4 ms |
|---|---:|---:|---:|
| 0 (today) | 0.0 % | -8.8 % | -22.8 % (k=3) |
| +0.05 | +6.7 % | -2.6 % | -19.1 % (k=3) |
| +0.10 | +13.5 % | +3.5 % | -14.2 % |
| +0.20 | +25.9 % | +14.9 % | -4.7 % |
| perfect survival (S = 1), k = 7 | +53.8 % (8.0 tokens) | +40.3 % | +16.3 % |

- Break-even uplift: a drafter at 2 ms needs +0.08 at every depth; at 4 ms +0.26; at 6.3 ms (DFlash2) it cannot break
  even below perfect survival (+0.0 survival yields -22 % or worse). A 4 ms drafter also shifts the best k to 3.
- Sensitivity: +0.01 survival at every depth = +1.35 % decode; +0.01 *conditional* acceptance at every depth (S_7 from
  0.424 to 0.459) = +3.4 %, +0.02 = +6.9 %, +0.03 = +10.5 %, +0.05 = +18.3 %.
- The asymptote is capped by the engine: width 8, and 8.0 tokens per pass at perfect survival. The measured pass table
  has 3.0 ms (+6 %) per further position (width 8 to 9 extrapolated), so lifting the cap would add at most a few percent
  at the top of the curve (same result as research/37 section 6).
- Draft tree / width: research/37 has no per-tree pass cost, only chains. An extra position costs 3 to 5 ms in the pass
  table (+7 to +12 % of a 42 ms pass), and on a DeltaNet trunk a branching tree needs the state-reconstruction machinery
  (Bole, ReplaySSM [known]). A sibling at depth 1 can recover at most the 0.149 of passes where the first draft misses,
  for +0.149 tokens (+3.7 % of 3.98) at +10 % pass time: negative. A tree needs a second drafter with calibrated
  top-2 mass to be worth more than that; not available.
- Against the noise: every cell above +2 % is outside the 0.6 % worst-case sd; the bound is not limited by noise.

## 4. What the bound needs the drafter to do

The bound is large; what it requires is not. At 1.15 ms the step must be the MTP head's own shape. A different
architecture at 2 ms must raise survival by 0.08 at every depth including depth 1, where conditional acceptance is
already 0.85 (a tuned head would have to reach 0.93 at depth 1 for +0.08 of S_1 alone) and where on our model the
published third-party drafters measure below or equal to MTP (DSpark, DFlash2). No source in section 1 reports such a
jump over a native head.

## 5. Realistic uplift and what a retrain would cost

Realistic uplift of a retrained MTP head: FastMTP's gain came from fixing a head that was untrained for depth (10 %
at depth 2). Ours is at 0.85 to 0.93 conditional; the distillation papers quote +7.5 % acceptance (MTP-D) and +8 to +20 %
(HASS over EAGLE-2), relative values on weaker starting points. If a retrain gave +0.02 conditional at every depth
(optimistic) it is +6.9 % in the model; a more likely +0.005 to +0.01 gives +1.7 to +3.4 %, which sits at the 2 %
band the project used for the draft-length analysis (research/37) and is unmeasurable without a quantised-model run of
several rounds. The only way to find out is to train; a held-out acceptance measurement offline does not need the
engine, but needs the trained head.

Cost of that experiment (this machine vs rented), from our own files and FastMTP's setup:

| Item | Local (RX 9060 XT 16 GB, 64 GB RAM, 9600X) | Rented |
|---|---|---|
| Teacher data from the shipped quantised model (hidden states plus next-token logits per position) | teacher-forced prefill of existing code corpora, rate = prefill rate of the engine (see BENCHMARKS.md), tens of millions of tokens = hours to days of exclusive GPU; self-sampling at decode rate (67-92 tok/s) is 20-50 days for FastMTP's 389K samples | not needed if the local engine is the teacher; data collection stays local either way |
| Training the head | the MTP GGUF is 4.14 GB at Q8; a BF16 training state with optimiser is several times that, does not fit 16 GB VRAM | one 80 GB GPU for the head alone if trunk activations are dumped (disk: tens of GB per 10M tokens at 4096 dims BF16 is an estimate to verify) |
| Fit for the rule "this PC or a short rented run" | no | yes, but with local data collection of hours to days |

## 6. What we would do differently (if the question comes back)

- Test the cheap hypothesis before any training: measure per-position **top-2 mass** and the conditional acceptance
  of the head on teacher-forced code, offline, from the production draft log (`TONA_DRAFT_LOG` records hits per depth;
  no GPU needed to analyse it). If depth-3 to 7 conditional acceptance is already 0.89 to 0.93, the head's remaining
  error is mostly irreducible and retraining cannot reach +0.02.
- If a retrain is ever run, make the objective **tokens per ms at our cost model**, not acceptance: train with the
  verify-time table so deeper positions are not rewarded where the pass cost (3 to 5 ms per width) makes them
  negative. The adaptive-depth survival curve overstates deep survival through selection; collect the curve at fixed
  depth 7 (teacher forced) so the training set has the same positions as the bound.
- Train against the **quantised** trunk (1-3 bit): the shipped model's logits differ from the BF16 teacher the vendor head
  saw, which is the only plausible source of a gain over a native head at all.
- Keep the head shape and the 1.15 ms step. A different architecture is closed by section 3 (break-even +0.08 at 2 ms,
  none at 6.3 ms).
- Stop reading acceptance length as the result. The EAGLE-3 vs MTP card above shows lower acceptance but higher
  throughput in a framework where the head costs more; here only tokens per ms counts (research/37, 56).
- Orthogonal and already ahead of any new drafter: suffix/history drafts for code (Baseten +34 % acceptance on code
  edits [known]; our own emission-form simulation showed no gain in the corpus, 2026-10-24-suffix-policy), trimmed draft LM head
  (#31031-style, +2 to +9 %), and fixing the verify pass per token (3.4+ ms per token, research/37 section 4), which moves the whole curve.

Coverage state of the scan used here: arXiv 2026-11-11..14 complete (13 requests, 61 hits, only 2 tagged `speculative`,
neither a drafter); 2026-11-15 was too young for the API to list it. The GitHub run for 2026-11-12..14 was stopped before
the PR listing finished (the issue queries were slow), so tracker status above comes from papers.md and web search, not from
a complete scan. The coverage book was not touched; the weekly search owner should close 2026-11-11 to 2026-11-15 there.

## 7. Proposed rows for research/papers.md

```
| 2026-03 | [MTP-D, arXiv 2603.23911](https://arxiv.org/abs/2603.23911) (abstract, 6 auth.) | Self-distillation of MTP heads, +7.5 % head acceptance, looped extension of heads; "minimal extra training cost", no figure | medium: the only method that keeps the MTP step; our head is at 0.85-0.93 conditional, headroom small (research/59) | new |
| 2025-09 | [FastMTP, arXiv 2509.18362](https://arxiv.org/abs/2509.18362) (HTML read, 10 auth.) | Training facts: 389.4K self-distilled samples, 3 epochs, MiMo-7B-RL, head 210.8M, one H20 server under 1 day; acceptance k=1/2/3 70/10/0 -> 80/56/36 % | medium: shows the recipe and the cost; our head starts at 0.85/0.845/0.866 conditional | read |
| 2025-03 | [EAGLE-3, arXiv 2503.01840](https://arxiv.org/abs/2503.01840) (abstract) + Qwen3.5-397B community head ([card](https://huggingface.co/AQ-MedAI/Qwen3.5-397B-A17B-eagle3)) | Up to 6.5x vs none, 1.4x vs EAGLE-2; on Qwen3.5-397B acceptance length below native MTP on every listed benchmark (MT-Bench 3.03 vs 3.28), throughput near MTP | high as a yardstick: confirms native MTP is the strongest cheap draft on Qwen | read |
| 2024-08 | [HASS, arXiv 2408.15766](https://arxiv.org/abs/2408.15766) (abstract) | +8-20 % over EAGLE-2 (four LLaMA), harmonised objective and context, no inference cost | low: no comparison with a native MTP head | read |
| 2025-02 | [GRIFFIN, arXiv 2502.11018](https://arxiv.org/abs/2502.11018) (abstract) | Acceptance length +8 %, speedup +7 % over best prior; masks misaligned tokens | low | read |
| 2025-04 | [PARD, arXiv 2504.18583](https://arxiv.org/abs/2504.18583) (abstract) | Small AR model adapted to a parallel drafter, 3x training efficiency; vs AR drafters only | low | read |
| 2024-12 | [Falcon, arXiv 2412.12639](https://arxiv.org/abs/2412.12639) (abstract) | Semi-autoregressive drafter, 2.91-3.51x on Vicuna/LLaMA2-Chat | low | read |
| 2026-11-16 | research/59 (own arithmetic) | Bound: survival +0.05/+0.1/+0.2 at 1.15 ms = +6.7/+13.5/+25.9 %; at 2 ms break-even +0.08, at 4 ms +0.26; no published drafter shows a gain over a native Qwen MTP head | verdict: drafter architectures closed, MTP-head retrain stays a maintainer decision | tested (desk) |
```
