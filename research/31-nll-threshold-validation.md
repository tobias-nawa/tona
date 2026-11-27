# 31 — Is 0.01 nats per token a sensible quality threshold? Literature and an experiment plan

Agent research (Sonnet), 2026-10-29

Method: web search and page retrieval (arXiv abstracts/HTML, GitHub PR page). Nothing was measured or built here, no
engine code changed, no GPU used. Marking per entry:
- **[HTML]**: paper HTML read (numbers are the authors').
- **[abstract]**: abstract page only.
- **[PR page]**: PR page read.
- **[snippet]**: only a search-result snippet seen.
- **[estimate]**: own calculation, not measured.
- **[unverified]**: could not be confirmed (e.g. a licence).

Units: Δ in nats per token = ln(PPL_new / PPL_old). KL is in nats per token. Accuracy differences are in percentage
points (pp).

---

## 0. Summary

1. **There is no published mapping "Δ-NLL of x nats -> y pp of code accuracy".** The literature has (a) quantization
   studies with accuracy and sometimes KL/flips, (b) sparse-attention papers with perplexity OR tasks, almost never
   both with numbers in one table. The 0.01 nats limit is therefore not contradicted and not supported by direct
   evidence; it is a detection limit that happens to be small.
2. **Evidence that 0.01 is conservative for short-context quality:** at mean KL 0.006 (BnB W8A8, Llama2-13b) MMLU
   changes by -0.10 pp, at KL 0.02 (GPTQ W4A16) by -0.14 pp (2407.09141 [HTML]). In a 2026 cohort study, quants with
   WikiText KL up to 0.064 (Qwen3.6-35B-A3B) and 0.027 (Devstral-Small-2-24B) show *no* significant correlation between
   KL and benchmark score, including HumanEval and LiveCodeBench (2606.19558 [HTML]). Our patterns have KL 0.003-0.045
   (indexcache.md), the same range.
3. **Evidence that the mean can hide the relevant loss for long context:** plain PPL correlates -0.18 / +0.24 / +0.27
   with LongBench / LongEval / RULER, a key-token variant (LongPPL) -0.96 / -0.90 / -0.90 (2410.23771 [HTML]); an L2-norm
   KV-eviction paper keeps PPL unchanged at 50 % eviction while needle retrieval fails above ~30 % (2406.11430
   [snippet]). Our texts are mostly locally predictable code, so a mean over all tokens dilutes exactly the few tokens
   that need cells beyond the 2,048 selected. A retrieval loss of a few percent of tokens could hide under 0.01.
4. **Mean KL and mean Δ-NLL are not better than each other for near-baseline checkpoints**: both lose predictive power
   inside the "silent zone" (rho -0.24 to +0.00 for KL, p99 KL -0.11/-0.21, top-1 +0.25/-0.04). They discriminate
   damaged from undamaged, not between small differences (2606.19558 [HTML]). Reading: the 0.01 rule is a good
   *screen for gross damage*; it cannot rank a 0.002 against a 0.008 change, and no metric can.
5. **Recommendation:** keep 0.01 as the cheap screen, but (i) add a key-token / tail metric for long-range tokens,
   (ii) validate with a dose-response run on an executable long-context code benchmark built on one shared context
   (section 3), (iii) read the limit off the curve (section 4). Whole plan fits in one GPU day (section 3.4).

---

## 1. Literature: delta metric against downstream accuracy

### 1.1 Quantization and compression (flips, KL, accuracy)

| Source | Setting | Metric pair | Reading |
|---|---|---|---|
| 2407.09141 "Accuracy is Not All You Need" [HTML], NeurIPS 2024 | Llama2-7b/13b/70b chat, Yi, MMLU 5-shot, 16-bit baseline | Table 5, accuracy change pp / flips %: Llama2-7b SQ W8A8 -2.36 / 13.62; GPTQ W4A16 -0.58 / 8.26; BnB W8A8 -0.30 / 4.15; GPTQ W8A16 +0.08 / 0.60. Llama2-70b GPTQ W4A16 -0.45 / 4.26 | Accuracy moves < 2.4 pp while 0.3-14 % of answers flip. Flips are 4-30x larger than the accuracy change; flips cancel. |
| same | Llama2-13b, Wiki-2 | KL: GPTQ W4A16 0.02, GPTQ W8A16 7.9e-5, BnB W4A4 0.03, BnB W8A8 5.8e-3; perplexity 4.6-4.7 for all four | PPL cannot separate them, KL can. MMLU Spearman(KL, flips) = 0.981. |
| same | MT-Bench (free-form), Llama2-70b | GPT-4 score 7.431 -> 7.059 (BnB W8A8, -5 %), 6.801 (GPTQ W4A16, -8 %) while MMLU changed -0.55 / -0.45 pp | Free-form generation reacts more than multiple choice. Relevant: our tasks are free-form code. |
| same | GSM8K | flips 12-30 % for BnB W8A8/W4A4 [HTML] | Long generated chains flip far more than single-token answers. |
| 2411.02355 "Give me BF16 or give me death" [HTML] | Llama 3.1 8B/70B/405B | HumanEval+ pass@1 8B: BF16 60.7, W8A8-INT 60.0, W4A16 59.1 (-1.6 pp, 97.4 %); 70B 74.8 -> 74.2 (W4A16); 405B 80.1 -> 78.9 (-1.2 pp). RULER 8B 82.8 -> 81.1 (INT4, -1.7 pp) | 4-bit weights cost 0-1.7 pp on code and RULER at 8B; no KL reported. Their HumanEval has 164 problems: 1 problem = 0.6 pp, so the differences are 1-3 problems and not significant. |
| 2504.04823 "Quantization Hurts Reasoning?" [HTML] | DS-R1-Qwen 1.5B-32B, QwQ-32B, Qwen3-8B, LiveCodeBench | W4 AWQ -1.4 to -2.5 pp; W3 AWQ -5.6 to -21.9 pp; KV4 about 0; KV3 -4.4 to -13.7 pp; W4A4 -1.7 to -24.3 pp. No perplexity. | Typical cliff: 4-bit nearly free, 3-bit costs 5-20 pp. The cliff is steep, so a limit anywhere in the flat part is cheap to hold. |
| 2606.19558 "Displacement Is Not Direction" [HTML], 2026-06 | 28 quants of Qwen3.6-35B-A3B, 41 of Devstral-Small-2-24B; HumanEval, LiveCodeBench, GSM8K, MATH-500, MMLU, IFEval, BFCL | Full cohort Spearman(KL, composite) -0.72 / -0.86. "Silent zone" (KL <= 0.064 Qwen, <= 0.027 Devstral, WikiText c512): rho +0.00 / -0.24 (p = 0.36). Per-benchmark silent-zone rho for HumanEval +0.20 / -0.14, LiveCodeBench +0.08 / -0.20. Quants with >= 99 % of the BF16 composite reach KL 0.094 / 0.060. | The closest direct evidence for code: below KL ~0.03-0.06 no measurable relation to code scores. Cutoffs are cohort-specific "descriptive, not general decision rules". Benchmark sizes and noise floors are not reported (a weakness). |
| same | per-prompt LiveCodeBench, 5 quants | geometric-mean KLD ratio failed/passed prompts 1.08-1.22; routing by lower KLD wins 42-49 % of disagreement prompts (163-190 prompts) | A prompt's KL says almost nothing about whether *that* prompt fails. Only aggregates carry signal. |
| 2607.08734 "Illusion of Equivalency" [abstract] | 8 to 2 bit, ARC/HellaSwag/Winogrande | "behavior shifts even when accuracy and perplexity are preserved"; proposes Correctness Agreement | Same message as flips; no numbers seen. |
| 2310.01382 LLM-KICK [abstract] | pruning and quantization | "perplexity fails to capture subtle changes"; pruning degrades downstream at 25-30 % sparsity despite negligible PPL change | Older evidence that PPL is blind for knowledge tasks. |
| llama.cpp PR #5076 "KL-divergence" [PR page], merged 2024-01-22 | Mistral-7B k-quants | Author: `ln(PPL(Q)/PPL(fp16))` and KLD "track closely"; Q4_K_S Bagel-34B showed ~7 % *lower* PPL than fp16, i.e. PPL meaningless there; KL percentiles and same-top still decreased consistently. A 20B Q2_K_S: same-top 0.74-0.83, mean KLD 0.36, median 0.18, q99 2.6 | Supports tail/percentile reporting (q99, same-top) in addition to the mean. Follow-up: `perplexity` tool statistics (PR #6936). No accuracy link in the PR. |

Reading for the threshold. On quantization, KL 0.006-0.03 per token costs 0.1-0.6 pp on multiple choice with 3-8 %
flips; 0.01 nats of Δ-NLL is of the same order as the *noise* of these studies. That the rule is 0.01 and not 0.03 is
safe on this evidence for short-context quality. This evidence does not transfer to sparse attention, see 1.3.

### 1.2 Mean versus key tokens for long context

- LongPPL (2410.23771, ICLR 2025 [HTML]): Pearson with LongBench / LongEval / RULER: PPL -0.18 / 0.24 / 0.27 (wrong sign),
  LongPPL-hard -0.96 / -0.90 / -0.90. Key tokens = tokens whose log-probability rises by a threshold when the long
  context is present. LongEval example (Table 9): Yi-6B-200K, answer-token PPL 2.29 vs 2.28 but accuracy 54 % vs 60 %.
  Caveat: correlations are across *different models*, not across perturbations of one model.
- Quest (2406.10774 [HTML]): the paper itself says that language-modelling evaluation "only involves local
  dependencies", and reports retrieval instead: passkey at 100K with 2,048 pages 100 % (Quest) vs 2 % (H2O, TOVA,
  StreamingLLM); LongBench lossless at 1/5-1/10 of the cache depending on the task (budget about 11K tokens).
- L2-norm KV compression (2406.11430 [snippet]): PPL unchanged up to 50 % eviction, needle accuracy holds only to ~30 %.
- Our own data point (indexcache.md): the control with only layer 0 full gives 22/24 single retrieval at 128K while the
  rule-passing pattern (d) gives 24/24; the NLL checks were not run for that control, so we do not know whether
  Δ-NLL would have flagged it. This is a cheap thing to measure first (step 1 below).

### 1.3 Sparse attention papers: what they report

| Source | PPL | Tasks | Usable pair? |
|---|---|---|---|
| IndexCache 2603.12201 [HTML] | LM loss only as search objective, no PPL table | 30B DSA model, Table 2: original Long Avg 50.2, LiveCodeBench 71.4, RULER 87.9. 1/2 searched 50.3 / 71.3 / 87.8; 1/4 searched 49.9 / 70.0 / 87.6; 1/8 searched 46.1 / 69.6 / 82.0; 1/4 uniform 43.0 / 69.4 / 79.2; 1/8 uniform 35.3 / 58.7 / 68.8 | No NLL given. Code (LCB) is the *least* sensitive column: uniform 1/4 loses 2.0 pp on LCB but 7.2 on Long Avg and 8.7 on RULER. So long-context retrieval, not code generation, is where sharing hurts first. |
| DeepSeek-V3.2-Exp (DSA) [snippet] | not found in snippets | "on par" with V3.1-Terminus on public benchmarks, e.g. MMLU-Pro 85.0 both, GPQA-Diamond 80.7 vs 79.9 | No pair; the report PDF (github.com/deepseek-ai/DeepSeek-V3.2-Exp) was not read. |
| NSA, Quest PPL tables | not retrieved (search gave nothing usable) | Quest: see 1.2 | Open: read NSA (2502.11089) and the DSA report for loss-vs-benchmark curves. |

Result: **no sparse-attention paper gives a Δ-NLL / task table that could calibrate our limit.** This is the gap our
experiment would fill, and it is publishable in the engine's docs as "our measured curve".

### 1.4 What predicts task loss best

- **Task-level flips (or correctness agreement) track behaviour; mean accuracy hides it.** Flips are 4-30x the
  accuracy change (1.1). But flips are *not* quality: they are dominated by cancellation. For a decision rule they
  are an upper bound on change, accuracy delta is the lower bound.
- **Mean KL vs mean Δ-NLL:** same information at small values (PR #5076: `ln(PPL ratio)` tracks KLD). KL is
  non-negative and sees changes that cancel in Δ-NLL (2407.09141 Fig. 9 case: PPL equal, KL 5.8e-3 to 0.03). Δ-NLL has
  the advantage of a sign and a paired bootstrap with small variance (own data: sd ~0.275 per token).
- **Tail metrics:** p99 KL had full-cohort rho -0.75 / -0.86, silent zone -0.11 / -0.21, i.e. not better than the mean
  there (2606.19558). Top-1 agreement is no better near baseline (rho -0.04 / +0.25). Only in the damaged region do all
  of them work. Nobody has shown "fraction of tokens with KL > x" to predict code pass rate; [unverified claim in
  the other direction, none found].
- **For long-range loss specifically:** key-token restricted NLL (LongPPL idea) is the only metric with published
  evidence of tracking long-context benchmarks (-0.96). Cheap for us: the tail run already has per-token NLL for the
  reference; add a second reference run with the context truncated to 2,048 (the QSA threshold), mark tokens where
  the long context lowers NLL by >= 2 nats (LongPPL threshold, [estimate] adopt as is), and report Δ-NLL on that
  subset. Expect 1-3 % of tokens; the bootstrap interval will be ~5-10x wider, so it needs more tail tokens or
  texts with real long-range structure (repeated identifiers, calls to functions defined 50K tokens earlier).

Conclusion for question 2: no metric dominates near baseline. Ranked by evidence: (1) task accuracy paired with
McNemar on long-context tasks, (2) key-token Δ-NLL, (3) mean Δ-NLL ≈ mean KL (screen), (4) top-1 / tail KL (only
as a damage alarm).

---

## 2. Is 0.01 sensible? Assessment

- **As a detection limit:** yes, it is what 10K tokens resolve (sd 0.275 / sqrt(effective n); with 64-token blocks the
  interval half-width in indexcache.md is about 0.002-0.01 for 4,096 tokens, which means 0.01 is at the edge for
  the 4,096-token tail sets). Our (a) intervals are +0.0049 [0.0002, 0.0098]: the rule rejects it on the interval
  excluding 0, not on size.
- **As a quality threshold:** weakly supported. In the quantization literature, effects of that size are far below task
  noise (section 1.1). For sparse attention the harm is concentrated on few tokens (1.2), so a mean of 0.01 could be
  composed of 0.5 nats on 2 % of tokens (the retrieval tokens), which is exactly a visible retrieval loss. The rule
  is neither shown too strict nor too lax; it is unvalidated for the failure mode that matters.
- **Rule 3 "interval contains 0":** with a rule that fails a pattern whose interval merely excludes 0 at +0.004, the
  gate rejects effects 2.5x below the stated limit. If the limit is "0.01", the clean statement is "upper bound
  < 0.01", and the "contains 0" clause can be dropped; if the aim is "no measurable change", say so. Choose one.
  (Proposal only, `docs/backends/README.md` is not changed here.)

---

## 3. Benchmark for a dose-response measurement

### 3.1 Requirements from the platform

- Context >= 32K (QSA selects nothing below 2,051 positions; indexcache.md: the 23 tasks "are mostly shorter than
  2,051 positions").
- IE_IDX_SHARE acts only in decode: a task whose answer is 1-3 tokens exercises it for 1-3 passes. Answers should be
  50-300 generated tokens, or the question tokens must be run in the decode path (teacher tail mode).
- One day of GPU. Prompt cache (glossary: checkpoints in RAM and SSD, reused for a shared prefix): a design with
  one shared long context means prefill is paid once (128K at ~1,600 tok/s prefill = ~80 s; indexcache.md measured
  92.7 s at 128K), each task then costs only its short prefill plus decode.
- Automatic check, locally runnable, deterministic.

### 3.2 Candidates

| Candidate | Size | Context | Scoring | Shared context? | Licence | Verdict |
|---|---|---|---|---|---|---|
| LongCodeBench 2505.07897 [HTML] (HF `Steefano/LCB`, code `Zteefano/long-code-bench`) | LongCodeQA 443 multiple-choice from 98 repos; LongSWE-Bench 600 from 61 Python repos | 32K-1M buckets | QA: multiple choice accuracy; SWE: run unit tests in Docker per instance | No: each instance has its own repo files plus distractors | HF licence [unverified]; paper CC0 | Contexts differ per task, so prefill is paid per task (128K = 80+ s). LongSWE needs a patch of tens of lines and Docker per instance; reference models score 0-29 % at 32K (Qwen2.5-14B: 0 %), a 1-bit MoE will be near the floor, f tiny. LongCodeQA answers are one letter, decode-only part negligible. Not suitable. |
| RepoBench 2306.03091 [abstract] (`Leolty/repobench`) | thousands per setting | 2K-~24K in the original (not >= 32K) [unverified] | exact match / edit similarity of one line | per-task contexts | [unverified] | Too short, one-line answers. Not suitable. |
| Long Code Arena 2406.11612 [snippet] | six tasks; project-level completion split by context size | up to ~ tens of K [unverified] | mixed (CI repair tested, others metric/LLM-judge) | no | [unverified] | Mostly not exact/executable. No. |
| RULER 2404.06654 (NVIDIA/RULER, Apache-2.0 [unverified]) | generated; any n | any (32K-128K) | exact-match string | one context, several questions in the multi-key/multi-query tasks | | Synthetic, not code, easy for a retrieval model (our retr.py was 24/24). Good as a *sensitivity* control, not as the code curve. |
| SWE-bench Lite (300) / Verified (500) | 300 / 500 | repo context via agent or retrieval, 10K-100K+ | run tests in Docker | no | MIT | Needs an agent loop (minutes per task), floor-level pass rates for this model class, 300 tasks give 6 pp resolution at best. Not for a one-day dose-response. |
| HumanEval+ (164) / MBPP+ (378) with prepended filler | 164 / 378 | artificial | tests | yes (filler is shared) | Apache-2.0 [unverified] | The task needs no long context; QSA would not hurt it even if retrieval degraded. Measures nothing about the mechanism. Only as a "no collateral damage" check. |
| **Own construction from a foreign repo (recommended)** | thousands generated | 32K / 64K / 128K, one shared prefix | exact match (AST-derived questions) and unit tests (hidden function bodies) | **yes by construction** | repo's licence (pick MIT/BSD/Apache) | See 3.3. |

### 3.3 Recommended design: one repository in context, two question families

Pick one permissively licensed Python repo with a good test suite that the model has not seen verbatim as an
"evaluation" item (any public repo is probably in the training data; use one that is *not* in our calibration texts: py
= Python stdlib, cpp = llama.cpp, sh = system scripts) and concatenate it to 128K tokens (also cut variants 32K /
64K for a context-length axis). Context is identical for all tasks, so the prompt cache holds it.

**Family A, retrieval questions with exact answers (cheap, thousands, generated by AST):** "In which file is `X`
defined?", "What is the default value of parameter `p` of `f`?", "Which function calls `g` in module `m`?", "What
does `CONST` equal?", "Copy lines of function `h`". Exact-match scoring after normalisation, 5-60 generated tokens.
Pick questions whose reference answer is correct ~50-80 % of the time (calibrate on the reference first) so that f
is maximal; drop questions the reference gets 0 % (no information) or 100 % at 2K context (do not need QSA).
Also produce the control: the same question with the relevant file placed within the last 2K tokens (local) -> must
be unaffected by IE_IDX_SHARE, proving the effect is long-range.

**Family B, hidden-body completion with the repo's own tests (code, executable):** remove the bodies of 100-200 chosen
functions from the shared context once (leave signature + docstring + `...`), then ask "write the body of `f` in
`m`". Check by substituting the generated body into the repo and running the covering tests (seconds on CPU,
parallel to the GPU). Context identical for all tasks, 100-300 generated tokens per answer. This is "repo-level
function completion", the same task type RepoBench/ClassEval-style benchmarks use, but with 128K context and one
prefix.

Cost per task [estimate]: question prefill ~300 tokens at ~1,600 tok/s = 0.2 s; decode 150 tokens at ~55 tok/s
(long-context decode, indexcache.md: 45-54 tok/s at 128K-250K) = ~3 s; prompt-cache restore of a 128K checkpoint
per task [unverified, must be measured: host-RAM copy of ~150 MB KV (1,152 B x 12 layers x 128K x ... check) plus
DeltaNet state]. So ~3.5-4 s per task, ~1,000 tasks = ~1 h per condition, i.e. 5-6 conditions in one day on a
single GPU. Teacher-forced NLL runs on the same questions are a by-product (IE_TEACHER_TAIL).

Pitfall: the answer must be verifiable without our model's text style: normalise whitespace, accept any body that
passes the tests; for Family A write the extraction regex before the run.

### 3.4 Power (paired McNemar)

Smallest detectable difference d at 80 % power, two-sided alpha 0.05: d ≈ 2.8 sqrt(f / n), f = fraction of tasks
that flip either way, n = tasks. n = 7.84 f / d².

| f | n for d = 3 pp | d at n = 200 | 500 | 1000 | 2000 |
|---|---:|---:|---:|---:|---:|
| 0.03 | 261 | 3.4 pp | 2.2 | 1.5 | 1.1 |
| 0.05 | 436 | 4.4 | 2.8 | 2.0 | 1.4 |
| 0.10 | 871 | 6.3 | 4.0 | 2.8 | 2.0 |
| 0.20 | 1,742 | 8.9 | 5.6 | 4.0 | 2.8 |

For comparison the published f: 0.3-14 % per MMLU answer (2407.09141), 12-30 % for GSM8K-style multi-step
generation. Greedy code completion with 100+ tokens will be at the high end (any divergence in the token stream
flips the text, only some flip the verdict). With 1,000 tasks at f = 0.10 we resolve 2.8 pp, with 2,000 at
f = 0.20 also 2.8 pp. The unconditional two-proportion test would need 3-5x more tasks; pairing is essential.
Note that the detectable limit moves with f: a *reference vs reference* run (different pass width, MTP on/off,
non-deterministic mode) gives the floor f0; real flips are f - f0.

### 3.5 Pilot to measure f (about 2 h of GPU)

1. Generate 300 Family-A questions and 60 Family-B tasks for the 64K context. Run the reference twice: once as is,
   once with a harmless change that alters rounding but not quality (MTP off or on, `IE_DETERMINISTIC` unset
   against set, or a different pass width). Gives f0 and the reference accuracy per family.
2. Run one rule-passing pattern (d, 8 full layers) and one clearly failing pattern (layer 0 only, "1 full") on the same
   tasks. Gives f for a null change, a mild and a strong dose; the strong one must show a clear accuracy drop (the
   control's 22/24 on retrieval says it will).
3. Keep tasks with reference accuracy between 20 % and 90 % in at least one run (selection on the reference only; never
   on the variants, which would bias the paired test).
4. Report f, accuracy and the McNemar discordant counts (b, c), then compute n for 3 pp from the table.

---

## 4. Experiment plan: dose-response and reading the threshold off the curve

1. **Calibrate the benchmark (pilot above).** Decide n per condition from the measured f. Fix the task list, seeds,
   decoding (greedy), extraction code and the analysis before the first variant run (as for the OEA protocol).
2. **Doses.** IE_IDX_SHARE with 12 (reference), 8 (pattern d), 6 (a), 5 (c), 4 (e), 2, 1 full layers; use the greedy
   or uniform patterns already measured so the NLL numbers exist. Optional artificial dose that is independent of
   the QSA mechanism: temperature-free logit noise (Gaussian sigma on the final logits in the teacher-forced run
   gives a *known* Δ-NLL ≈ small, quadratic in sigma) to test whether the benchmark reacts to a generic
   Δ-NLL of 0.01 / 0.03 / 0.1 at all. If it does not react to a mean-NLL change of 0.03 from noise but reacts to
   the 1-full pattern, then the mean is the wrong axis and only the long-range metric matters. Both are cheap.
3. **Per dose (same tasks):** teacher-forced Δ-NLL (mean and key-token), KL, top-1 on the tail of the same context
   (IE_TEACHER_TAIL), plus greedy task accuracy for both families. One run per dose is enough for the paired test;
   repeat the reference twice (f0) and the pivotal doses (6, 4) three times with different task shuffles.
4. **Statistics:** paired McNemar (exact, mid-p) for accuracy difference with 95 % CI (Newcombe/Wilson or paired
   bootstrap over tasks, clustered by source function/file so correlated questions do not inflate n);
   Δ-NLL by the 64-token block bootstrap as now; Spearman or a monotone fit (isotonic regression / logistic with
   dose) of accuracy loss against Δ-NLL, KL and key-token Δ-NLL across the 6-7 doses, with a bootstrap over tasks for
   the curve. Report which x-axis (mean Δ-NLL, key-token Δ-NLL, KL, top-1 mismatch) predicts accuracy best; the
   literature (1.4) suggests key-token.
5. **Reading the limit:** define the acceptable loss first (e.g. "no more than 1 pp on Family B and 2 pp on
   Family A at the 95 % upper bound", stated with the resolution the pilot gave, e.g. 2.8 pp). The limit is the largest
   Δ-NLL (or key-token Δ-NLL) at which the *upper* confidence bound of the fitted curve stays below that loss.
   Since 3 pp resolution is coarser than "1 pp", the honest result may be "limit between x and y, the benchmark cannot
   refine it"; say so rather than extrapolating. Compare with 0.01: if the curve is flat up to ~0.03 for mean and
   only the 1-2 full patterns lose, keep 0.01 as the screen and add the key-token check; if (a) (6 full, +0.005,
   KL 0.004-0.02) already shows a loss on Family A, the limit must be tightened or replaced.

### Pitfalls

- **Greedy vs sampling.** Greedy is deterministic but a tiny logit change flips the whole continuation (a per-token
  top-1 mismatch of 1-4 % (indexcache.md) makes a 150-token greedy completion differ with probability 78-99 % [estimate:
  1 - (1 - p)^150]); sequence divergence is not quality loss, only the *verdict* counts. Sampling would add noise and
  needs k >= 4 samples per task with pass@k-style averaging; for a dose-response curve greedy plus paired verdicts is
  cheaper.
- **Reference nondeterminism.** Without `IE_DETERMINISTIC=1` wide passes round differently (oea.md: 0.004-0.010 alone vs
  together). MTP/draft acceptance changes pass widths. Fix pass width, or run all conditions through one path, and
  measure f0.
- **Contexts < 2,051 show nothing.** Check per task that the answer-relevant cells are > 2,048 positions away; keep the
  local-control questions to prove this.
- **Contamination / copy shortcut.** A public repo may be memorised, which makes the model answer without the
  context and hides QSA damage. Control: ask the same questions with the repository's identifiers renamed
  consistently (cheap AST rewrite) or use a repo newer than the training cut-off.
- **Task selection bias.** Select by reference behaviour only; never drop tasks after seeing variant results.
- **Variance across prefixes.** One repo = one cluster. Use two repos (different languages if possible, as 2606.19558's
  cohort effects differ per model) before generalising; clustered CIs by file.
- **Ceiling/floor effects.** The IQ1_M model's reference accuracy on hard completion tasks may be low; Family A
  keeps the average near 50-70 %.
- **Retrieval test too easy** (indexcache.md: retr.py 24/24, still found the 1-full control at 22/24): with 100 %
  at the reference there is nothing to lose; Family A deliberately uses harder multi-hop questions (callers of X, defaults).
- **Different quantities in different papers:** KL values from 2407.09141 and 2606.19558 are WikiText with 512-token
  context and a different tokenizer/model; do not compare absolutely with our 4,096-token tails of 128K contexts.
- **Cut-off dates:** the 2026 papers (2606.19558, 2607.08734) are recent preprints; peer review unknown.

---

## 5. Status of open issues / PRs and open points

- llama.cpp PR #5076 (KL divergence): merged 2024-01-22 [PR page]. Statistics follow-up PR #6936: found only in
  a search result mirror, not read.
- 2606.19558 and 2607.08734: arXiv preprints, no review status found.
- IndexCache 2603.12201: code and vLLM docs exist (research/30); no quality-vs-NLL table, as shown above.
- Not read: NSA (2502.11089), the DeepSeek-V3.2-Exp technical report PDF, 2406.11430 full text, LongBench-style
  per-budget tables of Quest. Licences of LongCodeBench data, RepoBench, RULER, EvalPlus unverified.
- Own open points: key-token Δ-NLL tool (needs a second reference run with 2K context); prompt-cache restore cost for a
  128K checkpoint; chosen repository.

## Result

Measured in #76: the first significant accuracy loss (A -3.4 pp, B -8.6 pp) appears at Δ-NLL +0.016 on the 32K texts, 1.6
times the limit of 0.01; the stage below (+0.009) shows none the benchmark can resolve (A 2.3 pp, B 7 pp). On repo-code tails
the Δ-NLL is higher for the same stages and the boltons tail is unusable; "0.01 is too strict on repo code" is not shown.
Details: `research/measurements/2026-10-30-threshold-main.md` (pilot: `2026-10-30-threshold-pilot.md`).

## Sources

- https://arxiv.org/abs/2407.09141 (Accuracy is Not All You Need), HTML https://arxiv.org/html/2407.09141
- https://arxiv.org/abs/2411.02355 (Give me BF16 or give me death)
- https://arxiv.org/abs/2504.04823 (Quantization Hurts Reasoning?)
- https://arxiv.org/abs/2606.19558 (Displacement Is Not Direction)
- https://arxiv.org/abs/2607.08734 (Illusion of Equivalency)
- https://arxiv.org/abs/2410.23771 (What is Wrong with Perplexity for Long-context Language Modeling?)
- https://arxiv.org/abs/2310.01382 (LLM-KICK, Compressing LLMs: The Truth is Rarely Pure and Never Simple)
- https://arxiv.org/abs/2406.10774 (Quest), https://arxiv.org/abs/2406.11430 (L2-norm KV compression)
- https://arxiv.org/abs/2603.12201 (IndexCache)
- https://github.com/ggml-org/llama.cpp/pull/5076 (KL divergence)
- https://arxiv.org/abs/2505.07897 (LongCodeBench), https://arxiv.org/abs/2306.03091 (RepoBench), https://arxiv.org/abs/2406.11612 (Long Code Arena)
- https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp (model card)
