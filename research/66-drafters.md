# 66 - Drafters beside MTP: DNeo and the session-local drafter (#173)

Desk analysis 2026-11-18. Marking as in `research/65-gdn-verify.md`: **[own]**, **[abs]** (abstract read today),
**[known]** (`research/papers.md`). Nothing built beyond what is named.

## 0. Verdict

1. **Session-local drafting is already in the engine and tuned.** `server/suffix.{h,cpp}` (SuffixIndex, arXiv 2411.04975
   style) drafts from the history when the end of the text matched at least `TONA_SUFFIX_MIN` = 8 tokens; the longest match
   wins, **the most recent on a tie** [own]. The paper's two findings are (a) most-recent beats earliest match, which our rule
   has; (b) a minimum match length of 3 does about as well as a fancier session memory, whose weighting and backoff add
   nothing measurable at 30 to 40 times the drafting cost [abs]. Our simulation on 40 agent sessions (1.2 M tokens) found
   minimum 8 the optimum and 4 worse (105.3 vs 108.9 in the time model; 2026-10-24-suffix-policy.md). The paper tested
   Mistral-7B on general, medical and legal text, no code [abs], so it cannot overturn a result on our own corpus.
   **No prototype needed.** The "greedy tokens identical by construction" property holds for every history draft (the verify
   accepts only what the target would produce itself).
2. **DNeo (arXiv 2611.21925) needs a trained drafter.** Abstract only [abs]: Mixture-of-Features Attention encoder, working
   memory of rejected proposals, 2-tap causal convolutions, prefix-only cross-attention, non-uniform target layer selection;
   1.3x faster generation than DFlash-2 at equal size and quality. No training data, compute, weights or code are named on the
   page. Our Coder shard is a 1-3 bit requantisation, and a drafter has to match the trunk's errors (research/59 section 0.4), so
   we would have to train it ourselves, which does not fit this PC. **Not buildable; watch for a release.** Even at +30 % over
   DFlash-2: our DFlash2 path ties MTP at 7 drafts (5.26 vs 5.27 tokens per pass code, 2.88 vs 3.01 text, step 6.3 ms vs 1.15 ms
   [own, research/59]), so the expected gain over MTP is 0 to +10 % before the step cost.
3. **One small open item that is cheap and CPU-only:** the paper's recency idea is covered, but the *combination rule*
   (history draft overrides MTP above a length, or both are verified as one block) is what Baseten reports as +34 % acceptance on
   code edits [known]. We override. A test on the bench set (short single-file prompts, 400 tokens) would show nothing: those
   outputs hardly repeat their own prompt. The place that tests it is the agent corpus (`tona-sfx-policy`, CPU only).

## 1. What each needs

| | Session-local drafter 2611.19455 | DNeo 2611.21925 |
|---|---|---|
| Training | none | trained drafter (target features) |
| Weights | none | not stated |
| Code | github.com/antalvdb/olifant-speculative (named on the page) | not stated |
| Fit in our pipeline | exists: `SuffixIndex` + `Generator::draft` branch `from_sfx` | DFlash2 slot (`dfl_`, `draft_dflash`), needs a new sidecar for our target |
| Output identical to greedy | yes by construction (verify); the paper's 94 % token-for-token is its numerical near-ties on batch changes | same |
| Evidence | Mistral-7B, general / medical / legal text; speedup +0.09 over earliest-match lookup at 500 tokens; with repetition penalty 1.4-1.7x falls to 1.1-1.2x | 1.3x over DFlash-2 (authors, abstract) |

## 2. Decision

- #173 as posed (session-local drafter on the code bench set): **close as already implemented**; measuring `TONA_SUFFIX_MIN=0`
  against default on `bench-set` would measure nothing (single short prompts); the agent-corpus simulation is the right place and
  gave minimum 8 as optimum. Reopen only with an edit-heavy agent trace set (the 40 sessions had 59 Edit calls).
- DNeo: record in `papers.md` as "needs a trained drafter, no weights, step cost unknown; DFlash2 ties MTP here".
- A one-line extension worth a ticket: the policy simulation with a recency-weighted tie rule over the **last N occurrences**
  instead of the single longest match, on the edit-heavy sessions, once such traces exist.
