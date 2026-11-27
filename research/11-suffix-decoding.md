# 11: Model-free drafts for coding agents (SuffixDecoding and relatives)

**Agent research, as of 2026-10-24.** Papers not read in full: numbers come from abstracts,
HTML versions, PR texts and issue comments, not from our own measurements. PR/issue states
queried with `gh` on 2026-10-24. Vendor claims (Baseten, Snowflake, NVIDIA) are marked as such;
independent measurements are the comments in llama.cpp PRs and the papers of third-party authors.
Own simulation: `measurements/2026-10-22-suffix-sim.md`.

---

## Key points

1. **Our plan (a suffix hit overrides the drafter) is state of the art.** TensorRT-LLM
   (`use_sa_spec`, `sa_spec_threshold` = 4, default) and Baseten (suffix automaton + MTP) do exactly that;
   the SuffixDecoding paper itself calls it "hybrid" (fall back to EAGLE-3 when the score is below a
   threshold). Baseten (vendor): +34% acceptance length, +30–33% throughput against MTP alone, code-edit
   dataset only, one model.
2. **The newest and, for us, most important item is AgSpec (arXiv 2610.01108, 2026-10-06, KAIST):** separate the corpus by
   source (session, workspace files, global), index files **in the form in which the agent
   emits them** (diff lines with prefix, escaped version), and choose **draft length cost-aware** from a
   break-even λ plus online scaling. It builds on SAM-Decoding/SuffixDecoding and is 18%
   above the best previous method; a fixed draft of K=32 is ~30% slower than K=8 at batch 16.
3. **Cost-aware length has a closed-form rule** (AgSpec: λ = Ā·c/S). For us c is not constant
   but the measured expert union u(b). The rule then needs the marginal time ΔS(k) instead of
   a fixed c (own derivation below, section 3).
4. **Tool-call syntax as a draft** (vLLM #59712, SGLang #42313, llama.cpp discussion #29405, all open,
   one author): render the chat template twice (with/without an example tool call), use the difference as a draft for
   the scaffold `<tool_call>…`. Cold tool call: acceptance length 1.1 up to 3.3–5.0 (Qwen3-0.6B, RTX 5080,
   small models, one author). For us: nearly free, scaffold tokens have the highest acceptance probability.
5. **Independent practice shows the same pattern as our simulation:** gain only on repetition
   (ggerganov in llama.cpp #19164), unfiltered short drafts cost (dense 42 down to 31 tok/s decode; MoE 185 down to
   110 tok/s decode, llama.cpp #26283) and a minimum match (n_min 3 or 24) is the most important knob.
6. **Hybrid models (DeltaNet):** llama.cpp solves it with checkpoint + recomputation (#19493, merged 2026-04-19),
   "discard the last draft token with low probability" (#22506, merged 2026-04-29) lowers
   the number of rollbacks. For trees: Bole (2608.01651, already in papers.md) and TreeWY (2608.20961).

---

## 1. What is new in model-free drafts (2025/2026)

### 1.1 Methods and status

| Method | Core | Status / source |
|---|---|---|
| **SuffixDecoding** | Suffix tree per request + global tree, score = expected accepted tokens, MAX_SPEC = α·match length, fall back to model drafter ("hybrid"). ~20 µs per draft token, no GPU | arXiv 2411.04975 v3 (NeurIPS 2025 Spotlight). Paper numbers (authors): AgenticSQL up to 5.3× (EAGLE-2/3 2.8×), SWE-Bench 1.8–4.5× against no speculation, 1.3–3× against prompt lookup; on Spec-Bench alone worse than EAGLE, hybrid 2.5× against 2.4× |
| **vLLM `method="suffix"`** | Integration of Arctic Inference | PR #25784 merged 2025-11-03 (Snowflake author). Spec-Bench tables there: suffix beats ngram almost everywhere, "larger gains on real agent requests" (vendor claim, no numbers for agents). Docs from v0.19 |
| vLLM suffix on GPU / Model Runner V2 | GPU speculator without Arctic, with async scheduling | **#60339 open** (as of 2026-10-20): acceptance 1.937 tokens/step against Arctic 1.923 (k=8) and 1.948 (k=24) on a 21K-request replay; **#52097 open** (2026-10-09, `suffix_gpu`); vLLM default for `num_speculative_tokens` with suffix: 8 instead of 24 |
| **SAM-Decoding** | Suffix automaton instead of tree, exactly longest match, O(1) per step, static + dynamic, combinable with EAGLE-2 | arXiv 2411.10666 (ACL 2025). v3: +3.3–11.1% from combining with EAGLE-2, >18% faster than other retrieval methods on Spec-Bench |
| **Baseten sa_spec** | SAM on the GPU, built during prefill, integrated in TRT-LLM | Blog 2026-01-23 (update 2026-05-05), vendor. See 2.1 |
| **llama.cpp ngram-mod** | Hash pool (LCG) n-gram → next token, ~16 MB, shared across all slots | PR #19164 merged 2026-01-30 (ggerganov). Recommendation `n=24`, `--draft-min 48`, `--draft-max 64`; "MoE needs long drafts" (remark by the author, not our measurement) |
| llama.cpp ngram-map-k/k4v, ngram-cache | n-gram → up to 4 continuations, most frequent | merged, k4v fix #28431 open |
| **llama.cpp suffix tree** | Tree of the running request only, no global tree | **PR #26283 open** (kashif, 2026-07-29, as of 09-07, review pending) |
| llama.cpp ngram-mod with score pruning | Hits +, misses −, remove weak n-grams after a streak | #19294 open (since 2026-02-03, PoC) |
| **SGLang NGRAM** | Trie + external corpora (SAM), CUDA only, async scheduler off | Docs; open PRs below (section 4) |
| Cacheback | LRU tables of n-grams | arXiv 2511.21699 (EMNLP 2025), no numbers in the abstract |
| ReSpec ("When, What, How") | Retrieve only at low entropy, several candidates, verify retrieval drafts loosely | arXiv 2511.01282, 2025-11-03: +33% against EAGLE-2, +25% against SAM-Decoding on Spec-Bench |
| Goose | Asymmetric tree: reliable n-gram tokens as a deep chain, uncertain ones as wide branches | arXiv 2604.02047 (2026-04-02, v2 2026-08-10): acceptance of the two training-free sources differs by a median of 6×; 1.9–4.3× lossless on 7–33B (dense models) |

### 1.2 Specifically for agents, coding, edits

- **AgSpec (arXiv 2610.01108, 2026-10-06, KAIST; 3 quantized models: Devstral-24B, Gemma3-27B,
  Qwen3.6-27B; SWE-bench Verified and TeamBench, 30 instances each):**
  - Three separate corpora: **session**, **workspace** (opened repository files) and fixed **global**.
    The longer match wins, ties go to the session, global only with a lead of δ = 5 tokens.
  - **Emission form:** On first opening a file, two copies go into the workspace corpus (every line
    with a space prefix, every line with a minus prefix, i.e. diff lines) plus the raw version; for whole-file tool calls
    one copy with escaped newlines and quotes. Effect: +13–39% acceptance length for the coder
    agents in round 1, ≤ 3% in round 5 (the session then contains the form anyway).
  - Session corpus instead of a corpus per call: **acceptance length ×1.6–2.1**. Workspace on top: +2.2–7.0%.
    A workspace corpus growing across sessions: no measurable benefit.
  - Throughput against no speculation 2.27–4.37× at batch 1, 1.08–4.76× at batch 16; on average +18.0% against the
    fastest previous method (with SuffixDecoding +16.5%, with SAM +64.6%). Retrieval overhead 1.1–5.5% of the
    step time. LiveCodeBench 4.87–7.94×, Terminal-Bench +9.3–27.2% against previous methods.
  - **Assessment:** The comparison includes EAGLE-3, REST, PLD, FastCoder, SuffixDecoding, SAM-Decoding. The
    models are dense and the runs small (30 instances, uncorrected significance tests, as stated by the authors).
    No MoE, no offload, no hybrid model.
- **Tool-call scaffold:** vLLM `ngram_hint` (**#59712 open**, 2026-10-02, as of 10-19, merge conflicts, one
  contributor), SGLang **#42313 open** (10-03, as of 10-14), llama.cpp discussion #29405 (2026-09-25).
  Measurements by the authors (small models, no agent): vLLM Qwen3-0.6B 20-token tool call, acceptance 22% →
  89% (16 of 18 drafts); SGLang RTX 5080: 18 → 4 verification steps (acceptance length 1.11 → 5.0), second tool
  3.29, because all hints up to `{"name": "` are the same; llama.cpp Qwen3-14B: 20 → 5 steps, 220 → 69 ms,
  prose unchanged (0 drafts). The authors themselves note: with many tools the prefix matches only up to the
  name, stopping the draft there would be the next step; with a thinking block in the template the hint fails early
  (not measured).
- **SGLang #45054 open** (2026-10-19): every request can draft from its **own prompt**
  (`--speculative-ngram-prompt-sam-budget`, 32,768 most recent tokens), because so far only the last
  `max_trie_depth` input tokens land in the trie. Motivated with "editing code, rewriting text, quoting".
- **Prompt lookup as "speculative edits"** (code repair, arXiv 2604.05963, 2026): acceptance rises when
  the output deviates little from the input code. "Speculative Decoding: Performance or Illusion?" (arXiv 2601.11580,
  MLSys 2026): n-gram is weak on most loads, except for code edits.
- **Tool wait time instead of tokens:** SPORK (arXiv 2607.03333, 2026-07-03) guesses the next tool name with a probe
  (74.6–99.6%) and uses rejected probes as a draft; OoO-Spec (2608.00814), AOSpec (2608.00881), Speculative
  Macro Commit (2609.03236) work at the agent level, not the token level. Less relevant for B=1 with a local tool run,
  read as title/abstract only.
- **AgentSpec (arXiv 2608.24004, 2026-08-25, EMNLP 2026):** drafts only within semantically coherent
  sections of the agent workflow; targets batch inference, not B=1.
- **SSR (arXiv 2608.20359):** suffix cache from the partial thinking; stated 24.1% less latency; purely a
  reasoning topic. The arXiv ID (2608) does not match the submission date 2026-06-17 on the page: ID and date
  unresolved.

---

## 2. Combining model-free with a drafter

### 2.1 Patterns

| Pattern | Where | Numbers |
|---|---|---|
| **Threshold on match length, hit overrides drafter** | TRT-LLM `use_sa_spec`, `sa_spec_threshold` = 4 (default); Baseten sa_spec; SuffixDecoding paper ("hybrid" above a score threshold) | Baseten (vendor): DeepSeek-V3.1-NVFP4, `glaiveai/code-edit-samples`: acceptance length +34%, throughput per request +30–33% against MTP alone; "in production with coding agents up to 40%" (not substantiated). SA acceptance length on code with long context 10+, MTP 2–4. On thinking/prose acceptance ~0. Zero extra cost measured by setting the threshold to ∞ and latency stays equal to MTP (vendor, Nsight profile) |
| Hybrid from the paper | SuffixDecoding v3 | Spec-Bench hybrid ~2.5× against EAGLE-3 2.4× (authors) |
| **Both running together** | llama.cpp `--spec-type draft-mtp,ngram-mod` (command line, discussion #27154, Strix Halo, Qwen3.8-27B) | independent user: code trace acceptance ~0.96 at `n-max 12`, `ngram-mod n-min 24`; cold 59.7 tok/s decode against MTP n=9 55.7; "warm 148–163" is a repetition artifact according to the user. Heavy-reasoning traffic: acceptance below 50%, n-max 12 is slower there than n-max 2 |
| **Tree from both (asymmetric)** | Goose (arXiv 2604.02047) | Chain of the reliable source, wide branches of the uncertain one; provably at least as many tokens/step as either source alone (dense models) |
| Entropy as a switch | ReSpec (2511.01282) | Retrieve only at low uncertainty; verify retrieval drafts loosely (lossy, not directly usable for us) |
| Choose source with a lead | AgSpec | Session before workspace before global, global only with δ=5 |

### 2.2 Conclusion for us

Our plan matches the TRT-LLM pattern. Open parameters: the **threshold** (TRT: 4; llama.cpp ngram-mod: 24 for
long drafts; user in #26283: n_min 3 restored 41 tok/s decode dense) and **what happens to the
DFlash2 draft on a hit**. Because of block diffusion (one pass, 3 tokens fixed), the DFlash2 run costs 0.65 GB
of reads even when the suffix hit overrules it: on a hit ≥ threshold, **do not start the drafter pass
at all** (cf. Baseten: there MTP runs along in parallel without extra cost because MTP is cheap). This is a
difference from our situation (own consideration, not in the sources).

---

## 3. Cost-aware draft length

### 3.1 AgSpec rule (arXiv 2610.01108)

- Draft length per step: `K_i = clip(round(α^(a_i) · m_i), 1, K_max)`, m_i match length, α online scale per agent.
- Upper bound offline: `K_max = max{k : Pr(R ≥ k) ≥ λ}`, R = reusable length, measured by replaying
  recorded trajectories.
- **Break-even: λ = Ā·c/S** (S step time of the target model, c = dS/dL marginal cost of one more draft
  token, Ā mean accepted tokens per step). Measured λ: 0.022 (Devstral/TeamBench) to 0.299
  (SWE-bench, Devstral).
- Online: `α ← α + η(τ − 1[A_i < D_i])`, τ = 1 − λ, η = 0.5, α₀ = 1, α in [0.05, 12]; the update is skipped when a
  fully accepted draft reaches K_max (censored observation).
- Effect (batch 16, Devstral): fixed K=32 discards 11–26 tokens/step, AgSpec at most 6; the online scale alone
  reduces the discarded share from 78% to 50% and raises throughput by 66%.

### 3.2 Derivation for MoE (own consideration, not from a source)

Throughput = A/S. One more draft token k with acceptance probability p_k (cumulative: all previous ones
accepted) pays off if p_k/ΔS(k) > A/S, i.e. **p_k > A·ΔS(k)/S** (= AgSpec's λ with c = ΔS). For us
ΔS(k) = (u(b+1) − u(b))·0.98 GB/rate, hence **decreasing in k** (the union grows sublinearly) and strongly dependent on the
expert cache hit rate. We have already measured u(b) (`2026-10-22-expert-routing.md`); the
rule can be inserted directly into `ie-spec-sim`. Our finding "drafts with min_prob 0.8 outperform long ones"
fits this.

### 3.3 Calibrating the acceptance probability

- Our finding: frequencies in the trie are too optimistic (estimate ~1, acceptance ~22%). AgSpec avoids the
  estimation: instead of estimating p directly, a **quantile of the observed reusable length** (offline replay)
  plus **online feedback** on the hit/miss bit. This is robust against miscalibration.
- Alternative from llama.cpp: score per n-gram in the pool (#19294, open): hit up, miss down, pruning after a
  streak. Observation of the author: more even speedup, no collapse after a streak of misses.
- Adaptive depth with a model drafter: llama.cpp **#27210 open** (draft-mtp-adaptive, stew675, 101 comments):
  counter + weighted miss pressure, lower bound 3; user measurement (independent, Intel/AMD/Vulkan, MTP): code 157.7 →
  163.2 tok/s decode, prose 147.7 → 149.1 (+1–3.5%). Conclusion of the comments: fixed n-max too large for thinking, too small for code.
- MoE-specific: **EVICT** (arXiv 2605.00342, 2026-05-01, v2 2026-09-05): prunes the draft tree before
  verification to the cost-effective prefix, benefit from drafter signals times offline-profiled verification
  cost; up to 2.35× against no speculation, on average 1.21× against EAGLE-3 (authors, formula in the full text only).
  **DraftExpert** (2607.24434) truncates when an uncertain candidate would bring too many new experts (already in
  papers.md). **MoE-Spec** (arXiv 2602.16052, 2026-02): fixed expert limit per layer during verification.
  **AcceptMoE** (2608.02989) see papers.md.
- DSpark (arXiv 2607.05147, 2026-07-06): probability per prefix from a confidence head, combined with a
  throughput profile; vLLM RFC #48202 (closed) and reference PR #48204 (open): on Qwen3-4B, truncating per
  request brings **no** wall-clock improvement (author), mechanism only.
- vLLM #63151 (open, 2026-10-22): upper limit of verified drafts per request from the client.

---

## 4. Community, real gains with coding agents

All statements are user or author statements, not reproduced.

| Source | Finding |
|---|---|
| llama.cpp #19164 (ngram-mod), Jan. 2026 | ggerganov: helps only on repetition ("unless your use case involves such repeating blocks of text, this method won't help"). Users: acceptance 0–92%, gpt-oss-120b 181 → 814 tok/s decode at 92% (two runs, repetition); Devstral with CRLF files 11% acceptance (2.3 tok/s decode) against 83% with LF (29.8 tok/s decode): **line endings of the input change tokenization and kill the match**; Qwen-30B 4–63%. One user sees OpenCode + GLM-4.7-Flash "almost double" (unsure whether placebo) |
| llama.cpp #26283 (suffix, open), 2026-08-03 | ServeurpersoCom (independent): the default configuration lowers dense 42 → 31 tok/s decode on normal prompts, because the per-request tree has count 1 almost everywhere and `min_prob` filters nothing; with n_min=3: 41 tok/s decode; **MoE 110 → 185 tok/s decode** with the minimum length (against none). Demo: XML generation 41 → 122 tok/s decode once the tree is warm. Global tree still missing, kashif took over the patch, waiting for review |
| llama.cpp #19493 (checkpoint, merged 2026-04-19), stsydow | Hybrid model (Qwen3.5), code: **>30% faster**, normal text: ~10% slower |
| llama.cpp #27154 / #27210 (Strix Halo, 7900 XTX) | See 2.1 and 3.3. Neural drafters on APUs cost bandwidth: DSpark 1.36B drafter 32.6 tok/s decode cold against 59.6 with ngram; "ngram (from the prompt, zero bandwidth) wins on iGPUs" (one user) |
| SGLang open | #45054 (prompt SAM), #44957 (corpus per request), #42313 (tool-call seed), #44496 (external SAMs compact), #43954/#44147 (greedy verification for NGRAM), all open, as of 2026-10-14 to 10-23; tracking issue #21052 |
| vLLM open | #59712 (ngram_hint), #60339/#52097 (suffix GPU), #63151 (per-request limit), #28586 (suffix example, since 2025-11-12) |

Reading: **The real gain depends on the repetition rate, not on the method**, and there is no
independent measurement with Claude-Code-like traffic on a hybrid MoE. This supports our own
decision to measure that with the real model on recorded sessions.

---

## 5. Genuinely new, fitting our situation

1. **Index the emission form (AgSpec).** Our corpus contains tool results. Claude Code's read tool
   emits files with a line-number prefix (cat -n style); when writing (Edit/Write) the model writes them
   **without** a prefix and in tool-call syntax. A match against the raw tool-result text breaks at every
   line start. Remedy: **normalize** file contents from tool results **before indexing** (remove the prefix)
   and possibly store them in the escape form of the Qwen tool-call syntax. This is AgSpec's finding (+13–39%
   acceptance in round 1) transferred to our format (own transfer, effect unmeasured).
2. **Tool-call scaffold from the template** (vLLM #59712 and others). We know the chat template; difference rendering
   is trivial and yields drafts for `<tool_call>`, function name, parameter tags. The closing block
   (`</parameter>\n</function>\n</tool_call>`) and the `<|im_end|>` sequence belong to it as well. Stop the draft
   at the name and continue it only after the name has been chosen (limit noted by the author).
3. **Drafts from the file system:** AgSpec's workspace corpus (opened files) brings only +2.2–7.0% over the
   session, because files are mostly already in the session as a tool result. Pre-indexing whole repositories
   would hardly be useful according to AgSpec; only files the agent touches count.
4. **Mind line endings and whitespace** (llama.cpp #19164: CRLF 11% against LF 83%). Tokenization of the source
   and of the output must be the same; normalize tool-result text before indexing the same way as the output.
5. **Detect non-repetition:** In thinking blocks and prose acceptance is ~0 (Baseten, llama.cpp users).
   Speculation off during `<think>` or with a high minimum match.
6. **Tree with a hybrid model:** Bole (2608.01651) and TreeWY (2608.20961, 2026-08-21, Qwen3.5 35B/397B,
   one author, cost "a few percent" when memory is not tight) make trees over DeltaNet practical;
   but our simulation says: a tree brings hardly more than a path (2.33 against 2.27). Goose (asymmetric tree)
   would be the variant with a justified structure (deep suffix chain, wide drafter branches); effort versus benefit
   open.

---

## New papers (not checked in papers.md)

| Date | Paper | Relevance |
|---|---|---|
| 2026-10-06 | AgSpec, arXiv 2610.01108 | Corpus by source, emission form, cost-aware length with λ; directly transferable |
| 2026-04-02 (v2 08/2026) | Goose, arXiv 2604.02047 | Asymmetric tree from two training-free sources |
| 05/2026 (v2 2026-09-05) | EVICT, arXiv 2605.00342 | Cost-effective prefix before MoE verification |
| 02/2026 | MoE-Spec, arXiv 2602.16052 | Expert budget per layer during verification |
| 2026-08-21 | TreeWY, arXiv 2608.20961 | Tree verification over GDN without state copies |
| 2026-07-06 | DSpark, arXiv 2607.05147 | Confidence-driven verification length |
| 2026-08-25 | AgentSpec, arXiv 2608.24004 | Speculation in agent workflows (batch) |
| 2026-07-03 | SPORK, arXiv 2607.03333 | Probe for the next tool name, rejected probes as a draft |
| 11/2025 | ReSpec, arXiv 2511.01282 | Entropy as a switch for retrieval |
| 11/2025 | Cacheback, arXiv 2511.21699 | LRU n-gram tables |
| 01/2026 | Speculative Decoding: Performance or Illusion?, arXiv 2601.11580 | Independent measurement: n-gram strong only on code edits |
| 07/2026 | WAR, arXiv 2607.17299 | Suffix only at low load (RL rollouts): 1.4× |

## Open points

- Not read: full texts of EVICT (formula), AgSpec per-model tables, TreeWY, SPORK.
- Not found: SGLang suffix-decoding PR (there is NGRAM-SAM), Reddit measurements with Claude-Code traffic
  (the search returned only GitHub comments), independent measurement of TRT-LLM `use_sa_spec`.
- Date note: arXiv ID 2608.20359 does not match the submission date on the page.
