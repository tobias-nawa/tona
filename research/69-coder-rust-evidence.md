# 69: Weak Rust in the Coder IQ1_M shard: public evidence (#165)

Web research 2026-11-18. Pages were read through summaries (WebFetch); quotes are short and should be rechecked at the
links before they go into a public file. Read first: research/64-coder-shard-public-evidence.md.

## Question

Is bad Rust (and Go) a known property of ISTA's expert-pruned Coder IQ1_M, or of the Qwen3.8-Flash-Next line, or of
low-bit quantization in general? Do we need tonight's llama.cpp cross-check to rule out an engine bug, and what do we
tell installer users about Rust and Go?

## What exists

### Q1 Is weak Rust known across the family (base, other quants, other prunes)?
- Base / other quants: **no per-language Rust or Go number found.** The only polyglot figure is overall: 167/225 =
  74.2 % (NVFP4, 2x DGX Spark, author's own run, 95.6 % well-formed edits), no split by language. Measured, not
  official. <https://forums.developer.nvidia.com/t/multi-agent-workload-qwen3-8-flash-next-nvfp4-on-2x-dgx-spark-174-t-s-aggregate-74-aider-polyglot-262k-context/385731>
- Unsloth reports the 1-bit quant at 54.0 % DeepSWE v1.1 and 68.3 % Terminal Bench 2.1 (agentic, not per language).
  <https://unsloth.ai/docs/models/qwen3.8-next>
- Ollama card: SWE-bench Multilingual 81.0 (base), no Rust split. <https://ollama.com/library/qwen3.8-flash-next>
- Other expert-pruned variants (REAP-style, community prunes) with Rust numbers: **not found.**
  The REAP paper and follow-ups report overall coding scores only: <https://arxiv.org/html/2510.13999v3>.
  "Half the Experts, All the Code" (Qwen3.6-35B-A3B, Gemma-4-26B-A4B): half the experts removable with no detectable
  loss on its primary code benchmark; per-language results not visible in the abstract (full paper unread).
  <https://arxiv.org/abs/2607.16721>

### Q2 Reports on this file or the Coder line
All on <https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/discussions> (30 open threads,
#1-#30; #5, #9, #17 absent, probably closed; dates only relative on the page).
- **#28 "Bad Rust coding quality on new project?"** (reporter on another engine), open, about 17 days old, last reply an ISTA member.
  Reporter: model mixed Rust with Python/C++ (`for arg in os.args: println("{arg}");`) and invented that "Rust 2024"
  is a dialect. Downloaded 2-3 days before posting, so probably the updated shard (not stated). ISTA: the model "only
  keeps the capabilities that we wanted to preserve during pruning"; preserved JS, C++, C#, PHP, SQL, Bash; surprised
  Rust was dropped entirely; will add Rust in future models; Lua: "will try". A question about LoRA fixes is unanswered.
- #19 (open, Oct 1, no reply): Coder vs unpruned IQ3_XXS, one quant tier each, thinking off: HumanEval+ 93.3 vs 93.9
  (Python), but factual knowledge 63.3 % to 32.9 %. Author sees the same loss as with REAP on this base.
- #29 (open): answers switch to English or garbled CJK after English code; #27 "Language Broken === Model Broken",
  #13 Japanese broken, #4 Chinese collapses reasoning: natural-language gaps outside the calibration mix are reported
  repeatedly. #30 (open): 5/6 one-shot C tasks on an iGPU, so C works. #26: "seemingly at parity with the IQ3S on
  coding tasks". #14 "zero tasks accomplished SWE Verified", #16 "noticeably less impressive than the full IQ3_S".
  None names a revision; all anecdotal.
- Reddit r/LocalLLaMA, blogs: not searched in depth; nothing surfaced for Rust on this model.

### Q3 Do very low bits or calibrated pruning hurt uncalibrated languages disproportionately?
- Pruning, natural language (measured): English calibration at 25 % pruning cut Lithuanian comprehension by a median
  21.4 points (Qwen3-30B: 89.9 % to 22.3 %, near chance) against 1.7 for English; calibration in the target language
  largely closes the gap. This is the same mechanism as a missing programming language, shown for human languages.
  <https://arxiv.org/html/2611.07654v1>
- Calibration choice can more than halve the coding score of a pruned Qwen3.6 (secondary summary, paper not opened).
  Domain-matched calibration is near-lossless in-domain, collapses outside (REAP literature, same link as Q1).
- Quantization, code: Lua (low-resource) paper: sharp drop at 2-bit, 7B models below 50 % on Lua, 4-bit best trade-off;
  1-bit not tested. <https://arxiv.org/pdf/2410.14766> (read via secondary summary). A replication (model card, not
  peer reviewed) shows 2-bit AQLM hurting Java (32.2 to 23.2) more than Python (29.8 to 27.6) on CodeLlama-7B
  MultiPL-E, i.e. the less frequent language loses more. <https://huggingface.co/Devy1/CodeLlama-7b-hf-AQLM-2bit-code-finetuned-1x15>
- "Half the Experts" claims one repair turn with execution feedback erases the 2-bit penalty. Our 4-attempt eval has
  that feedback and still fails Rust, so Rust is not a plain quantization penalty. <https://arxiv.org/abs/2607.16721>
- Go: **no evidence found either way.** ISTA never mentions Go in a preserved or dropped list. Go 3/17 fits the same
  explanation (not in the preserved list) but that is inference, not a finding.
- No study found for 1-bit code quality per language.

### Q4 Calibration set of the expert selection
- Model card: calibration "mainly code and agentic", the release searched with vision included; no language mix.
  <https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF>
- Discussion #22 (open, ISTA answer): `sci-datasets/sci-code`, `gaia-benchmark/GAIA`,
  `HuggingFaceM4/the_cauldron`; later `open-thoughts/OpenThoughts3-1.2M` for harder coding. Per-expert RCO scores are
  no longer available to ISTA; sample counts, seed and revisions unanswered.
  <https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/discussions/22>
- The card calls degradation outside the targeted set "an accepted cost of the method" and recommends the unpruned
  GSQ-RCO releases for general use. No language mix stated anywhere (not found).

### Q5 Chat template / output format
- No problem specific to this file found. Model card has a "Chat template" heading without content.
- Family: tool-call parsing problems with Qwen3.5/3.8 XML-style templates in llama.cpp (issue #20837) and community
  template fixes (froggeric/Qwen-Fixed-Chat-Templates); these concern tool calls, not fenced blocks.
  <https://github.com/ggml-org/llama.cpp/issues/20837>, <https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates>
- Card: long multi-step agentic tasks retain less than single-problem code; looping was reduced in the update (so it
  existed before). A missing fenced file block is not documented anywhere (not found).
- To separate format from skill locally: score only attempts that did produce a parseable block (the 7 of 10 Rust
  runs) and check whether the compile errors persist there.

## What it means for #165

- The Rust result is **explained by the maker's own statement** (#28): Rust was not among the preserved languages. The
  evidence is a maintainer statement plus anecdotes, not a measurement, but it matches the mechanism measured for
  natural languages (Lithuanian paper). 0/10 on easy exercises and invented APIs fit capability loss, not a template bug.
- Installer wording (suggestion): Rust: "not recommended; the expert selection did not preserve Rust (maker statement,
  discussion #28), our eval 0/10 with test feedback." Go: "weak in our eval (3/17); the maker lists Go neither as
  preserved nor as dropped; use at your own risk." Recommend per language: JavaScript (18/27 here, on ISTA's list),
  Python (12/18, not on ISTA's list but works), C/C++/C#/PHP/SQL/Bash per the maker's list (C++ and C untested by us).
  Point Rust/Go users to the unpruned GSQ-RCO IQ3 release, which the card itself recommends for general use. Do not
  claim the unpruned model is better at Rust: no Rust number exists for it.
- Is tonight's llama.cpp check still needed? Public evidence makes an engine bug an unlikely sole cause for Rust
  (an independent user reports the same with a different engine, #28). But Python 12/18 and JS 18/27 working
  only shows the engine is not broken in general, not that it is bit-correct. The check remains cheap insurance; it was
  extended to the 17 Go exercises, since it is the only thing that would separate engine error from model error for
  Go (3/17), where no public data exists.
  Suggested reading: if llama.cpp matches us on Rust/Go within noise, write the installer text above; if it is clearly
  better on Go, investigate the engine before publishing the Go wording.
- Open gaps worth a follow-up if wanted: per-language numbers for the unpruned IQ3_S on our eval (would split pruning
  from 1-bit quantization); Reddit/blog search; full text of 2607.16721 for per-language results.
