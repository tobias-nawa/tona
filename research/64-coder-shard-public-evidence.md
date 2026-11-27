# 64: Public evidence on the Coder shard update of 2026-10-19 (#165)

Question: does public evidence say whether ISTA-DASLab's shard-1 update of the IQ1_M Coder (revision bba88856, commit
d1d94679, "Replace first shard with updated expert-pruned model") is better or worse than the 2026-09-29 files, and can
it replace our own agentic eval? Web research of 2026-11-18; pages were read through summaries, so exact quotes need a
check at the links.

## What exists

- **ISTA's own numbers** (model card, discussion #25): SWE-bench Verified 75.60 (old card) to 79.20 (current card;
  #25 says 75.8 to 79.2), LiveCodeBench v6 86.28 to 87.43, base model 82.80 and 87.43, all at xhigh effort. Single
  numbers: no spread, run count or seeds. The aim of the update was reliability (no loops or early stops "in limited
  tests"); about 25 % of the expert selection changed, no retained weight changed against the IQ3_S base.
  <https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/commits/main>,
  <https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/discussions>
- **ISTA on languages** (discussion #28): only JS, C++, C#, PHP, SQL and Bash were preserved by the expert selection;
  Rust is left for future models. Our polyglot shard covers Python, Go, Rust and JS, so part of it lies outside what
  the selection preserved.
- **User reports** (discussions #25, #26, #28, #30): both directions, none names a revision, no sample over a few
  tasks. #25 says the update "broke the model" (unresolved); #26 reports long subagent use without loops; #28 bad Rust;
  #30 5/6 one-shot C tasks on a Vulkan iGPU.
- **Leaderboards**: Artificial Analysis lists the base Qwen3.8-Flash-Next only and runs its own evaluations
  (<https://artificialanalysis.ai/methodology>); its GGUF set covers small mobile models, not community quants of server
  models. LMArena ranks by crowd votes (not checked for this model). No leaderboard lists this IQ1_M file or either
  revision.
- **Do leaderboards transfer to quant differences?** No rigorous evidence found. Work on answer flips under
  quantization reports near-equal accuracy with up to 13.6 % flipped answers; blog measurements
  (<https://kaitchup.substack.com/p/lessons-from-gguf-evaluations-ternary>,
  <https://localbench.substack.com/p/gguf-benchmark-methodology>) say small KLD or perplexity changes can hide large
  task differences.

## What it means for #165

- There is no independent comparison of the two revisions. The only one is ISTA's own (+3.6 points SWE-bench, one run).
- Our night-1 design resolves about 16 points per night (research/measurements/2026-11-18-agent-eval-shard.md), so it
  cannot see a difference of ISTA's size; it can see a large break in our engine and on our task mix.
- Rust is a known weak spot of the selection by ISTA's own statement; a per-language split of the polyglot result is
  the one place where our eval can say more than ISTA's numbers.
