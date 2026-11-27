# 2026-10-22: SuffixDecoding simulated on own agent sessions

Own measurement (simulation). Tool: `ie-spec-sim` (`engine/spec/suffix.{h,cpp}`), corpus from
`tools/agent_corpus.py`. After SuffixDecoding, arXiv 2411.04975.

## Setup

- **Corpus:** 805 Claude Code sessions of this machine (transcripts ≥ 200 KB, all projects),
  chronological. Context = user + tool results, output = text + tool calls of the assistant, in the
  Qwen chat format, tokenized with our tokenizer. The corpus stays in `work/` (private).
- **Test:** the last 40 sessions, 2.84 M context and 1.20 M output tokens.
  **Global trie:** outputs of the 60 sessions before, extended online by each test session.
- **Acceptance:** the model produces exactly the recorded output. A pass yields the accepted
  draft tokens plus one.
- Trie depth 40, patterns up to 32 tokens, draft per pass from the trie with the higher score (request or
  global), as in the paper.
- **Cost model:** time per pass = 4.11 GB dense / 315 GB/s + u(b) × 0.98 GB experts, of which 86.5 %
  from VRAM and the rest over PCIe (55 GB/s). u(b) is the measured union of the experts
  (2026-10-22-expert-routing.md). Without speculation: 55.1 tokens/s.

## Results

| Variant | Tokens/pass | Batch avg | Acceptance | Speedup (model) |
|---|---:|---:|---:|---:|
| path, α 2, max 16 (default) | 2.27 | 6.86 | 21.6 % | 1.36× |
| request trie only | 2.15 | 5.65 | 24.7 % | – |
| global trie only | 1.66 | 5.71 | 14.0 % | – |
| path, min_prob 0.8 | 2.07 | 4.36 | 31.9 % | **1.51×** |
| path, α 1, max 8 | 2.10 | 4.18 | 34.7 % | 1.48× |
| path, α 4, max 32 | 2.35 | 11.92 | 12.3 % | – |
| path, cost-aware trimmed | 2.26 | 6.43 | 23.1 % | 1.39× |
| tree, max 16 | 2.33 | 6.84 | 22.7 % | – |
| tree, min_prob 0.3 | 2.27 | 5.99 | 25.4 % | 1.45× |

"–": run before the cost model, tokens/pass only. In 63–70 % of the passes no draft token is accepted.

## Interpretation

- **~2.1 tokens per pass** without a drafter model, almost only from the running session. Earlier sessions
  add +0.1.
- **A tree brings barely more than a path** (2.33 against 2.27). Because of DeltaNet (state per branch)
  we take the path.
- **Guessing less is better:** because the experts of neighboring tokens overlap only little,
  the cautious variant (1.51×) wins against long drafts (1.20×). This confirms DraftExpert and
  AcceptMoE for our platform.
- The cost-aware trimming barely helps because the probabilities from the trie frequencies
  are too optimistic (estimate ~1, acceptance ~22 %). Next step: calibrate the acceptance from pattern length and
  frequency.
- The paper reports 7.8 tokens/step on SWE-Bench. Here it is less: the sessions write a lot of
  new text (answers in German, new files).

## Limits

- The outputs come from Claude, not from Qwen. How often the agent copies depends mostly on
  the task; the exact acceptance has to be measured with the real model.
- The cost model only accounts for bandwidth, without compute time, kernel launches and DeltaNet rollback.
  The union was measured on 512 tokens of Python.
- One run per variant. The simulation is deterministic; the spread across sessions has not been
  evaluated.
