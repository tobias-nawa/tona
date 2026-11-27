# Prefix cache for hybrid models with several conversations

**Agent research (Sonnet), 2026-10-23.** One search run, papers read as abstracts only. Numbers are the authors' claims.
"Inference" means: derived, not documented.

## Motivation (own measurement)

Claude Code in auto mode (log 2026-10-23 22:24–22:52, measurements/2026-10-23-claude-code-real.md): every second request was
a side request with ~41–46K tokens and a 7-token answer, presumably the auto-mode classifier. The server held only one
conversation. Each side request evicted it, after which the main thread was recomputed too: 45–90 s per
switch, 90 s at 71K tokens.

## How other engines solve it

- **SGLang MambaRadixCache**: the radix tree returns the deepest node with a stored state, the state is copied
  out. With `extra_buffer` there are states at branch points, cost 2 slots per request.
  https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/
- **SGLang Unified Radix Cache** (2026-08-12): copy-on-write for states, session-aware eviction,
  "tombstone" nodes without state. Prototype Qwen3-Next-80B: TTFT only 5–7% better.
  https://www.lmsys.org/blog/2026-08-11-unified-radix-cache
- **vLLM hybrid KV cache manager**: states only on the block grid (`--mamba-cache-mode align`).
  `--enable-mamba-shared-prefix-checkpoint` additionally stores a state at the end of a shared prefix.
  https://docs.vllm.ai/en/v0.32.0/design/hybrid_kv_cache_manager/
- **llama.cpp**: context checkpoints `-ctxcp` (default 32), `--checkpoint-min-step`.
  - Issue #22384 (closed 2026-04-26, "nothing to fix"): Qwen3.5/3.6 recomputes every turn completely, causes include
    thinking blocks that are dropped in the following turn.
  - More issues about full recomputation: #23013, #24587, #21690 (status not checked).
- **LMCache**: Gated-DeltaNet hybrids are not supported in the multiprocess connector.
  https://docs.lmcache.ai/mp/hybrid_models.html

## Papers

| Paper | Key statement | Relevance |
|---|---|---|
| Marconi, arXiv 2411.19379 (MLSys 2025) | State only at branch points of the radix tree plus after the last decode step; eviction by FLOP savings per byte; up to 34.4× hit rate | high |
| Sparse Prefix Caching, arXiv 2605.05219 (05/2026) | Checkpoint positions via DP from the distribution of overlap depths, beats fixed grids | medium |
| Tail-Replay, arXiv 2608.30310 | Store only attention KV, rebuild DeltaNet state from a 5–10% suffix (approximate, 92.8–99.9% quality), 9–14× TTFT at 32K | medium (only with quality measurement) |
| SuffixReplay (09/2026, arXiv unconfirmed) | similar, 0.36–0.51× memory against SGLang checkpoints | low |
| KVFlow, arXiv 2507.07400 (NeurIPS 2025) | Eviction by "steps until next execution" + prefetch, up to 2.19× against SGLang HiCache | low |

The agent found no source on coding-agent traffic specifically.

## Plan for us

1. **Now:** N conversations (IE_PROMPT_SLOTS, default 3). The running conversation stays in VRAM. The others lie
   as a checkpoint in RAM: states, KV rows, indexer pool, drafter ring. The longest exact prefix is chosen.
   Inference: this removes the recomputation on side requests, because the classifier and the main thread each
   continue their own prompt end.
2. **Afterwards:** checkpoints within the prompt (Marconi, vLLM shared-prefix): at the end of system prompt and tools and
   at message boundaries, taken during prefill without recomputation. This helps sub-agents that share the same system
   prompt.
3. Eviction by recompute time per byte instead of LRU.
4. Tail-Replay only with a quality measurement of our own.
