# Prompt cache: what a hit computes

The prompt cache keeps states of earlier prompts (RAM stash, SSD checkpoint files) and continues a new prompt from the
longest one that is the start of it. Sizes, budgets and switches are in [environment.md](environment.md); this page is
about the numbers: a hit must not change the answer.

## The rule

**A prompt is computed in one canonical way, whatever the cache holds:** all tokens but the last through the prefill
kernels, the last one through `forward` (it leaves the logits the sampler needs and writes the drafter state).
Where the prefill is cut does not matter: with `TONA_DETERMINISTIC=1` the state after a prefill of n tokens in one call
and in any number of pieces is byte-equal, in the chunk-by-chunk plan and in the layer-by-layer plan, with pieces that
cross the chunk edge (6,144) and the group edge (12,288)
([measurement](../../research/measurements/2026-11-02-prefix-bitexact.md), section 2). Chunk size and group are chosen from
the length of the piece (`PrefillRun::choose_plan`), but no kernel adds in an order that depends on where a chunk
starts: dense and expert matmuls compute every row alone, DeltaNet runs token by token, attention and the indexer per
token.

What can differ between a hit and a cold run is therefore only a token that goes through another path. Two kinds of
state exist:

| State | Made by | Last token | Logits stored | Continues |
|---|---|---|---|---|
| checkpoint inside a prompt (fork, start of the last message, head of a warm-up, slice boundary of an interrupted warm-up) | `Generator::checkpoint_at`, `prefill_warm` | through the prefill, like in a cold run | no | every longer prompt that starts with it |
| end of a prompt (every request, second stage of a warm-up) | `Generator::prefill` | through `forward` | yes | longer prompts (not canonical, see below) and exactly this prompt again |

A checkpoint inside a prompt that ended in a forward token would round that one position differently from a cold run;
the rest of the prompt then inherits the difference and the answer departs after a few hundred characters (cold,
after a fork, after a warm-up each gave another answer, each repeatable). That was the cause behind it; the chunk
plan was not (section 2 of the measurement). `split_run` (`engine/server/run_split.h`) states the rule in one place.

Because a state without logits cannot answer the prompt it ends, `Generator::find_source` never picks one for a prompt
of exactly its length; the next candidate (or a cold run) takes over. Checkpoint files record whether they hold logits
(`DISK_VERSION` 3; older files have the forward token and are ignored).

Behind a checkpoint even a short rest (at most `TONA_PREFILL_MIN` = 32 tokens, which a cold start would send through
`forward`) stays in the prefill, because the cold run of the whole prompt prefilled those tokens.

## Tests

- `ctest -L unit`: `run_split` (which tokens go where, for any cut), `checkpoint_file` (files without logits).
- `scripts/golden-server.sh` case `hit` (`GOLDEN_MODEL=mini` takes seconds): the start of the last message, the fork, a
  warm-up head, a warm-up of exactly the prompt, and the same prompt again, each answered by a server with checkpoints
  and by a server with `TONA_PROMPT_SLOTS=1` (none written); the pairs must be byte-equal. On main every pair differs.
- `tona-gpu-gen --teacher TEXT --mega` with `TONA_TEACHER_TAIL` and `TONA_TEACHER_CUTS`: logits of the tail after a prefill
  cut at chosen tokens against the unsplit prefill, for a quality or invariance check of a kernel change.

## Known limits

- **A turn that continues a conversation is not bit-equal to a cold run of the same tokens.** The state at the end of
  the previous request holds the tokens the model generated, and those went through decode passes (width 1-8, other
  roundings in the dense part, see [how-it-works.md](../how-it-works.md#how-the-engine-computes) on speculation). A cold prefill of the same text computes them in the
  prefill. The answer is deterministic and of the same quality, but it is not the cold answer. This covers the running
  conversation, the end state (`Source` 0 and 1) and a stashed conversation end; only the three checkpoint kinds above
  are canonical. Making them canonical would mean recomputing the generated tokens through the prefill after every
  turn.
- **An exact repeat of a prompt whose only stored state is a checkpoint inside it** (no end state) is computed from the
  next shorter state or from scratch: the checkpoint has no logits.
- **Without `TONA_DETERMINISTIC=1`** the prefill and `forward` for one token are the same code, but wide passes of the
  decode path round differently; the equality above is tested with the switch.
