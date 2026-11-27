# Prefill in slices, checkpoint at the branch point, effort per role

Own measurement, 2026-10-26, server with MTP drafter, `--ctx 16384`, two slots.

## Prefill in slices

Conversation A streams a long answer (1,800 tokens), after 4 s B sends a prompt of ~12,000 tokens
(engine/gpu/mega.hip). 3 runs per setting; run 2 hits the prompt cache (B almost immediately) and does not count.

| Setting | B: first token | A: longest pause |
|---|---:|---:|
| in one piece (`IE_PREFILL_YIELD_MS=0`) | 9.19 / 9.51 s | 9.13 / 9.48 s |
| **slices of 4,096, 1,000 ms decode in between (default)** | 12.64 / 13.08 s | 2.52 / 2.59 s |
| slices of 2,048, 500 ms | 14.25 / 14.84 s | 1.43 / 1.53 s |
| slices of 2,048, 1,000 ms | 17.86 / 18.27 s | 1.44 / 1.49 s |

Smaller slices compute more slowly (2,048: prefill itself ~25 % longer), hence 4,096. An auto-mode checker
(priority over all running requests) keeps computing in one piece.

## Checkpoint at the branch point

Four requests, system prompt ~11,400 tokens, after ~7,100 tokens a different working directory follows (like Claude Code in
different projects). Before, every request recomputed everything (6.8–7.1 s). Now the second one saves a checkpoint at the
branch point; the third and fourth reuse 7,125 tokens: **3.1 s instead of 6.9 s**.
Found along the way: the checkpoints were processed in a fixed order, a later one prevented earlier ones; now ascending.

## Effort per role

`IE_ROLE_EFFORT=main=low`: a request with `reasoning_effort: high` runs with "effort low" (log). Default off.

## One conversation alone (bench-set, decode)

| | Code | Text |
|---|---:|---:|
| before (IE_SEQS=2) | 89.4 / 89.6 / 89.7 tok/s | 64.9 |
| after | 89.5 / 89.5 tok/s | 64.8 / 64.9 |

Same tokens (code 3,139, text 1,148).
