# 2026-10-26: more slots for saved conversations

Own measurement. `scripts/bench-multi.py`: 6 conversations with their own beginning (~13.4K tokens of code each, task and
starting point different), 3 rounds in turn, per round a short message and 24 tokens of answer. Server with
`--ctx 65536`, without SSD cache. Counted from round 1 (round 0 computes each conversation once in full).

| `IE_PROMPT_SLOTS` | recomputed | Prefill | RSS afterwards |
|---:|---:|---:|---:|
| 4 (before) | 161,315 tokens (13.4K per request) | 97.0 s (8.1 s per request) | 31.2 GB |
| 16 | 551 tokens (46 per request) | 4.7 s (0.39 s per request) | 32.3 GB |

With 4 slots and 6 conversations in turn, LRU evicts exactly what is needed next. The byte limit
`IE_PROMPT_RAM_GB` (12) stays. A saved conversation costs ~190 MB fixed (state 149, drafter 42) plus ~15.4 KB
per token (KV 13.9, pool 1.6). At 40K tokens that is ~0.8 GB, so the 12 GB hold ~15 such conversations.

Server RAM at `--ctx 262144`: RSS 33.3 GB, of which 25.1 GB pinned (experts, KV int8 3.62 GB). With 12 GB of saved
conversations ~15 GB remain for the desktop and sessions.

Step 3 of the plan (eviction by recompute time per byte) postponed. It only takes effect once the 12 GB are full,
i.e. from ~15 long conversations at the same time. With more conversations in turn than fit, no per-byte
scoring helps against LRU eviction in a circle either.
