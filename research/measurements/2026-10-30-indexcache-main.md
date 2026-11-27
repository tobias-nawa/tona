# IndexCache decode gain on main after #74 (2026-10-30)

Question: #73 measured the decode gain of `IE_IDX_SHARE` before #74 made the indexer and top-k of the megakernel
cheaper (`2026-10-29-indexcache.md`: 250K pass 65.0 -> 53.2 ms with 6 full layers). The threshold run
(`2026-10-30-threshold-main.md`) passed 8, 6, 5 and 4 full layers. What is the gain on main now?

Setup: `work/build/main` (60d26f3, #73 and #74 in), server with MTP, `--ctx 262144` (KV int8 3.62 GB in pinned RAM,
79 expert slots per layer), engine sources as prompt (`code.txt`), new server per variant (empty cache), 130,788 and
249,732 prompt tokens, 200 tokens greedy, thinking off. 3 rounds alternating full / 6 / 4. Desktop running on the same
GPU. Script and logs: `/var/tmp/claude/idx/` (not in the repository).

Pass time = tokens per pass / decode rate, from the server log line of the request (200 tokens are about 70 passes, so
this is rough; the decode rate itself moves with MTP acceptance and is not compared).

| full indexer layers | `IE_IDX_SHARE` | 128K pass, ms (3 rounds) | 250K pass, ms (3 rounds) |
|---|---|---|---|
| 12 (default) | unset | 43.6 / 45.0 / 43.9 | 47.1 / 46.4 / 46.0 |
| 6 | `0,2,4,6,8,10` | 42.7 / 42.6 / 43.7 | 44.1 / 44.9 / 44.3 |
| 4 | `0,3,6,9` | 45.0 / 43.9 / 44.9 | 44.2 / 44.2 / 43.0 |

Prefill is unchanged (the prefill path ignores the switch): 1,392-1,393 tok/s at 128K and 1,291-1,292 tok/s at 250K
for all variants.

Result: after #74 the gain is small. 250K: 6 full layers -4.5 % pass time (median 44.3 against 46.4 ms, all three
rounds below the lowest default round), 4 full layers -4.7 %. 128K: 6 full layers -2.7 % (ranges touch), 4 full layers
no gain. #74 removed most of what IndexCache saved (#73: -18 % at 250K before #74).

Open: sharing in the prefill path (about 5 % of a 250K prefill at the #73 ceiling, less after #71).
