# SuffixDecoding in the server, 2026-10-23

Draft from the history (server/suffix.{h,cpp}): index over 4-grams of all tokens in the context (prompt + answers),
longest earlier match of the end of the text (up to 64 tokens, most recent on a tie); if it is ≥ IE_SUFFIX_MIN
(default 8), the up to 3 following tokens are verified, the drafter only catches up on its context.
Otherwise DFlash2 as before. ie-server, 64 GB, --ctx 16384, 2 runs in a row per task, greedy.

| IE_SUFFIX_MIN | Edit file (800 tokens) | Running text "explain" (300 tokens) |
|---:|---|---|
| 0 (off) | 44.6 / 42.6 tok/s, 3.96 / 3.77 tokens per pass | 32.4 / 39.6 tok/s, 2.34 / 2.29 |
| 8 | **50.2 / 49.1 tok/s**, 3.96 / 3.92; 179/202 and 181/204 passes from the history | 34.3 / 40.2, no matches |
| 4 | 49.7 / 49.4, 3.96 / 3.94; 184/202, 185/203 | 31.8 / 37.6, 1 match |

"Edit file": tools/server_smoke.py in the prompt, "Add Python type hints … return the complete file". Rates are decode.

Result: +13–15 % for agent-typical copying, running text unchanged. Threshold 8 stays the default (4 adds nothing
and occasionally hits wrongly on running text). The gain comes almost entirely from the saved drafter step: tokens
per pass were already at 3.8–4.0 before, i.e. at the limit of 4 tokens per pass (BMAX). Longer drafts need
larger passes (kernel rework BMAX 8/16).
