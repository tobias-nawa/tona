# Multiple conversations and intermediate checkpoint in the server, 2026-10-23/24, own measurement

ie-server like `just serve` (context 262,144, DFlash), Claude Code in auto mode.

## What the side requests are (IE_DUMP_DIR)

- Model `claude-sonnet-5`, max_tokens 64, no tools: the **auto-mode checker** ("You are a security monitor
  for autonomous AI coding agents"). Claude Code sends it to ANTHROPIC_BASE_URL, i.e. to our local model.
- Structure: system ~139,000 characters (≈ 30,500 tokens) identical, then the user's CLAUDE.md as a message (identical), then
  `<transcript>` with all actions in one message; each check appends one line. Answer `<severity>N</severity>`.
- A checkpoint at the end of the prompt therefore never matches: the start of the last message is identical, its end is not.

## Result

| State | Check request | Main thread after a check request |
|---|---|---|
| one conversation (until the evening of 10-23) | 41–50K tokens new, 45–51 s | 42–74K tokens new, 49–95 s |
| + conversations in RAM (IE_PROMPT_SLOTS) | 46–53K tokens new, 44–52 s | from RAM in 230–330 ms, only new tokens |
| + intermediate checkpoint at the start of the last message | **1,250–1,390 tokens new, 1.7–1.8 s** (first: 32 s) | from RAM in 160–200 ms |

- Run with conversations but without intermediate checkpoint (test project, 23:10–23:51, main thread 75–105K tokens): 30
  check requests of ~48 s each = ~24 of 41 minutes.
- Run with intermediate checkpoint (create and run 5 files, 00:06–00:08): 16 requests in 93 s, of which 5 check requests.
- Stored per conversation: 134 MB state + 84 MB drafter ring + ~13.9 KB per token (KV 12.9 + pool 1.5):
  16K tokens 459 MB, 100K tokens 1,763 MB. Storing or fetching 100–330 ms.
- Eviction: first whatever was not fetched in the last 16 operations (Marconi idea); an end that continues an
  older one inherits its hits. Without decay, entries of old conversations stayed around and displaced the
  main thread (measurement 00:03).

## Control without auto mode (00:08)

The same task with `--permission-mode acceptEdits --allowedTools "Bash(python3:*)"`: 11 requests, no
check request, 56 s instead of 93 s. So the side requests come only from auto mode.

## Side finding: answers not reproducible

The same prompt (22,020 tokens of code, greedy) computed fresh gives 5 different answers in 5 runs (from token
~10), another prompt (15,701 tokens) 3 of 3 equal. Probably floating-point atomics (order of
sums). Not caused by the conversations. A comparison with checkpoint but without recomputation (23:09) was
bit-identical.

## Intermediate checkpoints on the SSD (09:34–09:37)

`IE_PROMPT_CACHE=work/prompt-cache`. Run 1 writes the intermediate checkpoints (main thread 4,481 tokens = 288 MB, checker
30,531 tokens = 688 MB). Then server restart, run 2 (different task, different folder):

| | Run 1 (fresh) | Run 2 (after restart) |
|---|---|---|
| first check request | 31,756 tokens new, 48.6 s | 1,225 tokens new, **3.6 s** (SSD 256 ms, inserting 1,516 ms) |
| further check requests | 1.7 s | 1.7 s |

The intermediate checkpoint of the main thread does not match in run 2: Claude Code writes the working folder into the
system prompt, and it was a different one.
