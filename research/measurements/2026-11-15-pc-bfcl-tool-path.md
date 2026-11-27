# Does Tona's tool-call path cost BFCL cases? (Coder IQ1_M, 52 multi-turn cases)

## Question and decision

Is part of the Coder's low BFCL score (17/52 on Tona, `2026-11-16-pc-dm-vs-coder.md`) caused by Tona (template, tool-call
parsing, stop handling, sampling) and not by the model? Outcomes: Tona bug found by reading -> fix with unit tests from
the real outputs, re-run BFCL on Tona once; no bug and llama.cpp within 5 cases -> model; llama.cpp clearly lower -> note.

## Plan

### Scope

1. Read the 52 case results of the #145 Coder run (no GPU): failure kinds, what the model emitted, what the parser made of it.
2. If reading shows a bug: fix it, then one BFCL pass of the same 52 cases on Tona with the fix (one queue job, build
   `work/build/147`, `bench/polyglot/model_job.sh coder-tona bfcl`, ctx 131072, MTP, production mode, scope
   `MemoryMax=54G`, port 8091, BFCL temperature 0.001). llama.cpp is then not run: the question is already answered by the
   reading, and the fixed Tona run is the check.
3. If it does not: llama.cpp-hip on the same GGUF, same cases (`--jinja`).

### Expectation

Source of the expectation: the #145 result files. 8 of the 35 failed Coder cases are VehicleControl cases in which
`lockDoors(unlock=False)` arrived as the string `"False"` (256 such calls in the Coder run, 0 as JSON `false`), and
the `ls(a=True)` loops (`"a":"True"`) account for further force-terminated cases. Expectation after the fix: Coder
**at least 24 of 52** (the 17 plus most of the 10 to 14 cases that failed on this); the Ornith reference on llama.cpp was
27/52. Tolerance: one pass at temperature 0.001 is not fully deterministic (MTP), +-3 cases. If the score stays below 22,
the literal bug is not the main cause and the remaining failures are read again.

### Active tooling

`TONA_DIAG` default, no profilers, no `TONA_DETERMINISTIC` (production mode as in #145), no `TONA_RAM_GB`, no `TONA_PIN_GB`.
The machine is shared with other workers (the quiet check warned before every job in #145 as well).

## Test conditions

`tools/hw_report.py` of the job (`tc-coder-tona-bfcl-153731.hw.md`, not committed): Ryzen 5 9600X, RX 9060 XT gfx1200
15.9 GiB (1.17 GiB VRAM held by others), 60.5 GiB RAM (54.6 GiB available), swap 2.16 GiB in use, load average 3.35,
Fedora 44, kernel 7.2.15, ROCm 10.1, no sandbox. Commit 5a53b24 (the fix on `main` 0a926d9), build `work/build/147`
(deleted afterwards), Coder IQ1_M + MTP drafter, ctx 131072, production mode. `tools/quiet_check.py`: **warning** before
(other programs on the CPU, VRAM held by others, swapping) and **warning** after, unchanged. The machine was shared with
other workers (their GPU jobs through the queue, CPU work beside mine). Speed is not part of this question.

## Step 1: reading the 52 results of the #145 Coder run (no GPU)

Raw files: `result` and `score` of `bfcl-coder-tona` (not committed). The inference log of each case holds what the
parser handed to the harness. Findings per category:

- **force_terminated** (base_55, base_86, miss_func_43, miss_param_32, ...): the model calls `lockDoors` with
  `"unlock":"False"` (a **string**) or `ls` with `"a":"True"`. In the harness both strings are truthy, so
  `lockDoors(unlock="False")` *unlocks* (tool answer: `lockStatus "unlocked", remainingUnlockedDoors 4`, then the same call
  again until the step limit of 20), and `ls(a="True")` takes the wrong branch. The schema says `boolean`.
- **instance_state_mismatch** (base_79, miss_*_79, long_context_79 and _86, ...): the same VehicleControl cases ended in the
  wrong end state (`engine_state stopped` against `running`, 4 doors still unlocked).
- **empty turn** (miss_func_31, _178): not examined in this ticket; the turn ended without a call after text answers.
- **wrong args / execution mismatch** (base_188 `authentication_status`, base_178): not caused by the literal bug as far as
  read; left to the model (**ungeprueft** in detail).
- Count over all four result files of the Coder run: **256** `"unlock":"False"` and 12 `"unlock":"True"` as strings, no
  JSON booleans.

**Cause in Tona.** The model writes Python literals (`True`, `False`, `None`) in the `<parameter=...>` text. (Correction by the coordinator: the first version of this file blamed our template; `to_pyjson` in `engine/server/chat.cpp` renders booleans and null in JSON spelling, so the habit is the model's own, cause unchecked.) `engine/server/chat_qwen3_parser.cpp` parsed a non-string
parameter with `json::parse` and, when that failed (`False` is not JSON), kept the raw text as a string. A bug in Tona's parser, which must accept what the model writes. (The DM run shows the same: 324 `"False"` strings.)

## Fix

`parse_value` in `chat_qwen3_parser.cpp`: after a failed JSON parse the literals `True`, `False`, `None` become boolean
and null; string-typed parameters are untouched (`"False"` stays a string for `type: string`). Unit test
`test_python_literals` in `engine/tests/test_chat.cpp` (the `lockDoors` call of the failing case, with 1-char and
3-char streaming steps). Commit 5a53b24.

## Result: the same 52 cases on Tona with the fix (one pass, temperature 0.001)

| Run | base | miss_func | miss_param | long_context | total |
|---|---|---|---|---|---|
| Coder, Tona before the fix (#145) | 7 | 2 | 2 | 6 | **17/52** |
| Coder, Tona with the fix | 9 | 4 | 4 | 8 | **25/52** |

Ornith on llama.cpp (2026-11-15 evaluation): 27/52. Table: `2026-11-15-pc-bfcl-tool-path-bfcl-multi_turn.csv`.
Remaining failures (27): 12 state mismatches, 7 force_terminated (e.g. miss_func_32 loops on `power` with changing
exponents, no literal involved), 5 execution mismatches, 3 empty turns. The `lockDoors` and `ls` string literals are gone
from the results (24 JSON `false`, 8 `true`); the remaining `"None"` strings (access_token, name) are string parameters.
One pass: the run-to-run spread is not measured (**ungeprueft**; MTP and 0.001 temperature are not deterministic), the
gain of 8 cases is larger than the 3-case tolerance of the plan.

## Decision

Outcome 1 of the question: a Tona bug, found by reading, fixed, confirmed once. llama.cpp was **not** run: the reading
already answered the question and the fixed Tona run reached the expectation (25 against at least 24); a llama.cpp pass
would only measure how much of the remaining 27 failures is the model, which is a separate question. The DM result of
#145 (3/52) was produced with the same bug; its re-run is not part of this ticket, so the DM verdict of #145 is
**open** (the bug hit both models; the gap may shrink).

Goldens: the change is in the output parser after generation; token and logit goldens (`golden.sh`,
`golden-server.sh`) do not pass through it, so they cannot change. Not re-run (**ungeprueft**, reasoned only).

## Prior art

- vLLM's Qwen3-Coder tool parser converts parameter values by the schema type; its boolean handling was not looked up
  in this run (no network search done): **ungeprueft**.
- `research/measurements/2026-11-15-model-eval-ornith-bolt.md` and `2026-11-16-pc-dm-vs-coder.md`: read; the harness and
  the 17/52 baseline come from there.

Change type: bugfix
Search: tools/papers/scan.py not run in this ticket; window n/a (the cause is a local parsing gap found in the result files)


## Addendum (coordinator, 2026-11-15)

Second commit 9d00a10: Python literals inside arrays and objects (`[True, {"a": None}]`) are converted too, by replacing the bare words outside double-quoted strings and parsing again; vLLM's tool parsers fall back the same way (`json.loads`, then `safe_literal_eval`, `vllm/tool_parsers/hy_v3_tool_parser.py`). Unit test `test_nested_python_literals`. Not re-measured on BFCL: it only changes values that failed to parse before. Open: the #145 DM BFCL score (3/52) was measured with the bug and needs a re-run before its verdict stands.
