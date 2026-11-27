# 71: How our coding evals should be run, and where we deviate (2026-11-19)

Scope: read-only audit of `bench/polyglot/`, `bench/agent/` and the Rust/Go run of 2026-11-19
(`research/measurements/2026-11-19-rust-go-ornith-bolt.md`) against the primary sources named below. No harness code was
changed, nothing was run. "Raw replies" means `/var/tmp/claude/w170rg/{coder,ornith}/*.jsonl`.

## Summary

- The polyglot harness is a faithful cousin of Aider's, not a copy. The parts that decide pass or fail match: the model
  cannot edit tests or `Cargo.toml`, tests are restored from a clean copy, Rust runs with `--include-ignored`, JS `xtest`
  is enabled, Go runs `go test ./...`. Nothing found invalidates a comparison between models.
- The strict-parser mistake is real and the fix in the working tree is what Aider does: with exactly one file in the chat, a
  fence without a file name is that file (`wholefile_coder.py`, source `"chat"`).
- The biggest remaining risks are not in the sandbox but in the request: (1) Ornith and Bolt run on llama.cpp, whose server
  applies `min_p = 0.05` by default, Tona's sampler has no `min_p`; (2) thinking is off for models whose card numbers are
  thinking-on numbers; (3) Bolt in particular reasons by design; (4) `passed2` is "within 4 attempts" and must never be
  quoted as Aider's pass_rate_2; (5) n = 22 / 39 and one sample per exercise make the decision rule (net +5, p < 0.05)
  almost unreachable for Rust.
- In the raw replies the failures are model failures: Rust runs show compile errors (E0308, E0599, lifetimes) that rustc
  explains correctly, 1 of 88 Coder Rust attempts and 1 of 60 Ornith attempts hit the 16384-token limit, no
  reasoning text in the content (`reasoning_chars` is 0 everywhere), no wrong-file edits (`files_parsed` is the one allowed
  file). I found no harness-caused failure.

## Findings: Aider reference (question A)

Sources: `benchmark/benchmark.py` (https://raw.githubusercontent.com/Aider-AI/aider/main/benchmark/benchmark.py),
`aider/coders/wholefile_coder.py`, `benchmark/npm-test.sh`, blog https://aider.chat/2024/12/21/polyglot.html.

- Tries: `--tries` default 2. Retry only when the unit tests fail (`for i in range(tries)`); the test output is the next
  message ("tests are correct, do not change them, fix the code in <files>"). Reported: `pass_rate_1`, `pass_rate_2`,
  `percent_cases_well_formed` (from `num_malformed_responses`). A malformed reply is not re-prompted separately; it just
  makes the tests fail and uses one try.
- Editable files: `files.solution` minus `.meta/**`, `.docs/**`, test and example files, **`Cargo.toml` and
  `CMakeLists.txt`**. Test files are re-copied from the original before every try.
- Prompt: `introduction.md` (if present) + `instructions.md` + `instructions.append.md` + addendum ("modify the supplied
  files ... do not change names of functions or classes ... only standard libraries"). `hints.md` is not used.
- Whole format: file name line, fence, complete content; fence without name: use the saw-name, else the single chat file.
- Commands: `pytest`; `cargo test -- --include-ignored`; `go test ./...`; JS `sed 's/\bxtest(/test(/g' *.spec.js; npm run
  test`; Java removes `@Disabled`. Timeout 180 s per test run.
- Selection: the 225 hardest of 697 Exercism exercises (solved by 3 or fewer of 7 models). Sampling and temperature are not
  stated in the benchmark README or blog.

## Deviation table

| # | Deviation | Where in our code | What the standard does (source) | Effect | Fix |
|---|---|---|---|---|---|
| 1 | llama.cpp samples with its own defaults on top of ours: `min_p` 0.05, `top_k` 40 when we send none (Bolt). Tona has no `min_p`. | `bench/polyglot/model_job.sh`, `w170rg/job.sh` (SAMPLING), `engine/server/sampling.cpp` (no min_p) | Model cards give temperature, top_p, top_k (and sometimes min_p); llama.cpp `/completion` defaults: `min_p` 0.05, `top_k` 40 (https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md). The server's `repeat_penalty` default is inconsistent in that README, unverified. | Biases between models (Ornith/Bolt get a tail cut the Coder does not; favours Ornith/Bolt slightly). | Send `min_p: 0`, `repeat_penalty: 1.0`, explicit `top_k` for every arm, or state the difference in the result. Check the llama.cpp slot log for the effective sampler chain. |
| 2 | Thinking off for all arms, but the sampling comes from the thinking-mode card rows (`2026-11-15-model-eval-ornith-bolt.md`, line 45 "thinking on"). Bolt reasons first by design (same file, line 224). | `w170rg/job.sh` (`--reasoning off`), `bench/agent/suites.py` SAMPLING | Vendors report thinking-on numbers with thinking-mode sampling; Qwen publishes separate settings per mode (https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct: 0.7 / 0.8 / 20 / rep 1.05, 65536 output tokens). Artificial Analysis: model lab's recommendation overrides its default, reasoning models get the maximum disclosed output length (https://artificialanalysis.ai/methodology/intelligence-benchmarking). | Biases between models (thinking-capable models are measured off-distribution) and makes our numbers non-comparable to vendor numbers. For Bolt it may invalidate "no clear gain". | Report thinking-off as a separate, labelled mode; for a vendor comparison add a thinking-on run with the card's thinking settings and a larger `max_tokens`. Read Ornith's and Bolt's non-thinking recommendation from the model cards (not verified here). |
| 3 | `passed2` means "within 4 attempts" but is named like Aider's pass@2; `run.py` docstring still says "once" / "pass@2 as in Aider". | `bench/polyglot/run.py` (docstring, `passed2`), `report.py` | Aider: `pass_rate_2` = within 2 tries. | Harmless if labelled, invalidates a cross-reference to the Aider leaderboard if not. Four attempts have no public counterpart. | Report `pass_rate_1` (`passed_at == 1`), `pass_rate_2` (`passed_at <= 2`, computable offline from the saved files), and `pass_rate_4` separately; fix the docstring. |
| 4 | Retry message lacks "the tests are correct, do not change them"; no `introduction.md`; `hints.md` added; no Aider system prompt (own FILE-block format). | `bench/polyglot/tasks.py` PROMPT, RETRY, `instructions()` | Aider prompts as above. | Harmless between models (same prompt for all); not comparable to the Aider leaderboard in absolute terms. | State "Aider-style, own prompt" in every report; keep identical across arms. |
| 5 | Test output given to the model is the last 6000 characters. Rust/Go put the first errors at the top; a tail cut can drop them. The saved record keeps only 1500 characters, so the effect cannot be measured afterwards. | `bench/polyglot/sandbox.py` `OUTPUT_TAIL`, `run.py` `test_tail` | Aider feeds the test output (truncation rule not verified here). | Biases between models only if models differ in error volume; likely small, unmeasured. | Keep head 2000 + tail 4000; save the full text sent to the model in the record. |
| 6 | Python runs `pytest -x`: only the first failing test is shown. | `sandbox.py` COMMANDS | Aider runs plain `pytest`. | Harmless between models, slightly fewer hints per attempt. | Drop `-x`. |
| 7 | JS prep also un-skips `xit(` and `xdescribe(`. | `sandbox.py` JS_PREP | Aider only `xtest(` (`npm-test.sh`). | Harmless (more tests run, same for all). | Keep; note it. |
| 8 | Test timeout 240 s, 2 CPUs, debug Rust builds, `cargo --offline`. | `sandbox.py` | Aider 180 s. | Harmless. | None. |
| 9 | Only 22 of 30 Rust exercises are valid (reference solution fails in our sandbox). 3 need crates offline (gigasecond, grep, simple-cipher); the other 5 (alphametics, decimal, pig-latin, poker, robot-name) have no dependency in `Cargo.toml`, the cause is not documented. | `valid.txt`, `Containerfile.rust` comment | Aider uses all 30 (network available in its container). | Biases the subset (possibly drops harder exercises), harmless between models but unknown. | Rerun `run.py reference` for those 5 and write the failure reason into the measurement file before quoting "Rust". |
| 10 | Unpinned toolchains: `rust:1-slim`, `golang:1.24`, `python:3.13-slim`, `node:22-slim`. Aider's image is pinned per commit. | `Containerfile.*` | Aider: model + edit format + commit hash reproduce a run (benchmark README). | Invalidates reproducibility, not a comparison inside one night. | Pin by digest; write the digest into every measurement file. |
| 11 | Header-less replies: strict parser (fixed in working tree, not committed). The working-tree fallback takes the longest fenced block when exactly one file is editable. | `tasks.py` `parse_reply` | Aider takes the single chat file for a nameless fence. | Was: invalidates between models (Coder 21 of 61 replies without header). Now matches Aider; commit it. In the raw replies, Ornith has 0 of 66 attempts without a parsed file, Coder 1 of 88 (Rust) and 2 of 145 (Go). | Commit with a test; keep the `files_parsed` field. Multi-file exercises still need the header (Aider has the same limit). |
| 12 | A reply cut off at `max_tokens` (`finish == length`) costs an attempt and the retry says "cut off". | `run.py` `attempt` | Aider: a cut-off reply is just a failing try. | Harmless (1 of 88 Coder Rust, 1 of 60 Ornith Rust). The Coder cases show analysis prose before the file at 16k. | Report truncation counts per arm (report.py already does). Qwen suggests 65536 output tokens for non-thinking Coder models. |
| 13 | Different engines, weight quantisations and KV types per arm (Tona IQ1_M, llama.cpp Q4_K_M, q8_0 KV). | plan table | Known confound; engines are not templates. | Biases between models, known and named in the plan. | Never write "model A is better at Rust" without "as run on engine X at quant Y". |
| 14 | Early-stop (futility at half, net < +2) is an interim look at the data. Bolt stops after 5 Rust with 0 passes. | `w170rg/driver.py` | Sequential stopping inflates false positives only for "success"; futility stops are conservative for the stated rule. | Harmless for the +5 rule, but a stopped language gives no full pass rate for the table. | Say "stopped at n/N" next to every number. |
| 15 | Seed is fixed at 1165 for all requests of an arm. | `job.sh`, `suites.py` | EvalPlus/pass@k practice: independent samples (https://github.com/evalplus/evalplus uses n samples at T 1.0 for EvalPerf). | Harmless for pairing, but a different seed per exercise would be cleaner. | Derive the seed from base seed + exercise index. |
| 16 | JSON suite is greedy ("the seed does not matter") while the other suites sample at T 1.0; BFCL runs at T 0.001. | `bench/agent/suites.py` `tool_json`, `model_job.sh` | BFCL's own default is 0.001 (as our note says). | Harmless if labelled; breaks the sentence "same settings for every suite". | Say so in the report header. |
| 17 | SWE suite: `step_limit` 100, command timeout 120 s, 17 tasks; the 40-minute task cap and "stop at first client timeout" are our own. | `bench/agent/suites.py`, `model_job.sh` | mini-swe-agent `swebench.yaml`: `step_limit` 250, `cost_limit` 3, environment timeout 60 s, no temperature set (https://raw.githubusercontent.com/SWE-agent/mini-swe-agent/main/src/minisweagent/config/benchmarks/swebench.yaml). | Biases against slow or step-hungry models (100 vs 250 steps); the stop rule counts loops as failures, so SWE numbers are not comparable to published ones. 17 tasks give an interval of roughly +/- 20 points. | Report steps used and how many tasks ended on the step limit, loop stop or cap. A published-comparable run needs 250 steps and the full Verified set. |
| 18 | `report.py` and the decision rule treat one sample per exercise as the model's ability. | `bench/polyglot/report.py`, plan | Miller, "Adding error bars to evals" (https://arxiv.org/abs/2411.00640): report standard errors, use paired differences, plan power; Artificial Analysis averages repeats (e.g. 3). | See next section. | Repeat runs with different seeds; report the paired difference with an interval, not only McNemar p. |

## Statistics (question E)

- Exact McNemar on discordant pairs is the right paired test for two models on the same items. It ignores sampling noise of
  the model itself: with one sample at T 1.0, an exercise that the model solves 50 % of the time is a coin flip, so the
  test answers "did these two draws differ", not "does model A beat model B".
- Power: with n = 22 and the rule "net >= +5 and p < 0.05" Rust needs 6 wins to 0 losses (exact two-sided p = 0.031);
  5 to 0 gives p = 0.0625, 7 to 1 gives p about 0.07. At n = 39 (Go) 8 to 1 or better is needed. The rule can only be met by a
  large effect; a null result means "not shown", not "no gain". This follows from the binomial, nothing to look up.
- The Wilson interval on 22 items is about +/- 20 points around 30 %. Differences of a few exercises are noise.
- Recommended: several samples per exercise (pass@k with the unbiased estimator of Chen et al. 2021 when sampling n > k;
  or the mean of n independent runs as pass@1), paired analysis on exercise means, and effect size with an interval. EvalPlus
  uses greedy for HumanEval+/MBPP+ and T 1.0 with many samples only for EvalPerf; Artificial Analysis uses T 0 for
  non-reasoning models and repeats for noisy evals. A mixed policy is normal, a hidden one is not.
- Our repeated runs already exist for the agent night (seed = base + 1000 x repeat); use them as repeats and report the
  spread between repeats.

## Effect on the run in progress (Ornith/Bolt, Rust/Go)

Raw replies are saved, so re-scoring offline is possible only for attempt 1 (later attempts depend on the earlier feedback).
Rows 1, 2 (request settings) would need a rerun; rows 3, 5, 9 and 18 are reporting issues; the parser fix (11) is in
force for all arms of this run (Coder was rerun after it; `*.strict.jsonl` files are the old ones).

## Checklist before every eval run

1. Write the plan first: scope, expectation with tolerance, active tooling, decision rule with the power it can reach.
2. Same prompt, same parser, same sandbox image (digest written down) for every arm; commit the harness before the run
   (Aider labels uncommitted runs "dirty").
3. Parser: run it on the smoke-test replies of every arm; count replies without a parsed file per arm and per attempt.
4. Sampling: send every field explicitly (`temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `seed`) and read the
   server's effective values from its log; note engine, quantisation, KV type, thinking mode.
5. Thinking mode per arm: say which it is, which card row the sampling comes from, and the `max_tokens`; do not use
   thinking-on numbers as a target for a thinking-off run.
6. Reference check: every reference solution passes in the sandbox; list excluded exercises with the reason.
7. Test files and manifests are not editable by the model; ignored and skipped tests run (`--include-ignored`, `xtest`).
8. Name the rate: pass_rate_1, pass_rate_2 (<= 2 attempts), pass_rate_4; never "pass@2" for 4 attempts.
9. Report per arm: truncations, no-file replies, loop/timeouts, early stops, steps used (SWE).
10. Read ten raw replies per arm and one failing test output before believing a number.
11. Use repeats with different seeds; paired difference with an interval, not only a p value; state n and what the rule
    can reach.
12. Label what is not comparable to published numbers (own prompt, 4 attempts, 100 SWE steps, subset size).
