# 2026-11-19: Rust and Go with Ornith-1.5-35B-A3B and Nemotron 3.5 Bolt 74B-A9B, against the Coder

Status: finished with deviations (parser fallback, Bolt not run, Ornith Go stopped at 9/39); plan unchanged below, deviations and results at the end.

## Question and decision

The Coder (ISTA expert-pruned Qwen3.8-Flash-Next Coder IQ1_M, pinned shard 2026-10-19) solved 0/10 Rust and 1/17 Go on
the polyglot agent eval of 2026-11-18 (`2026-11-18-agent-eval-shard.md`, `research/69-coder-rust-evidence.md`). Question:
does Ornith-1.5-35B-A3B or Nemotron 3.5 Bolt 74B-A9B write clearly better Rust or Go, so the installer can recommend it
for those languages?

**Decision rule (per model and language, written before the run, unchanged):** paired against the Coder on the same
exercises, the model is "recommended for <language>" if it solves at least 5 more exercises net and the exact McNemar p
is below 0.05; "no clear gain" otherwise. Also reported: each model's absolute pass rate with a Wilson interval.

## Scope

Polyglot suite, Rust and Go only, all valid exercises: 22 Rust and 39 Go = 61 (`/var/tmp/claude/139/valid.txt`, data
`/var/tmp/claude/139/polyglot-benchmark-main`). Harness `bench/polyglot/run.py` (the one `bench/agent` calls): per
exercise the instructions and stubs, answer in FILE blocks, tests in the rootless sandbox, up to **4 attempts** with the
test output fed back in one conversation, `--max-tokens 16384`, server context 65536, **thinking off for all three**.
Order within a model: Rust (22) first, then Go (39), in the harness's fixed shuffled order (seed 139), so every model sees
the same sequence. Order of models: Coder, Ornith, Bolt; one `gpu` queue job each (labels `w170rg-coder`,
`w170rg-ornith`, `w170rg-bolt`), a marker `/var/tmp/claude/w170rg/<model>.DONE` after each.
A driver (`/var/tmp/claude/w170rg/driver.py`, outside the repository, not part of the harness) calls `run.py` once per
exercise and adds the rules below; `run.py` itself is unchanged.

- **Wall cap:** 40 minutes per exercise for every model alike (all 4 attempts together). An exercise that hits it is
  killed, counted as failed and listed as "loop/timeout" (`loop_timeout` in the result file).
- **Smoke test:** before each long run, one exercise (the first of the Rust order) on the model's own server, saved in
  `smoke.jsonl`; I read its FILE block and chat template handling before the run is allowed to go on (the driver starts
  right after in the same job; a bad format means I stop the job by PID and note it).
- **Coder results are all re-run** (61 exercises), not reused from 2026-11-18: that run used another build (the worktree
  build `w9`, not main) and only covered 10 Rust and 17 Go, so the settings are not identical.

## Why this size

The question needs a paired difference of 5 net exercises on the same exercises. The valid pool is only 22 Rust and 39
Go, so all of them are used (no sampling of a subset); one pass, no repeats: with the Coder at about 0 to 10 % a model that
solves 5 or more net is at 25 % or more on Rust and 13 % or more on Go, which one pass shows; the rule is stated in
exercises, not in percent. A smaller run could not reach +5 net for Rust at all below 22. No effect is expected to hide
in sampling noise at that size; a result close to the +5 line is reported with its counts, not rounded.

## Early-stop rules (2026-11-19; do not test what is known)

1. **Futility, per model and language:** after half of that language's exercises (11 Rust, 20 Go) compare paired with the
   Coder on the same exercises. If the model's net gain is below +2 the language is stopped for that model: it cannot
   plausibly reach +5 net with p < 0.05. Recorded as "stopped for futility at n/N" with the counts so far
   (`STOPPED-<lang>` file). The Coder arm itself has no futility check.
2. Success is not stopped early: a model that passes the check runs the whole language, the full pass rate goes into
   the installer table.
3. **Bolt:** if the smoke test or the first 5 Rust exercises show the same failure types as the Coder (no FILE block,
   invented APIs, compile errors in every attempt) and 0 passes, Bolt is stopped and that is recorded (the driver stops
   after 5 exercises with 0 passes; I read the failure types and resume only if they differ from the Coder's).
4. Rust runs before Go for every model, so the check comes early.
Every stop of a job or server by me is noted below with time and reason (by PID only).
If the Bolt GGUF does not load in the llama.cpp build we have (`work/build/llama.cpp-hip`), that is the result: I do not
update or rebuild llama.cpp (coordinator's rule), and Bolt is skipped. The 1.0 GB LiLiCorr file is NVIDIA's 1.13B
speculative drafter, not a quality correction (`research/70-ornith-bolt-rust-go.md`); it is not used.

## Models, engines and settings

| Arm | Engine | File | Server flags | Sampling (per model card) |
|---|---|---|---|---|
| Coder | Tona `tona-server`, build of main (`work/build/w170`, commit `1db5a2d`, via the cpu queue), production mode, MTP drafter | `coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-*.gguf` (pinned shard 10-19), drafter `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | `--ctx 65536`, `TONA_EXPERT_PROFILE=experts-coder-1019.bin`, no `TONA_DETERMINISTIC`, default `TONA_*`, thinking off (engine default) | temperature 1.0, top_p 0.95, top_k 20, seed 1165 (as the night's arm) |
| Ornith | llama.cpp HIP build (`work/build/llama.cpp-hip`, `scripts/baseline.sh` flags) | `Ornith-1.5-35B-Q4_K_M.gguf` | `-c 65536 -np 1 -ngl 999 --cpu-moe --moe-cache-mib 8192 -fa on -ctk q8_0 -ctv q8_0 -t 16 --jinja --chat-template-kwargs '{"enable_thinking":false}'` | temperature 1.0, top_p 0.95, top_k 20 (card: 1.0 reproduces the vendor numbers, `2026-11-15-model-eval-ornith-bolt.md`), seed 1165 |
| Bolt | llama.cpp HIP build, same flags | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` | same as Ornith | temperature 1.0, top_p 0.95, top_k server default (card "Recommended Sampling"), seed 1165 |

Why llama.cpp for Ornith and Bolt: Tona runs these architectures only on its CPU reference path, so llama.cpp is the
engine here; the engines differ (a comparison of pass rates, not of speed). `-t 16` is the physical core count of this
CPU (`scripts/baseline.sh`). Servers run in `systemd-run --user --scope -p MemoryMax=54G -p MemorySwapMax=0` on port
8091. Decode tok/s per engine are a side note only (different engines, no speed claim).

## Expectation (with tolerance and source)

No per-language Rust or Go number exists for either model (`research/70-ornith-bolt-rust-go.md`), so the tolerances are wide.
- **Coder:** Rust 0 to 2 of 22 (0/10 on 2026-11-18), Go 1 to 6 of 39 (1/17 new shard, 3/17 old shard), tolerance +/-3.
- **Ornith:** Rust 25 to 75 % and Go 20 to 70 % within 4 attempts. Source: vendor SWE-bench Multilingual 71.4, third-party
  MultiPL-E Rust 0.75 for the base model (single-function problems, easier than Exercism with 4 attempts), no Go number,
  Q4_K_M quality unmeasured, thinking off. Central guess: it clears +5 net on both languages; a result below that is a
  finding against the vendor numbers.
- **Bolt:** Rust and Go 5 to 50 %; no coding accuracy published, one failing Go report (HF discussion), pruned with 2-bit
  experts like the Coder. Central guess: between the Coder and Ornith.
- Duration: Coder about 2 to 3 h (61 exercises, failures use all 4 attempts, MTP decode), plus about 5 min server start;
  Ornith and Bolt 2 to 4 h each (decode 30 to 40 tok/s, up to 4 x 16k tokens worst case), less when stopped for futility.
- Active tooling: no profiler, no diagnostics beyond the defaults, no `TONA_DETERMINISTIC`, desktop guard on, queue kind
  `gpu`. Quality run, no speed claim. `hw_report` and `quiet_check` before and after every model in the model's run
  directory.

## Test conditions

Per model in `/var/tmp/claude/w170rg/<model>/`: `hw_report.txt`, `quiet_before.txt`, `quiet_after.txt`, `server.log`,
`server.cmd`. Summaries are copied below with the results.

## Results

### Deviations from the plan (all decided on 2026-11-19, in the order they happened)

1. **Parser fallback (harness change, all models alike).** The Ornith smoke test (rust/react, strict parser) found 4 of 4 attempts
   without a `FILE:` line: the code was in a plain ```rust block, so the strict harness scored "no file". The Coder had
   the same habit in 44 attempts of its first run (41 of them with a fenced block). `bench/polyglot/tasks.py parse_reply`
   now takes, when no FILE block is found and the exercise has exactly one editable file, the longest fenced block (as
   Aider's whole-file coder does for a single chat file; `research/71-agentic-eval-methodology.md`). Prompts are unchanged.
   Where a conversation had no header-less reply the two parsers behave identically, so those 40 Coder results (13 Rust,
   27 Go) from the strict run were kept; the 21 Coder exercises with at least one header-less attempt (9 Rust, 12 Go) were
   deleted and rerun with the new parser (08:09 to 08:51). The strict files are kept as `coder/{rust,go}.strict.jsonl`
   and `ornith/smoke-strict.jsonl`. Unit test: `tools/tests/test_polyglot_parse.py`.
   Strict Coder numbers for the record: Rust 2/22, Go 6/39; with the parser fallback Rust 2/22, Go 7/39.
2. **Reasoning switch:** llama.cpp deprecates `--chat-template-kwargs '{"enable_thinking":false}'`, so Ornith ran with
   `--reasoning off` (same effect, thinking off; 0 reasoning characters in all replies).
3. **Failures before the long runs:** first chain start failed instantly (`job.sh` not executable); the first Ornith and Bolt
   jobs failed instantly with "couldn't bind port 8091" (the Coder server's socket was in TIME_WAIT; the job now waits for
   any socket state). No GPU time was lost; the failed directories are `ornith-failed-bind`, `bolt-failed-bind`.
4. **Stop log** (by PID, 2026-11-19): about 08:05, I stopped the first Ornith job (chain.sh 2775340, driver 2781987/2781988,
   gpu_queue 2775342) after the smoke test showed the missing FILE header; the lock was released at once. About 10:15 the
   coordinator stopped chain.sh (PID 2782625) so **Bolt did not run**: reason, Bolt is not a recommendation candidate
   (maintainer, 2026-11-15, #144 comment) and Ornith and Bolt will be measured on Tona, where the sampling equals the Coder's.
   10:18 (log line in `ornith/STOPPED-go-manual`), about 10:40 per the coordinator: Ornith Go on llama.cpp stopped by
   hand (driver 2919301 and its run.py child) at 9 of 39 exercises; reason: "maintainer: measure Ornith on Tona
   instead (equal sampler); llama.cpp Go not needed". **The Go numbers for Ornith are incomplete (9 of 39) and are not a
   verdict.**
5. **Bolt: not run.** It was never loaded in llama.cpp (no loading test, no smoke test), so nothing is known from this run
   about whether the stock build runs the file; the plan's Bolt rules (zero-5 stop, load check) were not exercised.
   The LiLiCorr file is NVIDIA's speculative drafter and was not used.

### Sampling caveat (equal sampling is not given)

llama.cpp applies server defaults when a request sends none: `min_p` 0.05 and `top_k` 40 where none is sent (Ornith got
top_k 20 and top_p 0.95 explicitly, so only the **min_p 0.05 tail cut** differs). Tona has no min_p, so the Coder ran
without a tail cut that Ornith had. The bias favours Ornith slightly; the positive Rust result needs confirmation on
Tona with equal sampling. Different engines also mean different chat template renderers and sampler implementations.

### Results (pass in 4 attempts = `passed2`; first try = pass@1; "within 2" = `passed_at` <= 2, comparable to Aider pass_rate_2)

| Model | Language | n | pass in 4 attempts (Wilson 95 %) | first try | within 2 attempts (Wilson) | loop/timeout |
|---|---|---|---|---|---|---|
| Coder (Tona, shard 10-19) | Rust | 22/22 | 2 (9 %, 3-28 %) | 0 | 0 (0-15 %) | 0 |
| Coder | Go | 39/39 | 7 (18 %, 9-33 %) | 1 | 4 (10 %, 4-24 %) | 0 |
| Ornith Q4_K_M (llama.cpp) | Rust | 22/22 | 8 (36 %, 20-57 %) | 4 | 6 (27 %, 13-48 %) | 0 |
| Ornith | Go | **9/39, incomplete** | 7 of 9 (78 %, 45-94 %; first 9 of the fixed order, not a representative rate) | 2 | 4 of 9 | 0 |
| Bolt | Rust, Go | not run | | | | |

Paired verdict against the Coder (rule: net +5 or more and exact McNemar p < 0.05):

| Model | Language | both | only model | only Coder | neither | net | exact McNemar p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Ornith | Rust | 2 | 6 | 0 | 14 | +6 | 0.031 | **recommended for Rust by the rule**, with the min_p caveat above: confirm on Tona with equal sampling |
| Ornith | Go (9 of 39) | 2 | 5 | 0 | 2 | +5 | 0.0625 | **no verdict, incomplete**: net +5 on 9 pairs but p above 0.05 and 30 exercises missing |
| Bolt | both | | | | | | | not run |

Rust detail: Ornith passed grade-school, book-store, variable-length-quantity, macros, luhn-from, two-bucket, word-count,
accumulate; the Coder passed macros and two-bucket (the two shared passes, both of them also found by Ornith). Pass attempt
distribution Ornith Rust: 4 on the first, 2 on the second, 2 on the fourth. Go (first 9 of the order): Ornith failed only
robot-simulator and forth, both also failed by the Coder; the Coder passed book-store and counter.

Futility log: Ornith Rust at 11/22: net +2, not below +2, so the run continued (finished 22/22). Ornith Go: no check
(stopped by hand at 9/39). Coder: no check by design. No exercise hit the 40-minute wall cap in any arm (0 loop/timeout).
Failure types in one line each (heuristic from the last attempt's test output, not hand-read): Coder Rust, all 20 failures
end in compile errors (invented APIs, wrong syntax such as `class`, `Vec()`, `range()`); Coder Go, 32 failures, mostly
compile or test failures in the last attempt, plus attempts without a FILE block (1 after the fallback); Ornith Rust, 14
failures split about evenly between compile errors (7) and failing tests (7), 0 attempts without code; Ornith Go too small
to characterise (2 failures, both with errors in the last attempt).

Side note, decode rate (different engines, no speed claim): Ornith on llama.cpp about 45 tok/s decode in the smoke run
(MoE experts in RAM, GPU cache 8 GiB); the Coder on Tona with MTP 107 tok/s on a 2119-token reply (token per pass 6.4).
Wall time of the sums of requests: Coder Rust 1,963 s and Go 2,569 s (4 attempts where failing), Ornith Rust 3,715 s.

### Test conditions

Hardware report (same for every job, `ornith/hw_report.txt`): Ryzen 9 9950X, 16 cores / 32 threads; RX 9060 XT gfx1200,
15.9 GiB VRAM, 1.26 GiB held by the desktop before the Ornith run (1.69 to 1.89 GiB before the Coder run); 60.5 GiB RAM,
56.0 GiB available; Fedora 44, kernel 7.3.4; ROCm 10.1; engine commit `1db5a2d`/branch head of the plan commit.
Quiet check (`python3 tools/quiet_check.py`; the report files are in `/var/tmp/claude/w170rg/{coder,ornith}/`):

| Run | before | after |
|---|---|---|
| Coder (first run, strict parser) | warning: 1.69 GiB VRAM held by others (06:39) | warning: 1.89 GiB held (08:01) |
| Coder (rerun, 21 exercises) | warning: 1.89 GiB held (08:09) | warning: 1.26 GiB held (08:51) |
| Ornith | warning: 1.26 GiB held (08:52) | warning: 1.31 GiB held (10:18) |

The warnings are the desktop's own VRAM share (the owner agreed to the run on this machine; the desktop guard stayed on, no
exit 75 occurred). Swap in use 1.4 to 2.0 GiB before the runs. The Coder server ran with 76 expert slots per layer (VRAM
13.37 GB). An unrelated page-cache or CPU load from others was not seen (CPU use by others 0.1 % before the Ornith run).
Not measured, so not claimed: speed, the effect of the 40-minute cap (never reached), Bolt, Go for Ornith beyond 9
exercises, Ornith on Tona.

### Answer to the question

Ornith-1.5-35B-A3B Q4_K_M on llama.cpp solved 8 of 22 Rust exercises against the Coder's 2 (6 only-Ornith, 0 only-Coder,
p = 0.031): the decision rule is met for Rust, but with a slightly favourable sampler (min_p) and a different engine, so
it should be confirmed on Tona with the Coder's exact sampling before the installer text says so. Go is undecided (9 of 39
run). Bolt was not tested.
