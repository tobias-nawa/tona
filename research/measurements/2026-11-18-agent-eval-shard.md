# 2026-11-18: agentic evaluation of the two Coder shards, old 09-29 against pinned 10-19 (#165)

Status: plan and pilot. Night 1 is armed to start back to back with the scale run: it starts when the file
`/var/tmp/claude/scale1118/DONE` exists (written by the scale-run worker after its last block; between blocks no queue ticket
is visible, and an earlier start would evict the page cache). Armed 2026-11-18 13:19 CET.
Actual start time: to be filled in from `night1.out`. Results are appended below the plan.

## Question and decision

The teacher-forced NLL on 25 stored agent turns says the old shard (09-29) is 0.21 nat better; public texts, tool-call
JSON and bench-effort say equal (`2026-11-17-shard-agent-turns.md`). NLL on those turns mostly measures closeness to
Claude's style, not task success. Question: does the old shard solve more real tasks than the pinned one?

- Old clearly better (lower bound of the paired difference above 0 in SWE or polyglot, or pooled): propose re-pinning to
  09-29 to the maintainer (the pin stays untouched until decided).
- No resolvable difference: say "inconclusive at X points" and keep the pin; the shard question then rests on the
  smaller measured effects only.
- New better: the pin was right; the NLL gap was style.

## Changed purpose (2026-11-18): a guard, not a ranking

The decision is made: **the pin stays on the new shard (10-19)**. Public evidence
(`research/64-coder-shard-public-evidence.md`): the only comparison of the two revisions is ISTA's own, +3.6 points
SWE-bench Verified in one run without spread, which is far below what one night resolves (about 16 points). Night 1
therefore guards against a **large break** of the new shard in our engine and on our task mix; it cannot rank the shards
and is not read as a ranking. The section "Question and decision" above is superseded by these rules:

1. The new shard is clearly worse overall, beyond the night's resolution (pooled SWE plus polyglot: the 95 % interval of
   B - A lies entirely below 0, or the exact McNemar test favours the old shard at p < 0.05): reported as a finding,
   and the pin is reconsidered by the maintainer.
2. Otherwise the pin stays and **there is no night 2** (the night-2 command in the plan is withdrawn).
3. Polyglot is reported **per language** (Python, Go, Rust, JS; `report.py` prints one row per language). ISTA says the
   expert selection did not preserve Rust, so a Rust-only drop is a finding even if the total is fine. The per-language
   cells are small (about 18 pairs each, resolution about 35 points or more), so a Rust finding means: the old shard
   solves clearly more Rust exercises (only-old minus only-new of 3 or more, or p < 0.05); it is then reported with
   the counts and not treated as proof.

Infrastructure failures (server crash, timeout, grading error) are listed separately from model failures in every report.

## Changed goal (2026-11-18): both shards are offered in the installer

Both shards stay on disk (old 09-29 files are kept, nothing is deleted) and the installer will offer each with the
languages we recommend it for and the reason. **The per-language polyglot split is now the main output of the night**,
not a side check; the guard rule 1 above still applies. Measurement only: no code or installer change.

Report per language (Python, Go, Rust, JS), one row each, from the polyglot pairs of night 1 (about 18 exercises per
language, 72 of the 144 valid ones; the SWE and json suites stay separate):

| Language | both | only old | only new | neither | Verdict |
|---|---|---|---|---|---|

Verdict rule, fixed before the run: **"old better"** if only-old minus only-new is 3 or more; **"new better"** if
only-new minus only-old is 3 or more; otherwise **"no difference seen"** (this is not "equal": the night resolves about
20 to 30 points per language, see below). The exact McNemar p-value and the 4-attempt versus first-try split are given
next to every verdict; a verdict by the 3-task rule with p above 0.05 is marked weak.

Follow-up size for a 3-task difference per language (Connor, 80 % power, alpha 0.05, `pairs_needed`): 3 tasks of 18 are
16.7 points; that needs **55 pairs per language at 20 % discordance, 83 at 30 %, 111 at 40 %**. The valid pool has 36
exercises per language (144 over four), night 1 covers 18 of them, so a follow-up would run the other 18 and then
repeat all 36 with new seeds, 2 to 3 repeats per exercise for 55 to 111 pairs (repeats of one exercise are not
independent tasks, so the real power is somewhat lower). That is about 2 to 3 nights per arm for all four languages, or
about half of it for one language (Rust alone: 36 exercises x 2 to 3 repeats, about 1 to 1.5 h per arm at 120 s per
exercise). Whether to run it is the maintainer's decision after the night-1 table; it is not launched with night 1.

## Change during the run: loop stop for SWE tasks (2026-11-18 16:47, coordinator finding)

Finding: arm old, r1, stuck on `django__django-15563` from 14:46 to 16:47. The server log shows the same request 11
times: 22,833 prompt tokens (all reused), then 38.8k generated tokens in 600 s (2.01 tokens per pass), then the client
timeout and an identical retry: a repetition loop. A retry with the same seed repeats it, so the task would have run to
the 5 h container cap. Both arms get the same rule, written before arm new starts:

- `bench/agent/suites.py`: each SWE task is a call of its own (finished tasks are skipped, so an interrupted arm
  resumes). The first client timeout (`APITimeoutError` in the agent output) stops the task, as does a wall-clock cap of
  **40 min per task** (5 x the 8.5 min median of the 2026-11-16 run; the first eight tasks of this night took 2 to 5
  min). The container is removed. The task is scored failed with reason "loop/timeout", listed in `swe.loops.jsonl`.
- It is a **model failure**, counted apart from infrastructure failures (server crash, grading error); `report.py`
  prints the number of loop/timeout tasks per arm. This is a number for the shard question: ISTA says the update
  reduced loops.
- Arm old: its first eight tasks keep their results (not redone). The running task was stopped by hand at 16:47 (agent
  process ended by PID, container removed) and entered in `swe.loops.jsonl`. The block continued with polyglot and json
  on the old code; its SWE result file then listed the nine unrun tasks as failed without being run, so it is deleted
  and the remaining SWE tasks of arm old run in a separate queue job (same server settings, same seed, the new code)
  after the block. The seed of a task does not depend on its position, so these tasks are the same as in an
  uninterrupted run.
- Arm new runs with the cap from its first task.

## Addition: C# (.NET) as a fifth language (2026-11-18; planned before queueing)

Why: ISTA says the expert selection of the new shard preserved C# (`research/64-coder-shard-public-evidence.md`), so C#
is where the new shard should be at least as good as the old one. The result feeds the per-language installer
recommendation. Own suite `csharp` (same method as polyglot: whole files, tests, output fed back, 4 attempts in one
conversation), own result files, so the night-1 polyglot order and its 72-exercise prefix do not change.

- Exercises: exercism/csharp, practice exercises, MIT (commit f4e2411 of 2026-11-15), fetched to
  `/var/tmp/claude/139/exercism-csharp`, not into the repository. Tests run in a rootless Podman container
  (`localhost/tona-poly-csharp`, debian slim with the native libraries only) with no network, as the other four tracks;
  the host's .NET SDK 11.0.100 and an offline NuGet cache are mounted read-only. The exercises target net10.0 and are
  built for net11.0 (`-p:TargetFramework`); `Skip = "..."` is removed from the tests as the other tracks un-ignore theirs.
- Validation (CPU job, 2026-11-18): the reference solution (`.meta/Example.cs`) passes for **143 of 144** exercises
  (`ledger` fails and is dropped, `valid-csharp.txt`), **384 s for all 144, 2.7 s per exercise** for restore, build and
  test, so a test attempt does not dominate the time. Order: valid ids shuffled with the fixed seed of the polyglot
  runner; both arms take the same prefix.
- **Why this size:** question: does one shard solve clearly more C# exercises? 100 of the 143 exercises, 100 pairs: at 20
  to 30 % discordance 100 pairs resolve about 12 to 14 points (about 14 tasks); 143 would resolve 15 to 18 tasks (about
  12 points at 30 %) for 40 % more GPU time; 60 pairs (the lower bound asked for) would resolve about 17 points. A 3-task
  difference per language is not resolvable at this size (3 of 100 is 3 points; see "Follow-up size" above): the table
  states "no difference seen" for it, not "equal". The pool has 143 valid exercises, so a larger run means all of
  them, then repeats with new seeds.
- Duration: about 59 s per exercise (arm old, polyglot, 4263 s for 72) plus up to four test attempts at 2.7 s: **about
  65 s per exercise, 1.8 h per arm**, plus a server start of about 5 min; both arms about 3.8 h.
- Expectation: pass within 4 attempts **40 % (tolerance 25 to 60 %)** for each arm, from the old arm's other languages
  (JavaScript 18/27 = 67 %, Python 12/18 = 67 %, Go 3/17 = 18 %, Rust 0/10 = 0 % on 72 exercises; C# is a mainstream
  language with a strong type checker, so between Go and Python); first try 5 to 15 %. Difference between the arms
  between -10 and +10 points; a difference above 15 points is outside the expectation and a finding. Verdict rule as for
  the other languages: net difference of 3 or more tasks decides ("old better" or "new better", marked weak at p above
  0.05), otherwise "no difference seen".
- Active tooling: as night 1 (`TONA_DIAG` default, no profiler, no deterministic mode, desktop guard on, sampling and
  seed as the other agent runs); the loop stop does not apply (polyglot requests have no agent loop; a polyglot request
  that times out is a model failure of that exercise).
- Queue: `csharp-old`, then `csharp-new`, one `gpu` job each (kind `gpu`), behind everything queued before them; a
  `DONE` marker (`/var/tmp/claude/w159/csharp-DONE`) after the last. Out directory `/var/tmp/claude/w159/agent-shard-cs`.

Correction to "Follow-up size" above: the valid pool is 39 Go, 49 JavaScript, 34 Python, 22 Rust (144), not 36 per
language, and the 72-exercise prefix of night 1 holds 27 JavaScript, 18 Python, 17 Go and only 10 Rust exercises.

## Test conditions that need a note (arm new, r1)

- **Container image build outside the queue, 2026-11-18 about 18:15 to 18:16 CET** (the image `localhost/tona-poly-csharp`
  was created at 18:16:03; the build, one `apt-get install` of two libraries, ran about 40 s), while the job
  `agent-new-r1` (arm new, night 1, started 18:03) held the GPU lock. It was a small CPU and network job started by hand;
  from now on builds of any size go through the cpu queue. A loop, timeout or slow request of arm new in that window
  can be checked against it (the server log of the arm is `new-r1.server.log`).
- **Queue label:** the C# jobs carry the same labels as the night-1 jobs: `agent-old-r1` (csharp-old, ticket 1183, queued
  2026-11-18 18:26) and `agent-new-r1` (csharp-new). In the queue history, a label `agent-old-r1` or `agent-new-r1`
  after 18:26 can be a C# job; the out directory tells them apart (`agent-shard` for night 1, `agent-shard-cs` for C#).
  Later jobs get unique labels (for example `agent-cs-old-r1`).

## Addition: llama.cpp cross-check of the Rust result (2026-11-18; planned before queueing)

Question: the old shard solved 0 of 10 Rust exercises (arm old, r1, 4 attempts; 10 pairs only). Is that a model property
or an engine fault of ours? Decision rule (coordinator): if llama.cpp with the same file solves **0 to 2 of the 10**, it is
the model (the installer text says Rust is weak with both shards); if it solves **3 or more**, there is an engine bug to
hunt and the Rust verdict of the per-language table is void until it is found.

- Scope: the same 10 Rust exercises as in `old-r1-polyglot/polyglot.jsonl` (`nucleotide-codons`, `word-count`, `fizzy`,
  `parallel-letter-frequency`, `two-bucket`, `doubly-linked-list`, `grade-school`, `luhn-from`, `accumulate`,
  `variable-length-quantity`), the same file (old shard 09-29, both parts), the same runner (`bench/polyglot/run.py`,
  same prompts, 4 attempts in one conversation, 16384 max tokens), context 65536, one server start. Two passes on that
  server: **sampled** (the arm's sampling: temperature 1.0, top_p 0.95, top_k 20, seed 1165 = 165 + 1000) and **greedy**
  (temperature 0). The coordinator's text says "same seed, greedy"; the arm itself was sampled, so the sampled pass is the
  like-for-like one and the greedy pass removes sampling noise. llama.cpp's random stream differs from ours, so the
  sampled pass is a second draw, not a replay. A single pass of 10 is a weak test: the rule above is applied to the
  better of the two passes, and per-exercise passes of both are listed.
- Server: the llama.cpp HIP build of `scripts/baseline.sh` (`work/build/llama.cpp-hip`), its settings (`--cpu-moe
  --moe-cache-mib 8192 -fa on -ctk q8_0 -ctv q8_0 --jinja`), `-t 16` (physical cores of this CPU; the script's 6 was the
  old CPU), port 8091, `systemd-run --user --scope -p MemoryMax=54G -p MemorySwapMax=0`, one `gpu` queue job, label
  `agent-rust-llamacpp`, started after the file `/var/tmp/claude/w159/csharp-DONE` exists. No drafter on llama.cpp (the
  GGUF has no nextn layers; speculation is lossless).
- Equal request side: our server's default is thinking off (`think_default` false; the arm's results hold no reasoning
  text, 0 reasoning characters over all 72 exercises); llama.cpp gets `--chat-template-kwargs
  '{"enable_thinking":false}'` with the GGUF's own template (`--jinja`). The runner sends the same messages and no template
  fields to both. **Differences that cannot be removed:** the chat template is rendered by two implementations (ours
  `chat_qwen3_render.cpp`, llama.cpp's jinja); the check is that the prompt token counts of the first attempt per exercise
  match the arm's (`prompt_tokens` in the result lines); the KV cache is int8 on llama.cpp and the engine's own format on
  ours; the sampler implementations differ.
- Expected duration: **20 to 30 min** for Rust alone (server start about 5 min; 10 exercises x 2 passes x about 60 s; with Go see below).
- Expectation: **0 to 2 of 10** per pass if it is the model (the engine's 4-attempt failure types were compile errors of
  the model's Rust in 7 of 10, no parseable file in 3: `fizzy`, `two-bucket` and `doubly-linked-list` first attempts).
- Report: passes per exercise for both engines side by side, with sampled and greedy as separate columns (the bug alarm
  stays on the better pass: 3 or more of 10 in either), and one line on whether the failure types look alike. The
  three exercises whose first attempt had no parseable FILE block on our engine (`fizzy`, `two-bucket`,
  `doubly-linked-list`) are reported separately for both engines: valid FILE blocks on llama.cpp and none on ours would
  point at our template or stop handling, not at Rust skill.
- **Extension to Go (2026-11-18, coordinator, before the job starts):** the same job, the same server start and the
  same two passes also run the **17 Go exercises** of old-r1 (`say`, `robot-simulator`, `connect`, `octal`, `dominoes`,
  `food-chain`, `poker`, `kindergarten-garden`, `zebra-puzzle`, `paasio`, `dnd-character`, `hexadecimal`, `markdown`,
  `alphametics`, `crypto-square`, `word-search`, `simple-linked-list`; arm old solved 3 of 17). Reason: for Rust the public
  evidence (ISTA's own statement that the selection did not preserve it, `research/69-coder-rust-evidence.md`) explains the
  result; for Go there is no public data, so this check is the only way to separate an engine error from a model property.
  **Go rule: llama.cpp solves 7 or more of 17 in either pass means an engine bug to hunt; 0 to 6 means a model property.**
  The Rust rule stays as written above. Extra duration: 17 x 2 passes x about 60 s, **30 to 40 min more**, so the whole
  job takes about **50 to 70 min**. Order on the server: Rust sampled, Rust greedy, Go sampled, Go greedy; results in
  `rust-llamacpp/` as `sampled.jsonl`, `greedy.jsonl` (Rust) and `go-sampled.jsonl`, `go-greedy.jsonl`. Label unchanged.
- Active tooling: llama.cpp defaults, no profiler; the desktop guard is on; quiet check before and after in the out
  directory `/var/tmp/claude/w159/rust-llamacpp/`. Marker after the job: `/var/tmp/claude/w159/rust-llamacpp-DONE`.

## Method (`bench/agent/`, `run.py compare`)

Same build for both arms (`work/build/w9`, commit a1fda4c, `TONA_GEMMQ_FP8` off, production mode, MTP draft, no
`TONA_DETERMINISTIC`). Arms differ only in the model file and the matching expert profile (`experts-coder-0929.bin`,
`experts-coder-1019.bin`). Sampling as the model card and all agent runs since #139: temperature 1.0, top_p 0.95,
top_k 20, seed = 165 + 1000 x repeat, the same for both arms. Jobs: A r1, B r1 (one queue job each, kind `gpu`, quality
without timing claim). Suites are reported separately:

| Suite | Tasks | Agentic form |
|---|---|---|
| SWE-bench Verified subset | 17 (`bench/swe/subset.txt`) | mini-swe-agent shell agent in rootless Podman, step limit 100, official grading |
| Polyglot (Python, Go, Rust, JS) | first 72 of the fixed shuffled order of 144 valid exercises | write files, tests run, output fed back, up to 4 attempts in one conversation |
| Tool-call JSON | 32 | single call, greedy; a gate, not a discriminator |

Dropped: own effort tasks (the pilot needed 450 s per task at level high, and the earlier runs show 43/46 passed, near
the ceiling), Terminal-Bench (Docker only, near the floor), SWE-bench Pro (too hard for this model).

## Size by decision

Paired exact McNemar, 80 % power, alpha 0.05, Connor's formula (`bench/needle/stats.py`). **X = 10 points** is the
difference we want to resolve; pairs needed: 234 at 30 % discordant pairs, 155 at 20 %, 312 at 40 %.

Budget: about 5 h GPU per arm. Measured pilot (old shard, 2026-11-18): **polyglot 957 s for 8 exercises = about 120 s per
exercise** (up to 4 attempts), json 29 s, effort 2,245 s for 5 tasks; SWE about 8.5 min per task (17 tasks in 144 min
in the 2026-11-16 run, `2026-11-16-pc-dm-vs-coder.md`); server start about 5 min.

| Part per arm | Pairs | Time per arm |
|---|---|---|
| SWE 17 | 17 | about 145 min |
| polyglot 72 | 72 | about 145 min |
| json 32 | 32 | 1 min |
| two server starts and grading | | about 15 min |
| Total | 121 | about 5.1 h; both arms about 10.2 h |

**One night does not reach X = 10.** 72 polyglot pairs resolve about 18 points at 30 % discordance, SWE (17 pairs) about 40;
the sum of 89 SWE and polyglot pairs about 16. 10 points needs 234 pairs, that is about 2.5 nights per arm for polyglot
alone; the result file says what the night resolved, and a second night (exercises 73 to 144, then repeats with a
new seed) is the way to X = 10. If the shard effect is small (as the public-text and JSON results suggest), "inconclusive"
is the expected verdict and is reported as such, not as a tie.

## Expectation (written before the run)

- Pilot, old shard: polyglot 2 of 8 passed within 4 attempts, bench-effort 4 of 5, json 32 of 32.
- Old shard SWE 8 to 13 of 17 (the pinned shard solved 12 of 17 on 2026-11-16, one pass, sampling noise about 2 tasks);
  polyglot pass within 4 attempts 25 to 45 %; json 31 to 32 of 32.
- Difference old minus new: between 0 and +8 points (direction from the NLL result), true effect below 5 points more
  likely than not. Probability that the night shows a significant difference (p < 0.05) in any suite: about 20 %. A
  difference above 15 points or new better by more than 5 points is outside the expectation and a finding.
- Risk: SWE tasks that run into the 5 h cap or the step limit without a patch (counted as failed, same for both arms);
  a server crash (the block reports it and the file of that arm and suite is missing, the pair is then dropped).

## Active tooling

`TONA_DIAG` default, no profiler, no deterministic mode, desktop guard on (jobs wait while the desktop is in use). Other
workers share the machine through the queue; the quiet check ran before and after every block
(`<arm>-r<k>.quiet_check*.txt`, `.hw_report.txt` in the run directory).

## Result of night 1 (2026-11-18, 14:16 to 20:27; arm old r1 and arm new r1)

Run as planned except for the loop stop and the old arm's SWE rest (see "Change during the run"). Test conditions: `old-r1.hw_report.txt`
and `new-r1.hw_report.txt`, quiet checks `*.quiet_check*.txt` in `/var/tmp/claude/w159/agent-shard/`. Before arm new the quiet
check said "do not measure" (other programs used 47 % of the CPU threads, load average 29, 1.9 GiB of VRAM held by others);
this is a quality run without a timing claim, and the load lengthened arm new only slightly (polyglot 4,618 s against 4,263 s,
SWE 2,705 s for 17 tasks). Arm old before: warning, 1.9 GiB VRAM held by others, light swapping; after both: VRAM only.
Report file: `agent-shard/report.md` (made by `run.py report`; the end-of-run report of the compare process crashed on a
pair-key bug of the all-suites row, fixed in `report.py`; no result was lost).

| Suite | Pairs | old pass | new pass | both / only old / only new / none | new - old points [95 %] (p) | Verdict |
|---|---|---|---|---|---|---|
| SWE-bench Verified | 17 | 8/17 = 47.1 % [26, 69] | 12/17 = 70.6 % [47, 87] | 6 / 2 / 6 / 3 | +23.5 [-7.1, +54.2] (p 0.289) | new better (weak) |
| polyglot | 72 | 33/72 = 45.8 % [35, 57] | 32/72 = 44.4 % [34, 56] | 25 / 8 / 7 / 32 | -1.4 [-11.9, +9.1] (p 1.000) | no difference seen |
| polyglot / JavaScript | 27 | 18/27 | 18/27 | 15 / 3 / 3 / 6 | +0.0 (p 1.000) | no difference seen |
| polyglot / Python | 18 | 12/18 | 13/18 | 10 / 2 / 3 / 3 | +5.6 (p 1.000) | no difference seen |
| polyglot / Go | 17 | 3/17 | 1/17 | 0 / 3 / 1 / 13 | -11.8 (p 0.625) | no difference seen |
| polyglot / Rust | 10 | 0/10 | 0/10 | 0 / 0 / 0 / 10 | +0.0 | no difference seen (both 0) |
| tool-call JSON | 32 | 32/32 | 32/32 | 32 / 0 / 0 / 0 | +0.0 | gate passed |
| all suites | 121 | 73/121 = 60.3 % | 76/121 = 62.8 % | 63 / 10 / 13 / 35 | +2.5 [-5.3, +10.2] (p 0.678) | new better (weak) |

Resolution (80 % power, alpha 0.05): polyglot 72 pairs at 21 % discordance resolve 15 points or more; SWE 17 pairs at 47 %
discordance 43 points or more; all suites 121 pairs at 19 % about 11 points. First-try passes on polyglot (old): 7 of 72.

Time per task: SWE old about 2.3 min per task for the first eight (18 min), the rest in 1,102 s; SWE new 2,705 s for 17
(2.7 min); polyglot old 59 s and new 64 s per exercise (up to four attempts); json 27 to 29 s.

Loops and failures: **loop/timeout stops in SWE: old 1/17 (`django__django-15563`, 11 identical 600 s requests of 38.8k tokens,
stopped by hand at 16:47 after two hours), new 0/17.** SWE exit statuses of arm new: 14 submitted, 3 limits exceeded
(step limit). Infrastructure failures: none (no error, abort or fault line in either server log; no grading error; every
task graded). One polyglot request of arm old ended at the token limit in Rust (`doubly-linked-list`, 4 attempts).

Against the expectation (written before the run):

- Old SWE 8 to 13 of 17: **8, inside**; polyglot 25 to 45 %: **45.8 %, at the top edge** (new 44.4 %, inside); json 31 to
  32: 32 both, inside.
- Difference old minus new between 0 and +8 points: **SWE -23.5 points (new better by four tasks) is outside the
  expectation** (rule: new better by more than 5 points is a finding), but it is a single run of 17 tasks, p 0.29, interval
  from -54 to +7; one old task was lost to a loop and three hit the step limit. Polyglot -1.4 and all suites +2.5 are
  inside. The probability estimate "significant difference in any suite about 20 %" held: none is significant.
- The pinned shard solved 12/17 on 2026-11-16 (`2026-11-16-pc-dm-vs-coder.md`) and 12/17 now: it reproduces.

Decision by the rules (guard, not a ranking): rule 1 (new clearly worse beyond the night's resolution) **does not apply**:
the interval of the pooled difference is [-5.3, +10.2] in favour of neither shard, and the one weak signal (SWE) favours the new
shard. Rule 2: the pin stays on the new shard (10-19) and there is **no night 2**. Per language (rule 3), nothing is
resolved: Rust 0/10 for both (a Rust drop of the new shard cannot show against a floor of zero; see the llama.cpp
cross-check below), Go 3/17 against 1/17 (net difference 2 tasks, below the 3-task rule). What the night cannot say: a
difference below 15 points per suite, and anything about C#, which is a separate suite (results appended below when done).

## Result: C# (2026-11-18, 20:27 to 22:29; arm old r1, arm new r1, `agent-shard-cs/`)

Test conditions: `agent-shard-cs/*.hw_report.txt`, quiet checks `*.quiet_check*.txt` (warning only: 2 GiB of VRAM held by
others, before both arms). Quality run, no timing claim. Both jobs ran under the queue label of the night-1 jobs
(`agent-old-r1`, ticket 1183; `agent-new-r1`, ticket 1185).

| Suite | Pairs | old pass | new pass | both / only old / only new / none | new - old points [95 %] (p) | Verdict |
|---|---|---|---|---|---|---|
| csharp (4 attempts) | 100 | 58/100 = 58.0 % [48, 67] | 82/100 = 82.0 % [73, 88] | 56 / 2 / 26 / 16 | **+24.0 [+14.8, +33.2] (p < 0.001)** | **new better** |
| csharp, first try | 100 | 19/100 | 51/100 | | | |

Resolution: 100 pairs at 28 % discordance resolve 15 points or more. Time: old 3,095 s (31 s per exercise), new 4,172 s (42 s per
exercise; the reason for the longer time of the new arm was not checked), half to two thirds of the planned 65 s. No error in either
server log, no infrastructure failure.

Against the expectation (written before): pass within 4 attempts 40 % (25 to 60): old 58 % inside at the top edge, **new 82 %
outside, above**; first try 5 to 15 %: old 19 %, new 51 %, both above. Difference between the arms between -10 and +10 points:
**+24 points is outside the expectation and a finding**, the only significant difference of the whole evaluation. It agrees with
ISTA's statement that the expert selection of the new shard preserved C#. The verdict rule (net difference of 3 or more tasks)
gives "new better" with p < 0.001, not weak. Only 2 exercises are solved by the old shard alone, 26 by the new one alone.

For the installer: C# is the one language where the data say the pinned shard (10-19) is clearly better; for JavaScript, Python,
Go and Rust nothing is resolved. Not checked: whether the old shard's low C# score is partly an engine effect (the llama.cpp
cross-check covers Rust and Go only); a C# run of the old shard on llama.cpp would separate that.

## Result: llama.cpp cross-check of Rust and Go (job `agent-rust-llamacpp`, 2026-11-18 22:29 to 2026-11-19 03:25 CET)

Old shard 09-29, llama.cpp HIP build, thinking off, 4 attempts, same exercises and prompts as old-r1. Out directory
`/var/tmp/claude/w159/rust-llamacpp/` (hw_report, quiet checks before and after, server log). llama.cpp decoded at about 22 tokens/s
(decode, one slot, no drafter) against about 64 for our server, so the job took 5 h instead of the 50 to 70 min planned.

| | our engine (old-r1, sampled) | llama.cpp sampled (seed 1165) | llama.cpp greedy |
|---|---|---|---|
| Rust, 10 exercises | **0/10** | **0/10** | **1/10** (`two-bucket`, 4th attempt) |
| Go, 17 exercises | **3/17** (`alphametics`, `paasio`, `word-search`) | **4/17** (`dnd-character`, `dominoes`, `markdown`, `say`) | partial, stopped: 0/1 |

Rules: Rust 3 or more of 10 in either pass would mean an engine bug: **0 and 1, so the Rust result is a model property** (the
installer text: Rust is weak with both files). Go 7 or more of 17 in either pass would mean an engine bug: **4**, so the Go
result is a model property too. The Go greedy pass was **stopped by hand after one exercise** (coordinator, 2026-11-19 03:25)
and is partial data: `robot-simulator` looped to the token limit in all 4 attempts (14,855 to 16,384 tokens each, about 15 min per
attempt at 22 tokens/s, about 1 h per exercise), so the pass would have taken up to 16 h more while holding the GPU, it cannot
reach 7 of 17 on a looping run, and the sampled pass is the like-for-like one. The job then ended normally (server stop,
quiet check after, DONE marker). The Go pass sets of the two engines differ (3 against 4, no exercise in both): at 17
exercises and a temperature of 1.0 this is sampling noise, not a signal about either engine.

Equal request side, checked: the **prompt token counts of the first attempt are identical for all 27 exercises** (Rust and Go,
both engines, for example 517 and 517, 1,285 and 1,285), so the two chat template renderings produce the same number of tokens.

Failure types (from the attempt records and test tails; categories: compile error of the model's code, no parseable FILE block,
answer cut at the token limit, failing test):

- Rust, 40 attempts per run: ours compile 33, no FILE block 5, cut 1, test 1; llama.cpp sampled compile 34, no FILE 5, cut 1;
  llama.cpp greedy compile 30, no FILE 5, test 2, cut 2, other 1. **They look alike:** invented or wrongly used APIs and type
  errors that do not compile.
- Go, 68 attempts per run: ours compile 33, no FILE block 24, test 6, other 3, cut 1; llama.cpp sampled compile 26, no FILE 24,
  test 3, other 4, cut 3. The no-FILE count is the same (24 of 68) on both engines.
- **The three Rust exercises whose first attempt had no parseable FILE block on our engine** (`fizzy`, `two-bucket`,
  `doubly-linked-list`; the last one was a cut-off answer first): on llama.cpp `doubly-linked-list` has no FILE block in the first
  attempt in both passes (as ours), `two-bucket` has none in the sampled pass and a compiling file in the greedy pass,
  `fizzy` is cut at the token limit in both passes (up to 18,907 tokens over four attempts). So llama.cpp produces no FILE block
  on the same exercises, which points at the model's output and not at our template or stop handling.

Conclusion for the per-language table: the 0/10 Rust and 3/17 Go of the old shard, and by the same code path the new shard's
1/17 Go and 0/10 Rust, are model properties; the Rust and Go rows stand. Correction to an interim message: the Rust greedy
pass was reported there as 0/10 while it was still running; the finished pass is 1/10.

## Pilot result (2026-11-18, old shard, 1 block, 04:55 to 05:55)

| Suite | Result | Time |
|---|---|---|
| polyglot, 8 exercises, 4 attempts | 2 passed (1 at first try, 1 at second) | 957 s |
| bench-effort, 5 tasks, high | 4 passed | 2,245 s |
| json | 32 of 32 | 29 s |

Quiet check before: warning, 2.06 GiB of VRAM held by others (desktop). The pilot shows the harness runs end to end
for the three suites; SWE was not part of the pilot (images are pulled and 17 gold patches graded in #139).

## Run command (for the night)

```
python3 bench/agent/run.py compare --out /var/tmp/claude/w159/agent-shard \
  --a 'old|<repo>/work/models/coder-iq1_m-0929/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf|<build w9>|TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder-0929.bin' \
  --b 'new|<repo>/work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf|<build w9>|TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder-1019.bin' \
  --suites swe,polyglot,json --limit polyglot=72 --repeats 1 --drafter <repo>/work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
```
