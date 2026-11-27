# 2026-11-15: Ornith-1.5-35B-A3B, Nemotron 3.5 Bolt and our Coder files, measured with our own method

Status: **stopped by the maintainer's decision on 2026-11-15 (09:20)**, with the results at the end ("Outcome and
results"). The sections from "Question" to "Safety and tooling rules kept" are the plan as written before and during the
runs (its sizes and orders were shrunk several times; the final scope is in the outcome table). Reason for stopping:
knowledge gain too small at 4 tasks per model on llama.cpp; the models get Tona support and the same suites run there.
The harness is the reusable part (`bench/polyglot/`, `bench/swe/`, `bench/bfcl/`, `bench/chat/`).

## Question

Is Ornith-1.5-35B-A3B or Nemotron 3.5 Bolt better than the Coder model Tona runs today, for coding-agent workers on the
reference PC (RX 9060 XT 16 GB, 64 GB RAM)? Vendor numbers do not answer it: the vendor figures for the Coder (DeepSWE
58.7, Toolathlon 73.5, SWE-bench Multilingual 81.0) belong to the unpruned BF16 original, not to our IQ1_M file. This
measurement reports the real numbers of **our files**.

## Models (what is compared)

| Name | File | Quantization | Size | License | Notes |
|---|---|---|---|---|---|
| Coder | `work/models/coder-iq1_m/` (ISTA Qwen3.8-Flash-Next Coder GSQ-RCO IQ1_M, shard pinned 2026-09-29 or later) | IQ1_M, about 1.5 to 2 bpw on the experts (exact bits per weight: `tools/gguf_info.py` at run time) | 58 GB | Apache-2.0 base, ISTA quant | what Tona runs |
| DM | `work/models/coder-dm/` (Coder cut to 164 of 256 experts per layer, byte copy; shard 1 `requant/dm/coder-DM-164e.gguf`, shard 2 the Coder's) | as the Coder | smaller | as the Coder | used by the Mac backend; derived from the superseded ISTA shard (research/45) |
| Ornith | `work/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf` | Q4_K_M, about 4.9 bpw (21.7 GB for 36 B parameters) | 21.7 GB | MIT | official GGUF of ornith-ai |
| Bolt | `work/models/nemotron-bolt-74b-a9b-24gb/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` | experts IQ2_XXS (2.06 bpw, QAT by NVIDIA), Mamba layers NVFP4, embeddings Q8_0 | 22.0 GB | OpenMDW-1.1 | NVIDIA's own GGUF |

Deviation from the ticket: research/49 names the aidevco W4A16 quant (about 52 GB). It is compressed-tensors safetensors
and the pinned llama.cpp (b11822) cannot load it, so the official 24GB GGUF is used. Whether b11822 loads it is
unchecked; if not, the newest llama.cpp release is built under `work/build/` as `scripts/build-llamacpp.sh` does, through
the GPU queue. A second Bolt row with the unsloth UD-Q3_K_XL (39 GB) is optional.

Engines (final scope, maintainer, one night): **Coder and DM on Tona** (`work/build/engine` from main, defaults, MTP on;
DM only if it loads: the Mac backend is the one it was made for, whether the HIP backend loads a 164-expert file is an
open check, test: server start, one-line request; a failed load's error text is the result), **Ornith and Bolt on
llama.cpp-hip** (pinned build, no speculative decoding). There is **no Coder run on llama.cpp**: its speed on the same
pinned build and flags was measured in the scale run of 2026-11-14, block i
(`research/measurements/2026-11-14-scale-run.md`): **decode** 21.6 / 22.5 / 17.6 tok/s (short-code, long-decode,
needle-30k) and needle-30k **prefill** 411 tok/s, against Tona's numbers of the same block; this is the engine factor
and is cited, not re-measured. Quality does not depend on the engine (same weights, same sampling), so the quality suites
compare the models, each on the engine that runs it; engine differences in the tool-call parser or the chat template are
possible and are noted where seen (ungeprueft until a case shows one).

llama.cpp flags for every model: `-c 65536 -np 1 -ngl 999 --cpu-moe --moe-cache-mib 8192 -fa on -ctk q8_0 -ctv q8_0
-t 6 --jinja` (as `scripts/baseline.sh`). A model that needs a different setting (Bolt's Mamba state, a 128K context for
the agent suite) says so in its run block.

Sampling (written down before the run; from the model cards, thinking on, the model's default chat template):

| Model | temperature | top_p | top_k | source |
|---|---|---|---|---|
| Coder | 1.0 | 0.95 | 20 | Qwen3.8-Flash-Next card, thinking mode |
| DM | as the Coder | | | same model family |
| Ornith | 1.0 | 0.95 | 20 | card: 1.0 reproduces the vendor benchmarks, 0.6/0.95/20 for general use; top_k 20 from the card |
| Bolt | 1.0 | 0.95 | server default | card "Recommended Sampling" |

## Suites (shrunk by the maintainer, 2026-11-15: "orientation, not precision")

The goal is a rough ranking, not a significance claim. No repetitions, no confidence-interval sizing. Results are
reported as counts (passed of n); differences of a few tasks (about 3 of 20, 5 of 50) are noise and are said to be noise.
Aim: one or two nights for all models, in short GPU-queue chunks (one chunk = start server, run up to N tasks, stop;
resumable) so the other worker's jobs interleave. All suites use the OpenAI-compatible API on port 8091, one request
stream at a time, the same tasks, prompts and tool schemas for every model. Raw results: one JSON line per task under
`research/measurements/` (date in the name).

Priority: S2 and S3 first, then S1 and S5, S4 last and only if trivial.

### S2. Long agentic tasks (first priority): about 20 SWE-bench Verified tasks per model

Harness: `mini-swe-agent` 2.4.6 (minimal bash-only loop) with the official `swebench` 5.0.2 images, run in rootless
Podman (installed and working; `docker` is its emulation). Terminal-Bench 0.2.18 is the fallback if the images do not run
rootless. One pass per task, no repetition. Subset: 20 tasks drawn with seed 139 from the 500 Verified tasks, written to
`bench/swe/subset.txt` before the first run, the same for every model; tasks whose reference patch does not pass in the
local sandbox are dropped for all models. Step cap as in mini-swe-agent's defaults plus a wall-clock cap and a 128K
context, the same for every model; a task that hits a cap counts as failed. Open facts to settle on a 2-task pilot
before the first model run: rootless image pull and disk (371 GB free at the start; several GB per task), the
OpenAI-endpoint settings of mini-swe-agent. Expected cost: about 15 to 40 minutes per task at 50 to 90 tok/s **decode**
(ungeprueft), so 20 tasks are about 5 to 13 GPU hours per model; run in chunks of 5 tasks.

S2 gold-patch check (done 2026-11-15, CPU only, `bench/swe/evaluate.py` with the official swebench grading code
on the instance images, rootless Podman): the three matplotlib tasks of the draw (matplotlib-25960, -20826, -23314)
**cannot run rootless here**: their image layers need UIDs outside the user's subuid range ("potentially insufficient
UIDs or GIDs available in user namespace", `lchown /testbed/build/qhull-2020.2`), and no root is available. They are
dropped for every model. The other **17 tasks: gold patch resolved 17 of 17**, so the subset is 17 tasks
(`bench/swe/subset.txt`; django 11, xarray 2, sphinx 2, pytest 1, sympy 1). Step limit 100 (mini-swe-agent default is
250): a decision for runtime, the same for every model; a task that hits it counts as failed.

### S3. Tool use (first priority): about 50 BFCL multi-turn cases per model

`bfcl-eval` 2026.3.23, multi-turn categories (base, miss-func, miss-param, long-context), about 50 cases drawn with
seed 139, the same for every model, through the server's own tool-call parser (llama.cpp `--jinja`, Tona's parser), so
the result includes how each engine carries the tool call. Open: whether the package accepts a generic OpenAI endpoint
(pilot with 3 cases; otherwise a small own harness over the same case JSON). Tau-bench is out (its user simulator would be
the model under test).

### S1. Aider polyglot: about 50 exercises per model

`bench/polyglot/` (Python, Go, Rust, JavaScript; whole-file replies; pass@1 and pass@2; rootless Podman without network;
reference solutions checked first). The first 50 of the fixed seed-139 shuffle of the valid exercises, the same for every
model (`--limit 50`). Counts only; `report.py` also prints intervals, which are not used for claims.

### S5. Speed: one run per model and engine

`bench/run` cases short-code, long-decode, needle-30k, **one run** per case (plus the warm-up of the harness), **decode**
and **prefill** named, client and server values both. The S2 and S3 sessions give extra decode and prefill samples from
the server timings. Expectations (tolerance, source): Coder on Tona **decode** about 87 to 91 tok/s on the code set
(BENCHMARKS.md 2026-11-14, +-5 %); Ornith on llama.cpp **decode** 80 to 150 tok/s (from 1.8 GB
per pass at 315 GB/s plus offload, research/08, ungeprueft); Bolt on llama.cpp **decode** 30 to 60 tok/s (9 B active, IQ2,
22 GB partly in RAM, ungeprueft). With one run the spread is unknown, said in the result. Active tooling: `TONA_DIAG`
default, no profilers, no `TONA_DETERMINISTIC` (production mode).

### S4. German and chat: only if trivial

The 20 fixed prompts and the fixed rubric (`bench/chat/prompts.txt`, `bench/chat/rubric.md`, written before the run),
answers shuffled with the model names removed, scored blind by the maintainer; no judge model. Done only if S1 to S3 and
S5 are finished and it costs little; otherwise listed as not done.

## Order

Priority set by the maintainer (2026-11-15): S2 long agentic tasks and S3 tool use first, then S1 polyglot and S5
speed, S4 (German and chat) last and only when everything else is done; S4 stays at its 20 prompts.

1. Preparation that needs no GPU: reference check of the polyglot exercises (queued as ticket 492; short), downloads
   finish outside the queue (`sha256 ok` in `/var/tmp/claude/139-download.log` before a model is used).
2. Load tests: Ornith and Bolt on llama.cpp-hip; DM on Tona. Result per model: loads or the error text.
3. S2 pilot (2 tasks), then the 20-task subset per model in chunks of 5.
4. S3 pilot (3 cases), then the 50 cases.
5. S1 (50 exercises) and S5 (one speed run) for every model that loads; speed is also read from the S2 and S3 sessions.
6. S4 last, only if S1 to S3 and S5 are done.
7. Evaluation: one table per suite with counts (passed of n) per model and its engine, bits per weight next to every
   result, and the speed table with the engine factor of the scale run cited above.

## Safety and tooling rules kept

Every job, build and test run through `scripts/gpu_queue.py`; servers only on port 8091 under `systemd-run --user
--scope -p MemoryMax=54G -p MemorySwapMax=0`; port 8662, 8097 and the unit `tona-server` untouched; no program names in
this file (sums only); foreign code (the exercises, the SWE-bench repositories) runs only in the rootless Podman
sandbox without network; tools install under `work/` or `/var/tmp/claude/139`, never system-wide.

## Prior art

- SWE-bench Verified saturation and contamination: OpenAI stopped reporting it (2026-02), recommends SWE-bench Pro;
  third-party leaderboards (vals.ai, mini-swe-agent) show frontier models at 96 to 97 %, so it separates our small
  models but not the frontier (search snippets, not first-hand pages; status: read as snippet).
- Aider polyglot: 225 Exercism exercises in six languages, whole-file or diff edit formats, two attempts (aider.chat
  leaderboard documentation, read as snippet); our harness reimplements the method for four languages.
- mini-swe-agent 2.4.6, swebench 5.0.2, terminal-bench 0.2.18, bfcl-eval 2026.3.23: versions read from PyPI on
  2026-11-15; their documents are not yet read in full.
- No study of the correlation between polyglot and SWE-bench for local quantized models was found.

Change type: none
Search: tools/papers/scan.py not run (no engine code changes in this file); web search 2026-11-15, window 2026-02..2026-11 (9 months), terms: aider polyglot swe-bench correlation, swe-bench verified saturation, mini-swe-agent

## Outcome and results

### What was run, and what was not

| Part | Ornith (llama.cpp) | Bolt (llama.cpp) | Coder (Tona) | DM (Tona) |
|---|---|---|---|---|
| Load test | loads (b11822) | loads (b11822) | loads | **loads on Tona HIP** |
| S5 speed, 1 run per case | done | done | done | done |
| S3 BFCL multi-turn, 52 cases | **27/52** | not run | not run | not run |
| S2 SWE-bench Verified subset | 3 tasks graded (below) | 0 finished in 3 h, stopped | not run | not run |
| S1 polyglot (harness and 144 valid exercises ready) | not run | not run | not run | not run |
| S4 chat (prompts and rubric written) | not run | not run | not run | not run |

There is therefore **no quality comparison between the models**: only Ornith has quality numbers, and they are 3 SWE
tasks and 52 BFCL cases. The speed numbers are complete for all four. The Coder was not run on llama.cpp (the scale run
of 2026-11-14, block i, is cited as the engine factor).

### Test conditions

`tools/hw_report.py` of the first Ornith speed run (`tc-ornith-speed-013925`, commit 508e201 with a dirty tree; the
other runs were on the same machine): Ryzen 5 9600X (6 cores / 12 threads), RX 9060 XT gfx1200 15.9 GiB, 60.5 GiB RAM
(57.1 GiB available before the run, 0.96 GiB swap in use), Fedora 44, kernel 7.2.15, ROCm 10.1.0, no sandbox, load
average 2.3 (1 min). `tools/quiet_check.py` before every run said **warning**: 2.3 to 2.6 GiB of VRAM held by others
(the desktop; the reference notes about 1.4 GB), CPU use of others below 1 %, in two runs a little swapping. The GPU was
held by the queue slot for each run, so no other GPU job ran. The after-check of the first Ornith and the Bolt speed run
is invalid (it ran while our own server still held 13 to 15 GiB of VRAM; fixed for later runs); later after-checks say
warning again with the same findings (the compare tool labels warning to warning "DISTURBED"; the cause of that label
was not checked, **ungeprueft**). The benchmark reports are in
`research/measurements/2026-11-15-ibench-tona-base-m139-*.{md,json}` (the "tona" in the file name is the harness's
default label; the engine is named below).

Engines: llama.cpp-hip b11822 (d9ee007, pinned build) with `-c 65536 -np 1 -ngl 999 --cpu-moe --moe-cache-mib 8192
-fa on -ctk q8_0 -ctv q8_0 -t 6 --jinja`, no speculative decoding, for Ornith and Bolt; Tona `work/build/engine`, built at
**65af5f9** (f42258b plus a Metal-only merge; the HIP code equals f42258b, the merge of #136), defaults, MTP on,
`--ctx 65536`, for Coder and DM. Port 8091, server in a systemd scope with MemoryMax=54G and no swap. Production mode,
no `TONA_DETERMINISTIC`.

### Models and bits per weight

| Model | File | Bits per weight | License |
|---|---|---|---|
| Ornith-1.5-35B-A3B | Q4_K_M, 21.71 GB, sha256 42739874... verified | 4.83 average (21.71 GB over 35.95 B parameters) | MIT |
| Nemotron 3.5 Bolt 74B-A9B 24GB | official IQ2 GGUF, 22.02 GB, sha256 91d63363... verified | 2.38 average (22.02 GB over 74 B); experts 2.0625 (IQ2_XXS, QAT), Mamba layers NVFP4, embeddings Q8_0 | OpenMDW-1.1 |
| Coder (ISTA GSQ-RCO IQ1_M) | 29.6 + 28.8 GB = 58.4 GB | IQ1_M experts (nominal 1.75); average over all weights not computed (**ungeprueft**: total parameter count not read) | Apache-2.0 base, ISTA quant |
| DM (Coder cut to 164 of 256 experts per layer) | shard 1 20.5 GB + the Coder's shard 2 28.8 GB = 49.3 GB | as the Coder | as the Coder |

The Coder's GGUF carries the sampling it should use (`general.sampling`: temperature 1.0, top_p 0.95, top_k 20), the same
as the card values used in the plan. The Bolt quant of research/49 (aidevco W4A16, about 52 GB) is safetensors that the
pinned llama.cpp cannot load; the official GGUF was used instead.

### S5 speed (one run per case, so the spread is unknown; **decode** and **prefill** named)

Prompt token counts differ a little between tokenizers (30.2 to 30.9 K tokens in needle-30k).

| Model, engine | **decode** short-code / long-decode / needle-30k (tok/s, server) | needle-30k **prefill** | needle-30k first token | VRAM peak (whole GPU, MiB) | RAM available min (MiB) |
|---|---|---|---|---|---|
| Ornith Q4_K_M, llama.cpp | 44.7 / 47.7 / 34.0 | 713 tok/s (server) | 43.5 s | 13,597 | 56,421 |
| Bolt IQ2, llama.cpp | 33.1 / 33.4 / 30.2 | 478 tok/s (server) | 63.4 s | 16,155 | 54,627 |
| Coder IQ1_M, Tona (MTP) | 98.8 (59 tokens only) / 79.2 / 68.3 | about 1,912 tok/s (client; Tona reports none) | 16.2 s | 15,729 | 30,345 |
| DM, Tona (MTP) | 115.5 (43 tokens only) / 71.7 / 69.2 | about 1,966 tok/s (client) | 15.7 s | 15,752 | 39,443 |
| Coder IQ1_M, llama.cpp, no draft (**cited**: scale run 2026-11-14 block i, 3 runs, +-3 %) | 21.6 / 22.5 / 17.6 | 411 tok/s | 75.4 s | | |

- The expectations of the plan: Coder on Tona 87 to 91 tok/s on the code set: long-decode 79.2 and short-code 98.8 are
  different workloads, so no deviation is claimed (the plan's reference was the bench-set code average); Ornith 80 to 150
  **decode**: measured 34 to 48, **below the expectation** (the estimate of research/08 assumed the active experts' bytes
  per pass at VRAM bandwidth; here `--cpu-moe` with an 8 GiB expert cache runs part of the experts from RAM; cause not
  isolated, **ungeprueft**); Bolt 30 to 60: measured 30 to 33, inside.
- On one engine (llama.cpp, same flags) Ornith decodes about 2.1 times as fast as the Coder (47.7 against 22.5 tok/s in
  long-decode) and Bolt about 1.5 times; Ornith's needle-30k prefill is 1.7 times, Bolt's 1.2 times the Coder's. The
  Coder on Tona decodes 3.5 times and prefills 4.6 times as fast as on llama.cpp (the engine factor; measured on
  different builds and times, so an order of magnitude) and decodes faster than Ornith on llama.cpp.
- The DM decodes like the Coder on Tona (71.7 against 79.2 and 69.2 against 68.3 tok/s; differences within one-run
  noise, not claimed) and needs 9 GB less file; its RAM left free is 39.4 GB against 30.3 GB for the Coder.
- Bolt reasons first: all three speed cases ended with `finish length` inside the reasoning (300, 1024, 768 tokens), so
  the suite's "expected" check says WRONG for Bolt. That is truncation by the suite's token cap, not wrong answers.

### S3 tool use: BFCL multi-turn (Ornith only)

`bfcl-eval` 2026.3.23 through its chat-completions function-calling handler against the llama.cpp server (so llama.cpp's
tool-call parser and Ornith's template are in the path), temperature 0.001 (BFCL's default; BFCL sends no top_p or
top_k), 13 cases per category drawn with seed 139 (`bench/bfcl/ids.json`), one pass.

| Category | Passed |
|---|---|
| multi_turn_base | 9/13 |
| multi_turn_miss_func | 4/13 |
| multi_turn_miss_param | 7/13 |
| multi_turn_long_context | 7/13 |
| **Total** | **27/52 = 51.9 %** |

With n = 52 the 95 % Wilson interval of this result is about 38 to 65 %. Nothing to compare against: no other model was
run.

### S2 long agentic tasks (SWE-bench Verified subset, mini-swe-agent 2.4.6)

Subset: 17 tasks after the gold-patch check (gold patch resolved 17 of 17; the three matplotlib tasks cannot run rootless
here, their image layers need UIDs outside the user's subuid range, and are dropped; `bench/swe/subset.txt`). Step limit
100, 262,144-token context (a first pilot at 131,072 ran out of context after 97 steps), temperature 1.0, top_p 0.95,
top_k 20.

**The podman-banner bug.** `docker` here is Podman's emulation, which prints "Emulate Docker CLI using podman..." before
the command output. mini-swe-agent recognises a submission by the marker `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as the
**first** output line, so the submission was not recognised, the agent ran on to the step limit and `preds.json` came
out empty. For the Ornith tasks the patch each agent printed with its first submission was recovered from the
trajectories (`bench/swe/salvage.py`) and graded with the official swebench log parsers (`bench/swe/evaluate.py`, Podman).
**These Ornith tasks ran with the bug: their step counts and wall times are not comparable with anything else; only the
pass or fail of the first submission counts.** The harness is fixed (`environment.executable=podman`); no run with the
fix finished.

| Task | Ornith (llama.cpp) |
|---|---|
| django__django-10554 | not resolved (no submission within 100 steps) |
| django__django-15268 | **resolved** (patch of 941 bytes, applied, tests passed) |
| django__django-16082 | **resolved** (patch of 403 bytes, applied, tests passed) |

2 of 3 resolved. The fourth task drawn for the first chunk (sphinx-doc__sphinx-7454) was cut when the run was stopped;
the earlier pilot ran it and django-10554 at 131,072 context with the same bug (0 of 2: context exhaustion and the step
limit). **Bolt: no task finished in the 3 hours it was allowed** (started 06:40, stopped 09:40; decode about 26 tok/s and
long reasoning on every step; the first task, django-10554, was still running). "Not run" for Bolt on S2, with that
reason; it is a finding for agent use in itself, not a quality score. Coder and DM: not run on S2.

Ornith's tasks in the first chunk took more than 1 hour each at 18 to 36 tok/s decode (the long context lowers the rate;
the bug added steps), so 17 tasks per model would have been about 7 hours per model, which is why the plan shrank to 4
tasks per model and the run was then stopped.

### Findings

1. **Speed ranking on this PC, one run each:** Coder on Tona (68 to 79 **decode**, 1.9 K **prefill**) is the fastest;
   Ornith on llama.cpp second (34 to 48, 713 prefill), Bolt on llama.cpp third (30 to 33, 478 prefill); the Coder on
   llama.cpp is last (18 to 22, 411). The engine matters more than the model for the Coder.
2. **DM loads and runs on Tona's HIP backend** at the speed of the Coder, with less file and more free RAM.
3. **No quality verdict.** The only quality numbers are Ornith's (BFCL 27/52; 2 of 3 SWE tasks resolved). The vendor
   figures for the Coder (DeepSWE 58.7, Toolathlon 73.5, Multilingual 81.0) are for the unpruned BF16 original; our
   IQ1_M file was not measured on these suites, and the question "is Ornith or Bolt better than the Coder for agent
   coding" stays open.
4. **Bolt as measured is a poor agent-worker candidate on this PC:** 30 to 33 tok/s decode with reasoning that fills the
   token budget before the answer; no SWE task finished in 3 hours. One task, one run: an orientation, not a verdict on
   its quality.
5. **Harness findings** (reusable): the podman banner breaks mini-swe-agent's submit detection; rootless Podman cannot
   unpack some SWE-bench images (matplotlib); a 128K context is too small for SWE tasks of 100 steps; BFCL has no generic
   OpenAI endpoint but its chat-completions function-calling handler honours `OPENAI_BASE_URL`; port 8099 is used by
   another worker.

### What stayed unchecked

- Quality of the Coder, DM and Bolt on every suite; S1 polyglot and S4 chat for all models (harness ready, 144 valid
  polyglot exercises).
- Tool-call handling differences between llama.cpp's parser and Tona's: never exercised (no Tona agent run).
- Spread of every speed number (one run each); the Ornith decode below the plan's expectation (cause not isolated).
- Whether the matplotlib SWE images would run with a larger subuid range (no root here).
- Cause of the "DISTURBED" label on warning-to-warning quiet checks.
