# 2026-11-16: DM against the Coder on the PC: speed by RAM class and agent quality

Status: plan written first (2026-11-15, before the first run); results are appended below the plan.

## Questions and the decisions they lead to

1. **Speed by RAM class.** How fast are DM (the Coder cut to 164 of 256 experts per layer) and the Coder on this PC
   when the engine may use only the RAM a 16, 32 or 64 GB machine leaves? DM clearly faster (a factor, not percent) at
   16 and 32 GB: DM becomes the recommended model for those classes (installer, docs/hardware). Equal: question 2
   decides. DM slower: it stays Mac-only.
2. **Agent quality, DM against the Coder, both on Tona.** Does dropping 92 experts cost agent ability? DM within 2
   tasks of the Coder on SWE (17 tasks) and within 5 cases on BFCL (52 cases): "good enough" for the low-RAM classes.
   Clearly worse: DM only where the Coder cannot run, gap documented.

## Plan

### Scope

Engine: current `main` (4caacd0, HIP code unchanged from 65af5f9 of the model evaluation) built into
`work/build/dm145`; production mode (no `TONA_DETERMINISTIC`), MTP draft as in the evaluation, default settings, no
expert profile (as in the evaluation runs). Same build, sampling (temperature 1.0, top_p 0.95, top_k 20 from the
Coder's GGUF, SWE; BFCL temperature 0.001 as its default) and harness as `2026-11-15-model-eval-ornith-bolt.md`.

**RAM classes.** The installer's budget for the reference backend is `ram_gb = max(8, floor(machine RAM - 6))`
(`installer/modules/hip_gfx1200.cpp`, `recommend`: "everything except 6 GB for system and desktop"). The server runs in
a systemd scope with `MemoryMax` = that value in decimal bytes and `MemorySwapMax=0`, plus `TONA_RAM_GB` = the same
value, exactly as the installer's unit (`installer/core/unit.cpp`):

| Machine class | Installer budget (`TONA_RAM_GB`, `MemoryMax`) | Expected behaviour (budget.h: pin all experts or none; the all-experts step is at about 31.6 GB for the Coder's 25.15 GB of experts) |
|---|---|---|
| 16 GB | 10 GB (10,000,000,000 bytes) | no pins for either model, experts from the page cache and the SSD |
| 32 GB | 26 GB | Coder: no pins (page cache); DM: 164/256 x 25.15 = 16.1 GB of experts could fit if budget.h counts the file's real expert bytes (open: whether the rule uses the file size or a fixed 25.15 GB; read from the server log) |
| 64 GB | 58 GB | both pinned |

**Question 1 (speed).** `bench/run` cases short-code and long-decode, 1 run per model and class (warm-up of the harness
included), plus needle-30k at 32 GB only: 2 models x (2 + 2 + 3 cases) in 6 queue jobs, each one model x one class
x one case group, ctx 65536 (the installer's context for a 10 GB budget is 65536; kept for all classes so classes
compare). **Decode** and, for needle-30k, **prefill** are named. One run each gives no spread; the effect sought is a
factor, and a result within about 20 % of equal is read as "equal", not as a difference.

**Question 2 (quality).** The 17 SWE tasks of `bench/swe/subset.txt` (mini-swe-agent 2.4.6, rootless Podman, step limit
100, ctx 262144, fixed harness `environment.executable=podman`) and the 52 BFCL multi-turn cases (`bench/bfcl/ids.json`),
one pass per model, paired (same tasks and cases). Run at the 58 GB budget (the quality does not depend on the RAM class;
the 64 GB class is the fastest, which keeps GPU time short). Every SWE task is one queue job (model x one task), BFCL is
one job per model x category (13 cases). The SWE images (17 tasks, about 50 GB) are pulled again at idle priority
outside the queue and removed after the run. The 3 matplotlib tasks stay dropped (rootless UID mapping, see the
evaluation file). Detects only large gaps, which is the question; a difference of 1 to 2 tasks of 17 is noise.

### Expectation (written before the runs; sources and reasons)

- **Speed at 64 GB (58 GB budget):** both pinned: Coder **decode** 70 to 80 tok/s (long-decode 79.2, evaluation
  2026-11-15, one run), DM 70 to 80 (71.7, same run) -> equal, tolerance +-20 %.
- **Speed at 32 GB (26 GB budget):** Coder unpinned: **decode** about 50 tok/s on code (50.3 at a 26 GB limit,
  2026-10-27-ram-limit.md; long-decode text-like about 36), tolerance +-25 %. DM: if the rule lets its 16.1 GB of experts
  pin, **decode** about 70 (like the 64 GB class), a factor of 1.4 to 2 over the Coder; if not, about equal to the Coder
  (smaller page-cache working set, so somewhat faster, 1.0 to 1.3 x). Reason: decode of an unpinned model is bound by
  the page-cache misses, and DM's experts are 36 % fewer bytes.
- **Speed at 16 GB (10 GB budget):** Coder 20 to 30 **decode** (estimate in docs/hardware/pc-16gb-vram.md, unmeasured);
  DM 25 to 45 (16.1 GB of experts against 10 GB of cache: a larger hit share than the Coder's 25.15 GB), a factor of
  1.2 to 1.6. Sources are estimates (research/22, docs/hardware), so a wider tolerance of +-40 %.
- **needle-30k at 32 GB, prefill:** Coder 600 to 800 tok/s (659 / 770 at 26 / 29 GB, ram-limit file), DM equal or
  somewhat higher.
- **Decision by speed (expected):** DM faster by a factor at 16 GB perhaps, at 32 GB only if it pins. If neither is a
  clear factor, question 2 decides.
- **Quality:** DM is a pure cut (KLD 0.053 on code, 0.199 on text; JSON 32/32 as the Coder; requant file), so its
  tool-call format and code ability should be nearly kept, its long-tail knowledge reduced. Expected SWE: Coder 5 to 9
  of 17 (no measurement of this file exists; the unpruned vendor numbers are far higher and do not apply), DM 1 to 2
  tasks fewer. Expected BFCL: Coder 24 to 32 of 52 (Ornith on another engine: 27), DM 0 to 4 fewer. Probability that
  both thresholds hold: about 50 %; the noise of one pass (17 tasks, standard deviation of the difference about 2 tasks)
  is as large as the SWE threshold, so the verdict "good enough" can only fail on a large gap.
- **Risk:** the HIP server and the 164-expert file: the evaluation loaded it and ran speed; a tool-call or long-context
  failure specific to DM is possible and is a finding.

### Active tooling

`TONA_DIAG` default, no profilers, no `TONA_DETERMINISTIC`, `TONA_RAM_GB` per class as above, no `TONA_PIN_GB` /
`TONA_VRAM_GB`. Other refactor workers share the machine (CPU and GPU through the queue; their CPU work between jobs
may disturb a run: the quiet check before and after every job is kept and any "warning" is reported).

## Test conditions

(start of the work; every job also writes `tc-*.hw.md`, `quiet-before.json` and `quiet-after.txt` under
`/var/tmp/claude/145/results/`, summarised below)
Start of the work (2026-11-15 10:06, `tools/hw_report.py`): Ryzen 5 9600X (6 cores / 12 threads), RX 9060 XT gfx1200 15.9 GiB
(1.14 GiB held by others), 60.5 GiB RAM (53.5 GiB available, 1.1 GiB swap in use), Fedora 44, kernel 7.2.15, hip
7.16.26385, no sandbox, no `TONA_*` set in the shell, load average 4.05, commit 4caacd0. `tools/quiet_check.py`: **do not
measure** (other programs 22.7 % of the CPU threads; load average high; VRAM held by others). The machine was shared with
three refactor workers for the whole time (GPU through the queue, but their CPU, build and test work ran between and
beside my jobs), so the quiet check said warning before every job and warning after every job (every job's
`tc-*.hw.md`, `quiet-before.json`, `quiet-after.txt` is in `/var/tmp/claude/145/results/`, not committed). I did not ask the
maintainer to quiet the machine because the question is a factor (several times), not percent; the speed numbers carry
that disturbance and one run each, so they are read at factor level only. At the end (15:14): load average 1.3, 15.3 GiB
of VRAM held by another worker's job (not mine; mine had finished), 25.7 GiB RAM available, swap 2.1 GiB: "do not measure"
again (VRAM). Engine: Tona main 4caacd0 (build `dm145`, deleted), production mode, MTP, ctx 65536 speed / 262144 SWE /
131072 BFCL (as `model_job.sh`), the two model files of the evaluation (`work/models/coder-iq1_m`, `work/models/coder-dm`).

## Results

### Question 1: speed by RAM class (**decode**; server tok/s; one run each, spread unknown)

Machine class -> installer budget (`TONA_RAM_GB` = `MemoryMax`): 16 GB -> 10, 32 GB -> 26, 64 GB -> 58. ctx 65536.

| Class (budget) | short-code **decode** DM / Coder | long-decode **decode** DM / Coder | needle-30k at 32 GB only |
|---|---|---|---|
| 64 GB (58) | 108.7 / 109.0 | 67.9 (stopped after 129 tokens) / 77.3 | not run |
| 32 GB (26) | 121.1 / 63.2 | 80.5 / 41.6 | **decode** 81.1 / 49.6; **prefill** (30.9 K tokens / time to first token 15.7 s and 42.8 s) about 1,970 / about 720 tok/s |
| 16 GB (10) | 20.9 / 10.1 | 52.9 / 13.7 | not run |

Factors DM over the Coder: 64 GB: 1.0 / 0.9 (equal); 32 GB: 1.9 / 1.9 **decode**, needle **decode** 1.6, needle
**prefill** 2.7; 16 GB: 2.1 / 3.9 **decode**. The short-code case emits only 43 to 62 tokens, so its rate carries the first-token
ramp (DM at 10 GB: 3.2 s to the first token, 20.9 tok/s) and is the weaker number; long-decode (1,024 tokens, except
the DM 64 GB run that stopped at 129) is the better one. Reading: both models are pinned at 64 GB and equal; at 32 GB
the DM (164 experts = 16.1 GB) fits next to the limit and is pinned (its 26 GB result equals its 58 GB result, 80.5 against 67.9
to 77 within one-run noise) while the Coder's 25.15 GB of experts are not pinned (page cache), so the DM is about twice as
fast; at 16 GB neither is pinned and the DM is faster (**decode** 53 against 14 tok/s on long-decode; a smaller expert set
in the page cache would explain it, **ungeprueft**, no cache counters read). Against the plan: 64 GB as expected (equal); 32 GB: Coder 63 / 42 against the expected about 50 (inside
+-25 %), DM pinned as the first alternative predicted (about 70 expected, 80 to 121 measured, a little above); 16 GB: Coder
10 / 14 against the expected 20 to 30 (**below**, the estimate in docs/hardware was unmeasured), DM 21 / 53 against 25 to 45
(short-code below, long-decode above). Whether `budget.h` pins the DM from the file's real expert bytes was read from
the result, not from the server log (**ungeprueft** in the log; the equal 26 GB and 58 GB numbers are the evidence).

**Decision by speed:** DM clearly faster (a factor of about 2 or more) at 16 and 32 GB, equal at 64 GB. By the ticket's
rule that would make DM the recommended model for those classes, **unless question 2 says it is too weak**.

### Question 2: agent quality, same engine, same cases (64 GB budget, 58 GB)

| Suite | DM | Coder | Gap | Threshold | Result |
|---|---|---|---|---|---|
| BFCL multi-turn, 52 cases (13 per category, `bench/bfcl/ids.json`, temperature 0.001) | **3/52** (base 1, miss_func 0, miss_param 1, long_context 1) | **17/52** (base 7, miss_func 2, miss_param 2, long_context 6) | 14 | within 5 | **fails** |
| SWE-bench Verified subset, 17 tasks (`bench/swe/subset.txt`, step limit 100, ctx 262144, T 1.0, top_p 0.95, top_k 20) | **9/17** (6 of the 8 failed tasks ended with an empty patch: step limit without submission, 2 with a patch that does not resolve) | **12/17** (2 empty patches, 3 patches that do not resolve) | 3 | within 2 | **fails by one task** (noise of one pass about 2 tasks, so by itself it is borderline) |

Paired per task: both resolved 9; Coder only 3 (sphinx-7454, django-15268, xarray-4687); DM only 0. Per-task files:
`2026-11-16-pc-dm-vs-coder/swe-{dm,coder}.eval.jsonl`, BFCL tables `bfcl-{dm,coder}-multi_turn.csv`. For reference the
2026-11-15 evaluation measured Ornith (llama.cpp) at BFCL 27/52 and 2 of 3 SWE tasks.

What the DM BFCL failures look like (read from the result and score files of the DM run, five cases examined, not all 52):
`force_terminated` (the model returned text without a tool call in turn 1 or 2, so the harness stopped the case), one
empty turn response, a run of repeated `ls`/`cd` calls until the step limit in base_43, and two requests that hit the
client timeout (base_152 and base_197, cause not examined). The tool-call parser is the same Tona one the Coder passed
through on the same cases (17/52), so the weak result is a property of this model file in this path, not of the parser
alone; **ungeprueft**: whether it is caused by the cut (92 experts removed, KLD 0.199 on text), by the near-greedy
temperature of BFCL (0.001; the SWE runs at T 1.0 show a smaller gap), or by the MTP draft model (trained for 256 experts).
The DM SWE run needed 9 minutes per 2 tasks against 17 for the Coder because it ended tasks early at the step limit
without submitting.

**Decision by quality:** BFCL fails by a wide margin (14 against 5), SWE by one task. The ticket's rule for "clearly
worse": DM only where the Coder cannot run, with the gap documented here.

### Combined verdict and recommendation

- DM is **not** made the recommended model by this measurement: the speed gain is real (about 2x at 32 GB, 2 to 4x at
  16 GB **decode**), but the tool-use gap (3/52 against 17/52) is large and the SWE gap (9 against 12 of 17) is
  as large as the noise. No installer or docs/hardware change is made.
- The trade-off the maintainer may still want to decide: at 16 GB the Coder decodes only 10 to 14 tok/s on this PC
  (below the docs/hardware estimate of 20 to 30), so "Coder at 14 tok/s" against "DM at 53 tok/s that fails tool calls
  in most BFCL cases" has no good side. The docs/hardware estimate for the 16 GB row should be corrected to the measured
  10 to 14 (**decode**, one run, disturbed machine) when someone repeats it with 3 runs. Not done here (one run).
- Cheap follow-ups that would separate the causes of the DM result (not run, each one job): BFCL at the production
  sampling (T 1.0, top_p 0.95, top_k 20) for both models; DM without MTP; DM with a 164-expert-aware draft.

## What stayed unchecked

- Speed spread: one run per model and class, on a shared machine (quiet check warned every time); the factors are
  large but the 64 GB equal result (108.7 against 109.0, 67.9 against 77.3) shows run-to-run noise of 10 % or more.
- Cause of the DM tool-use failures (cut, temperature, MTP draft) and the two request timeouts.
- BFCL at the SWE sampling; DM and Coder on polyglot and chat; needle-30k at 16 and 64 GB; prefill by class.
- Whether `budget.h` pins the DM's 16.1 GB (inferred from equal speed at 26 and 58 GB, not read from a log).
- Real 16 and 32 GB machines (cgroup emulation only, as in docs/hardware).
- The matplotlib SWE tasks (dropped, as in the evaluation).
- `model_job.sh` returned exit status 1 after every job because of its EXIT trap under `set -e` (a `kill` on a finished
  PID); the data was complete each time. Fixed in this branch (`|| true`), the fixed script was not run again.

## Prior art

- `research/measurements/2026-11-11-requant.md` (DM: cut to 164 of 256 experts, KLD 0.053 code / 0.199 text, JSON 32/32)
  and `2026-11-15-model-eval-ornith-bolt.md` (harness, DM speed at 64 GB, pitfalls): read; used as the basis; this file adds
  the first agent-suite numbers for the DM and the RAM-class speed.
- BFCL multi-turn (bfcl-eval 2026.3.23) and mini-swe-agent 2.4.6 / swebench 5.0.2 as in the evaluation file.

Change type: none
Search: tools/papers/scan.py not run (no engine code change; harness script fix only); no new web search this time, window n/a
