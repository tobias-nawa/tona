# Prefill in FP8 (IE_PREFILL_FP8): shared activation conversion and quality check (#100, part 3)

Date 2026-11-03. Branch `100-fp8-quality` (from main 9563c46). Builds: `work/build/w103-base` (main 9563c46, the
arm "old") and `work/build/w103` (this branch, the arm "new"). All GPU runs under `/var/tmp/claude/gpu.lock`, raw
data under `/var/tmp/claude/w103/`. This file is written in two parts: the plan (this section, before any run) and
the results below it.

## 0. Plan (written before the first run)

### 0.1 What is checked, and the trap in the tools

`IE_PREFILL_FP8=1` changes only the dense prefill GEMMs. Two of the four checks the ticket lists would not touch
that path if run as named, so the plan replaces the tool use and says why:

- `ie-gpu-gen --check N` runs the prompt **step by step through the decode kernels** (`ie_gpu_gen_plain.cpp`), never
  the prefill. FP8 on or off gives identical numbers there. Replacement: `--teacher TEXT --mega --prefill --check N`
  (new in this branch, `ie_gpu_gen_teacher.cpp`): the text goes through `Mega::prefill`; the CPU reference
  (`cpu/ref.cpp`, f32) computes -log p of the same positions token by token; the tool prints the signed and absolute
  difference.
- `--multi-check` runs **decode passes** (`forward_multi`), never the prefill. Replacement for the 10,000-token
  Δ-NLL: `--teacher TEXT --mega --prefill` with the new `IE_TEACHER_NLL=FILE` (one line "pos -log p" per position,
  from the prefill) and the paired bootstrap of `scripts/nll-compare.py` (via a merge of two such files into its
  dump format). The prefill keeps no logits, so KL and top-1 cannot come from this run; they come from the
  teacher tail (next point), which sees the FP8-prefilled state through the decode kernels.
- The teacher tail of #73 (`IE_TEACHER_TAIL`) prefills the head through `Mega::prefill` and runs the tail in the
  megakernel. FP8 acts on the head only, which is the effect on the state (KV, DeltaNet state, indexer ring) that a
  later decode sees. This is the one run that gives KL and top-1.

How I make sure that every run goes through the FP8 prefill (also in the ticket's words, "write down how"):

1. The dense FP8 GEMM is `k_gemm8b`. One run with `rocprofv3 --kernel-trace --stats` per arm on a short prompt
   (2,048 tokens): the FP8 arm must list `k_gemm8b`, `k_dequant_fp8` and `k_to_fp8` with non-zero calls, the off arm
   none of them (§1.4).
2. Every FP8-on / off pair is compared by its output numbers: if the NLL of the two runs were bit-equal, the run
   would have measured nothing, and the file says so. A prefill pair that is bit-equal is a failure of the setup.
3. Every run is under `IE_DIAG=1`; the tools print the prefill plan with `IE_PF_DEBUG=1` (which shows layered or
   chunk-by-chunk), saved in the raw logs.

### 0.2 Step 1: shared conversion of the activation

Change: `PrefillRun::share_x8(true/false)` brackets the groups of GEMMs that read the same f16 input (q/k/v/idx_k in
`qsa_layer`; qkv/gate/beta/alpha in `dn_layer`; shared expert gate/up in `moe_route`). Inside a group `gemmq_fp8`
converts the input once (`to_fp8_rows`) and reuses `B.x8` / `B.sx` while pointer, leading dimension, rows and width
equal the converted ones. Closed at the end of each group, before any producer rewrites the input. Same bytes, same
factors as today: the conversion is deterministic per row.

- Bit-exact test: `scripts/ab.py`, FP8 on in both arms (`--env IE_PREFILL_FP8=1`): the SHAs of the three requests
  (2000, 13000, 30000) must be equal between old and new. Plus FP8 off: `scripts/golden.sh compare` for the prefill
  goldens (`GOLDEN_MODE=prefill`) and the standard ones named by `tools/select_checks.py`.
- Speed: `scripts/ab.py --a w103-base --b w103 --env IE_PREFILL_FP8=1 --env IE_DIAG=1`, `--workload server --targets
  2000:400,13000:200,30000:200 --ctx 140000`, MTP drafter, alternating AB/BA, alpha 0.05, margin 0.5 %, rounds 2 to 6,
  one discarded warm-up start.
  - Why this size: the same targets as the FP8 A/B (`2026-11-02-prefill-int8-ab-fp8-dense2.md`), so the number is
    comparable to the +1.64 / +2.96 / +3.18 % of the switch itself and its noise entries exist in `bench/noise.toml`
    (prefill@2000 sd 0.09 %, prefill@30000 sd 0.04 %, decode sd 0.23 to 0.28 %).
  - Expectation (B against A): prefill +1.2 % +-0.8 % (source: the ticket's estimate from the analysis: conversions
    0.27 s at 13K of a prefill of about 7 s, shared among about 1.5 GEMMs per conversion). Decode: 0 +-1 % (the
    decode path is untouched; the FP8 arm shows the decode@30000 gap to off, which both arms share).
  - Expected duration 3 to 10 min (2 to 6 rounds of 1.6 min plus warm-up).
  - Drop rule (ticket): if the confidence interval of prefill contains 0 and the mean is below 0.3 %, the change is
    removed again and the number goes in this file.
- Active tooling: `IE_PREFILL_FP8=1` in both arms, `IE_DIAG=1`, `IE_EXPERT_PROFILE` of the repository profile,
  `OMP_NUM_THREADS=6` (ab.py sets). Nothing else (no profilers) during the timing rounds.

### 0.3 Step 2: quality, FP8 on against off (end state of step 1, build `w103`)

Rule (`docs/backends/README.md#checks`, fixed here before any result): for every text, 95 % interval (block bootstrap, 64-token
blocks, 4,000 draws) of the mean Δ(-log p) = FP8 minus off:
- upper end below 0.01 nats: **pass**;
- mean below 0.01, upper end at or above 0.01: **undecided**;
- mean at or above 0.01: **fail**.
KL and top-1 reported next to it. Expectation for all Δ-NLL runs: mean +0.003 nats +-0.004 (source:
`2026-10-24-prefill-fp8.md`: perplexity 2.520 against 2.510 with the old dense GEMM, ln(2.520/2.510) = +0.004; the new
GEMM and the f16 fallback for small weights should not be worse). KL expectation 0.002 to 0.02 (the order of
`IE_IDX_SHARE` pattern (d), the one that passed).

All runs: `IE_DETERMINISTIC=1`, `IE_DIAG=1`, `OMP_NUM_THREADS=6`, `systemd-run --user --scope -p MemoryMax=... -p
MemorySwapMax=0`, model Coder IQ1_M (big). The tiny model is not used for quality.

| id | What runs | Why this size | Expectation | Duration |
|---|---|---|---|---|
| a | CPU reference: `ie-gpu-gen MODEL "" --mega --prefill --teacher work/corpus/code.txt --max 256 --check 256`, FP8 off and on, plus the same on shell text | the CPU reference is seconds per token; 256 positions run in about 10 min per run and tell whether the FP8 prefill is nearer to or further from f32 than the f16 prefill; more tokens would only narrow the interval | mean abs difference to CPU: off x, on at most 2x (tolerance +-100 %, from the earlier +0.004 nats mean shift against scatter 0.275 per token) | 2 x 2 x ~10 min |
| b1 | Δ-NLL of the prefill, 10,000 tokens of code (`/var/tmp/claude/w73/corp/cpp.txt`) and shell (`sh.txt`), `--teacher TEXT --mega --prefill --max 10000`, FP8 off twice (repeatability: must be bit-equal) and on once; `IE_TEACHER_NLL` | the size of the written rule; a supplementary 100,000-token run of both texts narrows the interval (cost: 1 min of prefill per run) | mean +0.003 +-0.004; the off pair bit-equal | 6 x ~20 s plus load |
| b2 | teacher tail at 32K for code (cpp, py) and shell (sh): prefill 32,768 tokens with FP8 on, then 4,096 tokens teacher-forced in the megakernel with `IE_TEACHER_REF` of the off run (`IE_TEACHER_DUMP`, method `2026-10-29-indexcache.md`) | the size of the #73 phase C at 32K; FP8 acts on the whole prefill so the tail at 32K shows the state after a long prefill. The 128K tail is the needle check (d) instead | Δ(-log p) +0.003 +-0.004, KL 0.002 to 0.02, top-1 equal 97 to 99.5 % | 3 x 2 x ~4 min |
| c1 | task benchmark `bench/threshold/` (pydash, boltons; questions A and code holes B), server on `w103` with FP8 off and on, ctx 69632, MTP, greedy, one hold of the lock per repo and run (about 45 min each, lock released in between); analysis `analyze.py` paired against the off run | calibration `2026-10-30-threshold-main.md`: resolves about 2.5 pp on questions and 7 pp on holes; ticket demands the task benchmark once next to Δ-NLL because FP8 is a new kind of change | no significant loss: A -0.5 +-1.5 pp, B -1 +-5 pp | 2 x ~1.3 h |
| c2 | `scripts/bench-effort.py medium` (23 tasks), FP8 off and on, ctx 65536 | gross-failure check; the reference varies by 1 to 2 of 46 | both within 1 to 2 tasks of each other | 2 x ~10 min |
| d | needle `bench/needle/run.py --hard --length 131072 --texts 0,1,2,3` against one server per arm (`IE_SEQS=1 IE_PROMPT_RAM_GB=0`, ctx 140000, pool `/var/tmp/claude/w77b/pool.txt`), FP8 off and on, then `report.py compare` (McNemar) | 128K is the length the ticket names; 4 texts = 160 pairs find about 12 pp at psi 0.3 (`bench/needle/README.md`, plan table); the hard variant because the easy one is saturated at 100 % | off hit rate about 50 % (README: 128K 50 % of 160); on within +-8 pp, McNemar p above 0.05 | 2 x ~15 min |
| e | cut invariance (#81): `IE_TEACHER_CUTS` as in `2026-11-02-prefix-bitexact.md`, `IE_PREFILL_FP8=1`, tail 64: mini head 20,000 with cuts 4097+9001 and 6143+13000+13001 and 1+19999, big head 14,000 (layered, group 12,288) with cuts 4001 and 12,500; every dump `cmp`-equal to the unsplit FP8 run | the sizes that put cuts on and across chunk and group edges for the mini and the big plan; the shared conversion adds state across GEMMs, so every cut that moves a chunk edge is a test of its reset | all dumps byte-equal | mini seconds, big 2 min per run |

### 0.4 Rules for the runs

Pause 60 s between GPU blocks; no build while a measurement runs; the lock is released between the checks a to e and
never held longer than about 45 minutes in one piece. 128K context on the big model: the coordinator is told before
the needle run starts. No blocking HIP call is added by this change; the server runs under `systemd-run ... -p
MemoryMax -p MemorySwapMax=0`, port other than 8080.

## Results


Deviations from the plan: (1) the plan named the effort benchmark once per arm; the first pair differed by 2 tasks, so
a second pair was run to see the run-to-run spread of the server (c2). (2) The 100,000-token prefill runs of b1 and the
`py` tail of b2 are supplementary (cheap). (3) `bench/threshold` was split per repository (one lock hold of about 25 to
40 min each) instead of one hold per variant. (4) A `--check` of 256 positions per text instead of the plan's single
text: code and shell (the second took 17 min because the CPU reference recomputes in every run).

### 1. Step 1: shared conversion (build `w103` against `w103-base` = main 9563c46)

`scripts/ab.py --a w103-base --b w103 --env IE_PREFILL_FP8=1 --env IE_DIAG=1 --workload server --targets
2000:400,13000:200,30000:200 --ctx 140000`, 6 rounds, one warm-up start. Raw: `/var/tmp/claude/w103/ab/step1.md`.

| Metric (B = shared conversion, A = main, FP8 on in both) | B vs A, mean [95 %] | Expected | Verdict |
|---|---|---|---|
| prefill tok/s @2000 | +0.40 % [+0.24, +0.56] | +0.4 to +2.0 % | at the lower edge |
| prefill tok/s @13000 | +0.57 % [+0.07, +1.07] | +0.4 to +2.0 % | within |
| prefill tok/s @30000 | +0.63 % [+0.53, +0.72] | +0.4 to +2.0 % | within |
| decode tok/s @2000 / 13000 / 30000 | -0.02 % / -0.04 % / +0.12 % (all "equal") | -1 to +1 % | within |

The estimate of +1.2 % was too high by half: the gain is +0.4 to +0.6 % of prefill. The confidence intervals exclude 0
and the means exceed 0.3 %, so the drop rule of the ticket does not apply and the change stays (commit 0d8abae).
Bit-exact: the SHAs of the three requests are equal between old and new and equal to those of the FP8 A/B of
2026-11-02 (`a1c43447d6e6c021`, `d3a8e69dc4cfb94d`, `9aaec77fd9926cc1`), the expert cache counts are identical. With FP8
off nothing changes: `scripts/gpu-status.sh --own --build work/build/w103` (golden gpu, cpu plan, prefill, prefill
dflash, server) reports "golden identical" on ca9a3c2; the `IE_PREFILL_FP8=0` reference of the 14K cut test (NLL
1.117333) equals the one recorded on 2026-11-02.

### 2. Did every run go through the FP8 prefill? (§0.1)

`rocprofv3 --kernel-trace` on a 2,048-token prefill (`/var/tmp/claude/w103/q/prof{0,1}`): FP8 off 0 launches of
`k_gemm8b`, `k_to_fp8`, `k_dequant_fp8`; FP8 on 300 `k_gemm8b` (216 K128 and 84 K256), 192 `k_to_fp8` (so one
conversion serves 1.56 GEMMs) and 300 `k_dequant_fp8`. The NLL of the two runs differs (0.755238 against 0.758756).
In all pairs below the outputs of on and off differ (b1, b2, e: the dumps are not byte-equal; threshold: 589 of 892
replies identical, c1). Server runs: the log shows the prompts going through the prefill (the 62,366-token context of
`bench/threshold` computed in 34.6 s = 1,800 tok/s). The effort tasks have prompts of 78 to 130 tokens; their
prefill is tiny, so that benchmark tests FP8 mostly through the decode of the answers after a small FP8 prefill; I
state this rather than count it as a long-prefill test.

### 3. Quality, FP8 on against off (end state, build `w103`)

Rule fixed in §0.3 before the runs.

**a. CPU reference** (`--teacher TEXT --mega --prefill --max 256 --check 256`, f32 reference, first 256 positions of
the text; -log p of the prefill minus CPU per position):

| Text | | mean (signed) | mean abs | max abs | NLL (prefill / CPU) |
|---|---|---|---|---|---|
| code (`work/corpus/code.txt`) | off | +0.0002 | 0.0920 | 0.96 | 1.5825 / 1.5823 |
| | on | +0.0029 | 0.1496 | 2.30 | 1.5852 / 1.5823 |
| shell (`corp/sh.txt`) | off | -0.0106 | 0.0515 | 2.44 | 0.8208 / 0.8313 |
| | on | +0.0000 | 0.0865 | 2.25 | 0.8314 / 0.8313 |

FP8 deviates from the f32 reference per token 1.63 times (code) and 1.68 times (shell) as much as the f16 prefill
(expectation: at most 2x; within). The mean shift against the reference is small in both directions. A single position
reaches 2.3 nats, as the f16 prefill does (2.4). There is no CI here (256 correlated positions).

**b1. Δ(-log p) of the prefill** (`--teacher TEXT --mega --prefill`, `IE_TEACHER_NLL`, `scripts/nll-pair.py`,
`scripts/nll-compare.py`, 64-token blocks, 4,000 draws). Off twice: bit-equal in both texts. KL and top-1 do not exist
for the prefill (no logits kept).

| Text | Tokens | Δ(-log p), FP8 minus off, mean [95 %] | Rule |
|---|---|---|---|
| code (cpp) | 10,000 | +0.0093 [+0.0040, +0.0147] | **undecided** (mean < 0.01, upper end >= 0.01) |
| shell | 10,000 | +0.0084 [+0.0033, +0.0139] | **undecided** |
| code (cpp), supplementary | 100,000 | +0.0048 [+0.0034, +0.0061] | upper end < 0.01 |
| shell, supplementary | 100,000 | +0.0063 [+0.0046, +0.0079] | upper end < 0.01 |

Against the expectation (+0.003 +- 0.004) the 10K means are **outside** (a finding): FP8 costs about 0.005 to 0.009
nats in the prefill itself, more in the first 10K than later (code: 0.022 over the first 2,000 positions, 0.006 over
2,000 to 10,000; shell 0.011 over 2,000 to 10,000). The 100K sets are the same texts, so they are not independent
evidence for the rule, only a narrower estimate of the same shift (their upper ends stay below 0.01). Under the rule
as written the verdict of the 10K sets is "undecided", which the rule answers with the task benchmark.

**b2. Teacher tail at 32K** (prefill 32,768 tokens in FP8, 4,096 tokens teacher-forced in the megakernel, reference = FP8
off, same text; `IE_TEACHER_DUMP` / `IE_TEACHER_REF`, `nll-compare.py` on `IE_TEACHER_OUT`):

| Text | Δ(-log p), mean [95 %] | KL (mean [95 %]) | top-1 equal | Rule |
|---|---|---|---|---|
| cpp | +0.0003 [-0.0029, +0.0035] | 0.00513 [0.00402, 0.00644] | 99.05 % | **pass** |
| shell | -0.0013 [-0.0056, +0.0030] | 0.01113 [0.00875, 0.01420] | 97.12 % | **pass** |
| py | +0.0009 [-0.0013, +0.0032] | 0.00325 [0.00234, 0.00423] | 99.51 % | **pass** |

Expectation (Δ +0.003 +- 0.004, KL 0.002 to 0.02, top-1 97 to 99.5 %): within. This is the effect of the FP8 state on
the decode-side tokens; the prefill's own predictions (b1) shift more than the tokens after it.

**c1. Tasks, `bench/threshold`** (pydash + boltons, ctx 69,632, MTP, greedy, `IE_DIAG=1`, the same 892 tasks; the off
arm reproduces the earlier reference: holes 86/292 as ref1; questions 556/600 against 557/598):

| Family | off | on | Paired difference [95 % Newcombe] | b lost / c gained | McNemar p |
|---|---|---|---|---|---|
| questions (A) | 556/600 = 92.7 % | 565/600 = 94.2 % | +1.5 pp [+0.1, +3.0] | 4 / 13 | 0.049 |
| code holes (B) | 86/292 = 29.5 % | 90/292 = 30.8 % | +1.4 pp [-3.4, +6.1] | 23 / 27 | 0.67 |

No loss (calibration: first significant loss came with 0.016 nats, -3.4 pp questions and -8.6 pp holes; the
benchmark resolves about 2.5 and 7 pp). The +1.5 pp for questions is a gain at the edge of significance of 12 tests
and is not read as an improvement. The replies change (589 of 892 identical), so FP8 does reach the long context.

**c2. `bench-effort medium`** (23 tasks, ctx 65,536, MTP, greedy; the effort server is not deterministic between
runs): off 22/23 and 22/23, on 20/23 and 22/23. Failures: off (roman) and (csv); on (roman, edit, weekday) and (roman).
The two off runs differ in thinking tokens (32,769 and 40,550), so the spread of the reference is a few tasks. 44/46
against 42/46 is within the "1 to 2 of 46" spread of the reference; the benchmark detects only gross failures.

**d. Needle, 128K, hard** (`bench/needle/run.py --hard --length 131072 --texts 0,1,2,3 --require-cache`, server
`IE_SEQS=1 IE_PROMPT_RAM_GB=0`, ctx 140,000, one server per arm, prefill of 128K about 75 s per text):

| | all | single | multi |
|---|---|---|---|
| off | 80/160 = 50.0 % [42.3, 57.7] | 48/80 | 32/80 |
| on | 78/160 = 48.8 % [41.1, 56.4] | 48/80 | 30/80 |
| on minus off (160 pairs) | -1.2 pp [-5.1, +2.6], McNemar p 0.75 | +0.0, p 1.0 | -2.5, p 0.50 |

Expectation (off about 50 %, on within +-8 pp, p above 0.05): met. By depth: the same curve (81 / 81 / 47 / 34 / 6 %
off against 81 / 81 / 50 / 28 / 3 % on). The needles of one text are not independent (README of the test); 4 texts.

**e. Cut invariance (#81) with `IE_PREFILL_FP8=1`**, tail 64, `IE_DETERMINISTIC=1`, `cmp` of the dumps (-log p and the
logits row of every tail token): mini head 20,000 (chunk by chunk): reference twice and cuts 4097+9001, 6143+13000+13001,
1+19999: all byte-equal, with FP8 on and with FP8 off. Big model, head 14,000 (layer by layer, chunk 6,144): reference
twice, cuts 4001, 12500 and 6000+12000: all byte-equal with FP8 on. FP8 on differs from off in both models (the FP8
path ran: mini NLL 34.657675 against 34.666086, big 1.126413 against 1.117333). **Pass: the prefill with FP8 does not
depend on its cuts**, also with the shared conversion.

### 4. Reading against the rule and recommendation

| Check | Result | Verdict |
|---|---|---|
| a CPU reference | 1.6x the deviation of f16, mean shift +0.003 / +0.000 | within the expectation |
| b1 prefill Δ-NLL 10K, code / shell | +0.0093 / +0.0084, upper ends 0.0147 / 0.0139 | **undecided** (outside the expected +0.003) |
| b2 tail 32K cpp / sh / py | upper ends 0.0035 / 0.0030 / 0.0032, KL 0.003 to 0.011 | pass |
| c1 tasks | questions +1.5 pp, holes +1.4 pp | no loss |
| c2 effort | 44/46 off, 42/46 on | within the spread |
| d needle 128K | 80 against 78 of 160, p 0.75 | no difference |
| e cut invariance | all dumps byte-equal | pass |

The rule gives "undecided" for the prefill's own Δ-NLL at 10K; the answer it names (task benchmark) shows no loss, as
do the tail, the needle and the cut test. Whether "undecided + tasks without loss" is enough to switch the default on
is the maintainer's decision: the measurable cost is about 0.005 to 0.009 nats per prefilled token (perplexity about
+0.5 to +0.9 %) for +1.6 to +3.3 % prefill speed (2026-11-02) plus +0.4 to +0.6 % from step 1; the effect on later
tokens is not visible (+0.000 +- 0.003).

## Decision (maintainer, 2026-11-03): FP8 on by default

The rule of docs/backends/README.md leaves the prefill Δ-NLL at 10K undecided (upper ends 0.0147 and 0.0139), which sends it to
the task benchmark; that shows no loss (questions +1.5 pp, code holes +1.4 pp), and neither do the 32K teacher tail,
the needle at 128K and bench-effort. The cost sits in the prediction of the prompt's own tokens, which no answer uses;
the tokens after the prompt are unchanged within ±0.003 nats. Against it: prefill +2 to +4 % on every prompt. The
default is on since the merge of `100-fp8-default`; `IE_PREFILL_FP8=0` returns to f16. The golden references of the
prefill path and `bench/sha.toml` were recorded again in the same change, after the goldens with `IE_PREFILL_FP8=0`
were identical to the old references.

### R4 after the switch (`2026-11-03-fp8-default-r4.md`, same build, A = `IE_PREFILL_FP8=0`, B = default, 2 rounds)

- A reproduces `bench/sha.toml` (08f0bd31e0c0277f, 536592a9aa29bd56); B is stable over both rounds (6ef803431673370a,
  453d981764536e4f) and is the new reference. Goldens: with `IE_PREFILL_FP8=0` all five identical to the old references;
  recorded again with FP8 on: prefill-dflash, server, server-dflash and four cases of the mini server golden.
- Prefill +4.09 % at 30K and +4.10 % at 130K (expected +2..+4 %: just above).
- Decode -10.4 % at 30K and -6.3 % at 130K (expected -2..+2 %: finding). The cause is the text, not the pass: tokens per
  pass 2.97 -> 2.55 at 30K and 2.82 -> 2.55 at 130K, time per pass unchanged (41.8 and 39.7 ms at 30K). With one answer
  per length this is one sample of the MTP acceptance; the A/B of 2026-11-02 had the other sign at 30K (+9.0 %). The
  `set` workload of the scale run of 2026-11-03 (several answers, FP8 off in part 1 and on in part 2) decides whether
  the acceptance moves systematically.
