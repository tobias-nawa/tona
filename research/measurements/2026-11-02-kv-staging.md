# Where the staged prefill loses against the VRAM KV (#82, part 2)

## Plan 1 (written before the first run)

**Question.** With the KV in pinned RAM (`IE_KV_HOST=1`) the prefill of 6-16K new tokens is 1-2 % slower than with the KV
in VRAM (`2026-11-02-kv-threshold.md`: ctx 49152, 8000 cold -1.5 %, 16000 follow-up -1.9 %). Where does the time go: the
plan (chunk, group, borrowed layers), the copies (`stage_in` / `stage_out`, same stream as the kernels, so they wait),
the attention, or the return of the borrowed slots?

**Step A, counters (cheapest).** `ab.py`, build `w82` (main c3ac8ac), A `IE_KV_HOST=0`, B `=1`, `--ctx 49152`,
one cold prompt of 8000 tokens (7.7K new tokens, the case that lost -1.5 %), 200 decode tokens, 2 rounds without
warm-up, `IE_DIAG=1 IE_PF_DEBUG=1`. Expectation: the same loss as before (-1.5 %, tolerance -3..0 %) and, in the logs of
`IE_PF_DEBUG`, the plan lines (chunk, group, loan limit, KV staging MB, layers borrowed) of both arms; a difference in
chunk/group or in the number of borrowed layers would explain the loss without any copy.
Active tooling: `IE_DIAG=1`, `IE_PF_DEBUG=1` on both arms (debug lines only, no extra sync).

**Step B, profile (one GPU block, if A does not explain it).** `rocprofv3 --kernel-trace --memory-copy-trace` of one 13K
prompt at ctx 49152, both arms; sums per kernel family and of the copies; the profiler slows both arms alike and is
used for shares, not for rates.

## Result of step A (`work/ab/w82-stagingA*`, 2 fixed rounds AB/BA, no warm-up)

Prefill@8000 (7,721 new tokens): A 1,713.3 tok/s (1,713.2-1,713.4), B 1,689.0 (1,688.6-1,689.4), B against A -1.42 %
[-2.23, -0.61]: the loss of the threshold run is reproduced (inside the plan). Decode@8000 +0.79 % (open).
**The plan differs** (`IE_PF_DEBUG` lines):

| arm | slots | layered prefill chunk / group | chunks of the 7,721 tokens | KV staging | layers borrowed |
|---|---:|---|---|---|---:|
| A VRAM | 71 | 4096 / 8192 | 4096 + 3625 | none | not printed |
| B RAM | 87 | 6144 / 12288 | 6144 + 1577 | 9 MB | 23 |

So B takes the large chunk (`IE_PF_LCHUNK` = 6144; the plan of A is limited by its smaller loan room) and the second chunk
holds only 1,577 tokens. The copies are small here (the staging holds 9 MB at pos 0: `stage_in` copies nothing, `stage_out`
9 MB). First hypothesis: the chunk split, not the copies.

## Plan 2 (before the next run)

Same call as step A, B with `IE_PF_LCHUNK=4096` (chunk 4096 / group 8192 like A, staging unchanged). Expectation: if
the chunk split causes the loss, B against A is within -0.5..+1 % (tolerance of the 2-round spread, 0.1 %). If the loss
stays at about -1.4 %, the copies or the borrowed layers cause it and step B (profile) follows.

## Result of plan 2 (`work/ab/w82-stagingB*`)

B (RAM) with `IE_PF_LCHUNK=4096`: prefill@8000 1,712.9 against A (VRAM) 1,712.9 (1,711.7-1,714.1): **-0.00 %**
[-0.82, +0.82]; decode@8000 +1.61 %. Plan lines of B: chunk 4096, group 8192, staging 9 MB, 18 layers borrowed. The loss
is the **chunk split**, not the staging copies: 6144 + 1577 tokens is slower than 4096 + 3625 by 1.4 %. The staging
itself costs nothing measurable at this size (the copies are 9 MB per layer; rocprofv3 not needed).

## Change and plan 3 (before the runs)

`choose_plan` (engine/gpu/prefill_entry.cpp): for a prompt of one group with at least two chunks the chunk becomes
`ceil(n / k)` rounded up to 128 (k = chunks at the old size), if that fits the loan limit. 7,721 tokens: 2 x 3968
instead of 6144 + 1577; 12.6K: 3 x 4224 instead of 6144 + 6144 + 400. Layered plans only; chunk-by-chunk and the groups
of long prompts are unchanged. Numbers: a different chunk changes the GEMM tile grouping, so bit-exactness is not
given a priori: `IE_DETERMINISTIC=1` SHAs against the base build and the goldens decide.

Runs (`ab.py`, A `work/build/w82-base` = c3ac8ac binary, B `work/build/w82` with the change, same env, 2 fixed rounds,
no warm-up, `IE_DIAG=1 IE_PF_DEBUG=1`, ctx 49152):

| run | env both arms | prompt (new tokens) | expectation, B against A prefill |
|---|---|---|---|
| R1 | `IE_KV_HOST=1` | 8000 cold (7.7K) | +1.0..+2.0 % (the loss of step A is recovered) |
| R2 | `IE_KV_HOST=1` | 13000 cold (12.6K; 6144+6144+400 before) | 0..+4 % |
| R3 | `IE_KV_HOST=0` (the reference at small ctx) | 8000 cold | -0.5..+1.5 %, must not be worse |
| R4 | `IE_KV_HOST=1` | 30000 cold (several chunks) | -0.5..+1 % |
Decode is unchanged by the change (-1..+1 % expected, same pass).

## Result of plan 3 (`work/ab/w82-chunk-R*`, B = balanced chunks, A = c3ac8ac binary, 2 fixed rounds)

| run | plan A -> B (chunk, layers borrowed) | prefill B against A | decode B against A |
|---|---|---|---|
| R1 RAM, 8000 | 6144, 23 -> 3968, 17 | **+1.52 %** [+0.70, +2.33] (1,689.1 -> 1,714.8 tok/s) | +0.86 % |
| R2 RAM, 13000 | 6144, 24 -> 4352, 19 | **-0.58 %** [-1.39, +0.24], open (outside the plan: finding) | +0.73 % |
| R3 VRAM, 8000 | 4096 -> 3968 | +0.22 % [-0.59, +1.04] | +1.73 % |
| R4 RAM, 30000 | identical plans (several groups) | +0.08 % [-0.74, +0.90] | +1.93 % |

R4 is an A/A of the two binaries (same plan): prefill equal, so prefill comparisons are sound; the decode rows of R3 and
R4 (+1.7, +1.9 % with no change in the decode path) are the noise of 200 decoded tokens in 2 rounds (A: 66.9 / 68.4
tok/s) and say nothing. Findings:

- Balanced chunks recover the loss at 8K (R1) and cost 0.6 % at 13K (R2): the rule "equal chunks" is not right on its
  own. The number of **borrowed layers** matters too: 23 against 17 layers at 8K. Returning the loan (the same experts
  back into the slots over PCIe) is part of the prefill time: about 11.5 ms per borrowed layer by the difference of R1
  (6 layers, 69 ms of 4.57 s).
- A bigger chunk is more efficient per token (R2: 6144 + 6144 + 611 beats 3 x 4352 although 5 layers more are
  borrowed).

Next: compare the arms that decide (VRAM against RAM) with this build, and sweep the chunk for 13K if RAM stays behind.

## Plan 4 (before the runs): the deciding arms with the new build

`work/build/w82` (balanced chunks), A `IE_KV_HOST=0`, B `IE_KV_HOST=1`, `IE_DIAG=1 IE_PF_DEBUG=1`, no other switch.
(a) ctx 49152, the chain of the threshold run: 2000, 8000, 16000, 32000, 45000, 600 decode tokens, `bench-set`, 4 rounds
max with warm-up (as `2026-11-02-kv-threshold.md`; there the prefill rows were -1.5, -1.9, +3.5, +0.2 %, decode +1 to
+2 %). Expectation: prefill@8000 and @16000 within -0.5..+1 % (the loss gone), the others unchanged (+/-0.7 %),
decode +1..+2 %. (b) the same at ctx 32768 with prompts 2000, 8000, 16000, 31000 (before: -2.1, -1.7, -2.1 %): same
expectation for 8000 and 16000. (c) cold 13000 at ctx 49152, 2 fixed rounds: unknown, the base had no VRAM/RAM pair; the
decision criterion is B not slower than -0.5 %.

## Result of plan 4 (`work/ab/w82-dec-{a,b,c}`), RAM against VRAM with the balanced chunks, B vs A in percent

Bold = decided (interval excludes 0). Rounds: a 4, b 4, c 2. Before (threshold run, base build) in brackets.

| ctx / run | 2000 prefill | 8000 prefill | 16000 prefill | 31-32K prefill | 45K prefill | decode 2000 / 8000 / 16000 / 31-32K / 45K | set code / text |
|---|---|---|---|---|---|---|---|
| 49152 (a) | **+1.33** [+1.11] | -0.27 [-1.49] | -0.24 [-1.89] | **+3.88** [+3.51] | +0.65 [+0.16] | **+1.43 / +1.66 / +1.24 / +1.23 / +1.53** | **+1.74 / +2.37** |
| 32768 (b) | +0.65 [+0.73] | -0.39 [-2.14] | -0.24 [-1.71] | **-1.26** (31000) [-2.09] | | +1.02 / +0.89 / +0.61 / +0.71 (open) | **+1.17 / +1.39** |
| 49152 cold 13000 (c) | | | 13000: **+2.32** (not measured before) | | | **+2.04** | |

Open = the interval contains 0 (all of them within [-1.1, +0.5] for the negative means). SHAs of A and B equal in all
rounds and prompts (the same values as in the threshold run, which also shows the new chunks change no output of the
greedy runs). The remaining loss at ctx 32768 is the 15,288-token prefill (31000 request, chunk 5120 in both arms, B
borrows 21 layers, staging 36 MB): the slot gap between the arms is only 10 slots there (87 against 77; 16 at ctx 49152,
21 at 70000), which pays for the staging at 49152 and not at 32768. The return of the loan also copies more slots per
layer in B (87 against 77).

Decision for the threshold: **0.6 GiB of KV** (ctx about 46.5K at 13.86 KB per position): ctx 49152 (0.63 GiB) is
inside and nowhere decided worse; ctx 32768 (0.42 GiB) stays in VRAM with its -1.26 % at 15K new tokens.
Still to show: the reference (ctx 140000 and 262144, RAM already) does not get worse by the chunk change.

## Plan 5 (before the runs): the chunk change against the reference

A `work/build/w82-base` (c3ac8ac), B `work/build/w82`, both arms default KV place, `IE_DIAG=1`, 2 fixed rounds, no
warm-up: (a) ctx 262144 (service default), cold 8000 and cold 13000 prompts (one run each), (b) `mid` (ctx 140000, 30K
cold, several chunks in one group), (c) `long` is not needed (groups of 12288 and more are unchanged: only single-group
plans are rebalanced). Expectation: (a) 8000 +1..+2 %, 13000 -0.7..+0.5 % (R2 at ctx 49152 gave -0.58 % open: if
this is the real effect, the rule needs a refinement), (b) -0.5..+0.5 % (R4-like). Then deterministic SHAs of base against
new (`IE_DETERMINISTIC=1`, 8000 and 13000 and 30000 at ctx 262144).

## Result of plan 5 (`work/ab/w82-ref-*`, A = c3ac8ac binary, B = balanced chunks, both default KV place, ctx 262144 / 140000)

| run | plan A -> B (chunk) | prefill B against A | decode | SHA A = B |
|---|---|---|---|---|
| a8 (8000 cold) | 5120 -> 3968 | +0.78 % [-0.05, +1.60] | +0.77 % | yes |
| a13 (13000 cold) | 5120 -> 4352 | **-0.21 %** [-1.03, +0.61] | -0.35 % | yes |
| b (`mid`, 30K, ctx 140000) | 6144 -> 5120 | -0.41 % [-1.38, +0.56] | +0.77 % | yes |
| d8 (`IE_DETERMINISTIC=1`, 8000) | 5120 -> 3968 | +2.36 % [+0.24, +4.48] | +1.96 % | yes (1aaf29af01af6f0d, as at ctx 16384-70000) |
| d13 (deterministic, 13000) | 5120 -> 4352 | -0.39 % [-1.22, +0.45] | +2.27 % | yes (522cb626022ab223) |
| d30 (deterministic, 30000) | | +0.08 % [-0.96, +1.13] | +0.55 % | yes, and equal to `bench/sha.toml` (08f0bd31e0c0277f) |

Bit-exact at all sizes (chunk size does not change a number). Rates: one chunk plan helps (8K, two chunks), the
three-chunk plans lose 0.2-0.6 % (a13, d13, R2 of plan 3; `mid` -0.4 % open). So the rule is narrowed to **exactly two
chunks** (k = 2): there B gains +0.8 to +2.4 % at 8K; for k >= 3 the plan is the old one (identical to the base build, so
a13, d13 and b are then trivially equal). The decision numbers of plan 4 were taken with the all-k rule: they are repeated
with the final build.

## Plan 6 (before the final runs)

Final build `work/build/w82` = k = 2 rule + threshold 0.6 GiB (the threshold changes nothing in `IE_KV_HOST=0/1` runs).
Runs as plan 4 (a: ctx 49152 chain, b: ctx 32768 chain, c: cold 13000 at 49152; A `IE_KV_HOST=0`, B `=1`; now the
VRAM arm also has the new plan). Expectation: as plan 4's results: 8000 and 16000 within -0.5..+0.5 % (the k = 2 cases),
13000 and 32000 positive (+2..+4 %), decode +1..+2 %. Then default-place runs of the new default (ctx 49152 without
`IE_KV_HOST`, one start each, the log line must say "in RAM") and the checks of the Rules (mini goldens, `gpu-status.sh
--own`).

## Result of plan 6 (`work/ab/w82-fin-{a,b,c}`), final build, RAM (B) against VRAM (A), percent, as expected by plan 6

Both arms with the k = 2 rule. Bold = decided. Rounds a 4, b 4, c 2.

| ctx / run | 2000 prefill | 8000 prefill | 16000 prefill | 31-32K prefill | 45K prefill | decode 2000 / 8000 / 16000 / 31-32K / 45K | set code / text |
|---|---|---|---|---|---|---|---|
| 49152 (a) | **+1.05** | -0.30 [-1.05, +0.45] | -0.27 [-0.96, +0.41] | **+2.84** | +0.14 [-0.59, +0.87] | **+1.69 / +1.57 / +1.45 / +2.16 / +2.31** | **+2.17 / +2.12** |
| 32768 (b) | +1.12 | -0.22 [-0.94, +0.50] | +0.47 | **-1.40** (31000) | | **+1.68** / +0.40 / +0.69 / +0.59 | **+1.61 / +1.96** |
| 49152, cold 13000 (c) | | 13000: **+1.77** | | | | +0.60 | |

(Before, threshold run, ctx 49152: 8000 -1.49 **decided**, 16000 -1.89 **decided**; ctx 32768: 8000 -2.14, 16000 -1.71,
31000 -2.09, all decided.) At ctx 49152 the RAM place is no longer decided slower anywhere; 8000 and 16000 remain at
-0.3 % with intervals that include 0 (margin 0.5 %: "equal" is not proven, a loss up to about 1 % is not excluded).
At ctx 32768 the 15,288-token prefill (31000) stays decided slower, -1.4 %.

## Decision and change

- `choose_plan`: one group of exactly two chunks gets equal halves (prefill_entry.cpp). Bit-exact (SHAs, also
  deterministic and equal to `bench/sha.toml`), +0.8 to +2.4 % prefill at 8K on the reference place (ctx 262144), the
  loss of the RAM place at 6-16K new tokens (-1.5 to -1.9 %) shrinks to -0.3 % (open).
- Threshold `kv_bytes > 0.6 GiB` (ctx about 46.5K) instead of 1 GiB: ctx 49152 and 70000 move to RAM (decode +1.2 to
  +3 % and 16-21 more slots), ctx 32768 stays in VRAM (-1.4 % prefill at 15K new tokens, decode +0.4 to +1.7 %).
- Not solved: the staging loss at ctx 32768 with only 10 more slots. A lower threshold needs the 21-layer loan (return of
  4.5 GB of experts, about 11.5 ms per layer) to get cheaper or the plan to borrow fewer layers: an idea, not part of
  this change.

## Checks of the final commit

`just check` green, `ctest -L unit` 41 of 41, mini goldens (`golden.sh` gpu, cpu plan, prefill; `golden-server.sh`) all
identical, `scripts/gpu-status.sh --own --build work/build/w82` on the big model: golden identical (prefill dflash, gpu,
cpu plan, server, prefill). `choose_plan` was split into `plan_fits` and `balanced_plan` after the measurements (the
cognitive complexity gate); same decisions, the goldens ran on the split version.

## Plan 7 (before the runs): final build against the old build, default place, cold prompts, prefill only

Why: `balanced_plan` acts on every layered plan with exactly two chunks, also with the KV in VRAM and in RAM at the
reference contexts (Rule 1: the reference does not get worse). The earlier old-against-new runs (plan 5) used the
rule for all k; the final rule (k = 2) was measured only RAM against VRAM.

A = `work/build/w82-base` (the ie-server binary of c3ac8ac, copied before the change; `work/build/merged` is not used because its commit is not guaranteed), B = `work/build/w82` (final, 5bdc1e6 + 6cde2fa). Both arms the same environment (no
`IE_KV_HOST`, so the default place: ctx 32768 VRAM, ctx 262144 RAM), `IE_DIAG=1 IE_PF_DEBUG=1` (plan lines of both
arms in the logs before the rates), one cold prompt per start, 200 decode tokens, `ab.py` with warm-up, 2 to 4 rounds
(no noise entry for these targets at these contexts: at least 2 rounds).

| run | ctx (place) | prompt (new tokens) | expectation, B against A prefill |
|---|---|---|---|
| a | 32768 (VRAM) | 8000 (7.7K, k = 2 at A's chunk) | 0..+1.5 % (halves instead of 4096 + 3625; R3 of plan 3 gave +0.22 %) |
| b | 32768 (VRAM) | 13000 (12.6K) | -0.5..+0.5 % (k = 3 or 4: plan unchanged if k != 2) |
| c | 262144 (RAM) | 13000 | -0.5..+0.5 % (k = 3 at chunk 5120: unchanged; if the plan line shows k = 2 the rule acts) |
| d | 262144 (RAM) | 30000 | -0.5..+0.5 % (several chunks) |

Decode is not evaluated (same pass; expectation -1.5..+1.5 %). Decision rule: any prefill "B slower" decided -> narrow
`balanced_plan` to the cases where it is measured to help and measure again.

## Result of plan 7 (`work/ab/w82-old-{a,b,c,d}`, A = c3ac8ac binary, B = final build, default place)

Plan lines first (`IE_PF_DEBUG`, both arms, all rounds):

| run | A plan | B plan |
|---|---|---|
| a, ctx 32768 VRAM, 7,721 tokens | chunk 5120 (5120 + 2601), group 10240, loan limit 3988 MB | chunk 3968 (2 x 3968) |
| b, ctx 32768 VRAM, 12,899 | chunk 5120 (3 chunks), group 15360 | the same (k = 3, rule not applied) |
| c, ctx 262144 RAM, 12,899 | chunk 5120, group 15360, loan limit 4091 MB | the same |
| d, ctx 262144 RAM, 29,641 | chunk 5120, group 15360 | the same |

Prefill, B against A (rounds 2, 4, 4, 1):

| run | prefill | decode | verdict |
|---|---|---|---|
| a | **+1.24 %** [+0.19, +2.29] (1,685.9 -> about 1,707 tok/s) | +0.58 % (open) | faster, as expected |
| b | -0.03 % [-0.72, +0.65] | +0.51 % | equal plan, noise |
| c | -0.43 % [-1.42, +0.55] | +0.75 % | equal plan, noise (A rounds 1,740-1,761 tok/s) |
| d | +0.36 % [+0.24, +0.48] (decided after round 1 by the known noise of 30K) | +0.96 % | equal plan, noise |

Nowhere decided slower. In b, c and d the plan is identical (the rule changes nothing for k >= 3: 13K and 30K have
three and six chunks at chunk 5120), so the differences are the noise of two binaries with the same plan. The only
plan the final rule changes in these cases (a) is 1.2 % faster. A chunk of 6,500 or more with 13K tokens would be
k = 2: it does not occur at the default (chunk 5120 at both contexts; 6144 only with a larger loan limit).

## Decision (maintainer, 2026-11-02): threshold 0.4 GiB

The rule is not "nowhere worse": a loss in one case is fine when the gain elsewhere outweighs it and is justified. At
ctx 32768 (plan 6, b) the agent set is decided faster in RAM, +1.61 % (code) and +1.96 % (text), and so is decode at
2000 tokens (+1.68 %). The one decided loss is the cold 31000-token request (15,288 new tokens), -1.40 % prefill, a
full-context fill that agent turns rarely make (the prefix cache serves the history). So the KV goes to RAM from 0.4
GiB (ctx about 31K). At ctx 16384 (2026-11-02-kv-threshold.md) RAM wins nothing decided, so that stays in VRAM.
