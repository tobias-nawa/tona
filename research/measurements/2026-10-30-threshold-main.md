# Threshold main run: does 0.01 nats predict task accuracy? (#76, part 2)

Question: is the quality rule of `docs/backends/README.md#checks` (95 % interval of Δ(-log p) contains 0 and lies below
0.01 nats) tied to a loss in task success, and where on the Δ-NLL axis does a loss first become significant?
Method: graded degradation with `IE_IDX_SHARE` (decode only; 12 / 8 / 6 / 5 / 4 / 2 / 1 full indexer layers), per stage
Δ-NLL (method of #73) and the same stage on the benchmark of part 1 (`bench/threshold/README.md`,
`2026-10-30-threshold-pilot.md`). Raw data (contexts, tasks, all runs, logs, dumps): `/var/tmp/claude/w76/`.

Rule and analysis fixed and committed before the first stage run: **306c1d8** (README section "Main run (part 2)",
`curve.py` with the zones). Later commits only added the Δ-NLL scripts and the options `--axis`/`--repo` of
`curve.py`; the definitions did not change.

## Setup

- Build `work/build/w76` built from this worktree at main 9b8a50e (the whole main run and all Δ-NLL runs come from
  this one binary). MTP, greedy (`temperature 0`), **thinking off** (`enable_thinking=false`),
  `--ctx 69632`, port 8092, one hold of the GPU lock per run and 2 minutes of pause after it (worker #78 slotted
  runs in between).
- Tasks: pydash (MIT, 812eb13, 62,269-token context, 116 holes) and boltons (BSD-3, d31d821, 62,130 tokens, 176 holes).
  Family B = every function that qualifies (3 to 35 body lines, >= 80 % covered, the suite fails without its body;
  292 tasks; the self-test passes the original function for all 292). Family A = 300 questions per repo, eight kinds
  (default, constant, base, caller, doc, returns, import, owner), sources at least 2,400 tokens before the end except 18
  local controls. File budget adjusted until the holed context is about 62.3K tokens (prompt below 64K). B reply limit
  2,000 tokens (30 of 292 replies of ref1 still hit it).
- Hole positions in eighths of the context: pydash 11 / 16 / 14 / 8 / 6 / 19 / 23 / 19, boltons 11 / 8 / 11 / 20 / 39 /
  48 / 25 / 14: all qualifying functions, so spread by the code and not even (pydash is thin in the 4th and 5th eighth, boltons in
  the first three).
- Stages (layer 0 always full, spread over the 12 QSA layers): 8 = 0,1,3,4,6,7,9,10 (#73), 6 = 0,2,4,6,8,10, 5 =
  0,2,5,7,10 (spacing 2,3,2,3,2; my choice), 4 = 0,3,6,9, 2 = 0,6, 1 = 0; 12 full layers = ref1. Stages 6 and 4 once
  more (`s6b`, `s4b`) with another task order (`--shuffle 76`: another cache history). `ref2` = `IE_DRAFT_N=1` (the server has
  no mode without a drafter; the other pass widths change the rounding, so ref2 measures f0, the noise of the reference
  itself). Every variant differs from ref1 (`check_differs.py`): identical replies s6b/s6 847 of 891, s4b/s4 846 of 892.
- Reference accuracy: A 557 of 598 (93.1 %; pydash 267/299, boltons 290/299, local controls 16/18), B 86 of 292
  (29.5 %; pydash 47/116, boltons 39/176). Closed-book (repo removed, A only): 81 of 600 (13.5 %; doc 48/137, default 18/132,
  import 8/72, caller 3/39, constant 1/59, local 3/18, base, owner, returns 0).
- Requests that failed with a barrier abort (HTTP 500) are missing from the run and drop out of the pairing:
  ref1 pydash-A007 and boltons-A161, s6 pydash-A051, s1 pydash-A250 (A: 598 pairs instead of 600).
- Request time: sum over the 11 runs 12.0 h; per run 1.1 to 1.5 h (A 0.8 s per question, B 10 to 14 s per task, mean 880
  generated tokens: Qwen writes a short analysis before the code although thinking is off).

## Accuracy per stage, paired against ref1 (all tasks)

b = right in ref1 and wrong in the run, c = the opposite, f = (b + c) / n; McNemar exact; Newcombe 95 % interval of the
difference (paired). Medians of seconds per task.

| run | family | n | acc % | b (lost) | c (gained) | f | McNemar p | diff pp [95 % Newcombe] | s/task |
|---|---|---|---|---|---|---|---|---|---|
| ref2 | A | 598 | 93.3 | 1 | 2 | 0.005 | 1.000 | +0.2 [-0.6, +1.0] | 0.8 |
| ref2 | B | 292 | 27.7 | 15 | 10 | 0.086 | 0.424 | -1.7 [-5.1, +1.7] | 13.8 |
| s8 | A | 598 | 93.3 | 4 | 5 | 0.015 | 1.000 | +0.2 [-0.9, +1.3] | 0.8 |
| s8 | B | 292 | 28.1 | 19 | 15 | 0.116 | 0.608 | -1.4 [-5.3, +2.6] | 10.4 |
| s6 | A | 597 | 93.3 | 4 | 5 | 0.015 | 1.000 | +0.2 [-0.9, +1.3] | 0.8 |
| s6 | B | 292 | 30.1 | 19 | 21 | 0.137 | 0.875 | +0.7 [-3.6, +4.9] | 10.1 |
| s6b | A | 598 | 93.3 | 4 | 5 | 0.015 | 1.000 | +0.2 [-0.9, +1.3] | 0.8 |
| s6b | B | 292 | 29.8 | 20 | 21 | 0.140 | 1.000 | +0.3 [-4.0, +4.6] | 10.6 |
| s5 | A | 598 | 93.3 | 5 | 6 | 0.018 | 1.000 | +0.2 [-1.0, +1.4] | 0.8 |
| s5 | B | 292 | 30.5 | 22 | 25 | 0.161 | 0.771 | +1.0 [-3.6, +5.6] | 11.0 |
| s4 | A | 598 | 92.5 | 7 | 3 | 0.017 | 0.344 | -0.7 [-1.9, +0.5] | 0.8 |
| s4 | B | 292 | 29.5 | 18 | 18 | 0.123 | 1.000 | +0.0 [-4.0, +4.0] | 11.1 |
| s4b | A | 598 | 92.5 | 7 | 3 | 0.017 | 0.344 | -0.7 [-1.9, +0.5] | 0.8 |
| s4b | B | 292 | 29.5 | 18 | 18 | 0.123 | 1.000 | +0.0 [-4.0, +4.0] | 11.1 |
| s2 | A | 598 | 92.3 | 15 | 10 | 0.042 | 0.424 | -0.8 [-2.6, +0.9] | 0.8 |
| s2 | B | 292 | 26.0 | 30 | 20 | 0.171 | 0.203 | -3.4 [-8.2, +1.3] | 10.0 |
| s1 | A | 597 | 89.8 | 29 | 9 | 0.064 | 0.002 | -3.4 [-5.5, -1.4] | 0.8 |
| s1 | B | 292 | 20.9 | 49 | 24 | 0.250 | 0.005 | -8.6 [-14.2, -2.9] | 10.1 |

- Only the stage with one full layer loses significantly: A -3.4 pp [-5.5, -1.4], B -8.6 pp [-14.2, -2.9].
- Two full layers: A -0.8 pp [-2.6, +0.9], B -3.4 pp [-8.2, +1.3], not significant, but the interval does not exclude
  -2.6 pp (A) and -8.2 pp (B). 4 to 8 full layers: |difference| <= 1.4 pp, all intervals contain 0.
- Noise of the reference itself (ref2): f0 = 0.005 (A) and 0.086 (B); f of the stages rises 0.015 (8 layers) to 0.064
  (1 layer) on A and 0.116 to 0.250 on B. The noise floor f0 on B (8.6 %) is half of the flip rate at two layers (17 %).
- Resolution reached (d = 2.8 sqrt(f / n)): A at f = 0.042 (2 layers) 2.3 pp, at f = 0.064 2.9 pp; B at f = 0.171 6.8 pp,
  at f = 0.25 8.2 pp. So A resolves 3 pp, B only about 7 to 8 pp (the plan asked for 5 pp, which would need 540 to 800
  B tasks at these f; two repos give 292).
- Repeats with another task order: s6b equals s6 on A and within 0.4 pp on B; s4b equals s4 in all counts (846 of 892 replies identical).

## Δ-NLL per stage

Method of #73: 4,096 tail tokens per text run teacher-forced through the megakernel (the decode path, `IE_TEACHER_TAIL`)
after the prefill of the rest; Δ = -log p of the stage minus that of the reference on the same tokens; 95 % interval by
block bootstrap (64-token blocks, 4,000 draws, `scripts/nll-compare.py`); KL(reference || stage) and top-1 agreement are
means. Texts: py, cpp, sh at 32K prefill + 4,096 tail (as in #73; 12,288 tokens pooled) and the tails of the two benchmark
contexts (pydash 58,173 + 4,096, boltons 58,034 + 4,096; 8,192 tokens pooled). The benchmark tail is the holed
context file as plain text, so it ends in the last file of the context (`functions.py` / `statsutils.py`, cut at a statement).
8 / 6 / 5 / 4 full layers on py/cpp/sh match the numbers of #73 where it has them (6: +0.0008, +0.0045, -0.0011).

| full layers | texts | n tokens | dNLL mean [95 %] | KL | top-1 equal % |
|---|---|---|---|---|---|
| 8 | py-32768-4096 | 4096 | +0.0029 [-0.0009, +0.0068] | 0.0033 | 99.4 |
| 8 | cpp-32768-4096 | 4096 | +0.0033 [-0.0006, +0.0071] | 0.0060 | 99.0 |
| 8 | sh-32768-4096 | 4096 | -0.0033 [-0.0091, +0.0025] | 0.0096 | 97.8 |
| 8 | pydash-58173-4096 | 4096 | +0.0018 [-0.0006, +0.0042] | 0.0033 | 99.2 |
| 8 | boltons-58034-4096 | 4096 | +0.0324 [-0.0138, +0.0832] | 0.0836 | 94.6 |
| 8 | **pooled lm** | 12288 | +0.0010 [-0.0018, +0.0036] | 0.0063 | 98.7 |
| 8 | **pooled bench** | 8192 | +0.0171 [-0.0051, +0.0429] | 0.0435 | 96.9 |
| 6 | py-32768-4096 | 4096 | +0.0008 [-0.0018, +0.0034] | 0.0037 | 99.4 |
| 6 | cpp-32768-4096 | 4096 | +0.0045 [+0.0004, +0.0087] | 0.0079 | 98.6 |
| 6 | sh-32768-4096 | 4096 | -0.0011 [-0.0059, +0.0037] | 0.0111 | 97.3 |
| 6 | pydash-58173-4096 | 4096 | +0.0058 [+0.0014, +0.0110] | 0.0074 | 99.2 |
| 6 | boltons-58034-4096 | 4096 | +0.0360 [-0.0124, +0.0873] | 0.0992 | 94.5 |
| 6 | **pooled lm** | 12288 | +0.0014 [-0.0010, +0.0038] | 0.0076 | 98.4 |
| 6 | **pooled bench** | 8192 | +0.0209 [-0.0039, +0.0486] | 0.0533 | 96.8 |
| 5 | py-32768-4096 | 4096 | +0.0017 [-0.0016, +0.0055] | 0.0045 | 99.3 |
| 5 | cpp-32768-4096 | 4096 | +0.0035 [-0.0007, +0.0076] | 0.0093 | 98.8 |
| 5 | sh-32768-4096 | 4096 | -0.0006 [-0.0059, +0.0047] | 0.0120 | 96.6 |
| 5 | pydash-58173-4096 | 4096 | +0.0036 [-0.0005, +0.0077] | 0.0065 | 99.1 |
| 5 | boltons-58034-4096 | 4096 | +0.0119 [-0.0185, +0.0483] | 0.1007 | 94.0 |
| 5 | **pooled lm** | 12288 | +0.0015 [-0.0009, +0.0041] | 0.0086 | 98.3 |
| 5 | **pooled bench** | 8192 | +0.0078 [-0.0081, +0.0263] | 0.0536 | 96.5 |
| 4 | py-32768-4096 | 4096 | +0.0032 [-0.0008, +0.0076] | 0.0056 | 99.4 |
| 4 | cpp-32768-4096 | 4096 | +0.0050 [+0.0003, +0.0103] | 0.0114 | 98.6 |
| 4 | sh-32768-4096 | 4096 | -0.0007 [-0.0059, +0.0048] | 0.0139 | 97.3 |
| 4 | pydash-58173-4096 | 4096 | +0.0122 [+0.0045, +0.0217] | 0.0138 | 99.1 |
| 4 | boltons-58034-4096 | 4096 | +0.0436 [-0.0071, +0.0988] | 0.0930 | 94.4 |
| 4 | **pooled lm** | 12288 | +0.0025 [-0.0004, +0.0055] | 0.0103 | 98.4 |
| 4 | **pooled bench** | 8192 | +0.0279 [+0.0015, +0.0579] | 0.0534 | 96.8 |
| 2 | py-32768-4096 | 4096 | +0.0091 [+0.0027, +0.0164] | 0.0124 | 99.1 |
| 2 | cpp-32768-4096 | 4096 | +0.0216 [+0.0099, +0.0346] | 0.0298 | 98.0 |
| 2 | sh-32768-4096 | 4096 | -0.0034 [-0.0100, +0.0038] | 0.0225 | 96.5 |
| 2 | pydash-58173-4096 | 4096 | +0.0220 [+0.0116, +0.0339] | 0.0191 | 98.5 |
| 2 | boltons-58034-4096 | 4096 | +0.0212 [-0.0344, +0.0831] | 0.1577 | 92.3 |
| 2 | **pooled lm** | 12288 | +0.0091 [+0.0038, +0.0147] | 0.0216 | 97.9 |
| 2 | **pooled bench** | 8192 | +0.0216 [-0.0061, +0.0534] | 0.0884 | 95.4 |
| 1 | py-32768-4096 | 4096 | +0.0147 [+0.0029, +0.0277] | 0.0256 | 98.5 |
| 1 | cpp-32768-4096 | 4096 | +0.0324 [+0.0193, +0.0479] | 0.0462 | 97.1 |
| 1 | sh-32768-4096 | 4096 | +0.0017 [-0.0080, +0.0133] | 0.0313 | 95.9 |
| 1 | pydash-58173-4096 | 4096 | +0.0264 [+0.0154, +0.0377] | 0.0291 | 97.9 |
| 1 | boltons-58034-4096 | 4096 | -0.1499 [-0.2563, -0.0553] | 0.2036 | 91.2 |
| 1 | **pooled lm** | 12288 | +0.0163 [+0.0091, +0.0240] | 0.0344 | 97.2 |
| 1 | **pooled bench** | 8192 | -0.0618 [-0.1201, -0.0127] | 0.1163 | 94.5 |

- On the 32K texts Δ-NLL grows smoothly: +0.001 (8 layers), +0.0014, +0.0015, +0.0025 (4), +0.0091 (2), +0.0163 (1).
- The tails of the benchmark contexts are different. pydash is smooth and larger: +0.002, +0.006, +0.004, +0.012, +0.022,
  +0.026. boltons is dominated by a few tokens: KL 0.08 to 0.20, intervals of +-0.05 and a **negative** Δ-NLL of -0.15
  [-0.26, -0.06] for one full layer (the stage with the largest damage), top-1 agreement 91 to 95 %. Its mean Δ-NLL is
  not a usable axis; pooling it with pydash hides that (pooled 1 layer: -0.062).
- The bootstrap intervals assume 64-token blocks are independent; the boltons tail has fewer effective blocks than the
  32K texts (64 blocks per text), so its intervals are rough.

## Curve: accuracy loss against Δ-NLL

Zone: *damage* = upper bound of the difference < 0; *safe* = lower bound > -3 pp (A) / -5 pp (B); else *open*.

### Δ-NLL of the 32K texts (py + cpp + sh pooled), all 598 / 292 tasks

| full layers | family | n | loss pp [95 %] | b / c | McNemar p | zone | dNLL mean [95 %] | KL | top-1 % |
|---|---|---|---|---|---|---|---|---|---|
| 8 (s8) | A | 598 | +0.2 [-0.9, +1.3] | 4 / 5 | 1.000 | safe | +0.0010 [-0.0018, +0.0036] | 0.0063 | 98.7 |
| 8 (s8) | B | 292 | -1.4 [-5.3, +2.6] | 19 / 15 | 0.608 | open | +0.0010 [-0.0018, +0.0036] | 0.0063 | 98.7 |
| 6 (s6) | A | 597 | +0.2 [-0.9, +1.3] | 4 / 5 | 1.000 | safe | +0.0014 [-0.0010, +0.0038] | 0.0076 | 98.4 |
| 6 (s6b) | A | 598 | +0.2 [-0.9, +1.3] | 4 / 5 | 1.000 | safe | +0.0014 [-0.0010, +0.0038] | 0.0076 | 98.4 |
| 6 (s6) | B | 292 | +0.7 [-3.6, +4.9] | 19 / 21 | 0.875 | safe | +0.0014 [-0.0010, +0.0038] | 0.0076 | 98.4 |
| 6 (s6b) | B | 292 | +0.3 [-4.0, +4.6] | 20 / 21 | 1.000 | safe | +0.0014 [-0.0010, +0.0038] | 0.0076 | 98.4 |
| 5 (s5) | A | 598 | +0.2 [-1.0, +1.4] | 5 / 6 | 1.000 | safe | +0.0015 [-0.0009, +0.0041] | 0.0086 | 98.3 |
| 5 (s5) | B | 292 | +1.0 [-3.6, +5.6] | 22 / 25 | 0.771 | safe | +0.0015 [-0.0009, +0.0041] | 0.0086 | 98.3 |
| 4 (s4) | A | 598 | -0.7 [-1.9, +0.5] | 7 / 3 | 0.344 | safe | +0.0025 [-0.0004, +0.0055] | 0.0103 | 98.4 |
| 4 (s4b) | A | 598 | -0.7 [-1.9, +0.5] | 7 / 3 | 0.344 | safe | +0.0025 [-0.0004, +0.0055] | 0.0103 | 98.4 |
| 4 (s4) | B | 292 | +0.0 [-4.0, +4.0] | 18 / 18 | 1.000 | safe | +0.0025 [-0.0004, +0.0055] | 0.0103 | 98.4 |
| 4 (s4b) | B | 292 | +0.0 [-4.0, +4.0] | 18 / 18 | 1.000 | safe | +0.0025 [-0.0004, +0.0055] | 0.0103 | 98.4 |
| 2 (s2) | A | 598 | -0.8 [-2.6, +0.9] | 15 / 10 | 0.424 | safe | +0.0091 [+0.0038, +0.0147] | 0.0216 | 97.9 |
| 2 (s2) | B | 292 | -3.4 [-8.2, +1.3] | 30 / 20 | 0.203 | open | +0.0091 [+0.0038, +0.0147] | 0.0216 | 97.9 |
| 1 (s1) | A | 597 | -3.4 [-5.5, -1.4] | 29 / 9 | 0.002 | damage | +0.0163 [+0.0091, +0.0240] | 0.0344 | 97.2 |
| 1 (s1) | B | 292 | -8.6 [-14.2, -2.9] | 49 / 24 | 0.005 | damage | +0.0163 [+0.0091, +0.0240] | 0.0344 | 97.2 |
Spearman, family A: loss vs dNLL +1.00, loss vs KL +1.00 (6 stages)
Spearman, family B: loss vs dNLL +0.60, loss vs KL +0.60 (6 stages)

### Δ-NLL of the benchmark tails (pydash + boltons pooled), same tasks

| full layers | family | n | loss pp [95 %] | b / c | McNemar p | zone | dNLL mean [95 %] | KL | top-1 % |
|---|---|---|---|---|---|---|---|---|---|
| 8 (s8) | A | 598 | +0.2 [-0.9, +1.3] | 4 / 5 | 1.000 | safe | +0.0171 [-0.0051, +0.0429] | 0.0435 | 96.9 |
| 8 (s8) | B | 292 | -1.4 [-5.3, +2.6] | 19 / 15 | 0.608 | open | +0.0171 [-0.0051, +0.0429] | 0.0435 | 96.9 |
| 6 (s6) | A | 597 | +0.2 [-0.9, +1.3] | 4 / 5 | 1.000 | safe | +0.0209 [-0.0039, +0.0486] | 0.0533 | 96.8 |
| 6 (s6b) | A | 598 | +0.2 [-0.9, +1.3] | 4 / 5 | 1.000 | safe | +0.0209 [-0.0039, +0.0486] | 0.0533 | 96.8 |
| 6 (s6) | B | 292 | +0.7 [-3.6, +4.9] | 19 / 21 | 0.875 | safe | +0.0209 [-0.0039, +0.0486] | 0.0533 | 96.8 |
| 6 (s6b) | B | 292 | +0.3 [-4.0, +4.6] | 20 / 21 | 1.000 | safe | +0.0209 [-0.0039, +0.0486] | 0.0533 | 96.8 |
| 5 (s5) | A | 598 | +0.2 [-1.0, +1.4] | 5 / 6 | 1.000 | safe | +0.0078 [-0.0081, +0.0263] | 0.0536 | 96.5 |
| 5 (s5) | B | 292 | +1.0 [-3.6, +5.6] | 22 / 25 | 0.771 | safe | +0.0078 [-0.0081, +0.0263] | 0.0536 | 96.5 |
| 4 (s4) | A | 598 | -0.7 [-1.9, +0.5] | 7 / 3 | 0.344 | safe | +0.0279 [+0.0015, +0.0579] | 0.0534 | 96.8 |
| 4 (s4b) | A | 598 | -0.7 [-1.9, +0.5] | 7 / 3 | 0.344 | safe | +0.0279 [+0.0015, +0.0579] | 0.0534 | 96.8 |
| 4 (s4) | B | 292 | +0.0 [-4.0, +4.0] | 18 / 18 | 1.000 | safe | +0.0279 [+0.0015, +0.0579] | 0.0534 | 96.8 |
| 4 (s4b) | B | 292 | +0.0 [-4.0, +4.0] | 18 / 18 | 1.000 | safe | +0.0279 [+0.0015, +0.0579] | 0.0534 | 96.8 |
| 2 (s2) | A | 598 | -0.8 [-2.6, +0.9] | 15 / 10 | 0.424 | safe | +0.0216 [-0.0061, +0.0534] | 0.0884 | 95.4 |
| 2 (s2) | B | 292 | -3.4 [-8.2, +1.3] | 30 / 20 | 0.203 | open | +0.0216 [-0.0061, +0.0534] | 0.0884 | 95.4 |
| 1 (s1) | A | 597 | -3.4 [-5.5, -1.4] | 29 / 9 | 0.002 | damage | -0.0618 [-0.1201, -0.0127] | 0.1163 | 94.5 |
| 1 (s1) | B | 292 | -8.6 [-14.2, -2.9] | 49 / 24 | 0.005 | damage | -0.0618 [-0.1201, -0.0127] | 0.1163 | 94.5 |
Spearman, family A: loss vs dNLL -0.09, loss vs KL +0.94 (6 stages)
Spearman, family B: loss vs dNLL -0.14, loss vs KL +0.49 (6 stages)

### Per repo (own tasks against the own tail), pydash then boltons

| full layers | family | n | loss pp [95 %] | b / c | McNemar p | zone | dNLL mean [95 %] | KL | top-1 % |
|---|---|---|---|---|---|---|---|---|---|
| 8 (s8) | A | 299 | -0.3 [-2.1, +1.4] | 3 / 2 | 1.000 | safe | +0.0018 [-0.0006, +0.0042] | 0.0033 | 99.2 |
| 8 (s8) | B | 116 | +0.0 [-5.8, +5.8] | 6 / 6 | 1.000 | open | +0.0018 [-0.0006, +0.0042] | 0.0033 | 99.2 |
| 6 (s6) | A | 298 | +0.0 [-1.6, +1.6] | 2 / 2 | 1.000 | safe | +0.0058 [+0.0014, +0.0110] | 0.0074 | 99.2 |
| 6 (s6) | B | 116 | -1.7 [-8.0, +4.5] | 8 / 6 | 0.791 | open | +0.0058 [+0.0014, +0.0110] | 0.0074 | 99.2 |
| 5 (s5) | A | 299 | +1.3 [-0.4, +3.3] | 1 / 5 | 0.219 | safe | +0.0036 [-0.0005, +0.0077] | 0.0065 | 99.1 |
| 5 (s5) | B | 116 | -1.7 [-8.8, +5.4] | 10 / 8 | 0.815 | open | +0.0036 [-0.0005, +0.0077] | 0.0065 | 99.1 |
| 4 (s4) | A | 299 | -1.0 [-3.0, +0.9] | 5 / 2 | 0.453 | open | +0.0122 [+0.0045, +0.0217] | 0.0138 | 99.1 |
| 4 (s4) | B | 116 | +0.0 [-5.8, +5.8] | 6 / 6 | 1.000 | open | +0.0122 [+0.0045, +0.0217] | 0.0138 | 99.1 |
| 2 (s2) | A | 299 | +0.3 [-2.2, +2.9] | 6 / 7 | 1.000 | safe | +0.0220 [+0.0116, +0.0339] | 0.0191 | 98.5 |
| 2 (s2) | B | 116 | -8.6 [-15.5, -1.6] | 14 / 4 | 0.031 | damage | +0.0220 [+0.0116, +0.0339] | 0.0191 | 98.5 |
| 1 (s1) | A | 298 | -2.7 [-5.7, +0.1] | 13 / 5 | 0.096 | open | +0.0264 [+0.0154, +0.0377] | 0.0291 | 97.9 |
| 1 (s1) | B | 116 | -19.8 [-28.9, -10.3] | 29 / 6 | 0.000 | damage | +0.0264 [+0.0154, +0.0377] | 0.0291 | 97.9 |
Spearman, family A: loss vs dNLL +0.43, loss vs KL +0.43 (6 stages)
Spearman, family B: loss vs dNLL +0.77, loss vs KL +0.77 (6 stages)

| full layers | family | n | loss pp [95 %] | b / c | McNemar p | zone | dNLL mean [95 %] | KL | top-1 % |
|---|---|---|---|---|---|---|---|---|---|
| 8 (s8) | A | 299 | +0.7 [-1.0, +2.6] | 1 / 3 | 0.625 | safe | +0.0324 [-0.0138, +0.0832] | 0.0836 | 94.6 |
| 8 (s8) | B | 176 | -2.3 [-7.6, +3.0] | 13 / 9 | 0.523 | open | +0.0324 [-0.0138, +0.0832] | 0.0836 | 94.6 |
| 6 (s6) | A | 299 | +0.3 [-1.5, +2.3] | 2 / 3 | 1.000 | safe | +0.0360 [-0.0124, +0.0873] | 0.0992 | 94.5 |
| 6 (s6) | B | 176 | +2.3 [-3.4, +8.0] | 11 / 15 | 0.557 | safe | +0.0360 [-0.0124, +0.0873] | 0.0992 | 94.5 |
| 5 (s5) | A | 299 | -1.0 [-3.0, +0.7] | 4 / 1 | 0.375 | open | +0.0119 [-0.0185, +0.0483] | 0.1007 | 94.0 |
| 5 (s5) | B | 176 | +2.8 [-3.2, +8.9] | 12 / 17 | 0.458 | safe | +0.0119 [-0.0185, +0.0483] | 0.1007 | 94.0 |
| 4 (s4) | A | 299 | -0.3 [-2.0, +1.2] | 2 / 1 | 1.000 | safe | +0.0436 [-0.0071, +0.0988] | 0.0930 | 94.4 |
| 4 (s4) | B | 176 | +0.0 [-5.5, +5.5] | 12 / 12 | 1.000 | open | +0.0436 [-0.0071, +0.0988] | 0.0930 | 94.4 |
| 2 (s2) | A | 299 | -2.0 [-4.7, +0.4] | 9 / 3 | 0.146 | open | +0.0212 [-0.0344, +0.0831] | 0.1577 | 92.3 |
| 2 (s2) | B | 176 | +0.0 [-6.3, +6.3] | 16 / 16 | 1.000 | open | +0.0212 [-0.0344, +0.0831] | 0.1577 | 92.3 |
| 1 (s1) | A | 299 | -4.0 [-7.3, -1.1] | 16 / 4 | 0.012 | damage | -0.1499 [-0.2563, -0.0553] | 0.2036 | 91.2 |
| 1 (s1) | B | 176 | -1.1 [-8.0, +5.8] | 20 / 18 | 0.871 | open | -0.1499 [-0.2563, -0.0553] | 0.2036 | 91.2 |
Spearman, family A: loss vs dNLL -0.71, loss vs KL +0.94 (6 stages)
Spearman, family B: loss vs dNLL -0.14, loss vs KL -0.09 (6 stages)

(per repo and family 6 stages x 2 families x 2 repos = 24 tests: one isolated *damage* hit such as pydash B at two layers
(p = 0.031) is weak evidence on its own.)

## Reading

1. **First significant loss: one full layer.** On the pooled 32K axis it sits at Δ-NLL +0.0163 [+0.0091, +0.0240], 1.6
   times the limit of 0.01; the stage below it (two layers, +0.0091 [+0.0038, +0.0147]) shows no significant loss. The curve
   therefore only resolves "the onset lies between Δ-NLL 0.009 and 0.016 (32K texts)". 0.01 lies inside that gap, so the
   rule's limit and the onset of measurable damage are the same order, and nothing indicates that 0.01 is lax: all four
   stages up to 0.0025 (8 to 4 layers) lose no accuracy that the benchmark can see (|difference| <= 1.4 pp, intervals
   +-1.3 pp on A and +-4.5 pp on B).
2. **Is 0.01 too strict?** The rule rejects the two-layer stage (interval excludes 0, upper bound 0.0147). That stage costs
   at most 2.6 pp (A) / 8.2 pp (B) at the 95 % bound and the point estimates are -0.8 / -3.4 pp. Whether that is acceptable
   is a decision; the benchmark cannot show a loss there but cannot exclude one of that size either (B resolves about
   7 pp). The Spearman correlation over the six stages is +1.00 for A and +0.60 for B (loss against Δ-NLL), identical to
   KL since both order the stages alike: the order of the stages is monotone, a single stage cannot separate the axes.
3. **Repo code (the benchmark tails).** For pydash the same stages have up to 5 times the Δ-NLL of the 32K texts (four layers
   +0.0122 [+0.0045, +0.0217]: above 0.01, no accuracy loss on pydash tasks: A -1.0 pp [-3.0, +0.9], B 0.0 pp [-5.8, +5.8]).
   If the pydash tail is the yardstick, the limit of 0.01 would reject a stage (4) whose loss we cannot see; but B on
   116 tasks resolves only +-5.8 pp, so "too strict on repo code" is **not shown, only not excluded**. The same axis at
   two layers (+0.022, B -8.6 pp [-15.5, -1.6]) and one layer (+0.026, B -19.8 pp [-28.9, -10.3]) shows the loss on pydash B
   at 2.2 and 2.6 times the limit. For boltons the tail-Δ-NLL is unusable (see above), A loses significantly only at one
   layer (-4.0 pp [-7.3, -1.1]), B not at all (-1.1 pp [-8.0, +5.8]); boltons B has 176 tasks and a lower reference
   accuracy (22 %).
4. **Axes that disagree.** The loss at one layer is the largest of all stages on every task set, and the 32K texts and the
   pydash tail order the stages almost the same way (5 and 6 layers swap); the pooled benchmark tail does not (Spearman -0.09 for A against Δ-NLL)
   because of boltons. KL orders the boltons tail correctly (+0.94 for A). So for repo code KL is the steadier axis
   in this data; for a rule on Δ-NLL a larger tail than 4,096 tokens per context would be needed.
5. **Not enough power for the question "does 0.01 predict a 1 pp loss".** The benchmark resolves 3 pp on A and about 7 pp on
   B, and the curve has two stages between "no effect" and "clear effect". A statement stronger than "the limit and the
   onset of detectable damage are of the same order, damage is first visible at 1.6 times the limit on the 32K texts and at
   about 2 times on the pydash tail" is not supported by these numbers.

## Selection rule (secondary analysis), and why it is not usable here

Rule of part 1 (selection run ref2 solved, closed-book did not solve for A): 563 of 892 tasks kept (A 482, B 81). Paired
against ref1, the 8-layer stage already shows B -12.3 pp [-20.9, -4.4] (p = 0.006) and A -0.4 pp, with a Δ-NLL of +0.001. That
is the selection, not the degradation: ref1 and ref2 differ only in the pass width, 630 of 890 replies are identical, so
a task chosen because ref2 solved it is mostly also solved by ref1 and less often by any run with a different numeric
path. The selection run is not independent of the reference run it is paired with. The secondary analysis is therefore
dropped (tables in `/var/tmp/claude/w76/analysis/curve-*-sel.md`); an independent selection would need a reference that
computes differently from every other run, which we do not have. The primary analysis (all tasks) stands.

## Observations

- `work/build/engine` of the part-1 pilot was stale (no `IE_IDX_SHARE`); this run only uses `work/build/w76`.
- Barrier aborts, each a single request answered with HTTP 500, process kept running; each failed request came right after
  a request that restored a conversation from RAM (every request after the first one per repo does, so this does not
  discriminate). Lines from `barriers.py`:
  - ref1 (no `IE_IDX_SHARE`) 22:55:46: `Pass failed: gpu::Mega: barrier aborted (time limit) at position 62366 (step 696: matvec moe.down, longest step 17 moe.gate_up 0.9 ms, pass up to there 51.3 ms, blocks at step 696, all equal)`, after `Request: 62364 prompt tokens (62318 reused, 46 computed in 0.6 s, conversation from RAM in 205 ms, slot 0)`.
  - ref1 23:22:11: `... at position 62228 (step 27: matvec gr.up, longest step 15 moe.shared 0.2 ms, pass up to there 0.9 ms ...)`, after `62219 prompt tokens (62182 reused, 37 computed in 0.6 s, conversation from RAM in 268 ms, slot 1)`.
  - s6 (`IE_IDX_SHARE=0,2,4,6,8,10`) 05:12:10: `... at position 62368 (step 98: dn.conv dn.conv, longest step 19 moe.down 0.4 ms, pass up to there 5.1 ms ...)`, after a restore from RAM (slot 0).
  - s1 (`IE_IDX_SHARE=0`) 00:35:19: `... at position 62369 (step 440: matvec moe.gate_up, longest step 17 moe.gate_up 0.7 ms, pass up to there 28.6 ms ...)`, after a restore from RAM (slot 1).
  Two of the four are in the unchanged path (ref1). All happened during the main run with other workers on the machine;
  I did not check what else used the GPU at those times.
- Qwen runs with thinking off in all of this; with thinking on the picture of B (reference accuracy, flips) may differ
  (the coordinator's own thinking runs are separate).

## Open points

- B power: 292 tasks resolve about 7 pp. For 5 pp at f = 0.17 to 0.25 about 540 to 800 tasks are needed, i.e. two to three
  more repositories of the size of boltons.
- Δ-NLL on repo code needs a larger tail (several 4,096-token windows per context) or a robust estimate; the boltons
  tail is not usable as is.
- Stages between 2 and 1 full layer do not exist on the 12-layer model, so the onset cannot be resolved finer in
  this dose axis; a different degradation (e.g. a smaller selection budget) would be needed for a finer curve.
