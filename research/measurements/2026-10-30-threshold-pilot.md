# Threshold benchmark: build and pilot (#76, part 1)

Question: does the quality rule (95 % interval of Δ(-log p) contains 0 and lies below 0.01 nats, `docs/backends/README.md#checks`)
predict task accuracy? This part builds the benchmark (`bench/threshold/`, README there) and measures in a pilot how
often tasks flip (f), from which the number of tasks follows. Plan: `research/31-nll-threshold-validation.md`.
Raw data (contexts, tasks, runs, logs): `/var/tmp/claude/w76/`, not in the repository.

## Build

- Engine: `work/build/w76`, built from main at 32f79f2 (includes `IE_IDX_SHARE`, #73). All pilot runs come from this
  build. Server on port 8092, `--ctx 65536`, MTP, greedy, **thinking off** (`enable_thinking=false`), under the GPU lock.
  A first set of runs with `work/build/engine` was discarded: that binary (2026-10-29 02:39) predates #73 and has no
  `IE_IDX_SHARE`; its "variants" gave 360 of 360 identical replies and identical decode time (14.18 vs 14.19 ms/token).
  `bench/threshold/check_differs.py` now aborts when a variant is identical to the reference.
- Context: one per repo, system message; question in the last user message. Sources are read with `ast.parse` only; tests
  run in a rootless Podman container (no network, read-only root, tmpfs `/tmp`, user nobody, all capabilities dropped, 2 GB,
  2 CPUs, time limit), see the README.

| Repo | Licence | Commit | Context | Holes (family B) | Questions (family A) |
|---|---|---|---|---|---|
| pydash | MIT | 812eb13 (2026-10-28) | 62,954 tokens, 4 files | 30 of 79 qualifying functions | 150 (13 local), 391 generated |
| boltons | BSD-3-Clause | d31d821 (2026-10-25) | 62,538 tokens, 8 files | 30 of 123 qualifying functions | 150 (13 local), 319 generated |

Both are well-known public packages, probably in the training data; the closed-book run below measures how much that
helps (A: 11.3 %). A function qualifies when 3 to 35 body lines are at least 80 % covered and the suite fails with the
body replaced by `raise NotImplementedError`. Self-test: the original function text passes in the sandbox for all 60
tasks. The prompt for B asks for the whole definition; the body is extracted and put into the file.

**Prompt cache works at 64K** (pydash, 5 requests): first request 63,054 tokens computed in 37.1 s (prefill, about 1,700
tok/s); every following request: `cache_n` 63,003, 43 to 71 tokens computed, restored from RAM in 25 to 310 ms, 0.4 to
0.7 s for the question prefill. The context sits in the system message because the server's checkpoint is at the start of
the last message.

## Pilot runs (360 tasks per run: 300 A, 60 B)

Setups: `ref1` main, MTP (default); `ref2` same with `IE_DRAFT_N=1` (the server has no mode without a drafter; this
changes the pass widths and so the rounding, which is what f0 measures); `closed` ref1 without the repository (A only);
`d8` `IE_IDX_SHARE=0,1,3,4,6,7,9,10` (pattern d, passes the NLL rule); `l0` `IE_IDX_SHARE=0`. One run each.

Accuracy of the reference: A 95.7 % (far 261/274, local controls 26/26), B 25/60 = 41.7 %. In 22 of 60 B replies of ref1
the 700-token limit cut the answer (counted as wrong): the cap probably costs B accuracy and is a parameter to review
before the main run. Closed-book A: 34/300 = 11.3 % (doc 23, default 8, caller 2, constant 1, base/returns/local 0).

Paired against ref1; b = right in ref1 and wrong in the run, c = the opposite, f = (b + c) / n; exact McNemar; 95 %
Newcombe interval of the accuracy difference (paired); seconds = median per task.

| run | family | n | acc % | b | c | f | McNemar p | difference pp [95 %] | s/task |
|---|---|---|---|---|---|---|---|---|---|
| ref2 (f0) | A | 299 | 95.7 | 0 | 0 | 0.000 | 1.00 | +0.0 [-1.2, +1.2] | 0.8 |
| ref2 (f0) | B | 60 | 38.3 | 5 | 3 | 0.133 | 0.73 | -3.3 [-12.3, +5.7] | 10.5 |
| d8 | A | 298 | 95.6 | 2 | 2 | 0.013 | 1.00 | +0.0 [-1.7, +1.7] | 0.8 |
| d8 | B | 60 | 35.0 | 10 | 6 | 0.267 | 0.45 | -6.7 [-19.1, +6.2] | 7.7 |
| l0 | A | 300 | 91.3 | 17 | 4 | 0.070 | 0.007 | -4.3 [-7.6, -1.4] | 0.8 |
| l0 | B | 60 | 21.7 | 16 | 4 | 0.333 | 0.012 | -20.0 [-33.0, -6.0] | 8.4 |

- The benchmark reacts: l0 loses 4.3 pp on A and 20 pp on B, both significant. Of the 17 A questions lost, 2 are local
  controls (should not change), 15 are long-range.
- Pattern d8 (rule-passing, Δ-NLL within ±0.008) shows no loss on A (0.0 pp) and a non-significant loss on B
  (-6.7 pp, interval -19 to +6); the pilot is too small to say whether B moves. d8 has 299 of 358 identical replies to
  ref1, so the switch acts, but most of those replies are short.
- Rounding noise f0: A 0.000 (299 pairs, 0 flips), B 0.133 (8 of 60; 50 of 360 reply texts differ). A is nearly
  deterministic against pass-width rounding; B is not.
- Time per task (wall, median): A 0.8 s, B 7.7 to 10.5 s (mean 490 generated tokens). A run of 300 A + 60 B takes about
  15 min including the two 37 s prefills and the 20 s load.

### Selection rule applied (fixed in the README before the variant runs)

Selection run ref2 (solved) and closed-book (not solved): 277 of 360 kept (A 254, B 23). Compared against ref1:

| run | family | n | acc % | b | c | f | McNemar p | difference pp [95 %] |
|---|---|---|---|---|---|---|---|---|
| d8 | A | 252 | 99.2 | 2 | 0 | 0.008 | 0.50 | -0.8 [-2.8, +0.8] |
| d8 | B | 23 | 60.9 | 7 | 1 | 0.348 | 0.07 | -26.1 [-46.1, -3.5] |
| l0 | A | 254 | 93.3 | 17 | 0 | 0.067 | <0.001 | -6.7 [-10.5, -3.8] |
| l0 | B | 23 | 43.5 | 11 | 1 | 0.522 | 0.006 | -43.5 [-62.6, -17.2] |

Caution: the B selection keeps only 23 of 60 tasks, and d8/l0 share the MTP setting with ref1 but not with ref2 (the
selection run), so differences in rounding noise between the settings may leak into "lost" counts. The unselected table
above is the unbiased one. With selection the A ceiling is 100 % by construction for ref2 and the interval gets
narrower, B gets wide: the selection helps A (removes tasks nobody can lose) and hurts B (too few tasks left).

## Number of tasks and GPU time

Smallest detectable difference d = 2.8 sqrt(f / n) at 80 % power, alpha 0.05, so n = 7.84 f / d² (f = flip rate of the
stage measured against ref1; f for 12/8/6/5/4/2/1 full layers lies between the d8 and the l0 value: the mild stage
decides how many tasks are needed).

| Target | f used | n |
|---|---|---|
| A, 3 pp | 0.013 (d8) | 114 |
| A, 3 pp | 0.070 (l0) | 610 |
| B, 5 pp | 0.133 (f0, pure noise) | 418 |
| B, 5 pp | 0.267 (d8) | 838 |
| B, 5 pp | 0.333 (l0) | 1,045 |

For the full curve A needs about 600 questions per stage; the two repos give 710 questions in total, so about 600 are
possible but without reserve for a separate selection run. **B is the bottleneck**: 5 pp needs 840 to 1,050 tasks, the
two repos give at most 79 + 123 qualifying functions (about 200 distinct tasks; each hole set needs its own 37 s
prefill, 30 holes per context as now). With 200 tasks the resolution is d = 10 pp at f = 0.27. For 5 pp on B about
5 times as many repositories (8 to 10) or all qualifying functions of more repos are needed; or accept 10 pp on B and
use A (3 pp) as the sharp instrument. Alternatively larger accuracy headroom: B reference accuracy is 42 % with a
700-token cap that cut 37 % of the replies.

GPU time [estimate from the measured times]: per stage A 600 questions x 1.07 s = 11 min, B 200 tasks x 7.7 s = 26 min,
7 contexts x 37 s prefill = 4 min, load 1 min: about 42 min. 7 stages 4.9 h; with ref2 (f0) and three repeats of the
pivotal stages (6 and 4 full layers) 5 more runs, about 8.5 h in total.

## Open points and observations

- No mode without a drafter in `ie-server` (needs `--mtp` or `--dflash`); ref2 uses `IE_DRAFT_N=1`.
- The scorer was fixed twice during the pilot (applies to all runs, which were scored again): indented code blocks
  (method pasted with its class indentation, found by the coordinator) and generic base classes (`Generic[T]` accepted as
  `Generic`).
- Replies cut by the 700-token cap count as wrong (22 of 60 in ref1).
- Barrier aborts (unchanged path or not, for #75), each a single failed request (HTTP 500, the task is missing from the
  run: pydash-A015 in ref2, boltons-A019 and boltons-A041 in d8), the process went on:
  - ref2 (`IE_DRAFT_N=1`, no `IE_IDX_SHARE`), 2026-10-29 21:29:27: `Pass failed: gpu::Mega: barrier aborted (time limit)
    at position 63052 (step 628: matvec moe.gate_up, longest step 17 moe.gate_up 0.5 ms, pass up to there 24.3 ms)`.
  - d8, 21:59:22: `... at position 62634 (step 19: matvec moe.down, longest step 17 moe.gate_up 0.7 ms, pass up to there 2.1
    ms)`; 21:59:41: `... at position 62633 (step 378: matvec moe.gate_up, longest step 368 dn.norm 61.9 ms, pass up to there 93.1
    ms)`.
  The discarded first set (old binary, `IE_IDX_SHARE=0` ignored) also had two, 2026-10-29 17:14:13 (position 63047,
  step 433 matvec qsa.in) and 17:14:47 (position 63057, step 990 matvec moe.down), in an unchanged path.
