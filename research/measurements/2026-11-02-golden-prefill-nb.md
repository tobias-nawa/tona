# Golden prefill in passes of width 4: is width 4 bit-identical to width 1? (#96)

Date: 2026-11-02. Branch `96-golden-prefill-nb` (from main 481227c). Build `work/build/wnb` (this branch, no code
change yet in step 1).

## Plan

Question: does the verify pass of width 4 write the same state (KV, DeltaNet, convolution) and produce the same logits
as single steps (width 1) over a 13K-token prompt? If yes, the golden prefill check can run its prompt in passes of
width 4 and keep the existing reference `work/golden/prefill/`. If no: stop, report the first deviation, record nothing.

What the code says (no run yet). `--batch-check N` (`engine/tools/ie_gpu_gen_batch.cpp`) computes the logits of the
first N prompt positions from single steps (`IE_BATCH_REF_NB=1`), resets the megakernel, and runs passes of exactly
`IE_BATCH_NB` tokens, all accepted, comparing every position bit for bit (`memcmp` of the full logit vector). It first
runs a mixed-width comparison (widths 4, 3, 2, 1 in turn, partly rejected), which is a deviation measure, not the
decision. State is compared through the logits: any KV, DeltaNet or convolution difference at position p shows in the
logits of a later position unless it is masked exactly, so equal logits at all 13,328 positions is strong evidence
for equal state, and it is exactly the property the check needs (the 64 decode tokens that follow read that state).
Earlier evidence: 120/120 positions bit-equal for width 4 against 1 and against 2 after the DeltaNet canonical
computation (`2026-10-25-bit-exact.md`, "DeltaNet canonical"); the split-K attention and the hyper-connection norm use
BMAX instead of nb. Not covered by that evidence: depth. At 13K the IndexCache and top-k selection run at their
deepest, the attention covers many chunks and the DeltaNet state has seen thousands of commits.

Expectation: bit-identical at all 13,328 positions (`bitwise (passes of 4): 13328 of 13328 positions equal, first
deviation at position -1`). Source: the 120-position result above plus the width-independent attention and norm.
Probability I put on it: about 75 %; the open risk is a width-dependent sum in the indexer or top-k at depth. The
mixed-width line is allowed to show deviations (it did before: top-1 222/267) and does not decide. 13,328 is the
prompt length of the golden check, so the tokenization is that of `scripts/bench-long-code.txt` without chat template
(as golden.sh feeds it).

Scope: one run, `ie-gpu-gen <model> <prompt> --mega --ctx 16384 --batch-check 13328`, `IE_BATCH_NB=4`,
`IE_BATCH_REF_NB=1`. Work: reference 13,328 single steps (~338 s, 25.4 ms each, `2026-11-02-golden-loads.md`),
mixed-width pass (about 5,300 passes of width 4..1 at 37..23 ms, ~150 s), bitwise pass (3,332 passes of 37 ms, ~125 s),
plus 17 s load: expected about 10 to 11 minutes (tolerance 9 to 14). Memory: the reference holds the logits of all
positions (13,328 x vocab x 4 B, about 8 GB); the run goes under `systemd-run --scope` with
`MemoryMax = MemTotal - 6 GB`, no swap, as golden.sh does.

Why this size: the whole point is depth. A shorter prompt (the 120 positions of 2026-10-25) is already known to be
equal and would not show a defect that appears only with a deep context, which is what #87 showed at 13K. One run is
enough because the comparison is bitwise and `IE_DETERMINISTIC=1`: a single deviation decides, and equality over
13,328 positions needs no repetition.

Active tooling: `IE_DETERMINISTIC=1`, `IE_DIAG=1` (allocation trace and GPU dump, as always on this machine),
`IE_PLAN=gpu`, `IE_EXPERT_PROFILE=work/profile/experts-coder.bin`, `OMP_NUM_THREADS=6` (the same as golden.sh). No
profiler. GPU lock `/var/tmp/claude/gpu.lock` around the run; no build while it runs.

Step 3 (only if bit-identical): acceptance runs of golden.sh, one run each, old and new: big model `long` against
`work/golden/prefill/`, and `GOLDEN_MODEL=mini` against `work/golden/mini/prefill/`. Expectation: byte-identical; new
about 145 s (17 s load + ~125 s prompt + 2 s decode), tolerance 130 to 170 s; old 359 s.

## Results

### Step 1: `--batch-check 13328`, `IE_BATCH_NB=4`, `IE_BATCH_REF_NB=1` (one run, 07:00:36 to 07:13:02)

Load 20.1 s, 12 min 26 s in all (expectation 10 to 11 min, tolerance 9 to 14: inside).

| | result |
|---|---|
| mixed widths (4, 3, 2, 1 in turn, partly rejected) | 13,328 positions, 7,109 passes, 17,772 comparisons, top-1 equal in 17,772, worst relative deviation 0.00e+00 |
| bitwise, passes of exactly 4 | 13,328 of 13,328 positions equal, first deviation at position -1 |

Read against the plan: bit-identical as expected. The result is stronger than planned: the mixed-width pass, which
deviated before the DeltaNet canonical change (top-1 222/267), is now exact as well, at depth 13K. Width 4 and width 1
give bit-identical logits at every position of the 13K prompt, so the state they write is equivalent for everything
that follows. Next: step 3.


### Step 3: `ie-gpu-gen --prompt-nb N`, `golden.sh` with `--prompt-nb 4` (1 run each, deterministic)

`--prompt-nb N` (1 to 4, default 1 = unchanged) feeds the prompt of the plain path (no drafter) in passes of N tokens;
`golden.sh` passes 4 for `GOLDEN_MODE=prefill` without drafter; `GOLDEN_PROMPT_NB=1` restores single steps.

| Check | Prompt | Load | Prompt time | 64 decode tokens | Total | Against the reference |
|---|---|---:|---:|---:|---:|---|
| big, `long`, single steps (old) | 13,328 x 1 | 16.9 s | 337.3 s (25.3 ms/pass) | 1.76 s (36.3 tok/s decode) | 356 s | identical (`work/golden/prefill/`) |
| big, `long`, width 4 (new) | 3,332 x 4 | 17.8 s | 136.9 s (41.1 ms/pass) | 1.71 s (37.4 tok/s decode) | 156 s | identical (`work/golden/prefill/`) |

Against the expectation (145 s, tolerance 130 to 170): 156 s, inside; the 13K pass of width 4 takes 41 ms, 4 ms more
than the 37 ms of the seed table (deeper context, more experts per pass). Wall time of the script was longer because
the lock was held by another worker for about 2 minutes before each run. The decode rate 37.4 against 36.3 tok/s decode
is inside the spread of these single runs and is not a claim.

Mini model (`GOLDEN_MODEL=mini`, `GOLDEN_MODE=prefill`): width 4 and width 1 give byte-identical output (`cmp`), but
**both differ from the recorded reference `work/golden/mini/prefill/long.txt`**. So does the mini decode golden
(`csv`, `rust`, `explain`, three of three) and so does every build tried (`work/build/merged`, `work/build/wgold`,
`work/build/wnb`), with and without `IE_DIAG`. The prefix of the text is equal, the text diverges later. The mini
references do not reproduce with the code of main before this change, so this is not an effect of #96; the
acceptance for the mini model is therefore the direct comparison width 4 against width 1 (equal), not the old reference.
The mini references need to be looked at separately (re-record is a decision for the maintainer).

Gates: `just check` green (pre-commit hooks and clang-tidy), `pytest tools/tests` 290 passed, `ctest -L unit` 37 passed.
