# Mini model for fast pre-checks (#91)

Question: can the engine load and run a model of the reference architecture with random weights, few layers and
few experts, without any change to the engine, and what does it save over the 55 GB model for the checks that
need no speed and no quality?

## Plan

Written before the first GPU run.

**Scope.** Generator `tools/make_mini_model.py` writes `work/models/mini/mini.gguf` (target) and `mini-mtp.gguf`
(MTP head) from the headers of the real files. Then, in this order, each step only if the one before is green:

1. CPU only: the loader accepts the files (`ie-model`), the CPU reference runs (`ie-ref`), logits are finite.
2. First GPU run, once: `ie-gpu-gen --mega -n 32 --check 4`, `IE_DETERMINISTIC=1`, `timeout 120`, under
   `/var/tmp/claude/gpu.lock`. The shapes are untested on the kernels; a hang or GPU fault ends the series.
3. The same run twice (tokens equal), the server with the MTP head on both APIs, `golden.sh` and
   `golden-server.sh` with `GOLDEN_MODEL=mini` (one recording, three compare runs), the slot pressure runs.

Duration: steps 1 and 2 seconds to minutes, step 3 minutes. Rounds: one for correctness checks (the result is
identical or not), three compare runs of the golden checks for their time.

**Expectation.**

- Size 150 to 200 MB, load in 2 to 5 s (estimate of the ticket; the file is 182 MB).
- Time of the five golden checks: about a minute against 601 s for the four comparable checks on the big model (medians of
  `research/measurements/2026-11-02-select-checks.md`; tolerance: anything under 5 minutes counts as met).
- `--check`: GPU and CPU reference agree on the top-1 token for most steps. Random weights have flat logits, so a
  flip at a near tie is possible and is judged by the logit deviation, not counted as a failure alone.
- Accepted drafts of the random MTP head: near chance, so the accept path is not covered by this model.

**Active tooling.** `IE_DETERMINISTIC=1`; no profiler, no `IE_DIAG`, no expert profile (the profile of the big model
does not match). Build: `work/build/engine` of main, unchanged. Scale of the weights: gains in
`tools/mini_model/tensors.py`.

## Results

Build: `work/build/engine` (main, no engine change in this branch). Model: `work/models/mini/`, seed 1; the files of two
generator runs have the same SHA-256 (mini.gguf `eb6c44a3...`, mini-mtp.gguf `dc066e6b...`). Raw logs:
`/var/tmp/claude/wmini/run/`, not in the repository. All GPU steps under `/var/tmp/claude/gpu.lock`, 60 s between
blocks; the lock was held by other workers for up to 20 minutes before a block, so waiting times are not in the table.

**Sizes.** Target 153.4 MB, MTP head 30.8 MB (expectation 150 to 200 MB: met). Generator 3.8 s (pure Python). Load
0.1 s (`ie-gpu-gen`), 0.2 s (server with MTP head; expectation 2 to 5 s: faster).

**Step 1 (CPU).** `ie-model` binds all tensors and checks all shapes; `ie-ref` runs 8 steps in 0.4 s with finite
logits (top-1 logit 33 to 38, gap to the second 0.5 to 2). The scale of the weights was set once (gains in
`tools/mini_model/tensors.py`), no iteration was needed.

**Step 2 (first GPU run).** `ie-gpu-gen --mega -n 32 --check 4`, `IE_DETERMINISTIC=1`, `timeout 120`: no hang, no
fault, no abort of the barrier, no engine change and no fallback of the table (d 512, hc_rank 128) needed. The
shapes of the plan fit the kernels as they are.

| Check | Result |
|---|---|
| `--check 4` against the CPU reference | top-1 equal in 4 of 4; worst relative deviation of the logits 2.75e-2 (steps: 2.73, 2.75, 2.24, 2.26e-2) |
| 3 identical runs of 32 tokens | tokens identical (cmp) |
| decode rate | about 370 tok/s decode (host-bound, a pass is a few ms of launch overhead; no meaning for speed) |

The deviation of 2.7e-2 is large against the reference model. It is not read as a finding about the kernels: logits
of a random model have a flat spectrum and 8 wide, and the comparison is int8 activations against a float CPU
reference. It is also why the mini model never judges quality.

**Golden checks with `GOLDEN_MODEL=mini`** (one recording, then compare; all identical):

| Check | Mini | Big model (median, `research/measurements/2026-11-02-select-checks.md`) |
|---|---|---|
| `golden.sh` decode, plan gpu (3 prompts x 200 tokens) | record 2.6 s, compare 2.60 / 2.59 / 2.58 s | 67 s |
| `golden.sh` plan cpu | record 2 s, compare 3 s | 65 s |
| `golden.sh` prefill (13K tokens, no drafter) | record 25 s, compare 24 s | 362 s |
| `golden.sh` prefill with dflash | not possible (no mini drafter) | 30 s |
| `golden-server.sh` (20 requests, both APIs, MTP head, stash, SSD checkpoints) | record 19 s, compare 18 s | 107 s |
| Sum of the four comparable checks | about 48 s | 601 s (all five: 631 s) |

The compare of `golden-server.sh` with `IE_SEQS=1` (reuse case, one slot, so every other conversation stashes the
running one: "Conversation of 2139 tokens stashed: 16 MB") is identical to the reference of two slots. The ssd case
writes one checkpoint and the restarted server finds it ("1 matching").

**Expert misses and CPU experts** (`ie-gpu-gen --mega -n 200`, deterministic; reference: 64 slots, 99.7 % hits, 18 MB
reloaded):

| Setting | Hits | Reloaded | Tokens |
|---|---|---|---|
| `--slots 16` | 80.9 % | 726 MB | identical to 64 slots |
| `--slots 10` (minimum) | 70.8 % | 639 MB | identical |
| `--slots 10`, `IE_PLAN=cpu` (CPU experts, 64 per layer pinned) | 70.4 % to 70.8 % | 160 to 639 MB | identical |
| `IE_VRAM_GB=0.5`, no `--slots` | 94.0 % | 52 MB | 42 slots |
| server `IE_VRAM_GB=0.79`, `--ctx 32768`, MTP | 18 slots per layer | | all 20 cases of `golden-server.sh` identical to the unpressured reference |

The server's limit is the VRAM in use plus slots x 0.5 MB plus about 3 MB: in use is 0.78 GB at `--ctx 32768` with
the MTP head (the error message of a too small limit names it), so 0.79 GB gives 18 slots. `IE_VRAM_LOG=1` prints the
numbers.

**Against the expectation.** Everything as expected or better, with two things to note: the flat logits (top-1 flips
at near ties are possible in other runs; none seen in 4 + 3 x 200 greedy steps), and that "reuse" needs `IE_SEQS=1`
to stash at all, because two slots hold both conversations.

**Not covered** (see docs/contribute/testing.md): speed, quality, DFlash, accepted drafts of the MTP head (acceptance is
chance: 1.00 tokens per pass in all server logs), long contexts, the shapes of the reference.
