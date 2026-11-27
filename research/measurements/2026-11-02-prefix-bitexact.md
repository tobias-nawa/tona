# Prefix-cache hits equal a cold prefill (#81): invariance of the prefill against its cuts

Date 2026-11-02. Branch `81-prefix-bitexact` (from main 98dc089), build `work/build/w81`, all runs under
`/var/tmp/claude/gpu.lock`.

## 1. Plan: is the prefill independent of where it is cut?

**Question.** The chunk plan (`PrefillRun::choose_plan`) depends on n and on the start position. Reading the kernels
(dense and expert GEMM, DeltaNet, pool/indexer, attention, MoE sum) finds no dependence on chunk boundaries. If that
holds, a prompt prefilled in one call and in pieces gives the same state, and #81 reduces to the one token that a
checkpoint computes through `forward` (`Generator::run_prompt`).

**What runs.** `ie-gpu-gen --mega --teacher TEXT` with `IE_TEACHER_TAIL=64`: the head goes through the prefill, the
last 64 tokens run teacher-forced through the megakernel, `IE_TEACHER_DUMP` writes -log p and the logits row of every
tail token. The reference run prefills the head in one call; the variants cut it with `IE_TEACHER_CUTS` (new, pure
prefill pieces, no forward token). The dumps are compared with `cmp`: any bit of difference in the state of the
head shows in the tail logits.

| Model | Head | Variants (cuts) | Why this size |
|---|---:|---|---|
| mini | 20,000 | none (reference), 4096, 4097 + 9001, 6144, 6143 + 13000 + 13001, 1 + 19999 | 20K crosses two layered groups of 6144 when the mini plan is layered (`IE_PF_DEBUG=1` shows the plan); odd cuts shift every chunk edge; the cuts at 6143/6144 sit on the group edge |
| mini | 20,000 | the same, `IE_PF_LAYERED=0` | the chunk-by-chunk plan |
| big | ~8,200 | one cut at ~8,000 (odd) | the big model uses other kernel paths by size; one run answers whether the mini result carries over; only if mini is bit-equal |

**Expectation.** All dumps byte-equal to the reference (`cmp` silent), under `IE_DETERMINISTIC=1`. Source: the kernel
reading in the report of step 1. A difference is a finding (a cut-dependent kernel), not noise: each run is
deterministic (same cuts twice give the same bytes; checked once per model with a repeat of one variant).
Expected duration: mini seconds per run, big about 2 min.

**Active tooling.** `IE_DETERMINISTIC=1`, `IE_DIAG=1`, `IE_PF_DEBUG=1` (prints the plan per prefill call).

## 2. Result: the prefill does not depend on its cuts

Build `work/build/w81` (main 98dc089 plus `IE_TEACHER_CUTS` in `engine/tools/ie_gpu_gen_tail.cpp`, which touches no
engine code), text the first 150 kB of `work/corpus/code.txt`; `IE_DETERMINISTIC=1`,
`IE_DIAG=1`, `IE_PF_DEBUG=1`. The tail is 64 teacher-forced tokens through the megakernel; `cmp` compares the dump of
-log p and the full logits row of every tail token (about 63 MB for the mini model). Raw logs and dumps:
`/var/tmp/claude/w81/inv/`.

**Mini model, head 20,000 tokens** (the plan was chunk by chunk, chunk 2,048: the mini model's loan limit is 16 MB, so
it never runs layer by layer; the layered path is covered by the big model below). Reference twice (`ref`, `ref2`):
identical. Variants, all byte-equal to the reference (`cmp` silent, NLL 34.666086 in every log):

| Cuts | Prefill calls (tokens) | Equal |
|---|---|---|
| 4096 | 4096 + 15904 | yes |
| 4097, 9001 | 4097 + 4904 + 10999 | yes |
| 6144 | 6144 + 13856 | yes |
| 6143, 13000, 13001 | 6143 + 6857 + 1 + 6999 | yes |
| 1, 19999 | 1 + 19998 + 1 | yes |
| 8191 | 8191 + 11809 | yes |
| `IE_PF_LAYERED=0` (on the mini model the plan is the same as by default): none against 4097, 9001 | as above | yes |

**Big model (Coder IQ1_M, MTP drafter not loaded), head 8,200 tokens**, plan of the reference: layer by layer, chunk
6,144, group 12,288 (so the chunk edge at 6,144 lies inside). Reference twice: identical (NLL 0.075354, top-1 98.44 %).

| Cuts | Prefill calls (tokens, plan) | Equal |
|---|---|---|
| 8000 | 8000 layered + 200 chunk by chunk (2,048) | yes |
| 4001 | 4001 layered + 4199 layered | yes |
| 4001, 6500 | 4001 + 2499 layered + 1700 chunk by chunk | yes |

So a layered piece, a piece of the chunk-by-chunk plan and the split of the group give the same bits: the chunk plan of
`PrefillRun::choose_plan` is not a source of difference under `IE_DETERMINISTIC=1`. What differs between a checkpoint
hit and a cold run is the token that `Generator::run_prompt` computes through forward at the checkpoint
(`engine/server/prefill.cpp`): the next sections remove it.

**Big model, head 14,000 tokens** (reference: layer by layer, chunk 6,144, group 18,432; NLL 1.117333, top-1 78.12 %):

| Cuts | Prefill calls | Equal |
|---|---|---|
| 12289 | 12289 layered + 1711 chunk by chunk | yes |
| 6000, 12000 | 6000 + 6000 layered (group 12,288) + 2000 chunk by chunk | yes |

All eight comparisons over the two models are byte-equal. Not repeated for the 14K reference (one run each; the
8,200-token reference was run twice and is identical).

## 3. Plan: what the change costs and which references move

**The change** (commit "Checkpoints inside a prompt are prefill-only"): the stretch up to a checkpoint inside a prompt
(fork, start of the last message, head of a warm-up, slice boundary of an interrupted warm-up) goes entirely through
the prefill and is stored without logits; the end of a prompt keeps its forward token and its logits (exact repeat).
Behind a checkpoint a rest of at most `IE_PREFILL_MIN` (32) tokens now also goes through the prefill, as a cold run of
the whole prompt does. Tokens change only where a checkpoint is written or read.

**Quality (rule 3, `docs/backends/README.md#checks`).** The new state of a checkpoint equals the unsplit prefill byte for byte
(section 2), and the unsplit prefill is the reference of the quality checks. Δ(-log p), KL and top-1 against that
reference are therefore 0 by construction, not an estimate; the proof is `cmp` on the full logits of the tail. The
numbers that move are those of main's checkpoints (forward token at the cut: Δ +0.0016 to +0.0025 nats, #79).
Expectation: `IE_TEACHER_REF` of one cut run against the unsplit dump prints dNLL +0.000000, KL 0.000000, top-1 equal
100 %. Run: big model, head 14,000, cuts 12289 against the unsplit reference of section 2.

**Which references move (prediction).** Only requests that write or read a checkpoint:
- `GOLDEN_MODEL=mini scripts/golden-server.sh compare`: `ssd-1` and `ssd-2` (a checkpoint at the start of the last
  message, 4,096 tokens or more) differ from the recorded ones; `base`, `sampled`, `thinking` and `reuse` (prompts
  under 4,096 tokens, no checkpoint) are identical. New files `hit-*` and `cold-*` (the new case `hit`) have no reference
  yet; every `hit-X` must equal its `cold-X`. Source: main's server log in section 2 shows checkpoints only for prompts
  of 4,096 tokens and more.
- The same on the big model (`scripts/golden-server.sh compare work/build/w81`, 601 s for the old cases plus the new
  case).
- `golden.sh` (cpu, gpu, prefill, prefill dflash): identical (`ie-gpu-gen` changed only in the teacher tool).
- `bench/sha.toml` (`scripts/ab.py`, IE_DETERMINISTIC=1, mid and long): identical. A single-message cold prompt writes
  no checkpoint (the start of the last message is at the start of the prompt, under the 4,096-token minimum), so the
  path is the old one.

**Speed (rule 1).** `scripts/ab.py --a w81-base --b w81` (w81-base is main 98dc089 built in `work/build/w81-base`),
workloads `mid` (prefill and decode at 30K) and `short` (decode at 2K), `IE_DETERMINISTIC=1` for `mid` to read the SHA,
`IE_DIAG=1`. Expectation: 0 % ± noise for prefill and decode, because the cold path of these requests does not touch
the changed code: `prefill@30000`, `decode@30000`, `decode@2000` within the sd of `bench/noise.toml`, margin 0.5 %.
One round decides where the table has df >= 4, otherwise at least two rounds, at most six.

**Cost of a short rest behind a checkpoint (the one place where the work changes).** On main a rest of at most 32
tokens runs as forward passes of 4 tokens; now as a prefill call. Measured on the big model with a throw-away server
per arm: a 6K-token head stored by a warm-up (`max_tokens: 0`), then 8 requests that continue it with 30 new tokens
each (distinct text), `max_tokens: 1`; wall time of the request, **prefill** of 30 tokens plus one decode token.
Expectation: unknown; the prefill of 30 tokens streams the experts of 48 layers once (about the time of 8 forward
passes of 4 tokens, 8 x 25 ms = 0.2 s on the decode path). A difference beyond 100 ms per such request is reported as
cost in the report. The answers of both arms differ by design (main's checkpoint has the forward token).

Active tooling for all GPU runs: `IE_DIAG=1`; `IE_DETERMINISTIC=1` where stated; every run under the GPU lock, 60 s
between blocks, no build during a measurement.

## 4. Result: quality (rule 3)

Big model, head 14,000 tokens, 64 teacher-forced tail tokens through the megakernel, `IE_DETERMINISTIC=1`. Reference: one
prefill call (section 2). Variant: prefill pieces cut at 12,289 (layered 12,289 + chunk by chunk 1,711), which is what a
checkpoint at that position now stores. `IE_TEACHER_REF` against the reference dump:

| | Δ(-log p) | KL(reference ‖ variant) | top-1 equal |
|---|---:|---:|---:|
| prefill cut at 12,289 | +0.000000 | 0.000000 | 100.00 % |

As planned: zero by construction. The numbers of main's checkpoints (a forward token at the cut) were +0.0016 to +0.0025
nats with an upper end of 0.0059 (#79, `2026-10-30-prewarm.md`, section 2.2); the new state does not have that token.
The rule of `docs/backends/README.md#checks` (upper end below 0.01 nats) is met with nothing to estimate.

## 5. Results: speed, bit-exactness, references

A = `work/build/w81-base` (main 98dc089, built from a worktree), B = `work/build/w81` (this branch). Raw files:
`2026-11-02-prefix-bitexact-ab*.md/.json` (ab.py reports with plan blocks and logs) and
`2026-11-02-prefix-bitexact-raw/` (the two small drivers' output).

### 5.1 `ab.py`, workloads `mid,short`, normal mode: one finding, explained by a control

Six rounds, expectation -0.5..+0.5 % per metric (plan above). Within the expectation: prefill@2000 +0.02 %,
decode@2000 -0.08 %, prefill@2000:write +0.01 %, decode@2000:write -0.07 % (open, interval -0.74..+0.60), prefill@30000
+0.25 % [+0.20, +0.30] (B faster, inside the margin). **Outside the expectation: decode@30000 -4.31 % [-4.82, -3.81]**,
and the SHA of the 30K request differs between the arms (A `cde3f1093dce48a1`, B `1fad1ead2f1ac34c`, each stable), the expert
counts of that request differ too (71,024 / 10,531 against 67,023 / 10,509).

Cause: in this workload order the 30K request is the third on the server; the two 2K requests before it leave a stash it
shares a start with, so it writes a checkpoint ("checkpoint at 29614 (before the fixed end)" in both server logs). In A that
checkpoint ends in a forward token and the 28-token rest goes through forward; in B it is prefill only. Different numbers,
another text of 200 tokens, another speed. Control (`2026-11-02-prefix-bitexact-ab-mid.md`): the 30K request alone on a fresh
server (cold, no checkpoint, the code of both arms is the same), 3 rounds: prefill@30000 +0.03 % [-0.05, +0.11] equal,
decode@30000 -0.01 % [-0.51, +0.49] equal, SHA `1fad1ead2f1ac34c` in both arms and all rounds, expert counts identical. That is
B's text of the first run: the text that B produces after the checkpoint is the cold text, and the cold text decodes at
67.7 tok/s in both builds. The 70.1 tok/s of A in the first run is the speed of main's checkpointed text (3.12 tokens per
pass against 3.02 for the cold text), not a speed that B lost.

### 5.2 `ab.py`, `mid,long`, `IE_DETERMINISTIC=1` (R4)

Six rounds; `2026-11-02-prefix-bitexact-ab-det.md`.

| | A (main) | B | Reference `bench/sha.toml` |
|---|---|---|---|
| SHA 30000 (cold, first request of the server) | `08f0bd31e0c0277f` | `08f0bd31e0c0277f` | `08f0bd31e0c0277f` (equal) |
| SHA 130000 | `e9545d70f2400b87` | `536592a9aa29bd56` | `e9545d70f2400b87` (B differs) |

As predicted for the 130K request: it follows the 30K one on the same server, shares its first 30,000 tokens and writes a
fork checkpoint there; in main that checkpoint ended in a forward token. **Re-recorded** in `bench/sha.toml`:
`"130000"` `e9545d70f2400b87` -> `536592a9aa29bd56`, because of the checkpoint (rule 3: numbers change on purpose; quality
section 4). `"30000"` is unchanged. The rates: prefill@30000 -0.01 % equal, decode@30000 +0.01 % equal, prefill@130000
+0.08 % (B faster, inside the margin), decode@130000 -0.68 % [-1.38, +0.03] open; that last request has another text in B
(expert counts 71,526 / 8,498 against 76,065 / 11,270), so it is the same effect as in 5.1 and not a comparison of equal work.
The 130K SHA of B was not compared with a 130K request alone on a fresh server; for the 30K request this comparison holds
(5.1).

### 5.3 Short rest behind a checkpoint (`shortrest.py`)

Big model, MTP, `IE_DETERMINISTIC=1`, warm-up of a 7,369-token head, then 8 requests of 29 new tokens each
(`max_tokens: 1`), two server starts per arm, wall time per request:

| | A (main: rest through forward passes of 4) | B (rest through the prefill) |
|---|---:|---:|
| request, run 1 / run 2 (mean of 8, sd) | 0.393 / 0.388 s (0.011, 0.012) | 0.314 / 0.314 s (0.010, 0.010) |
| warm-up of the head and the end (once) | 4.98 / 4.98 s | 4.87 / 4.89 s |

B is 0.075 s (19 %) faster for a rest of 29 tokens. The expectation of the plan (a difference beyond 100 ms is a cost) holds:
there is none at this size.

### 5.4 Agent-shaped requests (`agentic.py`)

16 conversations (a 4K-20K-token first message, "Noted.", a short question; the second request of each asks another short
question and continues the checkpoint at the start of the last message), two server starts per arm in the order A B B A.
Wall time with `max_tokens: 1` (pass L, 8 conversations): the hit request with a rest of 16-20 tokens takes 0.240 / 0.234 s on
A and 0.248 / 0.249 s on B (+0.010 s, +4 %); the first request of a conversation (6K-token prefill plus rest) 4.77 / 4.56 s on A,
4.54 / 4.54 s on B. So the prefill of a short rest costs about 10 ms more at 16-20 tokens and saves 75 ms at 29 (5.3): the break-even
is near 22 tokens. Decode rate with 200 tokens (pass D, 16 requests, 7 with 200 tokens in all four runs): tokens per pass
differ by request in both directions (e.g. 2.74 -> 2.88, 2.80 -> 2.50, 4.20 -> 3.50), paired mean of the tok/s difference
+2.1 % with sd 7.1 % over 7 pairs, tokens per pass -1.3 %: no systematic loss, a different text per request. The text
dependence is as large as the effect in 5.1.

### 5.5 References re-recorded (rule 3)

Predicted in section 3: only requests that write or read a checkpoint. Measured with `golden-server.sh compare` before
recording:

| Reference | Model | SHA before | SHA after | Why |
|---|---|---|---|---|
| `work/golden/server/ssd-1.json` | big | `c6fc78010c136b6f` | `ef53ac87f04846f7` | a 20,000-character first message: checkpoint at the start of the last message, prefill only now |
| `work/golden/server/ssd-2.json` | big | `9174c47c822943d7` | `2946b2a9c3155415` | continues that checkpoint (read from the SSD after a restart) |
| `work/golden/mini/server/ssd-1.json` | mini | `034b6a2b5309902e` | `58fb2e02840eec1b` | the same |
| `work/golden/mini/server/ssd-2.json` | mini | `1aa74d749e80a6bb` | `667ae60c1348a2f9` | the same |
| `hit-*.json`, `cold-*.json` (16 files each model) | big, mini | none | new | the new case `hit`; every `hit-X` equals its `cold-X` (checked by the script) |

Identical to the old references (not re-recorded): `anthropic*`, `openai*`, `sampled-*`, `thinking-anthropic`, `reuse-*`
(36 results, 20 of them old, 16 new). On main (`work/build/w81-base`) the case `hit` fails for all 8 pairs; on this branch for none
(mini; big: none).
