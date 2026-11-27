# IndexCache: share the block selection across QSA layers (#73, 2026-10-29)

Question: IndexCache (arXiv 2603.12201) lets only some "full" layers run the indexer (`k_idx_score_wmma` + `k_topk` in
prefill, `OP_IDX_SCORE` + `OP_TOPK` in the megakernel); the "shared" layers in between attend to the cells of the last
full layer. Our model has 12 QSA layers of 48. What does it bring, and what does it cost in quality?

Build: `IE_IDX_SHARE=<full QSA layers>` (index 0-11, layer 0 always full) in the decode program only; unset gives the
old program (golden decode, golden prefill identical, see below). The prefill path ignores the switch (not built, see
Open points). Raw data: `/var/tmp/claude/w73/` (not in the repository).

## Phase A: ceiling

Prefill (rocprofv3, main, 249,731-token prompt, `--ctx 262144`, GPU busy 191.4 s): `k_idx_score_wmma` 14.53 s +
`k_topk` 4.17 s = 18.7 s = 9.8 %; sharing every second indexer would save about 4.9 %. At 128K (#71,
`2026-10-29-prefill-topk.md`): 5.5 s of 92.7 s = 5.9 %, half of it 3.0 %.

Decode, pass time from the server log (200 tokens, 1 run each, rough): 128K prompt 51.8 ms for all 12 indexers,
47.3 ms with 6 full (-8.7 %), 43.0 ms with layer 0 only (-17 %); 250K prompt 65.0 / 53.2 (-18 %) / 46.5 ms (-28 %).
Decode is the lever; the abort rule (< 3 % on both) did not apply.

## Phase C: quality (decode path)

Method: the first 32,768 or 131,072 tokens of a text go through the prefill path (all indexers), the next 4,096
tokens run teacher-forced through the megakernel (`IE_TEACHER_TAIL=4096`, one token per pass), once for the reference
(all 12 indexers, `IE_TEACHER_DUMP`) and once per pattern (`IE_TEACHER_REF`, `IE_TEACHER_OUT`). Per token: -log p of
both, KL(reference || pattern) over the whole vocabulary, top-1 equal. Evaluation with
`scripts/nll-compare.py` (block bootstrap, 64-token blocks, 4,000 draws). Rule fixed beforehand (`docs/backends/README.md`):
the 95 % interval of Δ(-log p) contains 0 and lies below 0.01 nats.

Texts: py = 800 kB of the Python 3.14 standard library, cpp = 800 kB of llama.cpp (`src/`, `common/`), sh = 800 kB of
shell scripts and bash completions of the system. Calibration (greedy search only): the first 250 kB of the engine
sources, 32K prefill + 1,024 tail. Determinism check: the same setting twice gives KL 0.000000.

Greedy search (drop the full layer whose removal raises the NLL least; `greedy.log`): order of dropped layers 5, 6, 8,
7, 11, 10, 4, 1 (down to 4 full: 0,2,3,9). Single-run differences on 1,024 tokens were about ±0.01 and mostly
negative ("sharing improves"): noise, the choice overfits the calibration text.

Δ(-log p), pattern against reference, mean [95 % interval], 4,096 tail tokens:

| Pattern (full layers) | py 32K | cpp 32K | sh 32K | py 128K | cpp 128K | sh 128K |
|---|---|---|---|---|---|---|
| (a) 0,2,4,6,8,10 | +0.0008 [-0.0018, +0.0034] | +0.0045 [+0.0004, +0.0087] | -0.0011 [-0.0059, +0.0037] | -0.0015 [-0.0062, +0.0030] | +0.0049 [+0.0002, +0.0098] | +0.0039 [-0.0044, +0.0114] |
| (b) 0,1,2,3,4,9 | +0.0021 [-0.0024, +0.0074] | +0.0014 [-0.0035, +0.0067] | +0.0021 [-0.0030, +0.0077] | -0.0015 [-0.0083, +0.0051] | +0.0062 [+0.0014, +0.0111] | +0.0151 [-0.0007, +0.0323] |
| (c) 0,1,2,3,9 | -0.0002 [-0.0038, +0.0037] | +0.0072 [+0.0003, +0.0152] | +0.0040 [-0.0020, +0.0109] | +0.0011 [-0.0070, +0.0088] | +0.0096 [+0.0037, +0.0162] | +0.0260 [+0.0107, +0.0440] |
| (e) 0,2,3,9 | +0.0000 [-0.0051, +0.0068] | +0.0083 [+0.0008, +0.0170] | +0.0033 [-0.0034, +0.0111] | -0.0016 [-0.0091, +0.0057] | +0.0154 [+0.0015, +0.0380] | +0.0222 [+0.0068, +0.0386] |
| **(d) 0,1,3,4,6,7,9,10** | +0.0029 [-0.0009, +0.0068] | +0.0033 [-0.0006, +0.0071] | -0.0033 [-0.0091, +0.0025] | +0.0006 [-0.0034, +0.0048] | +0.0005 [-0.0026, +0.0037] | -0.0001 [-0.0076, +0.0076] |

KL (mean) and top-1 equal to the reference:

| Pattern | KL range over the six sets | top-1 equal range |
|---|---|---|
| (a) 6 full | 0.0037 - 0.0195 | 95.9 - 99.4 % |
| (b) 6 full, greedy | 0.0065 - 0.0351 | 96.1 - 99.5 % |
| (c) 5 full | 0.0070 - 0.0423 | 95.3 - 99.5 % |
| (e) 4 full | 0.0087 - 0.0449 | 94.8 - 99.4 % |
| (d) 8 full | 0.0034 - 0.0156 | 96.9 - 99.4 % |

Verdict against the rule: only (d) passes on all six sets. (a) fails strictly (cpp: the interval excludes 0; sh 128K:
upper bound +0.0114). The greedy patterns are not better than the uniform one and fail at 128K on cpp and sh. Fewer
full layers cost more on shell at 128K (the hardest set: top-1 equal 95 %).

Retrieval at 128K (`retr.py`: 8 key-value lines spread over about 125K tokens of C++, 8 single questions and 1 question
for all 8 codes, 3 seeds, greedy, server with MTP): reference 24/24 single and 24/24 multi; (d) 24/24 and 24/24;
(a) 24/24 and 24/24; control with layer 0 as the only full layer: 22/24 single, 23/24 multi (so the test can detect
losses, but is easy).

Tasks (`scripts/bench-effort.py medium`, 23 tasks, ctx 65,536): reference 22/23 (failed: csv), (d) 21/23 (failed:
roman, edit); within the spread of the reference (1-2 of 46). The tasks are mostly shorter than 2,051 positions,
where QSA selects nothing; this is a gross-failure check only.

## Phase D: speed (decode after a long prompt)

Server, MTP, `--ctx 262144`, engine sources as prompt, 600 tokens requested, 3 rounds interleaved, ms per pass =
generated tokens / tok/s / passes of the server log:

| Setup | 128K ms/pass | 250K ms/pass | 128K decode tok/s | 250K decode tok/s |
|---|---|---|---|---|
| main (`work/build/engine`) | 52.1 (51.4-52.9) | 61.5 (60.8-62.3) | 54.2 | 45.2 |
| w73, `IE_IDX_SHARE` unset | 52.6 (51.6-53.1) | 61.5 (59.7-62.8) | 53.0 | 43.6 |
| w73, (d) 8 full | 49.9 (48.9-50.5) = -4.2 % | 56.2 (55.4-57.0) = -8.6 % | 57.9 | 49.2 |
| w73, (a) 6 full | 46.5 (45.9-47.4) = -10.6 % | 52.8 (51.6-53.8) = -14.1 % | 54.6 | 51.2 |

The tok/s columns also move with the MTP acceptance (tokens per pass 2.56-2.89); the pass time is the clean number.
Prefill is unchanged (the switch does not touch it).

Reference unchanged (rule 5): `scripts/golden.sh compare`, `GOLDEN_MODE=prefill scripts/golden.sh compare` identical;
bench-set + lang, 3 rounds, short context: main 88.9 / 88.9 / 87.9 tok/s code (decode) and w73 89.1 / 88.8 / 88.8,
text decode 64.7 / 64.7 / 64.1 against 64.7 / 64.7 / 64.6; unchanged within the spread.

## Reading

- Decode only (d): -4 % pass time at 128K, -9 % at 250K, quality inside the rule on six sets and 24/24 retrieval.
- (a) saves about 2.5 times as much but fails the rule on C++ and shell by a small margin (mean +0.004 to +0.005
  nats).
- The paper's claim (about 75 % of the indexers removable) does not carry over without training: our model was
  trained with an indexer per layer and has only 12 QSA layers.

## Open points

- Prefill is not implemented. The cells of a chunk would have to survive the DeltaNet layers in between: 4,096 cells
  (16 KB) per token, 268 MB for a group of 16,384 tokens, in a plan that is already tight on VRAM. The ceiling is
  3-5 %.
- Measuring tool: the tail run is about 10 % slower with a reference (KL over 248,320 logits on the CPU per token),
  that is tool time, not decode time.

## Observations

- One barrier abort in an unchanged path (reference, no `IE_IDX_SHARE`): build `work/build/w73` at commit 58e81f4,
  command `ie-gpu-gen work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf "" --mega
  --teacher /var/tmp/claude/w73/corp/py.txt --max 135168 --ctx 136192` with `OMP_NUM_THREADS=6 IE_TEACHER_TAIL=4096
  IE_TEACHER_DUMP=...` under `systemd-run --user --scope`, 2026-10-29 between 10:05 and 10:07, text 131,072 tokens
  prefill then stepwise tail, 1,976 tokens into the tail:
  `Error: gpu::Mega: barrier aborted (time limit) at position 133048 (step 255: qsa.score qsa.score, longest step 76
  qsa.topk 0.3 ms, pass up to there 6.3 ms, blocks at step 255, all equal)`. The same command ran through directly
  afterwards (and in all later 128K runs). No GPU reset, the process exited with the error.
