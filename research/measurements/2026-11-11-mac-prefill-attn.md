# Mac M1 Max: prefill step 3, attention in tiles and the indexer select in O(n) (prefill)

Third of the five prefill steps: in a prefill chunk the indexer scores run as matrix products, each token's top
blocks come from a radix select into a bitmap, and attention runs per tile of 8 tokens × 3 query heads of one KV head
(`kernels/qsa_prefill.metal`) instead of one dispatch per token. Raw rounds:
[2026-11-11-mac-prefill-attn.jsonl](2026-11-11-mac-prefill-attn.jsonl).

## Plan (written before the run)

**Model.** `work/models/synth/synth-a-e256.gguf`, as in [2026-11-11-mac-prefill-experts.md](2026-11-11-mac-prefill-experts.md).

**Arms.** Two builds, both Release, kernels without fast math, `TONA_PREFILL_CHUNK=512`:

| Arm | Build | Commit |
|---|---|---|
| A | `work/build/mtl-s1` (worktree) | 995bb7f, step 2 |
| B | `work/build/mtl` | 88cc452, step 3 |

**Runs.** Two sets with `bench/mac/prefill_ab.py`, 30 s pause between processes:

1. `--arms mtl-s1:512,mtl:512 --procs 3 --prompt 1024 --rounds 2`: six processes A B A B A B, each one warmup and
   two counted rounds (prefill 1024, decode 16). ~6 min.
2. `--arms mtl-s1:512,mtl:512 --procs 2 --prompt 2048 --rounds 1`: four processes A B A B, one warmup and one counted
   round (prefill 2048, decode 16). ~6 min. Shows how the gain grows with the prompt.

**Expectation.**

- Set 1: A 67.8 tok/s ±1 % (step 2 measurement); B 122 tok/s ±6 (two smoke runs: 121.8, 122.0). Source of the size:
  the profile's attention fell from 3018 to 155 ms for two 512-token chunks.
- Set 2: A ~50 tok/s (one smoke run of step 2 at 2048: 50.0, chunk 2048); B 115 ±8 (one smoke run: 115.3).
- Decode after the prompt equal in both arms within 3 % (16.8 to 17.3 after 1024; 13.5 to 13.9 after 2048).

B outside these bands or a decode difference beyond 3 % is a finding.

**Active tooling.** No profiler. `TONA_SEQS=1`, `TONA_PREFILL_CHUNK=512`, `LC_ALL=C`; no other `TONA_*` variable.

**Correctness before the run** (commit 88cc452): `TONA_PREFILL_CHUNK=64 tona-check --prefill 5 --widths 1` on the
mini model (top_k 64, so the radix select runs) and on `synth-a-l4-e16.gguf` (all blocks selected): top-1 206/206,
widths 0/423, cut invariance byte-equal 3 of 3, prefill against decode nll ≤ 6.7e-5, then 69 of 69 top-1 equal.
`TONA_PREFILL_CHUNK=96 -n 600 --prefill 7` on the mini model: cut invariance 3 of 3 (9, 10, 7 cuts), 404 tokens
prefilled, nll deviation ≤ 4.2e-2 and 202 of 202 decode steps top-1 equal (worst 3.1e-4): the same numbers as step 2,
so not from step 3 (unchecked cause: a block selection flipped near a tie). `TONA_SEQS=2 --widths 2`: 0/786. A
negative control (attention scale dependent on the tile start) fails the cut test 3 of 3. While building this step a
kernel file was missing from the build's dependency list, so one round of checks ran on a stale library; all checks
above ran after the fix, on the library of 88cc452.

## Test conditions

Before the run (`tools/hw_report.py`, 17:26): Apple M1 Max, 10 cores, 32.0 GiB; macOS 27.0.1 (arm64); no `TONA_*`
variable in the shell; load average 2.41, 4.38, 4.01; other programs 7.8 % of all CPU threads. Swap 15.16 GB, free +
inactive memory 22.9 to 23.4 GB before each process.

Quiet check before (17:26:38): warning, other programs 6.5 % of all CPU threads. After (17:39:25): warning, 6.6 %;
10 s later 8.3 %. `pmset -g therm`: no warning before or after any process.

**Disturbance.** From about 17:27:30 to 17:28:36 a build of the engine (a diagnostic, see the next measurement) ran
on the CPU, during the start of process 3 (arm A). Process 3 is not different from processes 1 and 5 (below).

## Results

Set 1, prefill of 1024 tokens, tok/s (raw: the `.jsonl`):

| Process | Arm | Prefill | Decode after it (16 tokens) |
|---|---|---|---|
| 1 | A (step 2) | 68.05, 68.02 | 16.99, 16.98 |
| 2 | B (step 3) | 122.01, 122.15 | 16.96, 16.93 |
| 3 | A (build running at its start) | 68.09, 68.03 | 16.81, 16.80 |
| 4 | B | 123.18, 122.81 | 16.82, 16.80 |
| 5 | A | 68.10, 68.01 | 16.82, 16.84 |
| 6 | B | 123.13, 122.88 | 16.85, 16.84 |

Set 2, prefill of 2048 tokens, one counted round per process:

| Process | Arm | Prefill | Decode after it |
|---|---|---|---|
| 7 | A | 45.46 | 13.55 |
| 8 | B | 116.07 | 13.55 |
| 9 | A | 45.50 | 13.53 |
| 10 | B | 115.94 | 13.64 |

- **Set 1: B 122.7 tok/s** (122.01 to 123.18, n = 6) against A 68.05 (68.01 to 68.10): **×1.80**, inside the
  expectation (116 to 128).
- **Set 2: B 116.0 tok/s** (115.94, 116.07) against A 45.5 (45.46, 45.50): **×2.55**. B inside the expectation
  (107 to 123). **Finding:** A below the expected ~50: the smoke run behind that number used chunks of 2048, this
  run chunks of 512; an earlier step-2 run with chunks of 512 at 2048 tokens gave 45.3, so the expectation was taken
  from the wrong arm.
- Decode in both arms equal (16.8 to 17.0 after 1024, 13.5 to 13.6 after 2048).
- Over the three steps at 1024 tokens: 32.7 (decode passes) → 47.2 → 67.8 → 122.7 tok/s, ×3.75. Target ≥ 250; the
  llama.cpp yardstick on the real model 182 (pp512).

## Where the time goes now (profile, not a speed result)

`TONA_METAL_PROFILE=1`, two 512-token chunks: routed experts 43 %, DeltaNet projections 20 %, gated residual reads
12 %, DeltaNet scan 10 %, attention projections 6 %, indexer 4 %, shared expert 3 %, attention 2 % (155 ms, from
3018 ms in step 2).

## Addendum: where prefill and decode part on long prompts (checked after the run)

On the mini model with 404 prefilled tokens (`--prefill 7`, chunk 96) the -log p of the prefill differs from the
decode by up to 4.16e-2, above the 0.01 rule of thumb; step 2 gives the same numbers. A diagnostic build (not
committed) dumped, per token and layer, the router logits and the indexer's block selection of both paths:

- **First difference: the router of layer 0 at position 348.** Its logits differ between prefill and decode by at
  most 2.2e-6 (rounding: GEMM against mat-vec), and in the decode the 10th and 11th largest logit are 7.2e-7 apart:
  a tie, so the two paths pick a different expert set.
- From there position 348 differs in every later layer (router logits by up to 2.7e-2), with router flips in layers
  8, 9 and 10 at positions 348 to 358.
- The indexer picks different blocks in 3 of 1203 (layer, position) pairs, all in the last QSA layer at positions
  358, 366 and 403, all after the tie. Before position 256 the score differences stay ≤ 1.5e-5.
- The largest -log p differences sit at positions 348 to 368; the median |Δ| is 3.9e-6 before position 256 and 8.0e-6
  after.

Read: a tie in the expert routing, expected from rounding, not a fault of a prefill kernel. 937 prefilled tokens
(chunk 1024) show the same root (4.16e-2 there too, 8.55e-2 at most). Agreed with the maintainers: the measure that
counts is prefill against decode on the real model over ~2K tokens of code.
