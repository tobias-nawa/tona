# Mac M1 Max: prefill step 1b, GEMM tiles of 64 tokens and the chunk size (prefill)

Step 1b of the prefill path: the GEMMs take 64 rows × 64 tokens per threadgroup with 8 simdgroups (each decoded
weight serves up to 64 tokens; an expert's work follows its token count in steps of 16), attention runs in slices of
512 tokens and the long parts of a layer get command buffers of their own. The states are bit-identical to step 3.
Raw rounds: [2026-11-11-mac-prefill-gemm64.jsonl](2026-11-11-mac-prefill-gemm64.jsonl).

## Plan (written before the run)

**Model.** `work/models/synth/synth-a-e256.gguf`, as in [2026-11-11-mac-prefill-attn.md](2026-11-11-mac-prefill-attn.md).

**Arms.** Release builds, kernels without fast math:

| Arm | Build | Commit | `TONA_PREFILL_CHUNK` |
|---|---|---|---|
| A | `work/build/mtl-s1` (worktree) | 88cc452, step 3 | 512 |
| B | `work/build/mtl` | db4713a, step 1b | 512 |
| C | `work/build/mtl` | db4713a, step 1b | 2048 |

**Runs.** `bench/mac/prefill_ab.py --arms mtl-s1:512,mtl:512,mtl:2048 --procs 2 --prompt 2048 --rounds 1`: six
processes A B C A B C, 30 s pause, each one warmup and one counted round (prefill 2048, decode 16). ~7 min. No build
and no other GPU work during the run.

**Expectation.**

- A: 116.0 tok/s ±3 % (step 3 measurement at 2048 tokens, same build).
- B: 126.5 ±6 (one smoke run: 126.47).
- C: 179 ±10 (one smoke run of db4713a's GEMM before the slicing: 178.73; the slicing raised a 4096-token prefill
  from 157 to 173, so C may land at the upper end).
- Decode after the prompt: equal in all arms within 3 % (13.5 to 13.6).

An arm outside its band, or a decode difference beyond 3 %, is a finding.

**Active tooling.** No profiler, no `TONA_METAL_CB_LOG`. `TONA_SEQS=1`, `TONA_PREFILL_CHUNK` per arm, `LC_ALL=C`.

**Correctness before the run** (db4713a): `tona-check --prefill` on the mini model (chunks 64, 96 and 1024, the last
with 937 prefilled tokens, so two attention slices) and on `synth-a-l4-e16.gguf` (chunk 64): cut invariance
byte-equal in every round; prefill against decode the same numbers as step 3 (137 tokens: nll ≤ 2.7e-5 and 6.7e-5;
404 tokens: 4.16e-2) and as step 2 for 937 tokens (8.55e-2, decode afterwards 469 of 469 top-1, worst 1.9e-2), all
downstream of the router tie at layer 0, position 348 (gap 7.2e-7) shown in the addendum of [2026-11-11-mac-prefill-attn.md](2026-11-11-mac-prefill-attn.md);
Widths 0/423, `TONA_SEQS=2` 0/786, unit 45/45. Longest command buffer (`TONA_METAL_CB_LOG=1`, prefill of
4096 tokens in chunks of 2048): 0.53 s before the slicing, 0.22 s after.

## Test conditions

Before the run (`tools/hw_report.py`, 18:01): Apple M1 Max, 10 cores, 32.0 GiB; macOS 27.0.1 (arm64); no `TONA_*`
variable in the shell; load average 3.94, 4.33, 3.75 (my builds before; the run waited until the quiet check no
longer warned about it); other programs 5.7 % of all CPU threads. Swap 15.12 GB; free + inactive memory 20.9 GB
before process 1, 23.3 to 24.0 GB before the others.

Quiet check before (18:02:23): **ok**. After (18:08:09, right after the last process): "do not measure: the GPU is 99 %
busy", the tail of the last process; 10 s later (18:08:19): **ok**. `pmset -g therm`: no warning before or after any
process. No build or other GPU work ran during the run.

## Results

Prefill of 2048 tokens, tok/s, one counted round per process (warmup in brackets, not counted):

| Process | Arm | Prefill | Decode after it |
|---|---|---|---|
| 1 | A (step 3, chunk 512) | 114.41 (106.04) | 13.57 |
| 2 | B (1b, chunk 512) | 135.45 (133.15) | 13.75 |
| 3 | C (1b, chunk 2048) | 190.26 (183.71) | 13.71 |
| 4 | A | 116.05 (114.56) | 13.66 |
| 5 | B | 135.37 (133.04) | 13.71 |
| 6 | C | 190.56 (184.73) | 13.62 |

- A: 115.2 tok/s (114.41, 116.05), inside the expectation (116 ±3 %).
- **B: 135.4** (135.37, 135.45): **×1.18 over A**. **Finding:** above the expected 120.5 to 132.5; the smoke run
  behind the expectation (126.47) predates the attention slicing, which the plan already flagged as a possible gain.
- **C: 190.4** (190.26, 190.56): **×1.65 over A**, ×1.41 over B. **Finding:** slightly above the expected 169 to 189,
  same reason.
- Decode in all arms 13.6 to 13.8, within 3 %.
- Memory (`/usr/bin/time -l`, `--ctx 4096`): peak footprint 0.99 GB with chunks of 512, 1.87 GB with 2048. The
  per-token scratch grows with the chunk; the indexer scores per token also grow with the context (at 262k tokens of
  context ~0.5 GB for a 2048-token chunk).

**Decision:** default `TONA_PREFILL_CHUNK` 2048 (as the reference). The longest command buffer stays at 0.22 s.

At 2048 tokens the prefill went 45.5 (decode passes of step 2's baseline arm) → 116.0 (step 3) → 190.4 tok/s (1b,
chunk 2048). Target ≥ 250; llama.cpp on the real model 182 (pp512, different prompt length and the real weights).
