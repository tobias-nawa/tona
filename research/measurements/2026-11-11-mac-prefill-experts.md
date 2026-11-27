# Mac M1 Max: prefill step 2, routed experts grouped by expert as GEMMs (prefill)

Second of the five prefill steps: in a prefill chunk the routed experts run as GEMMs grouped by expert
(`moe_group` lists the slots per expert, `gemmx_T` one tile column per expert) instead of one mat-vec per slot. Raw
rounds: [2026-11-11-mac-prefill-experts.jsonl](2026-11-11-mac-prefill-experts.jsonl).

## Plan (written before the run)

**Model.** `work/models/synth/synth-a-e256.gguf`, as in [2026-11-11-mac-prefill-gemm.md](2026-11-11-mac-prefill-gemm.md).

**Arms.** Two builds, both Release, kernels without fast math, `TONA_PREFILL_CHUNK=512`:

| Arm | Build | Commit |
|---|---|---|
| A | `work/build/mtl-s1` (worktree) | 9618392, step 1 only; its `tona.metallib` byte-equal to the measured one of step 1 |
| B | `work/build/mtl` | 094858b, step 2 |

**Runs.** `bench/mac/prefill_ab.py --arms mtl-s1:512,mtl:512 --procs 3 --prompt 1024 --rounds 2`: six processes A B
A B A B, 30 s pause, each one warmup and two counted rounds of a 1024-token prefill and 16 decoded tokens. ~8 min.

**Expectation.**

- A: 47.2 tok/s ±1 % (step 1 measurement, same build).
- B: 68 tok/s ±5 (one smoke run: 67.7). Source of the size of the gain: in the profile the experts fell from 10.6 s to
  3.46 s for two 512-token chunks; attention (27 % at 512, 48 % at 1024 tokens) is unchanged.
- Decode after the prompt: equal in both arms within 3 % (~17 tok/s).

B outside 63 to 73 or a decode difference beyond 3 % is a finding.

**Active tooling.** No profiler. `TONA_SEQS=1`, `TONA_PREFILL_CHUNK=512`, `LC_ALL=C`; no other `TONA_*` variable.

**Correctness before the run** (commit 094858b): `TONA_PREFILL_CHUNK=64 tona-check --prefill 5 --widths 1` on the
mini model and on `synth-a-l4-e16.gguf`: top-1 206/206, widths 0/423, cut invariance byte-equal 3 of 3 (3, 6, 3
cuts), prefill against decode nll ≤ 6.8e-5, then 69 of 69 top-1 equal; `TONA_SEQS=2 --widths 2`: 0/786.

## Test conditions

Before the run (`tools/hw_report.py`, 16:39): Apple M1 Max, 10 cores, 32.0 GiB; macOS 27.0.1, kernel 27.0.0 (arm64);
no `TONA_*` variable in the shell; load average 2.41, 3.60, 3.22; other programs 8.7 % of all CPU threads. Swap 15.17
GB, free + inactive memory 22.8 to 24.3 GB before each process.

Quiet check before (16:39:38): warning, other programs 11.2 % of all CPU threads (agreed with the maintainers, as in
step 1). After (16:48:07, right after the last process): "do not measure: the GPU is 98 % busy", the tail of the
last process; 10 s later (16:48:17): **ok**. `pmset -g therm`: no warning before or after any process.

## Results

Prefill of 1024 tokens, tok/s (raw: the `.jsonl`):

| Process | Arm | Prefill | Decode after it (16 tokens, B = 1) |
|---|---|---|---|
| 1 | A (step 1) | 47.04, 47.03 | 17.01, 17.32 |
| 2 | B (step 2) | 67.76, 67.72 | 17.06, 17.07 |
| 3 | A | 47.04, 47.04 | 17.00, 17.04 |
| 4 | B | 67.81, 67.60 | 16.93, 17.26 |
| 5 | A | 47.15, 47.23 | 17.20, 17.11 |
| 6 | B | 68.09, 67.99 | 16.83, 16.79 |

- **B: 67.8 tok/s** (67.60 to 68.09, n = 6), inside the expectation (63 to 73).
- A: 47.1 tok/s (47.03 to 47.23), as step 1 measured (47.2).
- **Speed-up B / A: ×1.44**; from the old decode passes (32.7, step 1 file): ×2.07.
- Decode in both arms 16.8 to 17.3, within 3 %: unchanged, as planned.
- Swap-ins 0 to 84 per process, swap constant at 15.17 GB.

## Utilization (profile, not a speed result)

`TONA_METAL_PROFILE=1`, prefill of one chunk in warmup and round, FLOPs from the parameter counts of the model file
(active experts 10 of 256: 2.36 G parameters; DeltaNet projections 2.08 G):

| Chunk | Experts | DeltaNet projections (dense) | Attention share of GPU time |
|---|---|---|---|
| 512 | 1.40 TFLOPS | 2.67 TFLOPS | 27 % |
| 1024 | 1.89 | 2.74 | 48 % |
| 2048 | 2.18 | 2.78 | 66 % |

The expert GEMMs fill their 32-token tiles to ~62 % at 512 tokens (on average 20 tokens per expert); both GEMMs stay
below the 4.2 to 5.5 TFLOPS of plain f16 GEMMs ([2026-11-11-mac-synth.md](2026-11-11-mac-synth.md)), and attention
per token grows into the largest part. Next: step 3 (attention in tiles, indexer select without O(n²)), then GEMM
tuning (step 1b). Unchecked: a decode of 0.04 tok/s after a 2048-token chunk in profile mode.

## Addendum: the decode of 0.04 tok/s in profile mode (checked after the run)

Only in profile mode. Without the profiler, decode at the same position after a prefill in chunks of 2048 and of 512
ran equally fast (13.84 and 13.69 against 13.70 and 13.68 tok/s, `-p 2048 -d 32`, one round each). With it, the main
thread waited in `waitUntilCompleted` at 0 % CPU; GPU time 78.8 s against 113 s wall time. The GPU timestamps of the
command buffers showed idle gaps of 0.1 to 0.3 s between consecutive buffers, about 25 s over the ~200 profiled
buffers of the decode, after the prefill had run attention buffers of ~2.2 s (chunk 2048, step 2); not after ~0.5 s
buffers (chunk 1024). Read: the system delays the process's later submissions after long command buffers; no state
of the backend is left open. Rule since then: every command buffer at most 0.5 s
(`engine/backend/metal/AGENTS.md`). Unchecked: the exact threshold, and whether this is documented.
