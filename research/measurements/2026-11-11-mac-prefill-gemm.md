# Mac M1 Max: prefill step 1, dense matrices as GEMM (prefill)

First of the five prefill steps: in a prefill chunk every dense matrix (DeltaNet and attention projections, indexer,
router, shared expert, LM head) runs as one GEMM on the 8×8 matrix units (`kernels/gemm.metal`) instead of mat-vecs
over 4 tokens; everything else (routed experts, attention, DeltaNet scan, elementwise) is unchanged. A chunk spreads
over one command buffer per layer. Raw rounds: [2026-11-11-mac-prefill-gemm.jsonl](2026-11-11-mac-prefill-gemm.jsonl).

## Plan (written before the run)

**Model.** `work/models/synth/synth-a-e256.gguf` (`tools/make_synth_model.py --experts 256 --variant a`, 20.41 GB):
the Coder's shapes and types per role, the real expert count of variant A
([2026-11-11-mac-synth.md](2026-11-11-mac-synth.md)).

**Arms.** One build (`work/build/mtl`, Release, kernels without fast math, working tree on be15083), switched by
environment only:

| Arm | `TONA_PREFILL_CHUNK` | Path |
|---|---|---|
| A | 0 | the old prefill: decode passes of 8 tokens, mat-vecs over 4 tokens per dispatch |
| B | 512 | GEMM prefill in chunks of 512 tokens |

**Runs.** `bench/mac/prefill_ab.py --arms 0,512 --procs 3 --prompt 1024 --rounds 2`: six processes A B A B A B, 30 s
pause between them; each `tona-bench -p 1024 -d 16 -b 1 -r 2 --ctx 4096` (one warmup round not counted, two rounds:
reset, prefill of 1024 tokens, 16 decoded tokens). 6 counted prefill rounds per arm. `pmset -g therm`, free memory,
swap and page-ins before and after each process. A 1024-token prompt is two chunks of B, so every B round also cuts
once. ~1 min per process, ~9 min in all.

**Expectation.**

- A: 37–41 tok/s prefill (40 in rows 1–18 of the synth measurement at 256 tokens; 37–38.5 after the cooldown control).
  Longer prompt, more attention: the lower end.
- B: 47 tok/s ±6 (one smoke run at 1024 tokens: 47.0; at 512: 58.6). Why not more: in a profiled 512-token chunk
  (`TONA_METAL_PROFILE=1`, 18.0 s GPU over warmup and round) the routed experts take 59 % and per-token attention
  17 %, both still on the decode kernels (steps 2 and 3); the dense GEMMs take ~12 %.
- Decode (16 tokens, B=1) the same in both arms within ±3 % (the decode path is untouched), ~21 tok/s.

A B outside 41–53 or a decode difference beyond 3 % is a finding.

**Active tooling.** No profiler, no diagnostics. `TONA_SEQS=1`, `TONA_PREFILL_CHUNK` per arm, `LC_ALL=C`; no other
`TONA_*` variable.

**Correctness before the run** (same tree): `tona-check --prefill 5 --widths 1` with `TONA_PREFILL_CHUNK=64` on the
mini model and on the 4-layer model of real shapes (`synth-a-l4-e16.gguf`): top-1 206/206, widths 0/423 not
bit-identical, cut invariance byte-equal in 3 of 3 rounds (3, 6 and 3 random cuts), prefill against decode: nll
deviation ≤ 4.3e-5, then 69 of 69 decode steps top-1 equal (worst 1.55e-6). `TONA_SEQS=2 --widths 2`: 0/786. A
negative control (GEMM input scaled by `1 + t0 × 1e-6`, i.e. dependent on the tile position) fails the cut test in 3
of 3 rounds.

## Test conditions

Before the run (`tools/hw_report.py`, 16:16):

- CPU: Apple M1 Max, 10 cores / 10 threads; GPU: Apple M1 Max, 32.0 GiB unified memory; RAM 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64); no toolkit; no `TONA_*` variable set in the shell; no cgroup limits
- Load average (1, 5, 15 min): 2.43, 2.88, 2.71; CPU use by other programs 11.3 % of all threads
- Commit `be15083` plus the step-1 tree (dirty); the build's `tona.metallib` is byte-equal to the one of the commit
  that adds the GEMM prefill
- Swap 15.22 GB used; free + inactive memory 21.1 GB before the first process, 22.9 to 23.3 GB after

Quiet check before (16:16:05): **warning**, other programs 10.4 % of all CPU threads, GPU 6 % busy. After (16:26:55,
right after the last process): "do not measure: the GPU is 99 % busy", the tail of the last process in the
utilization counter; 10 s later (16:27:05): warning, other programs 10.3 %, GPU 6 % busy, the same as before. The
maintainers agreed to measure with the warning: the arms alternate, so a constant background load hits both.
`pmset -g therm`: no thermal or performance warning recorded before or after any process.

## Results

Prefill of 1024 tokens, tok/s, two counted rounds per process (raw: the `.jsonl`):

| Process | Arm | Prefill | Decode after it (16 tokens, B = 1) |
|---|---|---|---|
| 1 | A (chunk 0) | 32.88, 32.92 | 17.04, 17.14 |
| 2 | B (chunk 512) | 47.23, 47.25 | 16.79, 16.80 |
| 3 | A | 32.62, 32.62 | 16.83, 16.83 |
| 4 | B | 47.22, 47.23 | 16.90, 16.94 |
| 5 | A | 32.64, 32.64 | 16.91, 16.96 |
| 6 | B | 47.29, 47.20 | 16.82, 16.87 |

- **B: 47.2 tok/s** (47.20 to 47.29, n = 6), inside the expectation (41 to 53).
- **A: 32.7 tok/s** (32.62 to 32.92, n = 6). **Finding:** below the expected 37 to 41. The expectation came from
  prompts of 256 tokens; at 1024 the per-token attention and the indexer (idx_top is O(n) per token, O(n²) per
  prompt) cost more. Not separately measured; the profile below supports it (attention 17 % at 512 tokens).
- **Speed-up B / A: ×1.44** at 1024 tokens.
- **Decode** in both arms 16.8 to 17.1 tok/s, A and B within 1 %: the decode path is unchanged, as planned.
  **Finding:** ~17 instead of the expected ~21: these 16 tokens decode at positions 1024 to 1040, the earlier 21 was
  measured at 256 to 320; the attention and indexer cost per token grows with the position.
- Every process 68.6 s (B) or 97.6 s (A), no swap-ins worth noting (≤ 16 per process), swap constant at 15.22 GB.

## Where the time goes (profile, not a speed result)

`TONA_METAL_PROFILE=1` (one command buffer per part, GPU start to end; the split itself costs time), separate runs
before the measurement:

| Part | Prefill, 512-token chunk, % of GPU time | Decode B = 1, ms per pass |
|---|---|---|
| routed experts (mat-vec per slot) | 59 % | 13.9 |
| attention per token | 17 % | 0.6 |
| DeltaNet projections (GEMM in prefill) | 8.3 % | 11.6 |
| gated residual reads | 4.6 % | 8.8 |
| DeltaNet scan | 4.4 % | 2.0 |
| indexer, attention projections, shared expert, router, head, PLE | 7.1 % | 13.9 together |

Next step by this profile: routed experts grouped by expert as GEMMs (step 2), then attention (step 3). The gated
reads (8.8 ms of a ~51 ms profiled decode pass) were not in any kernel bench: most of the unexplained 11 ms of the B
= 1 pass ([2026-11-11-mac-synth.md](2026-11-11-mac-synth.md)); a separate point for the decode work.

Yardstick: llama.cpp prefill (pp512) on an M1 Max with the real Coder: 182 tok/s (lilting.ch); the plan band for the
finished prefill path: 310 to 425 tok/s, target ≥ 250.
