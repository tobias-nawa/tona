# Mac M1 Max: decode attention split over the cells and the indexer's radix select (decode)

Two decode changes on the real Coder (variant D, 164 experts):

- 1ca482b (D1): attention split over the cells (`attend_part`, 128 cells per threadgroup, `attend_join`), instead of
  one threadgroup per head serial over all cells;
- working tree on 1ca482b (D2): the indexer's top-k by radix select (`idx_top_r`, O(n_pool)) instead of the rank by
  counting (`idx_top`, O(n_pool²)).

Raw rounds: `2026-11-11-mac-decode-attn-{2k,8k,32k}.jsonl`.

## Plan (written before the run)

**Model.** `work/models/coder-d/coder-D-164e-0000{1,2}-of-00002.gguf`, sha256 of part 1 checked
(ef3d9e1b…4d36), part 2 the PLE table (316b46f3…e113).

**Arms.** Release builds, kernels without fast math, `TONA_PREFILL_CHUNK=2048`:

| Arm | Build | Code |
|---|---|---|
| A | `work/build/mtl-s0` (worktree) | 3ff1ffb: per-head attention, counting top-k |
| B | `work/build/mtl-s1` (worktree) | 1ca482b: D1 |
| C | `work/build/mtl` | D1 + D2 (uncommitted, committed after the run if it wins) |

**Runs.** `bench/mac/prefill_ab.py --arms mtl-s0:2048,mtl-s1:2048,mtl:2048 --decode 64 --rounds 1`, processes A B C
alternating, 30 s pause; each one warmup and one counted round (prefill, then 64 decoded tokens):

1. prompt 2048, 2 processes per arm, `--ctx 4096` (~6 min);
2. prompt 8192, 2 processes per arm, `--ctx 12288` (~15 min);
3. prompt 32768, 1 process per arm, `--ctx 36000` (~30 min).

**Expectation** (decode tok/s; source: smoke runs on D and A of this session, one round each):

- 2048: A 14 ±1.5 (A on variant A: 13.8), B 20.9 ±1.5, C 19.1 to 21.
- 8192: A 13 ±1.5, B 18.4 ±1.5, C 17 to 19.
- 32768: A unknown, below 12 (the counting top-k at n_pool 8192: 67M comparisons per layer and token), B unknown,
  C 19.5 ±3 (one smoke run, implausibly above its own 8K run).
- Prefill equal in all arms within 3 % (the prefill path is not touched).

The question this run decides: is C faster than B, at which length, and by how much. A result outside the bands is
a finding.

**Active tooling.** No profiler. `TONA_SEQS=1`, `LC_ALL=C`. No build and no other GPU work during the runs.

## Test conditions

Before the run (`tools/hw_report.py`, 21:07): Apple M1 Max, 10 cores, 32.0 GiB; macOS 27.0.1 (arm64); no `TONA_*`
variable in the shell. The run waited until the quiet check no longer warned (load average of my builds before).
Quiet check before (21:08:58): **ok**. After (21:58:40): "do not measure: the GPU is 97 % busy", the tail of the last
process; 10 s later (21:58:50): **ok**. `pmset -g therm`: no warning before or after any process. Free + inactive
memory 24.8 to 27.1 GB before each process. Swap-ins per process: up to 179 at 2K and 8K; at 32K 448 (A), 3982 (B),
274 (C), small against the run (~600 s). No build or other GPU work ran during the run.

## Results

Counted rounds (warmup not counted), tok/s:

| Prompt | Arm | Decode | Prefill |
|---|---|---|---|
| 2048 | A (3ff1ffb) | 13.84, 13.82 | 223.6, 224.9 |
| 2048 | B (D1) | 20.96, 21.06 | 224.9, 224.9 |
| 2048 | C (D1 + D2) | 20.47, 20.24 | 224.8, 224.8 |
| 8192 | A | 12.87, 12.89 | 178.1, 177.6 |
| 8192 | B | 18.96, 18.93 | 178.9, 177.7 |
| 8192 | C | 20.48, 20.43 | 179.0, 178.7 |
| 32768 | A | 7.29 | 108.5 |
| 32768 | B | 8.89 | 108.3 |
| 32768 | C | **20.01** | 108.7 |

- **D1 (B against A): decode +52 % at 2K** (21.0 against 13.8), +47 % at 8K, +22 % at 32K. Inside the expected bands
  at 2K and 8K.
- **D2 (C against B):** −2.9 % at 2K (20.4 against 21.0, at the lower edge of the expected 19.1 to 21), **+8 % at
  8K, ×2.25 at 32K**.
- **Finding:** at 32K the counting top-k is the bottleneck, as feared: A and B fall to 7.3 and 8.9 tok/s. With the
  radix select decode stays at **20.0 tok/s at 32K**, as fast as at 2K: the cost per decode pass no longer grows
  with the context in this range.
- **Finding:** at 2K the radix select costs ~1.4 ms more per pass than the counting (four passes over 512 scores
  with barriers against one); a switch at a block count would win those 3 % back for prompts below ~4K, not done
  (one path, and coding contexts are mostly longer).
- Prefill equal in all arms within 1 % at every length, as planned.

**Decision:** keep D2. Decode on the Coder (variant D): 13.8 → 20.4 tok/s at 2K, 12.9 → 20.5 at 8K, 7.3 → 20.0 at
32K.
