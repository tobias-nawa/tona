# Mac M1 Max: why decode slows down with the position (decode, profile)

Decode on the real Coder (variant A, 256 experts) runs at ~21 tok/s at position 256 and ~14 tok/s at 2K. The
decode pass still uses the per-token indexer of the first cut: `idx_score` and `idx_top` (rank by counting,
O(n_pool²) comparisons per token and QSA layer in one threadgroup of 1024), then `attend` (one threadgroup of 256 per
head, a serial loop over the cells for V). The prefill moved to `idx_select` (radix select) and `attend_t` (tiles) in
step 3.

## Plan (written before the run)

**Model.** `work/models/coder-a/coder-A-256e-00001-of-00002.gguf` (+ part 2), real weights; sha256 checked.

**Runs.** Build `work/build/mtl` at 47491af (Release, kernels without fast math).

1. Speed (decode, no profiler): `tona-bench -p P -d 32 -b 1 -r 1 --ctx 16384` for P = 256, 2048, 8192; one process
   each, one warmup and one counted round. Three numbers to place the curve, not an A/B. ~4 min.
2. Profile (`TONA_METAL_PROFILE=1`, one command buffer per part, GPU time; not a speed result): for each P the same
   run with `-d 0` and with `-d 8`; the decode share per part is the difference divided by 8 × 2 (warmup + round).
   ~10 min (profiled decode is slow after the long prefill buffers: the system gaps described in
   [2026-11-11-mac-prefill-experts.md](2026-11-11-mac-prefill-experts.md); GPU times are not affected).

**Expectation.**

- Decode: 21 ±2 tok/s at 256, 14 ±1.5 at 2048 (smoke runs of this session), at 8192 7 ±3 tok/s: the attention is
  capped at 2048 + 3 cells and does not grow beyond 2K, but `idx_top` grows with n_pool² (16× from 2K to 8K).
- Per pass at 2048 against 256 (+~24 ms): attention (`attend`, serial over 2048 instead of 256 cells per head)
  ~12 to 18 ms of it, the indexer (`idx_score` + `idx_top`) ~3 to 6 ms. At 8192 the indexer dominates (> 50 % of the
  pass).

A part outside these bands is a finding.

**Active tooling.** Run 2 with `TONA_METAL_PROFILE=1`; run 1 without. `TONA_SEQS=1`, `TONA_PREFILL_CHUNK` default
(2048), `LC_ALL=C`. No build and no other GPU work during the runs.

## Test conditions

Before the run (`tools/hw_report.py`, 20:12): Apple M1 Max, 10 cores, 32.0 GiB; macOS 27.0.1 (arm64); no `TONA_*`
variable in the shell. Quiet check before (20:12:09): **warning**, other programs 11.2 % of all CPU threads (a file
copy over the network ran during the runs); after (20:17:31): "do not measure: the GPU is 97 % busy", the tail of the
last run. Measured with the warning: one curve and profiles, no A/B. No build during the runs.

## Results

**Speed** (decode, 32 tokens after the prompt, one counted round each):

| Prompt | Prefill | Decode |
|---|---|---|
| 256 | 162.4 tok/s | **21.6 tok/s** |
| 2048 | 221.3 | **13.8** |
| 8192 | 173.2 | **12.8** |

- 256 and 2048 inside the expectation.
- **Finding:** at 8192 decode stays at 12.8 tok/s, far above the expected 7 ±3: the indexer does not grow as feared
  (`idx_top` at n_pool = 2048 costs ~5 ms more per pass than at 512, from 13.8 to 12.8 tok/s).

**Profile per decode pass** (ms; the difference of the runs with `-d 8` and `-d 1`, divided by 14 passes; the plan
said `-d 0`, which `tona-bench` rejects; first table of the run with `-d 0` therefore missing and `-d 1` run
afterwards):

| Part | Position 256 | Position 2048 |
|---|---|---|
| **attention (`attend`)** | 3.19 | **25.98** |
| routed experts | 10.80 | 9.15 |
| DeltaNet projections | 10.33 | 8.96 |
| gated reads | 7.92 | 7.28 |
| shared expert | 3.01 | 3.01 |
| attention projections | 3.15 | 2.69 |
| LM head | 2.61 | 2.61 |
| indexer (`idx_score` + `idx_top`) | 2.40 | 2.31 |
| router | 1.86 | 1.82 |
| DeltaNet scan | 1.57 | 1.54 |
| total (GPU) | 47.06 | 65.66 |

- **The attention explains the drop: +22.8 ms of +18.6 ms total** from 256 to 2048 (the other parts move by ±1.5 ms
  between the runs). **Finding:** above the expected 12 to 18 ms. `attend` runs one threadgroup of 256 per Q head
  (24 threadgroups on 32 GPU cores), each thread loops over all 2048 + 3 cells for its V column.
- **Finding:** the indexer stays at ~2.3 ms, below the expected 3 to 6.
- **The 8192 column is invalid:** at 8192 the profiled prefill varies between the two runs by more than the 14 decode
  passes it should cancel (negative differences). From the speeds: +5.6 ms per pass from 2048 to 8192, which fits
  the indexer's growth.

Next by this profile: decode attention split over the cells (flash-decoding: many threadgroups per head, each a
slice of the cells, then one combine), expected from ~26 ms to a few ms at 2048, decode ~13.8 → ~20 tok/s.
