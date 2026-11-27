# Prefill lever 7 (expert GEMM, 128-token tile): state at pause (#159)

Status: paused by the maintainer on 2026-11-16 before any lever 7 measurement. No result yet.

## Plan (not yet run)

- Scope: `tona-gemm-q-bench` from the lever 7 build, T = 64, 100, 160, 200, 320 tokens per expert segment,
  with the switch `TONA_GEMMQ_T128_MIN` set to 0 (old 64-token tile) against 1 (128-token tile).
- Expectation: identical output hash (the K-sum order per output is unchanged), fewer weight reads per token
  for the wide tile, so a faster expert GEMM from about T >= 96; tolerance of the speed gain not yet estimated
  (no prior measurement; the result sets the `TONA_GEMMQ_T128_MIN` default).
- Why this size: the micro benchmark is the smallest workload that answers "is the wide tile bit-exact and
  where does it pay off". Only after that an `ab.py` run at 30K or 130K.
- Decision per result: hashes differ -> drop the change. Faster from some T -> set the default threshold
  there and run `ab.py`, goldens, isa-diff. Not faster -> drop the change.

## Code state

- Branch `159-lever1`, uncommitted change in `engine/gpu/gemm_q.hip`: `k_gemm_q2<TY, TT>` templated on the
  token tile (64 or 128); `collect_q` picks 128 when the mean tokens per segment reaches
  `TONA_GEMMQ_T128_MIN` (default 96, 0 = off).
- Built in `work/build/w7` (links, no errors). Not yet run: isa-diff, goldens, unit tests, `just fmt`,
  `just check`. Not committed.

## Open findings from the earlier runs

- The plan changed after the clock-ramp merge (loan limit 4195 MB): default chunk 6144 at 30K, 5120 at
  130K. At 130K a chunk of 5120 is 1.13 % faster than 4096 (6 rounds, SHAs equal). At 30K a chunk of 5120 is
  0.56 % faster than 6144 (1 round, SHA equal). A default change to 5120 needs `ab.py` at 16K, 30K, 64K.
- `k_idx_score_wmma` per-call time rises with the chunk size (86 to 116 us at 130K); the cause is unexplained.
  The 30K profiles at chunk 4096, 5120, 6144 (`/var/tmp/claude/w159/c4096`, `c5120`, `c6144`) finished or are
  finishing; they are not analysed yet.
- Lever 1 (fp8 experts) prior art: no RDNA4 fp8 grouped MoE implementation found for gfx1200; vLLM supports
  RDNA4 fp8; VulkanForge reports +33 to 39 % MoE prefill from native fp8 WMMA (self-reported). Design: in-kernel
  unpack to e4m3 in LDS plus fp8 WMMA behind a switch, with the #106-style quality check. Not started.
- Lever 5 (idle gaps of 50 to 500 us, about 2.58 s at 130K): cause unknown, not host-bound.

## Next steps after the pause

1. Re-queue the `qbench` run (cancelled by the coordinator).
2. Decide the threshold, then `ab.py`, goldens, isa-diff, `just check`, commit.
3. Analyse the chunk profiles (CPU only), then the fp8 variant.

## Lever 7 microbenchmark (`tona-gemm-q-bench`, one run each, 32 experts per call, expert types 18/20/22/23/42)

- Output hashes are identical for the 64 and the 128-token tile at every T and type: bit-exact as designed.
- Time per call, 64-token tile / 128-token tile (ms, type 18; the other types move alike): T=64 0.324 / 0.329;
  T=100 0.540 / 0.469 (-13 %); T=160 0.739 / 0.780 (+6 %, other types up to +13 %); T=200 0.969 / 0.891 (-8 %);
  T=320 1.247 / 1.200 (-4 %).
- Reading: the wide tile pays where the last 128-token tile is well filled (100, 200, 320) and loses where it is
  mostly empty (160). The mean tokens per segment is a weak predictor; the real segment-size distribution decides.
  Spread not measured (the microbenchmark only answers "bit-exact, and where does it pay"). Decision by `ab.py`
  at 30K (`ab-lever7`, queued): keep the default threshold 96 only if prefill is faster, else drop the change or
  choose the tile per segment.

## Update after the pause was lifted (idx regression, step 1)

- Chunk profiles at 30K (kernel trace, same prompt): `k_idx_score_wmma` is 34.9, 34.1, 33.7 us per call at chunk
  4096, 5120, 6144 (10452 calls each), so the per-call time does **not** grow with the chunk size. The earlier
  explanation "the plan (chunk size) causes it" is wrong. Window 15.81 / 15.52 / 15.46 s.
- Earlier build at 30K (chunk 5120, group 15360, loan limit 3936 MB, no weight cache): 26.9 us per call. Now with the
  same chunk and group (loan limit 4195 MB, weight cache on): 34.1 us, +27 %. Of all kernels with equal call counts,
  only `k_idx_score_wmma` changed (the next is `k_gemm8b<256>` +6 %), and the window got shorter overall.
- No engine change between the two builds apart from levers 2 and 3. Candidates left: the weight cache and the
  kept loans change the memory layout; the loan limit changed. Test queued: the same 30K profile with the
  weight cache and loan keeping switched off (`off30`).

### Result of the `off30` profile (30K, chunk 5120 / group 15360, weight cache and loan keeping off)

- `k_idx_score_wmma`: 33.7 us per call, the same as with the switches on (34.1) and against 26.9 in the first
  profile (run1). The 30K window is 15.87 s (switches on 15.52 s, run1 15.80 s).
- So levers 2 and 3 do not cause the slower indexer, and neither does the chunk size or the weight-cache layout.
  The engine code of run1 and of this build is the same except the two switched-off levers. What remains is the
  state of the machine at the time of the profile (clock, memory state; run1 has no clock log). Effect on the total:
  the window did not get longer (+0.07 s for the indexer at 30K, within the spread of the window), so there is no
  lever here; the 130K "regression" of 1.4 s was part of a faster window overall (72.00 s against 72.93 s).
- Status: indexer regression closed as environmental and not actionable. A fresh profile of the old build on the
  current machine state would separate "build" from "machine"; not worth the GPU time because the window effect is nil.
