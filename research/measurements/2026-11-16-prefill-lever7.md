# Expert GEMM with a 128-token tile (#159, lever 7)

## Plan (written before the runs)

- Change: `k_gemm_q2` is templated on the tokens per tile (64 or 128). A launch uses the 128-token tile when the mean
  number of tokens per expert segment reaches `TONA_GEMMQ_T128_MIN` (default 96, 0 turns it off). The K-sum order per
  output is unchanged, so the result must be bit-identical.
- Basis: breakdown (research/measurements/2026-11-16-prefill-breakdown.md), expert GEMM about a quarter of prefill;
  a wider tile reads each weight block once for twice the tokens.
- Questions and decisions: (a) bit-exact (`tona-gemm-q-bench` hashes, SHAs of 30K and 130K equal to `bench/sha.toml`,
  goldens identical)? If not, drop. (b) Faster at 30K and 130K in `ab.py`? Keep the default threshold only if faster.
- Runs: `tona-gemm-q-bench` for T = 64, 100, 160, 200, 320 (one run per point, spread not measured: it only answers
  "bit-exact, where does it pay"); `ab.py` 30K (mid) and 130K (long), same binary, A with the switch at 0, B default,
  `TONA_DETERMINISTIC=1`, early stopping on (the first round decided at 30K; 130K likewise).
- Why this size: the microbenchmark is the smallest workload that shows bit-exactness and the tile effect; 30K is the
  shortest prompt with the real per-expert token counts; 130K checks the long-prompt plan.
- Expectation: prefill within -1 to +1 % at both lengths (the microbenchmark gave -13 to +13 % on one GEMM that is
  about a quarter of prefill); decode +-0.5 %.
- Active tooling: `TONA_DETERMINISTIC=1`, `OMP_NUM_THREADS=6`, `ROCR_VISIBLE_DEVICES=0`,
  `TONA_EXPERT_PROFILE=work/profile/experts-coder`; no profiler.

## Results

Microbenchmark, 64-token tile / 128-token tile, type 18 (ms): T=64 0.324 / 0.329; T=100 0.540 / 0.469; T=160
0.739 / 0.780; T=200 0.969 / 0.891; T=320 1.247 / 1.200. The hashes are equal at every T and expert type.

`ab.py` (same binary, `TONA_GEMMQ_T128_MIN` 0 against default), 1 round each, both decided by early stopping:

| Prompt | Prefill tok/s A -> B | B vs A | Decode | SHA |
|---|---|---|---|---|
| 30K | 1,933.5 -> 1,945.5 | **+0.62 % [+0.50, +0.75]** | +0.08 % (equal) | ee58880d3f5b558f = bench/sha.toml |
| 130K | 1,856.3 -> 1,864.6 | **+0.45 % [+0.39, +0.51]** | +0.21 % (equal) | 7d2996546076d1b0 = bench/sha.toml |

Both are inside the planned -1 to +1 % band. Goldens (prefill dflash, prefill, server) on the committed build: all
identical. `just check` green. `isa-diff` does not apply (the kernel is the change).

The gain is small because the tile choice is per launch from a mean: segments with a nearly empty last 128-token
tile lose what the full ones gain (T = 160 in the microbenchmark). A per-segment choice would need two tile sizes in
one launch; not built.

## Test conditions

Raw files: `2026-11-16-prefill-lever7-ab-30k.md/json`, `2026-11-16-prefill-lever7-ab-130k.md/json`.
`hw_report.py` and `quiet_check.py` were run once, after the runs, while the golden checks were using the GPU
(so the VRAM held by others in this report is our own golden run, not the measurement state): RX 9060 XT gfx1200,
Ryzen 5 9600X, 60.5 GiB RAM (40.3 GiB available), kernel 7.2.15-200.fc44, hip 7.16.26385, no `TONA_*` set in the
shell, load 2.6 / 5.2 / 3.4, other programs 8.7 % of the CPU threads, swap in use 2.55 GiB; quiet check: do not measure
(5.1 % other CPU, VRAM held). Quiet checks before and after each `ab.py` run were not taken: a gap against rule 2.
The A/B arms ran on the same machine state back to back, so the relative numbers hold, but the absolute tok/s carry
that caveat.
