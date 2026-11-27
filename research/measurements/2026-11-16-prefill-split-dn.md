# Prefill: DeltaNet kernels in pieces below 1 ms (#159, lever 5, part 3)

## Plan (written before the runs)

- Basis: research/measurements/2026-11-16-prefill-idle-gaps.md (a kernel of 1 ms and longer is followed by a stall of about
  280 us; the 1 ms is the KFD's MES gang quantum) and the 30K profile after the row-wise split (`sp30`): gaps after `k_dn_wave`
  249 (70 ms, kernel 5.4 ms) and `k_dn_conv_l2` 183 (51 ms, kernel 1.6 ms) of 15.5 s: 0.78 %. Both stream or run sequentially
  (the delta rule is latency-bound: 1.05 us per token), so a piece costs little; the GEMMs, which are compute-bound, did not
  pay (research/measurements/2026-11-16-prefill-split-gemm.md).
- Change: `launch_dn_wave` cuts the token range into pieces of at most `TONA_PF_SPLIT_DN` tokens (default 768: 0.8 ms; 0 = one
  launch; also off with `TONA_PF_SPLIT_ROWS=0`); the state is written to memory at the end of a piece and read again at the
  start of the next, in fp32, so the recurrence is the same arithmetic in the same order (host only, kernel unchanged:
  `isa-diff` identical). `launch_dn_conv_l2` cuts into the pieces of `TONA_PF_SPLIT_ROWS` rows (default 2048) with a new
  token offset `t0` and count `m` in the arguments (the causal conv reads earlier tokens of the chunk, so pointer shifting
  does not work; the kernel indexes `t = t0 + u / per`; `isa-diff`: only this kernel differs, by the argument offsets).
  `k_qsa_prep` (74 gaps, 21 ms, 0.14 %) is not done: its pool unit reads earlier tokens of the chunk as well and the gain is small.
- Questions and decisions: (a) tokens identical (`TONA_DETERMINISTIC=1` SHAs equal to `bench/sha.toml`, goldens identical)?
  If not, drop. (b) Faster at 30K and 130K? Keep only if both are faster beyond noise.
- Runs: `scripts/ab.py` A = the build before this change (`work/build/w7a`: split rows on), B = `work/build/w7`, mid (30K) and
  long (130K), `TONA_DETERMINISTIC=1`, `--primary prefill`, early stopping on; `hw_report.py` and `quiet_check.py` before and
  after each run; the desktop's VRAM and the load are named as conditions.
- Why this size: the smallest runs that show the effect: 30K (0.78 % of the window in these gaps), 130K as the long check.
- Expectation (source: the `sp30` profile): 121 ms of 15.5 s = 0.78 % at 30K, minus extra launches and the state reload
  (about 10 %): prefill +0.6 % at 30K, +0.6 % at 130K (the same kernels per layer), tolerance +-0.4 points; decode +-0.5 %.
- Active tooling: `TONA_DETERMINISTIC=1`, `OMP_NUM_THREADS=6`, `ROCR_VISIBLE_DEVICES=0`,
  `TONA_EXPERT_PROFILE=work/profile/experts-coder.bin`; no profiler.

## Run 1 result (dn_wave in 768-token pieces and conv_l2 in 2048-row pieces, against the build before): mixed

- `ab.py` (A `w7a`, B `w7`), SHAs equal to bench/sha.toml at both lengths: 30K prefill 1,917.6 -> 1,926.5 tok/s, **+0.46 %
  [+0.34, +0.59]** (1 round, B faster); 130K 1,798.7 -> 1,788.4 tok/s, **-0.57 % [-0.61, -0.53]** (2 rounds, B slower).
  Decode: 30K -2.26 % [-3.12, -1.39] in the single round (undecided; the decode path does not run these kernels; A itself read
  69.3 against 70.1 to 70.4 in earlier runs), 130K +0.09 % (equal).
- Test conditions: the desktop held 2.47 to 2.54 GiB of VRAM (1.5 GiB in the earlier runs of today), load average 1.1 to 1.2,
  other CPU use 0.3 to 0.5 %, 2.94 GiB swap in use, 2 pages swapped in during the 30K quiet check; `hw_after_130000` shows
  9.25 GiB held and 49 % busy (the GPU was not yet idle after the server stopped). The larger VRAM share is why every
  absolute rate is 3 to 5 % below the earlier runs (a smaller loan limit changes the plan); A and B share the plan.
- Reading: the state of the delta rule is `nv * 128 * 128` floats per layer (megabytes): every extra piece of `k_dn_wave` reads
  and writes it again, about 6 pieces of 0.8 ms for a 5.4 ms kernel, which costs as much as the 280 us stall it removes. The
  kernel is latency-bound, not streaming, so the premise "streams like the row-wise kernels" does not hold for it. `k_dn_conv_l2`
  has no state and two pieces of 0.5 ms.
- Decision: drop the `k_dn_wave` split (code removed); test `k_dn_conv_l2` alone (run 2: A `w7a`, B the rebuilt `w7`, 30K and
  130K, expectation +0.3 % = 51 ms of 15.5 s, tolerance +-0.3 points, decode +-0.5 %; fresh `hw_report` and `quiet_check`
  before and after each run, desktop VRAM and load named).

## Run 2 result (k_dn_conv_l2 in pieces only, against the build before): small gain, kept

`ab.py` (A `work/build/w7a`: the build before, B `work/build/w7`: conv_l2 split; the `k_dn_wave` split is removed), early
stopping decided after round 1 at both lengths:

| Prompt | Prefill tok/s A -> B | B vs A | Decode | SHA |
|---|---|---|---|---|
| 30K | 1,918.9 -> 1,922.2 | **+0.17 % [+0.05, +0.29]** | +0.20 % (equal) | ee58880d3f5b558f = bench/sha.toml |
| 130K | 1,799.4 -> 1,806.8 | **+0.41 % [+0.36, +0.47]** | -0.39 % (equal) | 7d2996546076d1b0 = bench/sha.toml |

Both inside the planned 0 to +0.6 %. `isa-diff` against the build before: only `k_dn_conv_l2` differs (kernel argument size
64 -> 72 bytes, same register use). Goldens (prefill dflash, prefill, server): 38 of 38 identical. `just check` green.

At 130K the expert-cache counts of the two arms differ (A 74,113 hits / 9,838 misses, B 73,884 / 10,067) although the tokens
are identical (SHA equal); the earlier runs of today had equal counts. The counters come from `/status` and depend on the
timing of the cache, not on the arithmetic; noted, not followed up.

Test conditions: raw files `-ab-30k.md/json`, `-ab-130k.md/json`; `hw_report` and `quiet_check` before and after each run
(`/var/tmp/claude/w159/split5/`): the desktop held 2.54 to 2.57 GiB of VRAM (warning on every check; 1.5 GiB in the earlier
runs of today, hence absolute rates 3 to 5 % lower than before), load average 0.9 to 1.1, other CPU use 0.2 to 2.7 %, 2.94 GiB
swap in use, GPU 0 % busy before the runs. A and B ran back to back on the same state.
