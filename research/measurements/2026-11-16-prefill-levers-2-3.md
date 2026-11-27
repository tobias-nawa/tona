# Prefill: slots stay borrowed between groups, dense weights unpacked once per layer (#159, levers 3 and 2)

## Plan (written before the first run)

- Changes (host code only, no kernel changes): (3) `PrefillRun::finish()` returns the borrowed expert slots only after the
  last group of a prompt (`TONA_PF_LOAN_KEEP=0` restores the copy back after every group); a later group borrows only the
  layers it needs beyond the ones still on loan. (2) The unpacked copies of the dense weights of a layer (fp8 for qkv, gate,
  q/k/v, shared expert; f16 for ssm_out, attn_output, hc down/up, router, indexer) stay in a 128 MB area of the loan for
  all chunks of the layer (`TONA_PF_WCACHE=0` unpacks per chunk as before). The GEMM calls (slices, tile choice) are
  unchanged, so every output value is computed from the same bytes.
- Basis: research/measurements/2026-11-16-prefill-breakdown.md: 9 group boundaries at 130K cost 129 to 348 ms idle each
  (2.05 s of gaps, about 1.2 to 1.5 s of them the copy back); dense unpack and input conversion 1.9 s of 72.9 s at 130K,
  of which 3/4 (4 chunks per group) are repeated work.
- Questions and decisions: (a) are the tokens identical (`TONA_DETERMINISTIC=1`, SHAs of 30K and 130K equal to
  `bench/sha.toml`, goldens identical)? If not, the change is dropped or fixed. (b) How much faster is prefill at 30K and
  130K? Keep both levers if each is faster beyond noise; a lever that is equal or slower is removed.
- Runs: (R4/R5 of docs/contribute/measurement.md) `scripts/ab.py --workload mid,long`, `--env TONA_DETERMINISTIC=1` (prefill is not
  affected by the deterministic cap on drafts; it is needed for the SHAs), `--primary prefill`.
  Run 1: A = build of main (`work/build/engine` of the main checkout, built 2026-11-16 11:18), B = `work/build/w159` of this
  branch: both levers together. Run 2: A = `w159` with `TONA_PF_WCACHE=0 TONA_PF_LOAN_KEEP=0`... (see results) to split
  the two levers: A = `w159` with `TONA_PF_WCACHE=0`, B = `w159` (lever 2 alone). Rounds as `ab.py` decides (known noise
  of mid and long in `bench/noise.toml`, 1 to 6 rounds, about 4 min per round).
- Expectation (source: breakdown file, levers 2 and 3): prefill@30000 +2.7 % (lever 3 +0.85, lever 2 +1.8), prefill@130000
  +3.7 % (lever 3 +1.7 to 2.1, lever 2 +2.0); tolerance +-1.5 points. Decode (**decode** tok/s) unchanged (+-0.5 %).
  Lever 2 alone (run 2): +1.8 % at 30K, +2.0 % at 130K.
- Goldens (`scripts/golden.sh compare`, prefill with dflash, prefill, `golden-server.sh compare`) on `w159`, kind gpu.
- Active tooling: `TONA_DETERMINISTIC=1` (both arms), `OMP_NUM_THREADS=6`, `ROCR_VISIBLE_DEVICES=0`,
  `TONA_EXPERT_PROFILE=work/profile/experts-coder.bin` (ab.py); no profiler, no `TONA_PF_DEBUG`.
- Quiet check: before and after each run (the machine was not quiet earlier today, other CPU load 67 to 76 %; the
  results below say what it was).

## Changes after the plan (before the final runs)

- The weight cache is not a fixed size: `Mega::pf_wcache_bytes()` computes it from the tensor shapes and the route of every dense
  weight (FP8 or f16, in slices as the GEMM does it, 256-byte aligned), the largest layer decides (107 MB for this model); `wc_get`
  throws if a layer needs more (the debug line `dense weight cache peak 107 of 107 MB` shows it is exact). The size is part of
  `pf_layered_need`.
- First run (`...-ab-run1.md`, 6 rounds): the 30K prefill got **no** faster (+0.27 %, undecided) and 130K +1.61 %. Cause found with
  `TONA_PF_DEBUG=1`: the 107 MB made the 30K plan chunk 4096 / group 16384 instead of chunk 5120 / group 15360, which cost what the
  cache saved. Fix: `choose_plan` uses the cache only when the plan is the same one as without it (30K: no cache, 130K: cache).
  So at 30K only lever 3 acts, at 130K both.

## Results

Test conditions: `tools/hw_report.py` and `tools/quiet_check.py` as in the breakdown file (quiet check: warning, load average 3.4 and
1.49 GiB VRAM held by others before the runs; the runs held the GPU lock alone). Build `work/build/w159` of branch 159-prefill
(base e4a4f41 + a7f0f5d), arm A = the same binary with `TONA_PF_WCACHE=0 TONA_PF_LOAN_KEEP=0` (= the old behaviour, so no other
change is in the comparison), arm B = default; `TONA_DETERMINISTIC=1`, prefill only decides.

**Bit-exactness:** SHAs of the 30K and 130K requests equal `bench/sha.toml` in every round of both arms (30000 ee58880d3f5b558f,
130000 7d2996546076d1b0); expert hits and misses equal; `scripts/isa-diff.py` main build against `w159`: `all kernels identical`;
goldens decode, CPU plan, prefill with dflash identical (prefill and server see below).

**Speed** (`scripts/ab.py`, `--workload mid,long`, 4 rounds, `...-ab.md`), **prefill** tok/s:

| Request | A (old) | B (new) | B against A | Decision |
|---|---:|---:|---:|---|
| 30K | 1,917.0 (1,916.9 to 1,917.2) | 1,927.4 (1,924.5 to 1,928.0) | **+0.51 % [+0.42, +0.60]** | B faster after round 1 (lever 3 only; plan expected +0.3 to +2) |
| 130K | 1,793.5 (1,791.3 to 1,794.5) | 1,825.0 (1,822.1 to 1,825.5) | **+1.74 % [+1.64, +1.84]** | B faster after round 4 (below the +2.2 to +5.2 % expected) |

Decode (**decode** tok/s) did not move beyond noise (30K +0.78 %, 130K +0.15 %, both open with intervals around 0).
Lever 2 alone at 130K (A with only `TONA_PF_WCACHE=0`, `...-ab-lever2.md`, 1 round, known noise): **+1.24 %**; so lever 3 gives about
+0.5 % at 130K (plan expected +1.7 to +2.1 %) and +0.5 % at 30K (expected +0.85 %).

**Why below the plan:** ungeprueft. The gains are 0.5 % (lever 3, 0.36 s of 72 s at 130K, against 1.2 s expected from the 8 boundaries) and 1.2 % (lever 2, against 2.0 %); the plan took the kernel and gap times of the profiled run as the saving, and a part of them evidently is not on the critical path or is paid again elsewhere. A new profile of the new build would show which.

Final build (after the gate refactor of plan_fits and borrow_slots): `...-ab-final.md`, 2 rounds, 30K +0.56 % [+0.49, +0.64], 130K +1.71 % [+1.67, +1.75], SHAs equal to bench/sha.toml; goldens decode, CPU plan, prefill dflash, prefill and server (36 cases) identical.
