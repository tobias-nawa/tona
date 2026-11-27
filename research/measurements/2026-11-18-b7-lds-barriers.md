# B7 (#163): LDS-only barriers instead of `__syncthreads()`

Host Ryzen 9 9950X, RX 9060 XT gfx1200, Coder IQ1_M. Base build `work/build/w163` (main 1eaa6d2 plus A2, A3 and the host
trace), change build `work/build/w163b7` (the same plus `gpu/lds_sync.h` and its 77 sites).

## What changes and why

`__syncthreads()` is a release fence + `s_barrier` + acquire fence over all address spaces. In the default WGP mode the
release waits for every wave's in-flight global loads and the acquire runs `global_inv scope:SCOPE_SE`, which drops L0.
`lds_sync()` fences only the local address space: `s_wait_dscnt 0` and the barrier. Register prefetches of the next tile
or row stay in flight across it. Source of the mechanism and a compiled probe for gfx1200:
`/var/tmp/claude/r9700/features.md` section 2.4 (probe: with `__syncthreads()` every barrier is
`s_wait_loadcnt_dscnt 0x0, s_barrier_signal -1, s_barrier_wait -1, global_inv scope:SCOPE_SE`; with the local fences
only `s_wait_dscnt 0x0` and the barrier).

Sites converted (77 of 128): megakernel op headers `mega_op_{common,norm,resnorm,moe,mtp,matvec,dn,qsa,attn,dattn,
fattn,route_sig}.h` (41), `fattn_tile.h` (5), `prefill_attn.hip` (12), `gemm_f16.hip` (8), `gemm_fp8.hip` (6),
`gemm_q.hip` (4), `gemm_dequant.hip` (1). Audit rule: after the barrier no thread reads global memory that another
thread of the block wrote before it. All converted sites move data global to registers to LDS (each thread consumes its
own loads) or reduce through LDS. Kept as `__syncthreads()`: the mailbox and dependency waits (`mega_op_wait.h`: thread 0
acquires, the other threads read after the barrier), the grid barrier (`matvec_barrier.h`), `mega_op_mtp_dyn.h` (writes
`out` to global memory and reads it back after a barrier), `mega_op_topk.h`, the profile stamp in `mega_kernel.h`, the
one-time table copies, and the small prefill kernels (`ops.hip`, `prefill_hc/dn/qsa/topk`, `topk_dev.h`) not audited yet.

## Plan (written before the runs)

**Question.** Does the LDS-only barrier change decode or prefill speed, and is it bit-exact?

**Checks and expectation.**
1. R0 `just check`, R1 unit tests: green.
2. R2 `scripts/isa-diff.py work/build/w163 work/build/w163b7`: the kernels that contain a converted site change, the
   others (matvec, experts, DeltaNet recurrence, MoE GEMM of the prefill variants without a barrier) are identical. The
   instruction difference per kernel is only barrier sequences (global_inv and loadcnt waits gone, dscnt waits kept);
   VGPR and spill counts may move.
3. R3 goldens (`scripts/gpu-status.sh --own --keep-going --base origin/main --build work/build/w163b7`): all identical.
4. R4 bit-exact `TONA_DETERMINISTIC=1`, `mid`: identical SHA between base and change (ab.py checks it).
5. R5 speed, `scripts/ab.py --a w163 --b w163b7 --workload short,set,mid`: decode (short, set) and prefill (mid).
   Expectation: decode -0.5 to +2.0 %, prefill 0 to +3 % (the GEMM main loops already consume their prefetch before the
   barrier, so the effect there should be small; attention and the decode op headers wait for loads at today's
   barriers). A decode loss above 0.5 % is a finding: the code generation of `k_mega` reacts to small changes
   (research/measurements/2026-10-28-mega-spills.md), so the spill counts of `k_mega` are read first.

**Why this size.** The question is one build against one build on two rates. Noise from bench/noise.toml: decode@2000
0.23 %, decode@2000:write 0.61 %, prefill@30000 0.04 %, `set` has 0.07 % (bench-set code). A 1 % decode effect needs 3 to
4 rounds (`ab.py` stops early); `--max-rounds 6`. `mid` is the shortest prompt where prefill runs full groups (30K);
130K only if mid shows an effect that attention would change.

**Active tooling.** None: no `TONA_MEGA_PROF`, no host trace, no `TONA_DETERMINISTIC` except in R4.

**Long-context run (R4, required because the change touches `fattn_tile.h` and `prefill_attn.hip`).**
`scripts/ab.py --a w163 --b w163b7 --env TONA_DETERMINISTIC=1 --workload mid,long --fixed-rounds --max-rounds 1`: one
round, both arms, 30K and 130K prompts, 200 decode tokens, MTP. Expectation: the output SHAs of both arms equal
`bench/sha.toml` (30000 `ee58880d3f5b558f`, 130000 `7d2996546076d1b0`); speed is not the question here (deterministic
mode switches the wide path off), the equality is. Why this size: one round is the recipe (R4), the SHA does not have a
spread; 130K is the length where the attention tiles run for minutes, 30K the full prefill groups.

## Results

Checks: `just check` green incl. clang-tidy (34 files); goldens identical (gpu, cpu plan, prefill, prefill dflash, server;
47 lines identical, 0 differing); ab.py output SHAs equal between A and B for 2000, 2000:write and 30000 (without
`TONA_DETERMINISTIC`, same text). The long-context run (R4, 30K and 130K, deterministic): see the end of this file.

Speed, `scripts/ab.py w163 against w163b7`, 6 rounds (full report `research/measurements/2026-11-18-b7-ab.md`, test
conditions in `research/measurements/2026-11-17-b1-cycle-breakdown.md` apply: same machine, 9950X):

| Metric (rate) | B vs A, mean [95 % CI] | Expected | Verdict |
|---|---:|---|---|
| prefill@30000 (prefill) | **+0.96 % [+0.46, +1.46]** | 0 to +3 % | B faster, decided |
| prefill@2000 (prefill) | +1.15 % [-0.52, +2.82] | 0 to +3 % | undecided, rounds 2 to 6 all positive |
| prefill@2000:write (prefill) | +1.14 % [-1.05, +3.3] | 0 to +3 % | undecided |
| decode@2000 (decode) | +0.31 % [-1.72, +2.34] | -0.5 to +2 % | undecided |
| decode@2000:write (decode) | +0.53 % [-0.36, +1.41] | -0.5 to +2 % | undecided |
| decode@30000 (decode) | +0.31 % [-0.02, +0.64] | -0.5 to +2 % | undecided, B ahead in 6 of 6 rounds |
| decode-code (decode, bench-set) | +0.75 % [-0.99, +2.48] | -0.5 to +2 % | undecided |
| decode-text (decode, bench-set) | +0.53 % [-1.37, +2.43] | -0.5 to +2 % | undecided |

Reading: prefill gains about 1 % (decided at 30K); decode gains 0.3 to 0.75 % in the mean of every metric, none decided
at 6 rounds (the rounds with a different order show the noise of the first rounds of a server start, the spread of the
decode@2000 rounds is larger than the noise table says). No metric is below the plan. Nothing got slower. Why this is
small: the prefill GEMM loops already consume their register prefetch before the barrier; what is saved is the L0
invalidate and the wait in the attention and op-header sites.

Note on the plan headers of the ab.py reports: "build commit" is `git describe` of the source tree at the time ab.py runs
(`ab_plan.build_commit`), so both arms show the source HEAD, not the commit that was built. The binaries: A `w163`
`tona-server` was built 2026-11-17 21:45 (before B7, committed 22:06), B `w163b7` at 23:02.

## Kernel identity (R2)

`scripts/isa-diff.py work/build/w163 work/build/w163b7`: 376 functions in each build, 140 differ in code, none added or
removed; the differing ones are the kernels that contain a converted site (all `k_mega` instantiations and the op
functions behind them, `k_attn_wmma`, `k_fattn_prefill`, `k_gemm_q2`, `k_dequant_fp8`, `k_pick`). Register and spill
counts are not listed as changed by the tool. The matvec, expert, DeltaNet-wave and other kernels without a converted
site are identical.

## Long-context run (R4)

`ab.py --a w163 --b w163b7 --env TONA_DETERMINISTIC=1 --workload mid,long`, one measured round after the warm-up (the
tool ran 1 round; its minimum of 2 includes the warm-up). Result: both arms reproduce `bench/sha.toml`, stable between
runs, A = B:

| Request | SHA A | SHA B | Reference |
|---|---|---|---|
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | ee58880d3f5b558f |
| 130000 | 7d2996546076d1b0 | 7d2996546076d1b0 | 7d2996546076d1b0 |

B7 is bit-exact at 30K and 130K context (prefill attention, fattn tile and the megakernel decode with MTP).
