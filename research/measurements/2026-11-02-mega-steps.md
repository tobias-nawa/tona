# Megakernel decode: the small steps close to the roof (#101, includes #37)

Date: 2026-11-02. Branch `101-mega-steps` (from main 6be2c3f). Build `work/build/w101`.

## Plan: step profile (block 1)

Question: which steps of a decode pass are still far below the byte roof, and how much time each costs, measured
fresh on this build instead of the 2026-10-25 numbers.

Scope: two `ie-gpu-gen` runs on the big model (coder IQ1_M), all under `/var/tmp/claude/gpu.lock`, one block, no build
during the block, `IE_DIAG=1`.

1. `IE_MEGA_PROF=1`, prompt "csv" of `scripts/bench-decode.sh` (code), 200 tokens, no drafter (`--mega`, every pass
   is B=1), plus `IE_BAR_PROF=<file>` for the waiting times at the barriers. About 1 min each (load 17 s, 200 passes
   of ~25 ms plus the first-pass work). Plain B=1 is the unit the ticket table uses. The B=1 pass is also what the
   production path runs for the verify of one token.
2. The same with the MTP drafter (`--mtp`), `IE_MEGA_PROF=1` only (profile mean over passes of width 1 to 4), to see
   the mix of the production path.

Rounds: one per variant. The step shares are deterministic to within a few percent (the 13K profile of
`2026-11-02-golden-loads.md` repeats within 1 pct. point); rates are not claimed from this block.

Expectation (source: ticket table from `2026-10-25-barriers.md`, tolerance +-30 %): fraction of the byte roof router
35 %, dn.out 54 %, gr.up/down 59 %, qsa.out 55 %; loss router 0.4 ms, dn.out 1.2 ms, gr.up/down 1.5 ms, qsa.out
0.4 ms. host.wait / host.y 735 + 443 us. gr.down 591 us. Pass B=1 about 25 ms (23-25 ms), profiled mean about 24.9 ms.
Anything outside +-30 % is a finding.

Active tooling: `IE_MEGA_PROF=1`, `IE_BAR_PROF` (time stamps per step and block, costs a stamp per step),
`IE_DIAG=1`, `OMP_NUM_THREADS=6`, `IE_EXPERT_PROFILE=work/profile/experts-coder.bin`, `IE_MEGA_INFO=1`. No profiler.
No rate from these runs goes into BENCHMARKS.md.

## Results: step profile (block 1)

Runs: two `ie-gpu-gen --mega` runs, prompt csv (code), 200 tokens, no drafter (every pass B=1), 32 blocks, 96 expert
slots per layer, 25.3 GB pinned. The second run was meant to use the MTP drafter, but `ie-gpu-gen` takes no `--mtp`
flag for this path; it ran without a drafter again and so gives the spread of the profile. Pass time:
23.85 ms and 23.71 ms profiled mean (plan: about 24.9; inside the tolerance). Decode 41.1 / 41.7 tok/s decode at
B=1 (one token per pass, no drafter; not a production rate).

Bytes per step from the tensor sizes of the model (`ie-gguf`), summed over the layers; GB/s = bytes / profiled step
time; roof = 315 GB/s (docs/backends/reference-platform.md). gr.* are Q8_0 (`hc8_`, 3.48 MB per matrix) plus the bf16 inject.

| Step | ms per pass (run 1 / 2) | MB per pass | GB/s | Share of roof | Loss to roof (ms) | Plan (10-25) |
|---|---:|---:|---:|---:|---:|---|
| head.out | 1.651 / 1.595 | 521 | 316 | 100 % | 0.00 | - |
| dn.in | 3.939 / 3.937 | 1,108 | 281 | 89 % | 0.42 | - |
| qsa.in | 1.220 / 1.220 | 326 | 267 | 85 % | 0.18 | - |
| qsa.out | 0.612 / 0.611 | 151 | 246 | 78 % | 0.13 | 55 %, 0.4 ms |
| dn.out | 1.855 / 1.851 | 444 | 239 | 76 % | 0.45 | 54 %, 1.2 ms |
| gr.up | 1.687 / 1.678 | 338 | 200 | 64 % | 0.61 | 59 %, 1.5 ms (both) |
| gr.down | 1.885 / 1.885 | 345 | 183 | 58 % | 0.79 | same |
| router | 0.630 / 0.624 | 63 | 100 | 32 % | 0.43 | 35 %, 0.4 ms |

Findings against the plan: router, gr.up and gr.down are as expected (within 30 %). dn.out (76 % against 54 %) and
qsa.out (78 % against 55 %) are outside the tolerance: they improved since 10-25 and have 0.45 and 0.13 ms left,
not 1.2 and 0.4. The ticket's 1 to 2 ms is therefore not reachable from these four steps alone (their loss to the
roof adds up to 2.4 ms, of which a realistic part is half).

Other steps (ms per pass, run 1): moe.gate_up 3.01, moe.down 2.24, dn.recur 0.92, combine 0.90 (97x, 7.4 us each),
host.wait 0.74 (48x, 15 us each), router 0.63, moe.shared 0.42, dn.conv 0.40, mix 0.34, moe.sum 0.28, moe.act 0.19,
host.y 0.14, moe.copy 0.11. Barriers (`IE_BAR_PROF`, 212 passes, 23.72 ms): latency 0.95 ms, imbalance 2.20 ms. Wait
for the slowest block per pass, us: gr.down 413, moe.gate_up 212, moe.down 200, gr.up 173, dn.recur 157, moe.sum 150,
mix 132.

Per block (us per step, mean over instances): gr.down blocks 0-21 15-18 us, blocks 22-31 9 us (320 rows on 20
blocks plus the inject on two); gr.up 13.4-15.9 us (10,240 rows with L = 2 lanes = 640 units on 512 waves, the first
128 waves do two); router 11.0-11.6 us (256 rows of bf16 on 256 of 512 waves); dn.out 48-50 us (all blocks).

CPU side (extra run with the DFlash drafter, width 4 in 51 of 52 passes, 200 tokens, step profile on): selection
0.58 ms per pass (12 us per layer at 4 tokens), megakernel 36.0 ms per pass. At width 1 the selection is smaller, so
most of the 15 us of host.wait at B=1 is the GPU-CPU-GPU round trip, not the selection. Expectation for lever 1 is
lowered accordingly (about -0.1 ms at B=1, more at width 4).

## Lever 1: host side of the router (softmax and top-k in one pass)

Plan: `route_softmax`/`route_top` (gpu/route_rank.h) replace iota + partial_sort over 256 indices; the probabilities
are computed as before (same expf, same double sum and division), only the ranking changes. Expectation: CPU
selection per pass -0.03 to -0.1 ms at B=1 (source: 0.175 ms per pass before at B=1, 3.6 us per layer, of which about
3 us were the ranking in a first guess). Tooling: `--batch-bench 300` with `IE_BENCH_NB`, step profile off,
`IE_DIAG=1`; a temporary timer around the sections of `ExpertSelect::run` (not committed).

Result (one run per arm, CPU time is deterministic to a few percent): CPU selection per pass at B=1 0.175 -> 0.151
and 0.157 ms (two runs after), at B=2 0.320 -> 0.279 ms. Pass time 23.76 -> 23.74 ms (inside the noise, as
expected: -0.02 ms). Timer inside `run` at B=1 after the change: ranking 1.7 us (the 256 expf and the double
divisions dominate; they stay, since another expf would change bits), union 0.2 us, plan_cpu 1.3 us, together 3.2 us
per layer. Of the 15 us per layer that host.wait lasts, 3.2 us are the selection; the rest is the GPU-CPU-GPU round
trip, which the CPU code cannot shorten. Result: a measured, deterministic CPU gain of 14 percent of the selection,
about -0.02 ms per pass of wall time at B=1, too small for `ab.py` to decide on its own; it is kept (bit-identical,
unit test `test_route_rank` against the old code with ties and extremes, mini goldens identical) and counted in the
final A/B of all levers.

## Lever 2: combine, first element requested early

Plan: in `op_combine_norm` request R, y and the norm gain of the thread's first element before the first barrier of
the step. Expectation: combine 0.89 ms -> 0.7 to 0.8 ms per pass at B=1 (source: combine 7.4 us per step of which
two serial global loads of about 1 us each can move), pass -0.1 to -0.2 ms (+0.4 to +0.8 % decode). Variants tried
with the step profile (plain B=1, 200 tokens, step profile on, mean per pass): base 892 / 886 us; unroll 10 instead
of 4 in the sum loop 987 (worse, rejected); first element of R and y early 754 / 763; additionally the first sum batch
(float4 of R and y) early 978 / 979 (worse, rejected, probably the changed code of the loop); first element of R, y and
w early 770 / 741 (kept). Pass 23.72 (base) -> 23.60 / 23.56 ms. isa-diff against main: op_combine_norm,
op_qsa_prep (inlined helper) and k_mega<1/4/8> differ; resources of k_mega<1> and <4> unchanged, k_mega<8> spills
181 -> 192 VGPRs (the 8-token kernel is a measuring mode). Mini goldens identical.

`ab.py` (`work/ab/w101-p12.md`, base `w101-base` = main 6be2c3f against `w101` = levers 1 and 2, workload set at
`--ctx 32768` and one prompt of 13,000 tokens with 200 decode tokens, `IE_DIAG=1`, 6 rounds, alternating, decode the
deciding metric; 19 min):

| Metric | B vs A, mean | CI | Expected | Verdict |
|---|---:|---|---|---|
| decode tok/s, bench-set code | +0.28 % | [-0.38, +0.95] | +0.2 .. +1.2 % | within, open |
| decode tok/s, bench-set text | +0.39 % | [-0.32, +1.10] | +0.2 .. +1.2 % | within, open |
| decode tok/s after 13,000 tokens | +0.22 % | [-0.43, +0.87] | +0.2 .. +1.2 % | within, open |
| prefill tok/s at 13,000 tokens | +0.02 % | | -0.7 .. +0.7 % | within |

The text SHA of the 13,000-token request is equal in both arms, expert hits and misses are equal
in all six rounds (69,037 / 9,435). Verdict: no decided gain (CI contains 0), all means positive and in the expected
direction, the profile shows a real -0.14 ms of 23.7 ms (+0.6 % at B=1); the A/B at +0.2 to +0.4 % is the same
order. Kept as a bit-identical change; levers 1 and 2 are not separable here (lever 1 is -0.02 ms).

## Lever 3: fuse moe.act into moe.down and mix into the next matvec (not built)

Plan: before writing a fusion, measure the upper bound: the pass time with the barrier after every `mix` step (97) or
every `moe.act` step (48) dropped (`IE_MEGA_NOBAR_STEPS`, measuring mode, output meaningless), `IE_FAKE_ROUTE=1
IE_FAKE_EXPERTS=1` as in `2026-10-25-barriers.md`, `--batch-bench 300` at B=1 (`IE_BENCH_NB=1`), two rounds per
arm, 60 s between runs, `IE_DIAG=1`. A fusion can save at most what dropping the barrier saves, and it adds the work of
the fused step to the consumer (every block would compute the mix of all 2,560 values, or the activation of ~9 experts,
itself). Expectation: barrier latency 1.1 us plus straggle 1.4 us per `mix` instance = 0.24 ms; moe.act 0.1 ms
(source: profile above, and -0.10 ms at B=4 in 2026-10-25-barriers.md).

Result, ms per pass at B=1 (two runs each): base 22.25 / 22.24, no barrier after mix 22.22 / 22.22, no barrier after
moe.act 22.21 / 22.26. The upper bound is 0.02 to 0.03 ms for mix and 0 to 0.04 ms for moe.act, far below the
expectation: at B=1 the barriers after these steps cost almost nothing (the blocks of such a short step arrive within
a microsecond of each other). Outside the expectation, so a finding: the 2.2 ms of waiting for the slowest block sit
in the long matvec steps (gr.down 413 us, moe.gate_up 212, moe.down 200, gr.up 173 per pass), not in the short
element-wise steps. Fusion would not even recover its added work. Lever 3 is dropped.

## Lever 4: router (not built as designed)

The design in the report (two waves per row, bit-identical through the first butterfly step) does not shorten the
chain that costs the time: each lane adds its 10 chunks one after the other, and that chain stays the same with
16 instead of 32 lanes per row. What can move bit-identically is the number of chunks in flight (any group size G of
`row_bf16` adds in ascending chunk order). Test: `ROWBF16_G` 4 -> 10 (all ten chunks of a lane in flight) for the whole
build: router 628 us -> 647 / 650 us per pass, pass 23.9 ms (worse; G = 8 was already worse for the inject, 2026-10-25-hc-matvec.md).
The router step (12.9 us each, 256 rows, 1.3 MB) is not limited by loads in flight. Lever 4 is dropped, the build
was restored.

## Final checks (levers 1 and 2 with the NaN guard of lever 1, final commit 07697b9)

- Goldens (`gpu-status.sh --own --build work/build/w101`): prefill dflash, gpu, cpu plan, server, prefill all identical,
  before the helper split and the NaN guard (fdf73a5) and after (07697b9).
- R4 bit-exact (`work/ab/w101-r4.md`, `IE_DETERMINISTIC=1`, workloads mid and long, 1 round, main against final): text
  SHAs of the 30K and 130K requests equal in both arms and equal to `bench/sha.toml`; expert hits and misses equal
  (68,220 / 7,862 and 72,627 / 7,397). Rates, one round only: prefill 30K +0.49 %, decode 30K +1.06 %, decode 130K
  +1.16 %, prefill 130K -0.34 % (inside the expectation of -1 to +1 %; the prefill path is not touched by the change).
- Speed, final code (`work/ab/w101-final.md`, same workloads as above, 6 rounds, decode deciding): decode of code
  +0.68 % (medians 88.9 -> 89.8 tok/s), decode after 13,000 tokens +0.42 % (69.4 -> 70.3), decode of text +3.24 % mean
  (**outside the expectation**; round 6 has arm A at 54.6 tok/s against 64 to 65 in all other rounds, a disturbance of
  that one run; the other five rounds give +0.15 / +0.78 / -0.31 / -0.47 / +0.62 %, median of all six +0.45 %), prefill
  at 13,000 tokens +0.04 %. All decode CIs contain 0 (stop rule: undecided after 6 rounds); text SHA and expert counts
  equal. In the plain B=1 step profile the final code gives pass 23.60 / 23.56 ms against 23.72 / 23.81 / 23.73 ms for
  main (one outlier run at 24.85 ms of the final code, see note).
- CPU selection per pass (`--batch-bench`, B=1 / B=2): main 0.175 / 0.320 ms, final 0.147 / 0.302 ms.
- Gates: `just check` green (clang-format, file length, lizard, gitleaks, clang-tidy), `ctest -L unit` 42 of 42,
  pytest 319 passed.

Note on the profile outlier: one of four alternating profile runs of the final code showed 24.85 ms while the next
showed 23.60 ms; other work on the machine (the build of another worker) was running. Not repeated.

Summary of the levers: 1 (host router) -0.03 ms at B=1 (CPU time -16 %), kept for the NaN safety it carries as
well; 2 (combine) -0.14 ms at B=1; 3 and 4 dropped with numbers. Total about -0.15 ms of a 23.7 ms pass (+0.6 %
decode), against a target of -1 to -2 ms: the target was not reached. The steps of the ticket have 2.4 ms to the
byte roof together, not 3.5 ms as in the 10-25 table, and what is left is limited by the latency of the sequential chain in
each lane, which the bit-exact rule leaves in place.
