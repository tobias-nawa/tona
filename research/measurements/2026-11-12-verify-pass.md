# Verify pass on the reference: where the cost per extra token goes (#126)

RX 9060 XT (gfx1200), Zen 5, DDR5, Coder IQ1_M shard 1 of 2026-10-19. Branch `126-verify-pass`.

## Plan (written before the first run)

**Question.** A verify pass of width W costs 24.2 / 29.5 / 33.1 / 36.7 / 43.7 / 49.3 / 52.7 / 58.1 ms for W = 1 to 8
(one adaptive run of 2026-10-26, old shard; ~4.9 ms per extra token). The expert bytes of the union explain only
1.2 to 1.9 ms per extra token at 315 GB/s (research/37-spec-draft-length-model.md). Which ops carry the other ~3 ms?

**Ceiling to work towards.** t(W) = t(1) + bytes(union(W) - union(1)) / bandwidth + compute, about 24 + 1.5 (W - 1) ms:
width 4 ~29 ms (36.7 today), width 8 ~35 ms (58.1 today). Decode value (model of research/37, tokens per pass 5.27 on
code at 7 drafts, 3.01 on text; each ms per token at k ~ 3 is worth 3 to 4 % decode): W = 4 saves ~7 ms of a ~45 ms
cycle, W = 8 saves ~23 ms of a ~75 ms cycle; with the width rule choosing wider passes more often the decode gain on
code is expected at +15 to +30 %, on text +5 to +10 %. The plan below measures, the gain is not assumed.

**Prior art read first** (technique, window 12 months): ReplaySSM 2610.13778 and SGLang PR 12892 (lazy state commit
instead of per-draft state copies: the engine already commits lazily, `op_dn_recur` reads the 108 MiB state once per
pass, writes it only after an accepted pass), Bole 2608.01651 / SpecLA 2607.16673 / TreeWY 2608.20961 (DeltaNet
verification in closed form, chunked), llama.cpp PRs 23174, 23531, 23187 (GDN with K > 1 for MTP on SYCL, Hexagon,
CPU) and 29353 (chunked GDN kernel for prompts), vLLM PR 62799 (RDNA4 skinny GEMM for 6 to 128 rows: hipBLASLt at
420 to 470 GB/s against ~600 GB/s for the 1-to-5-row kernel), 2610.34440 and 2610.38981 (expert bytes, unique experts),
2511.10676 (expert prediction before the attention, 93 to 98 %, relevant to the 7.7 ms of misses at B = 4), Cascade
2506.20675 (speculation on MoE: the verify width pulls 2 to 3 times more experts, up to 1.5x slowdown, a utility rule for the
width; relevant to the expert slope and the width choice, which this engine already makes per pass from measured costs), the
own k_wide work (research/measurements/2026-10-25-width-8.md).

**Scope, step 1: profile per op.** `tona-gpu-gen MODEL PROMPT --mega --batch-bench 400` with
`TONA_BENCH_WIDTHS=1,2,4,8 TONA_BENCH_REPEAT=3 TONA_MEGA_PROF=1`: all tokens accepted, real experts from RAM with the
profile `work/profile/experts-coder.bin` (the production expert set), the long code prompt `scripts/bench-long-code.txt`
(36 KB, longer than the 400 positions, so no text repeats). The profile is the step stamps of the kernel itself
(block 0, `wall_clock64`), not rocprofv3. The per-op average starts after the 64 warm-up tokens of each width. Three
rounds, the four widths interleaved; spread = min to max of the three rounds. One slope per op: (t(8) - t(1)) / 7 and
(t(4) - t(1)) / 3 ms per extra token. Expected duration: 4 widths x 400 positions x 3 rounds, about 8 min of GPU after
the load. Why this size: the question is which op grows with W, the pass has ~1,000 steps, 400 positions give 50 to
400 passes per width, the spread of the pass time between rounds is known to be below 1 % (2026-10-23-verify-pass.md).

**Expectation (with tolerance).** Pass time at W = 1 / 2 / 4 / 8: 24 / 30 / 37 / 58 ms (+-15 %, from the table above;
the shard and the expert set changed since). Largest single contributors to the slope by the old profile
(2026-10-23, 2026-10-25): moe.gate_up and moe.down (union, 1.2 to 1.9 ms per token together), dn.in (dense int8 matvec
turns compute-bound from 4 tokens), dn.recur (0.8 ms at W = 1, 3.5 at W = 4 in the old profile, i.e. ~0.9 ms per
token), combine (0.8 to 2.9 ms), head.out, qsa.attn, host.wait. A result in which the non-expert ops add up to less
than 2 ms per extra token, or where one op has a slope above 1 ms per token, is a finding.

**Scope, step 2: fixes** one op at a time, largest non-expert slope first. Each change:
- bit-identical output (`TONA_DETERMINISTIC=1`, `scripts/golden.sh compare`, `scripts/golden-server.sh compare`,
  `scripts/isa-diff.py` for untouched kernels) or the quality check of docs/backends/README.md when numbers move on purpose
  (widths above 4 use the WMMA path from 6 tokens, `TONA_WIDE_MIN`, which is not bit-identical to width 1);
- the same profile before and after (same rounds);
- `scripts/ab.py` for decode (workload `set` and `short`, spread of `bench/noise.toml`), base build against the change.

**Active tooling.** `TONA_MEGA_PROF=1` and `TONA_BENCH_*` in the profile runs only (the step stamps are always written
by the kernel, the host read-back per pass adds a `hipMemcpy` per pass: the profile run is therefore no speed result, the
plain `--batch-bench` without `TONA_MEGA_PROF` gives the pass times). `TONA_EXPERT_PROFILE` as production. No
`TONA_DETERMINISTIC` in speed runs (it switches off the wide path). The machine is shared with other workers: every GPU
run takes `work/gpu.lock` (builds too), one run at a time.

## Test conditions

`tools/hw_report.py` (2026-11-12 09:22): Ryzen 5 9600X, RX 9060 XT gfx1200 (15.9 GiB), 60.5 GiB RAM, Fedora 44,
kernel 7.2.15, rocm 10.1.0, commit `4178608` plus the profile tooling of this branch (`TONA_BENCH_WIDTHS`,
`TONA_BENCH_REPEAT`, `TONA_BENCH_ACCEPT`, `Mega::profile(reset)`; no change of device code). Quiet check before the
profile runs: warning "load average 7.88 is high" (a build of another worker ran next to it without the lock; the
coordinator has asked for all builds under the lock), VRAM 1.17 GiB held by others. Raw logs: `work/v126/*.log` (not
in the repository). Runs after the first one took `work/gpu.lock` per run.

## Results 1: profile of the baseline (main 4178608)

`--batch-bench 2400`, 3 rounds, widths interleaved, µs per pass and op (mean of 3 rounds, ± half the range); the sum
of the ops equals the pass time within 0.4 ms (host part). Spread of the pass time between rounds below 0.35 ms at W = 4
and 8, up to 1.1 ms at W = 1 (round 3 was 0.6 ms slower).

Pass ms (all rounds, min to max): W=1 24.21 (23.67-24.80), 2 29.40 (29.17-29.80), 3 33.0 (32.73-33.28), 4 36.22
(36.16-36.33), 5 44.16 (43.48-44.84), 6 48.66 (48.65-48.66), 7 52.25, 8 55.32 (55.08-55.56). Increments per token
5.2 / 3.6 / 3.2 / **8.0** / 4.5 / 3.6 / 3.0.

```
pass ms (wall, all rounds): {1: (24.21, 23.67, 24.8), 2: (29.4, 29.17, 29.8), 4: (36.22, 36.16, 36.33), 8: (55.32, 55.08, 55.56)}
profile total ms: {1: (23.83, 23.41, 24.45), 2: (29.11, 28.9, 29.5), 4: (35.85, 35.8, 35.96), 8: (54.69, 54.45, 54.92)}
op                         W=1             W=2             W=4             W=8  slope1-4  slope1-8
moe.gate_up        3182 ± 302      4522 ±  26      6571 ±   6     11530 ±  16       1130      1193
moe.down           2306 ± 112      3567 ±  20      5447 ±   2      9363 ±   4       1047      1008
dn.in              3932 ±  20      4242 ±   4      4401 ±   0      5373 ±   6        156       206
gr.down            1849 ±  10      2086 ±   4      2346 ±   4      3327 ±   3        166       211
gr.up              1679 ±  66      2053 ±   1      2256 ±   2      2924 ±  11        192       178
dn.out             1838 ±   4      2036 ±   2      2132 ±   6      2922 ±   2         98       155
dn.recur            910 ±   1       909 ±   1      1200 ±   0      2065 ±   1         97       165
combine             704 ±   2      1093 ±   5      1309 ±   1      2063 ±   2        202       194
qsa.in             1210 ±   8      1302 ±   1      1376 ±   0      1975 ±   2         55       109
head.out           1601 ±   1      1664 ±   0      1697 ±   1      1829 ±   2         32        32
host.wait           768 ± 104       985 ±  50      1236 ±   3      1766 ±  12        156       143
router              624 ±   3       753 ±   2       940 ±   0      1422 ±   0        105       114
host.y              203 ±  88       524 ± 211       836 ±  64      1350 ± 188        211       164
qsa.attn            257 ±   1       398 ±   2       692 ±   1      1277 ±   1        145       146
qsa.out             591 ±   2       653 ±   0       695 ±   2       949 ±   1         35        51
moe.sum             260 ±  16       329 ±   6       431 ±   2       914 ±   0         57        93
dn.conv             384 ±   2       340 ±   0       368 ±   0       794 ±   1         -6        58
moe.act             176 ±   1       224 ±   0       354 ±   0       748 ±   0         59        82
moe.shared          400 ±  39       401 ±  11       394 ±   2       653 ±   4         -2        36
mix                 306 ±   4       350 ±   2       366 ±   1       485 ±   0         20        26
qsa.prep            229 ±   2       238 ±   0       257 ±   0       278 ±   0          9         7
dn.norm             130 ±   2       144 ±   1       183 ±   0       225 ±   0         18        14
qsa.merge            97 ±   0       103 ±   1       162 ±   0       220 ±   0         22        18
moe.copy            103 ±   1       107 ±   0       109 ±   0       119 ±   0          2         2
qsa.topk             37 ±   0        39 ±   0        38 ±   0        39 ±   0          1         0
qsa.score            25 ±   1        25 ±   0        28 ±   0        35 ±   0          1         1
embed                 7 ±   0         8 ±   0        12 ±   0        20 ±   0          2         2
ple.conv              4 ±   0         5 ±   0         8 ±   0        13 ±   0          1         1
ple.gate             11 ±   0        11 ±   0        10 ±   0        10 ±   0         -0        -0
```

Findings:
- Slope 1 to 8: 4.41 ms per extra token. Experts (moe.gate_up + moe.down) 2.2 ms (bytes ideal 1.3 at 315 GB/s), all other
  ops 2.2 ms spread over 25 ops, none above 0.25 ms per token. Sum by kind: dense int8 matvec steps ~1.06 ms, element-wise and
  norm steps ~0.65, host steps (host.wait, host.y) 0.31, dn.recur 0.17.
- The DeltaNet state is read once per pass (registers over all tokens, `op_dn_recur`), dn.recur costs 0.91 ms at W = 1 and
  1.20 ms at W = 4 (0.1 to 0.2 ms per token, including the commit replay at m = nb). Dense projections read their weights once per pass
  (dn.in 3.9 -> 5.4 ms at W = 8).
- **The step at W = 5 is not a per-token cost.** W = 4 -> 5 costs 8.0 ms against 3.2 for 3 -> 4. Nearly every matvec step
  and dn.recur jump at that width (per op, W4 -> W5: moe.gate_up +1704, moe.down +1711, gr.down +572, gr.up +547, dn.in +487,
  dn.out +445, dn.recur +413, combine +344, router +234, moe.shared +188). Cause: width 5 and above run on `k_mega<8>`.
  Control, `TONA_MEGA_K8=1` (4 tokens or fewer on the 8-token kernel): W = 2 32.6 ms (+3.2 against k_mega<4>), W = 4 39.9 ms (+3.7);
  per op the same steps (W = 4: moe.down +783, moe.gate_up +689, gr.up +435, gr.down +422, dn.in +328, dn.out +327, dn.recur +247).
  `scripts/kernel-regs.py`: k_mega<4> 54 VGPR spills, <8> 141; the matvec path has 2,584 VGPR reload sites in <8> against 527 in <4>
  (research/measurements/2026-10-28-mega-spills.md counted the same).
- Wide path: `TONA_WIDE_MIN=4`: W = 4 37.9 (+1.6), W = 5 44.75 (equal); `=3`: W = 4 37.5; `TONA_WIDE_ROWS=2048`: W = 6 50.4
  (+1.7), W = 8 56.7 (+1.4); `TONA_WIDE_MIN=0` (no WMMA): W = 6 49.4 (+0.8), W = 8 56.6 (+1.3). The wide path is worth 1 ms at W = 8 and
  nothing below; the default is the optimum.
- Misses: `TONA_PREFETCH=0`: W = 4 41.4 (+5.2), W = 8 66.5 (+11.2); `TONA_PREFETCH=16`: W = 4 37.1 (+0.9), W = 8 56.1 (+0.8);
  default 8 is the optimum. Per pass at W = 8 (default): 1,595 hits, 52 GPU-direct, 164 CPU experts of 1,811 (12 % misses);
  without prefetch 1,050 hits, 466 GPU-direct, 294 CPU. `TONA_PLAN=gpu` (all misses read directly): W = 4 38.9, W = 8 62.1;
  `TONA_PLAN=cpu`: W = 4 37.4, W = 8 58.7: the planner beats both. host.wait + host.y are 1.0 ms at W = 1, 2.1 at W = 4, 3.1 at W = 8.

## Results 2: change 1, k_mega<8> without spills (commit 345c0ef)

Change: the matvec round (`row_q8_round`) runs vectors 5-8 as a second group behind `__builtin_amdgcn_sched_barrier(0)`; the weights are
unpacked once per round, so the arithmetic and its order are unchanged. Also `dn.recur` token loop unroll 1 for NB > 4 and a two-sum
`combine` loop. Spills (`scripts/kernel-regs.py`): k_mega<8> 141 -> 98 VGPR, scratch 560 -> 432; k<4>, k<1> unchanged.

Logits hash (FNV over the full logits of 600 passes, widths 1-8): identical to the base build at every width
(`work/v126/h-base.log`, `h-new2.log`, `h-new3.log`: 3a2febe7cd168de0 for 1-5, 22e6734613734eb9 / fcea32541e8bc009 / 22e6734613734eb9 for 6-8).

Pass time (batch-bench, 3 rounds for 1/2/4/8, 2 for the others), base profile -> new, ms per pass:

| W | base | new | change |
|---|---:|---:|---:|
| 1 | 23.82 | 23.51 | -0.3 |
| 2 | 29.11 | 28.97 | -0.1 |
| 3 | 32.46 | 32.48 | 0.0 |
| 4 | 35.85 | 35.78 | -0.1 |
| 5 | 43.28 | 41.24 | -2.0 |
| 6 | 48.09 | 45.51 | -2.6 |
| 7 | 51.64 | 48.89 | -2.8 |
| 8 | 54.69 | 51.95 (52.55-52.63 in the bench lines) | -2.7 |

Variant that did not hold (first attempt, 4-token halves as two full calls with shifted pointers): W = 8 57.8 ms (+3.1 against base), because
the weights were read and unpacked twice. Replaced by the two-group round above.

## Discarded or not yet taken (principle 8: reason, research, attempt to remove the cost)

- **Software pipeline of raw weight loads (`Raw<T>`, TONA_ROW_PIPE) in the row loop.** Probe, 8 lanes, nb = 1 / 4, large-array GB/s:
  IQ4_NL 163 -> 262 / 119 -> 131, IQ2_S 114 -> 132 / 76 -> 97, IQ3_S 155 -> 154 / 97 -> 107, IQ3_XXS 138 -> 136 / 111 -> 109. The type that
  matters (IQ4_NL is the down projection of most experts) gains at nb = 1 only. Not taken into the engine yet: the code costs +90 lines in four
  headers, the gain at nb >= 4 (what verify passes use) is 10 % for IQ4_NL and 0 % for the gate/up types in the probe, and the engine build was
  not tested. Attempts to remove the cost: unaligned 4-byte loads (`ld4_a2` as aligned int, i.e. fewer load instructions) gave the same numbers
  (`probe-lie`, `probe-liepipe`), a variant without the dot product (`probe-nodot`) reached 141 / 121 GB/s at nb = 4, so the dot is the limit
  at nb >= 4, not the loads. Prior art: Cascade (arXiv 2506.20675) and SGLang PR 12892 batch experts instead of pipelining loads. Experiment
  diff saved in `work/v126/patches/experiments-all.diff` (not in the repository).
- **Wide path, prefetch depth, planner switches:** default is the optimum, see Results 1.

## Mac findings assessed for the PC (assessment only, nothing implemented or measured here)

- **Mac 1, fragment-order Q6_K/IQ4 for dense matvecs at W >= 4: partly relevant, small ceiling.** The PC dense int8 path (k_mega matvec) does not use MMA
  fragments, so the Mac cost (8 rows per lane group, block header reloaded) does not apply there; it already streams one row per lane group. The
  fragment layout would only touch `k_wide` (`matvec_wide.h`, 106 lines, permlanex16 at lines 72/78), which runs from W = 6 (`wide_min_` default 6, mega.h:425)
  and is worth about 1 ms at W = 8 against int8 (Results 1, TONA_WIDE_MIN=0). The W = 4 -> 5 knee was the k_mega<8> spills (fixed: -2.0 to -2.8 ms at W = 5-8),
  not unpacking. Dense matvec total at W = 8 is about 1.06 ms of slope; even a Mac-sized 12 % gain on the wide part is under 0.15 ms per pass. A
  second copy would also cost RAM (dense weights read at W = 1 need the old layout unless the type serves both). Verdict: not relevant for the knee; worth a
  probe only after the experts (about 38 % of the W = 8 pass).
- **Mac 2, IQ3_XXS repacked for aligned loads (type 1018): relevant, to be probed.** The PC expert kernels read IQ3_XXS with 2-byte-aligned `ld4_a2` (two u16
  loads, `matvec_q8_core.h`), the same misalignment the Mac fixed. Probe evidence for the effect on the PC: `TONA_LD4_LIE` (aligned 4-byte load, wrong results but
  same traffic) gave IQ3_XXS (type 18) nb = 4 111 -> 108 GB/s, IQ3_S 97 -> 107, IQ2_S 76 -> 97 only when combined with the pipeline, so the loads
  are not the limit at nb >= 4 on RDNA4 (the dot is: `probe-nodot` 141 vs 111 GB/s for type 18 at nb = 4). Expected PC gain from alignment alone: below 3 %
  of the IQ3_XXS part (about 5 GiB of the experts). Next step if taken: repack offline into a new type, compare logits hash against the reference (bit-identical since
  only the byte order changes), then probe `--exps`. Not started.

## Results 3: isolating W3/4, raw-load pipeline in the engine, cost table

Test conditions: quiet_check reported the same two warnings in every run (about 1.2 GiB VRAM held by others, light swapping); runs went through the GPU queue
one at a time. Plan for all runs of this section: batch bench W = 1..8 (W = 1,2,3,4,8 for the intermediate variants), 3 rounds, real experts, long code prompt;
expectation: variants that touch only code for more than 4 vectors leave W <= 4 unchanged within 0.1 ms; the pipeline lowers moe.down by about
10 % of its time at 4 vectors (probe: +10 % bandwidth IQ4_NL).

- **The "+0.25 ms at W = 3/4" was drift, not code.** Base binary re-run in the same session (A/A): 23.75 / 29.23 / 32.77 / 36.12 / 43.55 / 48.67 / 52.19 / 55.24 ms
  (W = 1..8). The earlier base profile was taken under load 7.9. The two-sum `combine` loop, measured separately (reverted in 86d019a): 23.70 / 29.08 / 32.70 /
  36.06 / 41.65 / 46.12 / 49.60 / 52.57: it was 0.03-0.13 ms slower than with it at every width and never faster, so it is removed.
  `dn.recur` unroll only changes the NB > 4 instances; k_mega<1> and k_mega<4> compile to the same registers and spills.
- **Raw-load pipeline for IQ4_NL (moe.down), up to 4 vectors:** -0.16 to -0.29 ms per pass at W = 2..4 (moe.down -230 to -310 us), neutral at W = 1, hash unchanged.
  At 8 vectors the pipeline was +0.06 ms (moe.down +77 us), so it is compiled only for NB <= 4.
  Same pipeline for IQ2_S and IQ3_S as well: +0.40 to +0.48 ms at every width (moe.gate_up +200 to +234, dn.out +242 to +324, qsa.out +85 to +101 at W = 3/4),
  although the probe showed +28 % and +10 % bandwidth. Reason: in the engine kernel the extra raw registers cost more than the latency saved (the probe runs a
  single row loop per kernel, the engine instances carry the whole matvec step). Research: llama.cpp keeps IQ types on MMVQ without extra prefetch
  (PR 26079 notes; no PR on a pipelined IQ2_S/IQ3_S GEMV found, prior-art list below). Attempts to remove the cost: restricting to NB <= 4 did not help
  (the loss was at W = 3/4), restricting to type 20 removed it (what is shipped). Not tried: pipeline only for the gate_up role of IQ2_S with half
  the registers (one sub-block ahead instead of two); next step if experts stay the target. Patch kept in `work/v126/patches/pipe-all3.diff`.
- Final curve (IQ4_NL pipeline + spill fix, mean of 3 rounds, spread in brackets), base A/A -> new, ms per pass:
  W1 23.75 -> 23.74 (23.62-23.98), W2 29.23 -> 28.94 (28.92-28.96), W3 32.77 -> 32.52, W4 36.12 -> 35.89 (35.86-35.92), W5 43.55 -> 41.70, W6 48.67 -> 46.14,
  W7 52.19 -> 49.62, W8 55.24 -> 52.61 (52.57-52.66). Logits hash identical to base at W = 1..8 (`work/v126/h-p20.log`).
- The width cost table `t_mpass_` in `engine/server/generator.h` is set to this curve (it was the 2026-10-26 curve with 43.7 at W = 5).
  `t_xpass_` (two conversations) and `t_pass_` (no MTP) are not remeasured and keep their values.

## Prior art (agent search, snippet level, scan.py not yet run)

MonoMoE arXiv 2609.04244 (weight-major persistent megakernel for block-quantized MoE decode); Cursor warp decode (gate/up/down fused, warp per output); llama.cpp PR 20905
`mul_mat_vec_q_moe` (batch up to 8, two rows per warp), PR 19478/20831 (RDNA4 MMVQ table); PR 26079 (MMVQ/MMQ crossover by width); EVICT 2605.00342, MoE-Spec 2602.16052,
EcoSpec 2607.12696, S2-MoE 2608.15018 (verify width with expert-union cost); llama.cpp PR 22673 (GDN partial rollback); Cascade 2506.20675, SGLang PR 12892, arXiv 2511.10676.
Not found: anything on splitting token groups behind scheduling barriers to avoid spills, anything on software-pipelined IQ GEMV or aligned repack.

## Acceptance (commit d6581ce)

Test conditions: `tools/hw_report.py` and `tools/quiet_check.py` before and after each run are in `work/v126/ab1.log` and `ab2.log` (Ryzen 5 9600X, RX 9060 XT gfx1200,
60.5 GiB RAM, load 0.75-2.3, one warning in every check: 1.19 GiB VRAM held by others; swap in use 1.74 GiB, no swap traffic during the runs).

- `just check` steps run by hand with own compile databases (the shared `work/build/engine` belongs to the main checkout): all pre-commit hooks pass, `tidy: ok (9 checked)`.
- Unit tests (`ctest -L unit`, own CPU build): 44 of 44 passed.
- Goldens against `work/golden/`, build of d6581ce: decode (csv, rust, explain) identical, CPU plan identical, prefill identical, prefill with dflash identical,
  server 32 of 36 cases identical. The four `sampled-*` cases differ, and they differ the same way with the base build (`work/v126/g-server-base.log`),
  so they are not caused by this branch (the references seem to be outdated against main; to be checked by whoever recorded them).
- `--batch-check` with TONA_BATCH_NB=8 TONA_WIDE_MIN=0 TONA_DETERMINISTIC=1, 600 positions: top-1 equal 844 of 844, worst relative deviation 0, bitwise 600 of 600 positions equal.
- Logits hash widths 1-8 identical to the base build.
- Server A/B (`scripts/ab.py`, base build against d6581ce, production mode without TONA_DETERMINISTIC, port 8091, quiet machine), 2 rounds of short+set then 6 rounds of short:
  decode@2000 (74.1 -> 75.7 tok/s) +2.2 % [+1.9, +2.5]; decode@2000:write (121.2 -> 125.8) +3.7 % [+3.0, +4.4] (above the planned 3 %, a finding in the good direction);
  decode, bench-set code 90.8 -> 92.2 tok/s +1.5 % [+0.4, +2.7]; decode, bench-set text 68.3 -> 69.8 +2.1 % [+0.7, +3.4]; prefill@2000:write +0.1 %: equal.
  Text of the answers (SHA) identical between A and B in all rounds. The cost table `t_mpass_` changed in B as well, so these gains include the changed width choice.
- Prefill@2000 is bimodal in both arms (about 1507 or 1350 tok/s per server start, in A in rounds 1 and 6, in B in rounds 3 and 5 of the repeat); the result "-3.3 %, open" is that
  noise, not a change of the prefill path (the prefill kernels are not touched). Rounds without the low mode: -0.3 to +0.6 %.
- Not done: `scripts/isa-diff.py` of untouched kernels (R2), `t_xpass_`/`t_pass_` remeasurement, the `just test-sanitize` run.

## Prior art

Search for this section (replaces the agent's snippet-level list above where they differ; items marked unverified come from web search snippets only and are not in the book).
Ledger: `research/papers.md` and `research/papers-screened.tsv` grepped for spill, VGPR, sched_barrier, register, IQ quants, GEMV/matvec, bandwidth, verify, megakernel;
`scan.py arxiv` 2026-11-01..2026-11-09 with `--no-book`: 308 papers, all already in the book (screened rows); the GitHub scan of the same window (`--kind both --no-book`) finished complete: 595 issue hits and 4632 PR hits, 0 not in papers.md (the book covers GitHub to 2026-11-11), so the relevant ones are the rows above.

- [Utility-Driven Speculative Decoding for MoE (Cascade), arXiv 2506.20675](https://arxiv.org/abs/2506.20675), status: read (book row). Verifying several drafts touches more experts, so verify cost grows with
  width. Adopted: the cost-per-width view; our `t_mpass_` is that cost curve, now re-measured. Not adopted: their per-request on/off and K hill climbing, our adaptive width already chooses by the same utility.
- [MoESD, arXiv 2505.19645](https://arxiv.org/abs/2505.19645), status: read (book row). Speculation helps a sparse MoE only once enough experts are active anyway; at batch 1 wide verifies cost experts. Adopted: nothing in code, it confirms that the
  expert part of the pass (2.2 of 4.4 ms per extra token here) is the lever. Not adopted: their batch regime (medium batch) is not ours.
- [HybridSpec, arXiv 2611.04055](https://arxiv.org/abs/2611.04055), status: read (abstract, book row). On Qwen3.6-35B-A3B with DFlash the verify pass grows 71 % from length 4 to 16, almost entirely in GDN and MoE layers, and they
  choose the verification length from measured target latency. Same shape as our curve (23.7 -> 52.6 ms from W1 to W8, +122 %). Adopted: nothing; the measured curve feeds our existing width choice (`t_mpass_`). Not adopted: GPU-side length selection (future ticket).
- [SGLang #12892](https://github.com/sgl-project/sglang/pull/12892), status: open (book row). Rollback of the SSM/conv state after MTP verify without copies, 339 -> 50 us. Not adopted: our dn.recur slope is 0.17 ms per pass for all widths, rollback is not the width cost.
- [Pre-attention expert prediction, arXiv 2511.10676](https://arxiv.org/abs/2511.10676), status: read (book row). Earlier expert prediction. Not adopted here: misses are 12 % at W = 8 and the prefetch depth 8 was already the optimum (Results 1); a better predictor is a separate ticket.
- [llama.cpp #29910](https://github.com/ggml-org/llama.cpp/pull/29910) (closed, body read, with #30021, #30344 open), status: closed. VGPR spills in a quantized DP4A matvec path on AMD cost 140-180 % at larger batches; removing them is the fix. Adopted: the method (compare spill counts per kernel instance with `scripts/kernel-regs.py`),
  which is how the k_mega<8> instance was found (141 -> 98 VGPR spills).
- [llama.cpp #31804](https://github.com/ggml-org/llama.cpp/pull/31804), status: open (book row). On the proprietary RDNA4 Vulkan driver the default MMVQ row configuration beats the 4-row one at batch 5-8 for Q3_K, Q5_K, IQ4_XS. Same width range as our verify passes; Vulkan and not our kernel, so not adopted; it supports looking at the wide instances separately.
- [llama.cpp #19478](https://github.com/ggml-org/llama.cpp/pull/19478) and [#24386](https://github.com/ggml-org/llama.cpp/pull/24386), status: merged/open (book rows). RDNA4 MMVQ warp-count tables for batch 1. Not adopted: batch 1 only; our row loop is our own.
- [MonoMoE, arXiv 2609.04244](https://arxiv.org/abs/2609.04244), status: read (abstract, book row). Weight-major persistent megakernel for block-quantized MoE decode with the token tile on the N dimension: the design our k_mega already follows. NVIDIA/FP8, not taken over.
- [ik_llama.cpp #2633, #2676](https://github.com/ikawrakow/ik_llama.cpp), status: merged (book row). CPU TG kernels for IQ3_XXS/IQ2_S/IQ3_S with 4-row repack. CPU only; relevant to the IQ3_XXS repacking question (below).
- Not in the book and unverified (agent snippet): Cursor "warp decode" blog (fused gate/up/down, one warp per output), llama.cpp PRs 26079 (MMVQ/MMQ crossover by width) and 20831. No item on splitting token groups behind scheduling barriers to avoid spills, none on a software-pipelined IQ2_S/IQ3_S GEMV or an aligned-load repack was found in the book or the scan.

Search: tools/papers/scan.py topics v1; topics: moe-offload, speculative, megakernel, quantization, amd-rocm; terms: verify width, VGPR spill, sched_barrier, IQ4_NL GEMV, MoE kernel bandwidth; window: 2025-11-12..2026-11-12 (12 months); sources: arXiv (scan.py 2026-11-01..2026-11-09, `--no-book`), GitHub PRs and issues, research/papers.md, papers-screened.tsv.

## Open expert and verify-pass levers (for the follow-up ticket, with this branch's numbers)

Slope W1 -> W8 after this work: about 4.1 ms per extra token (23.74 -> 52.61 ms over 7 tokens); experts (moe.gate_up + moe.down) were 2.2 ms of 4.41 before, ideal 1.3 ms.
Per-op times (base profile, us, W1 / W4 / W8): moe.gate_up 3182 / 6571 / 11530, moe.down 2306 / 5447 / 9363, dn.in 3932 / 4401 / 5373, gr.down 1849 / 2346 / 3327, gr.up 1679 / 2256 / 2924, dn.out 1838 / 2132 / 2922, dn.recur 910 / 1200 / 2065, host.wait 768 / 1236 / 1766, router 624 / 940 / 1422.

1. **IQ2_S / IQ3_S pipeline with fewer registers.** The probe gained +28 % (IQ2_S) and +10 % (IQ3_S) bandwidth at nb = 4 with two sub-blocks ahead, the engine lost 0.40-0.48 ms per pass (gate_up +200..+234 us, dn.out +242..+324 us at W = 3/4).
   Try one sub-block ahead, or raw loads only (no early decode); check `kernel-regs.py` for k_mega<4> first (54 VGPR spills now). Where: moe.gate_up 11.5 ms at W = 8.
2. **Fuse gate_up and down** to remove one grid-wide step and the intermediate round trip: moe.gate_up + moe.down = 5.5 ms at W = 1, 12.0 ms at W = 4, 20.9 ms at W = 8; the Mac grouped-expert result (2026-11-12-mac-moe-grouped.md) is the comparison. No measurement of the barrier cost itself yet: first measure it by an empty step.
3. **Packed dot product.** The no-dot probe (`probe-nodot`) at nb = 4: IQ2_S 76 -> 101 GB/s (+33 %), IQ3_S 97 -> 121 (+25 %), IQ3_XXS 111 -> 141 (+27 %), IQ4_XS-like type 23 151 -> 181 (+20 %), IQ4_NL 119 -> 126 (+6 %). So the int8 dot and the unpacking behind it limit at nb >= 4 for the gate/up types;
   candidates are `v_dot4` with the sign applied once per row, packed f16 `v_pk_fma` (as k_wide), or fewer integer ops in the sign/mask expansion of IQ2_S.
4. **gr.up split:** 640 units on 512 waves, gr.up 1679 / 2256 / 2924 us; a split in 2 x 320 or more rows per unit removes the 128-unit tail (about 20 % of one round).
5. **dn.recur imbalance** (768 units on 512 waves; 910 / 1200 / 2065 us): not bit-exact fixable by reordering, needs a hash check; the NB > 4 unroll fix already took 0.3-0.5 ms off at W = 8.
6. **Quantization of units above 512 waves at W >= 5-7** in the elementwise ops: combine jumps at W5, mix and moe.sum at W7, dn.conv at W6.
7. **Misses and host:** 12 % misses per pass at W = 8 (52 GPU-direct, 164 CPU of 1811); host.wait + host.y 1.0 / 2.1 / 3.1 ms at W = 1 / 4 / 8; prefetch depth 8 is the optimum, a better predictor or more slots for verify passes is the next lever (2511.10676).
8. **IQ3_XXS aligned repack (Mac type 1018):** about 5 GiB of experts; the aligned-load probe (`TONA_LD4_LIE`) gave 111 -> 108 GB/s for the type at nb = 4, so alone it is below 3 %; combine with item 3 only.
9. **Fragment-order Q6_K/IQ4 for k_wide:** about 1 ms total at W = 8 for the whole wide path, see the Mac assessment above; last in line.

## Principle 8, completed for every item not taken over

| Item | Gain seen | Reason not taken | Research (book, this date) | Attempt to remove the cost |
|---|---|---|---|---|
| two-sum `combine` loop | none at W5-8, worse by 0.03-0.13 ms at all widths | the second sum's loads do not overlap better than the unrolled single loop; code shape cost at W <= 4 | nothing found on combine/norm loops in the book (grep `RMSNorm`, `norm` kernels: only fused-norm work on Metal, #16143) | tried the isolated revert (86d019a), curve with and without it measured |
| 4-token halves as two full calls | none (W8 +3.1 ms) | weights read and unpacked twice | llama.cpp MMVQ keeps one weight read per batch instance (#31804, #19478 text) | replaced by the two-group round behind `sched_barrier` (taken) |
| Raw-load pipeline for IQ2_S, IQ3_S | probe +28 % / +10 % bandwidth | +0.40..0.48 ms per pass in the engine (register cost) | #29910 (spill removal on AMD) says register pressure decides low-bit kernels; no pipelined IQ GEMV found | restricted to NB <= 4 (no help), restricted to IQ4_NL (shipped); one-sub-block-ahead variant listed as lever 1 |
| Raw-load pipeline for IQ4_NL at 8 vectors | none | +0.06 ms | same | restricted to NB <= 4 (shipped) |
| `TONA_WIDE_MIN`, `TONA_WIDE_ROWS`, `TONA_PREFETCH`, `TONA_PLAN` | none (defaults are optima) | default measured best | 2511.10676 (prediction), Bevy 2611.04794 (scheduling beats prefetch with a cache), Ostinato 2611.11533, Apto 2611.01859 (expert scheduling on shared PCs) | prefetch 0/16 and planner gpu/cpu measured, Results 1; scheduling by runahead is a different change |
| aligned 4-byte loads (`TONA_LD4_LIE`) | 0-3 % in the probe | not the limit at nb >= 4 | ik_llama.cpp #2633/#2676 (CPU repack) | tested alone and with the pipeline, `probe-lie`, `probe-liepipe` |

## After the merge of main (commit 30e17f7)

- Merge of main into the branch had one conflict, `t_mpass_` in `engine/server/generator.h` (comment only changed on main, kept our curve and comment).
- ISA diff (`scripts/isa-diff.py`, main build 392cdab against this branch, `work/v126/isa.log`): 198 functions in both; code differs in k_mega<1>, <4>, <8>, `op_dn_recur<8>`, `op_qsa_prep`, the chain kernels and the standalone `k_matvec_q8` instances
  (types 8, 12, 13, 14, 18, 20, 21, 22, 23, 42) and the probe kernels; all other kernels identical. Not a class C4 change, so goldens and the A/B speed run apply (done above). Register counts: k_mega<8> VGPR spills 141 -> 98, scratch 560 -> 432;
  k_mega<4> and k_mega<1> VGPR spills unchanged (54, 62), their SGPR spills dropped (4: 179 -> 113; 1: 106 -> 105), `k_matvec_q8<8>` VGPR 52 -> 51, `<20>` 58 -> 59.
  The standalone `k_matvec_q8` kernels differ because the shared row loop changed (`row_q8_round`); they are used by the golden-checked paths, which are identical.
- `test-sanitize` equivalent (CMake debug, `TONA_SANITIZE=ON`, CPU part only: the GPU part cannot be built with ASan on gfx1200, the compiler refuses `-fsanitize` for the offload target; CI builds without the toolchain too): 100 % of 46 unit tests passed under ASan and UBSan (`work/v126/san.log`).
