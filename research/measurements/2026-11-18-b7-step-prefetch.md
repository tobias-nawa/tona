# B2 (#163): prefetch of the next step descriptor in the verify kernel

Host Ryzen 9 9950X, RX 9060 XT gfx1200, Coder IQ1_M, MTP head. A `work/build/w163b7` (B7 head), B `work/build/w163p`
(the same plus `prefetch_next_op` in `engine/gpu/mega_kernel.h`).

## Prior art

- Hazy "no bubbles" megakernel (2025-05-27): the next instruction is loaded and its pages prefetched while the current
  one runs; the gap between instructions was the main loss of an inference megakernel.
- Mirage MPK (arXiv 2512.22219): task descriptors are prefetched, and a pre-load phase of a task overlaps the previous
  task.
- Programmatic dependent launch: the prologue (weight prefetch) overlaps the tail of the previous kernel.
- Here the 43.8 ms decode cycle has ~1,000 steps per pass; the trace (2026-11-17-b1-cycle-breakdown.md) puts 1 to 2 ms
  into step latency inside the verify kernel.

Hardware: gfx1200 has `s_prefetch_data` (SMEM prefetch into the scalar cache, no fault). The probe
`/var/tmp/claude/163/probe/pf.s` (compile only) shows: `__builtin_amdgcn_s_prefetch_data(ptr, len)` emits
`s_prefetch_data` on gfx1200; `__builtin_prefetch` emits nothing (dropped). Whether the instruction speeds up anything
on the card is exactly what the ab.py run decides; no GPU run of the probe was done.

## Plan (written before the runs)

**Question.** Does a one-line scalar prefetch of `ops[o+1]` (3 lines of 128 bytes) at the top of step o shorten the
step loop, and by how much?

**Expectation.** Decode +0.3 to +1.5 % (0.1 to 0.6 ms of 43.8 ms: ~1,000 steps x 0.1 to 0.6 us of scalar-load wait; the
upper bound of 1 to 2 ms from the breakdown assumes also the task descriptor and weight rows, not done here). Prefill
unchanged (prefill does not run k_mega). Risk: k_mega code generation reacts to small edits (spill comment in
`mega_kernel.h`); a loss is possible, then the change is rolled back.

**Checks.** isa-diff vs `w163b7` (expect only the `k_mega` instantiations to differ, spill counts reported), goldens
(bit-exact by construction: a prefetch changes no data), `scripts/ab.py --a w163b7 --b w163p --workload short,set`.

**Why this size.** Decode only; effect 0.3 to 1.5 % needs 4 to 6 rounds at the 0.23 percent noise of decode@2000.

**Active tooling.** None.

## Results

**Variant 1** (`__builtin_amdgcn_s_prefetch_data(ops + o + 1, 2)`, 619bce4, build `w163p`): goldens identical (47 of 47);
isa-diff vs `w163b7`: only `k_mega` instantiations change. In the hot variants (`Qwen4ExpProfile`; NB=1 draft, NB=8
verify at width 5 to 6) SGPR spills rise by +4 (NB=1), +4 (NB=4) and +9 to +14 (NB=8), VGPR spills by 0 to +1; the
+266 are in probe-profile variants that decode does not use. `scripts/ab.py w163b7 w163p`, 6 rounds, full report
`research/measurements/2026-11-18-b7-step-prefetch-ab.md`, SHAs equal:

| Metric (rate) | B vs A, mean [95 % CI] | Expected | Verdict |
|---|---:|---|---|
| decode@2000 | -1.02 % [-1.26, -0.79] | +0.3 to +1.5 % | B slower |
| decode@2000:write | -1.40 % [-2.07, -0.73] | +0.3 to +1.5 % | B slower |
| decode-code | -1.40 % [-1.90, -0.90] | +0.3 to +1.5 % | B slower |
| decode-text | -0.95 % [-1.45, -0.45] | +0.3 to +1.5 % | B slower |
| prefill@2000 | +0.08 % | -0.5 to +0.5 % | equal |

The prefetch makes decode 1 to 1.4 % slower, outside the plan: a finding. The few extra spills (and the extra address
computation) cost more than the prefetch saves, or the scalar cache already holds the next descriptor (all blocks read
the same ops; the first block to miss fills L2 for the rest). **Not verified** which of the two; only the speed is
measured. Variant 2 (one inline `s_prefetch_data` on the pointer of the current step with the descriptor size as offset,
334da79, build `w163q`) is the one try the plan allows before the rollback; its result follows below.

**Variant 2** (334da79, build `w163q`): isa-diff vs `w163b7`: hot-variant SGPR spills NB=1 unchanged, NB=4 -2, NB=8 +4,
VGPR spills unchanged. `scripts/ab.py w163b7 w163q`, 4 rounds (full report
`research/measurements/2026-11-18-b7-step-prefetch-ab2.md`), SHAs equal:

| Metric (rate) | B vs A | Expected | Verdict |
|---|---:|---|---|
| decode@2000 | +0.27 % | -0.5 to +1.5 % | within |
| decode@2000:write | -1.92 % | -0.5 to +1.5 % | B slower |
| decode-code | -1.48 % | -0.5 to +1.5 % | B slower |
| decode-text | -0.93 % | -0.5 to +1.5 % | B slower |
| prefill@2000 | +0.03 % | -0.5 to +0.5 % | equal |

Decode is again 1 to 2 % slower although the spill counts are back near the base: the spills were not the cause. The
prefetch itself costs time (an `s_prefetch_data` per step per wave is issued by all 16 waves of a block, ~1,000 steps)
or the scalar cache already had the next descriptor; the speed result does not separate the two. **Decision: both
variants reverted** (7333021 and the one before); the step-descriptor prefetch is not a lever here. The long-context
run was not queued (nothing to merge). The next item of the plan for step latency (task descriptor and first weight
rows) would need the same prefetch instruction and is not attempted on this evidence.

