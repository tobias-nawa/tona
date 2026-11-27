# Mac: Metal through the new registry entry, `Model` renamed (step M1)

Issue #142. The Metal backend registers through `Entry` with its kernel types (tier 3), the registry's Coder-only
path is gone, and `Model` is `Qwen4ExpModel`. No compute code changes.

## Plan (written before the run)

**Decision.** Whether M1 may merge: the gate is Metal goldens identical and Metal decode and prefill not slower.
Possible results: no metric decided slower and the checks identical → merge; a metric decided slower → a finding (no
compute changed, so it would point at the build, e.g. a different inlining), no merge until explained.
**Checks** (`engine/backend/metal/AGENTS.md`): `tona-check` 64 steps, `--widths 7`, `--prefill 5` with
`TONA_PREFILL_CHUNK=64`, `TONA_SEQS=2 --widths 7` against the references of the Metal branch; the MTP golden of the
reference (DM + MTP Q8_0, baseline 241 of 256, median 5.83e-02); the mini model, 200 steps against the CPU reference.
**Run.** `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, DM-T + MTP-164, A main 47c5a41
(`work/build/main`), B this branch (`work/build/m1`), 2 to 6 rounds, stop when decided (margin 0.5 %), `--load-log on`;
both arms run the keepalive default (500 ms), so their decode rates are also the check the coordinator asked for of
the keepalive's decode (the earlier A/B left decode open at -0.6 %).
**Expectation.** All four metrics -0.5 to +0.5 % (no compute change); answers identical.
**Active tooling.** The load log; defaults otherwise.

## Result

Checks (before the run): `tona-check` 64 steps, `--widths 7` and `--prefill 5` identical to the references,
`TONA_SEQS=2` unchanged against them, MTP golden 241 of 256 (median 5.83e-02, as the baseline), mini model 207 of 207.

A/B (6 rounds, [server A/B](2026-11-15-mac-m1-registry-server-ab.md), test conditions in
[the log](2026-11-15-mac-m1-registry.log)): no metric decided slower, all four open at margin 0.5 %; answers
identical (SHA 727ff658ae742fa7 in both arms, stable).

| Metric | A median | B median | B vs A, mean [CI] |
|---|---|---|---|
| prefill@8000 | 167.9 | 170.3 | +0.75 % [-0.69, +2.19] |
| decode@8000 | 25.2 | 25.2 | +0.07 % [-1.26, +1.39] |
| decode-code | 39.0 | 39.2 | +0.04 % [-0.65, +0.74] |
| decode-text | 27.0 | 27.4 | +0.62 % [-0.49, +1.72] |

Against the plan: prefill and decode-text lie above the expectation by their means, but every interval contains 0, so
neither is a decided change; per the plan's rule (nothing decided slower, checks identical) M1 may merge. The machine
was not quiet in rounds 1 and 2 (other programs 106 to 114 % of a core, up to 1.3 GB swap-ins, quiet check before:
warning, load 5.0); the after check 60 s past the end saw the GPU 98 % busy, the one 25 s later was ok.

**Keepalive decode.** Both arms run the fixed keepalive (500 ms). Not paired against keepalive off, so only a
comparison across runs: decode@8000 25.2, code 39.0 to 39.2, text 27.0 to 27.4 here against 25.1, 38.7 and 27.1 for
the arm without keepalive in [the keepalive A/B](2026-11-15-mac-keepalive-server-ab.md). No sign of a decode loss;
a paired A/B would be needed to decide it.
