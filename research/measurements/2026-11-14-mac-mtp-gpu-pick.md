# Mac: the MTP head's pick on the GPU (`TONA_METAL_GPU_PICK`)

Issue #108. Follows [2026-11-14-mac-gpu-gaps.md](2026-11-14-mac-gpu-gaps.md#decisions-on-the-candidates-coordinator-2026-11-14):
the host scan over the head's 49,681 rows costs about 0.33 ms per pick, most of the backend's 1.06 ms of GPU idle time
per decode cycle.

## Plan (written before the runs)

**Change under test.** `kernels/mtp_pick.metal`, appended to every MTP head pass that computes logits: each threadgroup
reduces a slice of at least 1024 rows to {row, max, sum of `fast::exp(l - max)`, NaN count}, one threadgroup merges
(at most 256 slices). Semantics of the reference's `op_mtp_pick` (`engine/gpu/mega_op_mtp.h`) and of the host scan:
logits clamped (NaN = −FLT_MAX, ±inf = ±FLT_MAX), free rows (token −1) skipped, the first valid row wins at the least,
on a tie the smaller row. The token of each row lives in a device buffer kept in step with the dynamic rows
(`MtpVocab::sync`). The host reads 4 floats: token of the row, q = 1 / sum. `TONA_METAL_GPU_PICK=1` (off by default
until measured); `TONA_METAL_PICK_CHECK=1` runs the host scan as well and prints per 200 picks the token mismatches and
q's relative deviation.

**Why this changes no output distribution.** The pick's q enters only the width of the draft chain: `chain_push`
(`engine/server/draft.cpp`) keeps it in `raw_q` for the calibration (`mcal.predict`, `observe`). Acceptance never
uses it: greedy accepts while the target's argmax equals the draft (`Generator::accept_drafts`,
`engine/server/verify.cpp`); speculative sampling uses `s.qd`, a distribution the host builds from the head's logits
(`chain_draw`, `make_dist_rows`), or none. The draft token itself is the argmax, which the GPU reduces exactly. So a
change of q's bits can change chain widths, never which tokens the target accepts; under `TONA_DETERMINISTIC=1` (pass
width does not change the target's numbers) the answers are identical.

**Smoke (done).** DM + MTP Q8_0 against the reference's golden file, `TONA_METAL_GPU_PICK=1 TONA_METAL_PICK_CHECK=1`:
200 picks, 0 token mismatches, q relative deviation median 7.4e-08, max 4.0e-07; golden 241 of 256, median 5.83e-02, as
before.

**Prior art.** The reference's own `op_mtp_pick` (two steps: blocks over slices, block 0 merges; "a single block over
all rows was the bottleneck at 41K rows"); llama.cpp samples on the host after reading all logits
(`common/sampling.cpp`), MLX keeps `argmax` as a GPU op (`mx.argmax`) and transfers one index.

**Run 1, identity (about 20 min).** `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, DM-T +
MTP-164, both arms `TONA_DETERMINISTIC=1`; A host pick, B `TONA_METAL_GPU_PICK=1 TONA_METAL_PICK_CHECK=1`;
`--fixed-rounds --max-rounds 2`, `--load-log on`. Expected: the 8000-token answers identical (SHA), the check lines of B
0 token mismatches, q deviation ≤ 1e-6. Speed not compared (B runs both picks).

**Run 2, speed (about 50 min).** Same workload, default mode; A host pick, B `TONA_METAL_GPU_PICK=1`; both
`TONA_GPU_GAPS=500` and a draft log per arm (`TONA_DRAFT_LOG`, width and accepted drafts per pass);
`--fixed-rounds --max-rounds 4`, `--load-log on`. Expected: backend idle per cycle 1.06 → about 0.1 ms; decode B
against A +0.5 to +1.5 % (code, text, decode@8000), prefill@8000 within ±1 %; chain widths and accepted drafts per
pass alike in both arms (q changes by 1e-7).

**Active tooling.** Run 1: `TONA_DETERMINISTIC=1` both, `TONA_METAL_PICK_CHECK=1` in B. Run 2: `TONA_GPU_GAPS=500`,
`TONA_DRAFT_LOG` both. The load log in both runs; defaults otherwise (`TONA_PLE_PREFETCH` 2).

## Result of run 2 (speed): 0.77 ms less host time per cycle, 0.25 ms more GPU time, decode undecided

Run order: run 1 stopped at once on a missing `--expect` of the script (no server started) and was repeated after run
2, below. Conditions of run 2: before ok, after "do not measure" (GPU 98 % busy, the run's own server) and 20 s later
ok; log in [2026-11-14-mac-mtp-gpu-pick.log](2026-11-14-mac-mtp-gpu-pick.log). Report and load table:
[2026-11-14-mac-mtp-gpu-pick-server-ab.md](2026-11-14-mac-mtp-gpu-pick-server-ab.md). Build 519cc97 plus the change
(the merge of main afterwards changed only a cast in `ple_stats.h`).

| Metric | A median (range) | B median (range) | B vs A, mean [CI] | Expected |
|---|---|---|---|---|
| decode-code tok/s | 38.5 (38.1-38.7) | 38.4 (38.3-38.4) | −0.13 % [−1.27, +1.02], open | +0.5 to +1.5 % |
| decode-text tok/s | 26.6 (26.5-27.1) | 26.7 (26.7-26.7) | +0.01 % [−1.36, +1.38], open | +0.5 to +1.5 % |
| decode@8000 tok/s | 24.9 (24.6-25.2) | 24.9 (24.7-24.9) | −0.12 % [−1.47, +1.23], open | +0.5 to +1.5 % |
| prefill@8000 tok/s | 171.7 (167.6-177.0) | 174.0 (169.6-177.2) | +0.97 % [−0.00, +1.94], open | ±1 % |

Answer of the 8000-token request identical in both arms in default mode too (SHA 727ff658ae742fa7, stable in both).

GPU gaps per cycle (17 starts, 3 lines each: the set's code and text, then the 8000-token request; ms, A's range
against B's, discarded starts included):

| Line | GPU busy A | GPU busy B | backend A | backend B | idle A | idle B |
|---|---|---|---|---|---|---|
| 1 | 91.5 to 93.4 | 93.3 to 93.7 | 1.11 to 1.17 | 0.33 to 0.42 | 4.15 to 4.40 | 3.32 to 3.45 |
| 2 | 96.9 to 98.8 | 99.0 to 99.1 | 1.08 to 1.28 | 0.34 to 0.37 | 4.15 to 5.25 | 3.37 to 3.43 |
| 3 | 66.1 to 68.2 | 68.2 to 68.4 | 0.79 to 0.92 | 0.28 to 0.30 | 3.08 to 5.27 | 2.56 to 2.61 |

The other parts (wake, outside, inputs, submit) are equal in both arms.

Draft log over all passes (A 11,862, B 10,544; A has the warm-up start more): mean width 2.276 in both, accepted
drafts per pass 1.768 in both (2.768 tokens per pass), acceptance 0.777 in both; the width distribution equal to three
decimals (width 0: 0.146, 1: 0.398, 2: 0.103, 3: 0.128, 4: 0.068, 5: 0.029, 6: 0.017, 7: 0.109).

**Finding: below the expectation.** The pick moves off the host as intended (backend 1.1 → 0.35 ms per cycle) and the
chain widths do not change, but the GPU's busy time grows by about 0.25 ms per cycle: the two pick dispatches per head
pass (about 3 per cycle) cost the GPU about 40 us each. Net per cycle about 0.6 ms (busy plus idle 97.5 → 96.9 ms in
line 1), about +0.6 %, inside the A/B's spread (CI about ±1.2 %): not decided by 4 rounds. Where the rest of the
backend's 0.35 ms goes (h copies, NaN count) and whether one dispatch instead of two halves the GPU cost is open.

## Result of run 1 (identity): 0 token mismatches in 13,600 picks, answers identical

Conditions: before ok, after "do not measure" (GPU 97 % busy, the run's own server) and 20 s later ok; build 42e1beb plus
the change. Report: [2026-11-14-mac-mtp-gpu-pick-identity.md](2026-11-14-mac-mtp-gpu-pick-identity.md). Both arms
`TONA_DETERMINISTIC=1`, 2 rounds.

- 8000-token answer: SHA 727ff658ae742fa7 in both arms, stable in both (the same as in default mode in run 2).
- Pick check of B (host scan beside the GPU's pick, 4 starts, 68 lines of 200 picks): 13,600 picks, **0 token
  mismatches**; q relative deviation median of the lines 7.5e-08, largest 4.4e-07 (float sum against double).

**Finding: inside the expectation.** The GPU's pick is the host's pick in every case seen; q differs by less than 1e-6
and changed no chain width (run 2). Together with run 2: bit-identical answers, 0.77 ms less host time per cycle,
0.25 ms more GPU time, a net of about +0.6 % that 4 rounds do not decide. Switch off until the coordinator decides.

## One dispatch instead of two (coordinator: choice b)

**Micro probe** (49,681 rows, 41,489 listed and every third dynamic row free, normal logits, 300 picks per variant,
GPU time of a command buffer with only the pick):

| Variant | us per pick | Against two dispatches |
|---|---|---|
| two dispatches (slices, then one merging threadgroup), as committed | 38.9 to 42.5 | - |
| one threadgroup of 1024 threads over all rows | 128.5 | rejected |
| one dispatch, the last threadgroup merges (atomic counter), 1024 rows per slice | 33.5 to 37.2 | all 4 floats bit-equal, 0 of 300 differ |
| the same with 256 / 512 / 2048 rows per slice | 40.3 / 32.7 / 36.6 | bit-equal; inside the spread of 3 us, 1024 kept |

The last threadgroup merges the slices by thread index (the fixed tree of the two-dispatch merge), not in the order
they finished, and sets the counter back to 0. MSL 3.1 has no device-scope fence; `atomic_thread_fence(mem_device,
seq_cst, thread_scope_device)` needs MSL 3.2 (macOS 15, the backend's minimum already), so the library is built with
`-std=metal3.2`. **Finding:** in the engine, without `coherent(device)` on the slice buffer, the merge read stale
slices: 42 and 48 of 200 picks with the wrong token, q off by up to 0.8, a different hash in each of two runs (the
probe, alone on the GPU, never showed it). With `coherent(device)`: 0 mismatches, q deviation as with two dispatches
(median 7.4e-08, max 4.0e-07), the same hash over token and q bits in two runs (`d8db34c300763244`).

Checks of the MSL 3.2 library (all kernels recompiled): tona-check DM-T `TONA_DETERMINISTIC=1` identical to the
3.1 build's output; widths (seed 7) 0 of 66 rows not bit-identical, with `TONA_SEQS=2` 0 of 160; prefill cut
(chunk 64, seed 5) identical; MTP golden 241 of 256, median 5.83e-02, as before; 54 of 54 unit tests.

### Plan of run 3 (written before the run)

**Scope.** `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, DM-T + MTP-164, 6 fixed rounds
(about 85 min with warm-up), `--load-log on`. A: the build of run 2 (MSL 3.1, host pick, the default), copied to
`work/build/mtl-pick2a` before the change. B: the new build with `TONA_METAL_GPU_PICK=1`. So B against A is what a
default change would bring (pick and library version together); the MSL 3.2 library alone is checked bit-identical
above.

**Expectation.** Backend idle per cycle 1.1 → 0.35 ms (as run 2); GPU busy per cycle B over A +0.15 to +0.25 ms (run 2:
+0.25 with two dispatches; the probe saves about 7 us of about 40 per pick, at about 3 picks per cycle); decode code,
text and @8000 +0.3 to +1.2 % (from the net 0.6 to 0.7 ms per cycle of about 97); prefill@8000 within ±1 %; answers of
the 8000-token request identical; chain widths and acceptance as in A (draft logs). Coordinator's rule: undecided
decode but measurably shorter GPU time per cycle (backend plus pick) → default on.

**Active tooling.** Both arms `TONA_GPU_GAPS=500` and `TONA_DRAFT_LOG` (one file per arm); the load log; defaults
otherwise.

### Result of run 3: decode +0.2 to +0.5 % (all open), 0.4 to 0.7 ms less per cycle, answers identical

Conditions: before ok; after "do not measure" (GPU 98 % busy, the run's own server), 20 s later a warning (other
programs 5.5 % of the CPU threads). 6 rounds, 78 min. Report and load table:
[2026-11-14-mac-mtp-gpu-pick1-server-ab.md](2026-11-14-mac-mtp-gpu-pick1-server-ab.md).

| Metric | A median (range) | B median (range) | B vs A, mean [CI] | Expected |
|---|---|---|---|---|
| decode-code tok/s | 38.4 (38.1-38.9) | 38.4 (38.4-39.0) | +0.22 % [−0.83, +1.27], open | +0.3 to +1.2 % |
| decode-text tok/s | 26.6 (26.6-27.1) | 26.8 (26.7-27.3) | +0.50 % [−0.75, +1.75], open | +0.3 to +1.2 % |
| decode@8000 tok/s | 24.9 (24.7-25.2) | 24.9 (24.8-25.3) | +0.22 % [−0.84, +1.28], open | +0.3 to +1.2 % |
| prefill@8000 tok/s | 169.5 (160.2-175.6) | 170.0 (168.1-176.0) | +1.74 % [−1.01, +4.48], open | ±1 % |

8000-token answer SHA 727ff658ae742fa7 in both arms, stable. Draft logs (A 17,134 passes, B 15,816): width 2.276,
accepted per pass 1.768, acceptance 0.777 in both, the width distribution equal to three decimals.

GPU gaps per cycle, medians over the starts (A 13, B 12; range in brackets), ms:

| Line | busy A | busy B | idle A | idle B | backend A | backend B | busy + idle A → B |
|---|---|---|---|---|---|---|---|
| code | 93.27 (90.99-93.45) | 93.47 (91.70-93.66) | 4.18 (4.06-4.54) | 3.34 (3.30-3.72) | 1.13 | 0.35 | 97.44 → 96.78 |
| text | 98.74 (96.22-98.88) | 98.88 (97.00-99.01) | 4.18 (4.12-4.39) | 3.42 (3.40-3.66) | 1.09 | 0.36 | 102.92 → 102.31 |
| 8000 | 68.01 (66.24-68.16) | 68.15 (66.68-68.39) | 3.13 (3.07-4.34) | 2.59 (2.55-2.77) | 0.80 | 0.29 | 71.13 → 70.75 |

**Reading against the plan.** The pick's GPU cost fell from +0.25 ms per cycle (two dispatches, run 2) to +0.14 to
+0.20; the idle ranges of the arms do not overlap; the cycle is 0.38 to 0.66 ms (0.5 to 0.7 %) shorter. Decode moves
the same way but stays inside its spread after 6 rounds; prefill's +1.7 % is A's spread (one round of A at 160).

**MSL 3.2 and registers (coordinator, point 3).** Xcode's pipeline statistics need a GPU capture; instead every kernel
of both libraries was built into a pipeline and its `maxTotalThreadsPerThreadgroup` (falls when a kernel needs more
registers), `staticThreadgroupMemoryLength` and `threadExecutionWidth` compared: all 463 kernels other than the pick
equal, including the register-bound ones below 1024 threads (`dn_step` 768, the `gemmx_*` 896). Library 4,305,556 →
4,301,812 bytes (the two pick kernels replaced by one). Spills themselves are not visible this way (unchecked).

### Plan of run 4: the MSL 3.2 library alone (written before the run)

Run 3 is positive in its direction but undecided, so (coordinator) a control: does the library version alone cost
anything? **Scope:** the same workload and models, 4 fixed rounds (about 55 min), `--load-log on`. A
`work/build/mtl-pick2a` (MSL 3.1), B `work/build/mtl` (MSL 3.2, commit 9f7af71) with `TONA_METAL_GPU_PICK=0`, so
both pick on the host and differ only in the compiled kernels (and the unused pick kernel). **Expectation:** decode
and prefill within ±0.5 % (all other kernels with the same pipeline limits); answers identical; backend idle equal
(about 1.1 ms in both). **Active tooling:** `TONA_GPU_GAPS=500` in both arms; the load log.

**Transfer to the reference PC (coordinator): none.** The HIP path picks inside the megakernel already
(`OP_MTP_PICK`, `engine/gpu/mega_op_mtp.h`); this change only brings Metal to the same place.

### Result of run 4: decode@8000 −0.87 % decided, code and text unchanged, GPU time per cycle unchanged

Conditions: before ok; after "do not measure" (GPU 97 % busy, the run's own server), 20 s later ok. 4 rounds, 53 min.
Report: [2026-11-14-mac-mtp-msl32-server-ab.md](2026-11-14-mac-mtp-msl32-server-ab.md).

| Metric | A (MSL 3.1) median (range) | B (MSL 3.2, host pick) median (range) | B vs A, mean [CI] | Expected |
|---|---|---|---|---|
| decode-code tok/s | 38.3 (38.2-38.5) | 38.3 (38.2-38.4) | −0.13 % [−0.92, +0.66], open | ±0.5 % |
| decode-text tok/s | 26.7 (26.6-26.8) | 26.7 (26.6-26.8) | +0.00 % [−0.92, +0.93], open | ±0.5 % |
| decode@8000 tok/s | 24.9 (24.8-25.1) | 24.7 (24.7-24.8) | **−0.87 % [−1.72, −0.01], B slower** | ±0.5 % |
| prefill@8000 tok/s | 166.9 (161.2-170.8) | 169.2 (167.2-170.5) | +1.57 % [−1.60, +4.74], open | ±0.5 % |

Answers identical (SHA 727ff658ae742fa7). GPU gaps per cycle, medians (A 9 starts, B 8): busy 92.98 / 98.37 / 67.77
(A) against 93.01 / 98.29 / 67.66 (B) ms, idle 4.22 / 4.21 / 3.15 against 4.20 / 4.22 / 3.15, backend equal.

**Finding: outside the expectation, and against the probe.** The one decided metric says MSL 3.2 costs 0.9 % of decode
after 8000 tokens of prompt; the GPU's own clock says busy and idle time per cycle are the same to 0.1 ms (B's busy
in the third line 0.11 ms lower). The CI touches −0.01, one of four metrics, each round's 8000-token decode about 100
tokens. Taken at face value it would undo most of the pick's gain (run 3: +0.22 % at 8000 for both changes
together), so the default stays off and 9f7af71 is not pushed until a repetition (run 5) decides. A narrow minus in 1 of 4 metrics while the GPU's clock is
unchanged is suspect of multiple comparison or noise (coordinator); run 5 decides with 8 rounds pooled. If the minus
holds: before a revert, check whether the one-dispatch pick works under MSL 3.1 (in 3.1 `atomic_thread_fence` with
`thread_scope_device` does not compile, tried 2026-11-14; remaining candidates: the slices' results exchanged through
device atomics). The revert only as a revert commit.

### Plan of run 5: run 4 repeated (written before the run)

Same arms, workload and switches as run 4, 4 fixed rounds (about 55 min), `--load-log on`, `TONA_GPU_GAPS=500`.
**Expectation:** if run 4's −0.87 % at 8000 is real, decode@8000 −0.5 to −1.3 % again and the GPU's busy time per
cycle of the third line higher in B; if it was noise, all within ±0.5 % and busy equal. Decision afterwards with both
runs together (8 rounds).

### Result of run 5 and decision: noise; MSL 3.2 costs nothing measured, the GPU pick on by default

Conditions: before ok; after "do not measure" (GPU 97 % busy, the run's own server), 20 s later ok. Report:
[2026-11-14-mac-mtp-msl32b-server-ab.md](2026-11-14-mac-mtp-msl32b-server-ab.md). Answers identical
(SHA 727ff658ae742fa7).

| Metric | Run 5, B vs A, mean [CI] | Runs 4 and 5 pooled (8 rounds), mean [CI, t with 7 df] |
|---|---|---|
| decode-code | +0.39 % [−0.61, +1.39], open | +0.13 % [−0.32, +0.58] |
| decode-text | +0.19 % [−0.63, +1.00], open | +0.09 % [−0.27, +0.46] |
| decode@8000 | +0.19 % [−1.45, +1.84], open | −0.35 % [−1.27, +0.57] |
| prefill@8000 | +0.08 % [−1.91, +2.06], open | +0.82 % [−0.97, +2.61] |

GPU gaps per cycle, medians: busy 93.33 / 98.64 / 68.02 (A) against 93.25 / 98.69 / 68.06 (B) ms, idle 4.13 / 4.19 /
3.12 against 4.12 / 4.17 / 3.12. Run 4's −0.87 % at 8000 did not come back: the multiple comparison of four metrics,
as suspected. Code and text are inside ±0.6 % pooled; decode@8000 (about 100 tokens per round) is the noisiest metric.

**Decision (coordinator's rule):** decode undecided but the GPU's cycle measurably shorter (run 3) and the library
version alone without a measured cost → `TONA_METAL_GPU_PICK` defaults to 1. The MSL 3.1 variant (per-word sentinels)
is not needed.
