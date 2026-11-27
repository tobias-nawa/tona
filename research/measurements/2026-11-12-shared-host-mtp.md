# Shared MTP host code (vocabulary map, bookkeeping, scorer cache), 2026-11-12

Status: **run on 2026-11-12, results below.** Verdict: tokens, goldens and SHAs identical, no measurable change in speed.

Refactor, step 2 of the shared host code ([research/47](../47-backend-shared-host-code.md)): `engine/backend/mtp_vocab.h`
(list and ring of the truncated head), `engine/backend/mtp_track.h` (which h is known, catch-up chunks, invariants),
the scorer cache moved from `engine/gpu/` to `engine/backend/`. The reference calls them; the Metal files call them too
(not built here). Rule 3 (AGENTS.md): identical tokens, identical goldens, unchanged speed.

## Test conditions

`tools/hw_report.py` before the runs (saved with the runs): Ryzen 5 9600X (6 cores / 12 threads), RX 9060 XT gfx1200
15.9 GiB VRAM, Fedora 44, kernel 7.2.15, HIP 7.16.26385, no `TONA_*` set. Other helpers use the same machine through
the GPU queue (`scripts/gpu_queue.py`), so the GPU is free while a run of ours holds the lock, but CPU and RAM are
shared: the quiet check before the runs warned (load from other work, 1.5 GiB VRAM held by the desktop); measured
anyway, available resources named in the results.

## Test conditions of the runs

`tools/hw_report.py` before each A/B run (identical in substance): Ryzen 5 9600X, RX 9060 XT gfx1200 15.9 GiB VRAM with
1.36 GiB held by the desktop, 0-1 % GPU busy, RAM 60.5 GiB with 55.7 to 56.1 GiB available, swap in use 1.77 GiB, load
average 1.6 to 1.9, CPU use by other programs 0 to 2 %, rocm 10.1.0, commits: A `126e0c7`, B `45349fa` (runs 3 and 4
first part) / `bcc7977` (second production run; only the text of one error message differs from `45349fa`). Quiet
check before and after every run: warning only for the 1.36 GiB of the desktop; measured anyway. Another helper used the
machine between our runs (through the queue, never at the same time).

## Plan (written before the speed runs; the golden checks ran earlier, see below)

### Scope

1. Unit: `test_mtp_shared` replays random token streams through a literal copy of the old ring (reference
   `mtp_sync_vocab`) and the new `MtpVocabMap`: rows, order of the row changes, first/count, scan, bounds; the old and
   the new `h_of`, the chunks of the catch-up loop (tokens, h sources, want flags, stale count) and the integers after
   save, load, unstash and reset. Expectation: all equal. Ran first (cheap, no GPU).
2. Goldens (bit-exact, `TONA_DETERMINISTIC=1` inside the scripts), reference outputs recorded from main:
   `scripts/golden.sh compare`, `GOLDEN_PLAN=cpu`, `GOLDEN_MODE=prefill`, `scripts/golden-server.sh compare` (all
   groups, port 8099). Expectation: every case identical. These ran before this plan file was written; their
   expectation (identical) was fixed by rule 3, not by a number.
3. Bit-exact A/B, `scripts/ab.py` A = main (126e0c7, `work/build/tona127b-base`), B = this branch, `TONA_DETERMINISTIC=1`,
   workloads `mid` (30K prompt) and `long` (130K prompt): SHA of every text equal between the arms and equal to
   `bench/sha.toml`. Why this size: the host paths touched run at a fill after a long prompt (catch-up chunks, ring
   after tens of thousands of context tokens, `mtp_scan`), which only `mid`/`long` reach. `mtp_dyn` (rows loaded on demand,
   `/status` and the server log line per request) must be equal per request: it counts the ring.
4. Speed in production mode (no `TONA_DETERMINISTIC`, drafts of up to 7), workload `short`, same arms. Why: one fill per
   decode round runs the new code; `short` is the cheapest workload with MTP drafting at 400 decoded tokens.

### Expectation

- Tokens, SHAs, `mtp_dyn`: identical (rule 3). Source: the code is a move; the unit test pins the logic.
- Decode and prefill rates: B against A within -1 % .. +1 %. Source: `bench/noise.toml` (sd of a paired difference
  0.23 % for `decode@2000`, 0.6 % for `decode@2000:write`, 0.28 % for `decode@30000`, 0.02 % for `prefill@130000`); the
  code runs once per decode round and loops over the new tokens, no change in the work.

### Active tooling

`TONA_DETERMINISTIC=1` in run 3 only. No `TONA_DIAG`, no profiler, no alloc trace beyond the defaults of the
server (`TONA_ALLOC_TRACE` is set by `golden-server.sh`). `--round-warmup` default of `ab.py`.

## Results

1. **Unit** `test_mtp_shared`: ok (token lists against Metal's old loader incl. a list file with duplicates, a negative
   id and an id outside; 6 ring replays of 400 steps each with resets, shrinking histories and bursts longer than the
   ring; 12,000 random states of `h_of`, the catch-up chunks and the stale count; save/load/unstash/reset). All 48 unit
   tests of the build pass (`ctest -L unit`, 22 s), before and after the last rebuild.
2. **Goldens** (all identical): `golden.sh compare` csv, rust, explain; `GOLDEN_PLAN=cpu` the same three;
   `GOLDEN_MODE=prefill` long; `golden-server.sh compare` 36 cases (base, sampled, thinking, reuse, ssd, hit) on port 8099.
   After the last rebuild `golden.sh compare` again: identical.
3. **Bit-exact A/B, `mid,long`, `TONA_DETERMINISTIC=1`, 6 rounds** (AB/BA alternating, 3.9 min per round). The raw
   report was overwritten by the next run under the same name; the numbers below are from its console output.

   | Metric | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
   |---|---|---|---|---|
   | prefill tok/s @30000 | 1879.9 (1869.6-1883.1) | 1877.1 (1860.9-1884.5) | -0.15 % [-0.42, +0.12] | open |
   | decode tok/s @30000 | 70.0 (69.2-70.3) | 70.0 (69.3-70.3) | -0.01 % [-0.35, +0.34] | equal after round 4 |
   | prefill tok/s @130000 | 1742.2 (1737.6-1743.6) | 1741.1 (1731.3-1741.5) | -0.13 % [-0.35, +0.09] | open |
   | decode tok/s @130000 | 62.4 (61.9-62.8) | 62.1 (61.4-62.4) | -0.42 % [-1.20, +0.37] | open |

   All four within the expected -1 to +1 %. SHA of the 30K text `ee58880d3f5b558f` and of the 130K text
   `7d2996546076d1b0`: equal between A and B in all 6 rounds and equal to `bench/sha.toml`; expert hits and misses per
   request identical (71,989 / 7,316 and 76,962 / 6,639).
4. **`short`, deterministic, `TONA_DECODE_DIAG=1`, 6 rounds** (`work/ab/shared-host-mtp-detshort.*`): SHAs equal
   (`11eacf1ee10017b4`, `a6f07fd6fb86e448`); in every round and both arms the per-request counters are identical:
   `mtp_stale 0`, `mtp_dyn 34` (2K summary) and `0` (write), hits, misses, prefetched, launches. Rates: decode@2000 +0.61 %
   [-0.85, +2.07], decode@2000:write +0.02 % [-0.61, +0.65], prefill@2000 -0.47 %, prefill@2000:write +1.34 % [-0.59, +3.27]
   (**outside the expectation, finding**: prefill of a 2K prompt is noisy here, A alone ranges from 1545 to 1609 tok/s in
   its six rounds and 5 of 6 round differences are flagged outlier or off plan; the prefill kernels are not touched,
   `prefill@30000` and `@130000` of run 3 are equal).
5. **`short`, production mode, `TONA_DECODE_DIAG=1`, 6 rounds**, twice (`work/ab/shared-host-mtp-prod.*`, `-prod2.*`;
   SHAs equal, hits and misses equal):

   | Metric | run 1 B vs A, mean [CI] | run 2 B vs A, mean [CI] |
   |---|---|---|
   | decode tok/s @2000 (summary) | **-0.71 % [-1.32, -0.10]** (B slower after round 6) | +0.21 % [-0.21, +0.64] |
   | decode tok/s @2000:write | +0.15 % [-0.53, +0.82] | +0.20 % [-0.70, +1.09] |
   | prefill tok/s @2000 | +0.28 % (equal) | -0.59 % [-2.04, +0.86] |
   | prefill tok/s @2000:write | -0.19 % [-0.84, +0.46] | -0.68 % [-4.10, +2.73] |

   Run 1 flagged decode@2000 slower (-0.71 %, five of six rounds negative, inside the expected band); run 2 on the
   binary of the final commit (rebuilt, same code) gave +0.21 % with the same noise table. The two runs together do not
   show a slowdown; the first result is not explained by a code path (the code runs once per MTP fill and loops over the
   same tokens as before), so it counts as noise until a third run says otherwise. "Ungeprueft": a cause in the code
   layout is not excluded.
6. **One 30K prompt, deterministic, DIAG, 2 rounds** (`work/ab/shared-host-mtp-dyn.*`): SHA equal, `mtp_dyn` 6 and
   `mtp_stale 0` equal in both arms and rounds. Limit: the counters cover the decode part of the request only (the rows
   taken at the prefill are before the snapshot), so the long-context ring is covered by the unit test and the SHAs, not
   by this counter.

## Findings

- Reference `mtp_fill` sets `hlast_pos` to `pos - 1` also when h was unknown there (a stale row, counted nowhere);
  Metal sets it only when h was found. Kept as it was in the reference (`after_fill(..., true)`, bit-identical drafts);
  `MtpTrack::after_fill` takes the flag, Metal passes the real value. Whether the reference can reach the case is
  ungeprueft (save_state before the first fill, then load_state and fill would do it).
- `golden-server.sh` took a fixed port 8091; it now takes `GOLDEN_PORT` (needed while another helper tests on 8091).
