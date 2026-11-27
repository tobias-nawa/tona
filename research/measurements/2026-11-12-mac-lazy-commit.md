# Mac: commit(m < nb) replays only the DeltaNet recurrence (#108)

M1 Max 32 GB, Metal backend, target DM, MTP head cut to 164 experts.

## Plan (written before the run)

**Question.** What does replaying only the DeltaNet recurrence after a partial commit (instead of a second whole pass
of the m committed tokens) bring the server? Behind it: 2026-11-12-mac-commit-replay.md (23 % of the verify passes
commit partially, their second passes ~11 % of the decode time).

**Arms.** A = `e6a556c` (merge of main) without the change (the six Metal files of `4c4fbac`), binaries in
`work/build/mtl-base`; B = `e6a556c` with it (`work/build/mtl`). Both with the same pass costs (`fa6cd48`).

**Scope.** `scripts/ab.py --a work/build/mtl-base --b work/build/mtl --workload set --ctx 16384 --model DM
--drafter MTP-164 --memory-max 40G` (first use of its macOS path, 2b1d282: resident-size guard instead of a systemd
scope; 40G because the resident size counts the mapped model files' resident pages, the n-gram table included).
Rounds AB, BA, ... until the result is clear (ab.py's rule), at most 6; one discarded warm-up start of A. ~4 min per
server start: 15 to 50 min.

**Expectation.** Greedy: identical texts in both arms (the change is bit-identical). Decode code +6 to +12 %
(28.4 -> 30 to 32 tok/s), text +3 to +10 %; the second passes were ~11 % of the round, the replay costs ~1 ms.

**Active tooling.** None in the servers (`ab.py` passes no `TONA_*`). `caffeinate -dimsu`. Quiet check and
`tools/hw_report.py` before and after.

Correctness before the run: widths with random partial commits bit-identical (mini seeds 7, 3, 19: 0 of 128, 147,
156 rows; 2 slots 0 of 292; DM 0 of 109), mini 207 of 207, DM 39 of 39, cut invariance byte-equal, prefill against
decode as before, MTP golden 241 of 256 (median 5.83e-2).

## Results

Quiet check before (09:44 and 09:45): warning both times, load average 3.1 and other programs 9.1 to 9.2 % of all CPU
threads (none of it from this work; measured anyway, the arms alternate under the same load); `tools/hw_report.py`:
M1 Max, 32 GiB, macOS 27.0.1, no `TONA_*` variable, load 3.12 / 3.94 / 3.63, commit `e6a556c` (dirty tree). After
(10:01:27): "GPU 96 % busy", the run's own tail. Raw: [2026-11-12-mac-lazy-commit.json](2026-11-12-mac-lazy-commit.json)
(ab.py's raw data).

Deviation from the plan: ab.py's `set` workload always runs at context 32768 (`scripts/ab_workloads.py`), not the
16384 of the plan; both arms alike (VRAM 25.35 GB, inside the 26.8 GB working set). ab.py compares the texts only for
its own server requests, not for bench-set; the bit-identity of the change rests on the width check above.

| Metric | A per round | B per round | B against A, mean [95 % CI] | Expected |
|---|---|---|---|---|
| decode code | 27.5 / 28.1 tok/s | 30.7 / 30.8 | **+10.6 % [+8.8, +12.4]** | +6 to +12 % |
| decode text | 19.6 / 20.3 | 23.6 / 23.8 | **+18.8 % [+16.3, +21.4]** | +3 to +10 %: finding |

Rounds AB +11.6 / +20.4 %, BA +9.6 / +17.2 %; decided after 2 rounds (alpha 0.05, margin 0.5 %), 15.6 min.

- Code inside the expectation. **Text gains more than expected**: text accepts fewer drafts, so more of its passes
  committed partially and paid a second pass; the expectation took the share over all passes (11 %) for both kinds.
- First use of ab.py on macOS (resident-size guard, 2b1d282): five server starts, none ended by the guard
  (`--memory-max 40G`), stops by PID as on Linux.
- The server now decodes code at ~30.8 tok/s and text at ~23.7 with the MTP head (trunk alone 20.4).
