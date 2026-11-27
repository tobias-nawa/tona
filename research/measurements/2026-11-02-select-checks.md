# select_checks: the checks chosen from the build graph (#89)

Date: 2026-11-02. Tool: `tools/select_checks.py` (commit decfa7e and following on branch 89-select-checks).

## Plan

Scope: no speed or quality claim, so no rates. Three parts: (1) the tool on the last 15 merges of main
(`git log --merges`), graph of `work/build/merged` (main of 2026-11-02), `--base <merge>^1 --head <merge>`; (2) an independent check of
the graph, the textual `#include` closure of the sources of `ie-gpu-gen` and `ie-server` against the graph; (3) two
real `scripts/gpu-status.sh` runs without `--post` under `/var/tmp/claude/gpu.lock`: one on the branch (scripts, tools,
docs only), one with a test commit that touches `engine/gpu/guarded_copy.h` (reverted afterwards).

Expectation: a merge that changes only documents, `research/` or `bench/` selects no golden check; a merge that changes
`engine/CMakeLists.txt` or a header of the GPU library selects all five; the status run on scripts alone skips all
five, the run with the header runs all five and all are identical (the binaries are the unchanged main build, so any
difference would be a bug of the flow). Duration of the all-golden run: about 631 s (sum of the run times below),
tolerance 10 %. Nothing is expected to be missing from a selection (checked by hand for 5 merges).

Active tooling: none of the engine diagnostics; `IE_DETERMINISTIC=1` is set by the golden scripts themselves. Python 3 with
ninja 1.13.2 and ctest of the system.

## Golden run times

Median of the logs under `/var/tmp/claude/golden.*` and `golden-server.*` (idle machine, whole check including model load):

| Check | Seconds | Source |
|---|---|---|
| golden gpu | 67 | ticket value; logs of the decode check do not tell gpu and cpu plan apart (median of both 71 s, n=79) |
| golden cpu plan | 65 | same |
| golden prefill | 362 | median of 45 complete runs (min 355, max 370), without drafter |
| golden prefill dflash | 30 | median of 42 runs |
| golden server | 107 | median of 54 complete runs (20 results; min 96, max 115) from creation of `server.log` to the last result; the ticket said about 300 s, the logs say 107 s |
| all five | 631 | sum; the first accidental run below took 628 s (10 min 28 s) |

## The 15 merges

Graph: `work/build/merged`. "Old" is the rule before this change: any changed file under `engine/`, `toolchain/`, the golden scripts,
their prompt, the two generators or `scripts/gpu-status.sh` runs all five (631 s). Seconds are golden time only.

| Merge | Subject | Files | Class | Golden checks | New s | Old s | Hints |
|---|---|---|---|---|---|---|---|
| 276f601 | Merge #84: every stash and restore copy has a time limit a | 16 | C8 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | long-context run, pressure series |
| ba03c36 | Merge #87: the prefill golden changed with the IndexCache  | 4 | C1 | none | 0 | 631 | - |
| 645db0b | Merge #83 stage 3 measurements: page migration evicts the  | 2 | C0 | none | 0 | 0 | - |
| a60325c | Merge #83 stage 2: the planner follows a slow CPU | 13 | C8 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | pressure series |
| 8634447 | Merge #83 stage 1: OMP_WAIT_POLICY=passive by default, IE_ | 11 | C8 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | pressure series |
| 7d5b00d | Merge #72: allocation trace and bounds check for the k_moe | 30 | C8 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | long-context run, pressure series |
| 64895f1 | Merge #77 phase 2: no OOM at 1M, a needle test that decide | 21 | C8 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | long-context run, pressure series |
| fb01c0e | Merge #75: barrier limits ignore the time a wave is evicte | 19 | C8 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | pressure series |
| 445e0bb | Merge main (#77 phase 1) into 75-barrier (#75) | 20 | C7 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | long-context run |
| f72c8de | Merge #77 phase 1: YaRN switch (off by default) and long-c | 20 | C7 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | long-context run |
| 817a63d | Merge branch 'main' into 77-long-ctx | 39 | C8 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | long-context run, pressure series |
| c1b42e3 | Merge #80: KV staging in the layer-by-layer prefill | 10 | C7 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | long-context run |
| 8f1539d | Merge #79: warm-up requests (max_tokens 0) and private che | 32 | C8 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | long-context run, pressure series |
| e47d7fd | Merge #76: threshold main run, dose-response of task accur | 20 | C1 | none | 0 | 0 | - |
| 680f7b2 | Merge #78: pipelined gather of k_attn_wmma for the KV cach | 8 | C7 | gpu, cpu plan, prefill, prefill dflash, server | 631 | 631 | long-context run |
| | | | | **Sum** | **7572** | **8203** | |

Golden time saved over the 15 merges: 631 s of 8203 s (7.7 %); one merge less to check on the GPU
(ba03c36 changed only `scripts/gpu-status.sh`).

Why so little is saved: most merges change a header or source of the GPU library, which both golden binaries contain.
10 of the 15 merges also change `engine/CMakeLists.txt`. With the first version of the tool that alone selected everything; after
the review the CMake change is compared (`tools/select_cmake.py`: base and HEAD configured, not built, and the commands of
`ie-gpu-gen` and `ie-server` compared). For these 15 merges the table did not change. The comparison finds the commands of both binaries changed in
3 merges (276f601, 7d5b00d, fb01c0e: new sources), of `ie-server` only in 2 (817a63d, 8f1539d) and of none in 5 (a60325c, 8634447,
445e0bb, f72c8de, 680f7b2: only test targets or options); but every one of these also changes sources that both binaries contain, so
the set stays all five. Of the last 80 non-merge commits of main the tool selects: 46 no golden check (C0, C1, or a CMake change that adds only a test),
27 all five, 6 the server golden only, 1 the four generator goldens only (earlier version: 45, 29, 5, 1; two commits whose only
build change was a test target now need no golden). Examples that select only
the server golden: 571b091 (`engine/server/engine.cpp`), 7d878de (`checkpoint_file.cpp`); only the four generator
goldens: c41b9d8 (`engine/tools/ie_gpu_gen_tail.cpp`).

## Checked by hand: nothing missing

For each of these merges I read the changed files and asked whether any golden check could reach them:

1. 276f601 (#84): changes `engine/gpu/mega.h`, `mega_state.cpp`, `backend/stall.h`, `budget.h` and server files. All in `ie-gpu-gen` and
   `ie-server` (graph), so all five run, as they must. Hints long-context and pressure are right: stash copies and time limits.
2. ba03c36 (#87): only `scripts/gpu-status.sh`. No golden check reads this script, none selected. Correct. (The merge changed
   which goldens the script requires; that is a change to the process, not to the engine's output.)
3. 645db0b (#83): documents and measurements only, C0. Correct.
4. a60325c (#83): `engine/gpu/cpu_cost.h`, `mega.h`, `mega_*.cpp`, `backend/omp_wait.h`, `CMakeLists.txt`. `omp_wait.h` is in both golden
   binaries. All five, correct.
5. 8634447 (#83): `engine/server/*`, `tools/ie_gpu_gen.cpp`, `omp_wait.h`. Both binaries are reached, all five, correct.
6. e47d7fd (#76): only `bench/threshold/*`. `scripts/golden.sh` and `golden-server.sh` do not mention it (grep), no golden check
   selected, C1 with the Python tests. Correct.
7. fb01c0e (#75): `matvec.hip`, `mega.hip`, `matvec_barrier.h`, wait paths: device code, all five and bit-exact, correct.

None of the 15 merges got a partial set, so for them an error could only be too many checks, never too few. The partial
selections come from single commits (list above). I checked two by hand: 571b091 changes `engine/server/engine.cpp`,
`engine.h` and `engine_report.cpp`, which no generator source includes (closure check above), so the server golden alone is
right; c41b9d8 changes `engine/tools/ie_gpu_gen_tail.cpp`, a source of `ie-gpu-gen` only, so the server golden is rightly skipped.

Independent check of the graph (`#include` closure of the compiled sources, quotes resolved against the file, `engine/` and
`engine/third_party/`): `ie-gpu-gen` 147 files in the graph and 147 in the closure, `ie-server` 181 and 181, no file on only one side.

Limit of the table: the graph is that of today's main, applied to older merges. A file that was renamed or deleted since then
is "in no binary" and selects everything (over-selection, never under-selection).

## Real gpu-status runs (no `--post`)

Under `flock /var/tmp/claude/gpu.lock`, 60 s after the previous GPU job, `--build work/build/merged`:

| Run | HEAD changes | Status description | Result |
|---|---|---|---|
| A | docs, scripts, tools only | `C1: no golden check needed; skipped 5` | no GPU used, exit 0 |
| B | plus a comment in `engine/gpu/guarded_copy.h` | `C8: golden identical (gpu;cpu plan;prefill;prefill dflash;server); needs: pressure series` | all five identical, exit 0 |

The test commit of run B was removed again (`git reset --hard HEAD~1`).

An earlier run, before the lock, was an accident of mine: I had committed a stray `.coverage` file, which the tool
rightly treats as an unknown file and answers with all five goldens. It ran for 10 min 28 s without the lock on
the GPU, all five identical (`C5: golden identical (...)`). It confirms the tool's fallback (unknown file: everything)
and the duration of 631 s (628 s measured, within the 10 % tolerance). The file was removed from the commit.

## Findings

- The tool saves nothing where the change reaches the GPU library, which is most engine changes; it saves where it should:
  documents, `bench/`, scripts, server-only and tool-only changes.
- The CMake comparison (approved in the review of #89) works as intended on single commits but saves nothing on these 15 merges,
  because they also change engine sources. Its cost: about 3.5 s per comparison (two configurations, measured over four comparisons: 13.8 s), only when a
  CMake file under `engine/` changed. `git archive` of `engine/` and `bench/corpus/` plus a link to `work/`; nothing is built.
- The golden scripts use no other file of the build than `ie-gpu-gen` and `ie-server` (`$build/ie-gpu-gen` in `golden.sh`,
  `$build/ie-server` in `golden-server.sh`); models, profile and references are under `work/`, the tracked profile under
  `toolchain/`, which triggers everything.
- The server golden takes 107 s, not 300 s; the ticket value was wrong.
