# Release checks 2026-11-10 (pre-release checks, no speed measurement)

Plan, written before the first check ran. Results are read against it below the line `## Results`.

## Plan

**Scope.** Three checks of the release checklist on the branch `43-release-prep` (worktree `tona-release-prep`, base main
2cf7e93; the code differences to main are one repository-name constant, `kReleaseRepo`, and its test):

1. gitleaks over the whole git history (`pre-commit run gitleaks-history --hook-stage manual`, the hook of
   `just check-all`; version pinned in `.pre-commit-config.yaml`). 742 commits on main. One run; the result is
   deterministic.
2. `just test-sanitize` (ASan and UBSan, label `unit`), in an own build folder `work/build/release-prep-sanitize` with
   the flags of the recipe (`-DCMAKE_BUILD_TYPE=Debug -DTONA_SANITIZE=ON -DTONA_MARCH=x86-64-v3`, no model, no
   llama.cpp). One run.
3. `scripts/fuzz.sh 300` (what `just fuzz` runs and what `docs/contribute/testing.md` names): three targets (`api`, `gguf`,
   `minisig`), 300 s each, about 15 min plus compile.

**Expectation.** (1) no finding in the history; a finding would be named by file and line only. (2) all unit tests pass
without a sanitizer report (the CI job `engine-sanitizers` is green on main); duration is the compile of a Debug build
with sanitizers, estimated 5 to 15 min on 12 threads. (3) no crash, leak, timeout or out-of-memory in any target; coverage
counters (`cov`, `ft`) at the end are about those of the earlier runs and only informative. Tolerance: none for the
three pass/fail results; any finding is a finding, not noise.

**Active tooling.** No GPU, no server, no `TONA_DIAG`. Builds under `flock work/gpu.lock`. `just check-all` is not run in
full (it runs clang-tidy and every hook on all files, not asked); `just check` with an own build folder is run for the
changed files. `scripts/fuzz.sh` writes to the shared `work/fuzz` (corpus grows, findings under `work/fuzz/findings`).

## Results

Commit under test: branch `43-release-prep`; checks 1 to 3 ran on the first commit of the branch (repository name, stale
names, plan; code identical to main apart from `kReleaseRepo`). The sanitizer run was repeated at the end on the final
branch state (installer, `fetch` resume, test kit), see below.

| Check | Result | Duration |
|---|---|---|
| gitleaks over the whole history (`pre-commit run gitleaks-history --hook-stage manual`) | passed, no finding (742 commits of main plus the branch) | 1.8 s |
| `just test-sanitize` equivalent in `work/build/release-prep-sanitize` (ASan + UBSan, Debug, `-L unit`) | 43 of 43 tests passed, no sanitizer report; repeated after the last change: 43 of 43 | build 51.5 s (first, full), 18.5 s (incremental); tests 22.8 s |
| `scripts/fuzz.sh 300` (`just fuzz` default): targets `api`, `gguf`, `minisig`, 300 s each | no crash, leak, timeout or out-of-memory in any target; no new file in `work/fuzz/findings/` | 3 x 301 s |

Fuzz details (libFuzzer, `-max_total_time=300 -max_len=8192 -timeout=10 -rss_limit_mb=4096`): `api` 2,091,427 runs (about
6,900 per s), peak RSS 564 MB; `gguf` 11,323,588 runs (about 37,600 per s), peak RSS 492 MB; `minisig` 6,318,845 runs
(20,992 per s), `cov` 2057, `ft` 8376, corpus 1687 inputs, peak RSS 562 MB. Slowest single input 0 s in all three.
`work/fuzz/findings/gguf/crash-eb2b7b1b...` is the file of 2026-10-28 (before the fix of #68); the saved input runs
without a crash on the current binary.

Against the plan: all three results inside the expectation (no finding). The sanitizer build took 51.5 s, far below the
estimate of 5 to 15 min (ccache warm, 12 threads); the estimate was too pessimistic, not a finding. The shared
`work/fuzz` corpus was extended by this run (about the same size as before; not compared).

Not run, on purpose: `just check-all` (clang-tidy on every file, not asked), GPU goldens and `ab.py` (no code of
`tona-server` changed: `tona_update` is linked only by `tona-top`, the installer and the update tests, see
`engine/CMakeLists.txt`), `scripts/isa-diff.py` (same reason: the HIP programs are not rebuilt differently).
