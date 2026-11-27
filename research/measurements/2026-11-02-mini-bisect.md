# Why the mini golden references no longer reproduce (#99)

Date: 2026-11-02. Branch `99-mini-bisect` (from main 4944759). Builds in `work/build/wbis` from the worktree
`.` (one commit after the other). Scripts and prompts come from this branch (main).

## Question

During #96 the mini references under `work/golden/mini/` (decode `csv`, `rust`, `explain`, and `prefill/long.txt`) did
not reproduce on any build tried (`work/build/merged`, `wgold`, `wnb`). The text agrees at the start and then
diverges. The mini model has flat logits, so it reacts to the smallest change in numbers; a change that is invisible
in the text of the big model may show here. Which commit changed the numbers (rule 3), or was it non-determinism
from the start?

## What the sources say about the recording

- References: decode written 2026-11-02 01:28:29 to 01:28:31, prefill 01:57:46 (`ls --full-time`).
- Log `/var/tmp/claude/wmini/run/blockA.log`: record, then three compares, all identical, and "deterministic: 3 runs
  identical" (r1 to r3, 01:16 to 01:28). So the recording was reproducible at that time, on its own build.
- #91 was developed on d5a3e1f (base of `bcf2b13^2`, 00:09), i.e. before #72 (IE_DIAG, merged 01:29), #89 (01:29),
  #88/#93 (01:57). The recorded build was therefore a build of the #91 branch at about 01:07 to 01:28, whose engine
  code equals d5a3e1f (the branch changes no engine code: "no engine change", `2026-11-02-mini-model.md`). The
  binary itself is gone (`work/build/engine` is from 2026-10-29).
- The first golden.sh of the branch (f70b61b) ran one `ie-gpu-gen` process per prompt with a positional prompt; the
  expert profile was off for mini.

## Plan

Candidates in order, one build each, the engine code of that commit: **d5a3e1f** (the base of the recording: the
reference must reproduce here, otherwise the recording is not reproducible and the bisect stops at "non-deterministic
from the start" or "environment"), then e4c1010 (before #91), bcf2b13 (#91), 9aaf6e9 (#92), b196c14 (#90), 87d65f2
(#86), 4944759 (main). Candidates between the first one that deviates and the last one that agrees are bisected
further (`git bisect` order on first-parent merges, then within the merge).

Per candidate: the four mini cases (csv, rust, explain at `-n 200`; the 13K prompt with `--prefill --ctx 16384 -n 64`)
with the arguments of the recording (one `ie-gpu-gen` process per prompt, positional prompt, `--mega`,
`IE_DETERMINISTIC=1`, `IE_PLAN=gpu`, `OMP_NUM_THREADS=6`, no expert profile), compared byte for byte against the
reference. Two runs per candidate, run 1 against run 2 too (non-determinism). The `--prompts-file` path of #90 is
tested separately on main (same process, prompts in turn) against the one-process-per-prompt output of the same
build. Cost: mini runs take 3 s (decode) and 25 s (prefill), about 1.5 minutes per candidate plus a build of 2 to 3
minutes (ccache). All under `/var/tmp/claude/gpu.lock`; 60 s between blocks; no build during a run.

Expectation: d5a3e1f reproduces the reference (the recording said so, three times). The first deviation is a commit
that touched numbers or the memory path: #90 (`reset_for_next_prompt`, only in the `--prompts-file` path, which this
bisect does not use for the candidates), #86 (GTT/userptr; should move no number), or something between d5a3e1f and
e4c1010 (#72 IE_DIAG, #89, #84 stash). I have no strong prior; if every candidate from d5a3e1f on deviates, the cause
is in the environment (a changed variable, the model file or a library), not in the code. The text of a deviation is
read against the reference for the position of the first different byte.

Active tooling: none of `IE_DIAG`, `IE_MEGA_PROF`, a profiler. `IE_DIAG` is unset in the runs; a separate run
on one candidate tests whether it changes the text. No reference is re-recorded and nothing is repaired.

## Results

Cause found after the first candidate; the rest of the plan was run to clear the other suspects.

**The recording used a stale binary.** The recorded text carries "megakernel: 278 steps" (`wmini/run/r2.err`); a build of
d5a3e1f (the base of #91) has 276. The only binary on the machine with 278 steps is `work/build/engine/ie-gpu-gen` of
2026-10-29 02:39, built before IndexCache became the default. That binary reproduces all four references.

| Build (engine code of the commit) | Setting | csv, rust, explain, long (2 runs each) | run 1 == run 2 |
|---|---|---|---|
| `work/build/engine` (2026-10-29, stale) | default | = reference, 8 of 8 | yes |
| d5a3e1f (base of #91) | default | differs, 8 of 8 | yes |
| d5a3e1f | `IE_IDX_SHARE=all` | = reference, 8 of 8 | yes |
| c25f9b2^ (parent of the commit below) | default | = reference, 8 of 8 | yes |
| bcf2b13 (#91) | `IE_IDX_SHARE=all` / default | = reference, 8 of 8 / differs 8 of 8 | yes |
| b196c14 (#90) | `IE_IDX_SHARE=all` / default | = reference, 8 of 8 / differs 8 of 8 | yes |
| 87d65f2 (#86) | `IE_IDX_SHARE=all` / default | = reference, 8 of 8 / differs 8 of 8 | yes |
| 4944759 (main) | `IE_IDX_SHARE=all` / default | = reference, 8 of 8 / differs 8 of 8 | yes |
| 4944759, `--prompts-file` (#90), one process, 3 prompts in turn | `IE_IDX_SHARE=all` | = reference, 3 of 3 | n/a |

**First commit that deviates: c25f9b2** "Share the decode block selection across every second QSA layer by default,
`IE_IDX_SHARE=all` for the old program (#73)", 2026-10-30 15:13. The default is `engine/backend/idx_share.h` (unset: every
second QSA layer shares the selection of the layer before it, line 25 "Every second layer"). It changes the numbers on
purpose (rule 3, measured and accepted in #73: Δ-NLL rule, no task loss), only in the decode program, so the decode
tokens of the mini goldens change; the mini model has QSA layers and flat logits and shows it at byte 544 of the csv
text. With `IE_IDX_SHARE=all` ("bit-identical to the engine before IndexCache") every reference reproduces on every
later commit tried.

**Suspects cleared.**
- GTT/userptr (#86): 87d65f2 and main with `IE_IDX_SHARE=all` = reference; no number moves.
- `reset_for_next_prompt` / `--prompts-file` (#90): b196c14 and main in `--prompts-file` mode (three prompts in one process,
  warm expert cache) = reference. (A first attempt looked different only because I wrote the prompt files with the
  trailing newlines that `$(chat ...)` strips in golden.sh; with the same bytes as golden.sh it is equal.)
- Non-determinism: none in any candidate (all run 1 == run 2, 2 runs per candidate and case).

**Consequence.** The big-model goldens are not affected by this finding (not examined here). Nothing
here shows an unintended number change in the engine: every change is the documented IndexCache default. Needed (not
done, maintainer's decision): record the mini references again with a current build, or set `IE_IDX_SHARE=all` in
`golden.sh` for `GOLDEN_MODEL=mini`.


## Follow-up: guard against stale builds, re-record the mini references (#99)

### Plan

1. Guard: `scripts/build-fresh.sh` (called by `golden.sh` and `golden-server.sh`, record and compare) exits 2 if a
   binary is older than any file of the source directory its build was configured from. Checked without GPU: the pytest
   `tools/tests/test_build_fresh.py`, and by hand: `work/build/engine` (2026-10-29) must give exit 2, the fresh build
   `work/build/w99` must run.
   Why not `ninja -n`: measured here, `ninja -C <build> -n` prints "Re-running CMake..." and stops, also for builds that
   are up to date (`-d explain`: `CMakeLists.txt` and the ROCm cmake files "dirty"), so it never reports on the
   sources; the dry run does not change `build.ninja` (mtime unchanged).
2. Default build of the scripts: stays `work/build/engine` (`just build` builds there, `gpu-status.sh` defaults to
   it); with the guard a stale default stops with exit 2 instead of passing a wrong verdict.
3. Re-record `work/golden/mini/` (gpu `csv rust explain`, cpu plan, prefill `long`, server) with a fresh build of this
   branch (main d71d8b8 plus the guard; no engine change on the branch) in `work/build/w99`. Old references first to
   `/var/tmp/claude/w99/mini-old`. Then `compare` twice in a row per check; all must be identical.
   Commands: `GOLDEN_MODEL=mini scripts/golden.sh record work/build/w99`, with `GOLDEN_PLAN=cpu`, with
   `GOLDEN_MODE=prefill`, and `GOLDEN_MODEL=mini scripts/golden-server.sh record work/build/w99` (`GOLDEN_FORCE=1`,
   it never overwrites).
   Scope and time: mini model, decode 3 s, cpu plan 3 s, prefill 25 s (now about 8 s with passes of width 4), server 19 s;
   about 2 minutes for record, 2 x 1 minute for compare, plus a build of 3 minutes.
   Expectation: the new references differ from the old ones (IndexCache is on by default now, as found above); the
   prefill text equals the one of single steps (shown in #96); `compare` twice in a row: identical in every case.
   Active tooling: `IE_DETERMINISTIC=1` (set by the scripts), `IE_DIAG=1`, no profiler, no expert profile (mini).

### Results

- Guard: `GOLDEN_MODEL=mini scripts/golden.sh compare work/build/engine` and `scripts/golden-server.sh compare
  work/build/engine` (build of 2026-10-29): exit 2, message names `engine/model/qwen4exp.cpp` as newer. The fresh build
  `work/build/w99` passes `build-fresh.sh` and runs. `ninja -n` dry run: see the plan (unusable, `build.ninja` mtime
  unchanged). pytest `test_build_fresh.py`: 5 passed.
- Old references saved to `/var/tmp/claude/w99/mini-old`, then recorded with `work/build/w99`: gpu (csv, rust,
  explain), cpu plan (3), prefill (`long`), server (20 cases). `compare` twice in a row (60 s between the blocks):
  all identical both times, server 20 of 20 (checked once more with a full count).
- Against the expectation: as expected, no surprise.

### Correction of the guard: only what the binary is built from (#99)

The first version compared the binary with every file under `engine/`. A change to a file that `ie-gpu-gen` does not
link (`engine/tests/*.cpp`, `engine/update/`, `AGENTS.md`) makes `just build` rebuild only the test, the binary stays
older and the guard would refuse for good. Now `scripts/build-fresh.sh` asks ninja: `ninja -t inputs <binary>` (141
lines for `ie-gpu-gen`: sources, objects, libraries) and `ninja -t deps` of those objects (headers), plus the
`CMakeLists.txt` and `*.cmake` of the source directory; `/usr` and the ROCm toolchain are skipped. Time: 0.5 s for
`ie-gpu-gen` and `ie-server` together (`ninja -t inputs` itself 2 ms, `-t deps` of everything 15 ms; the rest is the
bash loop over about 2,000 files).
Checked on `work/build/w99`: `touch engine/tests/test_class_lru.cpp` exit 0; `touch engine/gpu/mega_cache.h` exit 2
with the file named; after `just build` exit 0. pytest `test_build_fresh.py`: 9 passed, among them a unrelated file
newer than the binary (fresh) and a source, a header and a CMakeLists.txt newer (exit 2) on a small real CMake/ninja
project.
