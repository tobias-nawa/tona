# R0 baseline for the multi-architecture refactor (2026-11-15)

Ticket R0 of `docs/architecture/model-architectures.md` (section 7). Every later ticket of wave 1 compares against the
artifacts made here: the machine code (C1), the op programs (C2) and the goldens (C3). No speed is measured; the pass
curve (C4) is made from the same commit only by a ticket that needs it.

## Plan (written before the runs)

Commit: `64a8204` (`main`, 2026-11-15), built unchanged in `work/build/base`.

| Step | What runs | Expectation | If it differs |
|---|---|---|---|
| 1 Base build | full engine build in `work/build/base` | builds; `tona-server`, `tona-gpu-gen` exist | fix the environment, not the code |
| 2 Program dumps | `scripts/program-dump.sh work/build/base work/base-dumps`: `TONA_MEGA_DUMP_PROGRAM` of tona-server with the MTP head, tona-server with the DFlash2 drafter, and tona-gpu-gen with `TONA_PLAN=cpu TONA_SEQS=2`; `TONA_EXPERT_SLOTS=40` so pointers do not depend on free VRAM | three files; target, MTP and drafter programs present (`tasks ops`, `tasks mops`, `tasks mkops`, `tasks dops`) | a missing program: the script starts the wrong mode, fix the script |
| 3 Dump reproducibility | the same script into `work/base-dumps-2`, `cmp` per file | byte-equal (pointers are printed as region and offset, `docs/use/environment.md`) | a dump that is not reproducible cannot serve C2: find the varying field, mask it in the script |
| 4 Goldens stale? | `scripts/golden.sh compare` (gpu, `GOLDEN_PLAN=cpu`, `GOLDEN_MODE=prefill`, `GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash`) and `scripts/golden-server.sh compare` (MTP) against `work/golden/` with the base build; one run each | identical (references recorded 2026-10-27 and 2026-11-12 from earlier commits; any change of tokens since then would have been recorded with its commit) | a differing golden is stale: record it again from this build (`record`, `GOLDEN_FORCE=1` for the server), name the commit; the cause is not analysed here |
| 5 C1 of a rebuild | second build of the same commit in `work/build/base-check`, `scripts/isa-diff.py work/build/base work/build/base-check` | `all kernels identical` | the build is not reproducible, so C1 would give false alarms: find why before wave 1 goes on |

Active tooling: `TONA_DETERMINISTIC=1` in the goldens, `TONA_MEGA_DUMP_PROGRAM` in the dumps, `TONA_ALLOC_TRACE` in the
server golden (set by the script). No profiler, no diagnostics, no test switches otherwise. Expected wall time: build
about 10 minutes; dumps 3 x 1 minute; goldens (queue waits excluded) gpu 3 minutes, cpu 3, prefill 10, prefill-dflash 4,
server about 15; rebuild 10.

Quiet machine: these checks are about bits, not speed, so a busy machine does not falsify them; the conditions are
recorded anyway. Other workers' builds and GPU jobs run in the same line (`scripts/gpu_queue.py`), one at a time.

## Test conditions

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.14 GiB held by others before the run, 0 % busy before the run
- RAM: 60.5 GiB, 52.0 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 5.1, 2.35, 1.7
- CPU use by other programs over 0.5 s: 75.5 % of all threads, swap in use 1.1 GiB
- sandbox: no
- Commit: `64a8204` (dirty tree)

Quiet check before (09:47, taken while the base build of this ticket and builds of other workers' tickets were in the queue): "do not
measure" (55 % of the CPU threads used by other programs, load average 6.9, 1.14 GiB of VRAM held by the desktop). After (10:47, with
a GPU job of another worker running next to the last golden): "do not measure", DISTURBED (8.9 % CPU, 6.2 GiB VRAM held by others, 2
pages swapped in 1 s). The machine was not quiet; the runs of this ticket compare bits (goldens, machine code, program dumps), which a
busy machine does not falsify. No speed is claimed.

## Results

Commit `64a8204`, build `work/build/base` (CMake default, ROCm 10.1 toolchain of `work/rocm-10.1`).

| Step | Result | Expected | Read against the plan |
|---|---|---|---|
| 1 Base build | built, `tona-server` and `tona-gpu-gen` exist | builds | as expected |
| 2 Program dumps | `work/base-dumps/server-mtp.txt` (3.1 MB: `ops`, `mops`, `mkops`), `server-dflash.txt` (2.7 MB: `ops`, `dops`), `cpu-seqs2.txt` (3.1 MB: `ops` for two sequences, host-wait variants); 40 expert slots | three files with the target, MTP and drafter programs | as expected |
| 3 Reproducibility | second run `work/base-dumps-2`: `cmp` byte-equal for all three | byte-equal | as expected |
| 4 Goldens | identical on the base build: gpu (csv, rust, explain), cpu plan (3), prefill (long), prefill-dflash (long), server mtp (36 cases: both APIs, streaming, tools, sampling, thinking, reuse, ssd, hit and cold pairs) | identical; stale ones to be recorded again | none stale, nothing re-recorded |
| 5 C1 of a rebuild | `scripts/isa-diff.py work/build/base work/build/base-check` (second full build of the same commit in another directory): 217 functions in A and B, `all kernels identical` | identical | as expected: the build is reproducible, C1 gives no false alarms |

The server golden ran with the default TONA_DETERMINISTIC and TONA_PLAN settings of `scripts/golden-server.sh`. Not run here: the
server golden with the dflash drafter (the guard of section 7 names `golden-server.sh` once; the MTP drafter is the production default).

## What the later tickets use

- C1: `scripts/isa-diff.py work/build/base work/build/<ticket>`.
- C2: `scripts/program-dump.sh work/build/<ticket> <folder>`, then `cmp <folder>/<name>.txt work/base-dumps/<name>.txt` for the three names.
  The dump depends on the number of expert slots (fixed to 40 by the script) and on the code of the programs only.
- C3: `work/golden/**` is current as of this commit.
- C4: not produced here; a ticket that needs it runs the base build `work/build/base` against its own build in the same session.
- `work/build/base` was configured in the worktree of this ticket (its CMake cache names that path as the source); it must not be rebuilt
  or checked with `scripts/build-fresh.sh`. Keep it until the last ticket of wave 1 is merged.

