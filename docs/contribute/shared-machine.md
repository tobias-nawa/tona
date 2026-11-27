# Working on a shared machine: rules and traps

Builds, speed runs, quality runs, a desktop and the user's own server often share one machine. Each rule comes from a
failure (date given). Queue and guard: [testing.md](testing.md#speed-comparison-scriptsabpy); run sizes:
[measurement.md](measurement.md).

The rules apply whenever more than one job can touch the GPU, CPU or page cache. On a machine only one agent uses, the
queue is optional; the rest holds: no build or second GPU job during a measurement, no blocking HIP call while the
megakernel waits, stop processes by PID, own port (8662 is this repository's default).

## The queue

- **Every build and every GPU program goes through `scripts/gpu_queue.py --kind <exclusive|gpu|cpu> --label <x> --
  <cmd>`**: `exclusive` for speed measurements, `gpu` for quality, golden and correctness runs, `cpu` for builds and
  CPU-only work.
- **No bypass**, not even to "check a flag": never start `ab.py`, `tona-server` or another GPU program outside the
  queue, never pass `--no-lock`, never set `TONA_GPU_LOCK_HELD` or `TONA_CPU_LOCK_HELD` by hand. Read `--help` or the
  argument checks instead. (2026-11-16/17: a check run outside the queue collided with an exclusive run: out of
  memory, disturbed rounds.)
- **Every git worktree queues at the main checkout's lock** (`git rev-parse --git-common-dir`); no worktree gets its own
  `work/` with its own lock. (2026-11-17: one did and ran GPU jobs beside others.)
- **Never build while a speed measurement runs** (compiling distorts tok/s). The queue orders a build only as a `cpu`
  job.

## Telling when another run is done

- **An empty queue does not mean a run is over.** A long run (scale run, agentic eval) is several queue jobs with gaps.
  Wait for an explicit marker the other run writes after its last job (a `DONE` file in its run directory), not for an
  empty queue or a missing process. (2026-11-18: an agentic eval almost started in a gap of the scale run and would
  have evicted the page cache the next block measured.)
- **Files read during a page-cache measurement change it**: no large reads (traces, models, archives) while an R8 or
  any cold-path run measures.

## The desktop

- A `gpu` or `exclusive` job waits while somebody uses the desktop and is stopped (queue exit 75) when somebody comes
  back; a monitor woken to the lock screen does not count ([testing.md](testing.md), `scripts/desktop_guard.py`). Exit
  75: requeue the same job later (the round is lost, not wrong) and note it in the measurement file. (2026-11-17: a
  queued run took 13.6 of 16.3 GB VRAM while somebody worked; the desktop died with "Not enough memory for command
  submission".)
- `TONA_DESKTOP_GUARD=off` only when the owner of the machine explicitly allows a run beside their work.
- No VRAM-pressure tests (balloons, a second GPU program that fills VRAM) on the GPU that drives the desktop without a
  reserve and the owner's consent. (2026-11-15: such a test ended the desktop session.)
- The desktop's VRAM grows with uptime and open programs (1.4 GB after a reboot, 2 to 3 GB later on the reference
  machine) and long-context prefill depends on the expert slots that fit: write the desktop VRAM and the slot count of
  every server start into the measurement file.

## Processes and files

- **Stop processes by PID only.** `pkill -f <pattern>` matches the shell that runs it when the pattern is part of its
  own command line, and kills your own session. `pgrep -f` has the same trap when you act on its output.
- Test servers on their own port, never on the port of the machine owner's server, and under
  `systemd-run --user --scope -p MemoryMax=<class> -p MemorySwapMax=0`, so a runaway server cannot swap the desktop.
- Large files (models, builds, traces, benchmark data) under `work/` (ignored) or `/var/tmp`, never `/tmp` where it
  is RAM.
- **A build directory belongs to one worktree.** A CMake cache from another worktree points at that worktree's sources
  (configure fails, or the wrong sources build). Give every worktree its own (`just build_dir=<dir>
  installer_dir=<dir2> ...`).
- A golden check refuses a build that is older than its sources ("the build is stale"); rebuild, do not touch the
  timestamps.

## Comparisons

- **A yardstick gets the machine too:** when Tona runs with all physical cores, so does the engine it is compared with
  (`scripts/baseline.sh` takes the physical core count; 2026-11-18: llama.cpp was still pinned to the previous CPU's 6
  threads).
- **Agentic evals with greedy decoding: a retry repeats the answer.** When a model loops and the client times out, it
  resends the same request and a deterministic server answers identically until the task's wall cap. Stop a task at its
  first client timeout and at a wall cap, the same for every arm, and count "loop/timeout" as a model failure, separate
  from infrastructure failures. (2026-11-18: one SWE task repeated an identical 38,846-token loop 11 times in two
  hours.)
