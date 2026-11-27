# Golden check of the CET fix (hardened build), 2026-11-26

Rule 2 plan, written before the first run. Results follow below the line `## Results`.

## Plan

**Question:** does linking the hardened HIP programs with libgcc's runtime (`--rtlib=libgcc` instead of compiler-rt,
`engine/cmake/hardening.cmake`) change any output? Correctness only, speed is not measured.

**Scope:** one build, `work/build/cet29` from commit `0a5a3240`: `cmake -S engine -B work/build/cet29 -G Ninja -DTONA_HARDEN=ON`
(no `CMAKE_BUILD_TYPE`, as the development build), targets `tona-server` and `tona-gpu-gen`. Two runs through
`scripts/gpu_queue.py --kind gpu`, one after the other:

1. `scripts/golden.sh compare work/build/cet29` (decode, gpu plan, three prompts, Coder IQ1_M, `TONA_DETERMINISTIC=1`);
2. `scripts/golden-server.sh compare work/build/cet29` (both HTTP APIs, `tona-server` itself, MTP drafter).

The references in `work/golden/` were recorded from unhardened development builds.

**Expectation:** byte-identical output in both runs (zero diffs). Source: the change swaps only the link-time runtime for
host helper functions (integer and half-float conversions); hardening flags alter no device code and no floating-point
host code (AGENTS.md rule 3). Tolerance: none. Any diff is a finding; it is then compared against a hardened build with
the old link (compiler-rt) to separate this change from `TONA_HARDEN` itself.

**Active tooling:** `TONA_DETERMINISTIC=1` (set by the scripts), desktop guard on (the queue waits for monitors off), no
profilers, no `TONA_DIAG`.

## Test conditions before

### Test conditions

- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads
- GPU: AMD Ryzen 9 9950X 16-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 6.98 GiB held by others before the run, 5 % busy before the run
- RAM: 60.5 GiB, 48.0 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.3.4-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.25, 4.97, 3.99
- CPU use by other programs over 0.5 s: 1.4 % of all threads, swap in use 4.19 GiB
- sandbox: no
- Commit: `0a5a3240` (clean tree)

Quiet check before:

    - Quiet check (2026-11-26 16:22:41): do not measure
      - do not measure: 6.98 GiB of 15.9 GiB VRAM are held by others

The machine was not quiet (VRAM held by desktop programs); measured anyway, the check is correctness.

## Results

Build: `cmake -S engine -B work/build/cet29 -G Ninja -DTONA_HARDEN=ON` (commit `0a5a3240`, no `CMAKE_BUILD_TYPE`), targets
`tona-server` and `tona-gpu-gen`. The link line has `--rtlib=libgcc`; `readelf -n` shows IBT and SHSTK; `tools/check_hardening.py` passes.

| Run | Command (through `scripts/gpu_queue.py --kind gpu`) | Result |
|---|---|---|
| 1 | `scripts/golden.sh compare work/build/cet29` | csv, rust, explain: identical (3 of 3) |
| 2 | `scripts/golden-server.sh compare work/build/cet29` | the 34 result files of the groups base, sampled, thinking, reuse and hit are byte-equal to `work/golden/server/` (checked with `cmp`); the script itself exited 1 before the `ssd` group: `server exited early`, `gpu::Mega: VRAM only fits 9 expert slots (minimum 10; in use 5.80 GB)` |
| 3 | `GOLDEN_CASES=ssd scripts/golden-server.sh compare work/build/cet29` | not run: same error with 8 slots; others held 8.0 GiB VRAM, the desktop grew during the session |

Against the plan: the expectation (zero diffs) held for everything that ran. The `ssd` group (checkpoints on the SSD, server
restart) is open: it needs 2 GB more VRAM than the desktop left. It exercises host logic of the prompt cache, not code touched
by this change; to be repeated when the desktop holds less than about 6 GiB. Nothing ran in parallel; no drift in decode text.

## Test conditions after

### Test conditions

- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads
- GPU: AMD Ryzen 9 9950X 16-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 8.04 GiB held by others before the run, 5 % busy before the run
- RAM: 60.5 GiB, 49.7 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.3.4-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.64, 4.48, 4.52
- CPU use by other programs over 0.5 s: 1.7 % of all threads, swap in use 6.56 GiB
- sandbox: no
- Commit: `0a5a3240` (dirty tree)

Quiet check after:

    - Quiet check (2026-11-26 16:29:10): do not measure
      - do not measure: 8.05 GiB of 15.9 GiB VRAM are held by others
      - warning: the system is swapping (3 pages in 1 s)
