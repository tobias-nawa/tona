# Load log on Linux: A/A of the main build with `--load-log on` (2026-11-14)

## Plan

Written before round 1 by `scripts/ab.py` (its plan block, copied here) and in the run script; the machine was rebooted at
18:44 (no swap, 56.9 GiB available).

- Question: does the Linux load log (`scripts/ab_load_linux.py`, e76184f) change the decode rates, and are its columns
  sane on the reference PC? Only then may `AUTO` in `scripts/ab_load.py` include Linux.
- Scope: `scripts/ab.py --a engine --b engine --workload set --ctx 32768 --load-log on`, 4 rounds fixed (AB BA AB BA),
  one discarded warm-up start, about 30 min estimated (10.7 min measured). Build `work/build/engine` at e76184f.
- Why this size: bench-set is the production decode; 4 paired rounds resolve about +-1 %.
- Expectation: B against A -1 to +1 % for decode code and text (A/A, no difference by construction); absolute decode
  code 87.0 to 90.0 tok/s (arm A of 2026-11-13-pc-q6k-tile.md: 88.7; 2026-11-10-scale-run.md c1). The log samples every
  2 s at about 33 ms of one core (its own measurement in the ticket), so a cost below the noise is expected.
- Active tooling: the load log in both arms, `TONA_EXPERT_PROFILE=work/profile/experts-coder.bin`, no profiler.

## Test conditions

Before the run:

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.35 GiB held by others before the run, 4 % busy before the run
- RAM: 60.5 GiB, 56.9 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- TONA_* variables: `TONA_GPU_ARCH=gfx1200` `TONA_GPU_LOCK_HELD=1` `TONA_HIPCXX=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/amdclang++` `TONA_HIPFLAGS=-x hip --offload-arch=gfx1200 -isystem <repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/include` `TONA_HIPLDFLAGS=-L<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib`
- cgroup limits of this shell: MemoryMax=54.0G, MemorySwapMax=0.0G
- Load average (1, 5, 15 min): 1.59, 0.94, 0.4
- CPU use by other programs over 0.5 s: 0.8 % of all threads, swap in use 0.0 GiB
- sandbox: no
- Commit: `e76184f` (clean tree)
- Quiet check (2026-11-14 18:48:54): warning
  - warning: 1.35 GiB of 15.9 GiB VRAM are held by others

After the run:

- Load average (1, 5, 15 min): 1.46, 1.39, 0.96
- CPU use by other programs over 0.5 s: 0.2 % of all threads, swap in use 0.0 GiB
- sandbox: no
- Commit: `e76184f` (clean tree)
- Quiet check (2026-11-14 18:59:38): warning
  - warning: 1.38 GiB of 15.9 GiB VRAM are held by others

## Results

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 91.3 / 91.6 / 91.5 / 91.5 | 91.6 / 91.7 / 91.6 / 91.5 | 91.5 (91.3-91.6) | 91.6 (91.5-91.7) | +0.14 % [-0.32, +0.60] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 65.7 / 65.8 / 65.8 / 65.8 | 65.7 / 65.8 / 65.5 / 65.7 | 65.8 (65.7-65.8) | 65.7 (65.5-65.8) | -0.15 % [-0.63, +0.32] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +0.33 % | +0.00 % | 143 |
| 2 | BA | +0.11 % | +0.00 % | 142 |
| 3 | AB | +0.11 % | -0.46 % | 143 |
| 4 | BA | +0.00 % | -0.15 % | 143 |

Load per server start (percent of one core or of the GPU time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 166.4 | 9.9 | 4.4 | 0.0 | 0.0 | 10.9 | 27512.4 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | - | 18.0 |
| 2 | A | 167.9 | 9.5 | 6.6 | 0.0 | 0.0 | 15.5 | 27310.4 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 18.0 |
| 3 | B | 166.7 | 8.9 | 1.9 | 0.0 | 0.0 | 2.7 | 27230.4 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 18.0 |
| 4 | B | 166.6 | 9.1 | 2.0 | 0.0 | 0.0 | 2.9 | 27254.3 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 18.0 |
| 5 | A | 166.9 | 9.0 | 1.7 | 0.0 | 0.0 | 2.5 | 27278.2 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 18.0 |
| 6 | A | 166.9 | 9.0 | 2.0 | 0.0 | 0.0 | 2.9 | 27237.2 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 18.0 |
| 7 | B | 167.8 | 9.1 | 3.5 | 0.0 | 0.0 | 8.1 | 27272.1 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 18.0 |
| 8 | B | 167.2 | 9.2 | 3.0 | 0.0 | 0.0 | 3.2 | 27287.4 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 18.0 |
| 9 | A | 167.5 | 9.3 | 2.0 | 0.0 | 0.0 | 3.0 | 27477.2 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 18.0 |

Reading against the plan:

- **A/A within the expectation**: decode code +0.14 % [-0.32, +0.60], text -0.15 % [-0.63, +0.32].
- **Finding: the absolute decode is above the expected range**: decode code 91.5 tok/s (91.3 to 91.7 over all 8 starts)
  against 87.0 to 90.0; text 65.7 to 65.8. The log costs nothing measurable here (the rate is higher than every run without
  it). Cause of the higher rate unchecked: this is the first run after a reboot (before it, 31 GB of RAM sat in the amdgpu
  page pool, `nr_gpu_reclaim`, and 5.4 GB in zram swap), and the scale-run plan of today notes an unexplained 6 % spread
  between two runs of 11-13 (94.2 and 88.7). The scale run of tonight (block c1) gives a second value on the same boot.
- **Finding: "GPU ours" is always 0.0.** The server computes through ROCm user-mode queues (`/dev/kfd`); DRM fdinfo
  (`drm-engine-*`) counts only work submitted through the render node, so our compute never shows. "GPU other" (2.5 to
  15.5 %) is the desktop's render work. CPU ours (166 to 168 % of one core), system (8.9 to 9.9 %) and other (1.7 to 6.6 %)
  are plausible. Fix before `AUTO` includes Linux: a source for the compute time of KFD processes (candidates:
  `/sys/class/kfd/kfd/proc/<pid>/` per-process files, or the device total `gpu_busy_percent` minus the fdinfo of the
  others); unchecked which one is usable without root.
- page-ins 27.2 to 27.5 GiB per start: the server reads the model at every start (its scope's MemoryMax of 54 GB includes
  the page cache); memory level min 50 %, no swap traffic.

Decision: `AUTO` stays macOS only until "GPU ours" works on Linux.

Raw data: `work/ab/ab-aa-loadlog.json` (not in the repository).
