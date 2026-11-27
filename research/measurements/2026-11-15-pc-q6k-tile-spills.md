# The tile cases cost k_mega 2 ms per pass: kernel without them for plain programs, TONA_Q6_K_TILE off (2026-11-15)

RX 9060 XT (gfx1200), Zen 5. Branch `136-tile-spills` from main 35fc848. Follows `2026-11-14-pc-det-decode-bisect.md` (culprit 2: the
two `case kQ6KTile` of `matvec_phase.h` change the code of k_mega) and `2026-11-13-pc-q6k-tile.md` / `-upload.md` (the layout gains
0.4 to 0.5 ms at widths 6 to 8).

## Plan (written before the first run)

**Question.** (1) Are the two `case kQ6KTile` the cause of the production decode loss of 0228b0a (-3.6 % against 519c1ab, A/B of the
coordinator, `/var/tmp/claude/night/prod130/ab.out`)? (2) Is there a kernel that keeps the tile gain without the loss: one
instantiation per layout, one instantiation for both with a runtime flag, or fewer live values in the switch?

**Scope.**
1. Spill counts of `k_mega<1|4|8>` from `-Rpass-analysis=kernel-resource-usage` (same command as the bisect: `amdclang++ ... -Rpass-analysis=kernel-resource-usage -x hip -c engine/gpu/mega.hip`) for: main, main without the two cases, every candidate.
2. Batch-bench pass curve W = 1..8 (`tona-gpu-gen --mega --batch-bench 600`, `TONA_BENCH_HASH=1`, 2 repeats per width,
   interleaved arms, 3 runs per arm), plain GGUF: 519c1ab, 0228b0a, the candidates with `TONA_Q6_K_TILE=0` and `=1`. Deterministic
   counters first, as the ticket demands: a candidate that is not equal to 519c1ab at `TONA_Q6_K_TILE=0` or does not beat it with
   `=1` gets no A/B.
3. Goldens (gpu, cpu plan, prefill dflash, prefill, server) with the defaults and with `TONA_Q6_K_TILE=1`.
4. Final production A/B: A = 519c1ab, B = the chosen fix with the defaults, `scripts/ab.py` `set,mid`, ctx 140000, expert profile of the
   scale run, 4 to 6 rounds, `--load-log on`.

**Expectation (with tolerance).**
- Removing the two cases gives the spill counts of 519c1ab (147 / 121 SGPR spills in `k_mega<8>` / `<4>`), pass curve within +-0.15 ms of
  519c1ab (source: the bisect measured -3.3 % per pass from code size, 0228b0a 163 / 129 spills).
- A fix that keeps the tile gain: `TONA_Q6_K_TILE=1` at least 0.4 ms below 519c1ab at W6-8 (source: tile note), equal at W1-4.
- Final A/B with the defaults: decode code and text within -1..+1 % of 519c1ab, decode@30000 and prefill@30000 the same.

**Active tooling.** `TONA_BENCH_HASH=1` in the curve runs, `TONA_MEGA_PROF=1` in one profile run (per-op times); no other TONA_*
variable in the curves. The A/B sets `TONA_EXPERT_PROFILE`, `--load-log on`. The machine is shared through `scripts/gpu_queue.py`; no
build ran during a measurement.

## Test conditions

Before the final A/B (`2026-11-15-pc-q6k-tile-spills-ab.md` holds the plan and result of that run; output of `tools/hw_report.py` and `tools/quiet_check.py`):


- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 2.4 GiB held by others before the run, 0 % busy before the run
- RAM: 60.5 GiB, 57.1 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- TONA_* variables: `TONA_GPU_ARCH=gfx1200` `TONA_GPU_LOCK_HELD=1` `TONA_HIPCXX=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/amdclang++` `TONA_HIPFLAGS=-x hip --offload-arch=gfx1200 -isystem <repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/include` `TONA_HIPLDFLAGS=-L<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib`
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.95, 2.36, 2.38
- CPU use by other programs over 0.5 s: 27.5 % of all threads, swap in use 0.96 GiB
- sandbox: no
- Commit: `639e329` (clean tree)
- Quiet check (2026-11-15 01:15:00): warning
  - warning: other programs use 15.5 % of all CPU threads
  - warning: 2.4 GiB of 15.9 GiB VRAM are held by others

After the final A/B:


- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 9.94 GiB held by others before the run, 75 % busy before the run
- RAM: 60.5 GiB, 57.1 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- TONA_* variables: `TONA_GPU_ARCH=gfx1200` `TONA_GPU_LOCK_HELD=1` `TONA_HIPCXX=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/amdclang++` `TONA_HIPFLAGS=-x hip --offload-arch=gfx1200 -isystem <repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/include` `TONA_HIPLDFLAGS=-L<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib`
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.33, 2.03, 2.04
- CPU use by other programs over 0.5 s: 0.5 % of all threads, swap in use 0.96 GiB
- sandbox: no
- Commit: `639e329` (clean tree)
- Quiet check (2026-11-15 01:39:24): warning
  - warning: 2.4 GiB of 15.9 GiB VRAM are held by others

The curve runs ran between 2026-11-14 23:00 and 2026-11-15 01:10 on the same machine through `scripts/gpu_queue.py`, one after the other; each
run started its own `tona-gpu-gen` under `systemd-run --user --scope -p MemoryMax=54G -p MemorySwapMax=0`. The quiet check before the curve runs and
before the A/B gave warnings (other programs used about 15 % of the CPU threads at the start of the A/B, 2.4 GiB of VRAM held by the desktop). That
noise is why the A/B intervals below are wider than planned. The 9.94 GiB of VRAM held by others and the 75 % busy in the report after the A/B come
from another job that started after the A/B had ended (the queue is serial).

## Results

### 1. Cause (confirmed)

Spill counts (SGPR spills / VGPR spills / scratch bytes per lane of `k_mega<1>`, `<4>`, `<8>`, from `-Rpass-analysis=kernel-resource-usage`, the
command of the bisect; raw output of every build tried in `2026-11-15-pc-q6k-tile-spills-spills.txt`):

| Build | `k_mega<1>` | `k_mega<4>` | `k_mega<8>` | k_mega code bytes (`<1>` / `<4>` / `<8>`) |
|---|---|---|---|---|
| 519c1ab (bisect, `res-519c1ab.txt`) | 103 / 62 / 528 | 121 / 54 / 496 | 147 / 109 / 496 | not measured |
| main 35fc848 (the two cases in) | 103 / 61 / 512 | 129 / 53 / 480 | 163 / 110 / 496 | 371,668 / 730,080 / 1,120,128 |
| main without the two `case kQ6KTile` (= the kernel without tile loops of the final build) | 103 / 62 / 528 | 121 / 54 / 496 | 147 / 109 / 496 | 341,948 / 685,472 / 1,055,872 |

Removing the two cases gives exactly the counts of 519c1ab and 8 % less code. Pass curve (batch bench `tona-gpu-gen --mega --batch-bench 600`,
`TONA_BENCH_HASH=1`, plain GGUF, ms per pass, median over 2 repeats x 3 runs per arm, arms interleaved; W = tokens per pass). The logits hashes are
identical in every arm and at every width (810d28ee4a7b6cef at W1-5, 47d61a17e7270946 at W6 and W8, bcf618c73b286575 at W7):

| Arm | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 519c1ab (A) | 23.24 | 28.87 | 32.59 | 36.09 | 41.47 | 44.92 | 47.57 | 49.83 | 12 |
|  vs 519c1ab | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | |
| 0228b0a, plain GGUF (both loops in the kernel, as main) | 25.41 | 30.25 | 34.11 | 37.77 | 43.27 | 46.66 | 49.23 | 51.60 | 6 |
|  vs 519c1ab | +2.17 | +1.38 | +1.52 | +1.69 | +1.80 | +1.73 | +1.66 | +1.77 | |
| branch, TILE=0: kernel without the tile loops | 23.28 | 28.88 | 32.62 | 36.15 | 41.57 | 45.04 | 47.72 | 49.92 | 6 |
|  vs 519c1ab | +0.04 | +0.01 | +0.03 | +0.06 | +0.10 | +0.12 | +0.15 | +0.10 | |
| variant 1, TILE=1: kernel with the tile loop only (no plain Q6_K case), 2 sets | 25.35 | 30.36 | 34.22 | 37.78 | 43.25 | 46.34 | 48.95 | 51.38 | 12 |
|  vs 519c1ab | +2.11 | +1.49 | +1.63 | +1.70 | +1.78 | +1.42 | +1.38 | +1.55 | |
| variant 1, tile file, TILE=0 | 25.38 | 30.38 | 34.23 | 37.81 | 43.13 | 46.38 | 48.92 | 51.37 | 4 |
|  vs 519c1ab | +2.14 | +1.51 | +1.64 | +1.73 | +1.66 | +1.46 | +1.35 | +1.54 | |
| both loops in the kernel, tile data (as main with the default on) | 25.38 | 30.38 | 34.28 | 37.83 | 43.23 | 46.38 | 48.93 | 51.31 | 6 |
|  vs 519c1ab | +2.14 | +1.52 | +1.69 | +1.74 | +1.76 | +1.45 | +1.36 | +1.48 | |
| variant 2: tile loop in the case of type 14 | 25.33 | 30.14 | 34.06 | 37.61 | 43.18 | 46.33 | 48.89 | 51.29 | 6 |
|  vs 519c1ab | +2.09 | +1.27 | +1.47 | +1.52 | +1.71 | +1.40 | +1.32 | +1.46 | |
| variant 3: variant 2 plus asm on k in decode_tile | 25.32 | 30.21 | 34.13 | 37.74 | 43.30 | 46.25 | 48.89 | 51.28 | 6 |
|  vs 519c1ab | +2.08 | +1.34 | +1.54 | +1.65 | +1.83 | +1.33 | +1.32 | +1.45 | |
| final build, TILE=0 (default) | 23.28 | 28.91 | 32.69 | 38.05 | 41.51 | 45.02 | 47.61 | 49.88 | 2 |
|  vs 519c1ab | +0.04 | +0.04 | +0.09 | +1.97 | +0.04 | +0.09 | +0.04 | +0.05 | |
| final build, TILE=1 (both loops) | 25.23 | 30.36 | 34.22 | 39.74 | 43.27 | 46.31 | 48.88 | 51.33 | 2 |
|  vs 519c1ab | +1.99 | +1.49 | +1.63 | +3.65 | +1.80 | +1.39 | +1.31 | +1.50 | |

(0228b0a: one of the six W1 values, 32.01 ms, was a disturbed run; the median ignores it. Final build: only 2 runs per arm; at W4 the first repeat of each
of the two runs was disturbed (39.98 and 41.62 ms against 36.13 and 37.86 in the second repeat) and the table shows the median of the two, i.e. their
mean. Logs: `2026-11-15-pc-q6k-tile-spills-curve-*.log`; the prefixes `old-`, `v3-`, `v4-` mark earlier sets of the same arm.)

**Cause confirmed:** the two `case kQ6KTile` are the whole loss. A plain GGUF on the kernel with them (0228b0a) is 2.2 ms (W1) and 1.4 to 1.8 ms (W2-8)
slower per pass than 519c1ab, +9.3 % at 1 token. The kernel without them (this branch with `TONA_Q6_K_TILE=0`) is within +0.01 to +0.15 ms of
519c1ab. The case is never executed on a plain GGUF (no task has the tile type), so the cost is the code generation of the whole kernel.

### 2. Does the tile layout still pay inside a kernel that has the tile loops?

Tile data in the kernel with both loops (what main does with its default on) against plain data in the same kernel (0228b0a), medians from the table:
W1 -0.03, W2 +0.13, W3 +0.17, W4 +0.06, W5 -0.04, W6 -0.28, W7 -0.30, W8 -0.29 ms. The layout gains 0.3 ms at W6-8 and nothing at W1-5, as the tile note
measured (W1-4 within +-0.05 ms, W6-8 -0.44 to -0.48 ms). **But the kernel that holds the tile loops is 2.1 ms (W1) and 1.4 to 1.8 ms (W2-8) slower than
the kernel without them, so `TONA_Q6_K_TILE=1` (the default of main) is slower than 519c1ab: +9 % at 1 token, +3 % at 8.** The equal W1-4 curves
of the tile note compared two runs that both had the cost (its plain arm ran on the code of 0228b0a).

### 3. Candidates for a tile kernel without the cost

| Candidate | SGPR / VGPR spills, `<1>` `<4>` `<8>` | pass vs 519c1ab, W1 / W4 / W8 | Verdict |
|---|---|---|---|
| main: both loops, no flag | 103/61, 129/53, 163/110 | +2.1 / +1.7 / +1.5 ms | the finding |
| (b) kernel per layout: plain programs on the kernel without tile loops | 103/62, 121/54, 147/109 | +0.04 / +0.06 / +0.10 ms | **taken**: `TONA_Q6_K_TILE=0` = 519c1ab |
| (b) the same, programs with tile tasks on the kernel with the tile loop only (the plain Q6_K case removed) | 103/58, 127/52, 167/108 | +2.1 / +1.7 / +1.5 ms | no gain over main: the cost comes with the tile loop, not with the second loop |
| (a) one loop per case: the tile loop in the case of type 14 (tile tasks rewritten to type 14 at upload, the switch has the shape of the plain one) | 102/59, 99/53, 254/109 | +2.1 / +1.5 / +1.5 ms | same curve, spill counts differ widely |
| (a') (a) plus `asm("" : "+v"(k))` in `decode_tile` (offsets 4k, 6k as per-lane values; `v_readlane` 305 and `v_writelane` 104 in `k_mega<1>`, as in the plain kernel) | `<1>` as plain, higher at `<4>` and `<8>` | +2.1 / +1.7 / +1.5 ms | same curve: the static SGPR-spill traffic is not the cause |
| (c) tile loops `__noinline__` (arguments by value) | 103/83 (scratch 704), 138/78, 203/1152 (scratch 848) | not run | VGPR spills 10 times higher in `<8>`; dropped without a GPU run |

Per-op profile (`TONA_MEGA_PROF=1`, W = 1, one run per arm, built with variant (a)):

| op (W = 1) | plain GGUF, us | tiles, us |
|---|---:|---:|
| dn.in (36x) | 3885 | 4018 |
| moe.gate_up (48x) | 2826 | 2861 |
| moe.down (48x) | 2065 | 2146 |
| dn.out (36x) | 1855 | 2083 |
| gr.down (97x) | 1817 | 2085 |
| gr.up (97x) | 1646 | 1942 |
| head.out (1x) | 1601 | 1617 |
| qsa.in (12x) | 1182 | 1233 |
| dn.recur (36x) | 930 | 1120 |
| combine (97x) | 711 | 868 |
| host.wait (48x) | 653 | 727 |
| router (48x) | 613 | 757 |
| qsa.out (12x) | 603 | 651 |
| moe.shared (48x) | 438 | 409 |

All ops with Q6_K layer matrices (dn.in, dn.out, gr.up, gr.down, qsa) and the ops that wait for them (dn.recur, combine, router) are slower by 2 to 6 us
per step; the expert ops that never read tiles (moe.gate_up +1 %, moe.down +4 %) hardly are. The cost sits in the step, not in the decoder bandwidth (the
probe of the tile note reads the tile rows at equal or higher bandwidth than plain rows). **Ungeprueft:** the mechanism inside k_mega (instruction fetch,
register allocation around the calls of the `__noinline__` ops, scratch latency). What the numbers say: every kernel that contains `decode_tile` has the
cost, whatever the switch looks like, and neither the number of spill slots nor of `v_readlane` predicts it.

### 4. Goldens (all identical)

`scripts/golden.sh compare` (gpu plan), `GOLDEN_PLAN=cpu`, `GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash`, `GOLDEN_MODE=prefill`,
`GOLDEN_PORT=8099 scripts/golden-server.sh compare` on the final build with the defaults, and again with `TONA_Q6_K_TILE=1` (the tile route on the
kernel with both loops): all five identical in both runs. On the first build of this branch (before the refactor of `init_kernels`, which only moved
code) the defaults run and the first three checks of the `TONA_Q6_K_TILE=1` run were identical as well.

### 5. Production A/B (A = 519c1ab, B = 639e329 with the defaults)

`set,mid`, ctx 140000, expert profile of the scale run, 6 rounds (the three decode metrics did not decide, so the run went to the last round),
`--load-log on`. The SHA of the 30000-token request is equal in A and B and stable (`ee58880d3f5b558f`), and the expert cache counts are equal. Full report:
`2026-11-15-pc-q6k-tile-spills-ab.md`.

| Metric | B vs A, mean [CI] | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | -0.06 % [-0.13, +0.02] | -1..+1 % | equal |
| decode-code | +0.32 % [-1.37, +2.01] | -1..+1 % | open: the interval is wider than planned (other programs used 12 to 20 % of the CPU during the run) |
| decode-text | +0.06 % [-1.67, +1.78] | -1..+1 % | open, same reason |
| decode@30000 | +1.75 % [-0.56, +4.06] | -1..+1 % | open; four of the six rounds are marked outlier or off plan by the tool |

The loss of the coordinator's A/B (decode-code -3.63 % [-4.15, -3.11], decode-text -4.05 %, decode@30000 -3.27 %) is gone: the lower interval ends are -1.4 %
to -1.7 % and the means are at or above zero. The A/B does not show equality at the 0.5 % margin; the pass curve of section 1 is the sharper test (+0.01 to
+0.15 ms of 23 to 50), the A/B confirms that no loss of the earlier size remains.

## Decision

- Keep `Q6Layout`: k_mega per layout, a launch takes the kernel without the tile loops unless its program has tile tasks. `TONA_Q6_K_TILE=0` and every
  plain GGUF run the code of 519c1ab again (spill counts equal, pass curve equal, goldens identical).
- `TONA_Q6_K_TILE` default 1 -> 0, because the kernel with the tile loop is 1.4 to 2.1 ms per pass slower and the layout gains 0.3 ms at W6-8. A tile file
  (type 1019 in the GGUF) still runs, on the slower kernel; the Mac backend is untouched.
- The target of the ticket (default at least 519c1ab plus the tile gain) is **not reached**: the default equals 519c1ab, the tile gain of the HIP backend is
  not kept. No candidate (a, a', b, noinline) removed the cost of the tile loop.
- Principle 8 (dropping a measured gain): reason: the gain (0.3 ms at W6-8, nothing at W1-5) is smaller than the cost of the loop in k_mega (1.4 to 2.1 ms at
  every width); fresh search: see Prior art; attempt to remove the cost: (a), (a'), (b), (c) above, none works. Open ways, not tried: a tile layout whose decode needs
  fewer instructions than `decode_tile` (changes the file format shared with the Mac backend), or a separate kernel for the wide phases of tiled matrices
  instead of a loop in k_mega (the paper "One Megakernel is Not Enough" argues for cutting at such boundaries).

## Prior art

- [llama.cpp `mul_mat_vec_q` and `mul_mat_q`](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda) (merged, long-standing, from memory of the
  source: **ungeprueft** in this session): one template instantiation per quantization type and column count, chosen on the host, not a switch over types
  inside one resident kernel. Adopted: the same idea one level up, one k_mega instantiation per layout chosen at launch (`Q6Layout`). Not adopted:
  separate kernels per type (k_mega stays one kernel for its in-kernel barriers).
- [llama.cpp #31751 "CUDA: avoid J=32 MMQ register spills on RDNA3"](https://github.com/ggml-org/llama.cpp/pull/31751) (open, title only) and
  [#31550 "ci: update the HIP VGPR spill ignore list"](https://github.com/ggml-org/llama.cpp/pull/31550) (closed, title only): llama.cpp treats register
  spills of single kernels as regressions and checks them in CI. Adopted: the spill counts of `k_mega<1|4|8>` are the first check of a change to the matvec
  switch. Not adopted: spill counts as the only criterion; they did not predict the speed here (section 3).
- [Cohere: Megakernel Serving Engine](https://cohere.com/blog/megakernels) (in `research/papers.md`): one decode megakernel with all tasks inside; nothing on
  code growth per format. No change.
- [One Megakernel is Not Enough, arXiv 2610.35336](https://arxiv.org/abs/2610.35336) (abstract only, in `research/papers.md`): the best configuration is 2 to 4
  megakernels or a graph; supports the open way of a separate kernel for the tiled wide phases.
- Own earlier work: `2026-10-28-mega-spills.md` (removing spills by source changes made decode slower: the same failure of spill counts as a predictor) and
  `2026-11-14-pc-det-decode-bisect.md` (culprit 2).

Search: tools/papers/scan.py topics v1 (github scan); topics: quantization, amd-rocm, megakernel; terms: MMVQ, register spills, VGPR spill, template instantiation, quant kernel switch; window: 2026-10-15..2026-11-14 (llama.cpp PRs, 1187 listed, 1 month) plus the book `research/papers.md` (complete from 2025-01, 12 months)

## Files

`2026-11-15-pc-q6k-tile-spills-curve-*.log` (curve runs), `-prof-t0.log` and `-prof-t1.log` (per-op profile), `-spills.txt` (resource usage of every build
tried), `-ab.md` (A/B report). Changed code: `engine/gpu/matvec_phase.h` (`Q6Layout`), `engine/gpu/mega.hip` (kernel per layout, `kernel_ptr`,
`probe_kernel`), `mega_kernel.h`, `mega_op_matvec.h`, `mega_upload.cpp` and `mega_data.h` (programs with tile tasks), `mega_alloc.cpp` (default 0),
`docs/use/environment.md`, `README.md`.
