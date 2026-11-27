# F3: routing policy sigmoid with bias next to softmax, one definition for CPU and GPU, 2026-11-15

## Plan (written before the runs)

- Scope: unit tests (`test_route_policy` on the CPU, `test_gpu_route_sig` on the GPU: 32 sigmoid cases of the Python
  reference `tools/gen_route_ref.py`, bit-equal to the CPU); C2 program dumps of the three standard runs against
  `work/base-dumps`; C3 `golden.sh compare` for the gpu plan and `GOLDEN_PLAN=cpu`; C4 pass curve of
  `tona-gpu-gen --mega --batch-bench 600` (TONA_BENCH_HASH=1, widths 1, 4, 8, 2 repeats), 3 interleaved runs per arm,
  `work/build/base` against this branch. About 12 minutes of GPU time.
- Expectation: unit tests ok; dumps byte-equal; goldens identical; pass time per width within +-0.15 ms of the base
  (the softmax path only gained a dead float sum), logits hashes equal.
- Active tooling: TONA_BENCH_HASH=1 only.

## Prior art

- llama.cpp `build_moe_ffn` (sigmoid gating, selection by score + bias, unbiased weights normalised with the sum clamped
  at 2^-14, then scaled; Nemotron-H and DeepSeek-V3 style), open PR #31390 (Metal fusion of exactly this graph);
  DeepSeek-V3 (arXiv 2412.19437), auxiliary-loss-free balancing (arXiv 2408.15664), Nemotron-H (arXiv 2504.03624)
  (cited, the rule was taken from the llama.cpp code, not from the papers).
- Search: see the F1 file (same scan).

## Test conditions

### Test conditions

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.22 GiB held by others before the run, 0 % busy before the run
- RAM: 60.5 GiB, 54.4 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: `TONA_GPU_LOCK_HELD=1`
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.2, 5.83, 5.74
- CPU use by other programs over 0.5 s: 1.3 % of all threads, swap in use 2.11 GiB
- sandbox: no
- Commit: `302859e` (clean tree)

quiet_check before:
{
 "state": "warning",
 "findings": [
  {
   "level": "warning",
   "what": "1.22 GiB of 15.9 GiB VRAM are held by others"
  }
 ],
 "signals": {
  "taken": "2026-11-15 16:22:50",
  "threads": 12,
  "cpu_busy_percent": 0.7,
  "loadavg1": 2.2,
  "gpus": [
   {
    "busy": 0,
    "vram_gib": 15.9,
    "used_gib": 1.22
   }
  ],
  "ram_total_gib": 60.5,
  "ram_available_gib": 54.4,
  "swap_pages_moved": 0,
  "on_battery": null
 },
 "compare": null
}

quiet_check after:
- Quiet check (2026-11-15 16:34:19): warning
  - warning: load average 5.29 is high (it lags a minute)
  - warning: 1.22 GiB of 15.9 GiB VRAM are held by others
  - warning: the system is swapping (2 pages in 1 s)
- Quiet check before and after the run: warning -> warning, DISTURBED, repeat or mark the run
  - new: load average 5.29 is high (it lags a minute)
  - new: the system is swapping (2 pages in 1 s)
3

The after check says DISTURBED (load average 5.29, 2 pages swapped in 1 s); the load average lags a minute and the six
curve runs ended minutes before it, so the load is mostly our own. The two arms ran interleaved, so a disturbance would
hit both; the spread inside an arm is at most 0.3 ms.

## Results

Unit tests: `test_route_policy: ok`, `test_gpu_route_sig: ok (32 sigmoid cases)`.
C1: all Coder kernels identical to `work/build/base` with the two R5 `--rename` rules (217 in A, 220 in B; only
`k_route_sig`, `op_route_sigmoid` and the test step kernel added).
C2: `server-mtp`, `server-dflash`, `cpu-seqs2` byte-equal to `work/base-dumps`.
C3: golden gpu plan and cpu plan: csv, rust, explain identical.
C4 (decode-shaped verify passes, ms per pass, 3 runs per arm, the six values per cell, mean):

| width | base | this branch | difference |
|---|---|---|---|
| 1 | 23.63 to 23.78, mean 23.70 | 23.60 to 23.77, mean 23.69 | -0.02 ms |
| 4 | 37.86 to 37.96, mean 37.91 | 37.87 to 38.02, mean 37.93 | +0.02 ms |
| 8 | 57.56 to 57.87, mean 57.70 | 57.53 to 57.97, mean 57.71 | +0.01 ms |

Logits hashes equal at all three widths in all runs (width 1 and 4: 2e8e1b22c3a12255, width 8: 07dc3ad00fd072f2). Each
cell holds two repeats per run, so the six values are three runs of two.

## Reading against the plan

All as expected: no change for the Coder, differences far inside the +-0.15 ms tolerance.
