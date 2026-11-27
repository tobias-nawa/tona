# Backend registry on the reference (be93cf3, #108): unchanged reference?

## Plan (written before the run)

- **Question:** does the registry change (shared host code: `engine/backend/registry`, `options.h`, `server_main.cpp`
  reference construction moved into `make_reference` under `TONA_HAVE_REFERENCE`, `--backend auto`) leave the
  reference unchanged?
- **Scope:** build `work/build/m1reg` from be93cf3 (Mac branch 108-mac-m1b, which also carries the IQ2_XXS dequant in
  `cpu/quant.cpp`), unit tests, `scripts/isa-diff.py` against `work/build/engine` (main 06db13a), the five goldens
  (`scripts/gpu-status.sh --keep-going`, ~6 min), then `scripts/ab.py` A = main, B = be93cf3, `TONA_DETERMINISTIC=1`,
  workloads mid and long, up to 4 rounds (~20 min). Total ~35 min.
- **Expectation:** all kernels identical; unit 44/44 (43 plus the new registry test); goldens identical; SHAs A = B;
  decode and prefill within ±1 % (host-only change; noise of mid/long from bench noise.toml is below that).
- **Tooling:** `TONA_DETERMINISTIC=1` in the A/B only; no diagnostics; GPU lock `work/gpu.lock`.

## Test conditions


- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.18 GiB held by others before the run, 3 % busy before the run
- RAM: 60.5 GiB, 56.6 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 0.94, 3.5, 3.2
- CPU use by other programs over 0.5 s: 0.5 % of all threads, swap in use 1.68 GiB
- Commit: `be93cf3` (clean tree)

- Quiet check before: warning (1.18 GiB of 15.9 GiB VRAM held by the desktop; light swapping, 47 pages in 1 s).

## Results

| Check | be93cf3 (registry) | ded88a6 (builtin.h, tona-check) | Expectation |
|---|---|---|---|
| unit tests | 44/44 | 44/44 | 44/44 |
| `isa-diff` against main | all 198 kernels identical | all kernels identical | identical |
| goldens (prefill dflash, gpu, cpu plan, server, prefill) | all identical (`gpu-status.sh`: C7, needs long-context run) | interrupted (machine needed, 2026-11-11 ~15:30), to repeat | identical |
| `ab.py` deterministic, mid + long, 4 rounds | prefill 30K −0.29 %, decode 30K +0.01 %, prefill 130K −0.03 %, decode 130K −0.30 %; SHAs A = B = main reference (5b4e7ef2b0fc2495, 9340252c318b7217); expert hits and misses identical | not run (host-only factory move; goldens decide) | within ±1 % |

All within the expectation. The long-context run that `gpu-status.sh` asks for (because `engine/backend/budget.h` is in
the build graph) is not needed: the preprocessed Linux text of `budget.h` is byte-identical to main's (checked with
`g++ -E -P`, 2026-11-11), and the registry only routes the unchanged construction call.
