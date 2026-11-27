# TONA_Q6_K_T from the original GGUF: tiling the Q6_K layer matrices while they are uploaded to VRAM (#130)

RX 9060 XT (gfx1200), Zen 5. Branch `130-load-tile` from main 0228b0a. Follows `2026-11-13-pc-q6k-tile.md` (tile file: pass 0.44-0.48 ms
shorter at W6-8, production decode code +0.90 % [+0.04, +1.76], all goldens identical), which needs a second model file for PC users.

## Plan (written before the first run)

**Question.** Does tiling the 128 `blk.*` Q6_K matrices (1.36 GB) in host memory on the way to VRAM give the same pass gain from the original file,
and what does it cost at load?

**Where the weights go** (read in code, not guessed): every dense weight is uploaded by `Mega::upload` (`engine/gpu/mega_alloc.cpp`, one
`hipMemcpy` from the file mapping); the engine's view of the type is `TensorInfo::type`, read for dense matrices in `ProgramBuilder::addw` and `pw` only.
The CPU never reads these matrices (it reads experts, embedding and PLE rows; the file mappings of the dense tensors are released after the upload). So the
type is switched per device pointer (`Mega::weight_type`), the `TensorInfo` and every CPU reader stay on type 14. `output.weight` (read by row), the
MTP head and the drafter are not touched. Same name rule as `tools/coder_requant/repack_tile.py`, so the bytes equal a tile file.

**Switch.** `TONA_Q6_K_TILE` (integer, `docs/use/environment.md`), default 0 until this measurement decides. Tiler `tile_q6_k` (`model/quant_tile.cpp`), OpenMP over
groups of 16 tiles in `upload`; unit test against the independent packer of `test_quant_tile.cpp`. The 16-byte rule of the uploader holds (device allocations
are 256-byte aligned, a tile is a multiple of 16 bytes).

**Scope.**
1. Load time: `tona-gpu-gen --mega --batch-bench`, "loaded in" line. Arms off / on, interleaved (off, on, off, on, ...), 3 runs each cold (the model files evicted with
   `posix_fadvise(DONTNEED)` first; the 55 GB model is larger than the page cache, so "warm" below is whatever stays) and 3 runs each warm (right after, short bench).
2. Pass curve W = 1..8: the cold runs above run `--batch-bench 600` with `TONA_BENCH_HASH=1`, as the tile-file runs (6 per width and arm over 3 runs x 2 rounds in the
   earlier note; here 3 runs x 2 rounds as well). Compared to the tile-file numbers and hashes (3a2febe7cd168de0 W1-5, 22e6734613734eb9 W6/W8, fcea32541e8bc009 W7).
3. Production decode: `scripts/ab.py` workload `set`, A = switch off, B = on, same build, 4+ rounds AB/BA, alpha 0.05, margin 0.5 %.
4. Bit-exactness with the original file and the switch on: `scripts/golden.sh compare` (decode), `GOLDEN_PLAN=cpu`, `GOLDEN_MODE=prefill`, prefill with
   `GOLDEN_DRAFTER=dflash`, `scripts/golden-server.sh compare`.

**Expectation (with tolerance).** Pass gain as with the tile file: W1-4 within +-0.05 ms, W5-8 -0.1 to -0.5 ms (source: the earlier note, section 3); production decode code
+0.9 % (CI about +-0.9 %), text +0.6 %. Load time: the tiler reads 1.36 GB from the page cache and writes 1.36 GB; at 10 to 20 GB/s per direction that is 0.1 to 0.3 s
on 12 threads, hidden partly by the memcpy it replaces (+0.0 to +0.5 s expected; a cost over 1 s would be a finding, then threads or overlap). Cold load adds nothing
beyond the same file reads.

**Active tooling.** `TONA_BENCH_HASH=1` in the bench runs; no profiler, no other TONA_* variable. Test conditions below from `tools/hw_report.py`; the machine is shared
through `scripts/gpu_queue.py`.

## Test conditions


- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.9 GiB held by others before the run, 1 % busy before the run
- RAM: 60.5 GiB, 55.5 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.02, 4.18, 2.87
- CPU use by other programs over 0.5 s: 0.5 % of all threads, swap in use 2.26 GiB
- sandbox: no
- Commit: `0228b0a` (dirty tree)
Quiet check before (2026-11-13 07:40): warning, 1.9 GiB of 15.9 GiB VRAM are held by others (idle desktop).


## Results

Runs of 2026-11-13 (load times, pass curves; logs `2026-11-13-pc-q6k-tile-load-*.log`) and of 2026-11-14 after a reboot (A/B, tile-file
hash run, goldens; build `work/build/tona130b` at e18375d = this branch with main e76184f merged, switch default still 0 in these runs).
The 11-13 series was stopped once by the coordinator (machine load) and resumed on 11-14 from the A/B; the plan was not changed.

### 1. Load time (`loaded in`, tona-gpu-gen)

| Arm | cold (3 runs) | warm (3 runs) | warm, second series (3 runs) |
|---|---|---|---|
| off (0) | 8.8 / 8.8 / 8.8 s | 10.4 / 9.1 / 10.4 s | 10.3 / 10.4 / 10.4 s |
| on (1) | 10.6 / 10.7 / 10.5 s | 12.0 / 11.5 / 12.0 s | 11.0 / 10.9 / 10.9 s |

**Finding, outside the expectation (+0.0 to +0.5 s): +1.8 s cold, +0.5 to +1.6 s warm.** The tiler runs at about 0.75 GB/s for 1.36 GB, far
below the 10 to 20 GB/s assumed in the plan. Not yet profiled; candidates are the zero-fill of the staging vector per tensor and the
`hipMemcpy` from pageable memory after the tiling (the plan's "threads or overlap" follow-up). The server's "to ready" in the A/B below shows
the same: 18.0 s off, 20.0 s on.

### 2. Pass curve W = 1..8 (`--batch-bench 600`, 3 runs x 2 rounds, ms per pass, mean and range)

| W | off | on | on - off | tile file (11-14, 1 run x 2) |
|---|---|---|---|---|
| 1 | 25.39 (25.31-25.46) | 25.29 (25.21-25.38) | -0.10 | 25.25 |
| 2 | 30.25 (30.21-30.29) | 30.27 (30.24-30.30) | +0.02 | 30.26 |
| 3 | 34.13 (34.08-34.16) | 34.15 (34.11-34.21) | +0.02 | 34.20 |
| 4 | 37.74 (37.72-37.78) | 37.76 (37.73-37.78) | +0.02 | 37.80 |
| 5 | 43.28 (43.21-43.33) | 43.18 (43.12-43.22) | -0.10 | 43.34 |
| 6 | 46.67 (46.57-46.73) | 46.17 (46.11-46.25) | **-0.50** | 46.27 |
| 7 | 49.21 (49.15-49.29) | 48.76 (48.71-48.85) | **-0.45** | 48.85 |
| 8 | 51.53 (51.49-51.63) | 51.10 (51.03-51.15) | **-0.43** | 51.28 |

Within the expectation (W1-4 +-0.05 ms except W1 -0.10, W5-8 -0.1 to -0.5 ms), the same as the tile file of the earlier note (-0.44 to -0.48 ms).

**Logits hashes**: identical for off, on and the tile file at every width: 810d28ee4a7b6cef (W1-5), 47d61a17e7270946 (W6, W8),
bcf618c73b286575 (W7). They differ from the hashes of the earlier note (3a2febe7cd168de0 ...), which came from a different run setup;
which setting changes them is unchecked (candidate: the expert placement, `TONA_EXPERT_PROFILE` was not set here). Within one setup the three
routes are bit-identical.

### 3. Production decode (`scripts/ab.py` set, A = 0, B = 1, same build, 6 rounds, `--load-log on`)

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +0.44 % | -0.15 % | 144 |
| 2 | BA | +0.44 % | +0.00 % | 146 |
| 3 | AB | +0.33 % | +0.15 % | 144 |
| 4 | BA | +0.44 % | +0.15 % | 146 |
| 5 | AB | +0.33 % | +0.15 % | 144 |
| 6 | BA | +0.33 % | +0.30 % | 146 |

Within the expectation (+0.9 % code, CI about +-0.9 %): **decode code +0.38 % [-0.11, +0.88] (91.6 -> 91.9 tok/s), open after 6 rounds**;
text equal. Load columns: CPU ours 162 to 168 % of one core, system 8.4 to 10.3 %, other 1.7 to 3.1 %; "GPU ours" reads 0 because ROCm
compute is not in DRM fdinfo (2026-11-14-pc-load-log-aa.md); no swap traffic, memory level 50 %.

### 4. Goldens with `TONA_Q6_K_TILE=1`

`scripts/golden.sh compare` (csv, rust, explain) identical, rc 0; `scripts/golden-server.sh compare` all cases identical (base, sampled,
thinking, reuse, ssd, hit), rc 0. Not run in this series: `GOLDEN_PLAN=cpu` and the two prefill goldens of the plan (they run in the scale run
of 2026-11-14 with the default build; with the switch on they follow at the merge).

### Decision

Default **on** (`TONA_Q6_K_TILE=1`): bit-identical, the pass gain at W6-8 is deterministic and equals the tile file's, production decode
code moves by +0.38 % (open, consistent with +0.90 % [+0.04, +1.76] of the tile file). The cost is 1.8 s per cold start. Follow-up: make
the tiler fast enough to hide in the upload (target below +0.5 s).

### Test conditions of the 2026-11-14 runs

Before (18:59, end of the preceding A/A, same boot; the A/B started 19:01):

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 9.93 GiB held by others before the run, 69 % busy before the run
- RAM: 60.5 GiB, 57.0 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- TONA_* variables: `TONA_GPU_ARCH=gfx1200` `TONA_GPU_LOCK_HELD=1` `TONA_HIPCXX=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/amdclang++` `TONA_HIPFLAGS=-x hip --offload-arch=gfx1200 -isystem <repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/include` `TONA_HIPLDFLAGS=-L<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib`
- cgroup limits of this shell: MemoryMax=54.0G, MemorySwapMax=0.0G
- Load average (1, 5, 15 min): 1.46, 1.39, 0.96
- CPU use by other programs over 0.5 s: 0.2 % of all threads, swap in use 0.0 GiB
- sandbox: no
- Commit: `e76184f` (clean tree)
- Quiet check (2026-11-14 18:59:38): warning
  - warning: 1.38 GiB of 15.9 GiB VRAM are held by others

After (19:24, before the next job; the load average of 9.8 is the build of that job, after these runs):

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.36 GiB held by others before the run, 2 % busy before the run
- RAM: 60.5 GiB, 56.5 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- TONA_* variables: `TONA_DIAG=1` `TONA_GPU_ARCH=gfx1200` `TONA_HIPCXX=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/amdclang++` `TONA_HIPFLAGS=-x hip --offload-arch=gfx1200 -isystem <repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/include` `TONA_HIPLDFLAGS=-L<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib`
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 9.8, 5.27, 2.95
- CPU use by other programs over 0.5 s: 0.5 % of all threads, swap in use 0.0 GiB
- sandbox: no
- Commit: `7d7108a` (clean tree)
- Quiet check (2026-11-14 19:24:41): warning
  - warning: load average 9.8 is high (it lags a minute)
  - warning: 1.36 GiB of 15.9 GiB VRAM are held by others
