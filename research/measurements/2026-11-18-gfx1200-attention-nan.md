# gfx1200 attention: NaN and run-to-run variation (llama.cpp #32178) and VMM growth (llama.cpp #32350), 2026-11-18 (#171)

## What the two reports say

**llama.cpp #32178** (`test-backend-ops -o FLASH_ATTN_EXT`, backend ROCm0, RX 9060 XT gfx1200, 16,304 MiB, wave 32, HIP/ROCm 7.1.52801, clang 21.1.8, llama.cpp commit 956591f, Release, `GGML_HIP_GRAPHS=OFF`, `GGML_HIP_NO_VMM=ON`):

- Failing case: `hsk=192, hsv=128, nh=4, nr23=[16,1], kv=512, nb=76 (75 also fails), mask=1, sinks=1, max_bias=0`. Tensor type not stated.
- Symptoms: identical inputs (same CPU and input hashes) give different GPU output hashes between runs; NaN on the GPU where the CPU value is finite ("NaN at index 36953 (ROCm0=-nan CPU=0.001484)"). Forcing the TILE kernel (no matrix-core MMA path) passed 3 of 3 runs at 75 tokens. One head-256 sparse-mask case still failed after the narrow fix.
- Root cause unknown. Candidate workaround: `__forceinline__` on both `ggml_cuda_unroll::operator()` in `common.cuh` took the fixed-input suite from 4,316/4,322 to 4,322/4,322 (mechanism not established; no PR). A narrower inline (V loading only) gave 4,321/4,322.
- Reading: a compiler or code-generation effect in the MMA flash-attention path of the llama.cpp CUDA-to-HIP port, in a shape (K head 192 against V head 128, sinks, 16 query heads per KV head, ragged token count) that is the llama.cpp kernel's own.

**llama.cpp #32350** (ROCm 7.1.1, master `3549c274d`, `GGML_HIP_NO_VMM=OFF`, `test-backend-ops -o ADD`, no model):

- The first mapping is 4,096 bytes at a granule of 4,096 (the minimum granule); 9 cases pass. At the 10th the pool grows: `hipMemCreate` and `hipMemMap` of 45,056 bytes at offset 4,096 succeed, then `hipMemSetAccess(pool + 4096, 45056)` returns `hipErrorInvalidValue` (abort in `ggml-cuda.cu:707`).
- Setting access over the whole mapped range (offset 0, 49,152) returns success and `hipMemGetAccess` reports read/write, but the next kernel faults (`HSA_STATUS_ERROR_MEMORY_FAULT`). A control with one 64 MiB first mapping passed 173/173.
- Root cause not established, no verified fix. The reporter asks whether unmap-and-remap on growth would be acceptable.

## Which of our code can be hit

| Report | Our code | Match |
|---|---|---|
| #32178 shapes | The Coder uses QSA attention (`mega_op_attn.h`, `mega_op_dattn.h`, `prefill_attn.hip`, head size 256, 24 Q / 2 KV heads). Ornith and Bolt use full-context attention (`fattn_tile.h`, `mega_op_fattn.h`, `prefill_fattn.hip`, head size 256 / 128, groups 8 / 16). All are our own WMMA tiles (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`) with a hand-written softmax. There is no K 192 / V 128 shape, no sinks, no `ggml_cuda_unroll`. | Shape not present. Code generation on the same ISA is the common ground: the same compiler family produces our WMMA tiles. |
| #32350 | `ElasticPool` (`mega_elastic.cpp`): one reservation, the core from chunks of 32 MiB (2 MiB granule, `hipMemAllocationGranularityRecommended`), later blocks mapped behind the cursor, each with `hipMemSetAccess` on its own new range (nonzero offset). Never an unmap and remap at an address (docs/architecture/vram-giveback.md, #156). | Same call pattern (new sub-range, nonzero offset) at a different granule (2 MiB, not 4 KiB). |

Reading of the code (no run needed): in the full-attention tile the softmax cannot produce NaN from a fully masked chunk (`e = v > -inf ? exp(v - mx) : 0`, rescale factor 0 when the running maximum is still -inf), padded rows have a zero query and a finite score, and the per-row sum `lrow` is at least the weight of the row's own diagonal cell (prefill) or of any cached cell (decode). LDS reuse (`Ks` as `pt`) is separated by `lds_sync()` before the next staging. So no NaN or race path is visible in review; the check below is empirical.

## Plan (written before the runs)

Question: does our attention give different bits for identical inputs, or a NaN, in the conditions of #32178 (ragged token counts 75 and 76, 512 cached positions, 16 query heads per KV head, repeated runs), and does a pool that grows by mapping behind the mapped range work on this stack at the minimum granule (#32350)?

Why this size: the report's effect appears at the first repeats (3 of 3 runs differ at 75 tokens), so a repeat count of 200 per case is far above what reproduced it; each pass takes about a millisecond, so one case is about a second. The VMM micro needs 64 MB and ten growth steps. Smallest workload that answers it; deterministic counters (bit comparison, NaN and infinity counts) instead of timing. No speed claim is made, so no ab.py and no BENCHMARKS.md speed row; this is a correctness record.

Runs (all `gpu` kind through the queue, desktop guard on):

1. `test_gpu_fattn` (existing: oracle comparison, same bits across grid sizes and token batching) once.
2. `test_gpu_fattn --stress 200` (new): shapes Bolt hd 128 nh 32 / nkv 2 and nh 64 / nkv 4 (group 16), Ornith hd 256 nh 16 / nkv 2 and nh 64 / nkv 4 (group 8 and 16); decode passes of 1 and 4 tokens ending at 512 positions and at 1,004; prefill of 75 and 76 tokens ending at 512 and 76 tokens from 0. Each case 200 times; every output compared bit for bit with the first run and scanned for NaN and infinity (decode: q, o and the int8 output; prefill: f16 output).
3. `test_gpu_attn_wmma` once (Coder QSA tile against the oracle).
4. `just cross-check` (CPU oracle against the GPU path on the real model, 4 tokens, 6 steps) three times with `TONA_DETERMINISTIC=1`; the three outputs must be identical.
5. `vmm_grow` (new micro): minimum and recommended granule, access set on the new sub-range and on the whole range (4 runs).

Expectation: run 1 all ok; run 2 zero NaN, zero inf, zero differing runs in every case (our kernels are deterministic by design: same bits across grid sizes is already tested); run 3 ok; run 4 identical outputs; run 5 recommended-granule runs all steps ok (this is what production does), minimum-granule runs either ok or reproduce the #32350 error. Tolerance: none for counters (any NaN, inf or differing run is a finding). Any result other than zero is a finding, not noise.

Active tooling: none (no diagnostics, no profiler); `TONA_DETERMINISTIC=1` for run 4 only.

## Test conditions

Logs: `2026-11-18-gfx1200-attention-nan-logs/` (`hw.txt` is `tools/hw_report.py`, `quiet-before.txt` and `quiet-after.txt` are `tools/quiet_check.py`). Build: `/var/tmp/claude/w171-build` from this branch (commit ad6c5da, ROCm 10.1 toolchain of the main checkout, hip 7.16 at run time; `hw_report` ran outside the toolchain environment and so shows the system hip 7.1.52802), jobs through the gpu queue (`w171-attn`, `w171-cross`, `w171-vmm`), desktop guard on.

- CPU: AMD Ryzen 9 9950X, 16 cores / 32 threads; GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB; RAM 60.5 GiB; Fedora Linux 44, kernel 7.3.4-200.
- `quiet_check` before: "do not measure: 14.89 GiB of 15.9 GiB VRAM are held by others" (another worker's queue job held the card); after: the same plus 18 % CPU use by others and load 23. **This is a correctness record, not a speed measurement**: counters and bit comparisons do not depend on a quiet machine, so the runs went ahead; no speed number is reported.
- `TONA_*`: none for runs 1 to 3 and 5; `TONA_DETERMINISTIC=1` for run 4.

## Results

| # | Run | Result |
|---|---|---|
| 1 | `test_gpu_fattn` (oracle, same bits across grid sizes and token batching; Ornith hd 256, Bolt hd 128) | **all ok**, rc 0, tolerances as in `2026-11-15-multi-arch-f1-fattn.md` |
| 2 | `test_gpu_fattn --stress 200` (4 shapes x 6 cases, 200 runs each = 4,800 passes) | **0 NaN, 0 inf, 0 differing runs in every case**, rc 0. Shapes: hd 128 nh 32 / nkv 2 and nh 64 / nkv 4 (groups 16), hd 256 nh 16 / nkv 2 (group 8) and nh 64 / nkv 4 (group 16); decode 1 and 4 tokens ending at 512 positions, 4 tokens at 1,000; prefill 75 and 76 tokens ending at 512 positions, 76 tokens from 0 |
| 3 | `test_gpu_attn_wmma` (Coder QSA tile, repeat, ragged, drift, random cases up to 262,000 positions) | **all identical**, rc 0 |
| 4 | `just cross-check` equivalent (`tona-gpu-gen ... -n 4 --check 6 --mega`, Coder, CPU oracle against the GPU path) x 3, `TONA_DETERMINISTIC=1` | top-1 equal in 6 of 6 in all three, worst relative deviation 7.46e-02 in all three, **the six per-step lines and the summary are byte-identical across the three runs**, no NaN in the output. The deviation size matches earlier runs of this check (8.6e-2 on 2026-10-23, 7.8e-2 before it; `2026-10-23-cpu-experts.md`, `2026-11-12-shard-1019-switch.md`) |
| 5 | `vmm_grow`, minimum granule (4,096 B), the sequence of #32350 (1 granule, then 11, 3, 5, ...) | **all 10 steps ok** with access set on the new sub-range at a nonzero offset (the call that fails in #32350: 45,056 bytes at offset 4,096) and with access set on the whole range; the kernel over the whole mapped range wrote and verified its pattern after every step |
| 5 | `vmm_grow`, recommended granule (2 MiB, what Tona uses) | steps 1 to 7 ok (the first 62.9 MB; sub-range and whole range); the program then stopped because its 64 MB reservation was exhausted at step 8, which it printed as "FAILED": a limit of the micro, not an API error (no error string was printed). The reservation was raised to 128 MiB afterwards. Rerun 1 (`w171-vmm`, 2026-11-18 20:26) aborted in all four arms at start with "LLVM ERROR: out of memory" (host allocation, rc 134), during the agent eval's SWE grading; not investigated. Rerun 2 (`w171-vmm-r2`, 2026-11-19, queue otherwise empty): **all steps ok in all four arms** (4 KiB and 2 MiB granule, sub-range and whole-range access), rc 0. The logs here are the first run |

## Reading against the plan

- Expectation met on every counter: no NaN, no infinity, no differing run, oracle comparisons within tolerance. Nothing outside the expectation.
- #32178 does not reproduce in our attention. Our tile differs from the llama.cpp MMA kernel in the ways the report's candidate causes are about (own softmax, no `ggml_cuda_unroll`, no K 192 / V 128 shape, no sinks), so a null result here says our kernels are stable in the conditions of the report that they can run (ragged 75 / 76 tokens, 512 cached positions, groups of 16), not that the llama.cpp compiler problem is absent from the toolchain. The same compiler family builds both.
- #32350 does not reproduce at either granule on this stack (hip 7.16 of the ROCm 10.1 toolchain; the report's ROCm is 7.1.1). Tona's use is the same call pattern (sub-range access at a nonzero offset) at 2 MiB and works in production (chunked core, 262 chunks, measured 2026-11-18 in `2026-11-18-vram-chunks-gtt.md`). The report's suggested remedy, unmap and remap on growth, would break the never-remap rule (#156, ROCm/TheRock #8744) and is not for us.

## Fix proposal

None needed for our kernels. Kept as regression tests: `test_gpu_fattn --stress N` (bit comparison and NaN scan; add to `just test` only if wanted, it is a GPU test) and `bench/micro/vmm_grow.hip`. If a NaN is ever seen in our attention, the first suspects from #32178 are the compiler's handling of the WMMA loop and unrolled loads; the `lds_sync` barriers and the `-inf` masking in `softmax` were reviewed and look sound (see "Which of our code can be hit").
