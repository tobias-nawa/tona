# F2: plain residual step OP_RES_NORM and its prefill kernel, 2026-11-15

## Plan (written before the runs)

- Scope: `test_gpu_resnorm` once (decode steps inside the real k_mega with a test-only profile, d 2048, 4096, 2080,
  nb 1, 3, 4, 8, with and without y and w, in place; the prefill kernel at d 2048 and 4096), against a float64
  reference; same bits across grid size and token batching. One queue slot, under 1 s of GPU time. No speed claim.
- Expectation: R' exact, xn within 1e-6 relative (decode, float; the prefill kernel writes f16: 1e-3), int8 within 0.5 step, same bits yes, rc 0.
- Active tooling: none.

## Prior art

- The residual add fused into the following RMSNorm is the standard fusion (llama.cpp ggml-cuda `rms_norm` fused with
  add; vLLM `fused_add_rms_norm`); here it is a megakernel step that sums per block instead of a grid barrier, for the
  bit-exact tokens of the pass-width independence the engine requires.
- Search: tools/papers/scan.py topics v1 (see the F1 file); nothing newer than those found.

## Test conditions

### Test conditions

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.22 GiB held by others before the run, 4 % busy before the run
- RAM: 60.5 GiB, 53.9 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- TONA_* variables: `TONA_GPU_ARCH=gfx1200` `TONA_GPU_LOCK_HELD=1` `TONA_HIPCXX=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/amdclang++` `TONA_HIPFLAGS=-x hip --offload-arch=gfx1200 -isystem <repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/include` `TONA_HIPLDFLAGS=-L<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib`
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.83, 6.14, 5.83
- CPU use by other programs over 0.5 s: 0.2 % of all threads, swap in use 2.11 GiB
- sandbox: no
- Commit: `0d66187` (clean tree)

quiet_check before:
- Quiet check (2026-11-15 16:22:37): warning
  - warning: 1.22 GiB of 15.9 GiB VRAM are held by others

quiet_check after:
- Quiet check (2026-11-15 16:22:38): warning
  - warning: 1.22 GiB of 15.9 GiB VRAM are held by others

## Results

First run: 'same bits' failed for nb >= 3 without w. Cause: the test compared output buffers the step does not write in that variant (uninitialised device memory); the test now zeroes them. Second run:

```
decode d 2048 x 1 tokens, y 1 w 1 in place 0: R' exact, xn 1.2e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 1 tokens, y 0 w 1 in place 0: R' exact, xn 1.2e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 1 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 2048 x 1 tokens, y 1 w 1 in place 1: R' exact, xn 1.2e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 3 tokens, y 1 w 1 in place 0: R' exact, xn 2.2e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 3 tokens, y 0 w 1 in place 0: R' exact, xn 2.4e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 3 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 2048 x 3 tokens, y 1 w 1 in place 1: R' exact, xn 2.2e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 4 tokens, y 1 w 1 in place 0: R' exact, xn 1.9e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 4 tokens, y 0 w 1 in place 0: R' exact, xn 2.2e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 4 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 2048 x 4 tokens, y 1 w 1 in place 1: R' exact, xn 1.9e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 8 tokens, y 1 w 1 in place 0: R' exact, xn 1.7e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 8 tokens, y 0 w 1 in place 0: R' exact, xn 1.6e-07, q8 0.50 steps, same bits yes: ok
decode d 2048 x 8 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 2048 x 8 tokens, y 1 w 1 in place 1: R' exact, xn 1.7e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 1 tokens, y 1 w 1 in place 0: R' exact, xn 1.8e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 1 tokens, y 0 w 1 in place 0: R' exact, xn 1.2e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 1 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 4096 x 1 tokens, y 1 w 1 in place 1: R' exact, xn 1.8e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 3 tokens, y 1 w 1 in place 0: R' exact, xn 1.3e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 3 tokens, y 0 w 1 in place 0: R' exact, xn 1.7e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 3 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 4096 x 3 tokens, y 1 w 1 in place 1: R' exact, xn 1.3e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 4 tokens, y 1 w 1 in place 0: R' exact, xn 1.7e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 4 tokens, y 0 w 1 in place 0: R' exact, xn 1.9e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 4 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 4096 x 4 tokens, y 1 w 1 in place 1: R' exact, xn 1.7e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 8 tokens, y 1 w 1 in place 0: R' exact, xn 2.2e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 8 tokens, y 0 w 1 in place 0: R' exact, xn 2.1e-07, q8 0.50 steps, same bits yes: ok
decode d 4096 x 8 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 4096 x 8 tokens, y 1 w 1 in place 1: R' exact, xn 2.2e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 1 tokens, y 1 w 1 in place 0: R' exact, xn 1.5e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 1 tokens, y 0 w 1 in place 0: R' exact, xn 1.2e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 1 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 2080 x 1 tokens, y 1 w 1 in place 1: R' exact, xn 1.5e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 3 tokens, y 1 w 1 in place 0: R' exact, xn 1.5e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 3 tokens, y 0 w 1 in place 0: R' exact, xn 1.6e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 3 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 2080 x 3 tokens, y 1 w 1 in place 1: R' exact, xn 1.5e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 4 tokens, y 1 w 1 in place 0: R' exact, xn 1.5e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 4 tokens, y 0 w 1 in place 0: R' exact, xn 2.1e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 4 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 2080 x 4 tokens, y 1 w 1 in place 1: R' exact, xn 1.5e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 8 tokens, y 1 w 1 in place 0: R' exact, xn 2.3e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 8 tokens, y 0 w 1 in place 0: R' exact, xn 2.2e-07, q8 0.50 steps, same bits yes: ok
decode d 2080 x 8 tokens, y 1 w 0 in place 0: R' exact, xn 0.0e+00, q8 0.00 steps, same bits yes: ok
decode d 2080 x 8 tokens, y 1 w 1 in place 1: R' exact, xn 2.3e-07, q8 0.50 steps, same bits yes: ok
prefill d 2048 x 37 tokens, y 1 w 1: R' exact, xn 4.8e-04: ok
prefill d 2048 x 37 tokens, y 0 w 1: R' exact, xn 4.8e-04: ok
prefill d 2048 x 37 tokens, y 1 w 0: R' exact, xn 0.0e+00: ok
prefill d 4096 x 37 tokens, y 1 w 1: R' exact, xn 4.8e-04: ok
prefill d 4096 x 37 tokens, y 0 w 1: R' exact, xn 4.8e-04: ok
prefill d 4096 x 37 tokens, y 1 w 0: R' exact, xn 0.0e+00: ok
all ok
rc=0
```

C1: all Coder kernels identical to `work/build/base` with the two R5 `--rename` rules (217 functions in A, 254 in B,
only added probe kernels and ops; `op_qsa_prep` has a second body from the stacked F1 probe unit, the base body is
present). N1: every `k_mega` at most 192 VGPRs (the resnorm probe 188).

## Reading against the plan

As expected after the test fix; the first failure was a test defect, not a kernel one (the kernel output with w was
bit-equal in all runs, the unwritten buffers were the only difference).
