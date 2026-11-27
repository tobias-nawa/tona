# R5: kernel profile and translation unit of the Coder, bit-exact (2026-11-15)

Ticket R5 of `docs/architecture/model-architectures.md` (section 7). `k_mega` becomes `template <int NB, KernelProfile Prof>`, moves into
its own translation unit `gpu/arch/qwen4exp/kernel.hip`, and the host functions of `mega.hip` move into `gpu/mega_launch.cpp`, reaching the
kernels through a `KernelSet`. Base: the R0 build `work/build/base` of `main` 64a8204
(`2026-11-15-multi-arch-r0-baseline.md`). The Coder must stay bit-identical and not slower.

## Plan (written before the runs)

| Check | What runs | Expectation | If it differs |
|---|---|---|---|
| C1 machine code | `scripts/isa-diff.py` base against `work/build/r5`; the kernel name changes (template parameter), so two `--rename` rules map the base names onto the new ones (the rules are stated with the result) | all 6 `k_mega` instantiations equal instruction by instruction, with VGPR, SGPR, spill and LDS counts; every other function equal | a difference in a Coder kernel is unintended: find the cause or revert |
| C1h host code | `llvm-objdump` of `Mega::start_kernel`, `launch_program`, `launch`, `resume_if_yielded`, `init_barrier` in the base object and the new object | not identical: the kernel choice and the launch call moved behind the `KernelSet` (one indirect call per pass, about 20 instructions fewer inlined in `start_kernel`) | the difference is explained by C4, not argued away |
| C3 golden | `scripts/golden.sh compare work/build/r5` (gpu plan: csv, rust, explain) | identical (R0: identical on base) | a logic bug in the touched path; bisect |
| C4 pass curve | `tona-gpu-gen --mega --batch-bench 600`, `TONA_BENCH_HASH=1`, `TONA_BENCH_WIDTHS=1,4,8`, `TONA_BENCH_REPEAT=2`, expert profile of the Coder, 3 runs per arm, arms interleaved (base, r5, base, r5, ...) | equal logits hashes at every width; ms per pass of r5 within +-0.15 ms of base at W1, W4, W8 (source: `2026-11-15-pc-q6k-tile-spills.md` section on the curve: identical machine code differed by 0.01 to 0.15 ms between builds of the same kernel; `bench/noise.toml` has no entry for this tool) | slower beyond that: the cost is in the host path (indirect call, `KernelArgs` copy); fix, repeat; the ticket does not merge slower |
| C6 | `tools/gates/arch_isolation.py` (R8, branch `142-r8-isolation-gate`) and `just check` | clean | move the branch |

Why this size: W = 1, 4, 8 are the three instantiations of the kernel (one width per instantiation, section 6 of the design); the
ticket changes only the launch of a pass, so the rate in tok/s adds nothing the curve does not show. No A/B (C5 runs once at the end of
wave 1).

Active tooling: `TONA_BENCH_HASH=1`, `TONA_BENCH_WIDTHS`, `TONA_BENCH_REPEAT` in the curve; `TONA_DETERMINISTIC=1` in the golden (set by
the script). No profiler, no other TONA_* variable. Builds and GPU jobs of other workers share the line (`scripts/gpu_queue.py`), one at a
time, none during a run. Expected wall time of the queued runs: golden 5 min, each curve run 3 to 4 min.

## Test conditions

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.14 GiB held by others before the run, 0 % busy before the run
- RAM: 60.5 GiB, 54.8 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: `TONA_GPU_LOCK_HELD=1`
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.02, 1.66, 3.07
- CPU use by other programs over 0.5 s: 0.3 % of all threads, swap in use 1.63 GiB
- sandbox: no
- Commit: `8def57e` (clean tree)

Quiet check before the first curve run (10:35, taken inside the GPU slot of the run): warning, only "1.14 GiB of VRAM held by others" (the
desktop); CPU 0.6 % busy, load average 1.0. Between the six curve runs the GPU line was held by this ticket, so no other worker's GPU job
or build ran. The check after the last run was not possible in the slot (the line was occupied by another ticket's long job for more than
10 minutes); taken 11 minutes later it says "do not measure" (a GPU job of another worker was running), which describes that job, not the
curve runs. Branch `142-r5-kernel-profile` at commit `8def57e`, base `work/build/base` (R0, commit `64a8204`).

## Results

### C1 machine code (`scripts/isa-diff.py`, new `--rename` and per-body comparison)

First without rename: 217 functions in each build; only the six `k_mega<NB, ...>` instantiations differ, by name (the second template
parameter is now `Qwen4ExpProfile<Q6Layout>` instead of `Q6Layout`, the argument type `MegaScalars` left the anonymous namespace). With
the two rules

```
--rename 'k_mega<(\d), \(tona::gpu::\(anonymous namespace\)::Q6Layout\)(\d)>=k_mega<\1, tona::gpu::(anonymous namespace)::Qwen4ExpProfile<(tona::gpu::(anonymous namespace)::Q6Layout)\2>>'
--rename '\(anonymous namespace\)::MegaScalars=MegaScalars'
```

applied to the names of the base build: `217 functions in A, 217 in B`, **`all kernels identical`**: every instruction of all six `k_mega`
instantiations (NB 1, 4, 8, each plain and with the Q6_K tile loops) and the resource fields of their code-object records (VGPRs, SGPRs,
spill counts, LDS, scratch, kernarg size). Expected: identical; got: identical.

A first run of the unchanged isa-diff reported `code differs: extend_deadline` (a `__noinline__` helper of `matvec_barrier.h` that every
translation unit compiles with its own body; the tool kept the body of whichever object it saw first, and the object order changed with
the new files). isa-diff now keeps every distinct body per name and compares them as a set; the second run above is with that fix.

### C1h host code of the moved launch functions (`llvm-objdump -d -r` of the base `mega.hip.o` against the new `mega_launch.cpp.o`)

Not identical, as expected: the kernel choice and the launch moved behind the `KernelSet`.

| Function | Base | R5 | Difference |
|---|---|---|---|
| `Mega::launch(int)` | 21 instructions | 21 | identical |
| `Mega::launch_program` | 96 | 96 | identical except the numbers of two string literals |
| `Mega::start_kernel` | 665 lines, 522 instructions without relocations | 644 lines, 508 | 14 instructions fewer: selection of the instantiation and the six kernel arguments are now in `kernel.hip`; the new code is one indirect call (`callq *%rbx`) through `KernelSet::launch` plus filling a `KernelArgs` |
| `Mega::resume_if_yielded` | 246 | 247 | the inlined `start_kernel`; one instruction more, string numbers |
| `Mega::init_barrier` | 228 | 228 | register allocation and literal numbers differ (set-up only, once per process) |
| `Mega::init_kernels` | 902 | 281 | the probing moved into `KernelSet::probe` (set-up only) |

The per-pass path (`launch`, `launch_program`, `start_kernel`) differs only by the indirect call, so C4 decides.

### C3 golden

`scripts/golden.sh compare work/build/r5` (gpu plan): csv, rust, explain identical. Expected: identical.

### C4 pass curve (`tona-gpu-gen --mega --batch-bench 600`, `TONA_BENCH_HASH=1`, widths 1, 4, 8, 2 repeats, 3 runs per arm, interleaved base, r5)

Logs: `2026-11-15-multi-arch-r5-curve-{base,r5}-{1,2,3}.log`. ms per pass, median of 6 values (2 repeats x 3 runs) with minimum and maximum:

| W | base | r5 | r5 - base | logits hash (all runs, both arms) |
|---|---|---|---|---|
| 1 | 23.70 (23.61 to 23.77) | 23.66 (23.58 to 23.74) | -0.04 ms (-0.17 %) | 2e8e1b22c3a12255 |
| 4 | 37.92 (37.76 to 37.97) | 37.91 (37.87 to 38.00) | -0.02 ms (-0.04 %) | 2e8e1b22c3a12255 |
| 8 | 57.67 (57.63 to 57.72) | 57.65 (57.30 to 57.72) | -0.02 ms (-0.04 %) | 07dc3ad00fd072f2 |

Expected: equal hashes and within +-0.15 ms. Got: equal hashes at every width in all 6 runs; the differences are far inside the spread of
the base build itself (up to 0.16 ms between its own runs at W1). The indirect call costs nothing measurable per pass. The rate in tok/s
is not claimed separately; the curve is the evidence. Not run: `scripts/ab.py` (C5 runs once at the end of wave 1).

### C6

`tools/gates/arch_isolation.py` (branch `142-r8-isolation-gate`) on this tree: `arch-isolation: ok (374 checked)`. `just check`: see the
hand-back.

## Second run on the combined tree (R5 rebased onto main 33a4f91 with R1 to R4, R6, R7; `kernels_of` added)

The coordinator merged R4, R6, R1, R2, R3 and R7 while this ticket waited in the GPU line, so R5 was rebased onto `main` 33a4f91, and
`Mega` now picks its `KernelSet` through `kernels_of(const Model&)` (next to `make_arch_program`; a member of `ArchProgram` is not possible
because the kernels are probed before the builder exists). Commit `a5cbe9a`; all runs again against the same base `work/build/base`
(R0, 64a8204), plan and expectation as above.

| Check | Result |
|---|---|
| C1 (same two `--rename` rules) | 217 functions in A and B, `all kernels identical` |
| C3 golden gpu | csv, rust, explain identical |
| C6 | `arch-isolation: ok (403 checked)` with the R8 gate (rebased onto the same main) |
| C4 ms per pass, median of 6 (min to max) | W1: base 23.77 (23.61 to 27.66) / r5 23.71 (23.56 to 24.26), -0.05 ms; W4: 37.98 (37.85 to 54.16) / 37.89 (37.83 to 37.96), -0.09 ms; W8: 57.69 (57.45 to 67.48) / 57.50 (57.35 to 57.72), -0.18 ms. Logits hashes equal at every width in all 12 runs (2e8e1b22c3a12255 at W1 and W4, 07dc3ad00fd072f2 at W8) |

One base run (run 2) shows 27.7, 54.2 and 67.5 ms at the three widths: a disturbance of that run (the other arm did not show it), kept in
the table; the medians are robust against it. Logs: `2026-11-15-multi-arch-r5-curve2-{base,r5}-{1,2,3}.log`. Quiet check before the
first run: warning, only the 1.17 GiB the desktop holds; after the last run (taken right after it): warning, unchanged. Expected: equal
hashes, within +-0.15 ms; got: equal hashes, r5 0.05 to 0.18 ms faster than base. Not claimed as a gain: the machine code is identical, so
this is the spread between two builds of the same kernels (the first run showed 0.02 to 0.04 ms).
