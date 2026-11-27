# B1 (#163): parallel verification scan and decode embeddings

Host Ryzen 9 9950X, RX 9060 XT gfx1200, Coder IQ1_M, MTP head. Arm A `work/build/w163b7`, arm B `work/build/w163h`
(the same plus the commit "Parallel verification scan and decode embeddings"; both have `TONA_HOST_TRACE` compiled in, off).

## What changes and why

The host trace of a decode cycle (2026-11-17-b1-cycle-breakdown.md) shows 0.31 ms of serial `argmax` scans of the verify
logits (block of 5 to 6 rows of 248,320 floats, `Generator::accept_drafts`) and 0.30 ms of serial embedding and PLE row
dequantization (`Mega::embed_tokens`, `n >= 16` threshold for the prefill chunks) on the critical path of every cycle,
together 1.4 % of a 43.8 ms cycle. Both loops are over independent rows. B scans all rows of the block in parallel
(`argmax_scan`, NaN reported only for the rows the acceptance loop uses, as before) and lets `embed_tokens` use the
threads from 2 tokens on. Bit-exact by construction: the same function on the same row, the same dequantization per
token. Prior art: this is the host-side item 5 of the research list of the plan (vLLM Model Runner V2 removes host work
from the critical path); nothing here is new technique.

## Plan (written before the runs)

**Question.** Does it move decode, and by how much?

**Expectation.** Decode +0.3 to +1.2 % (saved: up to 0.31 + 0.30 ms of a 43.8 ms cycle, minus the cost of waking the
OpenMP threads, 5 to 20 us twice per cycle). Code prompts (5 tokens per pass, 6 rows) gain more than text (2.5 tokens per
pass). A loss is possible if waking the OpenMP threads costs more than the scan it saves, then the threshold for the
embeddings or the scan goes back up.

**Checks.** R1 unit tests, R0 `just check`; R3 goldens (`scripts/gpu-status.sh`, greedy tokens identical through the
server and generator paths, MTP drafter); R5 `scripts/ab.py --a w163b7 --b w163h --workload short,set`.

**Why this size.** One build pair, decode only (prefill uses the same `embed_tokens` in chunks of 2,048 tokens, where the
threshold did not matter). `short` is 2K context x 400 tokens x 2 requests (noise 0.23 percent for decode@2000, 0.61
for the `write` prompt), `set` is 14 prompts (noise 0.07 for code). An effect of 0.3 to 1.2 percent needs 4 to 6
rounds; `--max-rounds 6`.

**Active tooling.** None (no trace, no profiler, no deterministic mode).

## Results

`scripts/ab.py --a w163b7 --b w163h --workload short,set`, 6 rounds (report `work/ab/163-b1h.md`, kept out of the repo):
decode@2000 -0.21 %, decode@2000:write -0.23 %, decode-code -0.29 %, decode-text -0.02 %; prefill@2000 -0.10 %,
prefill@2000:write +0.08 %. Output SHAs and expert-cache counts of A and B equal. Expectation was decode +0.3 to
+1.2 %: **all four decode metrics are outside it, no gain.** The change is not merged.

## Finding: why the 0.6 ms of the trace did not show up

Question: was the host work overlapped with GPU work and so off the critical path? Answer from the existing trace of
2026-11-17 (251 cycles, `/var/tmp/claude/163/prof1/report_skip250.txt`), no new run:

- No. Both segments lie in gaps with no pass running. `fw_begin -> fw_staged` (0.300 ms, embeddings) runs after
  `drafted` and before `fw_launched`, so the verify kernel has not started; `fw_fetched -> verified` (0.307 ms, the
  scan) runs after the verify kernel was fetched and before the next `mtp_begin`, so the next draft has not started.
  The trace has no GPU work in either gap, so they were on the critical path, as the plan assumed.
- So the saving was real in the trace and the cause of the missing gain is on the host side of the change, not an
  overlap. What the trace does not show is whether the parallel versions are faster at all: both loops move 6 MB (scan:
  6 rows x 248,320 floats) or a few MB of dequantized rows, so they may already be limited by one core's streaming rate
  plus the wake-up of the OpenMP team (the CPU-expert pool just finished its 48th router, its threads may sleep or spin),
  and the 0.6 ms is then not reducible by threads. **Unverified**: no trace of arm B exists. Deciding it needs
  `TONA_HOST_TRACE` on w163h, one run of 250 cycles.
- **Trace of arm B (w163h, `TONA_HOST_TRACE`, same driver and workload, 251 cycles, mean 44.157 ms; run
  `/var/tmp/claude/163/prof2`):** `fw_begin -> fw_staged` 0.297 ms (A: 0.300), `fw_fetched -> verified` 0.345 ms
  (A: 0.307). The parallel loops are not faster at all: the embedding segment is unchanged, the scan segment 0.04 ms
  slower. So the end-to-end result is explained: the parallel versions save nothing, and the cause is not overlap. Why
  they save nothing (OpenMP wake-up of the team, memory-bound scan, the segment's other work) is still not separated.
  The segment `fw_fetched -> verified` also contains more than the scan, which this trace does not split.

- Do not retry the thread-parallel scan or embedding without that trace of B first; the end-to-end result says the
  parallel version does not beat the serial one by the 0.3 ms that was expected.

## Region marks (plan, written before the runs)

**Question.** Why are the parallel loops not faster: the wake-up of the OpenMP team, or the work itself? Main since
423d652/045271e sets `OMP_WAIT_POLICY=passive`, so idle threads sleep in the kernel and every parallel region pays a
futex wake; this build (a branch of an older main) does not set it, so both policies are set explicitly.

**Marks** (`RegionTrace` in `engine/backend/host_trace.h`, diagnosis only): `scan_enter` before the region,
`scan_worker_start` (earliest start of a row from the second on, i.e. of a thread other than the entering one),
`scan_first_done` (earliest row end), `scan_exit` after it; the same four with `emb_` for the PLE embedding loop.
`enter -> worker_start` is the wake-up latency, `worker_start -> exit` the work of the slowest thread.

**Runs.** Build `w163m` (= `w163h` plus the marks), TONA_HOST_TRACE, the driver of `prof1`, 251 cycles each, four
configurations: `OMP_WAIT_POLICY` passive/active x `OMP_NUM_THREADS` 6/16 (16 = physical cores of the 9950X). The active
policy is a diagnosis only, not a candidate default.

**Expectation.** Passive: wake latency 30 to 100 us per region (futex wake of up to 5 threads), so 2 regions cost
60 to 200 us of the 0.6 ms, the rest is the work. Active: latency below 5 us. If the latency in passive is below
20 us, the wake-up is not the cause and the work is memory-bound (6 MB per scan on 5 to 6 rows).

**Why this size.** The marks are per cycle and 251 cycles give the mean with a spread of a few percent; one server
start per configuration.

**Active tooling.** `TONA_HOST_TRACE` (the trace itself), `OMP_*` as listed.

### Results

Build `w163m`, 251 cycles per configuration, `/var/tmp/claude/163/prof3` (hw_report and quiet_check before and after
there). Means per cycle in us; "scan" = verify scan of 5 to 6 rows, "emb" = PLE embedding loop.

| Policy x threads | cycle ms | scan: enter -> first row done | scan: worker start -> exit | scan: whole region | `fw_begin -> emb_enter` | emb region |
|---|---:|---:|---:|---:|---:|---:|
| passive x 6 | 60.6 (disturbed) | 111 | 455 | 566 | 353 | 21 |
| active x 6 | 44.8 | 67 | 241 | 308 | 294 | 11 |
| passive x 16 | 43.8 | 66 | 239 | 305 | 175 | 11 |
| active x 16 | 43.8 | 81 | 286 | 367 | 250 | 10 |

(the passive x 6 run has a cycle of 60.6 ms against 43.8 to 44.8 in the other three: the machine was not quiet or the
run was disturbed, its numbers are not used.)

Findings, from these numbers:

1. **The embedding region is not the cost.** The whole parallel dequantization loop takes 10 to 21 us. The 0.3 ms of
   `fw_begin -> fw_staged` is spent before it (`fw_begin -> emb_enter`: 175 to 353 us, the PLE row indices and the
   io_uring read of the PLE rows, `ple_rows` and `ple_io_->read`), serial. Parallelizing the dequantization loop could
   not save more than the 10 to 20 us it costs. The change aimed at the wrong part of the segment.
2. **The scan does not scale with threads.** The whole parallel region takes 305 to 367 us (active and passive alike)
   against 307 us for the serial scan in the base trace. Within it the first 66 to 81 us pass before the earliest row ends
   and the slowest thread needs 240 to 290 us after the workers started, while one row serially takes about 50 us
   (307 us / 6 rows). A row is 5 times slower in a thread of the team than serially. Wake-up latency is not the cause:
   the active policy (no sleeping) is not faster than the passive one at 16 threads (305 vs 367 us, the order is
   within the noise of one run each), and the wait policy changes the cycle by 0.05 ms.
3. Why a row is slower in the team is **not determined**. Not tested, candidates: the logits are read from memory the
   GPU wrote (pinned host memory, possibly uncached for the CPU: reading it from several cores does not scale), or the
   scan is bound by memory latency of that region. A test would be one run of the scan on a copy of the logits in
   ordinary memory.
4. The odd result `scan_first_done -> scan_worker_start` = 0 (the earliest row end coincides with the earliest start of
   a second row, in all four configurations) is not explained; it may hint that the rows run one after the other
   because of a lock or of the memory, not at the same time. Not examined.

Conclusion: the 0.6 ms are real but not recoverable by threads in these two places. The embedding part is a serial
gather (0.18 to 0.35 ms) that a thread pool would not touch; the scan part is slowed down by the access to its input,
not by waking threads. A fixed thread pool outside OpenMP would save at most the 10 to 20 us of the embedding region
and the wake-up of the scan region. Not tried.

## Logits memory (plan, written before the run)

**Question.** Is the scan of the verify logits slow because of the memory they arrive in (pinned host memory written
by the GPU copy, `PinnedAlloc` = `hipHostMalloc` default flags, kernel GTT buffer with `HSA_USERPTR_FOR_PAGED_MEM=0`)?
In the region-mark runs one row took 240 to 290 us in a team thread against about 50 us serially.

**What the code says.** `fetch_logits` copies the logits with `hipMemcpyAsync` D2H into a `PinnedVec`; the scan reads
that buffer on the CPU. Whether the CPU maps it cached is not known from the code (the HIP default for
`hipHostMallocDefault` depends on `HIP_HOST_COHERENT`); the test measures it.

**Test.** `TONA_DIAG_SCAN=1` with `TONA_HOST_TRACE` on build `w163n` (= `w163m` plus `diag_scan` in `verify.cpp`, diagnosis
only): cycles alternate between (a) a serial scan of the pinned buffer (cold), (b) nothing, (c) a `std::copy` into
ordinary memory, then a serial and a parallel scan of the copy. One server start, the driver of `prof1`, 6 threads,
passive wait policy. `OMP_NUM_THREADS=6 OMP_WAIT_POLICY=passive`.

**Expectation.** If the buffer is uncached for the CPU: the serial scan of the pinned buffer takes much more than the
50 us per row of the base trace (about 300 us for 6 rows is the base: it would be 1 ms or more), the copy takes
1 ms or more, and the scans of the copy take 20 to 60 us per row serially and scale in parallel. If the buffer is
cached: serial scan of the pinned buffer about 300 us (as in the base trace), the copy about 150 to 300 us (6 MB at
20 to 40 GB/s), the scan of the copy about the same as the pinned one, and the parallel scan of the copy faster by
2 to 4 times (the rows are independent). Outside both: a finding.

**Why this size.** The marks are per cycle and cheap; 250 cycles give about 83 samples of each kind.

**Active tooling.** `TONA_HOST_TRACE`, `TONA_DIAG_SCAN`, `OMP_NUM_THREADS=6`, `OMP_WAIT_POLICY=passive`. The diagnosis
adds work to the cycle (the production scan still follows), so the cycle time of that run is not a result.

### Results

Build `w163n`, 251 cycles (the first 30 of each kind dropped), `/var/tmp/claude/163/prof4`, `OMP_NUM_THREADS=6`,
passive. Times in us:

| Measurement | n | mean | median |
|---|---:|---:|---:|
| serial scan of the pinned logits buffer (cold) | 137 | 285 | 256 |
| `std::copy` pinned -> ordinary memory | 137 | 161 | 131 |
| serial scan of the copy | 137 | 281 | 254 |
| parallel scan of the copy (`#pragma omp`, 6 threads) | 137 | 278 | 252 |
| PLE: `fw_begin -> emb_start` | 471 | 0.8 | 0.8 |
| PLE: `ple_rows` (row indices) | 471 | 0.7 | 0.6 |
| PLE: io_uring read of the rows | 471 | 161 | 62 |

Outcomes against the plan:
1. **The logits memory is not the cause.** The pinned buffer is read as fast as ordinary memory (285 vs 281 us). The
   "uncached" outcome of the plan did not occur; the cached outcome did, except for the last point.
2. **The parallel scan of the copy is not faster than the serial one (278 vs 281 us).** Outside the plan: nothing
   runs in parallel. Cause found in the build, not in the memory: `server/verify.cpp` and `gpu/prefill_entry.cpp` are
   compiled as HIP language (`set_source_files_properties(... LANGUAGE HIP)`), the compile line (`build.ninja`) has
   no `-fopenmp`, and their objects contain no `__kmpc_fork_call`/`GOMP_parallel` (`nm`): `#pragma omp` in them is a
   no-op. This explains every earlier result of this file: the "parallel" scan and embeddings of `w163h` ran serially
   (decode +/-0 %), the region marks showed rows one after another (`scan_first_done -> scan_worker_start` = 0).
   The same holds for the pragmas that existed before (`prefill_entry.cpp` `n >= 16`, and in `gpu/mega_experts.cpp`,
   `mega_select.cpp`, `mega_prefetch.cpp`, `mega_alloc.cpp`, `prefill_mtp.cpp`, `arch/qwen4exp/mtp.cpp`,
   `forward_moe.hip`: all HIP language, no `-fopenmp`).
3. A serial row of the scan takes 47 us (281 us / 6 rows), about 21 GB/s: the scan is compute-bound in a scalar loop,
   not in memory, so threads should help (6 rows on 6 threads: expect about 50 to 70 us for the region).
4. The PLE gather: the row indices cost 0.7 us; the io_uring read has a mean of 161 us and a median of 62 us. The cost
   is a heavy tail (rows not in the page cache, or the ring waking), not the median. The dequantization loop after it is
   10 us serial.

Consequence: the next steps are (a) `-fopenmp` on `verify.cpp` and `prefill_entry.cpp` only (no kernel file is touched),
(b) a prefetch of the PLE rows during the MTP chain to hide the tail of the read.

## OpenMP in HIP-language sources (survey, plan)

Every `#pragma omp` in a source compiled as HIP language was dead (no `-fopenmp` in its compile line, no
`__kmpc_fork_call` in its object). Survey of the sites, before turning them on (what the loop does, where it runs, race
check):

| Site | Path | Loop body | Race check |
|---|---|---|---|
| `gpu/mega_select.cpp:233` `read_misses` | **decode** (misses of the expert cache, every pass with misses) | `gguf().read` (pread) of 3 parts per miss into `staging_ + i * slot_bytes + offs[k]` | distinct destinations; pread is thread safe; stats updated after the loop |
| `gpu/mega_prefetch.cpp:77` `refill_cache` | **decode** (refill while the GPU finishes the layer) | `memcpy` of one expert part into `dst + i * slot_bytes + offs[k]` | distinct destinations; sources read only |
| `gpu/prefill_entry.cpp:270, 297` `embed_tokens` | **prefill** chunks (`n >= 16` before) and decode (`n >= 2`, mine) | `dequantize_row` of a token row into `out + b * d` | pure function, distinct rows |
| `gpu/prefill_mtp.cpp:74` `mtp_part` | **prefill** (MTP head input) | `dequantize_row` into `mtp_pf_e_host_ + i * d` | distinct rows |
| `server/verify.cpp` `accept_drafts` | **decode** (verify scan) | `argmax_scan` per row into `am[r]` | distinct elements; the NaN report is made after the loop |
| `gpu/mega_experts.cpp:42, 217` | load time (`prefill_cache`, pinned experts) | `gguf().read` into distinct slots | as `read_misses` |
| `gpu/mega_alloc.cpp:121, 148` | load time (`upload` tiling, `upload_q8_0`) | `tile_q6_k` per 16 tiles; Q8_0 quantization per block of 32 | distinct output ranges; `v` read only |
| `gpu/arch/qwen4exp/mtp.cpp:182` | load time (MTP head rows) | `Mega::head_row` (pread, local vectors, `iq4::quantize_iq4_nl`) | all state local; `mtp_head_q4_` read only |
| `gpu/forward_moe.hip:138` `load_misses` | the non-megakernel `Forward` path (reference), per layer | as `read_misses` | as `read_misses` |
| `tools/tona_gpu_matvec.hip:140 ff.` | tool (`tona-gpu-matvec`) | its own `parallel`/`for`/`critical` | not examined, a tool |

One behaviour change to note: an exception thrown inside an OpenMP region (a failing `pread`) terminates the process
instead of propagating; these loops had this property before the HIP move, and a read error here was fatal for the
pass before as well.

Plan: `tona_openmp_for_pragmas(<target>)` in `engine/openmp_pragmas.cmake` adds `-fopenmp` per source to every HIP-language
source with a pragma (of `tona_gpu`, `tona-server`, `tona-gpu-matvec`), so device code of the other files is unchanged;
the gate `tools/gates/openmp.py` (in `just check`, tests in `tools/tests/test_openmp_gate.py`) fails for a pragma in a
source without the flag. Checks: isa-diff (all kernels identical), `nm` shows `__kmpc_fork_call`, goldens (TONA_DETERMINISTIC),
`ab.py` of all together on short, set, mid against the build of the same tree without the flag.

### Run plan: all pragmas on (written before the runs)

**Question.** What do the pragmas that never ran do to decode and prefill when they run?

**Arms.** A `w163r` (this tree before the commit "OpenMP for the host code of HIP-language sources", with the host
trace marks, `TONA_DIAG_SCAN` and the parallel verify scan, all dead), B `w163o` (the same plus `-fopenmp` per source).

**Expectation.** Decode +0.2 to +2 %: the verify scan from about 280 us to about 60 us (+0.5 %), `read_misses` and
`refill_cache` (expert cache misses read and copied by 6 threads instead of 1) on the sets with misses (`2000:write`
81,757 hits / 28,520 misses per request), the embedding loop 10 us. Prefill +0 to +3 %: the PLE embedding of each chunk
of 2,048 tokens and the MTP head input go parallel (`mtp_part`: S rows). Risk: more threads compete with the CPU
experts and the router selection for the cores (6 OpenMP threads, passive wait); a loss on the decode of text is then a
finding. Source of the numbers: the trace of this file (scan 281 us serial) and `research/measurements/2026-11-17-b1-cycle-breakdown.md`.

**Confound.** The OpenMP runtime also differs: `w163r` uses the system libgomp, `w163o` the toolchain libomp, and the
runtime also serves the CPU expert path (`engine/cpu/experts.cpp`), which is not touched by the pragmas. If the result is
outside the expectation either way, one more run separates the effects: `w163r` against `w163r` with
`LD_LIBRARY_PATH` set to the toolchain libgomp directory (same binary, runtime only).

**Other checks.** `nm` shows `__kmpc_fork_call` in the objects of the 9 sources; isa-diff of `w163r` against `w163o`:
all kernels identical (the flag is host-only and per source); `tools/gates/openmp.py` green on the compile database;
goldens (`TONA_DETERMINISTIC` identical).

**Why this size.** `ab.py` short, set, mid, 6 rounds at most (noise 0.23 percent decode, 0.04 percent prefill); a split
per file follows only if the whole shows a gain or a loss.

**Active tooling.** None beyond `OMP_NUM_THREADS=6` of the ab.py servers.

### Results (all pragmas on)

`scripts/ab.py --a w163r --b w163o --workload short,set,mid --max-rounds 6` (exclusive queue job, 6 rounds, report
`work/ab/163-final.md`, kept out of the repo). Output SHAs and expert-cache counts of A and B are equal for all three
prompts. B against A, mean:

| Metric | B vs A | Expected | Verdict |
|---|---:|---|---|
| prefill@2000 | +0.37 % | 0 to +3 % | inside |
| prefill@2000:write | +0.28 % | 0 to +3 % | inside |
| prefill@30000 | +0.15 % | 0 to +3 % | inside |
| decode@2000 | +0.20 % [-0.03, +0.43] | +0.2 to +2 % | at the lower edge |
| decode@2000:write | +0.88 % | +0.2 to +2 % | inside |
| decode@30000 | +0.12 % [-0.35, +0.59] | +0.2 to +2 % | slightly below; one outlier round -0.79 %, others +0.19 to +0.50 % |
| decode-code | +0.64 % | +0.2 to +2 % | inside |
| decode-text | +0.27 % [-0.25, +0.78] | +0.2 to +2 % | inside |

All gains are small and none is negative. The sets with many cache misses (`2000:write`, code) gain most, as expected
from the parallel `read_misses` and `refill_cache`. The expected result for text and long context (few misses) is the
low end, which is what was measured. Test conditions: `hw_report.py` and `quiet_check.py` were not run around this ab
run (only around R8 below); ab.py records its own load log. The system libgomp (A) and the toolchain libomp (B) differ,
see the runtime run below.

### Release build 3 (final head, `scripts/release-build.sh work/release-163 --version 0.0.1`)

Built through the cpu queue, rc 0, 25 files. The assets include the toolchain `libgomp.so.1` and `libomp.so` (both
SONAME `libomp.so`; `libgomp.so.1` is the alias the objects link against). The bundle was assembled as `bin/` and
`lib/` (assets without the `rocm-<id>-` prefix), checked and removed afterwards:

- `release_needed.py --runpath-only` on `tona-server`, `tona-top`, `tona-install`: ok (RUNPATH of `tona-server` and
  `tona-top` is `$ORIGIN/../lib`; `tona-install` has none).
- `tona-server --help` with `env -i LD_LIBRARY_PATH=` starts (prints its argument message, exit 2); `ldd` resolves every
  non-base library inside the bundle `lib/`, base libraries from the system (libstdc++, libm, libgcc_s, libc):

```
libamdhip64.so.7, libgomp.so.1, librocm_kpack.so.0, librocprofiler-register.so.0, libamd_comgr.so.3,
libhsa-runtime64.so.1, libclang-cpp.so.24.0git, libLLVM.so.24.0git, librocm_sysdeps_*: all => <bundle>/bin/../lib/
libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6, libdl, librt, libpthread => /lib64
```

### Plan: R8 pressure series on the final build (written before the run)

**Why.** `select_checks` names "pressure series needed" (memory, pinning and stall paths): `read_misses` now issues its
preads from several threads under cache misses, and `refill_cache` copies in parallel; both sit on the stall path of a
pass. Recipe: R8 of `docs/contribute/measurement.md` (one series of 20 requests, 10 under page-cache pressure, `TONA_DIAG=1`), as
run for #83 (`2026-11-01-slow-decode.md`): the driver of that run is not in the repo; a copy with the paths, the port
8091 and the `TONA_*` names changed is `/var/tmp/claude/163/drive_r8.py` (mode `press`, N = 10: 10 pairs of a 30K and a
130K prompt, 200 tokens greedy, the odd pairs under pressure).

**RAM only, capped, no VRAM pressure.** The pressure is `cat` of the 55 GB model file the server does not use into the
page cache, in an own `systemd-run --user --scope` with `RuntimeMaxSec=3850` (it ends even if the driver dies); the server
runs in a scope with `MemoryMax=54G` and `MemorySwapMax=0`. The server itself is the normal one (`--ctx 140000`); the
driver waits for the VRAM of a previous server to be free (below 3 GB used) and adds no VRAM load. Port 8091.

**Build.** `w163o` rebuilt at the final head (the goldens run on it first).

**Expectation** (as in the #83 plan, register route: the pinned memory is not changed by this work): no aborted pass and
no fault in the 20 requests; slow decodes (under 55 tok/s) at most 3 of the 10 pressured requests; evicted time in the
decode of the pressured requests below 5 s in total; longest eviction episode below 3 s; decode of the unpressured
requests as in the ab.py of this file. A failed pass (barrier abort) is recorded and the server restarted. **Why this
size:** the size of the recipe; by the rule of three 0 faults in 10 bound the rate below 30 %, so a fault is only found
by chance, and a second series only follows a miss.

**Active tooling.** `TONA_DIAG=1`, `TONA_DECODE_DIAG=1`, the 10 ms eviction sampler and the PSI/vmstat sampler of the
driver; `OMP_NUM_THREADS=6`.

### Plan: R8 on the base build `w163r` (written before the run, after the MISS below)

**Question.** Is the total evicted time of 14.0 s in the 10 pressured requests of `w163o` caused by the change (parallel
`read_misses` preads and parallel `refill_cache` under page-cache pressure), or by the bound of 5 s (copied from another
series) or the machine?

**Run.** The same series (`scripts/r8_pressure.py --build w163r --tag r8r`, 10 pairs, odd pairs pressured, same page
cache pressure, same caps `MemoryMax=54G` `MemorySwapMax=0`, no VRAM pressure), one exclusive job, test conditions before
and after.

**Decision rule (written before the run, coordinator).** The total of evicted time over the 10 pressured requests is
compared with that of `w163o` (14.0 s). If `w163r` is in the same range, within its own spread over its 10 pressured
requests (per-request 30K 0.4 to 1.1 s, 130K 1.6 to 2.9 s for `w163o`), the MISS comes from the bound or the machine, not
from the change: the bound is corrected with a source and the change can merge. If `w163o` is clearly worse, the parallel
preads under pressure are the suspect and the change does not merge as it is.

**Expectation.** The same range (10 to 18 s) for `w163r`: the 2026-11-01 series (`s86f`, 5.3 s per 20 requests without
overflow) is a different build and machine state (9600X, a different server memory plan), so the bound was not a
measurement of this build. **Why this size:** one more series of the same size is the smallest run that gives a second
total with its own spread; a single series cannot separate a factor of 2.8 from the run-to-run spread of the eviction.

**Active tooling.** As for the R8 plan above.

### Results (R8)

`scripts/r8_pressure.py --build w163o --tag r8` (exclusive queue job on w163o built from `0f072b9`'s sources; raw
lines in `/var/tmp/claude/163/r8/r8.jsonl`, not in the repo), 10 pairs of a 30K and a 130K prompt, 200 tokens, odd
pairs under page-cache pressure (RAM only, no VRAM pressure, server `MemoryMax=54G`, `MemorySwapMax=0`).

| Check | Expected | Measured | |
|---|---|---|---|
| failed requests | 0 | 0 of 20 | ok |
| slow decodes (below 55 tok/s) of the 10 pressured | at most 3 | 0 (lowest pressured decode 59.1 tok/s) | ok |
| evicted time in the decode of the pressured requests | below 5 s in total | **14.0 s** (30K: 0.4 to 1.1 s each; 130K: 1.6 to 2.9 s each) | **MISS** |
| longest eviction episode | below 3 s | 0.8 s | ok |
| decode, unpressured | as in the ab | 30K: 70.4 to 73.6 tok/s, 130K: 61.8 to 71.6 tok/s | ok |

Decode (pressured): 30K 67.6 to 75.6 tok/s, 130K 59.1 to 64.9 tok/s. **Finding:** the total evicted time is 2.8 times the
bound copied from the #83 plan. No arm without this change ran in this series, so it is **not attributed**: the
expectation came from the 2026-11-01 series, where four of six slow decodes carried 1.9 to 3.1 s of evicted time under
pressure, the same range as the 130K requests here (1.6 to 2.9 s), so the single-request evictions are as before and
the sum is large because every pressured request has some. Whether the parallel `read_misses` or `refill_cache`
contribute is open; the check that answers it is the same series on `w163r` (no GPU job was queued for it, by the
coordinator's pause). Test conditions: `hw_report.py` (commit `a79171a` clean tree, 56.9 of 60.5 GiB RAM available,
swap 1.15 GiB in use, load 2.47) and `quiet_check.py` before and after the run: both **warned**, before 2.08 GiB VRAM
held by others, after the same plus "the system is swapping (2 pages in 1 s)"; measured anyway, the pressure series
itself fills the page cache.


## Draft-time PLE prefetch (plan, written before the run)

**Question.** Does asking the page cache for the PLE rows of the next verify block while the MTP head drafts remove the
tail of the serial io_uring read (mean 161 us, median 62 us, n = 471; table above, finding 4)?

**Prior art in this repo.** The Mac backend does exactly this for its mapped table: `backend/ple_prefetch.h`, level 2
asks for the next pass's rows while the MTP head drafts (`Pass::prefetch_rows`, decode +4.2 % code, +5.9 % text at level
1, +1.2 % more at level 2, `research/measurements/2026-11-14-mac-ple-prefetch.md`). On Linux the default is 0 ("off
until measured on the reference PC") and nothing calls it: the rows go through `DirectRows` (io_uring, buffered, no
readahead), which `madvise` on a mapping would not warm.

**Change.** `DirectRows::advise(rows, n)`: `posix_fadvise(POSIX_FADV_WILLNEED)` on the byte ranges of the rows (merged per
page by `ple_prefetch::ranges`, the shared helper) on the same file descriptor; `WILLNEED` starts the read into the page
cache even under `POSIX_FADV_RANDOM`. `Mega::prefetch_ple(seq, from, to)` computes the rows with `ple_rows` and calls
it; the generator calls it with the tokens drafted so far, so the next `embed_tokens` finds the rows cached. The bytes
the copy reads never change (bit-exact: only the page cache state differs). Gated by `TONA_PLE_PREFETCH=2`, default 0 on
Linux until measured. If the per-row syscalls cost more than 20 us on the draft path, they move to one helper thread.

**Expectation.** The read's mean falls from 161 us toward its median of 62 us, so at most about 100 us of a 44 ms cycle:
decode 0 to +0.3 % (code prompts, with more tokens per pass, the upper end); a loss when the advice itself costs more than
the tail it removes. Source: the 471-sample table above.

**Checks.** R1 unit test of `ranges` for the byte-range case; R3 goldens identical with `TONA_PLE_PREFETCH=2`
(`TONA_DETERMINISTIC`); R5 `scripts/ab.py --a w163o --b w163o --b-env TONA_PLE_PREFETCH=2 --workload short,set` (the
env switch needs one build only), rounds from noise.toml: effect 0.1 to 0.3 % against 0.23 % decode noise (0.07 for the
code set), `--max-rounds 6`. **Why this size:** the smallest workload with cold rows is the `set` run at 2K context; a
longer context adds nothing for a host-side row read.

**Active tooling.** `TONA_HOST_TRACE` compiled in but off; `TONA_PLE_PREFETCH` as above.

### Results (R8 on `w163r`)

Same script, caps and pressure, run after the `w163o` series (08:00 against 07:38, same quiet_check warning before and
after: 2.08 GiB VRAM held by others; `/var/tmp/claude/163/r8r.log`). 0 failed requests, 0 slow decodes (lowest pressured
decode 58.9 tok/s), longest eviction episode 0.6 s. Evicted time per pressured request in ms, pair order 1 to 10:

| Build | 30K, 130K per pair | Total | Median per request |
|---|---|---:|---:|
| `w163o` (change) | 870 1624, 1143 2596, 396 2871, 644 1663, 592 1628 | **14,027** | 1,383 |
| `w163r` (base) | 255 2919, 113 1215, 490 1097, 106 881, 273 771 | **8,120** | 631 |

The base also exceeds the 5 s bound (8.1 s), so the bound was too tight for this machine state and is not a property of
the change alone. But the two series are not in the same range by the numbers: `w163o` is higher in 8 of the 10
pressured requests by position (sign test p = 0.055), the totals differ by 5.9 s, and the per-request spread inside a
series (0.1 to 2.9 s) is wide enough that the difference of the means (1.40 against 0.81 s) has t about 1.6 (p about
0.13). **Reading:** not separable from the spread with 10 requests per arm, and the direction is against the change. The
series ran in one order (change first), so a drift of the page cache or the machine over the half hour is not excluded.
Per the decision rule (same range within its own spread: merge; clearly worse: no merge) this is **neither**: no
verdict from this data; the decision is the coordinator's, a swapped-order pair would settle it (not queued, by the
pause).


### Results (goldens on the final build)

`scripts/gpu-status.sh --own --keep-going --base origin/main --build work/build/w163o` (gpu queue job, build relinked
from the final sources after the first start refused a stale mtime; the second start was stopped by the desktop guard
after 35 cases with none different, the third ran through): golden identical for prefill dflash, gpu, cpu plan, server and
prefill, the cases listed `identical` (including `long` and `thinking-anthropic`); status line "C8: golden identical ...;
needs: pressure series". The pressure series is the R8 above (not green, see its results).

### Results (runtime separation)

`scripts/ab.py --a w163r --b w163r --b-env LD_LIBRARY_PATH=<toolchain llvm lib> --workload short,set --max-rounds 4`
(same binary, only the OpenMP runtime differs: system libgomp against the toolchain libomp; 4 rounds, report
`work/ab/163-runtime.md`, kept out of the repo; the first start of this run failed at the argument check because
ab.py wants an expectation for every metric, no round ran). B against A: decode@2000 -0.10 %, decode@2000:write
+0.16 %, decode-code +0.03 %, decode-text +0.00 %; prefill@2000 -0.23 %, prefill@2000:write +0.05 %; all inside the
planned -0.3 to +0.5 % (decode) and -0.3 to +0.3 % (prefill); output SHAs and expert-cache counts equal. **The runtime
alone moves nothing**, so the gains of the all-pragmas run above come from the pragmas, not from the runtime. (`hw_report`
and `quiet_check` were not run around this ab.)

## R8 follow-ups (plan, written before the runs; supersedes "next session" wording)

**Open.** `w163o` (all pragmas on) showed 14.0 s of evicted time in 10 pressured requests, `w163r` 8.1 s, `w163o` higher in
8 of 10 positions, not separable from the spread, in one order only (change first). Stability has priority.

**Runs** (each one exclusive queue job, `scripts/r8_pressure.py`, same series, caps and pressure as above, `hw_report` and
`quiet_check` before and after; the order below is the order of the runs):

1. `w163r` (tag `r8r2`), then 2. `w163o` (tag `r8o2`): the swapped-order pair.
3. `w163s` (tag `r8s`): `w163o` minus the pragma of `ExpertSelect::read_misses` (the only pragma of `mega_select.cpp`;
   one line removed, built from a throwaway worktree on branch `163-b1host-s`, not for merge). If `read_misses` is the
   suspect (parallel preads under page-cache pressure), the other 8 files stay on.

**Decisions (coordinator).** If 3 brings the eviction back to the `w163r` level (within its spread over the 10 pressured
requests) while it keeps most of the decode gain, the change merges without the `read_misses` pragma, with a note why:
parallel preads under page-cache pressure cost eviction time. If 1 and 2 show that the first difference was order or drift,
all of it merges. Pooled comparison: the four `w163r`/`w163o` series (first pair and swapped pair) give two totals per
arm; the per-request values are compared with the sign test by position and the totals with their spread.

**Decode gain of `w163s`** is judged by an `ab.py` run only if the R8 result says `w163s` is the candidate: `w163r`
against `w163s`, workloads short and set, 6 rounds (decode@2000:write and code are where `read_misses` mattered: +0.88 and
+0.64 % with all pragmas); not queued before that.

**Why this size.** 10 pressured requests per arm is the recipe size; the swapped pair and the third arm run in one session
so the machine state is as equal as possible. A fourth series only follows an unclear result. Deadline: all runs must end
before 20:00 (the night run of #165 takes the GPU); what does not fit waits.

**Still open from this file:** the first `just check` start failed at the configure step (`w163i` belongs to another
worktree); it is requeued with `installer_dir=work/build/w163i2`.

### Results (R8 follow-up 1: `w163r` again)

`scripts/r8_pressure.py --build w163r --tag r8r2` (exclusive job, 09:18 to 09:34, `/var/tmp/claude/163/r8r2.log`). Evicted
time per pressured request in ms, pair order 1 to 10: 1175 4377, 1485 1913, 528 2942, 554 1256, 690 1128; **total
16,048**, 0 failed requests, 0 slow decodes (lowest pressured decode 59.1 tok/s), longest episode 1.0 s. Test
conditions: `hw_report.py` (57.0 of 60.5 GiB RAM available, 1.94 GiB VRAM held by others), `quiet_check.py` before and
after both warned (1.94 GiB VRAM held by others; after: "the system is swapping (78 pages in 1 s)"), measured anyway.

| Series (pressured, total of 10) | Evicted time |
|---|---:|
| `w163r` first (07:57) | 8,120 ms |
| `w163o` (07:38) | 14,027 ms |
| `w163r` again (09:18) | **16,048 ms** |

**Reading.** The same base build gives 8.1 s and 16.0 s in two series, so the run-to-run range of the total (at least
2x) contains the 14.0 s of `w163o`; the base is not lower than the change in this second series. The first difference
(8.1 against 14.0 s) is therefore within the drift of the machine state and not a property of the change. This is the
coordinator's "order or drift" case for the swapped pair, shown by the second base series alone: the swapped
`w163o` series (`r8o2`) and the `w163s` arm (`r8s`, built as `w163s`) were **not run**, the pause of the #168 sweep
stopped chain4.sh before they were queued. Missing arms: `r8o2`, `r8s`. Without `r8o2` there is no `w163o` second
value, so the claim is "the base alone varies by 2x", not "`w163o` and `w163r` are equal". All bounds of the R8 plan
(5 s) were set from another series and are missed by every series of both builds (8.1 to 16.0 s); the bound needs a
source or a new value, it does not discriminate this change.

