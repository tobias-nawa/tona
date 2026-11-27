# Slow decode after a follow-up prefill (#83)

Date 2026-11-01. Platform: RX 9060 XT, Ryzen 5 9600X (6 cores, 12 threads), `OMP_NUM_THREADS=6`, build `w83` (main
7d5b00d plus `IE_DECODE_DIAG`), `--mtp --ctx 140000`. Flow: a 30K-token prompt, then an independent 130K-token prompt
(the 30K state is stashed), 200 tokens greedy after each. **Every rate in this file is a decode rate, except where it
says prefill.** "Load" means busy loops of a second process (python `while 1: pass`, 6 s, started when `/status` shows
the decode) running during the decode. Raw logs: `/var/tmp/claude/w83/` (not kept).

## Plan: #86 acceptance (ctx, bit-exact, goldens, pressure) of the register route

Class C8 (`docs/contribute/measurement.md`): R3, R4, R5 with `ctx`, R8. Build `work/build/w83` at bffee63 (no source change since the arms above).
1. Scope and expectation
   - R5 `ctx`: `ab.py --a merged --b w83 --workload ctx --primary decode`, contexts 140000 and 262144, 2-6 rounds, about 12-25 min.
     At 262144 the prefill of the first request is unstable (#88), decode is primary. Expectation: decode B against A -2..0 %
     (the 0.9-1.4 % of the arms above), expert counters of /status identical in both arms (all experts pinned in both: 256 per layer),
     B logs `host memory: GTT` and about 1.2 GB (140000) and about 5 GB (262144, KV 3.6 GB in the GTT) registered as userptr.
     Why this size: the pinned memory, the GTT use and the overflow depend on `--ctx` (the KV buffers), not on the prompt.
   - R4: `ab.py --a merged --b w83 --env IE_DETERMINISTIC=1 --workload mid,long --min-rounds 1 --max-rounds 1`; SHAs of 30000 and 130000
     equal to `bench/sha.toml` (08f0bd31e0c0277f, e9545d70f2400b87) in both arms. About 6 min. Tolerance: none (bits).
   - R3: `scripts/gpu-status.sh --base main --build work/build/w83` (no `--post`); the checks `select_checks.py` names, all five if it
     cannot decide; expectation: all identical, about 15-50 min.
   - R8: one series of 20 requests (30K and 130K alternating, 10 under page-cache pressure) with `IE_DIAG=1`, driver
     `/var/tmp/claude/w83/drive3.py`, 40-50 min. Expectation (register route: about 4 % of the pinned memory is userptr at ctx 140000):
     evicted time in the decode of the pressured requests below 5 s in total (series `s86f1`/`s86f2` without overflow: 5.3 s per 20
     pressured requests; default behavior 37 s per 20), slow decodes (under 55 tok/s) at most 3 of 10 pressured requests, no
     abort, no fault, longest eviction episode below 3 s. By the rule of three 0 faults in 10 bound the rate below 30 %, so a fault is
     only found by chance; it is recorded with the trace and the dump. A second series only if the first misses the expectation.
2. Active tooling: R3-R5 none beyond `ab.py`/`gpu-status.sh` (their own plan blocks); R4 `IE_DETERMINISTIC=1`; R8 `IE_DIAG=1`
   (allocation trace per PID, GPU dump, core limit unlimited), `IE_DECODE_DIAG=1`, the 10 ms eviction sampler of the driver.

## Plan: #86 repeat of the first comparison (the arms did not add up)

Arm D (code path alone, `2026-11-02-86-ab-codepath.md`): decode@2000 +0.17 %, decode@30000 +0.17 %, prefill 0.00 % - that is not the
-0.8 / -0.4 % that arms 1 and C leave for the code path (1: -0.91 / -1.23 %, C: -0.09 / -0.79 %). Either the first run was a drift
(all six rounds of B were lower, but the runs lie 40 minutes apart) or the memory model and the code interact.
1. Scope: the first comparison again, `scripts/ab.py --a merged --b w83 --workload mid,short --primary decode`, ctx 140000, 2-6 rounds.
2. Expectation: if the first run was a drift, B against A within -0.3..+0.3 % (arms C and D); if systematic, -1.5..-0.5 % as in run 1.
   The expectation range of the tool is the systematic case; a result near zero is then reported as outside it, and that is the answer.
3. Tooling: none beyond ab.py. Why this size: same workloads as run 1.

## Plan: #86 arm D, the code path alone (main against this build in the old memory model)

Arm C (`2026-11-02-86-ab-memmodel.md`) gave the memory model: GTT with the registered overflow against userptr, same build, decode@2000
-0.09 %, decode@30000 -0.79 %, prefill@30000 -0.18 %. Together with the -0.91 % and -1.23 % against main this leaves about -0.8 % (2K) and -0.4 %
(30K) for the code path. `head_row` is not in the decode loop (it runs only with `IE_MTP_HEAD=iq4` and at build time), so pread cannot be it.
1. Scope: `scripts/ab.py --a merged --b w83 --b-env HSA_USERPTR_FOR_PAGED_MEM=1 --workload mid,short --primary decode`; ctx 140000; 2-6 rounds.
2. Expectation: decode@2000 and decode@30000 B against A -1.2..-0.2 % (the code path share), prefill within -1..+1 %. If the result is
   0 +- 0.3 %, the code path is not the source and the sum of arms is noise.
3. Tooling: only the variable of arm B. Why this size: the same workloads as before.

## Plan: #86 where does the 1 % decode loss come from (arm C)

Question: the register route is -0.9 % (decode@2000) and -1.2 % (decode@30000) against main. Memory model (GTT against userptr)
or code path (mapping release, `head_row` with pread, the probe, the restart)?
1. Scope: `scripts/ab.py --a w83 --a-env HSA_USERPTR_FOR_PAGED_MEM=1 --b w83 --workload mid,short --primary decode`;
   the same build in both arms, so only the memory model differs (A: userptr as before, with the same code path; B: GTT with
   the 1.2 GB overflow registered). ctx 140000. 2-6 rounds, about 11 min at most.
2. Expectation: if the memory model costs the 1 %, B against A decode -1.5..-0.5 % (as against main); if the code path does, B against A
   -0.5..+0.5 %. Source: `2026-11-02-86-ab-prefill.md` of this branch. Pinned GB equal to the start log.
3. Tooling: only the variable of arm A; no test switch.
Why this size: the loss showed at 2K and 30K in `short` and `mid`; the same workloads show whether it goes.

## Plan: #86 prefill question with `ab.py` (after the rebase on main e4c1010)

Question: does the register route (default of the build) bring the prefill back to the speed of main, or is GTT itself slow?
1. Scope: `scripts/ab.py --a merged --b w83 --workload mid,short --primary prefill`; A = `work/build/merged` (main, built
   2026-11-02 01:17, engine code identical to e4c1010 except the IE_DIAG path), B = `work/build/w83` (this branch, 
   `IE_PIN_OVERFLOW=register`, `HSA_USERPTR_FOR_PAGED_MEM=0` by restart). `ctx` 140000. If the prefill falls by about 20 %
   again: arm C = B with `HSA_USERPTR_FOR_PAGED_MEM=1` (own build in the old memory model), which separates the code path
   from the memory model. Then decode with workload `ctx` (pinned GB and expert counters from `/status`; decode as primary at
   262144, the prefill of the first request is unstable there, #88).
2. Expectation: prefill@30000 and prefill@2000 within -1..+1 % (all 256 experts per layer stay pinned; earlier GTT runs without
   a cap: 16.8-16.9 s against 16.8 s default); decode within -2..+2 %. At ctx 140000 about 0.3 GB are registered as userptr.
   Source: `s86u1`-`s86u3`, `s86f1`, `s86f2` against `s86d1`, `s86d2` in this file.
3. Tooling: `IE_DECODE_DIAG` not set, no test switch; ab.py reads the variables itself. Why this size: the cap loss showed at a 30K
   prefill (-20 %) and at 130K (-27 %), so `mid` shows it; `short` adds the decode and the expert counters at 2K.

## Plan addendum: #86 overflow route (written before the runs, after the first cap measurement)

First result of (b), cap mode (`IE_PIN_OVERFLOW=cap`, binary of 3bfa1d8), ctx 140000, no pressure, 2 rounds alternating, fresh
server each: against the old behavior (`HSA_USERPTR_FOR_PAGED_MEM=1`) the decode at 30K fell from 74.2, 74.7 to 67.9, 67.5 tok/s
(-9 %), the prefill from 1806, 1799 to 1435, 1435 tok/s (-20 %) and at 130K from 1713, 1707 to 1256, 1254 tok/s (-27 %); decode at
130K 62.8, 66.2 against 63.1, 62.0 (inside the spread). The cap pins 24.9 of 25.2 GB there, yet the loss is far above 2 %: a few
unpinned experts per layer are read from the page cache. The cap route is out. Source of the cap:
`pin budget 28.4 GB -> 24.9 GB by the GTT (6.5 of 32.5 GB in use)`.

Route kept: all experts stay pinned; the part that does not fit the GTT is anonymous memory registered with `hipHostRegister`
(userptr, `gpu/gtt.cpp`, `IE_PIN_OVERFLOW=register`, default). libhsakmt reads `HSA_USERPTR_FOR_PAGED_MEM` only once at start
(`fmm.c:3846-3850` in ROCm/rocm-systems develop), so a per-allocation choice is only possible this way; `hipHostRegister` runs
only while loading.

Open question (from the review, to answer first after the restart): with `cap` the prefill falls 20-27 % although only 0.3 GB
fewer experts are pinned (24.9 instead of 25.2 GB). Suspicion: GTT memory itself is slower (CPU reads uncached or
write-combined, other DMA paths). Evidence so far against it: the series `s86u1`-`s86u3`, `s86f1`, `s86f2` ran before the cap
existed with all 256 experts per layer in GTT (allocation tally: 29.05 GB GTT) and showed no slowdown without pressure: prefill
16.8-16.9 s (30K) and 76.8-77.0 s (130K) like the default (16.8 and 76.8 s), decode 30K 72-74 and 130K 64.8-65.5 tok/s like the
default (72.3 and 64.8); the probe read 28 GB GTT at the same rate as userptr (0.73-0.75 s per 28 GB pass, 20 GB: 0.53 s against
0.52 s). So the loss most probably comes from the unpinned experts (read from the page cache, which `drop_range` empties for
pinned ones only) and not from GTT. First run after the restart: old (`HSA_USERPTR_FOR_PAGED_MEM=1`) against GTT without a cap
(a context at which everything fits, e.g. 65536) against `cap` at ctx 140000, prefill and decode, hit rate and pinned GB from the log.

1. Scope (class C8 of `docs/contribute/measurement.md` of the policy work: short runs for the rate, one pressure series for the acceptance)
   - Rate: ctx 140000 and ctx 262144, prompt about 2K tokens, 400 decode tokens, fresh server per run; arms old
     (`HSA_USERPTR_FOR_PAGED_MEM=1`), cap, register; 2 rounds, a third only if the two disagree. 12 starts, about 10 min.
     Why this size: the pinned memory, the cap and therefore the hit rate depend on `--ctx` (the KV buffers), not on the length of
     the prompt; the decode rate with a short prompt shows the share of unpinned experts. The 130K runs above stay as the first
     cap measurement.
   - Acceptance: one pressure series, 20 requests (10 under page-cache pressure, 30K and 130K), as `s86f`, default build.
2. Expectation
   - Pinned experts: old 256 per layer (25.3 GB at 140000); cap 252 at 140000 and 217 at 262144; register 256 in both, with
     about 0.3 GB (140000) and 4 GB (262144) registered as userptr.
   - Hit rate (log): register equal to old within 0.5 points; cap lower, more at 262144.
   - Decode (tok/s, 2K prompt): register within 2 % of old (spread of the old arm in this file: 74.2-74.7 at 30K); cap 5-10 % lower at
     140000 and more at 262144.
   - Under pressure the registered overflow is exposed again: evicted time in the decode between cap-free (`s86f`: 5.3 s per 40
     requests) and the default behavior (37 s per 40); slow decodes (under 55 tok/s) between 2 and 20 of 40 requests,
     proportional to the exposed share (about 1 % at 140000, about 14 % at 262144). Source: probe with a kept 3 GB mapping
     (0.9-3.7 s per minute) against 20 GB userptr (31-66 s per minute).
3. Active tooling
   - `IE_MEGA_INFO=1`, `IE_DECODE_DIAG=1` in the rate runs; the pressure series with `IE_ALLOC_TRACE`, GPU dump pattern, core limit
     unlimited, the 10 ms eviction sampler. No profiler, no test switch.

## Plan: #86 step 2 (variant A as the default), written before the runs below

Written after the first start-log runs (a); every run after this plan follows it. Build: `work/build/w83` at commit 3bfa1d8
(variable by restart, mappings removed after the upload, pin budget capped to the GTT). GPU lock for every run, port 8097,
`systemd-run --scope -p MemoryMax=54G -p MemorySwapMax=0`.

1. Scope
   - (a) Start logs, one start per cell (the number is deterministic): ctx 262144 and ctx 540672 with the #84 setup
     (`IE_SEQS=2`, `IE_TEST_STASH_RESERVE_GB=0`), new against `HSA_USERPTR_FOR_PAGED_MEM=1` (the old behavior). 4 starts, 2 min.
     The first three ran before this plan (the 540672 start of the new build hit a foreign server on port 8097 and is repeated).
   - (b) Only if the cap pins less: decode and prefill (tok/s), 30K and 130K prompt, 200 tokens, fresh server per round,
     3 rounds alternating old/new, no pressure. 6 starts, about 60 min.
   - (c) Checks: unit tests; `IE_DETERMINISTIC=1` SHA of 30K and 130K, 200 tokens; `scripts/gpu-status.sh --base main --build work/build/w83`
     (five goldens, about 50 min); `just check`.
   - (d) Acceptance series: 20 alternating requests (10 with page-cache pressure, 30K and 130K), as `s86f`, with
     `IE_ALLOC_TRACE` and the GPU dump, no test switch. About 60 min.
2. Expectation
   - (a) New: log `host memory: GTT`; the pinned experts are capped by the GTT (about 22-25 GB of 25.3 GB at 262144, less at 540672).
     Old: `host memory: userptr`, all 256 experts per layer. Source: GTT total 32.5 GB (`mem_info_gtt_total`), the build
     puts the KV cache and the staging buffers there first.
   - (b) A smaller pin share costs decode: the series with `IE_PIN_GB=18` (28 % fewer pinned) ran 44-54 tok/s against 64 (`s8g`
     in this file). For about 15 % fewer pinned I expect decode 130K 3-12 % lower (57-63 tok/s), 30K 2-8 % lower, prefill
     within 3 %. More than 12 % in decode or more than 5 % in prefill is a finding.
   - (c) Bits identical (065922c2ff602991 and 439aad472c59ae6f), goldens identical, `just check` green.
   - (d) Slow decodes (under 55 tok/s) in the order of `s86f1` and `s86f2` (2 of 40), evicted time in the decode below 1 s,
     no abort. A fault is recorded with trace and dump.
3. Active tooling
   - Diagnostics in (a)-(c): `IE_MEGA_INFO=1` (start log), `IE_DETERMINISTIC=1` only in the bit-exact check. No profiler, no test switch.
   - (d): `IE_ALLOC_TRACE`, `HSA_COREDUMP_PATTERN`, `ulimit -c unlimited`, `IE_DECODE_DIAG=1`; the driver script samples
     `evicted_ms` every 10 ms, `/proc/vmstat` and the GPU clock.

## Question

Sometimes the decode of the 130K request ran at 33-44 tok/s instead of 63-65 (issue #83; 10-30: 39.7, 10-31: 44.0,
40.2, 35-67 spread in the #72 runs).

## What the old logs say

In the logs of the slow runs the tokens per pass are the normal ones, only the pass is longer:

| Log | Decode | Tokens per pass | ms per pass |
|---|---|---|---|
| ab-merged-1793483391 (slow) | 44.0 tok/s | 2.78 | 63 |
| ab-merged-1793481938 (normal) | 64.4 tok/s | 2.78 | 43 |
| bx-main (slow) | 39.7 tok/s | 2.60 | 65 |
| bx-w80-def (normal) | 60.2 tok/s | 2.60 | 43 |

So MTP acceptance (hypothesis 1) is not the cause: same text, same tokens per pass, 45 % longer passes.

## Reproduction attempts without load

`IE_DECODE_DIAG=1` prints the counter differences of the decode (hits, misses, CPU experts, CPU time, yields, ...);
`drive.py` samples GPU clock, power, busy and CPU load every 100 ms and the PSI and `/proc/vmstat` counters between the
start of the decode and its end. Only this process and worker #84 (under the GPU lock) were running.

| Run | Decodes at 130K | Slow (< 55) | Decode min / mean / max (tok/s) | ms per pass |
|---|---|---|---|---|
| 20 x same server (30K, 130K, prompt salted per round) | 20 | 0 | 62.0 / 66.5 / 69.3 | 45.1-47.5 |
| 20 x same server (same salt for both prompts) | 20 | 0 | 60.9 / 64.9 / 68.1 | 45.7-48.7 |
| 10 x fresh server | 10 | 0 | 63.5 / 64.3 / 64.4 | 46.4-47.1 |

The same numbers at 30K: 0 slow of 50. Everything the ticket lists is flat: cache hits 62-73K and misses 8.2-10K per
200 tokens, CPU experts 73 per pass, wait for the CPU 1.9-2.2 s, yields 0, GPU clock 2950-3270 MHz (2990 average at 130K
after a fresh start, no influence on the pass time), GPU busy 98 %, CPU load 45 %. Hypotheses 2 (expert slots after the
loan), 3 (background SSD/stash work: there is no thread besides the prefill thread) and 4 (clock) have no support. Fresh
servers are bit-identical run to run (SHA `065922c2ff602991`, `439aad472c59ae6f`).

## Reproduction with load, deterministic mode (`IE_DETERMINISTIC=1`)

**Mode note:** the load script started the server with `IE_DETERMINISTIC=1` (passes of at most 4 tokens, fixed
plan) by mistake. The tables of this section and the next two (pinning, variants) are therefore deterministic-mode
numbers; the comparison between the variants holds, the absolute rates and the factor of the slowdown differ from the
normal mode (next section, where the old spinning policy gives 38-44 instead of 26 tok/s). 130K conversation, 200
tokens, repeated identical requests (prefix reused). Decode in tok/s, single runs, in order.

Default OpenMP wait policy (threads spin), loads unpinned on all CPUs (deterministic mode):

| N busy loops | Decode (tok/s) |
|---|---|
| 0 | 65.3, 67.1, 67.4, 66.9, 66.4, 67.9 |
| 1 | 67.3, 66.7 |
| 2 | 65.9, 65.8 |
| 4 | 65.1, 66.2 |
| 6 (nice 19) | 64.9 |
| 6 | 64.9 |
| 7 | 27.5 |
| 8 | 25.9, 25.8, 26.4, 25.9, 26.2 |
| 12 | 27.6, 32.1 |
| 12 (nice 19) | 25.2 |

With the load the counters look like in the normal runs (same hits, misses, 73 CPU experts per pass, yields 0, same
tokens per pass), **only the CPU time of the experts rises from 0.43 s to 3.7-5.3 s per 200 tokens** (about 10x). The
signature of a slow decode is that number.

Pinning (`OMP_PLACES={0},...,{5}`, `OMP_PROC_BIND=close`, loads with `taskset`), N=6, default policy (deterministic mode):

| Setup | Decode (tok/s) |
|---|---|
| pinned, load on the SMT siblings 6-11 | 65.7, 65.7 |
| pinned, load on the same hardware threads 0-5 | 24.7, 24.6 |
| not pinned, load on 6-11 | 64.6, 65.1 |
| not pinned, load on 0-5 | 65.6, 65.5 |

So it is not the SMT sibling (a busy sibling costs nothing measurable here) but the pre-emption of a team thread by a
load on its own hardware thread; unpinned, the scheduler moves the threads away until all 12 threads are busy.

OpenMP variants, 130K, N busy loops (deterministic mode):

| Variant | N=0 | N=4 | N=6 | N=8 | N=12 |
|---|---|---|---|---|---|
| default (spin) | 65.3-67.6 | 65.1-66.2 | 64.9 | 25.8-26.4 | 27.6, 32.1 |
| `OMP_WAIT_POLICY=passive` | 65.3-67.2 | 65.8 | 65.8 | 61.5, 62.1, 63.0 | 33.2, 30.9 |
| `GOMP_SPINCOUNT=2000` | 65.5-67.3 | | | 57.6 | 30.9 |
| `OMP_NUM_THREADS=5` | 65.3-67.0 | | | 30.7, 29.5 | |
| `IE_PLAN=gpu` (no CPU experts) | 60.1-62.9 | | | 62.8 | 62.4 |

## Reproduction with load, normal mode (`IE_DETERMINISTIC` unset)

Same flow, decode in tok/s, single runs in order. Before = old behavior: spinning wait policy, fixed plan
(`OMP_WAIT_POLICY=active`, `IE_CPU_ADAPT=0`); stage 1 = passive, fixed plan (`IE_CPU_ADAPT=0`); `IE_PLAN=gpu` = no CPU
experts.

| Variant | N=0 | N=4 | N=8 | N=12 |
|---|---|---|---|---|
| before (spin, fixed plan) | 67.4, 67.4, 67.3 | 66.8 | 38.3, 37.5 | 41.6, 40.5, 43.7 |
| stage 1 (passive, fixed plan) | 66.9, 67.3, 67.1 | 66.2 | 62.4, 61.8 | 33.0, 32.9, 32.7 |
| `IE_PLAN=gpu` | 62.2, 62.3, 62.2 | | 62.3 | 62.0 |

The old policy in the normal mode drops to 38-44 tok/s with 8 to 12 busy threads: the range of the slow decodes in the
ticket (33-44).

The experts of the CPU run in an OpenMP region of three loops with a barrier each, about 73 calls per pass. Waiting
threads that spin burn the time slice that the pre-empted thread needs. `passive` makes them sleep.

## Bits: CPU expert and GPU direct read

`ie-expert-check` (every expert format, CPU `matvec_q8` against GPU `matvec_q8`, rounding to int8, silu): "bit-exact",
0 differences in 1,048,576 silu values. End to end with `IE_DETERMINISTIC=1` and a fixed plan: `IE_PLAN=gpu` and
`IE_PLAN=cpu` both give SHA `065922c2ff602991` (30K) and `439aad472c59ae6f` (130K), the same as the default plan.
`IE_PLAN=gpu` is the lower bound without CPU experts: decode 59.6 (130K, deterministic; default plan 63.5), CPU load 9 %
(normal mode: 62.2 against 67.1).

## Other: memory reclaim during a decode

One of about 100 decodes without load was slower: 57.2 tok/s (52.3 ms per pass instead of 46-49) with GPU busy 93.7 % and
a clock average of 2595 MHz. It is the only one in which `kswapd` scanned during the decode (`pgscan_kswapd` 40,901
pages, `kcompactd` active, memory PSI 351 us), right after a server start whose model load filled the page cache. It
did not recur in the later fresh starts. A 10 % effect, not the 33 % of the ticket; noted, not pursued.

## Fix 1: restart with `OMP_WAIT_POLICY=passive`

`ie-server` and `ie-gpu-gen` start themselves once with the variable set if neither `OMP_WAIT_POLICY` nor
`GOMP_SPINCOUNT` is set (`backend/omp_wait.h`; libgomp reads the variable when it loads, so `main()` is too late).
`/proc/<pid>/environ` after the start: `OMP_WAIT_POLICY=passive` with nothing set; the user's `active` and
`GOMP_SPINCOUNT=1000` are kept.

Without load, 3 rounds alternating `merged` (main 7d5b00d, spin) and `w83` (passive):

| Measure | merged (3 rounds) | w83 (3 rounds) |
|---|---|---|
| bench-set code, decode (tok/s) | 86.9, 88.2, 88.0 | 87.3, 87.3, 87.4 |
| bench-set text, decode (tok/s) | 63.3, 63.9, 63.8 | 63.4, 63.5, 63.5 |
| 30K, decode (tok/s) | 74.3, 75.4, 75.4 | 74.8, 75.0, 74.8 |
| 130K, decode (tok/s) | 63.6, 64.3, 64.3 | 64.2, 64.1, 64.0 |
| 30K, prefill (tok/s) | 1806, 1807, 1806 | 1806, 1806, 1806 |
| 130K, prefill (tok/s) | 1713, 1714, 1713 | 1713, 1713, 1713 |
| CPU load during the decode | 44-45 % | 14-17 % |

Under load (130K, decode): deterministic mode N=8: `w83` 62.3, 62.2, `merged` 26.2, 25.9; N=12: `w83` 29.9. Normal mode (table above): N=8 62.4, 61.8 against 38.3, 37.5; N=12 33 against 41 (not fixed by this step alone).

Checks: `IE_DETERMINISTIC=1` 30K `065922c2ff602991`, 130K `439aad472c59ae6f`; unit tests 32 of 32; golden decode
(csv, rust, explain), golden prefill with DFlash (long) and golden server (20 of 20) all identical.

## Fix 2: the planner follows a slow CPU (`gpu/cpu_cost.h`)

With `passive` the decode survives 8 busy threads, but at 12 (all hardware threads) it still fell to 33 tok/s: the
planner kept ~73 experts per pass on the CPU, 10x slower than planned, and the GPU waits for them in the kernel.
`IE_PLAN=gpu` reads the pinned misses directly and runs at 62 tok/s whatever the load. The cost of a CPU expert for
the planner is now a moving average (weight 1/16 per call) of the measured time per expert relative to the base of 60
us, with a dead zone up to 2 x the base (measured in the normal state: 76-80 us), so without load the plan is the old
one. Above it the estimate rises, the misses go to the GPU; when no expert is left on the CPU the estimate decays
(1/160 per layer), which probes the CPU about every 17 layers (5 % of the layers under load). Only pinned experts count
as samples. The plan stays fixed with `IE_DETERMINISTIC=1` and `IE_PLAN`; `IE_CPU_ADAPT=0` switches it off, `=2` forces
it (test).

Normal mode, 130K, 200 tokens, decode in tok/s, single runs in order (rows before and stage 1 from above):

| Variant | N=0 | N=4 | N=8 | N=12 |
|---|---|---|---|---|
| before (spin, fixed plan) | 67.4, 67.4, 67.3 | 66.8 | 38.3, 37.5 | 41.6, 40.5, 43.7 |
| stage 1 (passive, fixed plan) | 66.9, 67.3, 67.1 | 66.2 | 62.4, 61.8 | 33.0, 32.9, 32.7 |
| stage 2 (passive, adaptive) | 67.1, 67.2, 66.9 | 66.2 | 62.7, 63.0 | 58.3, 57.8, 58.6 |
| stage 2 after the decay fix | 67.2, 67.2, 67.0 | 66.3 | 62.7, 63.3 | 58.5, 58.5, 58.7 |
| `IE_PLAN=gpu` | 62.2, 62.3, 62.2 | | 62.3 | 62.0 |

N=12 reaches 58 against the 62 of the pure GPU plan: the rest is the probing of the CPU and the first layers after the
load starts.

**Course over a decode with load that goes on and off** (`timeline.py`, 130K, 1800 tokens, 12 busy loops from second 6
to second 14 of the decode, normal mode; `2026-11-01-slow-decode-timeline.png`, left fixed plan, right stage 2). Stream
chunks per second (about tokens): fixed plan before / under load / after 61.6 / 27.0 / 55.8, stage 2 60.7 / 48.4 / 50.4
(the "after" window is the last 0.7 s of the stream, noise). CPU experts per pass: ~90 without load, ~2 under load,
back to ~80 within one pass after the load ends; the estimate ratio rises from 1.4 to 8-20 within one pass and falls
back to 1.4 at once; no oscillation. The 200-token table above is the one to compare without load (stage 2 equals the
others within the spread).

**Bits under a changing plan:** `IE_DETERMINISTIC=1 IE_CPU_ADAPT=2`, load N=12 in some of the requests: CPU experts per
pass 71-75 without load, 5 and 13 under load (gpu_direct 2,600 -> 7,000-7,500 per 200 tokens), SHA of the text
`a90744c565420c39` in all 8 requests, with and without load.

**Without load, 3 rounds alternating `merged` (main + stage 1) and `w83` (stage 2), normal mode** (decode unless
named):

| Measure | merged | w83 |
|---|---|---|
| bench-set code (tok/s) | 86.0, 87.4, 87.6 | 87.5, 87.3, 87.2 |
| bench-set text (tok/s) | 62.8, 63.4, 63.5 | 63.4, 63.4, 63.4 |
| 30K (tok/s) | 73.0, 74.7, 74.7 | 74.5, 75.0, 74.7 |
| 130K (tok/s) | 63.3, 64.2, 64.1 | 64.0, 64.0, 64.0 |
| 30K prefill (tok/s) | 1806, 1806, 1806 | 1805, 1806, 1806 |
| 130K prefill (tok/s) | 1712, 1712, 1712 | 1712, 1712, 1712 |

Unit test `cpu_cost` (dead zone, rise, fall back, fixed plan, closed loop over 500 + 1000 + 500 layers with 10x load
in the middle: plan unchanged without load, 52 probes under load, back to the normal plan after 34 layers, no flapping).

**Decay fix (review of stage 2).** A layer with an unpinned miss that must run on the CPU never emptied the CPU list, so
the decay did not run and the estimate stayed high after a load burst (every pinned miss on the GPU, 7 % slower without
load, as with `IE_PLAN=gpu`). The decay now runs whenever no pinned expert is on the CPU but some are on the GPU; a
layer with pinned and unpinned experts on the CPU takes no sample and does not decay. Unit test with an unpinned miss
in every layer after a burst: the plan returns after 153 layers (the old code: not in 1,500). Final checks on the
committed state (ninja -n empty): pre-commit and tidy green, unit 33 of 33, `IE_DETERMINISTIC=1` 30K
`065922c2ff602991` and 130K `439aad472c59ae6f`, golden decode / prefill with DFlash / server (20) all identical, load
table above (row "after the decay fix").

**Caveat for runs on this machine:** a `cat` loop of my own pressure test (page-cache fill, see #85) kept running for
45 minutes after a crash and made frozen servers, golden aborts ("pass deadline -2060 ms") and decodes of 22-46 tok/s
without any load; those runs are not in the tables. Check `ps` for stray readers before measuring.

## Stage 3: page migration evicts the KFD queues (memory pressure)

Question: what makes some decodes slow (33-44 tok/s) and some requests abort (barrier time limit, #85) when the page
cache and compaction are busy? Tools: `evicted_ms` of every KFD process (`/sys/class/kfd/kfd/proc/*/stats_*/evicted_ms`,
10 ms sampler, an episode is a run of growth), `/proc/vmstat`, PSI, GPU clock and busy. Driver script (not in the repo):
`/var/tmp/claude/w83/drive3.py`; raw data `s8d`..`s8g` there. Pressure = two big files that the server does not use are
read in a loop into the page cache (`cat > /dev/null`, own `systemd-run --user --scope` with `RuntimeMaxSec`). Normal mode,
server MTP, `--ctx 140000`, 30K then 130K prompt, 200 tokens, decode rates only. Slow = decode under 55 tok/s.

Caveat: a first pressure run (not counted) crashed with an HTTP 500 and left its reader and server running; the series
`session9`, `9b`, `8b` and part of `8c` were polluted and are not used. The numbers below come from `s8d` to `s8g`, each
started after a check that no stray reader runs.

### Server, 20 alternating rounds (`s8d`, default allocation), decode of one request

| | Decodes | Slow | Aborts (500) |
|---|---|---|---|
| without pressure | 20 | 1 (130K, 40.0 tok/s, evicted 3896 ms, first request after a fresh start) | 0 |
| with pressure | 19 | 6 (41.6 to 54.6 tok/s) | 1 (round 5, 130K: pass deadline -24.6 s, 116.9 s evicted in the request) |

Four of the six slow decodes carry 1.9 to 3.1 s of evicted time inside the decode, with 39K to 78K migrated pages
(`pgmigrate_success`) and GPU busy 69 to 79 % (healthy: 97 to 98 %). Rounds with heavy `kswapd` scanning (3 to 7 million
pages) but no migration (`pgmigrate_success` 0) show 0 ms evicted: migration, not scanning, triggers the eviction.
Longest single eviction episode inside a decode window: 3.0 s without pressure (the slow case above), 1.4 s with
pressure; over a whole request (prefill included) up to 77.7 s, and the abort happens in such a phase.

### Variants, same flow, 6 pressure rounds each (12 requests)

| Variant | Slow with pressure | Aborts | Evicted in decode (with pressure) | Decode without pressure |
|---|---|---|---|---|
| default (`s8e`) | 2 of 12 (50.1, 50.7) | 0 | 0 to 1470 ms | 59-75 (30K), 63-65 (130K) |
| `HSA_USERPTR_FOR_PAGED_MEM=0` (`s8f`) | 1 of 11 (42.7) | 1 (round 9, 130K, 117.3 s evicted) | 0 to 2171 ms | 60-74 (30K), 63-65 (130K) |
| same plus `IE_PIN_GB=18` (`s8g`) | all 39-44 (base 44-54, first request 31) | 0 | 0 to 159 ms | 44-54 |

The default and the env variable are not distinguishable on the server. Two series of the default differ (6 of 19
slow in `s8d`, 2 of 12 in `s8e`), so the rate depends on the memory state before the run (cache full, free memory 2 GB).
`IE_PIN_GB=18` leaves fewer experts in host memory and cuts the eviction by a factor of 10 or more, but costs rate.

### Probe: which host memory route gets evicted (`/var/tmp/claude/w83/probe/`)

A small HIP program reads a host buffer from the GPU in a loop for 60 s while the pressure reader runs (not the
megakernel, so synchronizing is allowed). Rows are 3 repetitions per size unless noted.

| Host buffer | Size | Evicted in 60 s | Worst single read |
|---|---|---|---|
| `hipHostMalloc` default (KFD userptr) | 20 GB, mapped flags | 31.6, 35.4, 65.7 s | 3.6-4.3 s (normal 0.55 s) |
| same, default flags | 12 GB | 4.4, 14.8, 35.0 s | 1.3-3.1 s |
| `hipHostMalloc` with `HSA_USERPTR_FOR_PAGED_MEM=0` (KFD GTT) | 12, 20, 28 GB (9 runs) | 0 ms in all | normal |
| `hipMemCreate` host location, pinned, mapped with `hipMemMap` | 12, 20, 28 GB (9 runs) | 44-221 ms (startup only) | normal |
| `hipMemCreate` as above plus a 3 GB mapped file copied to VRAM with `hipMemcpy`, mapping kept | 20 GB (6 runs) | 0.9-3.7 s | up to 0.68 s |
| same, mapping removed with `munmap` after the copy | 20 GB (3 runs) | 0.24-0.31 s | normal |
| same, `madvise(MADV_DONTNEED)` instead of `munmap` | 20 GB (3 runs) | 0.7-2.3 s | up to 0.68 s |

The memory type was checked with an `LD_PRELOAD` tally of `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` (`/var/tmp/claude/w83/probe/kfdshim.c`, not in the repo): a 1 GB
`hipHostMalloc` is a USERPTR allocation by default and a GTT allocation with the variable. The server with the variable
allocates 14.3 GB VRAM, 29.05 GB GTT and no USERPTR; its `smaps` shows `Pss_Shmem` 0 (the default server: 30.4 GB
`rw-s /dev/zero (deleted)`, `Locked` 0). `hipMemCreate` with a host location also reads at full rate and its handle can be
mapped by the CPU (`mmap` of the exported fd).

Findings:
1. The default `hipHostMalloc` memory is userptr: page migration invalidates it and the driver evicts all queues of
   the process. 4 to 66 s per minute in the probe.
2. KFD GTT memory (variable, or `hipMemCreate` with a host location) is not moved: 0 to 0.2 s in 18 probe runs of 12
   to 28 GB.
3. A mapped file that was the source of a `hipMemcpy` keeps the process exposed while the mapping exists (1 to 4 s per
   minute); `munmap` removes it, `MADV_DONTNEED` does not. The server keeps two GGUF mappings of about 2.7 GB resident
   each. Not yet proven that this is the remainder in the server with the variable (`s8f` still shows up to 2.2 s); the
   probe shows the mechanism.
4. `/dev/udmabuf` (64 MB per buffer, the size limit is root-owned) cannot be imported with `hipImportExternalMemory`
   (OpaqueFd): "out of memory" at the first chunk. Not usable through HIP.
5. libdrm headers are not installed (`/usr/include/libdrm/amdgpu.h` missing); a GEM BO in GTT ends in the same kernel
   object type as the variable or `hipMemCreate`, so it was not built.

## #86 step 1: variable and file mappings (`IE_TEST_UNMAP_GGUF`)

Test switch `IE_TEST_UNMAP_GGUF=1` (commits dbdb791, 8b9e73e): after the upload, the file mappings that nobody reads again are
replaced by inaccessible anonymous mappings (`MappedFile::release`): the dense tensors, the whole MTP file and the output
head (its rows are read with `pread` in `Mega::head_row`), 8.25 GB. Kept: the experts (the CPU reads the unpinned ones in
place: `gpu/mega_prefetch.cpp:48-50`, `:82`) and the two embeddings (rows read on the CPU: `gpu/mega_mtp.cpp:135`, `:284`,
`gpu/prefill_entry.cpp:243`, `gpu/prefill_mtp.cpp:74`, `cpu/dflash.cpp:265`). Always together with the external
`HSA_USERPTR_FOR_PAGED_MEM=0`; allocation tally: 0 USERPTR, 29.04 GB GTT (`/var/tmp/claude/w83/probe/kfdshim.c`).

Same flow as `s8d`: 20 alternating requests (10 without, 10 with pressure; 30K and 130K, 200 tokens, decode rates),
fresh server per series, one build (`work/build/w83`). Slow = decode under 55 tok/s.

| Variant (series) | Pressure requests | Slow | Aborts / faults | Evicted in the decode, sum (max) | Evicted per request, sum | Longest episode in a request |
|---|---|---|---|---|---|---|
| default (`s86d1`, `s86d2`) | 40 | 20 | 0 | 37.4 s (3.8 s) | 1519 s | 47 s |
| variable only (`s86v1`) | 20 | 2 | 1 abort (barrier, 51.8 s evicted) | 3.0 s (0.64 s) | 691 s | 89 s |
| variable + switch, output still mapped (`s86u1`) | 20 | 3 | 0 | 2.1 s (0.70 s) | 201 s | 12.3 s |
| variable + switch with output (`s86u2`, `s86u3`) | 40 | 5 | 1 GPU memory fault | 8.3 s (0.58 s) | 252 s | 1.4 s |

Without pressure (10 requests per target and series, any variant) nothing is slow: decode 30K 72-74 tok/s median, 130K
64.8-65.5 median, prefill 16.8-16.9 s and 76.8-77.0 s (30K and 130K) in all variants.

Findings:
1. The variable alone removes most of the eviction inside the decode (37 s to 3 s per 20 requests) and the slow decodes
   (20 of 40 to 2 of 20), but not the long evictions during the prefill (89 s longest episode, one abort).
2. The mapping release removes those: longest episode 47-89 s to 1.4 s, per-request sum 691 s to 252 s. The output head
   was the last uploaded tensor with a live mapping (12.3 s to 1.4 s).
3. Not reached: evicted time in the decode under 200 ms; the maximum is 0.58 s and the sum is 8 s per 40 requests. A rest
   remains (the kept mappings of the experts and embeddings are not hipMemcpy sources, so the cause is open).
4. One request of `s86u3` (round 17, 130K) ended in `Memory access fault by GPU ... Page not present`, kernel
   `k_moe_sum` (grid 13,107,200), faulting address 0x7f743de25000, right after the stash of the previous conversation. This
   is the symptom of #72. It is 1 of 60 pressure requests with the GTT variant and 0 of 40 with the default in these
   series; too few to say whether the variable causes or only exposes it. The memory type of the faulting address is not
   known yet.
5. Without pressure, 3 fresh-server rounds alternating (`s86m*` against `s86w*`, `merged` was main plus stage 1 before
   276f601; #84 changes only the stash copy): decode 30K 71.8, 66.1, 71.6 against 71.0, 68.7, 72.7 tok/s; 130K 66.8, 64.2,
   65.8 against 63.5, 63.1, 58.8 tok/s (the last one is the only value below the spread of the other series, 60.6-67.9);
   prefill 1790-1801 against 1777-1799 and 1703-1709 against 1704-1705 tok/s.

### #86: the fault hunt (`s86f1`, `s86f2`)

Question: where does the GPU memory fault of `s86u3` come from (k_moe_sum, #72)? Without the GPU: all six arguments of `k_moe_sum`
(`MoeSumArgs`, `gpu/prefill_kernels.h:336`, call `gpu/prefill_moe_run.cpp:214`) are carved from the prefill arena
(`moe_buffers()`, `gpu/prefill_moe_run.cpp:50`) or are `PfBufs`: VRAM, no pinned host memory, no GGUF mapping. The CPU
core dump of `s86u3` (12.7 GB) has no mapping around the fault address 0x7f743de25000; the nearest one starts 1.9 MB above
it. The faults of #72 happened before the switch existed (3 reports, userptr default), so the switch is not the cause.

GPU core dump: without `HSA_COREDUMP_PATTERN` the dump always fails ("Failed to write segment data: Broken pipe" in a
deliberate-fault test, "Invalid argument" in `s86u3`): the runtime pipes it to `systemd-coredump`, which does not take it.
With the pattern and `ulimit -c unlimited` it works; `ulimit -c 1024` gives a 1.9 KB file; the test process with 20 GB pinned
host memory and 10 GB VRAM wrote 32 GB. My runs had set no pattern.

Two series like `s86u3` (variable plus switch with output) with `IE_ALLOC_TRACE`, the pattern and an unlimited core limit:

| Series | Pressure requests | Slow | Faults / aborts | Evicted in the decode, sum (max) | Longest episode in a request |
|---|---|---|---|---|---|
| `s86f1` + `s86f2` | 40 | 2 | 0 | 5.3 s (0.57 s) | 1.2 s |

Decode with pressure: 30K 59.2-70.4 (median 63.8), 130K 50.2-63.3 (59.8) tok/s; without pressure 30K 71.6-76.5 (74.0), 130K
60.2-68.2 (64.8) tok/s. Together with `s86u2` and `s86u3`: 1 fault in 100 pressure requests with the variable and the switch.
The third series (`s86f3`) was stopped after 46 s and is not counted. The hunt was ended there; the trace and the dump
will run in every development run (own ticket), so the next fault is mapped without extra series.

## #86 where the 1 % decode loss comes from: result of arms 1, C, D and the repeat

All `ab.py`, ctx 140000, workloads `mid` and `short`, 6 rounds each (files `2026-11-02-86-ab-*.md`); B vs A, mean:

| Run | A | B | decode@2000 | decode@30000 | prefill@30000 |
|---|---|---|---|---|---|
| 1 `ab-prefill` | main (`merged`) | this build, default (GTT + registered overflow) | -0.91 % | -1.23 % | -0.79 % |
| repeat `ab-repeat` | main | this build, default | -0.86 % | -1.35 % | -0.45 % |
| C `ab-memmodel` | this build, `HSA_USERPTR_FOR_PAGED_MEM=1` | this build, default | -0.09 % | -0.79 % | -0.18 % |
| D `ab-codepath` | main | this build, `HSA_USERPTR_FOR_PAGED_MEM=1` | +0.17 % | +0.17 % | 0.00 % |

The default build is 0.9-1.4 % slower in decode than main, reproduced twice (the expectation of the plan, -1.5..-0.5 %, held; the
prefill@2000 of the repeat, -1.48 %, is outside its -1..+1 %). The split is unclear: the arms C and D explain about -0.1/-0.8 % (memory model) and +0.2 % (code
path), less than the two direct runs show, so there is an interaction or a drift between the runs that I cannot separate with
these arms. The code path is not the source (D), and `head_row` is not in the decode loop (it runs only with `IE_MTP_HEAD=iq4`
and while building). The expert counters and SHAs are identical in all arms. The loss is inside the 2 % that a stability fix may
cost, and it was not pursued further: noted, then on to the ctx and pressure runs.

## #86 acceptance of the register route (plan above)

Build `work/build/w83` (HEAD 0b02aee, same engine code as bffee63), against `merged` (main e4c1010) where there is an arm.

| Check | Result | Expectation | Verdict |
|---|---|---|---|
| R5 `ctx` (`2026-11-02-86-ab-ctx.md`), 6 rounds | decode@2000 ctx140000 +0.15 % [-0.38, +0.69], ctx262144 -0.29 % [-1.32, +0.75]; prefill@2000 +0.00 % and -0.33 %; all open (undecided: the intervals contain 0); expert counters identical in both arms; B: GTT, 1.2 GB (ctx 140000, 27.4 GB pinned) and 4.7 GB (ctx 262144, 29.2 GB pinned) registered as userptr | decode -2..0 %, counters identical | decode@2000 ctx140000 and decode@2000:write above the range (+0.15 %, +0.90 %: B not slower); the 0.9-1.4 % loss of the earlier arms did not show in this run: inside the noise of two builds that differ in memory model, not reproduced |
| R4 bit-exact (`2026-11-02-86-ab-bits.md`), `IE_DETERMINISTIC=1`, 2 rounds | SHA 30000 `08f0bd31e0c0277f`, 130000 `e9545d70f2400b87`, both arms stable and equal, equal to `bench/sha.toml` | equal | as expected |
| R3 goldens (`2026-11-02-86-gpu-status.txt`) | all five identical: gpu (csv, rust, explain), cpu plan, prefill, prefill dflash, server (20 of 20); 631 s | identical | as expected. This run had no GPU lock (see below) |
| R0/R1 | pre-commit on the branch against main: all hooks pass; tidy ok (14 files); 36 unit tests pass in `work/build/w83` | green | as expected. `just check` itself cannot run in the worktree: `work/build/engine` is configured for the main checkout |
| R8 pressure series `s86acc` (`IE_DIAG=1`), 20 pressured and 20 unpressured requests (30K and 130K) | slow decodes (under 55 tok/s) 0 of 20 pressured; aborts 0; GPU faults 0; evicted time in the decode of the pressured requests 0.36 s in total (max 117 ms); evicted per request 127 s in total; longest episode 1.9 s; decode with pressure 30K 56.7-68.6 (median 65.8), 130K 56.3-61.7 (58.7); without pressure 30K 66.0-76.8 (74.2), 130K 60.8-65.6 (63.7) | evicted in the decode below 5 s, at most 3 of 10 slow, no abort, no fault, longest episode below 3 s | better than expected on every number |

Against the default behavior (`s86d1`, `s86d2`: 20 of 40 pressured requests slow, 37.4 s evicted in the decode, longest episode 47 s) and
the cap-free GTT series (`s86f1`, `s86f2`: 2 of 40, 5.3 s): the register route with 1.2 GB of userptr at ctx 140000 loses nothing visible
under pressure. Rule of three: 0 faults in 20 pressured requests bound the rate below 15 %, so a fault near 1 % (as `s86u3`, 1 of 100)
is not measured here; the always-on trace and dump (`IE_DIAG=1`) are on for this.

Not done: R7 (long-context run, `ab.py` workload `huge`, needle) which `select_checks.py` names because of `gpu/gtt_budget.h` (the name
matched `ctx`; the change does not touch the context handling), and the test-py run (no pytest in the toolchain Python).

Process note: `scripts/gpu-status.sh` took no GPU lock, and I started it from my script without an outer `flock`; it ran about 03:19-03:30 on
2026-11-02 (631 s of golden checks on `work/build/w83`), while the #90 worker profiled. None of my own measurements ran at that time
(my chain was sequential), so no number of mine is affected; the goldens are identity checks, not rates. From now on every GPU call of mine
goes through a script that takes the lock (`flock` around `gpu-status.sh`); `gpu-status.sh` and `golden*.sh` should take it themselves.
