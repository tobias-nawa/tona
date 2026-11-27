# Barrier aborts in the megakernel (#75, with #64)

Question: why does a pass end now and then with `gpu::Mega: barrier aborted (time limit)` (HTTP 500) on the unchanged path,
and how do we stop that without weakening the protection against a real hang?

Build: `work/build/w75` (branch `75-barrier`, from main c1b42e3), reference `work/build/merged` (c1b42e3). Server on port
8097, `--mtp`, two slots, under the GPU lock. Raw data (logs, sampler files, scripts): `/var/tmp/claude/w75/`, not in the
repository. Desktop (compositor, shell, terminal) on the same card throughout; MemoryMax 50 or 56 GB.

## Reproduction

Pattern of #76: two conversations with a 64K system prompt each (`/var/tmp/claude/w75/repro.py`), then short questions
(48 tokens) in quick succession, alternating, so that every request restores its conversation from RAM (1.2 GB).
Diagnostics: `IE_MEGA_PROF` + `IE_MEGA_STALL_MS` (every step over n ms with the arrival of the blocks), the cause the
aborting block records, `IE_MEGA_ABORT_TRACE`, and a sampler that reads `evicted_ms` of the KFD process every 10 ms
(`/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms`), the DRM engine time of every client, and `/proc/vmstat`.

| Run | Requests | Aborts | Remark |
|---|---:|---:|---|
| r1, r2 | 100 + 40 | 0 | |
| r3 | 60 | **1** | pass of 1.4 s, last router waited 1,350 ms for the CPU |
| r4, r5 (56 GB), r6, r7 | 150 + 150 + 150 + 100 | 0 | r7 with `IE_SEQS=1` and 90 s idle afterwards |
| together, without the fix | 750 | 1 | about one abort in 750 requests (about 11,000 passes) |

Plus one more abort of worker #77 (server, 130K prompt, host.wait 547 ms, then abort at another step).

## Finding: the driver evicts all queues of the process

Both kinds of stall seen in the logs are KFD evictions, not the kernel and not the CPU.

- **Stall (a)**: the whole grid stands still for 50 to 890 ms inside an ordinary step (dn.in, moe.sum, qsa.attn, ...). All 32
  blocks reach the next barrier within 40 us of each other afterwards: the waves were not running, they were not slow.
- **Stall (b)**: `host.wait` 0.35 to 1.35 s. The CPU thread stands in `prefetch_settle` (`hipEventSynchronize` on the copy
  stream) with 0 page faults (`getrusage`): the copy queue is evicted as well.

Evidence (r4, 150 requests; sampler against the stall lines):

| Stall | Duration | `evicted_ms` in the window of the stall |
|---|---:|---:|
| CPU router 9 | 447 ms | 462 ms |
| dn.recur | 179 ms | 242 ms |
| moe.sum | 281 ms | 311 ms |
| moe.down | 138 ms | 208 ms |
| qsa.in | 78 ms | 88 ms |
| gr.up | 57 ms | 85 ms |

Whole runs: r4 10.8 s evicted in 280 s (182 events of 20 ms or more), r5 11.6 s in 213 s, s1 (1000 requests) 38.3 s in 2,647
events, longest 889 ms. Server idle after the load: 0 ms in 85 s.

Evictions come with page migration: in 10 s windows without `pgmigrate_success` about 200 ms were evicted, in windows with
200K to 550K migrated pages 700 to 1,300 ms (r5). `kcompactd` woke 2,541 times in r4, `kswapd` scanned 50K pages every
100 ms. The memory limit does not matter (50 GB: 10.8 s, 56 GB: 11.6 s evicted).

Memory of the engine: everything pinned is `hipHostMalloc(Mapped | NonCoherent)`, no `hipHostRegister`:
the expert stage (`pin_layer`, `engine/gpu/mega_experts.cpp`, 27.4 GB at `--ctx 140000`), the KV in RAM
(`alloc_kv`, `mega_build_bufs.cpp`, 1.94 GB at 140K, 3.62 GB at 262K) and the per-slot sequence buffers
(`alloc_seq_buffer`, `mega_seq.cpp`, 4.1 GB per further slot at 262K). The model file is plain `mmap` page cache. `Unevictable`
in `/proc/meminfo` is 672 kB: nothing of this is locked, `vm.compact_unevictable_allowed=1`, THP in `madvise` mode.

Micro-test without the engine (`/var/tmp/claude/w75/mb/evict_pressure.cpp`): 20 GB `hipHostMalloc` while another thread
reads a 29 GB file in a 30 GB cgroup, with small kernels launched every 5 ms. First runs: +1,352 ms and +475 ms evicted
within one second, with 830K and 8K migrated pages. Control with 0.01 GB pinned and 294K migrated pages: 0 ms. In a second
series (40 s, file already cached, almost no reclaim) all variants stayed near 0 (default 0, Mapped|NonCoherent 470 and 0,
Coherent 0, `hipMallocManaged` 16, `hipHostRegister` of THP memory 155 ms) and tell nothing about which allocation is
safe. `hipMemcpy` of 1 GB to and from pageable memory does not evict (`evict_mb.cpp`, 40 copies, +0 ms).

A GPU memory fault of a prefill kernel (`k_moe_sum`, address in host-mapped memory, "page not present") killed one w75
server in a bit-exact run (`/var/tmp/claude/w75/ab/ab-w75-1793438573.log`). That is #72, not changed here.

## How an eviction becomes an abort

A wave that was evicted while it waited at a barrier finds `now - t0 > limit` when it runs again (the clock kept going), before
the blocks that were evicted with it have arrived: a wrong alarm. Same race at `host.wait`: the blocks wait behind block 0
with the barrier limit (1.0 or 1.2 s) while block 0 waits for the CPU and yields only after 100 ms; after a freeze of
1 s the waiters count the abort first, the yield comes too late. r3 shows it: the pass ran through to the end, the abort was
counted without a step (`bar[3] == 0`).

With the new report the abort says it: `cause: barrier at its limit (block 14 waited 1100.0 ms of 1000.0 ms, pass deadline
+386.3 ms), 31 arrivals counted, generation 300`, with `IE_MEGA_ABORT_TRACE` also `arrivals at step 326: 31 of 32 blocks ...,
missing blocks 10`.

## Fix

Waits shift their limits by gaps between two polls longer than `IE_MEGA_FREEZE_US` (2 ms): a wave that polls does so every few
microseconds, a gap means it did not run. Time in which the wave did not run does not count; a block that never arrives is
still found after the limit (1 s) by waves that poll without gaps. `IE_MEGA_WAIT_CAP_MS` (5 s) bounds what the shifts of one wait
add up to. The deadline of the pass counts only for a wait that has lasted 100 ms (a freeze while nobody waits eats the
deadline too and the waits behind it are short). Code: `matvec_barrier.h` (`WaitLimits`, `WaitClock`, `barrier_wait`),
`mega_op_wait.h` (`dep_wait`), limits per launch in `mega.hip`. The arithmetic is in `gpu/wait_clock.h`, free of the GPU, and
runs in the CPU unit test `tests/test_wait_clock.cpp` (ctest label `unit`). The diagnostics (gap histogram,
`IE_TEST_FREEZE_*`) exist only in a build with `-DIE_WAIT_DIAG=ON`; in the production kernel they cost registers.
`CpuStall` (CPU stretch at a router, `IE_MEGA_STALL_MS`) reads no clock and calls no `getrusage` when the switch is 0.

Why this does not weaken the GPU safety: only time in which the wave was not scheduled is left out. A kernel that hangs keeps
its waves running and polling; their waits end at the same limits as before (1 s per wait, 1.5 s per pass, 100 ms yield
at routers). The 5 s cap bounds the sum of shifts.

**Poll gaps (`IE_MEGA_GAP_HIST`)**, r11 (100 requests) and r12 (80 requests, `vkcube` running on the desktop), together
1.08e9 polls: 77 gaps of 16 to 64 us, none between 64 us and 2 ms, 12 between 2 and 10 ms, 25 between 10 and 100 ms (r12), none longer
(waits that are polling when a freeze starts; the longest eviction in these runs was 575 ms). The largest gap of a wave that
ran is under 64 us, so 2 ms is more than 30 times above it; any threshold from 64 us to 2 ms gives the same result on this data.

## Tests

- **Simulated eviction** (`IE_TEST_FREEZE_MS=1100`, launch 20, barrier 300; one block arrives 5 ms later than the waiters stand
  still): without the shift (`IE_MEGA_FREEZE_US=0`) the pass aborts every time (HTTP 500, "waited 1100.0 ms of 1000.0 ms", missing
  block named); with the fix 0 aborts in 40 requests. Repeated on every build of the fix: always one abort without and none with the shift.
- **Soak** (s1): 1,000 requests, 14,899 passes, 0 aborts. It contained natural stalls of 889 ms (moe.down), 798 ms (CPU router),
  795 ms (host.y) and 343 ms. Without the fix the rate was 1 in 750 requests; 1,000 requests are 16 times the 60 requests
  that r3 needed for its abort.
- **Deadline as a safety net** (build with `IE_WAIT_DIAG`, `IE_MEGA_BUDGET_MS=300`, `IE_TEST_FREEZE_MS=700 IE_TEST_FREEZE_ALL=1`: all
  blocks stand still 700 ms at barrier 300, the waits behind it are short): 1 abort in 10 requests at 713.8 ms into the pass,
  the message names the cause "barrier at its pass deadline (block 11 waited 0.0 ms of 1000.0 ms, pass deadline -413.9 ms)";
  control with the default budget of 1,500 ms: 0 aborts. Simulated eviction again on this build (`IE_TEST_FREEZE_MS=1000`, 20 requests):
  1 abort with `IE_MEGA_FREEZE_US=0`, 0 with the shift. No stand-still longer than 1 s (the setting is clamped).
- **Unit test** `test_wait_clock` (CPU, in `just test-cpu`): gap below and above `freeze_ticks`, a hang is still found, shifting
  off, cap, deadline for short and long waits, the 300 ms rule, no deadline (`~0` does not overflow). 30 of 30 unit tests pass.
- **Bit-exact** after the merge with main f72c8de (`IE_DETERMINISTIC=1`, 200 tokens, w75 against p1 = f72c8de): SHA
  `065922c2ff602991` at 30K and `439aad472c59ae6f` at 130K on both. Golden on the merged state: decode 3 of 3, prefill dflash 1 of 1,
  server 8 of 8 shown (tail), all identical.
- **Bit-exact** before the merge (`IE_DETERMINISTIC=1`, 200 tokens, w75 against merged, 3 interleaved pairs): SHA `065922c2ff602991` at 30K and
  `439aad472c59ae6f` at 130K in all runs.
- **Golden**: `scripts/golden.sh compare` 3 of 3 identical, `GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash` 1 of 1,
  `scripts/golden-server.sh compare` 20 of 20, exit 0 each.

## Speed

Final state (merged with main f72c8de, reference `work/build/p1` = f72c8de), `scripts/bench-set.py` (server, MTP, ctx 32768),
decode tok/s, 3 rounds alternating p1 / w75:

| | p1 | w75 |
|---|---|---|
| code | 84.8 (warm-up), 88.9, 88.9 | 88.4, 88.3, 88.3 |
| text | 64.2, 64.8, 64.8 | 64.1, 64.0, 64.0 |

From the second round: code -0.7 %, text -1.2 % (the group has 3 prompts, 1,148 tokens). Decode of 200 tokens after the 30K / 130K
prompts, 3 rounds alternating (`ab75.sh`, ctx 140,000): p1 74.9, 75.9, 76.0 and 64.0, 64.9, 64.9; w75 75.5, 75.6, 75.5 and 64.4, 64.4, 64.4
(30K equal, 130K -0.3 %). Prefill 1,803 and 1,713 tok/s on both. Decision of the maintainer: accepted for the stability fix
(a stability fix may cost up to 2 % after cheaper variants were tried).

The rounds above ran on a w75 binary built before the last edit of the wait code. Re-check of the committed merge
(`work/build/merged` = main + this branch, same tree as `ca6cde8`), 4 rounds alternating against p1: code 87.8, 88.9, 88.9, 88.9
vs 88.4, 88.3, 88.3, 88.4; text 64.1, 64.7, 64.7, 64.7 vs 64.0, 64.0, 64.0, 64.0 (from round 2: code -0.6 %, text -1.1 %).
Same build: unit 30 / 30, golden decode 3 / 3, prefill dflash 1 / 1, server 20 / 20 identical, `IE_DETERMINISTIC=1` SHA
`065922c2ff602991` (30K) and `439aad472c59ae6f` (130K) as p1.

The cost is the code shape of the kernel, not the CPU side: `CpuStall` without a syscall did not bring the text rate back
(measured before: code 88.5 vs 88.5, text 64.1 vs 64.5 with the old `CpuStall`; after: 88.3 / 64.0 vs 88.9 / 64.8 against
a different main). VGPR / SGPR spills of `k_mega` (`res.sh`, kernel-resource-usage):

| kernel | p1 VGPR spills | w75 | p1 SGPR spills | w75 | p1 scratch B/lane | w75 |
|---|---|---|---|---|---|---|
| `k_mega<1>` | 58 | 61 | 93 | 107 | 372 | 448 |
| `k_mega<4>` | 49 | 49 | 97 | 107 | 324 | 400 |
| `k_mega<8>` | 126 | 149 | 135 | 257 | 468 | 544 |

Attempts to get closer (each judged by `res.sh` first): diagnostics compiled out (`IE_WAIT_DIAG`, kept: VGPR 64 / 57 / 162 -> 61 / 49 / 149);
cap as an absolute end instead of start + length (kept); deadline thresholds precomputed on the host (worse, `k_mega<8>`
179 VGPR spills); no cap in `WaitClock` (no gain). Test hooks and `count_poll` non-inlined: worse. A build with trivial wait
limits (no shift at all) has the same spills (`k_mega<8>` 189 VGPR), so the spills come from the changed surrounding code and
not from the limit arithmetic; the compiler reacts to every change of the wait code in an unpredictable way.

Older series, before the merge (merged vs w75, balanced): code 88.6, 88.4, 88.5 vs 88.5, 88.5, 88.4, text 64.5 vs 64.2, 64.1, 64.1.
The kernel reacts to the code shape of its waits: a variant with cheaper arithmetic per poll (shifts as ends of the limits)
was 1.1 % slower in code and text, one with the wait loop in a non-inlined function and a lean loop 1.1 % / 0.5 % (V3) and
3.1 % / 2.0 % (V4), and with non-inlined `dep_poll` 1.3 % / 0.8 % (V5). The inlined version is the fastest of the five.

## Stall time before and after

The fix removes the abort, not the latency: evictions go on. Sum of in-step stalls of 100 ms and more per 100 requests (varies by
a factor of 10 from run to run): without the fix r1 3,668 ms, r4 696, r5 1,517, r6 536, r7 1,741; with the fix r8 1,870, r11 403,
r12 890, s1 308 (1,000 requests). Evicted time per run is 3.9 % of the wall time in r4 and about 2.5 % in s1.

## Proposal D: remove the cause (system settings)

Not implemented, decision of the user: without root. Not tested; the effect is a hypothesis from the evidence above.

- `LimitMEMLOCK=infinity` for the service (hard limit is 8 MB now) and `mlock` of the `hipHostMalloc` regions plus
  `vm.compact_unevictable_allowed=0`: the kernel then cannot migrate these pages. Effect: stops the page-migration part (the
  windows with 700 to 1,300 ms per 10 s). Risk: locked memory cannot be reclaimed under pressure (27 GB of 60 GB), a
  mis-sized limit leads to the OOM killer instead of slow reclaim.
- `vm.compaction_proactiveness=0`: no background compaction (`compact_daemon_wake` 2,541 in r4). Risk: fewer free huge
  pages, higher allocation latency elsewhere.
- Less page-cache churn from the model file (kswapd 50K pages per 100 ms during decode).
- `amdgpu.debug_evictions=1` (needs root) would name the cause of every eviction and settle whether the hipHostMalloc
  pages or something else is invalidated. If #72 (memory fault in `k_moe_sum`, host-mapped address) comes from the same
  migration, this would help there too.

## Open

- The wait diagnostics need their own build (`-DIE_WAIT_DIAG=ON`); the simulated-eviction server tests run on it, not on the
  production binary (same sources otherwise).
- The decode loss of 0.7 % (code) and 1.2 % (text) from the changed code shape of `k_mega` stays (accepted).
