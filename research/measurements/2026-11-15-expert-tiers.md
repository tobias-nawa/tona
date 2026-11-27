# Expert tiers: probes and measurements for the usage-ordered placement, 2026-11-15

Design: [docs/architecture/expert-tiers.md](../../docs/architecture/expert-tiers.md). Each step below has its plan
written before its run, then its result and a short decision note (what the number says, what it means for the next
step). Test conditions are appended per step.

## Step 1: routing trace and offline replay (plan)

Question: how many of the VRAM misses of a coding decode reach a cold tier, for a RAM room split between pins and page
cache, and which ranking and split leave the fewest SSD reads? Decision: the pin/page-cache split and the page-cache
floor of the budget code (design values 4.3 GB and q = 0.2 are estimates), and whether the 32 GB estimate (about
2.3 % of the reads cold) holds.

Scope: one run of `scripts/bench-set.py` (11 code and 3 text prompts, greedy, up to 400 tokens, MTP) against the build
`tiers` (main 4259191, no engine change) at the 58 GB class (all experts pinned, as production, profile
`work/profile/experts-coder.bin`), with `TONA_ROUTE_TRACE` (union of experts per layer and pass) and
`TONA_ROUTE_TRACE_P`, `TONA_DECODE_DIAG=1`. The bench-set prompts are not in the profile's prompts
(`scripts/make-profile.sh`), so the profile is tested on held-out prompts. Offline replay with `tools/expert_tier_sim.py`
(new, no GPU): per-layer LRU of the VRAM slots seeded from the profile, VRAM misses sent to pins (ranked) or a page-cache
LRU; room R = class budget - 0.5 slack - 1.0 prompt reserve - 1.93 held = 6.6 GB (10 GB class) and 22.6 GB (26 GB class).
One run is enough: the output is a deterministic trace, the replay a deterministic function of it. Why this size: 14
prompts give about 4,000 tokens, more than the 1,920 of the profile, in one GPU job of a few minutes.

Expectation: the replay's VRAM hit rate within 5 points of the server's counter (74.8 to 88.5 % in
`2026-10-24-expert-prediction.md`, the replay has no preload, so rather the low end, 75 to 85 %); VRAM misses 13 to 25 %
of the unioned reads (about 50 to 95 expert reads per token at 384 reads per token, fewer per token when a pass
unions several tokens); at R = 22.6 GB pinning 15 to 19 GB leaves fewer SSD
reads than no pins by a factor of 2 or more; at R = 6.6 GB the split changes the SSD reads by less than 30 %. A
result outside this is a finding. Active tooling: route traces, `TONA_DECODE_DIAG`, `TONA_MEGA_INFO`, `TONA_VRAM_LOG`;
no profiler, no `TONA_DETERMINISTIC`.

### Step 1 results

Test conditions (`tools/hw_report.py` before the job): Ryzen 5 9600X, RX 9060 XT gfx1200 15.9 GiB (1.22 GiB held by the
desktop), RAM 60.5 GiB (54.5 available), swap 2.11 GiB in use, load average 5.3 (other workers' CPU work between GPU
jobs), commit 4259191 (tree with the new `tools/expert_tier_sim.py`). `quiet_check` before: warning (load 4.86, 1.22 GiB
VRAM held by the desktop), measured anyway: the output is a deterministic trace, not a rate. After: "do not measure"
because the quiet check ran while my own server still held 15.3 GiB of VRAM (not another program). Rates below are not
speed claims (one run, other workers on the CPU).

Run: server on port 8091, 58 GB class, all experts pinned (26.3 GB), 85 VRAM slots per layer, MTP, `bench-set.py` once:
**decode** code 92.0 tok/s, text 67.3 tok/s (4,087 tokens, 1,122 passes, 3.6 tokens per pass); server counters
`TONA_DECODE_DIAG`: hits 1,236,868 / misses 178,470 = 87.4 % hit with preloading. Replay output:
[2026-11-15-expert-tiers-replay.txt](2026-11-15-expert-tiers-replay.txt) (full tables; `tools/expert_tier_sim.py`),
files in `/var/tmp/claude/148/res/` (the 540 MB logit trace is not committed).

| Quantity | Expected | Measured |
|---|---|---|
| Replay VRAM hit rate (LRU, no preload) | 75 to 85 % | **77.0 %**; server 87.4 % (preload on) |
| VRAM misses per token (replay, no preload) | 50 to 95 | 85.4 = 175 MB |
| SSD reads per token, R = 22.6 GB, page cache only (today's rule) | not stated | **4.2** |
| same, 20.3 GB pinned by profile + 2.3 GB cache | at least 2x fewer than none | **1.7** (x2.5 fewer); 18.1 + 4.5 GB: 1.9 |
| SSD reads per token, R = 6.6 GB, page cache only | not stated | **66** (135 MB per token) |
| same, best split by profile ranking | within 30 % of none | **no gain**: 60 to 72 (+10 % to -9 %) |

Per room (SSD reads per token; the second half of a column is the same trace, ranking learned on the first half):

| Room (class) | page cache only (LRU) | managed cache, decayed-use replacement | pins by profile, best split | pins by own miss counts (first half -> second half, vs its own page-cache baseline) |
|---|---:|---:|---:|---:|
| 6.6 GB (16 GB class) | 66 | 55 (-16 %) | 60 (all pinned) | not run at this room |
| 10 GB | 44.8 | 39 (-13 %) | 42.4 (all pinned), 44 to 53 mixed | 36.6 -> 26.8 (-27 %) |
| 14 GB | 25.7 | 23 (-10 %) | 24.2 (all pinned), 25 to 30 mixed | 21.4 -> 16.3 (-24 %) |
| 18 GB | 11.9 | 11.1 (-7 %) | 10.9 (all pinned), 11.5 to 12.3 mixed | 10.9 -> 8.2 (-25 %) |
| 22.6 GB (26 GB class) | 4.2 | 4.0 | **1.7** at 20.3 GB pinned | 6.3 -> 1.6 |

Calibration against today's decode (**ungeprueft** beyond these two points; the replay has no preload, so absolute
counts are an upper bound): at the 26 GB class today's gap to the pinned run is 15.9 - 9.2 = 6.7 ms per token (63 against 109
tok/s, short-code) for 4.2 reads: 1.6 ms per cold read. At the 10 GB class 135 MB per token at 14 tok/s is 1.9 GB/s, 70 %
of the 2.65 GB/s sequential rate: that class is **bandwidth-bound by the number of reads**, not latency-bound.

### Step 1 decision note

- **The 2026-10-27 cause is confirmed in a new form.** Partial pins by the profile order *lose* to a plain page cache at
  rooms of 6 to 18 GB (up to 18 % more SSD reads), because the profile counts reads, not VRAM misses, and the page-cache
  LRU follows the workload. A static pin only beats the cache when the unpinned rest is truly cold: at 22.6 GB room with
  18 to 20 GB pinned (75 to 80 % of the experts) the SSD reads drop from 4.2 to 1.7 to 1.9.
- **Correction to the design note.** (a) The 16 GB class is bandwidth-bound by the count of reads (66 per token, 76 % of the
  VRAM misses reach the SSD), not 41 MB as in section 6; the claim that parallel reads would give 20 to 28 tok/s is wrong
  at that class: the read count is the lever, and ordering/policy gives only 10 to 27 %. (b) The profile alone is a weak
  ranking below about 75 % of the experts pinned; online VRAM-miss counts are the better key (-24 to -27 % in the honest
  first-half/second-half test).
- **Expected gain, revised (estimates):** 26 GB class: 4.2 -> 1.7 to 1.9 reads x 1.6 ms = -3.7 ms per token on 15.9:
  **about 80 to 85 tok/s on short-code (x1.3)**, text about x1.2 to 1.3; the design's 85 to 95 / 62 to 70 was too high.
  10 GB class: about x1.1 to 1.3 from a frequency-aware managed cache or miss-count pins; no more without fewer reads.
- **Plan change for steps 3 to 5:** step 3 (static tiering) targets the 26 GB class: pin by profile when the budget
  holds at least about 70 % of the experts, otherwise keep today's page cache (do not pin partially below that). Cache
  floor: 3.0 GB looks enough in the replay (20.3 + 2.3 beats 18.1 + 4.5); measure 3.0 against 4.3. The small classes
  need the online miss-count ranking (step 5's governor already needs online counters) and a frequency-aware replacement,
  not the profile: moved behind step 3 as step 3b. Step 4 (explicit cold reads) matters at the 26 GB class (latency of
  1.7 to 4 reads per token) and has at most about x1.4 at the 10 GB class (bandwidth 70 % used already).

## Step 2: cost of giving back and taking a pinned piece (plan, written before the job ran)

Question: what do `hipHostMalloc`+`hipHostFree` (GTT) and mmap(huge pages)+`hipHostRegister`+`hipHostUnregister` of a
32 MB piece cost, and does another stream's kernel gap grow while they run? Decision: the piece size and the release step
of the governor (8 pieces per gap between passes in the design). Under about 5 ms per piece and no kernel gap above
5 ms: pieces of 32 MB, 8 per gap. Above that: smaller steps or pieces by layer (a stall above the 100 ms yield would be
a safety problem).
Scope: `bench/micro/piece_cost.hip`, 40 cycles per kind at 32 MB and at 8 MB, an idle GPU (no server), a ticker thread
with a tiny kernel every 200 us on its own stream, `HSA_USERPTR_FOR_PAGED_MEM=0` as the engine sets it. Why this size:
the mechanism is a per-call cost, 40 cycles give median and p90.
Expectation: malloc 1 to 5 ms, free below 2 ms (GTT is a driver call plus page pinning, 32 MB = 8,192 pages of 4 KB;
estimate, no earlier measurement: research/44 names it unchecked); register with huge pages 1 to 3 ms; unregister plus
munmap below 3 ms; the memset (the floor of reading data into the piece) about 3 ms at about 10 GB/s single thread. Kernel gap
during the cycles below 2 ms; a gap above 20 ms would mean the queues stall (the 2026-11-11 eviction case). Active
tooling: none. A tool built outside the queue (one 5 s compile of one file, no GPU use) is the only deviation.

### Step 2 results

Conditions: `tools/hw_report.py` and quiet check before and after the job (`/var/tmp/claude/148/res/s2.*`): warning only
for the 1.22 GiB the desktop holds and a few swapped pages; idle GPU, no server. 40 cycles per kind, 32 MB and 8 MB
pieces, ticker thread with a tiny kernel every 200 us on its own stream.

| 32 MB piece | median | p90 | max | Expected |
|---|---:|---:|---:|---|
| `hipHostMalloc` (GTT) | 0.44 ms | 0.45 | 0.60 | 1 to 5 ms |
| `hipHostFree` | 0.12 ms | 0.14 | 0.22 | below 2 ms |
| mmap + madvise(huge) + `hipHostRegister` | 0.47 ms | 0.53 | 1.41 | 1 to 3 ms |
| `hipHostUnregister` + munmap | 0.10 ms | 0.13 | 0.23 | below 3 ms |
| first touch of the GTT piece (memset) | 1.98 ms | 1.99 | 2.01 | about 3 ms |
| kernel gap of the other stream, during GTT / registered cycles | 0.276 / 0.277 | 0.300 | 0.330 / 0.319 | below 2 ms (idle 0.276, max 0.316) |

The 8 MB pieces cost about a quarter of that. All of it is below the expectation and the other stream's gap does not move.

### Step 2 decision note

Giving back or taking a 32 MB piece costs 0.1 to 0.5 ms, so a release step of 8 pieces (260 MB) is about 1 to 4 ms:
less than one pass. Piece size 32 MB and 8 per gap stay (the governor may take 32 pieces in an emergency, about 15 ms).
Limits of this result: pieces that the GPU has not read; **ungeprueft** is the unregister of a piece a running kernel
read (KFD mapping to tear down), and the registered range under real page-cache pressure. That is for the step-5 test
with the balloon. For step 3: nothing to change, the static stage keeps one buffer per layer.

## Step 3: static tiering at the 26 GB class (plan, written before the runs)

Change (engine): with `TONA_TIERS=1`, a profile and `TONA_RAM_GB`, `pin_budget_from_limit` takes a part of the experts
when at least 70 % of the expert bytes fit (page-cache floor 3.0 GB, `TONA_TIERS_CACHE_GB`), and `pin_experts` chooses
them by one ranking over all layers (`backend/expert_tiers.h`, unit test `test_expert_tiers`). Default off: arm A is the
same binary without the flag, i.e. today's all-or-none rule.

Question and decision: does the partial pin lift **decode** and **prefill** at the 26 GB class (the 32 GB machine's
installer budget) against today, without costing anything where nothing changes? Decision: faster by more than 10 %
on decode and not slower on prefill: keep the code, decide on turning it on by default after the 10 GB class and the
dynamic part (steps 3b/5); equal or slower: step 3 stops, the replay's reason for it is wrong and the analysis restarts
at the page-cache behaviour (counters).

Scope: build `tiers` (main 4259191 + this change), `scripts/ab.py`, A = default, B = `TONA_TIERS=1`; both
`TONA_RAM_GB=26`, `--memory-max 26000000000`, profile `work/profile/experts-coder.bin` (ab.py sets it), `--ctx 65536`
(the installer's context for a budget below 32 GB; one ctx for all workloads, so `set` runs at 65536 and not 32768),
workloads `short,mid,set` in production mode (no `TONA_DETERMINISTIC`), `--fixed-rounds` with 3 rounds alternating
(AB, BA, AB), fresh server per start, page cache not cleared (warm steady state; a cold start is the second question and
is read from the start log: start time and pinned bytes). Port 8091. Counters: `TONA_DECODE_DIAG=1` on both arms
(counters only: hits, misses, cpu_experts, gpu_direct per request). Why this size: the effect is a factor (replay:
2.2 fewer SSD reads of 4.2 per token), 3 rounds of the three workloads give a spread; the 30K prompt (`mid`) is the
prefill case that touches every expert. Bit-exactness separately: `TONA_DETERMINISTIC=1`, `short`+`mid`, 2 rounds, the
SHAs of A and B equal.

Expectation (source: the replay of step 1 and the measured classes): at the 26 GB budget B pins 19.6 minus held about 1.9
= 17.7 to 19.6 GB, i.e. 70 to 78 % of the experts; **decode** B/A +20 to +40 % (63 -> 76 to 88 on short code; the
step-1 estimate x1.3 with 1.6 ms per cold read, one calibration point), text +15 to +35 %; **prefill** at 30K between
+0 % and +100 % (unpinned 720 tok/s, fully pinned 1,741 at 13K in the 2026-10-27 and 2026-11-16 files: a partial pin
lies between, wide because prefill reads every expert of a layer). Equal tokens per pass. A result outside this is a
finding. Active tooling: `TONA_DECODE_DIAG=1`, `TONA_VRAM_LOG=1`, `TONA_MEGA_INFO=1` (log lines only).

## VRAM resilience: what happens today when another GPU program takes VRAM (plan, written before any run)

Requirement (maintainer): after hours of work the user starts another GPU program (a 3D tool). Today the VRAM expert
slots are sized once at start (`size_slots`: free VRAM per sysfs minus `TONA_VRAM_RESERVE_MB`, 640) and never change.

What the code says (read, not run):
- All slots of all layers, and the prefill workspace behind them, are **one** `hipMalloc`
  (`ProgramBuilder::alloc_slots_and_arena`, `mega_build_slots.cpp`), the slot addresses are baked into the tasks
  (`install_slots`). Freeing slots to give VRAM back to another process therefore needs the allocation split (per layer or
  per slot group) or a virtual-memory reservation whose physical chunks can be unmapped (`hipMemAddressReserve` /
  `hipMemCreate` / `hipMemMap` / `hipMemUnmap`): **ungeprueft** whether that API works on this stack, first probe of the build.
- Under VRAM pressure the kernel's TTM evicts buffers of the process, KFD stops its queues until they are restored
  (`evicted_ms`, handled by `TONA_EVICT_CREDIT` in the wait clock: the pass survives, it is slow).

Question: how does today's build behave while a second process allocates VRAM in steps and holds it? Decision: the
size of the governor's problem: if the balloon is denied or the server stalls, shrinking is needed before the other
program starts; if it gets its memory with a small stall, the governor is a comfort feature. Outcomes to read: (a) the
balloon fails at step N (the card is full, our slots keep the memory: the other program fails or swaps, the user sees an
out-of-memory in it); (b) the balloon succeeds by evicting our buffers (TTM moves them to GTT): the server's decode drops;
(c) the desktop or a queue hangs (stop).

Scope: build `tiers` default behavior (no `TONA_TIERS` needed, VRAM is independent), server on port 8091 in the 58 GB
class (everything pinned, production), `bench/micro/vram_balloon.hip` (new: hipMalloc steps, no memset, no kernel, one
line per event with sums of sysfs) with 512 MB steps, 12 steps (up to 6 GB), 15 s between steps, 60 s hold, then free;
`/var/tmp/claude/148/decode_loop.py` runs 300-token decodes one after the other and logs rate, the server's KFD
`evicted_ms` and the card's VRAM used per request for the whole time (about 6 minutes: 30 s before, balloon, 90 s after).
One run: the question is qualitative (denied, stalled or fine), the numbers carry no spread. Why 512 MB steps: the
server leaves about 0.6 to 1.0 GB free (free VRAM per sysfs minus 640 MB reserve, 1.2 GB held by the desktop), so the
first step already forces the decision; small steps show at which size the stall starts.

**Warning, the maintainer is away and the card shares the desktop:** a VRAM balloon can stall the desktop (rule "no
blocking HIP call while k_mega waits": the balloon is a separate process and does not call anything blocking beyond
`hipMalloc`, the server's wait is protected by the 100 ms yield; the desktop's own jobs wait while k_mega runs, 2 s
`lockup_timeout`). Abort rule: if a request of the loop takes longer than 15 s, or `evicted_ms` rises by more than 5 s in
one window, or the balloon's `hipMalloc` blocks for more than 20 s, the driver script kills the balloon by its PID, waits
30 s and stops; on any kernel message about a ring timeout or reset: stop at once, save `journalctl -k -b | tail -80`
to `/var/tmp/claude/148/res/`, report. Never `pkill -f`; only own PIDs.

Expectation (no earlier measurement, from `2026-11-11-evict.md` and `docs/backends/reference-platform.md`): the first 1 to 2 steps
are granted from the 0.6 to 1.0 GB left plus the 640 MB reserve; from about 1.5 GB the card is full, `hipMalloc` of the
balloon either fails with out of memory or TTM evicts a part of the server's allocations; the decode rate drops by more
than half while `evicted_ms` grows, and recovers within 10 s of the release. Which of the two (denied or evicted) is the
finding; I do not know it. Active tooling: `TONA_EVICT_CREDIT` default, `TONA_VRAM_LOG=1`, `TONA_DECODE_DIAG=1`.

### Step 3 results

Test conditions: `tools/hw_report.py` at the start of both runs (`/var/tmp/claude/148/res/s3.hw.md`): Ryzen 5 9600X, RX 9060 XT
15.9 GiB (1.22 GiB held by the desktop), 60.5 GiB RAM (53.9 available), swap 2.1 GiB in use, load average 5.1, commit
e3737b3 plus the uncommitted tiers change (the committed code is the same except for the split of `pin_experts`). Quiet
check before each run: warning (load average, 1.22 GiB VRAM of the desktop), measured anyway: arms alternate, and the
effect is far above the noise. After each run: "do not measure" because of the server of the last start still holding
15.3 GiB VRAM when the check ran, and CPU use of other workers (6 % to 53 % of the threads) in between. The other workers'
CPU and build work ran between my jobs for the whole time; it affects both arms alike in each alternating pair.
Reports: [first run](2026-11-15-expert-tiers-step3-ab.md) and [second run](2026-11-15-expert-tiers-step3b-ab.md).
Names of the workloads fix their own context (`short`, `mid` at 140000, `set` at 32768; my `--ctx 65536` did not apply).

**First run** (budget without the pinned KV and sequence buffers, i.e. 21.25 GB offered, 216 experts per layer = 10,391 pinned
= 84 %, 21.9 GB at ctx 32768 and 23.5 GB at ctx 140000 including the pinned sequence buffers), B (`TONA_TIERS=1`) against A
(default), 3 rounds alternating AB, BA, AB, same binary, all in production mode:

| Workload | A | B | B against A [95 % CI] |
|---|---:|---:|---:|
| **decode** after 2K prompt (`short`) | 43.7 | 74.1 | +67 % [+61, +74] |
| **decode** after 2K prompt, write request | 57.5 | 116.8 | +100 % [+91, +109] |
| **decode** after 30K prompt (`mid`) | 42.3 | 69.2 | +63 % [+62, +64] |
| **decode** bench-set code | 62.0 | 92.5 | +49.5 % [+47, +52] |
| **decode** bench-set text | 35.6 | 66.5 | +87 % [+85, +89] |
| **prefill** 30K prompt | 713.8 | 1255 | +80 % [+77, +83] |
| **prefill** 2K prompt (`short`, write) | 750 / 774 | 722-1149 / 1357-1413 | +71 % (wide, one round off plan) / +87 % |

All tokens identical: the output SHA of every request is equal in A and B and stable over the rounds (2K, 2K write, 30K
tokens), in production mode. Expert hits: A 83,417 hits / 69,432 misses (55 %), B 130,198 / 22,651 (85 %) per request:
a partial pin raises the VRAM hit rate by 30 points because the preloading (`mega_prefetch.cpp`) takes only pinned
experts; the replay of step 1 has no preload and misses this. No yield, no abort, no page-fault problem in any B start
(`yields 0`), although at ctx 140000 only about 0.6 GB of the 26 GB were left for the page cache by the accounting
(the pinned KV buffers fill lazily).

**Second run** (budget counting the pinned buffers filled later: "filled later" 1.09 GB at ctx 32768, 4.18 GB at ctx
140000, floor 3.0 GB): at ctx 140000 the budget is 17.1 GB = 68 % of the experts, below the 70 % threshold, so **nothing was
pinned and B equals A** (decode/prefill within -1.2 % to +0.7 %, identical counters): the safety rule did what it says. At ctx
32768 (`set`) 205 experts per layer (9,861, 80 %, 20.8 GB) were pinned: **decode** code 62.5 -> 85.5 tok/s
(**+38.8 %** [+34.4, +43.2]), text 35.8 -> 64.7 (**+81.2 %** [+80.0, +82.4]). Against the first run (21.9 GB pinned,
decode code 92.5) 1.1 GB less pinned costs 7 tok/s on code: near the full set each GB matters.

Against the plan (expectation decode +15 to +40 %, prefill 0 to +100 %): decode above it (+49 to +100 % with 84 % pinned, the
replay without preload underestimated it), prefill inside it, tokens equal, no abort.

### Step 3 decision note

- **The change works and is large:** 78 to 84 % of the experts pinned lifts **decode** by 40 to 100 % and **prefill** by 70 to
  90 % at the 26 GB class, tokens identical. It stays behind `TONA_TIERS` until two points are settled: the floor and the
  threshold, and the bit-exact check in deterministic mode and the goldens (queued, below).
- **Why more than the replay said:** the VRAM hit rate depends on the pins (preloading takes only pinned experts): 55 % without
  pins, 85 % with 84 % pinned. The pins are worth more than the page cache they replace. This also revises step 1's
  conclusion for the small classes: the replay's "no gain from profile pins" did not count the preload effect; the 10 GB
  class needs a new measurement with pins by miss count before it is written off (step 3b).
- **Open decision for the coordinator: the page-cache floor.** The first run worked with 0.6 to 2 GB left for the cache and
  gave +60 to +100 %; the safe accounting (3.0 GB floor, KV and sequence buffers counted in full) pins nothing at ctx 140000
  on 26 GB and 80 % at ctx 32768, +39 % and +81 %. Each GB of pins near the top is worth about 6 tok/s on code. Options:
  (a) keep floor 3.0 and the full KV: safe, gives nothing at 26 GB with long contexts; (b) floor 1.5 GB and count the KV at
  min(ctx, 65536): the installer's context for budgets below 32 GB is 65536 (1.1 GB KV), pins about 82 %; (c) as the first run. I
  recommend (b) after one floor sweep (3.0, 1.5, 0.5 GB) with the page-cache counters and a long-context run under the
  same limit to see reclaim stalls; this is a measurement, not a guess, and takes about 1 hour of queue time.
- **Plan for the rest:** the VRAM governor part waits for the coordinator's answer (the maintainer's new requirement is
  written below); step 4 (explicit cold reads) now has a clear target: the 16 to 20 % of experts left cold at the 26 GB class
  also lack the preload; reading them early into slots would add the same effect for them.

## hipMem probe: can the slot pool return VRAM? (plan, written before the run)

Question: do `hipMemAddressReserve` / `hipMemCreate` / `hipMemMap` / `hipMemSetAccess` / `hipMemUnmap` / `hipMemRelease` work on
gfx1200 with ROCm 10.1 (HIP 7.16), what do they cost per 32 MB chunk, does an unmap and release give VRAM back to the card
(sysfs `mem_info_vram_used`), and does unmapping stall a kernel that reads other chunks of the same mapping? Decision:
works and gives VRAM back with a stall below 5 ms: the slot pool becomes one reserved address range with mapped chunks
(addresses stay fixed, so the task addresses stay valid; a dropped chunk is a set of slots that are never used again until
re-mapped). Does not work, gives nothing back or stalls: the pool must be split into per-layer allocations that are
freed whole (coarser, an address change of the tasks), or the VRAM part is limited to the stop-and-restart of the engine.
Scope: `bench/micro/vmm_probe.hip`, standalone (1 GB of VRAM for a few seconds, no server running; the queue holds the
GPU), 32 chunks of about 32 MB, a kernel reading the lower half on another stream while the upper half is unmapped chunk
by chunk, then a re-map. One run: the result is qualitative plus a per-call time. Expectation: supported (the HIP 7 runtime
lists the API), granularity 2 MiB, create and map 0.1 to 1 ms per chunk, unmap and release 0.1 to 1 ms, VRAM used falls by
the released size within a second, no stall above 5 ms, the lower half still reads correctly. No earlier measurement.
Active tooling: none. Safety: the probe runs short kernels on its own stream, no persistent kernel, no server.

## Step 3c: page-cache floor sweep and a long-context run (plan, written before the runs)

Decision made by the coordinator: floor 1.5 GB with the KV in RAM counted at min(ctx, 65536) (now the code default,
`kTiersCacheFloorGb`, `kTiersKvCtx`); this sweep settles the final value. Question: do floors of 3.0, 1.5 and 0.5 GB differ in
speed, and does a low floor cause reclaim stalls (yields, aborts, major faults, PSI) in a long-context run? Decision: the
lowest floor with no yield, no abort and no loss against 3.0 beyond noise; if 0.5 shows stalls, 1.5 stays; if 1.5 shows
stalls, 3.0.
Scope: build `tiers` (committed code plus the floor/KV change), `scripts/ab.py` with both arms `TONA_TIERS=1`,
`TONA_RAM_GB=26`, `MemoryMax` 26 GB, `TONA_DECODE_DIAG=1`, workloads `short,mid,set` (ctx 140000 and 32768), 2 rounds alternating,
production mode: run 1 floor 3.0 against 1.5, run 2 floor 1.5 against 0.5. Run 3: the long-context case `long` (130K prompt,
ctx 140000: the KV in RAM fills up to 1.8 GB), 1.5 against 0.5, 2 rounds. Per start the server's `yields`, aborts
(`barrier aborted` in the log), and the major page faults and PSI memory `some` total of the server's cgroup before and
after (sums read from `/sys/fs/cgroup/.../memory.pressure` and `memory.stat` `pgmajfault`, no process names) are
taken from the log lines and a sampler. Why this size: the question is a threshold, so 2 rounds per pair and one long run per
pair; the speed effect between floors is expected to be a few percent, below what 2 rounds can resolve, which is why
the stall counters decide, not the rates.
Expectation: at ctx 32768 floor 1.5 pins about 82 %, floor 3.0 about 80 %, floor 0.5 about 84 %: decode code 85 to 92 tok/s
(sources: the two step-3 runs), differences between floors below 8 % (about 6 tok/s per GB of pins; the floors differ by
at most 2.5 GB of pins). At ctx 140000 with the KV counted at 65536 floor 1.5 pins about 78 %. No yield and no abort at
1.5 or 3.0; at 0.5 GB of cache left I expect 0 to a few yields in the long run (the 2026-10-27 file saw aborts below about 3.5 GB,
with the hot head unpinned, which no longer holds). A result outside this is a finding.

## Step 3b: partial pins at the 10 GB class (plan, written before the run)

Question: with preloading counted (it works only for pinned experts), does a partial pin by the profile help at the 16 GB
class (budget 10 GB), where the 70 % rule pins nothing? Step 1's replay said no gain without preload; step 3 showed that the
preload effect (VRAM hit 55 -> 85 %) is large. Decision: B faster than A by more than 10 % on decode: build the pins by
online miss counts (a ranking that learns what the VRAM misses), as the governor of step 5 needs those counters anyway, and
lower the threshold for small budgets; equal or slower: keep the 70 % rule, the 10 GB class stays page-cache bound (66 SSD
reads per token) and gets nothing from pins; its levers are fewer VRAM misses (explicit cold reads, step 4) and a smaller model.
Scope: `scripts/ab.py`, A default (nothing pinned), B `TONA_TIERS=1 TONA_TIERS_MIN_FRACTION=0.05` (the profile ranking, any
share), both `TONA_RAM_GB=10`, `MemoryMax` 10 GB, workloads `short,set` (ctx 140000 and 32768), 3 rounds alternating, production
mode. The offered budget is 10 - 0.5 - 1.0 - 1.5 (floor) - 0.25 - filled later = about 3.6 GB at ctx 140000 and 5.7 GB at
32768, i.e. 14 to 23 % of the experts. Counters `TONA_DECODE_DIAG=1` (hits and misses per request). Why this size: the effect
to find or exclude is larger than 10 %, 3 rounds give the spread; the 30K prompt is left out (at 10 GB it takes minutes and
the question is decode).
Expectation: decode B/A between -5 % and +25 % (replay: SSD reads 66 -> 60 to 72 with profile pins, so about 0; the preload
effect adds up to +10 to +20 points of VRAM hit for the pinned 14 to 23 %, which is worth up to +15 % at 135 MB per token if the
SSD bandwidth bound moves with the misses); A decode 10 to 14 tok/s (2026-11-16 file). Tokens identical. A result outside
this is a finding.

## VRAM reserve for the desktop (prior art and plan, written before the runs)

Cause (2026-11-15, VRAM balloon test, 58 GB class, ctx 65536): tona-server held 14.35 GB, the card showed 15,702 of 16,304 MiB
used, 0.6 GB free, which is about the default `TONA_VRAM_RESERVE_MB` (640) minus what the desktop grew by since the slots were
sized. A second process taking +512 MiB made amdgpu refuse command submissions of the compositor ("Not enough memory for command
submission"); the desktop session ended, Tona survived. The reserve already exists in `ProgramBuilder::size_slots`
(`engine/gpu/mega_build_slots.cpp`: free VRAM per sysfs minus `TONA_VRAM_RESERVE_MB` minus pad minus the prefill arena); the
question is its default size.

Prior art (fetched 2026-11-15):
- llama.cpp `--fit` (on by default) keeps `--fit-target` free on each device, default **1024 MiB** per device
  (tools/server/README.md, `LLAMA_ARG_FIT_TARGET`).
- vLLM `--gpu-memory-utilization`, default **0.92** of the device (docs.vllm.ai, engine args): 8 % left, 1.3 GB of 16.3 GB; it
  is a fraction of the total, not a measurement of what the desktop uses.
- Ollama `OLLAMA_GPU_OVERHEAD` ("Reserve a portion of VRAM per GPU (bytes)", envconfig/config.go): default **0**, the user sets it;
  Ollama relies on a fixed per-library minimum plus this knob (internal minimum not verified here).
Reading: the common default for a desktop-shared card is 1 GiB or more; ours (640 MiB) is below it. Tona has the extra risk that
its slots are one allocation that is never given back at run time (until the hipMem part exists).

Proposal: default 1024 MiB (llama.cpp's value, equal to vLLM's 8 % rounded down), or 1536 MiB if the cost is within noise. Env override stays
`TONA_VRAM_RESERVE_MB` (document in docs/use/environment.md, with the cause).
Question and decision: what does the reserve cost in decode? Cost <= 2 % decode: take the larger one that stays within 2 %;
above 2 %: stop and ask the maintainer. Scope: `scripts/ab.py`, build `tiers`, 58 GB class (`TONA_RAM_GB=58`, `MemoryMax` 58 GB, profile
seeded), workloads `short,set`, 3 rounds alternating, production mode: run 1 A reserve 640 against B 1536; run 2 A 640 against B 1024.
Why this size: 1 GB of reserve is about 10 of the 86 slots (98 MB per slot over all layers), expected cost 0 to 2 %.
Expectation: decode B/A between -3 % and +1 % (the VRAM hit rate falls a little with about 12 % fewer slots; no earlier measurement of this lever, so the tolerance is my estimate); prefill -3 to +1 %. Tokens identical. Active tooling: `TONA_DECODE_DIAG=1`, `TONA_VRAM_LOG=1`,
`TONA_MEGA_INFO=1`, `TONA_EXPERT_PROFILE`. A result outside this is a finding.

## Step 3c result: page-cache floor sweep (26 GB class, 2026-11-15 18:02 to 18:31)

Raw: `ab-sw1` (floor 3.0 against 1.5), `ab-sw2` (1.5 against 0.5), `ab-sw3` (long, 130K prompt, 1.5 against 0.5), each file
with its plan. Build `tiers` at 3445216 plus the uncommitted `TONA_TIERS_MIN_FRACTION` change (no effect on these runs). Test conditions
(`tools/hw_report.py`, `tools/quiet_check.py` before and after, each run): the only warning was 1.03 GiB of 15.9 GiB VRAM held by
others (the desktop); CPU and RAM quiet, no swapping. The environment of every start is in the run files.

| Pair (B against A) | decode@2000 | decode@2000:write | decode@30000 | decode code / text (set) | prefill@2000 / :write / @30000 |
|---|---|---|---|---|---|
| floor 1.5 against 3.0 | +7.5 % [+6.6, +8.3] | +7.4 % | +5.0 % | +2.1 % / +2.9 % | +7.2 % (open) / +14.7 % / +5.9 % |
| floor 0.5 against 1.5 | +2.4 % | +7.2 % | +1.8 % | +0.9 % (open) / +1.7 % | +12.6 % (open) / +2.9 % / +3.7 % |
| long 130K, 0.5 against 1.5 | decode@130000 +2.0 % [+1.0, +3.1] | | | | prefill@130000 +3.6 % |

Pins (log line `experts per layer pinned`): floor 3.0 at ctx 32768 pinned 205 per layer (9,861, 20.8 GB), floor 1.5 at ctx 140000
pinned 199 per layer (9,589, 21.8 GB), floor 0.5 209 per layer (10,072, 22.8 GB). Each GB of floor given up is about 1 GB of pins,
about +2 to +7 % decode. Tokens identical in every pair (the SHA of each request equal in A and B and stable).
Stall counters: `yields` is 0 in every B start and every A start of the three runs (284 DIAG lines with 0, none with a higher value
in these runs), no `barrier aborted` in any log, including the 130K run at floor 0.5.

Against the plan: all rates were inside the -8..+8 % (-10..+10 % long) expectation except prefill@2000 (+12.6 % at 0.5 against 1.5, one
round off plan) and prefill@2000:write (+14.7 % at 1.5 against 3.0, a round outside the plan); the prefill rates at 2K are noisy
(rounds 631 to 840 tok/s) and the stall counters decide, as planned. Result: **no stall at 0.5 GB**, and a larger floor loses pins.
Decision by the plan's rule ("the lowest floor with no yield, no abort and no loss"): 0.5 GB would qualify on these runs, but they
cover two rounds and one long start; the coordinator chose 1.5 GB as the code default and this data does not contradict it
(the cost of 1.5 against 0.5 is 2 to 7 % decode). Not measured: concurrent memory pressure from the user's other programs;
the floor is what keeps their page cache alive. Open for the coordinator: keep 1.5, or take 0.5 with `TONA_TIERS_CACHE_GB`
documented as the lever.

## Step 3b result: partial pins at the 10 GB class (2026-11-15 18:31 to 18:46)

Raw: `ab-3b` (plan above). Quiet check: before 18:31 warning VRAM 1.0 GiB held by others; after 18:46 two warnings: 0.96 GiB of
VRAM held by others, and the system swapping (273 pages in 1 s) at the end (after the last server was stopped; no swap
within the servers, `MemorySwapMax=0`).

| Metric | A (nothing pinned) | B (profile pins, any share) | B vs A [CI] |
|---|---|---|---|
| decode@2000 | 44.0 / 43.6 / 43.9 | 44.7 / 46.5 / 47.5 | +5.5 % [+1.4, +9.6] |
| decode@2000:write | 59.4 / 59.1 / 58.6 | 56.5 / 56.9 / 57.4 | -3.6 % [-5.6, -1.6] |
| decode code (set) | 62.5 | 56.1 | **-10.4 %** [-11.0, -9.7] |
| decode text (set) | 36.1 | 35.9 | -0.1 % [-0.8, +0.6] |
| prefill@2000 / :write | 647 / 665 | 664 / 654 | +4.3 % / -1.3 % |

Expert hits per request: 2K A 84.8-85.5K hits / 67.3-68.0K misses, B 93.0K / 59.8K (VRAM hit rate +5 points); 2K write A 55K / 55K,
B 54.0K / 56.3K (no gain). Tokens identical. Against the plan (decode -5..+25 %): code -10.4 % is **outside the expectation, a
finding**: with 14 to 23 % of the experts pinned the pins take RAM from the page cache without lifting the VRAM hit rate enough
(the preload effect that gave +30 points at 80 % pinned gives +5 here), so code decode, which re-reads cold experts, loses.
Decision rule of the plan: B faster by more than 10 %? No (best +5.5 %, code -10.4 %). **Keep the 70 % rule; the 10 GB class gets
nothing from profile pins.** The step-4 and miss-count ideas stay open as listed in the plan (explicit cold reads, smaller
model) but the step-3 effect does not carry down to small budgets.

## hipMem probe result (run 2026-11-15 17:50, `bench/micro/vmm_probe.hip`, 1 GiB of the card's VRAM for a few seconds, no server)

VMM API supported on gfx1200 / HIP 7.16; granularity minimum 4 KiB, recommended 2 MiB. Reserve 1 GiB of address space 0.009 ms,
set access 0.132 ms. Per 32 MiB chunk (n=32): `hipMemCreate` 0.006 ms mean (max 0.028), `hipMemMap` 0.002 ms (max 0.006),
`hipMemUnmap` with the GPU busy 2.9 ms (n=16, max 2.97), `hipMemRelease` 0.007 ms. Card VRAM used: 1237 MiB before, 2261 after mapping
(+1024), 1967 after unmapping and releasing half (-480 of 512 MiB, sysfs figure). A kernel over the whole mapping summed correctly
(268,435,456 as expected); the lower half still read correctly after the upper half was unmapped; re-creating, mapping and setting access on
one chunk 0.035 ms. Against the plan (create/map 0.1 to 1 ms, unmap 0.1 to 1 ms, stall below 5 ms): create and map are 20 to 100 times
faster than expected, **unmap costs 2.9 ms per chunk with the GPU busy** (3 times the expectation, below the 5 ms limit; 16
chunks one by one take about 47 ms in total; whether the constant 2.9 ms is a per-call device sync that one batched unmap would pay once is unchecked). The
decision of the plan holds: the slot pool can become a reserved address range with mapped chunks, addresses stay fixed, VRAM
is returned (480 of 512 MiB seen at once, the rest within sysfs granularity). The kernel read the lower half on another stream during the unmaps (as planned) and the data stayed correct. Not measured: the
stall the persistent k_mega sees; that needs the engine, not this probe.

## VRAM reserve: result (58 GB class, 2026-11-15 18:59 to 19:23)

Raw: `ab-rs1` (A 640 MiB against B 1536 MiB) and `ab-rs2` (A 640 against B 1024), 3 rounds each, alternating, production mode,
`TONA_RAM_GB=58`, `MemoryMax` 58 GB, profile seeded; plans in the run files. Build `tiers`. Test conditions: hw_report in
`vmm.hw.md`-style header of each run; quiet_check before (state "do not measure": 0.96 to 1.03 GiB of VRAM held by others, the
desktop; no CPU or RAM finding) and after (same warning, 19:23). Another worker's builds and runs shared the queue, not the
GPU at the same time.

Slots (log line `VRAM:`): A 640 MiB 86 to 89 slots of 98.2 MB; B 1024 MiB 82 to 85; B 1536 MiB 76 to 80 (the free VRAM per
sysfs at start moved between 9.97 and 10.30 GB with the desktop's use, which is why the counts differ between starts). 1 GB of reserve is about 10 slots, 4 to 9
here.

| B against A | decode@2000 | decode@2000:write | decode code (set) | decode text (set) | prefill@2000 / :write |
|---|---|---|---|---|---|
| 1536 against 640 | -1.26 % [-1.9, -0.6] | -2.50 % [-3.7, -1.3] | -1.03 % [-1.7, -0.4] | -1.07 % [-2.0, -0.1] | -0.9 % / -2.3 % (both open) |
| 1024 against 640 | +0.41 % (open) | +0.75 % (open) | -0.67 % [-1.4, +0.06] | -0.63 % [-1.2, -0.03] | +15 % (one outlier round, 1013 vs 1465) / +1.0 % (open) |

Expert hits per request: A 134.6K / 18.3K misses (88 %), B 1024 133.2K / 19.7K, B 1536 130.8K / 22.1K. Tokens identical, no
yield in any start. Against the plan: 1536 inside the -3..+1 % expectation; 1024 inside on decode, prefill@2000 outside only through one
outlier round (A 1013 tok/s) which is noise of a known kind, not an effect of the reserve.
**Cost: 1024 MiB about -0.6 % decode (code and text), 1536 MiB about -1.0 % on the set and up to -2.5 % at 2K:write: both at or
below the 2 % limit; 1024 is well below.** Default set to **1024 MiB** (llama.cpp's `--fit-target`, vLLM's 8 % rounded). Not
measured and cannot be on this desktop: whether 1024 MiB is enough for the compositor (that was the balloon test, not repeated).
Arithmetic: the crash left 0.6 GB free and failed at +512 MiB; 1024 leaves about 1.0 GB at start, so a 0.5 GB second program
leaves 0.5 GB. This lowers the risk; it does not remove it; the run-time part (hipMem, returning slots) does. An override for
users with a bigger desktop load: `TONA_VRAM_RESERVE_MB`.

## Step 4 micro: what does an unmap cost, and what does a running kernel see (plan, written before the run)

Prior art first (fetched 2026-11-15; full list with status in the design note, docs/architecture/expert-tiers.md section 6):
[ROCm/TheRock #8744](https://github.com/ROCm/TheRock/issues/8744) (open; kernel-side root cause stated by a AMD contributor: after
`hipMemUnmap` the GPU TLB is not invalidated when VRAM is CPU-updated through Resizable BAR, a later map at the same address
reuses a stale translation; fix in review at amd-gfx, workaround: fresh address or keep the handle; reproduced on gfx1200 with
9 of 2100 iterations wrong for same address + new handle, 0 of 2100 for a fresh address or the same handle) is the hazard for any
design that returns slots and maps them back at the same address.
Question 1: is the 2.9 ms per `hipMemUnmap` of the first probe a device sync (the probe launched a kernel before each unmap), and does one call over
many chunks cost less? Question 2: what does a spinning persistent kernel that reads the kept chunks see while the others are
unmapped (longest gap between its iterations)? Question 3: on this kernel (7.2.15, in-tree amdgpu) does remapping the same address
with a new handle corrupt data? Decisions: unmap idle below 0.3 ms per chunk and no kernel gap above 1 ms: the pool can give slots back
at a safe point between passes without a yield; unmap waits for the device: do it only while k_mega is parked (the yield after
100 ms, or between requests); the call blocks the host for the rest of the running kernel, never call it while k_mega waits on the
CPU. Corruption at the same address: slots must come back at a fresh address (the pool is a ring of reserved ranges and the task
addresses are patched) or the pool must never release handles (no VRAM returned): the design changes. No corruption in 400
iterations does not prove the fix (the issue's rate is 0.4 %: 400 iterations would show about 1.7; zero has a probability of
about 19 % at that rate), so the result is only evidence.
Scope: `bench/micro/vmm_stall.hip` (new, standalone; 1 GiB of VRAM for a few seconds, no server, no second process; the spinning kernel
stops by its own wall clock after 60 ms and never waits for the CPU, so an unmap that blocks until idle cannot hang the GPU, and
no VRAM beyond Tona's own budget is used). E1 16 unmaps of 32 MiB, GPU idle, one by one and as one call over the range; E2 the
spinning kernel (4 blocks reading one word in each of the 16 kept chunks per iteration and verifying it), baseline without unmaps
and with the 16 unmaps starting after 10 ms, twice each; E3 400 times release, create, map at the same address, write a pattern,
read it back, and check the kept half. One run, qualitative plus per-call times. Active tooling: none.
Expectation: E1 idle unmap 0.05 to 0.5 ms per chunk (create and map were 0.006 and 0.002 ms); one call over a range may be
refused (HIP requires the range of one map) and then the answer is "no batching". E2 baseline longest gap below 0.05 ms (own estimate);
with unmaps the first unmap takes up to the 50 ms left of the kernel if unmap waits for the device, later ones short, and the longest
kernel gap above 1 ms only if the unmap stalls the compute queue. E3 0 to 3 wrong. A result outside this is a finding.

## Step 4 micro: result (2026-11-15 19:29 to 19:40, `bench/micro/vmm_stall.hip`)

Raw: `/var/tmp/claude/148/res/stall.out` (default run: E1, E2, E3 same handle-new), `stall2.out` (variants 1 and 2 of E3). Build `hipcc -O2
--offload-arch=gfx1200`, kernel 7.2.15, ROCm 10.1 (HIP 7.16). Test conditions: hw_report in `stall.hw.md`; quiet_check before: load average 4.65
(other workers' builds) and 0.96 GiB of VRAM held by others, after: the same; the micro uses its own 1 GiB and no other process on the GPU
(the queue held the GPU). No VRAM pressure was put on the card.

| Experiment | Result |
|---|---|
| E1 idle, `hipMemUnmap` per 32 MiB chunk (16) | mean 0.007 ms, max 0.024 ms, total 0.11 ms |
| E1b idle, one call over the range of 16 chunks mapped by 16 calls | no error, 0.065 ms (batching works) |
| E2 baseline spinning kernel (4 blocks, reads the 16 kept chunks, 60 ms, own clock) | longest gap 0.011 ms, 0 gaps above 0.5 ms, 0 wrong reads (two runs) |
| E2 with 16 unmaps of the other half starting after 10 ms | **first unmap 50.4 ms** (until the kernel ended), the other 15 about 0.1 ms; mean 3.15 ms per unmap; longest kernel gap 0.026 and 0.010 ms, 0 wrong reads |
| E3 new handle, same address, 400 times | **wrong in 29 of 400** (7 %); kept half intact |
| E3 variant 1: new handle, fresh address each time | wrong in 0 of 400; all addresses distinct |
| E3 variant 2: same handle mapped again at the same address | wrong in 0 of 400 |

Against the plan: E1 0.007 ms against 0.05 to 0.5 ms expected (faster: the finding of the first probe, 2.9 ms, was the kernel launched
just before the unmap); one call over a range accepted (the plan expected a possible refusal); E2 baseline gap 0.011 ms (below the 0.05 ms
estimate); with unmaps the call waited for the kernel (the plan's second branch): **`hipMemUnmap` blocks the host until the device is
idle**, so the kernel itself sees no stall, the host sees all of it; E3 **29 of 400 against 0 to 3 expected: outside the expectation,
a finding**: the stale-translation bug of ROCm/TheRock #8744 is present on this kernel and hits whole 32 MiB chunks. The first run
(default) wrote through possibly stale translations into released pages, about 460 chunk writes (400 in E3, the refills of E1 and E2); the pages were
given back to the driver and could have belonged to another program (no glitch seen). Mitigation variants that would write
through stale translations again (delays, flushes) were therefore **not run**; only variants safe by construction (fresh address, same
handle) were. Consequence in the design note (section 5a): an index range that was unmapped is never mapped again.

### Step 4 micro, follow-up (same day): the same-address variants are behind a flag, and a large address reservation works

The default run wrote through stale translations about 460 times (E3 400 plus the refills of E1 and E2) with no visible damage on the
desktop. Since then E1, E2 and E3 variant 0 (everything that maps an address again with a new handle) run only with
`--unsafe-same-address`, with a comment naming ROCm/TheRock #8744 at the top of `bench/micro/vmm_stall.hip`; the default run does not
write through any address that was unmapped. Default run (`stall3.out`, 2026-11-15): same handle at the same address 0 of 400 wrong, new
handle at a fresh address 0 of 400 wrong (all addresses distinct), and E4: `hipMemAddressReserve` of 64 GiB, 512 GiB and 4096 GiB
succeeds in 0.005 to 0.009 ms and a 32 MiB chunk mapped at the very end of each is read and written correctly. The slot index is
`int32` and the kernels multiply it by a 64-bit stride, so the address space, not the index type, limits the number of fresh generations
(design note 5a).

## Step 4 build: governor checks and a fake-free trace run (plan, written before the runs)

Built: `TONA_VRAM_GOVERNOR=1` (off by default), elastic slot generations, `TONA_VRAM_FAKE_FREE`, hard never-remap guard (design note 5a).
No second program takes VRAM in any of these runs (the coordinator's rule after the 2026-11-15 desktop crash).

Q1 (bit-exact with the governor off, rules 1 and 3 of AGENTS.md): after the change a default start must produce the tokens of
before: `scripts/golden.sh compare` (gpu, and `GOLDEN_PLAN=cpu`) on the new build, plus the selected checks of
`tools/select_checks.py`. Decision: any difference blocks the merge, nothing else is measured before it is fixed. Expectation: identical.

Q2 (cost of the governor idle, decode and prefill): `scripts/ab.py` A default, B `TONA_VRAM_GOVERNOR=1`, `TONA_RAM_GB=58`, `MemoryMax` 58 GB,
profile seeded, workloads `short,set`, 3 rounds alternating, production mode; B's governor reads the real card (about 1 GB free,
above its shrink mark of 768 MiB, so no shrink is expected; the log lines are counted and a round with a shrink is reported as such).
Why this size: the same as the reserve runs (3 rounds resolve 1 %). Expectation: decode and prefill B against A between -2 % and +1 %:
the core is the same slot count minus about 2 slots for the granule alignment of the elastic blocks (about -0.3 % from the reserve runs: 4 slots
cost 0.6 %), the idle tick is two sysfs reads per 250 ms. Tokens identical. A result outside this is a finding. Cost above 2 %
decode: stop and report.

Q3 (shrink and regrow, tokens identical, decode cost per phase): `/var/tmp/claude/148/gov_run.sh`: server as above with
`TONA_VRAM_GOVERNOR=1` and `TONA_VRAM_FAKE_FREE=res/fake-free-1.txt` (free MiB per second: 0 s 3000, 40 s 300, 80 s 100, 130 s 3000; the
clock starts at the first tick, about when the server is ready), a decode loop of 300-token requests (a different prompt per request,
temperature 0) for 270 s, one line per request with its SHA, the decode rate and the card's VRAM used. The same loop with the governor off
for the same time gives the reference SHAs. Expectation: at 40 s two generations are given back (free 300 MiB needs 724 MiB: about 2
generations of about 430 MB; card VRAM used falls by about that much), at 80 s the rest (emergency, 4 generations in all, so the
core only), at about 190 s the regrow starts (calm 60 s after the second shrink within 60 s of the first; the doubling window) and
brings one generation per 5 s back; decode in the phases: all generations 88 to 99 tok/s (the reserve runs), core only 3 to 5 %
lower than with all generations (4 generations are about 16 of 86 slots; 1 GB is -1 to -2.5 %), after the regrow the slots are empty
and fill by misses, back to the first level within a minute. Tokens: every request's SHA equal to the governor-off run for the same
n. No yield, no abort, no refused write (`refused` in the log), the sysfs free value of the card rises at the shrinks and falls at the
regrow. A result outside this is a finding.
Active tooling in all runs: `TONA_DECODE_DIAG=1`, `TONA_VRAM_LOG=1`, `TONA_MEGA_INFO=1`, `TONA_EXPERT_PROFILE`.

## Step 4 build: results (governor branch, 2026-11-15 to 2026-11-16)

Builds: `tiers` at the commits named in the run files (v0.0.2-856 and later). Test conditions per run in the run files
(`g-*.hw.md`, `quiet-before/after`); where a run was disturbed this is said.

**Q1, bit-exact with the governor off:** `scripts/golden.sh compare` (gpu plan) identical in every prompt, `GOLDEN_PLAN=cpu` identical
(both on the build that contained the whole governor code except the last three small commits; the final golden run on the last build is
queued, `gv3-golden.log`). `ab.py` with `TONA_DETERMINISTIC=1`, A default and B `TONA_VRAM_GOVERNOR=1` (the governor active: core 55
slots, 4 generations of 4 slots, 491 MB each): **SHA equal in A and B and stable for 2K, 2K write and 30K prompts**, the 30K SHA equals
the reference of main, hits and misses per request identical.

**Q2, cost of the governor idle** (`ab-gv-cost`, A default, B `TONA_VRAM_GOVERNOR=1`, 58 GB class, 3 rounds, build before the
alignment fix, so B had one slot more than the final code): decode 2K -0.78 % [-1.96, +0.40], 2K write +0.18 % [-1.59, +1.96], code
-0.30 % [-1.14, +0.55], text -0.46 % [-1.22, +0.30]; prefill 2K +0.03 % (write); prefill@2000 -5.1 % through one outlier round (-15 %,
the other two +/-0.5 %). All inside the plan (-2 to +1 %): **the idle governor costs 0.3 to 0.8 % decode**; tokens identical; the VRAM
hit rate per request is 1.1 % lower (128,505 against 129,884 hits). Governor lines in the log: core 55 slots, 4 generations of 491 MB, 1097
more address ranges fit.

**Q3, fake-free trace (`g-fake`, `g-ref`; trace 0 s 3000 MiB, 40 s 300, 80 s 100, 130 s 3000; decode loop of 300-token requests with
a different prompt each):** the governor gave back **2 generations at 40.3 s (973 MB) and the other 2 at 40.6 s (984 MB)**: the
fake value does not rise when memory is released (a real card's free value would), so the second sample still saw 300 MiB and took the
rest at once; a trace that follows the release would stop after the first step. The card's VRAM used (sysfs) fell from 15,151 to 13,285 MiB
(**-1.87 GB, real**, seen at the next request); at 190.6 s and 195.8 s, after the calm time of 60 s (doubled by the second shrink within 60 s
of the first, as designed), two generations were **taken again** (+466 MB in the card's used value after the first, 1 live then 2 live,
1096 and 1095 address ranges left). No yield, no abort, no refused write in the log, no `barrier aborted`. **Tokens: 49 requests in
both runs, 0 SHA mismatches against the governor-off run for the same prompts**, before, during and after the shrink and the regrow.
Decode per phase (mean of the requests starting in the window): 0 to 40 s 75.0 tok/s, 40 to 130 s (core only) 71.7, 130 to 190 s 79.0, from 190 s
76.2 (one request); the governor-off run in the same windows 89.4 / 89.1 / 90.5 / 89.3. **These absolute rates are not valid**: the quiet check
before the governed run said "do not measure" (other programs used 101 % of the CPU threads, load average 23.9, swapping), the planner
moved the expert work from the CPU to the GPU-direct path (cpu_experts 19 per pass against 155 in the reference start, 30.2 s against
12.4 s for pinning), and the first phase alone is 16 % below the reference although nothing had been shrunk yet. Within the governed run the core-only phase is
4.4 % below the all-generations phase (71.7 against 75.0), the same order as 4 generations (16 slots, 1.6 GB) cost through the static reserve
(1 to 2.5 % per GB), but with the load as a confounder. A repeat on a quiet machine (reference and governed run back to back, each after a
quiet check) is queued (`gov-both`, `g-ref2`, `g-fake2`).

## TONA_TIERS on by default: guards and the 58 GB class (plan, written before the runs)

Decision of the maintainer (2026-11-15): `TONA_TIERS` defaults to on (`TONA_TIERS=0` switches it off); the 70 % rule leaves classes
without a gain unchanged (10 GB: nothing pinned, step 3b). Branch `148-tiers-default` off main ce565fd, one code line
(`tiers_on` in `engine/gpu/mega_experts.cpp`) plus documents.

Guards: (1) the five goldens of `scripts/gpu-status.sh` (gpu, cpu plan, prefill, prefill dflash, server) on the new build, run
from the main checkout with `--head <sha> --build work/build/tiers-d --keep-going`, no post; (2) unit tests (`just test-cpu`) and
`just check`; (3) what the earlier gates showed on the governor branch (golden gpu and cpu identical, `ab det` SHAs equal) is quoted in
the report, not counted as a guard of this branch.
Question of the run: does anything get slower at the 58 GB class (reference default, all experts fit, so the tiers code should
change nothing there)? `scripts/ab.py`, A `TONA_TIERS=0`, B the new default (no variable), `TONA_RAM_GB=58`, `MemoryMax` 58 GB, profile
seeded (the profile is what activates tiers), workloads `short,set`, 3 rounds alternating, production mode, build `tiers-d` for both
arms. Why this size: 3 rounds resolve about 1 % (reserve runs). Expectation: B against A decode and prefill between -1 % and +1 %
(all experts pinned either way; the code differs only in the page-cache floor and KV accounting that do not bind when everything
fits), tokens identical, the log line `experts per layer pinned` equal in A and B. Decision: a loss above 2 % or different pins
at the 58 GB class: stop and report, the default stays off there. Active tooling: `TONA_DECODE_DIAG=1`, `TONA_VRAM_LOG=1`,
`TONA_MEGA_INFO=1`, `TONA_EXPERT_PROFILE`.

### Result: 58 GB class, A `TONA_TIERS=0` against B the new default (3 rounds, `/var/tmp/claude/148/res/ab-default`)

Build `tiers-d` (commit aecbaa0 plus the plan commit), both arms the same binary. Test conditions: `hw_report` and `quiet_check` in the run
files (the only warning of the earlier runs, 0.96 to 1.03 GiB of VRAM held by the desktop; the load of other workers' builds was kept
off by the queue). Pins: `256 experts per layer pinned` (all of them) in A and B, tokens identical (SHA equal and stable), **expert
hits and misses per request identical in A and B** (129,884 / 22,965 and 83,132 / 27,145 in every round): at this class the
default changes no decision of the engine.

| Metric | A, per round | B, per round | B vs A, mean [CI] |
|---|---|---|---|
| prefill@2000 | 1413.6 / 1438.2 / 1437.7 | 1410.5 / 1415.5 / 1447.1 | -0.38 % [-1.73, +0.96] (equal) |
| decode@2000 | 71.3 / 71.1 / 70.8 | 69.5 / 71.2 / 70.8 | -0.78 % [-2.57, +1.00] |
| prefill@2000:write | 1700.1 / 1712.6 / 1714.6 | 1693.6 / 1706.9 / 1721.4 | -0.11 % [-0.69, +0.48] |
| decode@2000:write | 113.4 / 114.9 / 114.4 | 112.4 / 113.3 / **98.7** | -5.31 % [-14.0, +3.4] |
| decode code (set) | 86.5 / 88.1 / 88.1 | 88.0 / 88.2 / 88.3 | +0.69 % [-0.24, +1.63] |
| decode text (set) | 63.4 / 64.9 / 64.8 | 64.8 / 64.8 / 64.6 | +0.58 % [-0.71, +1.87] |

Against the plan (-1 to +1 %): everything inside except decode@2000:write, whose third round of B (98.7 against 112 to 113; -13.7 %, marked
outlier by the tool) pulls the mean to -5.3 %; without that round B is -1.1 % (-0.8, -1.4), which is about 2 sd of the known noise of that
metric (0.6 %). Since pins, hits and misses are identical in A and B, no code path differs at this class, and the single slow round is
a disturbance of the run, not an effect of the default; the decision rule of the plan (a loss above 2 % or different pins: stop) is
met on the counters, not on that one metric. Open for the coordinator: a repeat of decode@2000:write (4 rounds, 14 min of queue time) if
that is wanted before the merge.

### Goldens for the tiers default (`scripts/gpu-status.sh --head <sha> --build work/build/tiers-d --keep-going`, no post)

Exit 0: golden gpu, cpu plan, prefill, prefill dflash and server all identical (every prompt "identical", no difference or failure line;
log `/var/tmp/claude/148/res/gs-default.log`). Unit tests `just test-cpu` exit 0, `just check` exit 0 (tidy ok) on the branch.

## Section 5b micros and the passive VRAM log (2026-11-16)

**Plan.** Question 1: can MADV_FREE data be trusted with "copy, then check the magic word of the source" (CPU only, `bench/micro/madvise_free.cpp`;
expectation: 0 valid-but-wrong pages, some pages lost under pressure, `mincore` unreliable). Question 2: what pinned host memory allows
(`bench/micro/host_pin.hip`, host memory only, no kernel, no VRAM; run through the GPU queue, once with `HSA_USERPTR_FOR_PAGED_MEM=0` and once
unset): file-backed `hipHostRegister` accepted and unreclaimable afterwards, register/unregister cost of a 32 MB piece, a `hipHostMalloc`
piece freed alone. Decision: the soft-warm/pinned split of 5b stands if file-backed registration pins the pages and a piece can be freed on its
own; otherwise only the page-cache half of the design remains.

**Passive VRAM log (burst distribution of the desktop, for the reserve decision of 5b).** `/var/tmp/claude/148/vram_passive_log.py`, started with
`nohup setsid systemd-run --user --scope -p MemoryMax=100M ...`; one line `epoch_ms used_MiB` every 100 ms (sums of the card, no process
names, no GPU call), one file per day `/var/tmp/claude/148/vramlog/vram-<date>.txt`, stops by itself after 7 days. PID in
`/var/tmp/claude/148/vramlog/pid`; stop it with `kill $(cat /var/tmp/claude/148/vramlog/pid)`. Output is buffered 64 KiB (about 5 minutes). Started
2026-11-16 07:0x, PID 2147606. **Stopped by the coordinator on 2026-11-16 and not analysed**: one desktop measures one user; section 5b now designs a reserve that calibrates itself on each machine instead. Server runs of the workers show up as steps in the
log (they are not separated out), so the analysis takes the rises while no run is queued (queue times from the gpu_queue log).

**Result 1 (madvise_free, 128 MiB, MemoryMax 300 MiB, swap off):** E1 without pressure all 32,768 pages valid; E2 under pressure 262,144 page
reads, 254,688 valid by copy-then-check, 7,456 lost, **0 valid-but-wrong**; E3 check-then-copy: 0 bad copies (a race, no evidence); E4 freed pages read
as zero without a fault (7,522), a page written after the advice stays valid, **mincore reports all pages resident after the reclaim**: not usable as a check.

## All slots from 32 MB chunks (step 4b, plan written before the runs, 2026-11-16)

Background (#157): the slots and the prefill workspace are one 8,370 MB `hipMalloc` (`mega_build_slots.cpp:91`); TTM evicts whole buffer objects, so a
0.5 GB desktop request cost Tona 8.4 GB. The elastic pool of step 4 maps only the last 1.5 GB in small blocks; the 7 GB core stays one BO. Goal of
this step: build **all** slots, core included, from 32 MB chunks (`hipMemCreate` each, one contiguous reserved address range, never the same
address again, kernels unchanged, bit-exact tokens), with the prefill workspace chunked or separate, as the base for #157's give-back.

**Micro first** (`bench/micro/vmm_chunks_build.hip`, 260 chunks of 32 MB = 8.7 GB, no pressure, the card otherwise unused, through the GPU queue). Questions: (1) is every
`hipMemCreate` its own allocation (VRAM used +32 MiB per create+map, -32 MiB when one handle is released); (2) what do 260 of them cost at build
time against one `hipMalloc`; (3) does one kernel fill and check the contiguous range unchanged. Expectation: +32 MiB per step, create+map about 0.04 ms per
chunk (earlier probe: `vmm_probe`), so 10 to 40 ms for 260 against a few ms for one `hipMalloc`; 0 wrong words; release of one chunk -32 MiB in under 1 ms.
Decision: all three as expected -> build the slots chunked (guards below). VRAM falls by other than 32 MiB per chunk (a chunk is not its own BO), or build time above
1 s -> stop and report. Active tooling: none beyond the micro. Test conditions: `hw_report` and `quiet_check` in the run output of the queue job.

**Guards for the build** (as for the governor): `golden.sh compare` and `GOLDEN_PLAN=cpu` identical; `TONA_DETERMINISTIC=1` A/B tokens identical (ab.py short,mid; A =
one allocation, B = chunked, switch `TONA_SLOTS_CHUNKED`, off by default until measured); speed A/B on the 58 GB class (decode and prefill expected within -1 to +1 %: the
kernels see the same addresses, only the mapping differs; TLB effects of 32 MB chunks against one big allocation are the risk: the micro reports kernel time over the range).

**Additions to the chunk plan (numbers from #157, 2026-11-16):** the same micro at 32 MB gave about 22 ms for 260 chunks and +280 KFD ALLOC ioctls (one BO per chunk); recommended granularity 2 MiB. Two things the first chunked run of this branch must measure: (1) the lag between `hipMemRelease` and the fall of `mem_info_vram_used` (it stayed 8.6 GB high right after a release in #157's run: the governor must not take VRAM for free the moment a release returns), logged as the time until the sysfs value falls; (2) the KFD restore cost with about 260 more BOs (`restore_process_bos` locks all of them): the decode rate and the `evicted_ms` of the chunked run against the one-allocation run. Step 5's "pin only the hot set" is what makes real GTT room (headroom is 1.15 GB at the 58 GB class today).


## Final-binary results of the governor (gv4, 2026-11-16, build `tiers` after the merge of main 3f987d6 and the from_reserve port)

Conditions: quiet check before every run "warning" only (2.22 GiB of VRAM held by the desktop, no other load; the earlier disturbed run is superseded). Build exit 0.
Core 49 to 52 slots, 4 elastic generations of 4 slots (about 493 MB each), 1,100 address ranges left.

**Q1 bit-exact.** `golden.sh compare` identical (csv, rust, explain; exit 0) on this binary with the governor off. `ab.py` with `TONA_DETERMINISTIC=1`, A default,
B `TONA_VRAM_GOVERNOR=1`: **SHA equal in A and B and stable for 2K, 2K write and 30K**; the 30K SHA equals the reference of main (ee58880d3f5b558f).

**Q2 cost with the governor on but idle** (`ab-gv-cost2`, 58 GB class, 3 rounds alternating, plan -2 to +1 %): decode@2K +1.27 % [-0.01, +2.55], @2K write +3.55 %
[-0.69, +7.80] (round 3 -0.3 %: the first two rounds are the noisy metric), code **-0.86 %** [-1.55, -0.17], text **-1.04 %** [-1.97, -0.10];
prefill@2K +0.11 %, @2K write **-0.31 %** [-0.62, -0.01]. All inside the plan; the three significant differences are the cost of 4 generations less VRAM-resident
(hit rate -2 % at 2K: 125,940 against 128,514 hits). Tokens identical. **The idle governor costs 0.3 to 1.0 % decode and 0.3 % prefill.**

**Q3 fake-free trace on a quiet machine** (`g-ref3`, `g-fake3`: 300-token requests, trace 0 s 3000 MiB free, 40 s 300, 80 s 100, 130 s 3000; 84 and 83 requests):
the governor gave back **one generation per poll at 40.2, 40.4, 40.7 and 41.0 s** (495, 491, 489, 489 MB; the free value in the fake trace does not rise,
so it went on to the floor; core only afterwards), the card's VRAM used fell from 15,147 to 13,275 MiB (**-1.87 GB, real**), and after 120 s of calm it **took a
generation back at 160.7, 165.9, 171.2 and 176.2 s** (5 s apart, to the full pool). **0 SHA mismatches in 83 common requests**, no yield, no refused write, no abort.
Decode (mean of the requests starting in the window; reference / governed): 0 to 40 s 92.6 / 92.3 (-0.3 %), 40 to 130 s (core only) 92.6 / **89.4 (-3.5 %)**,
130 to 190 s 94.6 / 92.1 (-2.6 %, the regrow phase), from 190 s 92.7 / 92.5 (-0.2 %). So the governor holds the rate within 0.3 % while whole, loses 3.5 % decode while it
has given back the 4 generations (1.97 GB), and is back to -0.2 % after the regrow.

**Section 5b micros (`host_pin`, host memory only, run with `HSA_USERPTR_FOR_PAGED_MEM=0` and unset).**
(a) `hipHostRegister` of a file-backed `mmap` range is **accepted, but does not pin the pages**: `Unevictable` and `Mlocked` stay 0 and `MADV_PAGEOUT`
takes all 16,384 resident pages away, exactly as for an unregistered control. Registered memory is therefore reclaimable, with a notifier into the driver
(the userptr path of `gtt.cpp`: the kernel may take the pages and the process's queues are evicted for it). A registered soft tier is **not** a safe soft tier
for a persistent kernel; the soft-warm tier of 5b must be read by the CPU path only, unregistered. (b) Registering 32 MB of anonymous memory costs 0.5 ms,
unregistering about 0, and it does not pin either (GTT +0, Unevictable +0). (c) With `HSA_USERPTR_FOR_PAGED_MEM=0` (the engine's setting) a
`hipHostMalloc` piece of 32 MB is GTT memory: 8 pieces took 23.5 ms (GTT +256 MB, MemAvailable -257 MB) and **one piece freed on its own in 0.35 ms returned 32 MB to GTT
and 31 MB to MemAvailable**: pieces can be released one by one, cheaply, which is what the reactive pin release of 5b needs. Unset (userptr path) a freed 32 MB piece
returned nothing to MemAvailable at once (2.1 ms): not the path for a quick release.

## Chunked core and long prefill: results (read 2026-11-16 during the pause; all runs had finished before it)

**Micro `vmm_chunks_build`, 8.72 GB from chunks of 2 to 64 MiB** (granularity: min 4 KiB, recommended 2 MiB; `chunks_sweep.out`): create+map+access of the whole
range **173, 119, 182, 158, 153, 147 ms** for 2, 4, 8, 16, 32, 64 MiB (4160 down to 130 chunks): the build cost does not depend on the chunk count in this range,
about 0.15 s in total. One kernel fills and checks the contiguous range in 68.2 to 69.3 ms for every size, **0 wrong words**, so chunking costs a streaming pass nothing. Releasing one
chunk takes 0.07 to 0.09 ms, all of 259 chunks 2.1 ms. The sysfs VRAM used after a release lags: successive runs started 5 GB "below" the previous run's end, so freeing is deferred
(consistent with #157's 8.6 GB): the governor must not take VRAM for free when a release returns. Decision of the sweep: any size from 2 to 64 MiB builds in the same time, so the
chunk size is not constrained by build cost; 32 MiB stays as the named constant (a fixed mechanism granule, chosen because the cost is flat and a chunk is then one BO of a size comparable to one elastic
block), the derived rule of `slots_chunk_bytes` may be dropped in favour of it.

**Chunked core in the engine (`TONA_SLOTS_CHUNK_MB=32`, 256 chunks of 33.5 MB for the 8,569 MB core), 58 GB class.** Deterministic A/B (one allocation against chunks): **SHA equal and stable for 2K, 2K write and
30K, 30K equals the reference of main** (`ab-chunks-det`). Speed A/B, 3 rounds, expectation -1 to +1 % (`ab-chunks-speed`): decode@2K -0.38 %, @2K write **-1.06 %** [-2.13, +0.02], code +0.43 %,
text +0.59 %; prefill@2K -0.66 % (one round 1422 against 1458: noisy), @2K write +0.15 %. All inside the plan except the 2K-write decode, which is at its edge and whose interval touches zero; tokens identical.
`golden.sh` on the chunks build did not run (the build was older than its sources, exit 2). KFD restore cost with the extra BOs and the evicted_ms were not compared in this run.

**Long prefill with the governor (`gl-ref`, `gl-fake`: one prompt of 133,390 tokens, trace free 3000 / 300 MiB at 30 s / 3000 at 60 s).** Governor off: 74.3 s. Governor on: **gave back 2 generations at 31.0 s
and 2 at 34.5 s, in the middle of the prefill (the tick between prefill groups works)**, the card's VRAM used fell 15,160 -> 13,303 MiB (-1.86 GB), prefill 76.6 s (**+3.1 % time**, with the pool smaller for 40 s of
the run). **The answer's SHA is identical (0df59fd6c41c).** Quiet check before: warning only (desktop VRAM).

**Merged head dc14e1e (main 63e420c merged):** build exit 0; `just test-cpu` exit 0; `just check` exit 0 (tidy ok, 24 files); **C1: all kernels identical** (264 functions, `scripts/isa-diff.py` against a build of main 63e420c). The five goldens did not
run: `gpu-status.sh` refused (run from the main checkout whose HEAD is ahead of origin/main); the next run needs `--own` or an updated base checkout. Pause by the maintainer from this point on.

## KFD restore cost with the chunked core (plan, 2026-11-16, written before the run)

Question: do about 260 more buffer objects change how long the KFD takes to evict and restore the process's buffers (`restore_process_bos` locks all of them)? Run: the 58 GB class
server, 150 s decode loop of 300-token requests (`gov_run2.sh`, the merged build), A one allocation, B `TONA_SLOTS_CHUNKED=1`, one run each, `evicted_ms` of the server's KFD process printed
per request, decode rate per request. Expectation: no evictions after the load in either run (the card is not under pressure), `evicted_ms` equal in A and B within the load-time evictions
that exist today (about 2.6 to 9 s per run, `svm`-triggered during the load, #157), decode equal within 1 %. Decision: B's `evicted_ms` more than 2x A's, or decode more than 1 % lower: report; otherwise the chunked pool is cleared for its own
step. Caveat named in advance: without pressure there is no eviction in the request phase, so this measures the load-time restores only; a restore under real pressure would need the pressure test that is forbidden.

### Result: KFD restore cost with the chunked core (`g-chk0` one allocation, `g-chk1` `TONA_SLOTS_CHUNKED=1`, 47 requests each, quiet check "warning" before both: desktop VRAM only)

Against the plan: **tokens identical in all 47 requests; decode 93.84 (A) against 93.52 tok/s (B), -0.34 %** (limit 1 %); `evicted_ms` of the server's KFD process: A 3,334 ms at the first request and 3,535 ms at the
last (+201 ms during the loop), B 2,448 ms and constant (the load-time evictions are of the same order, B lower); card VRAM used 15,294 against 15,312 MiB (+18 MiB for 256 handles). So **256 more buffer objects did not make the
eviction or restore time worse** in this run (B's `evicted_ms` is below A's, far under the 2x limit). The caveat stands: no pressure, so this covers the load-time restores only; the pinned build took 12.8 s (A) against 10.8 s (B). One run each, so a spread is not available;
the difference in decode is inside the 0.3 to 1 % noise of earlier runs.

### Final guards of the merge head 4919b35 (reported by the coordinator, queue run `final8`)
Unit tests 62/62, `just check` with tidy ok, the five goldens (gpu, cpu plan, prefill, prefill dflash, server) identical including the long one, `isa-diff` all kernels identical against main 63e420c.
The pressure series that `gpu-status.sh` asks for is the VRAM pressure test that the maintainer has forbidden; it is not run (see the PR text).
