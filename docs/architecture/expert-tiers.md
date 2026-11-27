# Expert tiers: placing experts by usage across VRAM, pinned RAM and SSD

Rule: the least used experts go to the SSD, on every backend, and placement follows the free RAM while other programs come and
go. Status 2026-11-25 (code read, not rerun); **estimate** and **unchecked** mark what is not measured.

| Part | State |
|---|---|
| Usage profile per model, global ranking, partial pinning from 70 % of the expert bytes (`engine/backend/expert_tiers.h`, `budget.h`; `TONA_TIERS`, on by default) | built, measured (section 8) |
| Pieces of 32 MB and the table `place[layer][expert]` (section 4) | not built |
| Explicit cold reads with prediction lookahead (section 4) | not built; io_uring serves the PLE rows only (`engine/io/uring_pread.cpp`) |
| RAM governor on the PC (section 5) | not built for the pinned tier; the state machine `engine/backend/ram_governor.h` exists with unit tests |
| VRAM give-back (section 5a) | built behind `TONA_VRAM_GOVERNOR` (off); chunked slots behind `TONA_SLOTS_CHUNKED` (off) |
| Soft tier, no idle memory (section 5b) | design only |

## Context

The Coder (IQ1_M: 48 layers x 256 experts of 2.05 MB, 25.15 GB) keeps experts in a per-layer VRAM slot cache (about 84 of 256
slots, LRU/ClassLRU, seeded from the profile), a pinned RAM stage (`Mega::pin_experts`) and the model file (page cache and SSD,
read in place by the CPU). `budget.h` pins "all or none". Decode on the reference PC before the 9950X
(`research/measurements/2026-11-16-pc-dm-vs-coder.md`, short code / long decode): 64 GB class 109 / 77 tok/s, 32 GB class (26 GB
budget) 63 / 42, 16 GB class (10 GB budget) 10 / 14.

## 1. Prior art

Usage-ordered caching with the SSD as bottom tier: MoE-Infinity ([2401.14361](https://arxiv.org/abs/2401.14361)). Prediction and
prefetch: ProMoE ([2410.22134](https://arxiv.org/abs/2410.22134); our next-layer prediction already catches 85.5 % of misses in the
top 24, `2026-10-24-expert-prediction.md`), HOBBIT ([2411.01433](https://arxiv.org/abs/2411.01433); its mixed precision is lossy and
out of scope). `io_uring` + `O_DIRECT` cold reads with LFU: llama.cpp PR 31469 (closed, unmerged). Policy replay: COPE
([2610.33404](https://arxiv.org/abs/2610.33404)), our method for tier sizes. Layer-aware caches and prefetch harm:
[2509.02408](https://arxiv.org/abs/2509.02408), [2511.05814](https://arxiv.org/abs/2511.05814). Precision under pressure: DynaExq
([2511.15015](https://arxiv.org/abs/2511.15015), lossy, text unread). No work combines usage-ordered pinning with a cgroup-aware
budget that follows other programs (research/44, search of 2026-11-15).

## 2. Why partial pinning lost on 2026-10-27

`pin_layer` pinned the same n experts per layer and left the **hot head** of the usage order unpinned, so the page cache had to
carry exactly the experts evicted from VRAM that come back all the time. Every bad run had page-cache room below the size of that
head ([2026-10-27-ram-limit.md](../../research/measurements/2026-10-27-ram-limit.md): 26 GB limit with 20.6 GB pinned and a 4.6
GB head: decode 20.9; 17.0 GB pinned with a 3.5 GB room: aborts; 21.7 GB pinned with 4.3 GB room: decode 51.2). **Ungeprueft**
whether the profile was loaded in those runs. The cause does not repeat if (a) the order is by usage with VRAM-resident experts
accounted for, (b) page-cache headroom stays above a floor (`CACHE_MIN_GB`), (c) cold misses are read by explicit I/O, not by
faulting in the engine thread. (c) is confirmed by phase B step 4, **ungeprueft** until then. Aborts are fixed by the 100 ms yield.

## 3. Usage skew of the Coder

Source: `work/profile/experts-coder.bin` (1,920 counted tokens per layer, decode with coding prompts plus one Claude Code request;
split-half shares within 0.7 points). Other workloads are **not measured** (a prefill mix changes the ranking,
`2026-11-11-requant.md`), so the shares bound the gain for coding decode only.

| Pinned bytes | Experts | Per-layer top n: share of reads | Global ranking: share | Left for the SSD tier |
|---:|---:|---:|---:|---:|
| 2.3 GB | 1,123 (9 %) | n=23: 0.368 | 0.378 | 62 % |
| 10.0 GB | 4,885 (40 %) | n=102: 0.814 | 0.818 | 18 % |
| 16.9 GB | 8,257 (67 %) | n=172: 0.952 | 0.963 | 3.7 % |
| 18.3 GB | 8,941 (73 %) | n=187: 0.969 | 0.977 | 2.3 % |
| 19.6 GB | 9,576 (78 %) | n=199: 0.979 | 0.987 | 1.3 % |
| 22.0 GB | 10,748 (87 %) | n=224: 0.994 | 0.998 | 0.2 % |

The top 84 per layer (the VRAM slots) carry 0.753 on average (0.553 to 0.919). A global ranking beats equal n per layer by 0.4 to
1.1 points. Of the 24.7 % of reads outside the top 84, pinning the next 6.6 GB covers 73 % and the next 10 GB 91 %. Measured VRAM
hit rate: 75 to 88 %. **Bound:** at the 32 GB class (about 18 GB pinned) at most 2 to 3 % of expert reads are cold.

## 4. Design

- **Profile.** One per model (`backend/expert_profile.h`, versions 1 and 2): half decode, half prefill, each normalized per layer
  (`tools/coder_requant/select_experts.py`). It ships with the model: `toolchain/profiles/<model id>.bin`, embedded in
  `tona` (`scripts/make_expert_profile.py`; `scripts/make-profile.sh` for the Coder). `rank_experts(profile, online,
  vram_slots)` in `engine/backend/expert_tiers.h` returns the global order for every backend.
- **Score.** `usage(l, e) x P(miss in the fast tier)`. On the PC the top `n_slots` of a layer are weighted with `q` (0.2,
  **estimate**, fixed by the phase B replay), others with 1. `usage` is the profile plus an online count of VRAM misses with a
  half-life of 20,000 tokens (**estimate**).
- **Tiers.** Hot: VRAM slot cache (dynamic, seeded by the profile). Warm: pinned RAM by score until the budget is full
  (`pin_experts`, growing and shrinking by pieces). Cold: page cache and SSD, explicit reads.
- **Static plus dynamic.** Static global ranking at start, only when 70 % of the experts fit (`MIN_PARTIAL_PIN_FRACTION`);
  re-ranking from online counts at most once per minute.
- **Budget.** With `TONA_TIERS`: `pin_bytes = limit - RAM_SLACK - prompt_reserve - cache_floor - held`, used if it holds at least
  70 % of the expert bytes (below that a profile pin loses to the page cache), capped at all. `cache_floor` = `CACHE_MIN_GB` (3.0;
  replay: 20.3 GB pinned + 2.3 GB cache gives 1.7 SSD reads per token, 18.1 + 4.5 gives 1.9; sweep against 4.3 with
  `TONA_TIERS_CACHE_GB`). `TONA_PIN_GB` takes precedence.
- **Pieces (not built).** The pinned stage becomes pieces of about 32 MB (16 adjacent experts of one layer) and a table
  `place[layer][expert]` replaces `PinLayer::idx`, so one expert leaves the pinned set alone; freed in reverse score order.
- **Cold reads (not built).** `pread`/`io_uring` at queue depth 8 or more into a staging pool, issued when the next-layer
  prediction names a cold expert (85 % recall in the top 24), demand reads for the rest. Page cache stays on. The Mac does the same
  at depth 16 (13 GB/s, `docs/hardware/mac-32gb.md`).
- **No change of numbers.** Same bytes, kernels and summation order for a plan (`TONA_DETERMINISTIC=1`, goldens). The CPU/GPU split
  plan depends on the pinned set and is not bit-exact in production mode (`docs/backends/README.md`): check tokens with `TONA_PLAN` fixed
  and with the default plan.

## 5. Dynamic rebalancing under memory pressure (design, not built for the pinned tier)

Free RAM changes by many GB within minutes. Shrink the warm tier when other programs need RAM, grow it back when free, without
aborts, fault storms or oscillation. Existing: `backend::free_ram_bytes()` (smallest room of `MemAvailable` and every cgroup up to
the root, counting only memory the kernel cannot take back) and `stash_fits` / `make_room`. The monitor and release order are in
`research/44-memory-priority.md` (sections 1 and 5); this design is its first consumer on the PC.

**Signal.** One thread samples `free_ram_bytes()` every 250 ms. PSI (`memory.pressure` of the own cgroup, else
`/proc/pressure/memory`) marks "short"; the unprivileged PSI trigger has a 2 s minimum window, too slow alone. `memory.events` is a
log diagnostic only.

**Reaction.** Release between two passes (about 20 ms); 8 pieces (260 MB) per gap, 32 below 2 GB free, everything while no pass
runs. At 1 GB/s growth the reaction needs about 1 GB of margin, at 4 GB/s about 4 GB (**estimate**). Release cost of a piece is
**unchecked**.

State machine (`engine/backend/ram_governor.h`, pure, unit-testable): input free bytes, time, pass running; output a signed byte
delta. All numbers are **starting values, unchecked**.

| State | Condition | Action |
|---|---|---|
| calm | free >= 10 GB for 30 s | grow one 512 MB step every 5 s while free - step >= 8 GB; stop at the section 4 budget |
| band | 4 GB <= free < 10 GB | nothing (dead band wider than the largest grow step) |
| tight | free < 4 GB | release stored conversations first (to a quarter of the budget), then the lowest-score pieces until free >= 6 GB |
| short | free < 2 GB or PSI "some" in the own cgroup | emergency step size, no waiting for a gap longer than one pass |

The 30 s calm time doubles with every tight event within 60 s of the last, up to 10 minutes.

Safety rules:
1. Pieces change only between passes, on the engine thread, after `prefetch_drain`; `place[]` is cleared before the memory is freed
   and set after the data is complete.
2. Growing reads the piece through the file and drops the page cache (`drop_range`), and never grows when `cgroup headroom - piece
   < cache_floor`. The page cache is never shrunk by us; the pinned range only by us, never by the kernel (a kernel invalidation
   of a registered range stops all queues, research/44).
3. A shrink turns pieces into cold experts, least used first; the 100 ms yield keeps a long cold wait from aborting a pass.
4. Off by default until speed-neutral and bit-exact without pressure (rules 1 and 3 of AGENTS.md).

Test: a balloon process outside the server's cgroup grows at 100 MB/s, 1 GB/s and 3 GB/s to 10, 20 and 30 GB, holds 20 s or 120 s,
releases, repeats every 90 s for 10 minutes, during decode. Metrics: tok/s per 5 s window against no balloon, longest pass, yields,
aborts (must be 0), `MemAvailable` minimum, evicted time (KFD counter), time to shrink and regrow, grow/shrink events per cycle (1
and 1), tokens identical with `TONA_DETERMINISTIC=1`. Arms: no governor, governor.

## 5a. Giving VRAM back at run time (built behind `TONA_VRAM_GOVERNOR`, off by default)

Another program must be able to take 0.5 to 1.5 GB of VRAM while Tona runs without the desktop losing command submissions
(2026-11-15: +512 MiB from a second process made amdgpu refuse the compositor's submissions; the session ended, Tona survived).
`TONA_VRAM_RESERVE_MB` (1024 MiB) absorbs a burst up to its size; the governor gives slots back after a burst so the reserve is
whole again. Numbers: `research/measurements/2026-11-15-expert-tiers.md`. No engine found shrinks a serving expert cache under
live decoding in response to another program (search of 2026-11-15); vLLM sleep mode and SGLang release whole pools while idle,
CUDA only.

**Measured (`bench/micro/vmm_stall.hip`).**
1. `hipMemUnmap` of a 32 MiB chunk on an idle GPU takes 0.007 ms and **blocks the host until the device is idle** (50.4 ms behind a
   60 ms spinning kernel). No stall inside a running kernel.
2. Remap at the same address with a **new handle: 29 of 400 chunks wrong**; fresh address: 0; same handle and address: 0 (kernel
   7.2.15). This is [ROCm/TheRock #8744](https://github.com/ROCm/TheRock/issues/8744) (open): the GPU TLB is not invalidated after
   unmap.

**Design as built.**

| Part | Rule |
|---|---|
| Address range | An address range that was ever unmapped is **never mapped again**. Shrinking kills the newest generation; growing maps a new generation at never-used addresses. `backend/va_ledger.h` refuses a map request touching an earlier range in every build type. |
| Pool | One reservation (`TONA_VRAM_VA_GB`, 512 GiB, about 800 generations at 0.6 GB) holds the core (slots and prefill workspace, one mapping) and elastic blocks. A layer's slots stay at `layer_base + index * slot_bytes`; far blocks have large indices (the address space is the limit, not the `int32` index). |
| Elastic part | The last `TONA_VRAM_ELASTIC_MB` (1536) of the slots live in generations of `TONA_VRAM_STEP_SLOTS` (4) slots per layer (blocks rounded to 2 MiB). The core keeps at least `n_expert_used x 8` slots on the GPU (80 for the Coder) and 32 (at least twice `n_expert_used`) in CPU mode (`TONA_CPU_EXPERTS=1`). |
| Live check | Every write into a slot goes through `ElasticPool::require_live` (refuses, logs and throws outside a live ledger range): decode misses, CPU-mode refills, cache preload, next-layer prefetch, return of borrowed slots. Hits are not checked. |
| Layered prefill | Borrows slot regions of the last layers (at most 50 %, `TONA_PF_LOAN_PCT`) in the never-unmapped core; `loan_fit` may need more layers (`TONA_PF_DEBUG=1`). |
| Signal | Free VRAM = `mem_info_vram_total` - `mem_info_vram_used` (sysfs), polled every 250 ms on the engine thread (also while idle, `Backend::vram_tick`). `TONA_VRAM_FAKE_FREE=<file>` ("<seconds> <free MiB>" lines) replaces it for tests. |
| Marks (`backend/vram_governor.h`) | A step-counting wrapper around `RamGovernor`. With R = 1024 MiB: emergency below 256 MiB, tight below 512, release aims at 768 (whole generations); growth only above R + a step (1424 MiB), one generation per 5 s; calm 30 s, doubling within 60 s up to 10 min. A first sample below the tight mark rebases R to what is free. |
| Safe point | `vram_tick` acts only with no loan open, no pass awaiting commit and an idle stream (`hipStreamQuery(0)`); it drains the prefetch stream and calls `hipDeviceSynchronize` before unmapping, never while k_mega runs or waits (AGENTS.md "GPU safety"). A long prefill ticks between its groups. |
| Cost | Unmap plus release of a generation about 48 x 0.014 ms; regrow 0.04 ms per block, new slots fill by misses. Standing cost: 1 to 2.5 % decode per GB of slots. |

Unit tests: `test_vram_governor`, `test_mega_extra`. Unmeasured on a loaded card: the core floor of 32, the 1536 MB elastic
default, the 250 ms poll.

**Drawbacks.** A reactive scheme cannot prevent a burst larger than the free VRAM (whether 1024 MiB is enough is unmeasured). The
elastic part is limited to 1.5 GB. A stale translation writes into pages that may belong to another program: the same-address
rule is a hard check, and the micro's same-address variants need `--unsafe-same-address` (never on a desktop GPU in use). Regrow
is slow by design. A kernel fix of #8744 would allow same-address reuse.

## 5b. No idle memory: what the OS can take back by itself (design only)

Principle: keep no memory free as a buffer; use memory the OS can take back instantly. The VRAM part is superseded by
[vram-giveback.md](vram-giveback.md); `TONA_VRAM_AUTO_RESERVE` is not built.

| Memory | Reclaimed under pressure | GPU-direct |
|---|---|---|
| Clean file-backed page cache (`mmap` of the GGUF, `pread`) | yes, first, no writeback; refault from NVMe | no |
| Private anonymous with `MADV_FREE` | yes, lazily; becomes a zero page **without notice**; a write cancels it | no |
| Pinned memory (GTT buffer) | **never**; only Tona can free it | yes |

Consequences:
1. **The warm tier splits in two.** *Pinned* GTT buffers (`HSA_USERPTR_FOR_PAGED_MEM=0`) hold what the GPU reads directly, sized
   by the measured GPU-direct share. *Soft warm* is the rest as clean page cache, read by the CPU path only, taken back by the kernel
   first.
2. **`MADV_FREE` is not for experts** (page cache is better) but fits data that is not in a file: stored conversations and
   prefix checkpoints (`stash`). Put a non-zero magic word at the start of every 4 KiB page. **Copy the page first, then check the
   magic word of the source page, never check then copy.** `mincore` cannot be the check. Micro `bench/micro/madvise_free.cpp`:
   under pressure 7,456 of 262,144 page reads were lost, **0 passed copy-then-check with wrong bytes**.
3. **No gap.** The RAM governor becomes reactive only: trigger is PSI `some` of memory or MemAvailable near zero; `R` is only the
   step size. Regrow needs a calm period, one step at a time, because pinning turns reclaimable cache into unreclaimable memory.
4. **No `memory.low`/`memory.min` for Tona.** **(unverified)** whether `memory.high` instead of `MemoryMax` fails more smoothly.

**Result of the `host_pin` micro (`research/measurements/2026-11-15-expert-tiers.md`).** `hipHostRegister` of a file-backed or
anonymous range is accepted but **does not pin** (`MADV_PAGEOUT` still removes every page); registered memory is reclaimable with a
driver notifier, whose reclaim evicts the process's queues. So a soft tier must never be registered, and GPU-direct pins must be GTT
buffers. GTT pieces of 32 MB free one by one (0.35 ms), so a reactive release by pieces is cheap.

**VRAM is the other case.** No OS reclaim exists: amdgpu evicts our buffers to GTT only by evicting the process's compute queue
(`evicted_ms` in `/sys/class/kfd/kfd/proc/<pid>/stats_*`), which stops the megakernel while it waits for the CPU. A self-calibrating
reserve was rejected (maintainer, 2026-11-16). See [vram-giveback.md](vram-giveback.md).

## 5c. No magic numbers

Amounts that depend on the user's free memory (budgets, reserves, floors) come from the machine at run time. Mechanism
granularities (chunk size, reaction step, poll interval, timing constants) may be fixed constants **chosen by measurement and named
with that source**.

| Constant | Where | Verdict |
|---|---|---|
| Allocation granule (2 MiB) | `vmm_granule()` | done: from `hipMemGetAllocationGranularity` at run time |
| Chunk size of the core, 32 MiB | `slots_chunk_bytes()` | done: named constant, sweep of `vmm_chunks_build` (2 to 64 MiB: same 0.15 s build, 68 ms pass); `TONA_SLOTS_CHUNKED` (off) |
| VRAM reserve R, 1024 MiB | `TONA_VRAM_RESERVE_MB` | interim, remove when an instant give-back exists |
| Core floor, GPU mode `n_expert_used x MEGA_BMAX` | `plan_elastic` | keep (model and kernel property) |
| Core floor in CPU mode 32; elastic share 1536 MB; slots per step 4; address reservation 512 GB | `plan_elastic`, `TONA_VRAM_*` | remove with the chunked core: elastic = all slots above the floor, step = a chunk, reserve what the API accepts |
| Poll 250 ms; calm 30 s, max 600 s; grow gap 5 s; doubling window 60 s | `ElasticPool::due`, `VramMarks` | mechanism constants: keep, name the measurement |
| `RAM_SLACK_GB` 0.5, `CACHE_MIN_GB` 3.0, `TIERED_CACHE_MIN_GB` 1.5, `PROMPT_MIN_GB` 1.0, `MIN_PARTIAL_PIN_FRACTION` 0.7, `kTiersKvCtx` 65536 | `backend/budget.h`, `mega_experts.cpp` | remove in step 5 (nothing is kept free); the 70 % rule stays as a result on this PC |
| PSI trigger `some 150000 1000000` | design only | remove: derive from the time to free one piece |

## 6. Expected gain per class (estimates)

Replay of a coding-decode routing trace (`tools/expert_tier_sim.py`, `research/measurements/2026-11-15-expert-tiers.md`): a
per-layer LRU of 85 slots misses 23 % of the unioned expert reads (server hit rate 87 % because it preloads). One cold read costs
1.6 ms; the 10 GB class reads 135 MB per token at 14 tok/s, 70 % of the SSD's 2.65 GB/s.

| Class (budget) | Room R | SSD reads per token, page cache only | with the tiers | Decode now (short code / long) | Decode expected |
|---|---:|---:|---|---|---|
| 64 GB (58) | all | 0 | unchanged | 109 / 77 | unchanged (rule 1) |
| 32 GB (26) | 22.6 GB | 4.2 | 1.7 to 1.9 with 18 to 20 GB pinned | 63 / 42 | **about 80 to 85 / 50 to 55** |
| 16 GB (10) | 6.6 GB | 66 | 55 (managed frequency cache) or 40 to 50 (pins by online miss counts); the profile alone does not help | 10 / 14 | **x1.1 to 1.3**: bandwidth-bound; needs more VRAM or a smaller model |

Partial pins by the profile lose to the page cache at rooms of 6 to 18 GB, hence the 70 % rule. Explicit parallel reads help the
32 GB class, not the 16 GB class (x1.4 at most).

## 7. What the Mac session needs

Read the same profile (`read_expert_profile`) and rank with `rank_experts`; `P(miss)` is 1 outside the arena. Fill the arena by
score, evict by lowest score (LRU tie-break), read cold experts by direct reads at depth 16 with next-layer lookahead. Pressure
(`DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`) feeds `ram_governor.h`; the actuator resizes the arena by pieces (test: balloon script plus
`TONA_METAL_PIN`). A new profile field goes into a version 3 header by agreement.

## 8. Decision and phase B

Phase B, each step with its own decision:
1. Probe: `TONA_ROUTE_TRACE` on a coding decode, replayed offline (COPE-style) to get `q`, the cold-miss share per budget and the
   reuse of the cold set; fixes the page-cache floor.
2. Probe: cost of freeing and re-pinning a 32 MB piece; fixes piece size and release step.
3. Static tiering (global ranking, no governor), 10 and 26 GB classes, 3 alternating rounds, tokens identical. **Done
   (2026-11-15):** +40 to +100 % decode, +70 to +90 % prefill at the 26 GB class, tokens identical; `TONA_TIERS` is on by default
   (`TONA_TIERS=0` switches it off), the 70 % rule leaves the 10 GB class unchanged.
4. Explicit cold reads with prediction lookahead. **Not built.**
5. Governor and balloon test. **VRAM side built behind a flag; the balloon test ran at level 0 only**
   ([2026-11-18-vram-level0.md](../../research/measurements/2026-11-18-vram-level0.md)); the RAM side is not built.
