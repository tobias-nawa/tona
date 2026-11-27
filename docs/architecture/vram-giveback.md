# VRAM give-back without idle memory

Rule (maintainer, 2026-11-16): Tona keeps no VRAM free proactively; the expert cache fills the card; when another program needs VRAM
it gets it at once at Tona's expense (Tona may get slower, neither may fail). Partly built (code read 2026-11-25, not rerun).
Mechanism and options: [research/58-vram-instant-giveback.md](../../research/58-vram-instant-giveback.md). Tags: [V-code] kernel or
ROCr source read, [V-local] measured here, [unchecked].

## Status per part

| Part | State |
|---|---|
| E0 (GTT headroom), E1 (KFD SMI log) | done: [2026-11-16-gtt-headroom.md](../../research/measurements/2026-11-16-gtt-headroom.md): 1.15 GB GTT headroom, desktop's own GTT use 35 MB, no `ttm` eviction in normal operation |
| E5 (many small BOs in one range) | done, below |
| E2, E3 (iGPU), E6 (root `amdgpu_gem_info`) | not run |
| (a) slots as chunks in one range | behind `TONA_SLOTS_CHUNKED` (off); fixed 32 MiB chunk ([expert-tiers.md](expert-tiers.md) section 5c). A/B: tokens identical, decode -0.05 to -0.6 % ([2026-11-18-vram-chunks-gtt-r3.md](../../research/measurements/2026-11-18-vram-chunks-gtt-r3.md)) |
| (b) GTT headroom | `TONA_GTT_MARGIN_MB` (default 1024; 3072 measured: tokens identical, decode +0.1 to +0.3 %, eviction risk untested) and `TONA_PIN_OVERFLOW`; `ttm.pages_limit` untouched |
| (c) KFD SMI events | logger built (`TONA_KFD_SMI_LOG`, `engine/gpu/kfd_smi.*`), not the trigger; the reaction is the VRAM governor (`TONA_VRAM_GOVERNOR`, off), polling sysfs every 250 ms ([expert-tiers.md](expert-tiers.md) section 5a) |
| (d) dmem protection | not built, not planned |
| Compositor survival test | level 0 ran ([2026-11-18-vram-level0.md](../../research/measurements/2026-11-18-vram-level0.md)): no eviction, no verdict on (a) or (b); level 1 (card to full) not approved |

## Rule: no magic numbers

Chunk size, GTT room and release amount come from the machine and the event at run time: allocation granularity
(`hipMemGetAllocationGranularity`), `gtt_total`/`gtt_used` from sysfs plus observed bursts as a decaying record, and the bytes the
event shows. Numbers below for this PC (8.4 GB BO, 31 of 32.5 GB GTT, 260 chunks, 6 GB burst) are measurements or examples.

## What is known

The kernel tries to evict our buffers when the desktop asks and fails for the desktop (`-ENOMEM`, "Not enough memory for command
submission") when the eviction finds nothing and the GTT fallback is full (research/58 section 2.3).

1. **Our big buffer is one BO:** a normal start has 1,787 `hipMalloc` calls (13.4 GB) and one of 8,370 MB (expert slots and prefill
   workspace, `mega_build_slots.cpp:91`) [V-local]. TTM evicts whole BOs, so one eviction takes 8.4 GB of 15.9 GB, stops every
   queue (all BOs of a process share one eviction fence) and the restore worker brings it back 100 ms later (balloon test
   15.9 -> 7.6 GB).
2. **The GTT has no room:** `cap_pin_budget()` leaves 1 GiB; 31.0 to 31.3 of 32.5 GB are in use with a 58 GB class server
   [V-local].

## The combination

### (a) Slot memory as many small BOs

Target: a desktop request of N MB evicts about N MB of Tona. Requirement for the elastic pool: all slots, core included, from chunks
in one reserved range, never remapped at the same address.

- **Today:** `ElasticPool` (`engine/gpu/mega_elastic.cpp`) maps the last 1.5 GB of the slots (`TONA_VRAM_ELASTIC_MB`) in generations,
  one `hipMemCreate` per layer of `step_slots` (4) slots, about 8 MB. The core (about 7 GB of slots plus 0.8 GB prefill workspace)
  stays one mapping, so under kernel eviction it is still one BO.
- **Each chunk is its own BO** [V-code: `Runtime::VMemoryHandleCreate`]: every `hipMemCreate` is one KFD `ALLOC_MEMORY_OF_GPU` call.
- **Proposal:** all expert slots become granule-sized chunks in one reserved range (`hipMemAddressReserve` once, then N x
  `hipMemCreate` + `hipMemMap` + `hipMemSetAccess`). Kernels keep their 64-bit slot offsets, so tokens stay identical. A test value
  of 32 MB gives about 260 BOs for 8.4 GB. Chunks are **never unmapped and remapped at the same address** (ROCm/TheRock #8744). The
  prefill workspace follows the last slot chunk (`mega_extra.h`).
- **Cost to check:** about 260 more BOs per restore (`restore_process_bos`), mapping time at load, TLB reach.
- **Not given:** which chunk the kernel takes (TTM LRU, roughly creation order, unchecked); Tona only makes any victim small and
  creates the coldest chunks first.

### (b) GTT headroom for those chunks

An evicted chunk goes to TT pages and counts in `mem_info_gtt_used` / `ttm.pages_limit` (`usage > size` gives `-ENOSPC`) [V-code].
Headroom must cover the largest foreign VRAM burst B that arrives faster than Tona reacts, plus the desktop's own GTT use d:

    gtt_headroom >= B + d,   gtt_total - tona_gtt_use >= headroom,   tona_gtt_use = pinned experts + stash + KV + prompt slots

| Quantity | Value | Source |
|---|---|---|
| `gtt_total` (= `ttm.pages_limit`, half of RAM) | 32.46 GB | `mem_info_gtt_total` |
| Tona GTT use (58 GB class) | 31.0 to 31.3 GB, 26.3 GB pinned experts | `mem_info_gtt_used` |
| Headroom today | 1.2 to 1.5 GB | `cap_pin_budget` margin |
| B | 0.5 GB (compositor and terminal burst) up to 4 to 8 GB (Blender, many tabs) | assumption, to be measured per user |
| d | about 0.5 GB | [unchecked] |

Example: a 6 GB burst needs about 6.5 GB headroom, so Tona's GTT use must fall from 31.3 to about 26 GB or the limit must rise, in
this order: (1) **raise `ttm.pages_limit`** (root, `modprobe.d`) from half to three quarters of RAM (+15 GB of limit at 60 GB RAM,
no extra pinned RAM; risk: the kernel reclaims page cache first, then fails; failure mode unchecked); (2) **pin less**: size the
pinned hot set by the measured GPU-direct share with the headroom as a parameter replacing the 1 GiB margin. Open: real `d` and
worst `B` of this machine; whether TTM charges eviction pages to the evictor's memcg [unchecked].

### (c) KFD SMI eviction events as the trigger to drop, not to restore

- **Signal:** `AMDKFD_IOC_SMI_EVENTS` on `/dev/kfd`, events `QUEUE_EVICTION` (trigger `ttm`) and `QUEUE_RESTORE`; no root, only the
  calling process's events [V-code]. Delivery 3.5 us on average, 85 us at most.
- **Reaction:** the reader thread sets an atomic and nothing else (no HIP call: AGENTS.md GPU safety). At the next safe point
  (`vram_tick`: no loan open, stream idle) the engine computes the bytes that left VRAM since the last tick and **releases that many
  of the coldest chunks** (never the core floor) with `hipMemUnmap` + `hipMemRelease`, dropping those slots from the cache index
  first. Without it the restore worker takes back what the desktop has not claimed and the next desktop allocation evicts again.
- **Weak point:** Tona cannot tell which chunks the kernel moved. Evicted chunks stay in GTT until restored and are read over PCIe
  (slower, correct). Options: accept it; predict victims from creation order; probe read bandwidth per chunk (32 MB at 315 GB/s
  against about 28 GB/s over PCIe; a GPU job).
- **Not solved:** a burst bigger than the free VRAM plus one reaction ends in a kernel eviction first (Tona stalls 0.1 to 4 s:
  `eviction credit` in the server log).

### (d) dmem protection: an optional root step

`dmem.min` on the desktop's cgroup makes the kernel evict Tona before the desktop (research/58 section 3.1). It needs root once
(`+dmem` down the cgroup chain, `dmem.min` on `user.slice` and below), decides only who is evicted, and Tona works without it. The
installer may print the commands, never run them. Not planned before E2 confirms the policy on 7.2.15.

## Experiments

E0 and E1 are done; E2/E3 (iGPU) were skipped by instruction; E6 (root `cat /sys/kernel/debug/dri/1/amdgpu_gem_info` while a
server runs) needs the user.

**E5 result** (idle card, `bench/micro/vmm_chunks.*`):
- Granularity: minimum 4096 bytes, recommended 2,097,152 bytes.
- KFD ioctl counts (strace): ALLOC_MEMORY_OF_GPU 8 baseline, 288 with 260 chunks (+280, 20 unattributed); consistent with one
  allocation per chunk, not isolated. MAP_MEMORY_TO_GPU rose by only 20. VRAM used grew by exactly the chunk sizes.
- 260 chunks of 32 MB in one range (median / p99 per chunk): `hipMemCreate` 0.035 / 0.056 ms, `hipMemMap` 0.006 / 0.011 ms,
  `hipMemSetAccess` 0.044 / 0.066 ms; about 22 ms in all. A fill and check of 8.72 GB across every border: **0 wrong words**.
  Unmap plus 260 releases: 12.8 ms.
- **Not understood:** `mem_info_vram_used` still read 8.636 GB above the start right after the release; do not assume VRAM is free
  at once after `hipMemRelease`.
- The restore cost of 260 more BOs is unmeasured.

## Order of work

1. E0/E1 and the H1 verdict; if the GTT is within 1.5 GB of full in normal operation, (b) comes first.
2. E5, then (a) behind a flag, bit-exact tokens (`TONA_DETERMINISTIC=1`, goldens).
3. (c) stage 2 inside the governor's safe-point machinery.
4. (d) as documented option.

(a) and (b) change allocation layout and pin budget, so they carry A/B runs (`scripts/ab.py`) with the plan first.
