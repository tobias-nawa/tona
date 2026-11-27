# VRAM instant give-back: fill the card, yield to the desktop at once (#157)

**Agent research, state of 2026-11-16** (a Sonnet agent; research only, no GPU job, no VRAM pressure test). Status tags:
[V-code] = kernel source read in full for the function named (Linux `master` of 2026-11-16 from raw.githubusercontent.com, **not**
the running 7.2.15 Fedora build; paths and names are stable since 6.x but line-level behaviour on 7.2.15 is unchecked),
[V-local] = read on this PC today (sysfs, journal, logs), [snippet] = search result only, [unchecked] = from general
knowledge or inference, with the check that would settle it.

## Goal and what happened

Maintainer rule: Tona keeps **no VRAM free proactively**; the expert cache fills the card; when the compositor, a browser
or Blender needs VRAM, it gets it **immediately**, at Tona's expense (slower is fine, a failure of Tona or of the other
program is not).

2026-11-15, 17:51 (`/var/tmp/claude/148/res/v-today.*`, `/var/tmp/claude/148/vram-crash-kernel.log`) [V-local]:

| Fact | Source |
|---|---|
| Tona held 15.7 of 16.3 GB VRAM (KFD process, persistent megakernel, 26.3 GB pinned host memory) | `v-today.balloon.txt` (start row), `v-today.server.log` line 36 |
| A second HIP process (the balloon) allocated +512 MiB steps; its step 3 (1.5 GiB held) failed with "out of memory" after 49.8 s | `v-today.balloon.txt` |
| Card VRAM used over time: 15.7 -> 15.9 -> **7.6** -> 15.9 -> **7.0 GB**: whole blocks of about 8 GB left VRAM and came back, although the balloon needed 0.5 GB per step | `v-today.balloon.txt` |
| `[drm] *ERROR* Not enough memory for command submission!` 3x for quickshell (`qs`), 2x for kitty; their contexts ended ("VM memory stats ... non-zero when fini") | kernel log |
| No GPU reset; Tona's queues were evicted (`evicted_ms` 91 s over the run) and recovered, decode back at 90 to 100 tok/s | `v-today.loop.txt` |
| Journal 17:40 to 18:00 holds **no** `Failed to find memory space for buffer ... eviction`, no `Buffer eviction failed`, no `amdgpu_vm_validate() failed.` | `journalctl -k` (read today) |
| `mem_info_gtt_used` **31.3 of 32.5 GB** (card1, Tona running), `ttm.pages_limit` 7,925,327 pages = 30.2 GiB | `/sys/class/drm/card1/device`, `/sys/module/ttm/parameters/pages_limit` |

The last two rows decide what the mechanism can be (section 2.3).

## 1. Short answer

1. The graphics process is **not** missing a feature: the kernel does try to evict Tona's BOs, and hipMalloc memory **is**
   evictable. What goes wrong is the allocation of a *destination or a placement*: `-ENOMEM` from `ttm_bo_validate()` is
   what TTM returns when neither the VRAM placement (after an eviction walk) nor the GTT fallback can be satisfied.
2. Tona's own memory policy removes the second place. `cap_pin_budget()` (`engine/gpu/gtt_budget.h`) fills the GTT budget
   up to **1 GiB** below `mem_info_gtt_total`; the card therefore has about 1 GiB of GTT for everything the desktop moves
   or creates there. Today: 31.3 of 32.5 GB in use. This is the best-supported candidate (H1, below), **not proven**.
3. Even when the eviction works, KFD gives back **all-or-nothing**: all BOs of a process share one eviction fence, so
   evicting any single BO preempts every queue of the process (the megakernel), and a worker restores everything after
   100 ms. That is the "evicted_ms 91 s" and the 8 GB swings above.
4. The only kernel feature that does what the maintainer wants (a protected cgroup wins, an unprotected one is evicted
   first) is the **dmem cgroup controller**. It exists and amdgpu registers its VRAM with it on this kernel; it is switched
   off for user sessions and needs **root once** to enable. It fixes *who is evicted*, not the GTT headroom problem nor the
   all-queues-stop cost.
5. XNACK/SVM migration is **not available on gfx1200** (kernel refuses retry-fault mode for GFX10.1.1 up to 12.0.x).

## 2. Mechanism (code-traced)

### 2.1 Where the message comes from [V-code]

`amdgpu_cs_ioctl()` (`drivers/gpu/drm/amd/amdgpu/amdgpu_cs.c`) calls `amdgpu_cs_parser_bos()` and prints
`Not enough memory for command submission!` for **any** `-ENOMEM` coming out of it. Inside, in order:

1. `amdgpu_vm_validate()` over the process's VM BOs (page tables, evicted per-VM BOs): on failure it prints its own
   `amdgpu_vm_validate() failed.`, which is **absent** from the journal.
2. `amdgpu_cs_bo_validate()` for each BO of the command stream (`drm_exec_for_each_locked_object`): domain =
   `preferred_domains` while the move budget lasts, else `allowed_domains`; on `-ENOMEM` once more with `allowed_domains`.
   Silent on failure apart from the generic message. **This is the path left over** by the journal.
3. userptr, `amdgpu_ttm_alloc_gart()`, BO-list paths (not relevant for a compositor).

`amdgpu_bo_create()` sets `allowed_domains = VRAM | GTT` for every user BO whose preferred domain is VRAM only (except
`AMDGPU_GEM_CREATE_DISCARDABLE`), so a compositor BO may land in GTT, **if GTT has room**.

### 2.2 How TTM evicts a KFD BO [V-code]

- `ttm_bo_validate()` -> `ttm_bo_mem_space()` -> `ttm_bo_alloc_resource()` runs twice: first without eviction on the
  fallback places, then with `force_space` on the desired places, where `ttm_bo_evict_alloc()` walks the VRAM LRU
  (`ttm_lru_walk_for_evict`, trylock pass, then a ticket-locking pass repeated while it makes progress). If the walk
  evicts nothing it returns `-EBUSY`, the place is skipped, and the function ends with `-ENOSPC`, which `ttm_bo_validate()`
  turns into **`-ENOMEM`** ("backward compatibility with userspace"). This is the only way for a compositor CS to get
  `-ENOMEM` while there is still evictable memory in the pool: *the walk found nothing it was able to evict and the GTT
  fallback was full.*
- Each victim goes through `ttm_bo_evict()`: `amdgpu_evict_flags()` asks for `GTT | CPU` placement for a VRAM BO, then
  `ttm_bo_mem_space()` for the destination. If that fails TTM logs `Failed to find memory space for buffer 0x... eviction`
  (`pr_err`). **Not in the journal today.** The destination allocation therefore did not fail in `ttm_bo_evict()`.
- `amdgpu_ttm_bo_eviction_valuable()` refuses a KFD BO only when the evictor has the same `mm` as the fence
  (`amdkfd_fence_check_mm`, "a KFD process needs all its BOs resident"); the compositor and the balloon are other
  processes, so Tona's BOs are legal victims. Pinned BOs (doorbells, MMIO, `amdgpu_amdkfd_gpuvm_pin_bo`) and
  `AMDGPU_PL_PREEMPT` (userptr) are not.
- **hipMalloc is evictable**: `amdgpu_amdkfd_gpuvm_alloc_memory_of_gpu()` creates VRAM BOs with `VRAM_WIPE_ON_RELEASE |
  VRAM_CLEARED` (+ `CPU_ACCESS_REQUIRED` for `KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC`, `VRAM_CONTIGUOUS` for `CONTIGUOUS`);
  `amdgpu_bo_create()` adds GTT to the allowed domains. There is **no evictable/non-evictable flag** in
  `kfd_ioctl.h`'s `KFD_IOC_ALLOC_MEM_FLAGS_*` list (VRAM, GTT, USERPTR, DOORBELL, MMIO_REMAP, WRITABLE, EXECUTABLE, PUBLIC,
  NO_SUBSTITUTE, AQL_QUEUE_MEM, COHERENT, UNCACHED, EXT_COHERENT, CONTIGUOUS). Only the pin calls make a KFD BO immovable.
- **Eviction is synchronous for the evictor, asynchronous in the process.** `ttm_bo_evict()`/`ttm_bo_wait_ctx()` waits for
  all fences of the BO including `DMA_RESV_USAGE_BOOKKEEP`, where the KFD eviction fence lives. Waiting on it calls
  `amdkfd_fence_enable_signaling()` -> `kgd2kfd_schedule_evict_and_restore_process()` (`amdgpu_amdkfd_fence.c`), which
  queues `evict_process_worker()` (`kfd_process.c`): it preempts all queues of the process
  (`kfd_process_evict_queues`, trigger `KFD_QUEUE_EVICTION_TRIGGER_TTM`), signals the fence, and schedules
  `restore_process_worker()` after `PROCESS_RESTORE_TIME_MS` = 100 ms (`kfd_priv.h`; `PROCESS_BACK_OFF_TIME_MS` 100,
  `PROCESS_ACTIVE_TIME_MS` 10). So the CS does **not** fail "before the eviction completes"; it blocks in an interruptible
  wait. The preemption of a megakernel with spinning blocks is the slow part (`gpu::Mega: eviction credit 2.3 to 3.8 s` in
  `v-today.server.log`; `queue_preemption_timeout_ms` = 9000 by default).
- **All BOs of the process share one fence** (`amdgpu_amdkfd_fence.c` header comment): evicting a single 98 MB slot stops
  *every* queue. There is no partial give-back at the queue level.
- **Restore ping-pong.** `amdgpu_amdkfd_gpuvm_restore_process_bos()` re-validates **every** KFD BO into `mem->domain`
  (VRAM) and falls back to GTT per BO only when the VRAM validate fails. It runs from the restore worker after 100 ms,
  holding the reservation locks of the process's BOs (`drm_exec`), and it takes back whatever the other process has not
  pinned in the meantime. The 15.9 -> 7.6 -> 15.9 -> 7.0 GB swings in the balloon log have exactly this shape.

### 2.3 Why `-ENOMEM`: candidates, ranked

| # | Hypothesis | For | Against / open | Settles it |
|---|---|---|---|---|
| H1 | **GTT placement impossible**: the compositor BO (or a BO the CS has to move) cannot go to VRAM (eviction walk finds nothing evictable at that instant) **and** `amdgpu_gtt_mgr_new()` returns `-ENOSPC` because `ttm_resource_manager_usage(man) > man->size` (GTT total 32.5 GB, 31.3 GB used by Tona's pinned GTT buffers + the rest; margin 1 GiB by design) | GTT used 31.3/32.5 GB today; `cap_pin_budget` margin of 1 GiB; no destination-failure `pr_err`; matches `-ENOSPC` -> `-ENOMEM` mapping | Needs the GTT use **at 17:51** (not logged). Today's number is a different run. Why the VRAM walk found nothing is open (H2) | log `mem_info_gtt_used` at 10 Hz in the next run; compare with the `-ENOMEM` moment |
| H2 | **Eviction walk finds nothing while the restore worker holds the locks**: `restore_process_bos` locks all of Tona's BOs (`drm_exec`) and revalidates them; a trylock walk skips locked BOs, and the ticket pass only repeats while it makes progress | the swings show restore running while the balloon still allocated; the walk skips locked BOs by design | not shown that the ticket pass ends without progress; kernel 7.2.15 code unchecked | KFD SMI events (section 3.4) timestamped against the `-ENOMEM`; `debug_evictions` |
| H3 | A BO of the compositor with `CPU_ACCESS_REQUIRED` or VRAM-only constraint (scanout) cannot take GTT at all | `amdgpu_cs_bo_validate` special-cases visible VRAM; card has resizable BAR (full visible) so weak | not checked which BOs qs submitted | `amdgpu_bo_list`/`amdgpu_gem_info` debugfs (root) |
| H4 | Eviction destination failed for lack of **TTM pages** (`ttm.pages_limit`, `ttm_tt_populate()` -> `ttm_global_swapout()`) | pages limit is the same 30.2 GiB as GTT | the destination failure would have logged `Failed to find memory space` | n/a, weakest |

What is **not** the cause: "Tona's BOs are not evictable" (they are), "the CS does not wait for the eviction" (it does).
The root cause is **open** until H1/H2 are separated by the logging experiment in section 6.

## 3. Levers

### 3.1 dmem cgroup controller (Q2)

- **Exists on this PC** [V-local]: `/sys/fs/cgroup/cgroup.controllers` lists `dmem`; `/sys/fs/cgroup/dmem.capacity` shows
  `drm/0000:03:00.0/vram 17095983104` (the RX 9060 XT) and `drm/0000:0f:00.0/vram 2147483648` (the iGPU), so
  **amdgpu registers its VRAM manager with dmem on kernel 7.2.15** (`amdgpu_vram_mgr_init()`:
  `dmem_cgroup_register_region("drm/%s/vram")`; commit "drm/amdgpu: Add cgroups implementation" 2025-03-27; controller
  itself "kernel/cgroup: Add dmem memory accounting cgroup" 2024-12-04, i.e. 6.14/6.15 [commit dates from the GitHub API;
  release numbers unchecked]). `CONFIG_CGROUP_DMEM=y`, `CONFIG_HSA_AMD_SVM=y` in `/boot/config-7.2.15`.
- Only **VRAM** is a dmem region for amdgpu; GTT and system pages are not. So dmem cannot help with H1.
- **Semantics** (`Documentation/admin-guide/cgroup-v2.rst` "DMEM", `kernel/cgroup/dmem.c`, `ttm_bo.c`): `dmem.min`,
  `dmem.low`, `dmem.max` per region, "the same as for the memory cgroup controller". In TTM (`ttm_bo_alloc_at_place`,
  `ttm_bo_evict_walk`): an allocation whose cgroup is below its `min`/`low` may evict BOs of cgroups that are above
  theirs, even when TTM would otherwise try other placements first; BOs inside `min` are never evicted, BOs inside `low`
  only in a second pass. This is exactly the wanted policy: **give the desktop cgroup `dmem.min` (say 2 GiB), leave Tona
  unprotected**; Tona then holds the rest and loses it first.
- **Charging** [V-code]: `ttm_resource_try_charge()` -> `dmem_cgroup_try_charge()` charges the cgroup of `current`
  at the time of the allocation (`get_current_dmemcs()`). KFD allocations come from Tona's process, so they are charged to
  Tona's cgroup. **Moves back by the KFD restore worker run as a kworker = root cgroup**, so restored BOs are charged
  to the root cgroup and count as unprotected. Not a problem for the policy, but it means `dmem.current` of Tona's
  unit is not a reliable measure of its VRAM (unchecked in practice).
- **Unprivileged use**: **no, not out of the box** [V-local]. `/sys/fs/cgroup/cgroup.subtree_control` is `cpu io memory pids`;
  `dmem` is missing at `/`, `user.slice`, `user-1000.slice` and `user@1000.service` (`Delegate=pids memory cpu`).
  systemd 259.10 has no `DMemory*=` settings (`man systemd.resource-control` has none). Enabling needs root once: write
  `+dmem` into `cgroup.subtree_control` from `/` down to the delegated subtree (a boot-time root unit, e.g. `ExecStart=` or
  `tmpfiles.d` `w`); after that the user can write `dmem.min/low/max` **below** `user@1000.service` (delegated, owned by
  the user), but protection is hierarchical like `memory.min`: it is effective only up to what the parents allow, so the
  root unit must also set `dmem.min` on `user.slice`/`user-1000.slice`/`user@1000.service`. Whether systemd overwrites a
  controller it does not know when it re-realizes the units is **unchecked**.
- **Version fit**: `dmem.min/low` protection in TTM is old enough for 7.2 [commit "drm/ttm: Handle cgroup based eviction"
  series Dec 2024 to Jan 2025, snippet; unchecked on 7.2.15]. The newer reclaim-on-lower-`max` callback
  (`amdgpu_vram_mgr_dmem_reclaim`, "Wire up dmem cgroup reclaim for VRAM manager", 2026-07-25) and the `dmem.high` soft
  limit (patch v6 2026-05-31 on dri-devel, **not merged** [snippet]) are **after** the code in 7.2 as far as the dates
  show; do not plan on them.
- **Interaction with KFD**: dmem decides *which BO is evicted*; for a KFD victim everything in 2.2 still applies (one
  fence, all queues stop, restore after 100 ms, 8 GB swings). `dmem.max` on Tona's cgroup is a hard cap (a static
  reserve in disguise, which the maintainer does not want) and is therefore not a candidate.

### 3.2 amdgpu/KFD parameters and sysfs (Q3, first half)

[V-local] list of `/sys/module/amdgpu/parameters` on 7.2.15, relevant entries only:

| Parameter | Value / default | Verdict |
|---|---|---|
| `gttsize` (deprecated, "use `ttm.pages_limit`"), `ttm.pages_limit` | auto = 30.2 GiB (half of RAM) | Raising `ttm.pages_limit` (module parameter of `ttm`, 0644, **root**) gives the desktop GTT room again without giving back VRAM. Cheap, a candidate for H1. Costs more pinnable RAM, so more pressure on the 60 GB host |
| `debug_evictions` | N | Prints every KFD eviction with its cause in dmesg; root only; diagnosis, not a fix |
| `queue_preemption_timeout_ms` | 9000 | Not a lever for latency; only the abort time of a stuck preemption |
| `no_system_mem_limit` | N | KFD's cap on system memory it may pin (half of RAM); irrelevant for VRAM |
| `no_queue_eviction_on_vm_fault`, `cwsr_enable`, `mcbp`, `sched_policy`, `noretry` | defaults | Change preemption behaviour, risk GPU hangs; **not recommended** (`cwsr_enable=0` appears in a Strix Halo workaround [snippet] and would remove context-save: eviction then cannot preempt a running kernel) |
| `moverate` | not readable (permission) | CS move budget (`amdgpu_cs_get_threshold_for_moves`); only decides whether `preferred` or `allowed` domain is tried first. Not a lever for the failure |
| `vm_update_mode`, `vm_fragment_size` | -1 | unrelated to eviction |

There is no module parameter that makes KFD BOs evict per BO, and none that "yields" to graphics.

### 3.3 BO flags, priorities, SVM/XNACK (Q3, second half)

- **No per-BO priority for KFD BOs.** The graphics-side `AMDGPU_GEM_CREATE_*` flags (`DISCARDABLE`, `PREEMPTIBLE`,
  `VRAM_CONTIGUOUS`) are not reachable from HIP. `DISCARDABLE` BOs are *dropped* on eviction (`amdgpu_evict_flags`: zero
  placements); that would be perfect for a cache whose content can be refetched, but KFD does not expose it (only the
  graphics ioctl `AMDGPU_GEM_CREATE` does; a dmabuf-shared BO across APIs would need an amdgpu GEM allocation imported
  into HIP; **unchecked** whether HIP can import and compute on such a BO without the KFD fence, and whether an
  eviction then stops the queues anyway).
- The 2024 RFC "TTM interface for managing VRAM oversubscription" (Friedrich Vock, amd-gfx 2024-04) and Tvrtko Ursulin's
  "Discussion around eviction improvements" (dri-devel 2024-05) propose eviction priorities and an "evicted VRAM" query
  [snippet; merge status **unchecked**, nothing of it found in the amdgpu sources read above].
- **XNACK/SVM is unavailable on gfx1200** [V-code]: `kfd_process_xnack_mode()` returns false for
  `KFD_GC_VERSION >= 10.1.1 && < 12.1.0` ("GFXv10 and later GPUs do not support shader preemption during page faults.
  This can lead to poor QoS ... or even deadlocks") unless the device has per-process XNACK (`KFD_SUPPORT_XNACK_PER_PROCESS`
  is a GFX9.4.x feature). gfx1200 = 12.0.0, so retry-fault SVM with on-demand migration (hipMallocManaged with
  `HSA_XNACK=1`) **does not run**; the 2026 SVM patch series was tested only on gfx943/gfx906 [snippet]. SVM ranges
  without XNACK exist (Tona's overflow uses one) but are mapped, not faulted: no migration on demand. Verify on the PC with
  `rocminfo | grep -i xnack` (read-only) if a second confirmation is wanted.
- **Contiguous/PUBLIC flags** add constraints (`VRAM_CONTIGUOUS` needs one hole, `CPU_ACCESS_REQUIRED` needs visible VRAM);
  avoid them for the expert cache.

### 3.4 User-space signals (Q4)

- **KFD SMI events** [V-code: `include/uapi/linux/kfd_ioctl.h`, `kfd_smi_events.c`]: `AMDKFD_IOC_SMI_EVENTS(gpuid)` on
  `/dev/kfd` returns an anonymous fd; write a `uint64_t` mask (`KFD_SMI_EVENT_MASK_FROM_INDEX(ev)`), then `poll()` and
  `read()` one-line text events (`KFD_SMI_EVENT_MSG_SIZE` 96). Events of interest: `QUEUE_EVICTION` (9, with trigger
  `KFD_QUEUE_EVICTION_TRIGGER_TTM` etc.), `QUEUE_RESTORE` (10), `MIGRATE_START/END` (5/6), `UNMAP_FROM_GPU` (11),
  `GPU_PRE/POST_RESET` (3/4). `/dev/kfd` is 0666 on this PC [V-local].
- **Permission**: without `CAP_SYS_ADMIN` a client receives only events of **its own process** (`kfd_smi_ev_enabled()`:
  `client->tgid != tgid && !client->suser`); `KFD_SMI_EVENT_ALL_PROCESS` (bit 64) needs root. For Tona this is the right
  scope: it sees the evictions of **itself**, no root needed. A separate unprivileged logger cannot see them.
- **Timing**: the event is emitted in `kfd_process_evict_queues()` **before** `evict_process_queues()` runs, so Tona
  learns of an eviction while the queues are being preempted: a notification with a few ms latency, but **after** the
  kernel started taking memory, not before. It is feedback ("pressure exists, shrink more now"), not an early warning.
  Events are dropped silently when the kfifo is full.
- **No VRAM pressure notification exists** otherwise [snippet, negative result]: the search found only the 2024 RFCs above,
  a 2026 drm fdinfo "evicted" accounting patch (not amdgpu-specific), and a 2026 thread where AMD mentions eventfd +
  device-specific wait ioctls for **user queues**, not for memory pressure. Sysfs `mem_info_vram_used`/`mem_info_gtt_used` are
  pollable counters (no `POLLPRI`) [unchecked for notification support, V-local for readability]. A 10 ms poll is
  already what `research/44-memory-priority.md` plans for host RAM.

## 4. What other engines do (Q5)

| Engine | VRAM policy | Source |
|---|---|---|
| vLLM (ROCm) | static fraction of **total** VRAM (`gpu_memory_utilization`, default 0.9), ignores memory other processes hold; users lower it (0.7 to 0.78) for display headroom | ROCm CLI vLLM adapter docs, vLLM forum [snippet] |
| llama.cpp | allocates what it estimates; `-fit`/`--n-gpu-layers` are static; UMA path for iGPU (PR #4449); reports of desktop/GPU freezes on AMD are mostly hangs and MES issues, **none found** that match "full VRAM -> compositor ENOMEM" | [snippet] |
| Ollama | `keep_alive` unload of the whole model; context capped by the user; 7900 XTX desktop-freeze reports are about the driver config | [snippet], also `research/44-memory-priority.md` (unchecked rows) |

None of them gives memory back on demand. The nearest published pattern is a **static reserve**, i.e. what the maintainer
rejected. No report of exactly the 2026-11-15 failure was found; the search tools available returned few direct hits
(GitLab and lore were behind bot protection), so "none found" is weak evidence.

## 5. Options ranked: closest to "no reserve, immediate give-back"

| Rank | Option | Reserve | Give-back | Needs | Risk | Fixes which part |
|---|---|---|---|---|---|---|
| 1 | **dmem protection of the desktop** (`dmem.min` on the session's cgroup, Tona unprotected) **plus** `ttm.pages_limit` raised **plus** Tona's own GTT margin sized to what the desktop may need | none (VRAM); GTT margin > 1 GiB | kernel evicts Tona first, at the next CS; Tona's queues stop for the preemption time (0.1 to 4 s), then Tona runs on what is left | root once (`+dmem` chain, `dmem.min` on parents, `ttm.pages_limit`); kernel 7.2.15 has the region | systemd may reset controllers (unchecked); `dmem.current` of Tona misattributed (kworker restores charge root) | who is evicted (policy), GTT headroom (H1) |
| 2 | **Tona-side elastic budget from KFD SMI + sysfs**: subscribe to `QUEUE_EVICTION/RESTORE` in-process (no root), poll `mem_info_vram_used` and `mem_info_gtt_used` at 10 ms; on an eviction event or when `used + desktop growth` nears the total, **free expert slots in small steps and drop the process's own BOs** before the kernel has to evict them | none while idle; a transient step during pressure | Tona frees its **small** BOs itself, so the kernel finds free VRAM and the walk never needs to evict a KFD BO (no all-queue stop) | no root; needs expert cache in small BOs (granularity: see below) | race: a desktop allocation within the poll interval still evicts (rank 1 or the kernel path covers it); thrash: restore ping-pong must be stopped by not letting the restore worker repopulate (we hold fewer BOs, nothing to restore) | latency of the first reaction, the 8 GB swings |
| 3 | **Small expert BOs** (e.g. 32 to 64 MiB hipMalloc each) instead of large slabs | none | a single eviction takes a small BO; the walk stops after the first success (`ttm_lru_walk_for_evict(..., 1)`), so only the needed amount leaves | Tona code (allocation layout); check `TONA_ALLOC_TRACE` for current BO sizes first | per-BO overhead (page table, `drm_exec` locks of N BOs in every restore), unchecked; does **not** remove the all-queue preemption | amount evicted per request |
| 4 | **`ttm.pages_limit` / GTT headroom alone** | GTT 1 GiB -> several GiB | none | root, module parameter at boot or runtime write (0644) | pinned RAM lowers the host's free memory | H1 only |
| 5 | Static reserve (what vLLM does) | yes | n/a | none | contradicts the goal | all, by reserve |
| 6 | SVM/XNACK migration, `hipMallocManaged` | none | kernel migrates on demand | **not possible on gfx1200** | | n/a |
| 7 | `DISCARDABLE` BOs via a graphics-API import | none | BO dropped, no copy | unchecked whether HIP can use it | large | n/a, speculative |
| x | `dmem.max` on Tona | static cap | n/a | root once | a reserve in disguise | rejected |
| x | `cwsr_enable=0`, `sched_policy`, `noretry` and similar | | | root, reboot | GPU hangs | rejected |

Combining 1 + 2 + 3 reaches the goal: the kernel is the safety net (policy: the desktop wins, with GTT room to land),
Tona's own early give-back makes the safety net rarely fire.

## 6. Proposal and experiments

All steps first measure, then decide; none touches the desktop session.

**E0 (read-only, zero risk): confirm or refute H1 in the next normal run.**
Log at 10 Hz into the measurement file: `mem_info_gtt_used`, `mem_info_gtt_total`, `mem_info_vram_used` of card1; run
`journalctl -k -f` filtered for `Not enough memory|Failed to find memory space|Buffer eviction failed|amdgpu_vm_validate`.
Decision: GTT within 0.5 GiB of the total when a `-ENOMEM` occurs -> H1 confirmed, the fix is GTT headroom
(options 1/4) before anything else; GTT far below -> H2/H3, go to E1.
Also read: `rocminfo | grep -i xnack`; `TONA_ALLOC_TRACE` of a start (BO sizes for option 3).

**E1 (read-only for the desktop): KFD SMI logger inside Tona.**
Add a diagnostic thread (behind `TONA_DIAG`) that opens `AMDKFD_IOC_SMI_EVENTS` for its own process, subscribes to events
3, 4, 5, 6, 9, 10, 11 and writes timestamped lines (monotonic ns) to the measurement log; no behaviour change. For a
cross-process view (the balloon and the compositor) use a **root** logger with `ALL_PROCESS` only on a day where the
user runs `sudo` consciously; it only reads.
Decision: time from the first `QUEUE_EVICTION` to `QUEUE_RESTORE` (expect >= 100 ms) and the number of cycles per
VRAM allocation of the other process; if the cycles show restore ping-pong, option 2/3 priority rises.

**E2 (iGPU sandbox, no desktop): dmem policy and GTT behaviour on the 2 GB iGPU.**
The iGPU (`card0`, `drm/0000:0f:00.0/vram`, 2 GiB) has **no connected display** [V-local: `card0-*` status
`disconnected`; both monitors are on `card1`]. On it, with `HIP_VISIBLE_DEVICES` selecting the iGPU (whether ROCm 10.1 runs
kernels on gfx1036 is **unchecked**; if not, use the `/dev/dri/renderD128` graphics path with an offscreen Vulkan or GL
client as both roles, one process as "Tona" filling the 2 GiB through a graphics BO, one as "desktop"): create two cgroups
under a **throwaway** `systemd-run --user` scope after a root one-off `+dmem` enable; set `dmem.min` on the "desktop"
cgroup; fill with "Tona"; allocate with "desktop"; record who gets evicted (`dmem.current` per cgroup, journal).
Decision: policy confirmed on amdgpu 7.2.15 -> propose the root boot unit (rank 1); not confirmed (e.g. no eviction
preference, or KFD BOs charged to root) -> keep rank 2/3 as the primary design and treat dmem as a bonus.
Failure mode of this experiment is a failed process on the iGPU only. Do **not** run it on `card1`.

**E3 (iGPU or after E0/E1): GTT headroom.**
On the iGPU: fill GTT through pinned host buffers to within 1 GiB of `mem_info_gtt_total`, then create a graphics client
that needs GTT placement; observe `-ENOMEM`. Raise `ttm.pages_limit` and repeat. This reproduces H1 off the desktop GPU.

**E4 (Tona side, no pressure on card1): elastic give-back prototype in a test harness.**
Unit-test the controller (poll loop, step size, hysteresis) as pure functions in the style of `gtt_budget.h`; drive it with
recorded `mem_info_vram_used` traces (the logs of 2026-11-15) instead of a live card. Live validation only after E0 to E3
and only on the iGPU or with the user present; the reference-platform GPU rule applies (no blocking HIP call while the
megakernel waits; `AGENTS.md` "GPU safety").

**Proposed order**: E0 (next run, free) -> E1 -> E2/E3 on the iGPU -> decide ranks 1, 2, 3 -> implementation tickets. The
design must also say what the Mac backend does (Metal has its own governor); option 2 is backend-neutral in spirit, the SMI part is AMD-only.

## 7. Open points

- Root cause of `-ENOMEM` unproven: H1 (GTT full) vs H2 (restore/locks) needs E0/E1 data at the failure moment.
- Behaviour of 7.2.15 itself (TTM dmem eviction, ticket-walk details): source read was `master`; check with E2.
- Whether `systemd` keeps `+dmem` in `cgroup.subtree_control` of its units: unchecked.
- Whether a restore of Tona's BOs after the desktop's allocation can itself steal VRAM back from the desktop: the restore
  worker validates into VRAM and evicts others only through the same TTM path, with the kworker's (root) cgroup, so under
  dmem it would be unprotected relative to the desktop's `min`; unchecked in practice.
- Sources searched but not readable from this environment: lore.kernel.org and lists.freedesktop.org (403), GitLab (bot
  protection). The amd-gfx 2024-08/09 threads that mention `amdgpu_vm_validate_pt_bos()` and spin loops of failing CS were
  seen as search results only [snippet].

## Sources

- amdgpu CS: `drivers/gpu/drm/amd/amdgpu/amdgpu_cs.c` (`amdgpu_cs_ioctl`, `amdgpu_cs_parser_bos`, `amdgpu_cs_bo_validate`)
  https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/amdgpu_cs.c
- TTM: `drivers/gpu/drm/ttm/ttm_bo.c` (`ttm_bo_validate`, `ttm_bo_alloc_resource`, `ttm_bo_evict_alloc`, `ttm_bo_evict`),
  `ttm_resource.c` (`ttm_resource_try_charge`), `ttm_tt.c` (`pages_limit`) https://github.com/torvalds/linux/tree/master/drivers/gpu/drm/ttm
- amdgpu TTM/KFD: `amdgpu_ttm.c` (`amdgpu_evict_flags`, `amdgpu_ttm_bo_eviction_valuable`, GTT size), `amdgpu_object.c`
  (`allowed_domains`), `amdgpu_gtt_mgr.c` (`amdgpu_gtt_mgr_new`), `amdgpu_vram_mgr.c` (dmem region),
  `amdgpu_amdkfd_gpuvm.c` (`alloc_memory_of_gpu`, `restore_process_bos`), `amdgpu_amdkfd_fence.c`
- amdkfd: `kfd_process.c` (`kfd_process_evict_queues`, `evict_process_worker`, `restore_process_worker`,
  `kfd_process_xnack_mode`), `kfd_priv.h` (`PROCESS_RESTORE_TIME_MS`), `kfd_smi_events.c`, `include/uapi/linux/kfd_ioctl.h`
- dmem: `kernel/cgroup/dmem.c`, `Documentation/admin-guide/cgroup-v2.rst` (DMEM); commits via
  `gh api repos/torvalds/linux/commits?path=...` (controller 2024-12-04, amdgpu 2025-03-27, reclaim callback 2026-07-25)
- dmem.high soft limit patches v2 to v6, May 2026: https://lore.gitlab.freedesktop.org/drm-ai-reviews/review-patch1-20260531-feature-dmem-high-v6-1-20563ecd6dc7@gmail.com/t/ , https://lwn.net/Articles/1072639/ [snippet]
- IGT dmem tests with amdgpu support (Cascardo) [snippet]
- RFC TTM interface for VRAM oversubscription, amd-gfx 2024-04: https://lists.freedesktop.org/archives/amd-gfx/2024-April/107332.html ; eviction improvements: https://lists.freedesktop.org/archives/dri-devel/2024-May/453022.html [snippet]
- Mesa/amd-gfx hits on the message: https://gitlab.freedesktop.org/mesa/mesa/-/issues/10000 , https://lists.freedesktop.org/archives/amd-gfx/2024-September/114700.html [snippet]
- vLLM on ROCm (fraction of total VRAM): https://rocm.docs.amd.com/projects/rocm-cli/en/latest/engines/vllm.html [snippet]
- Local: `research/09-gpu-reset.md`, `research/44-memory-priority.md`, `engine/gpu/gtt_budget.h`, `docs/backends/reference-platform.md`,
  `docs/use/environment.md` (`TONA_ALLOC_TRACE`, `TONA_DIAG`), `/var/tmp/claude/148/res/v-today.*`
