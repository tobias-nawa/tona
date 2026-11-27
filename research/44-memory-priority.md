# 44: Tona next to other work: memory priority, pressure signals, elastic memory

Research of 2026-11-11 for #116 (installer asks working RAM and who wins a shortage; the engine shrinks or is protected).
Read-only research: no code, no GPU, no server. Marks: [V] = primary document fetched and read on 2026-11-11; [S] = secondary
source or search snippet only; [L] = checked on the reference PC (Fedora 44, kernel 7.2.15) with read-only commands;
**unchecked** = not verified, with the check that would settle it. Apple pages came back without body text (title only), so
macOS statements are [S] or from general knowledge and marked **unchecked**.

## Result in short

1. **Signal on Linux: PSI, not `memory.events`.** `memory.pressure` of the server's own cgroup (and `/proc/pressure/memory`)
   with a poll trigger gives an event without polling loops. Unprivileged triggers need a window that is a multiple of 2 s
   [V psi], so the fastest reaction of a trigger is about 2 s plus one window step; add a 1 s sampling of `MemAvailable` and
   the cgroup's `memory.current` for the slope. `memory.events` (`high`, `max`, `low` counters) only counts after the fact and
   is useful as a diagnostic, not as the trigger.
2. **Priority "other programs" without root works through the user manager.** The user manager of the reference PC delegates
   `cpu io memory pids` [L]. `systemctl --user set-property --runtime tona-server MemoryHigh=...` changes the limit at run time
   with no root. `memory.high` throttles and reclaims, never kills [V cgroup-v2]; it is the soft boundary the server can move.
3. **Priority "AI server" cannot be fully done without root.** `memory.low` / `MemoryMin` protection is capped by every
   ancestor's own value [V cgroup-v2]; ancestors (`user@UID.service`, `user-UID.slice`, `user.slice`) belong to root. Without a
   root drop-in the protection is zero. What works without root: `CPUWeight` and `IOWeight` among siblings in the user manager,
   `MemoryHigh` on the *other* user units (they are the same user's units), `ManagedOOMPreference=omit` (partial, below).
4. **The hard part is the pinned overflow on the PC (userptr/SVM range).** Every page the kernel migrates or reclaims in a
   registered range stops all queues of the process until the range is validated again (#103 measurements). So Tona must
   never let the kernel decide which of those pages go: shrinking must be done by the server itself (unregister, free, later
   re-register), in pieces, at a moment between passes. The cost of unregister/register is **unchecked** (plan in "Open
   measurements").
5. **Order of release** (cheapest first): stored conversations (RAM stash), then the page cache of expert files it reads
   (the kernel does this itself; do not fight it), then expert-cache slots in host memory (pinned, GTT part before the
   registered part), then KV/state in RAM, last the arena. Details in the proposal.
6. **macOS** has a real push signal (`DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`: normal / warn / critical) and purgeable Metal
   resources; priority control without root is limited to lowering our own priority and QoS (`taskpolicy`, `setpriority`),
   not throttling other programs. "AI server first" is therefore not offered on macOS (see 4).

## 1. Linux, priority "other programs": detecting pressure without root

### PSI [V psi]

- Files: `/proc/pressure/{cpu,memory,io}` and per cgroup `memory.pressure`, same format. `some` = at least one task stalled,
  `full` = all non-idle tasks stalled. Averages over 10 s / 60 s / 300 s plus a `total` counter in microseconds.
- Trigger: write `some 150000 1000000` (150 ms stall within 1 s) to an opened file, `poll()` for `POLLPRI`. One trigger per
  file descriptor. Window 500 ms to 10 s, checked ten times per window. **Unprivileged users: the window must be a multiple of
  2 s** [V]. A monitor runs only while the resource is stalled, stays active at least one window, and notifies at most once
  per window. `POLLERR` means the source is gone (cgroup removed).
- On the reference PC `/proc/pressure/memory` is world writable (`-rw-rw-rw-`) [L], so a user-space trigger can be armed.
  For the cgroup file `memory.pressure` the owner is root unless the cgroup is delegated; the unit's own cgroup with
  `Delegate=yes` is chowned to the user per systemd [V cgroup-delegation] (unchecked whether the pressure file itself is
  included; fall back to the system-wide file, which includes the desktop, i.e. exactly the "other programs").
- Which one to watch: system-wide `memory` `some` tells "somebody waits for memory", including Tona itself. Own-cgroup
  `memory.pressure` tells "Tona waits". The decision needs both: system `some` high and own `some` low means others suffer
  and Tona is not the victim; this is the case where "other programs first" has to give memory back.
  Both are **unchecked** as an algorithm; the plan is in "Open measurements".
- Baseline caution [L]: on a busy developer machine the `avg300` of system `some` can sit at 30 to 40 %. A fixed threshold on
  the averages would never come back to "calm". Use triggers on stall *amount per window* and the derivative, not the
  absolute average.

### cgroup counters and limits [V cgroup-v2]

- `memory.events`: `low` (reclaim hit although under `memory.low`: protection overcommitted), `high` (throttled by `memory.high`),
  `max`, `oom`, `oom_kill`. Counters, so they need polling (or `inotify` on the file: the kernel signals `modified` on change;
  **unchecked** for unprivileged readers).
- `memory.high`: throttles the cgroup and applies heavy reclaim; "going over the high limit never invokes the OOM killer".
  This is the knob that turns "Tona holds too much" into slowness instead of a kill. `memory.max` kills (the unit already
  sets `MemoryMax` and `MemorySwapMax=0` in `installer/core/unit.cpp`).
- `memory.reclaim`: write-only, proactive reclaim of a cgroup. **Careful:** reclaim in the Tona cgroup also hits the pages of
  the registered range; do not call it on the server's cgroup.
- Per-unit runtime change without root: `systemctl --user set-property --runtime tona-server MemoryHigh=40G`. On the reference
  PC the user manager's `app.slice/memory.high` is owned by the user [L], and `user@UID.service` has `Delegate=yes`
  with `cpu io memory pids` [L]. Fedora's stock default is `Delegate=pids memory` [S packt/redpesk via search], other
  distributions may not delegate `cpu` and `io`: the installer must read `DelegateControllers` and tell the user.

### What systemd-oomd and the desktop do [V oomd, V resource-control]

- `systemd-oomd` polls PSI of units that have `ManagedOOMMemoryPressure=kill` (and swap) and SIGKILLs a whole cgroup (leaf
  or `memory.oom.group`) when the limit (default 60 %, longer than a duration) is exceeded; it is active on the reference PC [L].
  Which user units it watches is set by the distribution (**unchecked** for Fedora 44: `systemctl show -p ManagedOOMMemoryPressure
  user@1000.service user.slice`).
- `ManagedOOMPreference=avoid|omit` ranks the unit's cgroup; for memory-pressure decisions it is honoured if the unit's cgroup
  owner equals the owner of the monitored ancestor cgroup, or root owns it [V resource-control]. For a user unit under the
  user's manager that is the case, so `omit` or `avoid` for `tona-server.service` should work without root (**unchecked**).
  Caveat: this protects Tona from oomd but makes the *other* programs the victims. That is a deliberate effect of priority
  "AI server first", and the installer text must say so ("your browser tab or build may be killed").
- Desktop environments: GNOME and KDE rely on oomd/earlyoom variants or the kernel OOM killer, no cooperative signal to apps
  (**unchecked**; not researched further).

### How fast and how much

- A hang is not caused by the pressure itself but by direct reclaim in the engine thread (#103: 5 GB MemAvailable after a
  copy led to a 48 min stand; 29.6 s evicted per 120 s with 4 KB pages, 2.85 s with huge pages, measurements of 2026-11-11).
  The existing 8 GB reserve (`STASH_RESERVE_GB`) is the empirical "do not go below" line.
- Reaction time budget: kernel throttling starts when allocations hit `memory.high`; a user-space reaction of 2 to 4 s (trigger
  window 2 s) is slower than a burst of a build (several GB in seconds). So the proactive part matters more than the reactive:
  keep `MemAvailable` above the reserve by construction (working RAM, see proposal) and use the trigger only for the slow
  drift. Release in steps of 5 to 10 % of the elastic pool per event, never below the floor.

## 2. Linux, priority "AI server" and throttling others

| Knob | Without root | Effect | Verdict |
|---|---|---|---|
| `MemoryLow=` / `MemoryMin=` on `tona-server` | accepted, but **capped by ancestors** [V cgroup-v2, V resource-control: "needs matching allocations on ancestors"] | Protects only if `user@UID.service`, `user-UID.slice` have a matching `MemoryLow`; `memory.low` is best-effort and relative to siblings; `memory.min` can cause OOM if overcommitted | works only with one root drop-in (installer may print the two lines for `sudo`, never run it silently) |
| `MemoryHigh=` on *others* (`app.slice`, own user) | yes, same user manager [L writable] | Throttles the user's other programs; they slow, nothing is killed | the realistic "throttle others" lever; effect is blunt (all apps in `app.slice`), the user may notice stalls in IDE/browser |
| `CPUWeight=` / `IOWeight=` | yes if `cpu`/`io` delegated [L here, not Fedora default] | relative to siblings in the same slice; put Tona in its own slice and give `app.slice` less | useful for builds that compete for CPU; irrelevant for memory |
| `ManagedOOMPreference=omit` | yes (owner rule above) | oomd does not pick Tona | useful, unchecked |
| `OOMScoreAdjust=` (negative) | negative values need `CAP_SYS_RESOURCE`; positive ones are free | kernel OOM killer preference | for "others first": set Tona to a *positive* score is not wanted; for "Tona first" negative is root-only |
| `nice`, `SCHED_IDLE` on others | only own processes | not possible for programs we did not start | no |
| `memory.reclaim` on the others' cgroup | their cgroups are the user's, writable via the manager? **unchecked** | proactive reclaim of idle pages of the browser/IDE | interesting "soft throttle", needs a test |

Realistic reading of "the other will be throttled temporarily": on Linux without root this is `MemoryHigh` (and optionally a
lower `CPUWeight`) on `app.slice`, applied only while Tona is below its floor and removed with hysteresis. Everything beyond
that (protecting Tona's pages) needs root once. Cost: the user's UI can freeze for seconds if `MemoryHigh` is set below
what the apps need; a lower bound for `MemoryHigh` on others (their current usage minus 10 %, never below e.g. 4 GB) is
required (**unchecked** how the browser reacts).

## 3. Engine side: how others free memory, and what Tona has

| Engine | Mechanism | Source |
|---|---|---|
| vLLM | `sleep(level=1)`: weights to host RAM, KV discarded, over 90 % of GPU memory freed; level 2 drops weights too; explicit call, not pressure driven; CUDA | [V docs.vllm.ai sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/) |
| llama.cpp | host prompt cache `--cache-ram N` (MiB, `-1` unlimited, `0` off), entries evicted to fit, state that alone exceeds the limit is not saved; there were overshoot reports and a crash report with the default under sustained load | [S PR 16391](https://github.com/ggml-org/llama.cpp/pull/16391), [S issue 22128](https://github.com/ggml-org/llama.cpp/issues/22128) |
| Ollama | `keep_alive` unloads the whole model after idle time; no partial shrink under pressure (from general knowledge, **unchecked** against current docs) | general knowledge |
| SGLang | radix cache eviction (LRU) with a size fraction (`--mem-fraction-static`); no OS pressure signal (**unchecked**) | **unchecked** |
| MLX | `mx.set_memory_limit`, `mx.set_cache_limit`, `clear_cache` (from general knowledge, **unchecked**) | **unchecked** |

None of them listens to an OS pressure signal; all use a static budget plus explicit unload. A pressure-driven elastic budget
is new ground, so the design below is ours.

MoE literature on elastic or dynamic expert caches (search snippets only, [S]; none read in full):
[ExpertFlow 2410.17954](https://arxiv.org/abs/2410.17954) (predictive expert cache, LRU unstable),
[DynaExq 2511.15015](https://arxiv.org/abs/2511.15015) (expert precision as a runtime resource: hot experts high precision,
cold experts quantized or offloaded under memory pressure; closest to "shrink without stopping"),
[2509.08342](https://arxiv.org/abs/2509.08342) (per-layer VRAM budget and split ratio),
[ElasticMoE 2510.02613](https://arxiv.org/abs/2510.02613) (scaling devices, not host pressure; virtual-memory migration of
experts, not applicable). No paper found that resizes an expert cache from an OS pressure signal. Open, to read before
building: DynaExq.

### Tona's specifics

- **Pinned memory on the PC** comes in two kinds: GTT buffers (`HSA_USERPTR_FOR_PAGED_MEM=0`: page migration cannot invalidate
  them) and the overflow registered with `hipHostRegister` (SVM range, MMU notifier; one range of 684 MB at ctx 140000
  [measurement 2026-11-11-evict]). Under host pressure the GTT part is safe but cannot be taken away by the kernel either:
  it is *wired* from the OS view. Only the server can shrink it.
- **Unpin / re-pin cost under KFD (unchecked).** What is known: a userptr/SVM invalidation stops all queues until the range is
  revalidated, and the time grows with the range size (measurement 2026-11-11: chunking by 32 MB was planned as test E4).
  What is not known: the cost of an explicit `hipHostUnregister` plus `free` of a slice, and of `hipHostRegister` again
  (set_attr over a range, page faulting in the data on first GPU access). Hypothesis: an explicit unregister is a single
  event at a chosen moment between passes, cheaper than the kernel doing the same per page; test with 32 MB pieces.
- **`madvise`**: `MADV_COLD` / `MADV_PAGEOUT` (Linux 5.4) mark or reclaim private anonymous pages [S, man-pages madvise(2),
  **unchecked** for a registered range]. On a registered range they would trigger the MMU notifier, i.e. the stall we want to
  avoid: use them only on memory that is not registered (stored conversations, the CPU-read expert file mappings, stash
  buffers). `munlock` is irrelevant for KFD ranges (registration is not `mlock`), **unchecked**; it matters only if the server
  `mlock`s something.
- **File-backed expert reads** (experts the CPU reads in place from the model mapping) are page cache: the kernel drops them
  first without any action from us; the price is a re-read from the SSD (decode drops), no stall of queues. This is the
  natural first victim and the reason why page-cache pressure must not touch the registered range (that is the 29.6 s case).
- **Existing elastic parts**: `TONA_PROMPT_RAM_GB` and the stash budget already drop old conversations at the moment of a copy
  (docs/use/environment.md). The new part is dropping them *on pressure*, not only at the next copy.

## 4. macOS

- **Signal:** `dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0, mask, queue)` with
  `DISPATCH_MEMORYPRESSURE_NORMAL / WARN / CRITICAL`; events on level transitions, level read with `dispatch_source_get_data`.
  Page body not delivered by the fetch; content from general knowledge, **unchecked** against [Apple documentation](https://developer.apple.com/documentation/dispatch/dispatch_source_type_memorypressure).
  The kernel level also appears as `kern.memorystatus_vm_pressure_level` (sysctl, **unchecked**).
- `os_proc_available_memory()` is an iOS-family API (**unchecked** whether it is meaningful on macOS); on macOS use
  `host_statistics64` (free, inactive, compressed) and the sysctl above.
- **Releasing Metal memory:** `MTLResource.setPurgeableState(.volatile)` lets the system discard contents under pressure; after
  `.nonVolatile` the returned previous state `.empty` says it was purged and must be refilled
  ([Apple](https://developer.apple.com/documentation/metal/mtlresource/setpurgeablestate(_:)), body not delivered,
  **unchecked**). Fits *recomputable* data: KV of idle conversations, prompt cache, expert cache copies that can be reloaded
  from the file. Not suitable for weights without a reload path.
- **Wired / compressed:** Apple Silicon has unified memory; GPU buffers count as wired while in use, and the GPU working-set
  limit (`recommendedMaxWorkingSetSize`) is by default about two thirds to three quarters of RAM; raising it
  (`iogpu.wired_limit_mb`) needs root (**unchecked**). Compressed memory shrinks other programs' footprint before swapping, so
  macOS soft-pressures the *others* first, which already looks like "Tona first" when Tona's buffers are wired. Hence Tona's
  elastic part on macOS = purgeable buffers plus releasing caches on `WARN`, and a smaller KV/arena on `CRITICAL`.
- **Priority without root:** `taskpolicy -b` / `-c <utility|background|maintenance>` on a process lowers its QoS and I/O policy; the
  tool uses `setpriority(2)` and `setiopolicy_np(3)`; reports say it can only demote, not promote [S eclecticlight 2022/2025,
  [taskpolicy.8](https://keith.github.io/xcode-man-pages/taskpolicy.8.html)]. Lowering *others* is possible only for
  processes of the same user and is weak (it does not limit memory). **Recommendation: on macOS the installer offers
  only "work comes first" and a user-visible QoS setting for Tona itself; "AI server first" is not offered.** Unchecked:
  whether `taskpolicy -b -p <pid>` on foreign processes of the same user needs root.

## 5. Proposal (for maintainer decision)

1. **Budget formula (installer):** `tona_ram = RAM - system_reserve - work_ram`. System reserve = max(4 GB, 8 %) plus the
   existing 8 GB stash reserve that stays a separate headroom, not part of Tona's budget. Written as `TONA_RAM_GB`
   (already driving `MemoryMax`, pins, stash budget). New: `TONA_RAM_FLOOR_GB` (lowest elastic state) and
   `TONA_PRIORITY=work|server`.
2. **Signal (Linux):** one monitor thread: (a) PSI trigger on system `/proc/pressure/memory`, `some 300000 2000000`-class
   (2 s window, the unprivileged minimum), (b) the same on the server's cgroup `memory.pressure`, (c) a 1 s timer reading
   `MemAvailable` and cgroup `memory.current`. State machine: calm / tight / short. *Tight* = MemAvailable below reserve +
   2 GB or system trigger fired and own trigger quiet; *short* = MemAvailable below reserve or own trigger fired. The numbers
   (300 ms, 2 GB) are starting values, **unchecked**: calibrate in #110 with a ballast process (growing anonymous memory
   at 1 GB/s and at 100 MB/s).
3. **Signal (macOS):** `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` WARN = tight, CRITICAL = short, NORMAL = calm; plus
   `host_statistics64` once per second. Same state machine, so the engine code does not know the OS.
4. **Release order** on *tight*: (1) stored conversations (oldest first, down to a quarter of `TONA_PROMPT_RAM_GB`; cheap, nothing
   registered); (2) madvise/purgeable on non-registered caches; on *short* additionally (3) expert-cache slots in host memory,
   removed in 32 MB pieces from the registered overflow first (smaller range, then fewer invalidations), then from the GTT
   part; (4) smaller KV in RAM / smaller arena only at a request boundary (never mid-pass). The decode never waits for a release:
   release runs between passes; if a pass is running, the step is queued.
5. **Hysteresis:** shrink at once, grow back only after 30 s calm and then one piece per 5 s, at most to the pre-shrink size;
   a second *tight* within 60 s doubles the wait (max 10 min). Reason: build bursts repeat; re-pinning costs a queue stall
   (unchecked, see below), so oscillation would cost more than the shortage.
6. **Priority `work` (default):** `MemoryHigh` of the server = `tona_ram` (runtime-adjustable with `systemctl --user
   set-property --runtime`, so the server or `tona-top` can lower it without a restart), `MemoryMax` stays as the hard fence,
   `ManagedOOMPreference` left default. Tona shrinks as above, never the others.
7. **Priority `server`:** (a) the same server side without releasing below `TONA_RAM_FLOOR_GB`; (b) on *short*, the engine
   applies `MemoryHigh` to `app.slice` (computed as current usage minus the deficit, floor 4 GB) and lowers its `CPUWeight`,
   and removes it with the hysteresis of point 5; (c) `ManagedOOMPreference=omit` on the server unit; (d) optional, root once:
   the installer prints the two `MemoryLow` drop-ins for `user@UID.service` and the user slice and the `sudo` command, runs nothing
   itself. Installer text of the question, in English for the repo: "Who has priority in a shortage, the AI server or other
   programs? The other side is throttled temporarily if needed." The answer text in the installer must name the consequence
   (UI stalls for `server`, slower decode for `work`) and that nothing is killed by Tona.
8. **Installer asks and writes:** working RAM (3 suggestions from hardware, optional "measure my usage": sums of
   `MemTotal-MemAvailable` samples with min/median/p95 over a few minutes, no process names), priority, and reads
   `DelegateControllers`; unit gets `MemoryHigh=`, `Slice=tona.slice` (own slice for weights), `ManagedOOMPreference=omit` if
   `server`; config keeps `work_ram_gb`, `priority`; changeable later in `tona-top` (rewrites a drop-in and calls
   `set-property`). If `memory` is not delegated, fall back to the engine-only mode and say so.
9. **Disadvantages:** (i) thresholds are guesses until #110; (ii) each registered-range shrink has an unknown stall; (iii)
   `MemoryHigh` on `app.slice` is blunt and can freeze the UI; (iv) `server` priority without root protects only partially;
   (v) every elastic step lowers decode (experts from SSD, smaller KV) so a "tight" phase of minutes shows as slower, not hung;
   (vi) the stricter the unprivileged PSI rule (2 s windows), the later the reaction, hence the proactive reserve; (vii) the
   state machine adds a thread and code to a reference path: it must be off by default on the reference backend until it is
   measured bit-exact and speed-neutral without pressure (rules 1 and 3 of the repository).
10. **Tests (#110):** ballast process with build-like load (steps of 2 GB every few seconds, plateau, release); metrics:
    longest stall of the decode, evicted time (KFD counter), tok/s decode and prefill during and after, time to recover; PC and
    Mac separately; arms: `work`, `server`, no elastic (today's behavior).

## Open measurements (all unchecked)

1. Stall of one `hipHostUnregister` + free and of the following `hipHostRegister` of a 32 MB piece, on idle server; versus
   kernel-initiated invalidation of the same piece (E4 of the 2026-11-11 plan covers register chunking only).
2. Can the unit's own `memory.pressure` be read and armed by the user? And `inotify` on `memory.events`?
3. Does `MemoryHigh` on `app.slice` act within seconds on a user session, what happens to the desktop, and is `app.slice` the
   right target (some programs sit in `session.slice` or `background.slice`)?
4. `ManagedOOMPreference=omit` effective for a user unit under Fedora 44's oomd setup?
5. macOS: sysctl `kern.memorystatus_vm_pressure_level` versus the dispatch source; purged state after `volatile` under a
   ballast; whether `taskpolicy` on foreign same-user processes needs root.
6. Read DynaExq (2511.15015) and the adaptive-split paper (2509.08342) fully; check SGLang and MLX limits against current docs.

## Sources

- [V] Kernel PSI: https://docs.kernel.org/accounting/psi.html
- [V] Kernel cgroup v2 (memory.low/min/high/events/reclaim, delegation): https://docs.kernel.org/admin-guide/cgroup-v2.html
  (read to character 100000 of 143188; the remainder was not read)
- [V] systemd cgroup delegation: https://systemd.io/CGROUP_DELEGATION/ (does not cover `user@.service` controllers)
- [V] systemd.resource-control(5): https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html
- [V] systemd-oomd: https://www.freedesktop.org/software/systemd/man/latest/systemd-oomd.service.html
- [V] vLLM sleep mode: https://docs.vllm.ai/en/latest/features/sleep_mode/ (search summary)
- [S] llama.cpp host prompt cache PR https://github.com/ggml-org/llama.cpp/pull/16391 and issue https://github.com/ggml-org/llama.cpp/issues/22128
- [S] Fedora default `Delegate=pids memory` for `user@.service`: third-party documentation (redpesk, Packt) via search
- [S] taskpolicy: https://keith.github.io/xcode-man-pages/taskpolicy.8.html, https://eclecticlight.co/2022/01/24/how-you-cant-promote-threads-on-an-m1/
- [S] MoE: https://arxiv.org/abs/2410.17954, https://arxiv.org/abs/2511.15015, https://arxiv.org/abs/2509.08342, https://arxiv.org/abs/2510.02613
- Apple dispatch memory pressure and Metal purgeable state pages: fetched without body, content **unchecked**.
- [L] reference PC: `systemctl show user@UID.service -p DelegateControllers`, `/proc/pressure/memory` permissions, ownership of
  `app.slice/memory.high`.
- Local: `docs/use/environment.md` (RAM, pin, userptr), `research/measurements/2026-11-11-evict.md`, `installer/core/unit.cpp`.
