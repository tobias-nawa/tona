# Test plan: does the desktop survive a VRAM burst with chunked slots and a larger GTT margin? (#166)

**Plan only, 2026-11-18. Nothing here was run, built or queued. It goes to the maintainer first** (the desktop ended on 2026-11-15 in a balloon test and again on 2026-11-17).
Builds on [research/61](61-desktop-vram-reserve.md) (options C chunked slots, D GTT margin, E dmem), [research/58](58-vram-instant-giveback.md) (why the compositor failed) and
[measurements/2026-11-18-vram-chunks-gtt.md](measurements/2026-11-18-vram-chunks-gtt.md) (cost of C and D: tokens identical, speed within noise; benefit unmeasured).

## 1. The question and why it is hard to test safely

**Question:** with C and/or D on, does a second program that needs VRAM get it without a command-submission failure, and what does Tona pay (stalled queues, decode loss)?

**The constraint that shapes the design.** The kernel only evicts when VRAM is full. With Tona holding the card down to its reserve (about 1 GB free), a second program that
stays inside the free VRAM causes **no** eviction and shows nothing about C or D. The behaviour of interest starts exactly where the 2026-11-15 failure started: the request that
no longer fits. That moment is dangerous for **every** GPU process at once, including the compositor, if it needs memory in the same window. So "a second client that stops far
below the free memory" is safe and answers a different, smaller question (section 3, level 0). The real question needs a **sacrificial compositor** (section 2).

## 2. Design: take the real compositor out of the experiment

Three pieces, none of them the user's desktop:

| Piece | What it is | Role |
|---|---|---|
| **Tona** | the production server on the 58 GB class (`b164t` or the build under test), a decode loop of 400-token requests (`ab.py` workload `short`) | the thing under pressure; arms *default*, *chunked* (`TONA_SLOTS_CHUNKED=1`), *margin* (`TONA_GTT_MARGIN_MB=3072`), *both* |
| **Balloon** | a small HIP client (to write, `bench/micro/vram_balloon.hip`): allocates `hipMalloc` buffers of 64 MiB in steps, pauses 2 s per step, logs the latency of every allocation and the card's free VRAM and GTT headroom; holds the memory until told to release | the pressure; the memory a browser or a game would ask for |
| **Canary** | a small graphics-API client (to write, `bench/micro/vram_canary.*`, Vulkan or the amdgpu libdrm path, headless, no window): keeps a 64 MiB VRAM-only buffer and a command buffer, submits a trivial job every 100 ms and waits for it; logs the latency of every submission and every error | stands in for the compositor: it is a graphics process whose command submission needs its buffers resident, the path that failed for the bar and the terminal on 2026-11-15 (`Not enough memory for command submission`) |

**Where it runs, in decreasing safety (the maintainer picks one):**

1. **No graphical session at all** (recommended): the machine's display manager is stopped for the run (`systemctl stop` of the session from a text console or ssh, at a time the
   maintainer chooses), so no compositor, bar or terminal holds a context to lose. The canary is the only graphics client. A failure ends the canary, nothing else. The cost is the
   maintainer's session: they log out and back in.
2. **Desktop on the iGPU for the run** (if the monitors can be moved to the 2 GB iGPU; `/sys/class/drm` has card1 = the 16 GB card and a second device with 2 GB): the compositor lives on the
   other GPU and cannot be hurt by VRAM exhaustion on the 16 GB card. Not verified that the session can be moved without a restart.
3. **iGPU sandbox** (research/58 E2/E3): dmem and GTT behaviour on the 2 GB iGPU with no desktop on it. Safe, but it is another GPU with another VRAM size, so it only shows the mechanism
   (who is evicted, what error), not Tona's numbers.

The real compositor is **not** part of any level. Whether the real session would have survived is inferred from the canary and stays an inference; one final confirmation by the maintainer,
on their own day, is a separate decision.

## 3. Levels, each with its stop rule (the ladder)

Each level is a separate queue job (kind exclusive, desktop guard on), run in order; a level starts only if the one before ended cleanly.

| Level | Balloon takes | Session | What it shows | Can the desktop die? |
|---|---|---|---|---|
| 0 (no pressure) | up to `free_at_start - 1.5 GiB` (so at least 1.5 GiB stays free, above the 1 GiB the compositor needed on 2026-11-15) | the normal desktop | allocation latency and Tona's decode loss while the card is busy; **no eviction expected**; the governor (opt-in) is the only thing that reacts | no, by construction: free VRAM never below 1.5 GiB; the check runs **before** every allocation |
| 1 (pressure on a canary) | up to the full card: steps of 64 MiB until the canary fails, an allocation fails, or the limit below | session stopped, or the iGPU placement | where the kernel starts to evict, whose BOs, how long Tona's queues stop, whether the canary's submissions fail, per arm | the user's desktop: no (not running). The run ends, at worst, with a failed canary and a restarted Tona |

Level 0 may run on the live desktop and is the smallest useful test: it also measures what a polite foreign program costs Tona. **Level 1 is the one that needs the maintainer's decision.**

### Stop rules (abort at the first one that fires; the balloon frees everything, Tona and the canary are stopped by PID)

Level 1 only (level 0 has its own hard cap above, checked before every step):

1. **A canary failure is the result, not an accident:** the first `Not enough memory for command submission` or submission error of the canary ends the ladder for that arm (the balloon releases at once).
2. **Tona safety (AGENTS.md "GPU safety"):** `evicted_ms` of Tona's queues (`/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms`) rises by more than 5 s in one step, or a pass takes longer than 10 s (the preemption abort is 9 s, `queue_preemption_timeout_ms`): stop. No blocking HIP call is added to Tona; the balloon is a separate process, it never touches Tona's memory.
3. **Hard cap on the balloon:** never more than the whole card minus 256 MiB (the card total of `mem_info_vram_total` read at start), and never more than 20 steps.
4. **Time:** 15 minutes per arm, wall clock; the watchdog (below) kills at 17.
5. **Kernel log:** any of `GPU reset`, `ring ... timeout`, `amdgpu_vm_validate() failed` (these were absent on 2026-11-15) ends the whole job and the day's test; reported to the maintainer.
6. **Watchdog:** a third small process (plain shell, by PID, no `pkill`/`pgrep -f`) checks a heartbeat file written by the balloon every second; if it stops for 5 s, or the ssh/console session that started the job ends, it kills the balloon and the canary and releases the memory.
7. **Manual:** a file `STOP` in the run directory ends the job within 2 s.

## 4. What is measured

Per step (one line of a log, sums and numbers only, no program names): time, balloon bytes held, card `mem_info_vram_used`, `mem_info_gtt_used` and headroom, balloon allocation latency (ms) and result,
canary submission latency (p50, max over the step) and errors, Tona decode tok/s in the step window, Tona `evicted_ms` (cumulative) and the KFD SMI events of its own process (`TONA_KFD_SMI_LOG`: `QUEUE_EVICTION`, `QUEUE_RESTORE`, `UNMAP_FROM_GPU`), the filtered kernel log count (`Not enough memory for command submission`,
`Failed to find memory space`, `Buffer eviction failed`, `amdgpu_vm_validate() failed`; the existing `tools/vram_watch.py` already samples sysfs at 10 Hz and follows the kernel log). Test conditions: `hw_report.py` and `quiet_check.py` before and after, as always.

## 5. Plan per level (the measurement file opens with this, with the values filled in)

- **Scope:** per arm one ladder of at most 20 steps at level 1; arms in the order default, chunked, margin, both and repeated in the reverse order once (2 ladders per arm); level 0 once per arm; the same Tona decode loop in every ladder.
  GPU time: about 15 minutes per ladder at most, 8 ladders for level 1 = 2 hours plus level 0 = 1 hour: not minimised yet; a first pass of default and chunked only (4 ladders) answers whether C changes the outcome.
- **Expectation (with source, to be replaced by what level 0 shows):**

| Arm | Canary outcome | Tona's loss | Source of the expectation |
|---|---|---|---|
| default (one 8.4 GB BO, GTT margin 1 GiB) | fails when the balloon's step no longer fits and the eviction walk finds nothing it may take or the GTT is full (as 2026-11-15) | the whole 8.4 GB leaves VRAM at once (15.9 -> 7.6 GB swings), queues stop for the preemption, then restore | research/58 2.2, 2.3 (H1, H2); `v-today.balloon.txt` |
| chunked | the canary's step is served by evicting 34 MB chunks; fewer or no failures; Tona's queues still stop once per eviction (one fence per process) but only for the chunks' worth | a few chunks per step, `evicted_ms` small | `vram-giveback.md` (a), E5 (262 chunks, 22 ms to build) |
| margin (GTT 3072 MiB) | the compositor's fallback placement has room in GTT; failures at the same balloon size as *default* only if the cause is not H1 | like *default* | research/58 H1: 1.15 GB GTT headroom in normal operation |
| both | no failure up to a larger balloon than *default* | small | sum of the two |

  These are hypotheses with a mechanism, not measured numbers; an outcome outside the table is a finding, and a canary failure in **every** arm points to H2/H3 (research/58), where the fix is not these options.
- **Active tooling:** `TONA_KFD_SMI_LOG` on in the Tona arms (a logger, no behaviour change), `tools/vram_watch.py` at 10 Hz; no profiler; the arm variables only.
- **Decision per outcome:**
  - the canary survives in *chunked* and/or *both* where it failed in *default*: option C (and D if C alone is not enough) go to the maintainer as the protection, the default question (C has a possible -0.3 % decode cost, `2026-11-18-vram-chunks-gtt.md`) is then settled against a measured benefit;
  - the canary fails in every arm at the same balloon size: C and D do not fix the cause; the next candidates are E (`dmem.min`) and a measured look at H2 (restore locks);
  - Tona's `evicted_ms` stays high in *chunked*: chunking does not shorten the stall; the benefit is only on the desktop side.

## 6. What this cannot show, and what has to be built first

- It does not run the real compositor; a surviving canary is evidence, not proof. The last confirmation is the maintainer's decision.
- The balloon and the canary do not exist yet (`bench/micro/`). The balloon is about 80 lines of HIP; the canary needs a graphics API on this machine (Vulkan loader and headers present or not: **unchecked**) or direct libdrm command submission. If neither is workable, level 1 degrades to the balloon alone plus the kernel log of the failing allocation, which loses the "compositor stand-in".
- Level 0 and the ladder do not test the governor's reaction to a burst (that is the TONA_VRAM_FAKE_FREE trace of #148, already done); arm "governor on" can be added as a fifth arm if the maintainer wants the combination.
- The 2026-11-17 crash is not analysed here; if its cause was the same path (VRAM full), this plan covers it; if it was the queue or something else, it does not.

## 7. Decisions needed from the maintainer

1. May level 0 run on the live desktop? (Polite foreign program, free VRAM never below 1.5 GiB, hard cap checked before every allocation.)
2. For level 1: which placement (session stopped, desktop on the iGPU, iGPU sandbox only), and on which day?
3. Is the stand-in canary acceptable evidence, or is a one-time run with the real compositor wanted (his session, his risk, his day)?
