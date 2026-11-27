# Desktop VRAM when the display runs on the compute GPU: options, costs, recommendation (#166 part 2)

Research and plan, 2026-11-18. No engine code, no GPU job, no VRAM pressure test. Builds on
[58-vram-instant-giveback.md](58-vram-instant-giveback.md) (#157: why amdgpu failed the compositor),
[docs/architecture/vram-giveback.md](../docs/architecture/vram-giveback.md) and section 5a/5b of
[docs/architecture/expert-tiers.md](../docs/architecture/expert-tiers.md) (#148: the governor). Status tags: [V-local] read or measured on this PC today,
[V-repo] a measurement already in the repository, [web] fetched today (not independently confirmed), [unchecked] named with the check that settles it.

## 1. Answer in short

1. **Today two things protect the desktop, and only one is on.** A fixed reserve `TONA_VRAM_RESERVE_MB` = 1024 MiB is subtracted from the free VRAM when the
   slots are sized at start (`ProgramBuilder::size_slots`, `engine/gpu/mega_build_slots.cpp:70`). The governor `TONA_VRAM_GOVERNOR` (elastic slot generations in
   `engine/gpu/mega_elastic.*`, `mega_vram.cpp`) can give slots back between passes; it is off by default.
2. **The reserve is not what failed on 2026-11-15; the eviction path was.** With 0.6 GB free a +512 MiB second process made the compositor's command
   submission fail (`Not enough memory for command submission`). #157 found why: our 8.4 GB of slots is one buffer object, the kernel can only evict it whole, and
   the GTT (system-memory fallback) had 1.15 GB of room ([V-repo] `measurements/2026-11-16-gtt-headroom.md`), so the compositor got no place to land. A bigger
   reserve hides this; it does not fix it.
3. **Recommendation:** keep 1024 MiB as the interim floor; make chunked slot buffers (`TONA_SLOTS_CHUNKED`) plus GTT headroom the primary protection, after their A/B
   runs; use `dmem.min` for the desktop's cgroup if the maintainer accepts one root step; keep the governor opt-in until it has a signal that arrives before the
   burst (it is reactive by construction). Measure the desktop's real peak first (section 3, plan M1, passive, no GPU risk). Details and costs in sections 5 and 6.

## 2. What exists today, and why the governor is opt-in

| Item | State | Source |
|---|---|---|
| Reserve at start | default 1024 MiB (was 640); cost against 640: decode -0.6 % (code, text), about 4 of 86 slots; 1536 MiB: -1.0 % on the set, up to -2.5 % at 2K write; prefill within noise | [V-repo] BENCHMARKS 2026-11-15, `measurements/2026-11-15-expert-tiers.md` ("VRAM reserve for the desktop") |
| Whether 1024 is enough | **not measured**, the balloon test was not repeated (it ended the desktop session on 2026-11-15) | same file, last paragraph of the section |
| Governor (`TONA_VRAM_GOVERNOR=1`) | the last `TONA_VRAM_ELASTIC_MB` (1536) of the slots are elastic generations (4 slots per layer); marks derived from the reserve R (tight below R/2, release aims at 3R/4); poll of sysfs `vram_total - vram_used` every 250 ms on the engine thread at a safe point (no kernel in flight, `hipStreamQuery`); regrow after 30 s of calm, one generation per 5 s | `expert-tiers.md` 5a; `engine/gpu/mega_vram.cpp` |
| Governor cost | idle: decode -0.3 to -1.0 %, prefill -0.3 %, hit rate -1 to -2 %; shrunk (4 generations = 1.97 GB given back): decode -3.5 %; tokens identical in 83 requests and in deterministic mode; real give-back 1.87 GB in 0.8 s | BENCHMARKS 2026-11-16 (#148), `measurements/2026-11-15-expert-tiers.md` "Final-binary results of the governor" |
| Never remap at the same address | 7 % of remapped chunks read wrong data (29 of 400, kernel 7.2.15; ROCm/TheRock #8744), so regrowth takes fresh address space (512 GiB reserved, about 800 steps) and a ledger refuses any overlap | `expert-tiers.md` 5a |

**Why it is opt-in** (stated in the design note, not guessed):
1. **Reactive.** It restores the reserve after someone took it; "a burst larger than the free VRAM fails the compositor in the first milliseconds, the poll and the
   safe point come after" (drawback 1). So it does not replace the reserve; it needs R to define its marks.
2. **Limited reach.** Only the last 1.5 GB of slots are elastic, and the core keeps at least 32 slots in CPU mode (`n_expert_used x 8` on the GPU); a program that takes
   more can still hurt the desktop (drawback 2).
3. **Standing cost** for a benefit that only shows under pressure: -0.3 to -1.0 % decode idle, a 250 ms wake-up of the engine thread.
4. **Risk class.** A mistake in the same-address rule can write into another program's VRAM (drawback 4); the hard check is on in every build, but the feature
   has no soak time on a loaded card.
5. **The maintainer's rule of 2026-11-16** (no VRAM kept free proactively; memory must be there when needed) put the reserve and the self-calibrating reserve out of
   the final design; they are "interim, to be removed when #157 delivers" (`expert-tiers.md` 5c).

## 3. How much VRAM the desktop takes on this machine

Method: `drm-memory-vram` of every amdgpu client in `/proc/<pid>/fdinfo` (the kernel's per-client accounting), summed by role; the card's total from
`mem_info_vram_used`. Program names stay out of this file: roles are *compositor*, *shell* (the desktop bar and widgets), *other clients*.

**Snapshot, 2026-11-18 03:18 [V-local]** (idle desktop, no browser, no editor, no Tona process, no other GPU job):

| Role | VRAM (MiB) |
|---|---:|
| compositor | 702 |
| shell | 1,600 |
| other clients (terminal, password manager, X server) | 232 |
| sum of per-client values | 2,534 |
| **card `mem_info_vram_used`** | **2,017** |

The per-client values add up to more than the card shows because buffers shared between clients (the compositor and a client map the same buffer) count in each
of them. **The card figure is the one that matters for a reserve; the per-role split only says who holds it.** A passive sampler (10 s, same method, sums only)
runs while this note is written; its result goes into section 3.2.

Earlier readings of the card with only the desktop on it: 1.22 GiB (2026-11-15, [V-repo] `2026-11-15-expert-tiers.md`), 2.22 GiB (2026-11-16, same file, "gv4"),
1.97 GiB today. The desktop's own footprint therefore moves by about 1 GiB between sessions, which is as large as the reserve.

### 3.1 Not measurable without driving the desktop (plan, before any run)

The coordinator asked for "niri plus a browser plus VS Code, idle and at peak". Today no browser and no editor are open, and starting them on the user's session
to provoke a peak is a desktop action, not a GPU benchmark; I did not do it. Plan **M1** (passive, zero GPU risk, needs no one to do anything special):

- *Question:* what is the largest rise of the card's used VRAM that the desktop causes in normal use, over how many seconds, from which starting level?
- *Decision it feeds:* the size of the interim floor (1024 MiB or more) and the amount a give-back has to return in one reaction.
- *Scope:* `python3 tools/vram_watch.py --out work/vram-watch/day1 --seconds 28800` (existing read-only logger, 10 Hz) run for one working day with no Tona
  process on the card, plus the fdinfo role sampler for the split (sums only). No program is started or closed for it.
- *Expectation (with tolerance):* idle 1.2 to 2.3 GiB (the three readings above); a browser with several video tabs and an editor add 0.3 to 1.5 GiB (estimate, no
  source on this machine; the number that M1 replaces); peaks of 3.5 GiB or more would make a 1024 MiB reserve insufficient only if Tona holds the rest of the card.
- *Reading:* the worst rise within a 1 s and a 10 s window is the burst size a reserve or a give-back must cover.
- *Active tooling:* none (no TONA_* variable, no profiler, no Tona process). `hw_report.py` and `quiet_check.py` before and after are not meaningful for a desktop
  log; the file says so and records the card, the kernel and the compositor session instead.

### 3.2 Passive sampler result

A passive sampler (`/var/tmp/claude/166/vramwatch.py`, 10 s interval, sums only) started 03:18 and is set to run 90 minutes. At the time of writing it had
12 samples (03:18:14 to 03:20:06): the card's used VRAM stayed between 2017 and 2017 MiB, compositor 702, shell 1,600, other clients 232
MiB, unchanged. **That is a two-minute idle window, not a peak measurement**; it confirms the snapshot and nothing more. The log stays in `/var/tmp/claude/166/vram.jsonl`
(sums by role, no program names) and is the first input of M1.

## 4. How other engines handle it

Fetched today unless marked; primary sources where reachable, otherwise the weaker source is named.

| Engine | Mechanism | Default | Failure modes and gaps | Source |
|---|---|---|---|---|
| llama.cpp | `--fit` is on by default: test allocations decide what fits, context shrinks first, then tensors move to RAM; dense tensors stay on the GPU ahead of MoE tensors. A user-set `-ngl`, `--tensor-split` or `--override-tensor` turns fitting off for that allocation | `--fit-target` **1024 MiB free per device**, the same margin on every GPU | one static margin at load time, no run-time follow-up; on Windows the system-memory fallback can hide VRAM exhaustion. Not verified: whether other processes' VRAM is subtracted (the discussion does not say) | [web] ggml-org/llama.cpp discussion #18049 (2025-12-15, builds on #16653) |
| vLLM | `gpu_memory_utilization` is a fraction of the **total** device memory for this instance (weights, KV cache, activations); the startup profile separates other processes' memory from the instance's; nothing is rebalanced afterwards; V1 pre-allocates a KV pool for at least one maximum-length request | 0.9 in research/58, 0.92 in `measurements/2026-11-15`; **neither re-verified today** | a fraction of the total, not a measurement of what the desktop uses; a forum answer says it does not coordinate with other processes | [web] docs.vllm.ai `mem_utils`; vLLM forum threads |
| Ollama | free VRAM on AMD Linux is `mem_info_vram_total - mem_info_vram_used` from DRM sysfs (the KFD `used_memory` file is avoided on purpose because it ignores other consumers); refreshed before placement (`RefreshFreeMemory()`) | `OLLAMA_GPU_OVERHEAD` 0 (the user sets a reserve) | if the sysfs read fails, used stays 0 and free equals total; no guard against used above total; no desktop-freeze report tied to it found | [web] ollama v0.9.0 `discover/amd_linux.go`; issue #10883 |
| Kernel | sysfs counters `mem_info_vram_*` and `mem_info_gtt_*`; per-client fdinfo `drm-memory-vram`, `drm-memory-gtt`, and the amdgpu keys `amd-evicted-vram`, `amd-requested-vram` | n/a | what a compositor does when VRAM is full (evict to GTT, slow down, reset) was **not** confirmed by the web pass; on this PC it is the `-ENOMEM` of 2026-11-15, traced in research/58 | [web] docs.kernel.org amdgpu driver-misc; amd-gfx 2024-12 fdinfo series |

Findings: **all three engines use a static margin or a one-time check at placement; none of them polls free VRAM and gives memory back at run time.** "None found" is
a weak statement here (4 searches and 4 fetches, no issue-tracker sweep for AMD desktop freezes in llama.cpp or Ollama; research/58 section 4 had the same
limit). What this repository has and the others do not: the sysfs poll with give-back (the governor), eviction events of the process's own queues (`kfd_smi`,
`TONA_KFD_SMI_LOG`), chunked buffers so that a kernel eviction takes a chunk and not the whole pool.
The common default for a margin on a desktop-shared card is **1 GiB** (llama.cpp) or 8 to 10 % (vLLM); ours (1024 MiB, 6 % of 16.3 GB) is in that range.

## 5. Options and their costs

The cost unit is slots lost, expressed as decode: measured on the 58 GB class, **1 GB of reserve is about 10 of 86 slots (384 MiB, from 640 to 1024, cost 4 slots) and 1 to 2.5 % decode** ([V-repo] reserve runs, and `expert-tiers.md` 5a "about 1 to 2.5 % decode per GB of slots"). The 10 GB and 26 GB
classes have fewer slots and are not covered by these runs.

| # | Option | VRAM given up | Decode cost (measured or derived) | Protects against | Does not protect against | Needs  Fits which priority setting |
|---|---|---|---|---|---|---|---|
| A | Fixed reserve at start (today), 1024 MiB | 1 GiB | -0.6 % against 640 (measured) | desktop growth up to about 1 GiB after the start | a burst larger than the free VRAM; the all-or-nothing eviction | nothing  both (1024 MiB with work first; the smallest value that keeps the compositor alive with server first) |
| A2 | Larger fixed reserve (1536 or 2048 MiB) | 1.5 to 2 GiB | -1.0 to -2.5 % (1536 measured); 2048 about -1.5 to -3.5 % (extrapolated, unmeasured) | a desktop that grows by 1.5 to 2 GiB | same as A | nothing  work first only, and only if M1 shows a large desktop |
| B | Governor on (`TONA_VRAM_GOVERNOR=1`) on top of A | 0 while idle; gives back up to 1.5 GB under pressure | idle -0.3 to -1.0 % (measured); -3.5 % while shrunk | a slow rise: restores the reserve within 0.8 s | a burst that starts below the poll interval (250 ms) or while a pass runs | already built, off by default  work first (a daily cost for the slow-growth case) |
| C | Chunked slot buffers (`TONA_SLOTS_CHUNKED`, 32 MiB chunks, one buffer object each) | 0 | build cost 22 ms at load for 260 chunks (measured micro); decode and restore cost with 260 more buffers **not measured** | the kernel can take 32 MiB instead of 8.4 GB when the compositor asks; the all-queue preemption shrinks to a few chunks | still one eviction fence for the whole process (queues stop for the preemption time, 0.1 to 4 s measured in #157) | built behind a flag (research/58, `vram-giveback.md` (a)); needs the A/B run with the plan first  both (no VRAM cost) |
| D | GTT headroom: the pin budget leaves more than 1 GiB (`cap_pin_budget` margin) or `ttm.pages_limit` is raised | 0 VRAM; 1 to several GiB of pinned RAM less | RAM-side hit rate of the CPU experts (not measured for this change) | the fallback placement of the compositor (H1: the GTT was 31.3 of 32.5 GB full) | the eviction fence cost | the margin is a code change; `ttm.pages_limit` needs root  both (RAM-side cost only) |
| E | `dmem.min` for the desktop's cgroup, Tona unprotected | 0 | none expected; unmeasured | the kernel evicts Tona's buffers first when the desktop is below its protection | the GTT room, the fence cost | root once (enable `dmem` in the cgroup tree and set `dmem.min` on the session slice); not delegated by systemd 259  work first (states the priority to the kernel); not wanted with server first |
| F | KFD SMI eviction events plus sysfs poll as the governor's signal (stage 2 of (c)) | 0 | event thread, no decode cost expected | gives the governor an event after the first eviction, so it shrinks before the next burst | the first burst | built as a logger only (`TONA_KFD_SMI_LOG`)  work first, as the governor's signal |
| G | XNACK or SVM migration | | | | | **not available on gfx1200** (research/58 3.3)  neither (not available) |

Reading the table:
- A and B are the pair that exists. Together they cost about -1 % to -1.6 % decode on the 58 GB class and still fail the burst case that happened.
- C and D attack the cause (H1: GTT full, plus one 8.4 GB victim), not the symptom, and cost nothing in VRAM. Their open numbers are the restore cost with 260
  chunks and the RAM-side cost of a bigger GTT margin; both are single A/B runs.
- E is the only option that states the policy ("the desktop wins") to the kernel in one line; it costs one privileged step on this PC.

## 6. Recommendation

In this order; each step has its decision rule so that none depends on a guess.

1. **Now, no code:** keep A (1024 MiB). Do not raise it on the evidence so far: the loss is real (-0.6 % per 384 MiB step) and the failure it would hide has a
   different cause (section 1, point 2).
2. **M1 (section 3.1), passive and free of GPU risk.** Decides the floor: if the card's worst 10 s rise from the desktop is under 0.5 GiB, 1024 MiB stays; if it is above
   1 GiB, the reserve is too small for this machine and A2 is the stopgap until C and D are measured.
3. **C and D first, because they fix the mechanism.** Plan for the run (not started, written here so that the next worker has it): `scripts/ab.py`, A default against B
   `TONA_SLOTS_CHUNKED=1` (and a second pair with the GTT margin raised), workloads short and set at the 58 GB class, 3 rounds alternating, bit-exact tokens
   (`TONA_DETERMINISTIC=1`, goldens). Expectation: decode within -0.5 %; **decision:** within 1 % and tokens identical, chunked becomes the default; the gain
   it has to show is a different one, so it is checked by the balloon test of step 5.
4. **E if the maintainer accepts one root step** (a boot unit that enables `dmem` and sets `dmem.min` on the user slice). It is the one lever that states the
   priority to the kernel; research/58 ranks it first. It does not remove the need for C (fence cost) and D (GTT room).
5. **Balloon test with the desktop protected:** only after C and D, and only on a day the maintainer chooses (it ended the desktop on 2026-11-15). Plan: a second
   process takes VRAM in 256 MiB steps while a 58 GB class server decodes; pass = the compositor and the server both survive with a 1.0 GB reserve and with 0.6 GB,
   and the eviction events (`TONA_KFD_SMI_LOG`) show chunk-sized losses. Desktop guard rules apply: queue kind exclusive, the monitor off.
6. **Governor stays opt-in.** Default-on costs 0.3 to 1.0 % decode every day for protection against slow growth only; after C and D the kernel does the
   small-burst case itself and the governor's remaining job is a long-lived foreign consumer (a video editor, a game). Revisit when step 5 shows the residual risk,
   and when the KFD SMI signal (F) feeds it.
7. **No floor derived from the desktop's use at start, and no installer floor raise.** Decided by the maintainer on 2026-11-16: no proactive,
   self-calibrating reserve; Tona fills the memory and must give it back reactively; fixed reserves are interim only. So the 1024 MiB stays a fixed number
   that goes away when C, D and E (or the governor with an event signal) carry the protection; M1 only checks that the interim number is not
   obviously too small, it does not turn it into a function of the desktop.
8. **The priority setting decides how much protection is wanted.** "Work first" or "server first" is a user setting (installer and config), not a fixed rule
   (maintainer, 2026-11-18). The last column of the options table says which option fits which setting. With "work first" the desktop wins: C, D and E
   (no VRAM cost) plus the interim floor apply. With "server first" the desktop protection may be lighter, down to the reserve the compositor needs to survive
   and nothing more: A at its smallest value, with C and D (they cost no VRAM), while A2, B and E are not needed. The setting chooses among the options; it adds none.

### 6.1 Result of the C and D runs (2026-11-18, [measurements/2026-11-18-vram-chunks-gtt.md](measurements/2026-11-18-vram-chunks-gtt.md))

Cost only; no pressure test. **D (GTT margin 3072 MiB, 2.4 GB of pins to userptr): tokens identical, decode +0.1 to +0.3 %, prefill equal.** **C (chunked slots, 262 chunks of 34 MB): tokens identical
(goldens, SHAs), prefill equal, decode -0.25 to -0.62 % with every CI spanning zero** (round 1 an outlier at -1.5 %; rounds 3 and 4 within -0.2 %): it qualifies on the point estimates and needs one longer
A/B before it becomes the default. What they buy is still unmeasured: the balloon test (step 5) is not run.

## 7. What is not known (and the check)

| Open | Check |
|---|---|
| Desktop peak with a browser and an editor open | M1, one working day |
| Whether 1024 MiB is enough for the compositor | balloon test after C and D (step 5), a day chosen by the maintainer |
| Restore cost with 260 chunks per eviction | first run of `TONA_SLOTS_CHUNKED` under a balloon |
| Whether `dmem.min` survives systemd's cgroup management | test on the iGPU sandbox first (research/58 E2) |
| vLLM's default and ROCm specifics, llama.cpp's use of other processes' VRAM | read the current docs and source; one hour |

