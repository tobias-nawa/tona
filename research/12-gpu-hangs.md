# Complete system hang during compute on RDNA4 (gfx1200)

Agent research, as of 2026-10-24. Nothing measured or reproduced by us. Sources: issue pages, mailing-list
archives, forums, mostly read from summaries. Statements about causes are **hypotheses** unless noted otherwise.
Supplements `09-gpu-reset.md` (there: KFD eviction, #12528, TheRock #9279/#9299, module parameters), which is not repeated here.

## Key statements

1. **There is an almost identical third-party report:** ROCm/legacy-rocm-build #6397 (2026-07-05), RX 9060 XT (gfx1200),
   **Fedora 44**, kernel 7.0.14/7.0.13, linux-firmware 20260622: the whole machine hangs within seconds during rocBLAS GEMM
   (llama.cpp), screen black, network dead, **kernel log ends abruptly, no ring timeout, no reset**,
   `gpu_recovery` does not take effect. Not due to power (36 W, limited to the 127 W minimum: hang anyway), not due to temperature;
   pure FMA load stable. The reporter afterwards also reproduced the hang with **Vulkan/RADV** and suspects the kernel memory path
   or hardware; issue closed by the reporter, repo archived on 2026-10-19. No maintainer comment, no fix.
   https://github.com/ROCm/legacy-rocm-build/issues/6397
   The pattern (log ends without a message, no reset message) matches our incident exactly. It suggests that the
   cause is **not our megakernel design** but a gfx1200 peculiarity that heavy load triggers.
2. **MES doorbell bug on gfx12 (new, July/August 2026):** Two patches by Jesse Zhang (AMD) fix that with
   several concurrent compute queues (also under oversubscription: more ready queues than HQD slots) a dispatch
   **hangs silently, without a dmesg message**; the fence is never signaled. Cause: `unmapped_doorbell_handling` mode 1
   (basic) sets only a coarse "has ready queues" flag, the ringing queue can be skipped because of `work_done`. Fix: mode 2 (basic+).
   Tested on gfx1201.
   - v1 `use basic+ unmapped-doorbell handling for concurrent AQL queues`, 2026-07-30, Acked-by A. Deucher:
     https://ratatoskr.run/amd-gfx/2026/07/17340238/t
   - v2 `fix dropped dispatches under queue oversubscription`, 2026-08-04, Shaoyun Liu "LGTM", no formal Reviewed-by:
     https://ratatoskr.run/amd-gfx/2026/08/17361870/t
   - Merge status and kernel version **unknown**, the archive marks both as "STALE". Affects KFD **and** DRM user queues.
   We have megakernel plus copy streams plus desktop queues: exactly the constellation "several queues, doorbells".
   A silent hang of a queue whose result the CPU/compositor is waiting for fits the picture
   ("no message from router 15" = step not reached).
3. **Diagnosis without a second machine:** pstore (EFI) and a cleanly configured hardware/softlockup path are feasible;
   netconsole needs a receiver (a second device on the network is enough, no "second machine"
   in the sense of a workstation). If the hang is a hard bus/PCIe standstill, **none of these paths writes anything anymore**
   (as in #6397: log ends abruptly). Then only bisection by experiments helps.

## 1. Known causes of complete hangs (classification)

| Candidate | Evidence | Assessment for us |
|---|---|---|
| Silent MES queue hang (doorbell/`work_done`) | patches above (Jul/Aug 2026), gfx1201; "no dmesg error" | **high**: gfx12, several queues, no log. Explains the *hang of one queue*, however, not automatically the whole machine; whole machine only if the compositor (desktop queue) is also affected or a driver lock is held |
| gfx1200-specific hang under heavy load, independent of ROCm | #6397 (Vulkan reproduced), Fedora thread 9060 XT (see 09), CachyOS kernel 7.2.4 Navi 44 gfx ring timeout + resets without recovery: https://discuss.cachyos.org/t/rx-9060-xt-cinnamon-x11-hard-freeze-with-repeated-gfx-ring-timeout-while-running-mtool-under-wine/35614 | **high** as a basic risk of the card/driver state; the engine can only trigger it, not fix it |
| SMU does not answer (clock change) | TheRock #9299 (gfx1201, host freezes, `SMU: No response msg_reg: 12`); Strix Halo thread (SMU deadlock through VPE power gating, community patch `no_vpe_idle_pg`), different IP, not transferable: https://community.frame.work/t/smu-deadlock-system-freeze-on-fedora-43/81795 | **medium**: our load alternates strongly between spin phases (high utilization) and pauses; "short measurements with pauses run at lower clocks" (CLAUDE.md). The clock-change hypothesis of the #9299 reporter fits idle/load alternation. Parameters from forums (anecdotal, gfx1201/other GPUs): `amdgpu.runpm=0`, `amdgpu.ppfeaturemask` without GFXOFF/DS bits (0xfff7bfff or 0xfffd7fff) |
| MES REMOVE_QUEUE/CWSR on preemption | TheRock #9279 (09), Framework threads MES WAIT_REG_MEM (other generation) | **medium**: tends to produce reset messages in the log; our incident had none |
| PCIe atomics / host-memory polling | No relevant reports found. ROCm docs: coherent host memory is not GPU-cached (ok for spin), non-coherent does not allow synchronization inside a running kernel: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_manual.html | **low to medium, unchecked**: spinning on host memory produces permanent PCIe read traffic (at 0.73 µs round trip, up to 32 blocks). No report, but no counter-evidence either. Cheap to test (spin on device memory + occasional host poll by one block) |
| SDMA | the #9299 reporter suspects SDMA+PCIe traffic (MoE offload); no evidence | **low to medium** |
| VRAM overcommit with the desktop | 09: after the pinning change 12/12 stable, `evicted_ms` ~0 during generation | for a hang rather **low** (but our incident predates the change, its cause is open) |
| Display path (PSR, flip_done, sg_display) | Fedora 42 thread 9060 XT: `amdgpu.dcdebugmask=0x10` (PSR off) against occasional freezes: https://discussion.fedoraproject.org/t/occasional-screen-freezes-with-amd/181796 ; 9070 XT flip_done freezes with `runpm=0`: https://discussion.fedoraproject.org/t/drm-flip-done-timed-out-recurring-freeze-artifacting-on-rx-9070-xt-gfx1201-500hz-hdr-over-displayport/198858 | **medium**: two 4K monitors, Wayland; freeze possible without compute. Can only be ruled out by a long control run without the engine |

Further finding: the generally documented escalation on gfx12: `gfx_0.0.0 timeout` → MES reset fails → full
reset, which partly does not come back (kernel 7.0/7.1 RDNA4, gamegpu report; CachyOS 7.2.4). Such cases *log*, however.
Our incident and #6397 do **not** log: a hint of a standstill below the software (bus, SMU, memory controller, lock in the kernel without a watchdog hit).

## 2. Rules for persistent kernels next to the desktop

- **No AMD developer statement found** that names rules, limits or recommendations for persistent/spin kernels next to a
  desktop on the same GPU. The search for "persistent kernel" + MES/compositor only yielded general MES hangs. Honest situation: undocumented.
- Plausible but undocumented principles (own inference): the hardware scheduling layer (MES) shares queues in time or across
  slots; a kernel that never ends only works as long as preemption/CWSR is reliable. The oversubscription patch shows
  that slot allocation on gfx12 still had bugs (as of Aug 2026 unmerged/unclear).
- `lockup_timeout`, `cwsr_enable`, `uni_mes`, `mes`: see 09. New: a "complete" hang is not caught by `lockup_timeout`
  when the CPU itself stands still (#9299: recovery does not take effect, #6397 likewise).
- CU mask (`HSA_CU_MASK`, `hipExtStreamCreateWithCUMask`): applies only to our queue; measured in 09: no protection against timeout.
  No evidence of a protective effect against system hangs.
- Compute partitioning: does not exist on RDNA (consumer) (Instinct feature). Priorities: no documented effect with MES found.

## 3. Diagnosis when the machine stops completely at once

Order by effort and chance of success on Fedora 44, without a second workstation:

| Way | Works alone? | Effect on a hard standstill |
|---|---|---|
| `journalctl -b -1 -k` after reboot | yes | the journal is written only periodically: the last seconds are missing (this is how the log ended in our incident and #6397) |
| efi_pstore / ramoops (panic/oops path) | yes (check backend: `/sys/module/pstore/parameters/backend`; with EFI possibly `efi_pstore.pstore_disable=0`; Fedora/systemd moves entries to `/var/lib/systemd/pstore` at boot). Source: https://blogs.oracle.com/linux/pstore-linux-kernel-persistent-storage-file-system | takes effect **only** if the kernel is still running and triggers a panic/oops. With a standstill of the CPU/bus: nothing. kdumpst note: ramoops stores only a dmesg, truncated to `record_size`: https://gitlab.freedesktop.org/gpiccoli/kdumpst |
| Softlockup/hardlockup panic (`softlockup_panic=1`, `nmi_watchdog=1`, `hung_task_panic`) | yes (in scripts/debug-setup.sh) | triggers panic → pstore **if** CPU cores still service interrupts. With GPU/bus standstill with a blocked CPU (MMIO read hangs) the NMI watchdog helps because it fires independently of the scheduler; useful if the hang is a blocking register access |
| netconsole | needs a receiver on the LAN (e.g. `nc -u -l` on a second device), so no second workstation but a second device | delivers lines until shortly before the standstill; the network driver must still run. In #6397 the network was dead |
| Serial console / debug header | no (board without) | |
| kdump | limited | GPU stays "rogue" after kexec (https://lists.infradead.org/pipermail/kexec/2021-December/023524.html); with a hard standstill kdump does not start (needs a panic) |
| amdgpu devcoredump | only on reset path | the dump lives briefly in RAM (`/sys/class/devcoredump`), cannot be retrieved in a complete system hang; per the amd-gfx discussion the kernel knows only the IB that was running: https://lkml.iu.edu/hypermail/linux/kernel/2305.0/01149.html |
| umr wave dump | only if the system is alive | useless in a total hang |

Conclusion: for an incident of type "log ends abruptly" a hardware state is likely that none of the ways captures. Most likely
benefit: (a) provide NMI watchdog + softlockup panic + efi_pstore (if not already active, see debug-setup.sh) and
(b) `gpu-watch.sh` with a high-frequency **fsync** log (e.g. 100 ms) of `pp_dpm_sclk`, power, temperature, `mem_info_vram_used`,
`evicted_ms`, barrier counter. Shows the last state before the standstill (clock change? power spike?), although the cause stays open.

## 4. gfx1200/1201 specifically, kernel 6.15 to 7.x, firmware

- Reports (all 2026): #6397 (Fedora 44, 7.0.13/7.0.14, FW 20260622, gfx1200), CachyOS kernel 7.2.4 Navi 44 ring timeouts with
  unsuccessful reset, Fedora 43 / 6.18.x 9060 XT persistent ring timeout (09), 9070 XT flip_done freezes (above).
- No report found that documents a fix through a specific kernel or firmware. A blanket statement "6.18 causes
  RDNA3/4 crashes" in the Framework forum is anecdotal.
- Firmware: Debian `firmware-nonfree 20260810-1` and Fedora linux-firmware 20260810 list "amdgpu: update GC 12.0.0 firmware" without details
  (https://tracker.debian.org/media/packages/f/firmware-nonfree/changelog-20260810-1). The oversubscription patch mentions that the timer is tied to MES revision
  `< 0x8b`. An idle-power fix requires MES ≥ 0x8b (from a summary of the patch search). **To check on the machine:** loaded MES version
  (`amdgpu_firmware_info`, root needed, not run).
- Ubuntu SRU LP: #2122659: GC 12.0.1 firmware against power-state transition on navi4x, different chip derivative (12.0.1 = gfx1201), possibly related to the clock-change hypothesis.

## 5. Countermeasures in the engine

All without evidence of effectiveness for this error, therefore to be run as **experiments with A/B**. Prioritized by cost and
prospects:

1. **Reduce spin load:** spin on barriers/mailbox with `__builtin_amdgcn_s_sleep(n)` (small n, a few 64-clock units)
   and poll the host mailbox from **one** block only (the rest waits on a device-memory flag). Lowers permanent PCIe reads and the
   power spikes/clock jumps. Costs latency (measurement: 0.28 µs block→block at present). Addresses the SMU/PCIe hypotheses.
2. **Megakernel on fewer CUs (28 of 32):** measured in 09: no protection against ring timeout; effect on total hang unknown. Low priority, costs ~12% throughput.
3. **Shorten kernels / preemption points:** already implemented in prefill (ordinary kernels). Since the change the hang no longer occurred, which
   supports the hypothesis "continuous load of the megakernel triggers it" (n=1 incident, **no statistics**). The decode megakernel remains the test.
4. **Keep the queue count small:** after the MES doorbell patch the multi-queue constellation is the documented risk. Merge copy streams
   or run copies through the same stream as the compute while a kernel runs; no additional streams next to the megakernel.
5. **Watchdog outside the GPU:** CPU thread that checks the progress counters (step counter in host memory) every ~200 ms and on a standstill >1 s
   sets the abort flag **and** logs the state with fsync. Does not prevent the hang but delivers the last lines before the crash (see 3b).
6. **No continuous-run test without safeguards:** save work, `sync` beforehand; close parallel sessions until clarified.

## 6. Experiments (parameters, the maintainer decides; nothing changed)

A/B each with a control (engine off, desktop only) over hours, otherwise a non-occurrence is not meaningful:
- `amdgpu.runpm=0` and/or `amdgpu.ppfeaturemask` (disable GFXOFF/clock stages), hypothesis SMU/clock change.
- `amdgpu.dcdebugmask=0x10` (PSR off) or `amdgpu.sg_display=0`, hypothesis display.
- Kernel/firmware: `rpm -q linux-firmware kernel`; watch Fedora updates; search the stable changelog for the patch "mes12 basic+" (merge status open).
- Pin the clock (`power_dpm_force_performance_level=manual` / fixed stage) as a test of the clock-change hypothesis (root).
- Own bug report: at Mesa/drm-amd with reference to #6397, if reproducible again with Vulkan load without ROCm.

## Open questions / gaps

- No drm/amd GitLab issue queried directly (the search yields hardly any hits there); the state there is unknown.
- Merge status of the two MES12 doorbell patches and the kernel version containing them unknown.
- Loaded MES/PSP firmware versions on this machine not checked.
- No report found on missing PCIe atomics or host spin as a cause of the hang.
- Whether #6397 was a hardware fault of a single unit remains open (the reporter wanted to test the card in another system; no follow-up).
