# Mac: do volatile Metal buffers go before other programs' memory is compressed? (#108)

Question: the arena should fill all free RAM and leave its less used blocks to macOS as volatile (purgeable)
buffers, so the system takes them at once when another program needs memory, instead of Tona keeping a reserve free.
Research (xnu `vm_pageout.c`, `vm_purgeable.c`; Apple developer forum thread 698161; WWDC19 session 606;
llama.cpp `ggml-metal-device.m`): xnu purges a volatile object when its token is ripe (about as old, in the
inactive queue, as the object's last change to volatile) or when the pressure level forces it (warn: groups 0 to 1,
critical: all); it skips wired pages; a residency set wires its allocations. This probe measures it, without the
engine. Maintainer consent for a pressure test with a balloon (6 and 10 GB, emergency brake) given 2026-11-16.

## Plan (written before the run)

- Scope: `tools/metal_purge_probe.mm` (clang++, Metal), 256 blocks of 32 MB (8.6 GB) touched, the first 128 in a
  committed residency set, the second 128 not, all volatile; every 0.5 s for 240 s the count of empty blocks in
  each half and the VM counters. After 10 s `tools/mem_balloon.py` with `--rates 1 --targets 6,10 --phase-s 150
  --hold-s 20 --rest-s 30 --limit-s 300 --swap-brake-gb 2` (two cycles, hard end 5 min, brake at critical pressure
  or +2 GB swap). One run; about 5 min.
- Before: free about 14.3 GB, 18.4 of 19.5 GB swap in use, pressure normal. With the probe about 5.7 GB free.
- Expectation:
  - Without pressure (first probe, 1 GB, 10 s): no block empty; calls cost 0.3 us (state), 0.9 us (residency
    commit); the in-set half counts as purgeable wired (0.57 GB): measured 08:03.
  - 6 GB balloon: shortfall about 0.3 GB plus the kernel's free target: the out-of-set blocks are purged first, at
    least 1 GB of them; compression by the balloon under 0.5 GB; no in-set block empty.
  - 10 GB balloon: all 128 out-of-set blocks empty (4.3 GB); in-set blocks none (wired); compression of other
    programs 0 to 2 GB for the rest of the shortfall.
  - Outside: nothing purged and the compressor grows (tokens not ripe: then the mach API's
    `VM_PURGABLE_ORDERING_OBSOLETE` is the next step); or in-set blocks purged too (residency does not protect;
    then the set needs no removal step).
  - Every block either kept with its contents or empty and zero; none inconsistent.
- Active tooling: none in the engine (it does not run); the balloon's per-second log; `caffeinate`.

## Result

Logs: [probe](2026-11-16-mac-purgeable-probe-probe.log), [balloon](2026-11-16-mac-purgeable-probe-balloon.log)
(one line per second). The balloon ran all phases (173 s), no brake: pressure stayed normal, swap +0.00 GB.

| Time (probe) | Balloon | Free GB | Empty blocks in set / out | Purgeable GB | Compressions since start |
|---|---|---|---|---|---|
| 0 s | 0 | 6.56 | 0 / 0 | 8.61 (wired 4.33) | 0 |
| 5 s | 0 | 6.57 | 0 / 0 | 8.60 (wired 0.03) | 0 |
| 20 to 35 s | 6 GB | 0.35 | 0 / 0 | 8.61 | 0 |
| 75.2 s | 10 GB | 0.27 | 13 / 0 | 8.17 | 0 |
| 76.2 s | 10 GB | 0.27 | 45 / 0 | 7.09 | 0 |
| 100 s to end | released | 10.19 | 45 / 0 | 7.09 | 0 |

End check: 211 blocks kept with their contents, 45 empty and zero, none inconsistent.

- **macOS took the volatile blocks before compressing anything**: 1.5 GB purged, 0.00 GB compressed over the whole
  run, at normal pressure. The 6 GB balloon fit into free RAM and file cache without a purge; the 10 GB balloon
  purged 45 blocks within about 1 s, as fast as it grew (1 GB/s).
- **The residency set does not protect a volatile block**: all 45 purged blocks were in the set, the oldest
  volatile ones (marked first, indices 0 to 44). The set wires its allocations only at first (4.33 GB purgeable
  wired at 0 s, 0.03 GB after 5 s). Outside the plan, which expected the set to protect them: a finding. For the
  arena: the purge order is the order of the volatile marks (FIFO), whatever the residency; a block must be
  nonvolatile while a command buffer may read it, then volatile again; the residency set can keep all blocks.
- **Calls are cheap**: 0.3 us per state change, 0.2 us per query, 1.0 us per residency add or remove with commit.
  For 600 blocks per pass that is under 0.4 ms against a decode pass of about 30 ms; toggling only the blocks a pass
  uses is less.
- **Detection works**: an empty block reads `MTLPurgeableStateEmpty` on the way back to nonvolatile and zero
  contents; a kept one keeps its contents.
- Not measured: a volatile block read by the GPU (undefined by the header, avoided by design); purge order under
  warn or critical pressure; how the purge treats our blocks against other programs' file cache (the 6 GB balloon
  took free pages and file cache first).

Test conditions before:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.07, 2.21, 2.13
- CPU use by other programs over 0.5 s: 4.8 % of all threads
- sandbox: no
- Commit: `c0fc074` (dirty tree)
```

```
- Quiet check (2026-11-16 08:03:35): ok
```

Test conditions after:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.42, 1.82, 1.97
- CPU use by other programs over 0.5 s: 6.1 % of all threads
- sandbox: no
- Commit: `dc0b8cd` (dirty tree)
```

```
- Quiet check (2026-11-16 08:08:42): ok
```
