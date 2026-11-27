# GTT and VRAM headroom in normal operation, and KFD SMI events of our own process (#157, E0 and E1)

Date 2026-11-16. Goal: know how much GTT room Tona leaves in normal operation (hypothesis H1 of
[research/58](../58-vram-instant-giveback.md): the GTT is full, so the desktop's fallback placement fails) and check the
KFD SMI event logger (`TONA_KFD_SMI_LOG`, `engine/gpu/kfd_smi.*`). No speed claim, so no BENCHMARKS.md row.
No VRAM pressure is applied: nothing but the server and five requests runs on the GPU.

## Plan (written before the run)

- **Scope.** One start of `tona-server` (Coder IQ1_M, MTP drafter, `--ctx 262144`, `TONA_RAM_GB=58`, `MemoryMax` 58 GB, port 8099),
  load until `/status` says ready, five chat requests of at most 300 tokens (greedy), five seconds idle, stop by PID. About 2 min
  after the build. One run, no rounds: the question is a level (peak and lowest headroom), not a difference between arms.
  `tools/vram_watch.py` samples `mem_info_gtt_used`, `mem_info_gtt_total`, `mem_info_vram_used` and `mem_info_vram_total` of the
  discrete card at 10 Hz and follows `journalctl -k` for `Not enough memory|Failed to find memory space|Buffer eviction failed|
  amdgpu_vm_validate|ring timeout|GPU reset|evict`.
- **Expectation (from where).** GTT used at the peak 30.5 to 31.5 GB of 32.46 GB, lowest headroom 1.0 to 2.0 GB (maintainer's
  reading today: 31.0 of 32.5 GB with a 58 GB class server; `cap_pin_budget` margin 1 GiB). VRAM used 15.2 to 15.9 GB of 17.1 GB
  total (earlier sessions: 15.7 to 15.9 GB). Kernel log: no matching line (no pressure). SMI log: the START line; QUEUE_EVICTION
  and RESTORE events only around the load (pinning evicts about 1 s, research/measurements/2026-11-11-evict.md), **none during the
  requests**; a `ttm`-trigger eviction during the requests would be a finding.
- **Active tooling.** `TONA_DIAG=1` paths (`TONA_ALLOC_TRACE`, `TONA_KFD_SMI_LOG`), `TONA_VRAM_LOG=1`, `TONA_MEGA_INFO=1`. No profiler,
  no test switch. `tools/hw_report.py` and `tools/quiet_check.py` before and after (in the run directory).

## Result

Run directory `/var/tmp/claude/157/res/` (not kept in the repo). Build of this branch (ctest `kfd_smi`, `evict_counter`, `trace_path`: 3 of 3
passed). Server loaded in 23.4 s (context 262144, KV int8 3.62 GB in RAM, 29.2 GB pinned experts, 3.6 GB of them registered as userptr
because they do not fit the GTT, 64 expert slots per layer, VRAM 12.85 GB at the plan), then five requests (3.8 s for 296, 0.7 s for 34,
1.7 s for 81, 2.0 s for 156, 0.9 s for 47 completion tokens; **no speed claim**, the machine was not quiet: see below), stop by PID.
Sampling: 397 samples at 10 Hz over 39.6 s.

| Quantity | Expectation | Measured |
|---|---|---|
| GTT used, peak (of 32.46 GB total = 30.23 GiB) | 30.5 to 31.5 GB | **31.31 GB** (29.16 GiB); flat at that value from 20 s on, through all five requests |
| GTT headroom, lowest | 1.0 to 2.0 GB | **1.15 GB** (1.08 GiB), reached at 24.7 s (end of the load) and unchanged until the stop |
| GTT used without Tona (before the start, after the stop) | not planned | **0.035 GB**: the desktop's own GTT use `d` is about 35 MB on this PC |
| GTT used during the load | not planned | 9.6 GB at 0 to 10 s, 22.6 GB at 10 to 15 s, 31.3 GB from 20 s |
| VRAM used, peak (of 17.10 GB total) | 15.2 to 15.9 GB | **16.03 GB**; 2.35 GB were in use before the start (desktop), so 1.07 GB stayed free |
| Kernel lines matching the filter | none | **0** (`vw.kernel.log` empty) |
| SMI log | events around the load, none during the requests | see below: as expected, **none after the load** |

**H1 (the GTT is full in normal operation): confirmed as a level.** During normal operation Tona leaves 1.15 GB of GTT, by design of
`cap_pin_budget` (1 GiB margin), against a desktop that itself uses 35 MB. Whether the failure of 2026-11-15 was this H1 is still open:
the failure needs a burst and the GTT at that moment; this run shows only that the room is 1.15 GB, which is far below the 8.4 GB that one
eviction of the slot BO needs as destination.

### KFD SMI events (own process, no root), `kfd-smi-2234556.log`

1,190 lines: 1,072 `UNMAP_FROM_GPU` (trigger 2, unmap from CPU), 42 `QUEUE_EVICTION` (**all trigger `svm`**, none `ttm`), 69 `QUEUE_RESTORE`
(6 of them `R`, restore rescheduled), 3 `MIGRATE_START`/`END` pairs. **All of them fall in the load phase**: the first eviction at 11.6 s
after the logger started, the last event at 24.4 s, the server was ready at about 23.4 s plus the logger's start offset; **none during the
five requests or the idle seconds after them**. The `svm` trigger matches the 3.6 GB registered userptr overflow (an SVM range with an
MMU notifier: every page that the load touches or frees invalidates it), not the TTM path of a VRAM eviction. A burst of 20 evictions
within 0.1 s at 23.16 to 23.24 s is the end of the load. Eviction to restore: 1 to 300 ms per cycle. Delivery latency of an event
(arrival minus kernel timestamp, same clock): mean 3.5 us, maximum 85 us. So the SMI path delivers within microseconds; the cost of an
eviction is in the preemption and the restore (100 ms by design), not in the notification. `evicted_ms` was 0 at the start of the run
(server log line 9).

### Test conditions

`research/measurements` rule: `tools/hw_report.py` output in `/var/tmp/claude/157/res/hw.md` (Ryzen 5 9600X, RX 9060 XT gfx1200 15.9 GiB,
60.5 GiB RAM, kernel 7.2.15-200.fc44, ROCm 10.1; 2.19 GiB VRAM held by others before the run, 0 % GPU busy). `quiet_check.py` before and after: **warning**
both times ("2.19 GiB of 15.9 GiB VRAM held by others"), unchanged; the user was not asked to quiet the machine because the run makes no speed claim
and the held VRAM is the desktop, which is what the question is about.
