# The stash copy that stood for 48 minutes (#84)

Question: why did the engine thread stand in `hipMemcpy` of `Mega::stash` for 48 minutes, and how do we make sure no
copy of a stash or restore can stop the server for good?

Build: `work/build/w84` (branch `84-stash-hang`, from main 7d5b00d), reference `work/build/merged`. Server on port 8095,
`--mtp`, two slots, under the GPU lock and `systemd-run --scope -p MemoryMax=54G -p MemorySwapMax=0`. Raw data (logs,
sampler files, scripts): `/var/tmp/claude/w84/`, not in the repository.

## Reproduction

Three independent texts of about 520K tokens each (`--ctx 540672`, `IE_SEQS=2`), one after the other. The first two fill
both slots; the third needs a slot, so `stash_current` copies the running conversation (7.6 GB) to the host. The reserve
of 64895f1 is switched off with `IE_TEST_STASH_RESERVE_GB=0`, because with it this case is no longer stashed at all.
A watchdog ends the run after 300 s without progress (gdb of all threads, then the server is ended by PID).

| Run | Build | What | Result |
|---|---|---|---|
| r1 | without limit | 3 texts, reserve 0 | text 3: connection closed after 816 s (progress stood still) |
| r2 | without limit | the same, with gdb | watchdog after 300 s; engine thread in `hipMemcpy -> hsa_signal_wait_scacquire` |
| r3 | with diagnosis | the same | GPU memory access fault in `k_moe_sum` (a different problem, tracked in #72) |
| r3b | with diagnosis | the same | pool copy 0 stalled 15 s, no end after 120 s; the stash buffer is freed at 15 s |
| r5 | with diagnosis | the same | host buffers freed at 15.7 s and 18.6 s: the copy ended 6.4 s and 11 s later |
| r6 | with the fix | the same | copy ended 8.6 s and 19.4 s after the release; both requests went on |

The host during the stall (r1, `mem.txt`): MemAvailable 5.2 to 13 GB, swap 2.5 to 5 GB used by other processes, cgroup
shmem 43 GB (pinned memory), the stash adds 8.9 GB of anonymous memory.

## Finding

The destination of the copy is pageable RAM (`std::vector`). Allocating and touching its 8.9 GB with only 5 GB of host
MemAvailable left puts the host into direct reclaim (`pgscan_direct` and `pgmigrate_success` rise in `mem.txt`). During
that time the GPU of the process makes no progress with the copy: the engine thread polls `hsa_signal_wait_scacquire` at
100 % CPU, the GPU is idle, `journalctl -k` is empty. Handing the memory back ends the stall within 6 to 20 s (r5, r6).
Without handing it back the copy stood for 13 minutes and more (r1, r2, r3b). Root settings were ruled out.

Not established: why the reclaim of the host blocks the DMA of this process (candidate: the driver evicts the queues of
the process under memory pressure, see `2026-10-31-barrier-aborts.md`). It is not needed for the fix.

## Fix

- `engine/gpu/guarded_copy.{h,cpp}`: the copy goes through a ring of pinned bounce buffers (4 x 16 MB) on its own
  non-blocking stream, all regions of a stash in one pipeline, the CPU copy on 4 threads. Per chunk an event is polled with a limit of `2 s + chunk / 250 MB/s` (`backend/stall.h`).
- A stalled stash gives its host memory back, waits up to 60 s (`STALL_GRACE_S`, from the 8.6 to 19.4 s measured) and
  is dropped; the request goes on. A stalled restore invalidates the slot and the prompt is computed again.
- If the copy does not end within the grace period the process exits with code 75 (`EX_TEMPFAIL`), so the service
  restarts it. A watchdog thread does that because the enqueue of a host-to-device copy blocks the host as long as the
  stream stands (3.5 s measured), where polling cannot help.
- Buffers are never freed while a DMA can still run: the source of a restore stays held by the caller until the copy
  has ended; a stash is released only to make the stalled copy end, and the engine thread does nothing else on the GPU
  until it has.
- Stash and unstash run on the engine thread between passes; `k_mega` is not running then, so the polling is not the
  blocking HIP call that is forbidden while the kernel waits for the CPU.
- The reserve of 8 GB stays: the case that hung leaves 5 GB, the stash needs 8.9 GB, and with the reserve this case is
  not stashed. The measurements give no better value.

## Timeout paths with the test hooks

The stream waits for a host word (`hipStreamWaitValue32`) before the copy; the word is set after n ms or never. No
kernel spins, the copy stands as in the hang.

| Run | Hook | Result |
|---|---|---|
| hk1 | stash, word after 3.5 s | stall logged, host memory released, ended 1.4 s later, "not stashed", next request normal |
| hk2, hk2b | restore, word after 3.5 s | the enqueue blocked the host for 3.5 s (inside the limit of the watchdog), restore completed, request served from RAM |
| hk4 | stash, never set, grace 5 s | exit with code 75 |
| hk5 | restore, never set, `..._SAFE=1`, grace 5 s | `FATAL ... Exiting with code 75`, process gone at once, exit code 75 |
| hk6 | stash, never set, `..._SAFE=1`, grace 5 s | the same |
| hkb | stash, never set, grace 5 s, **without** `_SAFE` (queue never ends) | exit code 75 at once; no process left in state D, `journalctl -k` empty, VRAM free; the next server started and answered (bit-exact run below) |

Re-run on the final build (hk1n, hk2n, hk5n, hk6n): the same results (stash stall released and ended 1.4 s later;
restore enqueue blocked 3.5 s and went through; both exit tests ended with code 75).

The restore hook did not reach the path "restore stalled, slot dropped, prompt computed again" live. If the stream
stands, the enqueue of the next chunk blocks the host before any event can be polled (hk2b, hk2n: 3.5 s blocked), so
only the watchdog (`Armed` in `issue`) ends the process with exit code 75. The event polling is the second line, for a
DMA that stands without the enqueue blocking. No further code change for it.

## Checks

| Check | Result |
|---|---|
| bit-exact, `IE_DETERMINISTIC=1`, 200 tokens, 30K | SHA `065922c2ff602991`, w84 and merged |
| bit-exact, 130K | SHA `439aad472c59ae6f`, w84 and merged |
| bit-exact on the final build (2efcb9b) | both SHA identical again |
| golden decode (csv, rust, explain) | identical |
| golden prefill with dflash | identical |
| golden server (20 cases, including `reuse-*`, `ssd-*`) | 20 of 20 identical |
| golden prefill without drafter | DIFFERS, on w84 and on merged alike: not caused by this change |
| unit test `stall` (label `unit`) | passes |

Prefill and decode rates of the bit-exact runs (rate of one run each): 30K prefill 1800 (w84) and 1807 (merged) tok/s,
decode 75.4 and 74.5 tok/s; 130K prefill 1660 and 1713 tok/s (single runs, no spread; the speed of prefill is not what
this change touches).

## Speed of stash and unstash

Server with `--ctx 40000` (30K) and `--ctx 140000` (130K), texts t11, t12, t13, t11; the third text stashes the first
conversation (stash), the fourth restores it from RAM (unstash). Times from the server log (`stashing n ms` and
`conversation from RAM in n ms`). Three rounds, builds alternating (merged, w84, merged, ...). The first version of
the fix (one call per region, 2 x 32 MB, zero-filled host vectors) was slower than merged by 30 to 70 ms per step:

| Conversation | Step | merged (ms) | first fix (ms) | final (ms) | Final against merged |
|---|---|---|---|---|---|
| 30K (651 MB) | stash | 100 (100 to 101) | 141 (140 to 142) | 59 (59 to 60) | -41 ms, -41 % |
| 30K | unstash | 122 (121 to 122) | 188 (188 to 190) | 98 (96 to 99) | -24 ms, -20 % |
| 130K | stash | 398 (398 to 400) | 428 (425 to 429) | 360 (359 to 362) | -38 ms, -10 % |
| 130K | unstash | 498 (497 to 501) | 532 (532 to 536) | 460 (459 to 462) | -38 ms, -8 % |

Median and range of three runs.

### Where the time went (`IE_COPY_PROF=1`, 30K, stash of 651 MB, first version)

| Part | ms |
|---|---:|
| host allocation (`resize`: page faults and a pass of zeros over 651 MB) | 80 to 100 |
| CPU copy from the bounce buffer into the stash | 30 to 33 |
| wait for the DMA | 15 to 29 (serial: 27 calls, each one waited for its own last chunk) |
| enqueue | 0.3 to 8 |
| wait for idle streams | under 1 |

The suspicion of many small serial pool copies was right for the wait (16 ms); the larger parts were the zero fill and
the single-thread CPU copy. The copy is bound by DRAM bandwidth and page faults (the DMA, 0.3 ms per 16 MB, is never
the limit). Every optimization alone (30K, stash / unstash in ms; each row adds to the one above, same server, one run
each except the first and last):

| Step | stash | unstash |
|---|---:|---:|
| first fix (27 calls, 2 x 32 MB, 1 thread, zeros) | 141 | 188 |
| + one pipeline over all regions, 4 x 16 MB (1 thread, zeros) | 136 | 172 |
| + chunk 32 MB or 8 x 8 MB instead (no change) | 141 / 135 | 177 / 175 |
| + host vectors not zero-filled (1 thread) | 107 | 143 |
| + 2 threads | 76 | 114 |
| + 3 threads | 64 | 102 |
| + 4 threads (final) | 59 | 97 |
| + 6 threads (no gain, not kept) | 58 | 96 |

The chunk size and the number of buffers change nothing, so they stay constants. More than 4 threads gain nothing.

### Safety of the uninitialized host memory

`HostSnap` is complete or empty (comment in `backend/backend.h`): `Mega::stash` writes state, KV and pool in full by one
copy and resets the snapshot on a stall and on any exception; `stash_current` drops the entry if stash throws;
`read_checkpoint` resets the snapshot when a read is short or the tokens differ. The checkpoint file is written only
from a snapshot `stash` returned, and `unstash` reads only complete ones (`snap_fits` checks the sizes).
