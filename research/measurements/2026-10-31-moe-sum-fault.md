# GPU memory fault in `k_moe_sum` during prefill (#72)

Question: why does the server now and then die with `HSA_STATUS_ERROR_MEMORY_FAULT` in `k_moe_sum` during a long
prefill, and how do we stop it?

Status: **not reproduced, cause not found.** This file holds the analysis, the diagnostics added and the negative
results, so the next occurrence can be assigned to a buffer at once.

Build: `work/build/w72` (branch `72-moe-sum`, from main fb01c0e) plus the diagnostics below. Server on port 8095, `--mtp`,
two slots, always under the GPU lock. Raw data (logs, traces, scripts): `/var/tmp/claude/w72/`, not in the repository.

## The two crashes

| | 2026-10-29 00:12 | 2026-10-31 10:24 |
|---|---|---|
| build | main 3a269e8, `--ctx 131072`, 84 slots, loan 4,350 MB | w75, `--ctx 140000`, 81 slots |
| sequence | 65K request, then 127,838 tokens | 30K request, stash (643 MB), then 130K |
| grid of `k_moe_sum` | 15,728,640 = 6144 x 2560 (full chunk) | 13,107,200 = 5120 x 2560 (chunk 5120) |
| fault address | 0x7f61284cc000 | 0x7fd0f6204000 |
| kernarg address of the dispatch | 0x7f6668910800 | 0x7fd648738100 |

Both addresses lie 21.0 and 21.3 GiB below the kernarg pool of the dispatch. The kernel log has no amdgpu line for either
crash any more and both CPU core dumps are gone, so the address could not be assigned to a buffer.

## Analysis without the GPU

Pointers of `k_moe_sum` (`prefill_moe_run.cpp`, `moe()`): `b.O` (C x ne x d floats), `b.rowmap` and `b.sw` (C x ne), `b.sho`
(C x d), `b.shi` (C) and `B.y`. All come from the prefill arena. In the layer-by-layer prefill the arena is the first range of
the borrowed expert slots (`hipMalloc`, VRAM); no pinned host pointer is among the arguments.

- The arena size (`prefill_arena_bytes`), `pf_layered_need` and `borrow_slots` add up the same items (arena, residual
  streams, features, KV staging, two layer buffers); the arena lies at the start of the loan, so no sizing error can push it
  past the end of the slot allocation. Trace of the run a1: every loan ends exactly at the end of the slot allocation
  (0 bytes slack), the arena starts at the first loaned layer.
- `rowmap[a] = off[e] + fill[e]` stays below `n * ne` as long as the router counts add up. A router row without a finite
  logit (`be = 1 << 30`) would fault in `k_route` or `k_scatter` first, not in `k_moe_sum`.
- `rowmap` is the only index array that lives across the long expert GEMMs, so `k_moe_sum` is the first kernel to fault when
  something overwrites the arena in that window.

## Diagnostics added

- `IE_ALLOC_TRACE=<file>`: every `hipMalloc`, `hipHostMalloc`, free (wrappers in `gpu/alloc_trace.h`), and per group the
  borrowed ranges (loan, arena, residual streams, KV staging, layer buffers) with address, size, call site and time.
  `tools/alloc_trace_lookup.py TRACE ADDRESS` names the buffer of a fault address, live or freed, or the distance to the
  nearest one.
- `IE_PF_CHECK=1`: kernel `k_moe_check` checks the router's experts (before the scatter) and the `rowmap` against the
  count and offset of its expert (after the scatter and again before `k_moe_sum`), the host checks that the counts add up to
  `n * ne`. Violations are logged with layer and chunk start, the run goes on. Unit test `test_pf_check` (label `unit`) covers
  the bounds arithmetic in `gpu/prefill_check.h`.

## Reproduction attempts (all clean)

Sequence: alternating small and large prompts of different content (so every request stashes the other conversation),
`IE_PF_CHECK=1` (except e1), `IE_ALLOC_TRACE`, ctx as noted. Decode after each prompt: **16 tokens in a1, b1, c1 and d1**,
200 tokens (MTP) in e1, m1..m40 and s1..s2; the crashes followed 200-token decodes.

| Run | ctx | Prompts | Pressure | Result |
|---|---:|---|---|---|
| a1 | 131072 | 4 x (30K, 125K), 16 tokens | none | 8 prefills, 0 violations |
| b1 | 131072 | 6 x (30K, 125K), 16 tokens | Vulkan VRAM churn (4 x 256 MB held 6 s, free 6 s) | 12 prefills, 0 violations |
| c1 | 131072 | 8 x (30K, 125K), 16 tokens | VRAM churn + 6 GB random RAM touch + model file read in a loop | 16 prefills, 0 violations; 43.6 M pages migrated, 208 K compaction stalls, 1,049 s evicted per KFD node; one prefill took 737 s |
| d1 | 140000 | 10 x (30K, 130K), 16 tokens | VRAM churn + file read | 20 prefills, 0 violations; 705 s evicted |
| e1 | 140000 | 12 x (30K, 130K), 200 tokens | none, `IE_PF_CHECK=0` (timing as in production) | 24 prefills, no fault; 7.7 s evicted |
| m1..m40 | 140000 | 40 server starts, each 8K + 20K twice, `IE_PF_LCHUNK=2048`, `IE_PF_GROUP=4096` (five groups of borrow and return per prompt) | none | 160 prefills, 0 violations, no fault |

Total: more than 240 layer-by-layer prefills without a fault or a violation, 40 fresh server starts. VRAM used was
16.4 to 16.97 GB of 17.1 GB (reserve 640 MB).

## What the runs rule out

- A wrong `rowmap`, `sel` or count at any of the three check points, also under heavy eviction (c1, d1).
- A sizing error of the borrowed range (the trace shows consistent ranges in 56 groups of run a1).
- Page migration and queue eviction of the process on their own (c1 and d1 had 1,049 s and 705 s of `evicted_ms`).
- VRAM churn by a second client on its own.

## Open

The two crashes happened with 81 and 84 slots; the runs here had the slot count that the free VRAM gave (84 at both contexts, see `VRAM:` in the server logs). Remaining candidates
that the runs could not test: a fault that depends on the state after a long MTP session (the crashes followed 200-token decodes
of a real client), a rare race with the CPU running ahead (the check adds syncs; e1 and m1..m40 had them off or short prompts), and
a driver-side fault in the page tables of the 8.25 GB slot allocation. The next crash of a server started with
`IE_ALLOC_TRACE` and `HSA_COREDUMP_PATTERN` will settle it.

## At the next fault

Start the server like this (nothing else changes, the trace costs one `fprintf` per allocation and per group):

```sh
ulimit -c unlimited          # RLIMIT_CORE=0 suppresses the GPU core dump as well (tested)
IE_ALLOC_TRACE=/var/tmp/claude/trace-$(date +%s).txt IE_PF_CHECK=1 \
HSA_COREDUMP_PATTERN=/var/tmp/claude/gpucore-%p.core ie-server ...
```

- Files go to `/var/tmp/claude/` (disk, not the RAM-backed `/tmp`). The trace is a few hundred KB per server run (2,200 lines
  for 8 prefills). `ulimit -c unlimited` also lets the abort write the CPU core dump (25 GB each, via `coredumpctl`); delete
  it afterwards if it is not needed.
- Size of the GPU core dump: **155 MB for a trivial test binary** (`/var/tmp/claude/w72/cd/fault.hip`, one kernel, 1 MB
  allocation). For the server it is not measured; the dump contains memory segments of the process, so it can be much larger
  (the 8.25 GB slot allocation alone). Plan for several GB and check `df /var/tmp` first (519 GB were free).
- `HSA_COREDUMP_PATTERN` **does take effect**: a deliberate fault in a test binary (kernel writes to the unmapped address
  0x7f0000100000, the same `HSA_STATUS_ERROR_MEMORY_FAULT` and "Page not present" message as the crashes) wrote
  `gpucore-<pid>.core` with `ulimit -c unlimited` and nothing with `ulimit -c 0`. Without the variable the runtime tried to pipe
  the dump to the kernel handler, which failed with "Broken pipe" in both crashes. With the iGPU visible
  (`ROCR_VISIBLE_DEVICES` unset) `rocgdb` rejects the dump ("architecture gfx1036 not supported"); the dump of a run with
  `ROCR_VISIBLE_DEVICES=0` or `HIP_VISIBLE_DEVICES=0` lists the RX 9060 XT.
- **Untested:** whether `rocgdb` shows the faulting wave, its PC and registers. In the test dump `rocgdb -c <file>` printed
  `info agents` (gfx1200) but `info queues`, `info dispatches` and the wave list were empty, so the dump may not carry the wave
  state of a runtime-aborted queue. Do not rely on it; the allocation trace is the main tool.
- Assign the address:

```sh
tools/alloc_trace_lookup.py /var/tmp/claude/trace-<id>.txt 0x7f61284cc000
rocgdb -batch -ex "info agents" -ex "info queues" -ex "info dispatches" -c /var/tmp/claude/gpucore-<pid>.core
```

  The lookup prints every allocation or borrowed range that contains the address (with the time it was freed, if it was) or
  the distance to the nearest one. The trace times are milliseconds since the first traced call of the process; the server
  log's heartbeat (`IE_HEARTBEAT`) and the group lines (`IE_PF_DEBUG`) tell which group was running.
- Also keep `journalctl -k --since <start> | grep -i amdgpu` (the page-fault line with the faulting client) and the
  `IE_PF_CHECK` lines of the log.

## How to test the page-table hypothesis (not run)

Hypothesis: with the VRAM nearly full (16.4 to 16.97 of 17.1 GB used, reserve 640 MB) the driver sometimes leaves a page of
the 8.25 GB slot allocation unmapped during a validation. At about one crash in 20 server starts and 42 clean starts here, a
meaningful test needs several hundred starts. Procedure: `multi.sh N` (`/var/tmp/claude/w72/multi.sh`, one start = two
8K and two 20K prefills in groups of 4096) with `IE_VRAM_RESERVE_MB=640` (default) against `IE_VRAM_RESERVE_MB=1500`, 200
starts each, alternating in blocks of 20, counting faults. A difference needs about 10 faults in the default arm; at the
reported rate that is 200 starts, so with the rate here (0 in 42) it may take far more. Not run: it would cost expert slots and gains
nothing at this rate.

## Cost and bit-exactness of the diagnostics (without the variables)

Build: `work/build/w72` = commit e02c73d (`ninja -n` empty afterwards), reference `work/build/merged` = main fb01c0e.
Scripts and logs: `/var/tmp/claude/w72/` (`bx72.sh`, `gold72.sh`, `speed72.sh`). All under the GPU lock, desktop on the same card.

| Check | Command | Result |
|---|---|---|
| Bit-exact | `IE_DETERMINISTIC=1`, ctx 140000, 200 tokens, 3 rounds alternating w72 / merged | SHA `065922c2ff602991` (30K) and `439aad472c59ae6f` (130K) in all 12 runs |
| Golden decode | `scripts/golden.sh compare work/build/w72` | csv, rust, explain identical, exit 0 |
| Golden prefill | `GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash scripts/golden.sh compare work/build/w72` | long identical, exit 0 |
| Golden server | `scripts/golden-server.sh compare work/build/w72` | 20 of 20 identical, exit 0 |
| Unit | `test_pf_check` (new), `just test-cpu` not rerun as a whole | ok |

Speed, 3 rounds alternating (prefill: server ctx 140000, 200 tokens after the prompt; bench-set: ctx 32768, MTP):

| | merged | w72 |
|---|---|---|
| prefill 64K (65,124 tokens), tok/s | 1787, 1787, 1787 | 1787, 1788, 1787 |
| prefill 128K (130,981 tokens), tok/s | 1709, 1709, 1709 | 1709, 1709, 1709 |
| bench-set code, decode tok/s | 88.2, 88.3, 88.3 | 88.1, 88.1, 88.1 |
| bench-set text, decode tok/s | 64.0, 64.0, 64.0 | 63.9, 63.9, 63.9 |

Prefill is equal to the digit. Decode in bench-set is 0.1 % (code) and 0.15 % (text) lower with a spread of 0.1 tok/s between
rounds; the decode path has no changed kernel, the difference is at the level of the code layout of the binary (the trace wrappers sit
in the allocation calls, which the decode does not run). Decode after the 130K prompt varied from 35 to 67 tok/s between runs of
either build (desktop load, not the build).
