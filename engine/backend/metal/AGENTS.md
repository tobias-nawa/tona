# AGENTS.md: Metal backend

Read the root [AGENTS.md](../../../AGENTS.md) and [docs/backends/blueprint.md](../../../docs/backends/blueprint.md) first.
Design and decisions: `research/40-mac-backend-proposal.md` (D1 to D8; D6 changed 2026-11-11, see there), the
measurements behind them: [docs/backends/metal.md](../../../docs/backends/metal.md).

## Platform

Apple silicon GPUs through Metal (MSL 3.2, macOS 15 or newer), arm64 host code. Developed and measured on an M1 Max
32 GB, macOS 27. Tier 3 (experimental), maintainers: the project maintainers. Info name `metal-<device>`, e.g.
`metal-apple-m1-max`.

## Build

```sh
cmake -S engine -B work/build/mtl -G Ninja -DTONA_BACKEND_METAL=ON -DCMAKE_BUILD_TYPE=Release
ninja -C work/build/mtl tona-check tona-server
```

- Statically linked into `tona-server` and `tona-check` (maintainer decision 2026-11-11: one backend per macOS binary,
  no `dlopen`; [backends/README.md](../../../docs/backends/README.md#selection)).
- The kernels (`kernels/all.metal` and what it includes) are compiled at build time into `tona.metallib` next to the
  programs. That needs the Metal toolchain: `xcodebuild -downloadComponent MetalToolchain`. Releases ship only the
  prebuilt library.
- Development only: `TONA_METAL_SOURCES=engine/backend/metal/kernels` compiles the kernel sources at start instead.
  Never compare speeds of two builds while this is set: both read the same sources, whatever their binary
  (`research/measurements/2026-11-11-mac-densek.md`, the invalid first A/B).
- Host tests without GPU and model: `ctest -L unit` (includes `metal_host`).

## Safety

- Watchdog: a command buffer that runs or waits for about 5 s is aborted by macOS
  (`kIOGPUCommandBufferCallbackErrorTimeout`; 4.5 s completes, 5 s aborts,
  `research/measurements/2026-11-11-mac-timeout.md`). One abort leaves the queue usable; the pass throws and
  `recover()` resets the sequence.
- **Every command buffer at most 0.5 s.** After buffers of ~2.2 s the system left the GPU idle for 0.1 to 0.3 s
  before each later submission (25 s over ~200 buffers; not seen after buffers of ~0.5 s;
  `research/measurements/2026-11-11-mac-prefill-experts.md`). A prefill layer of a 512-token chunk takes ~0.1 s.
- The arena's stops (`runner_cold.mm`) give up only as `stop_wait.h` decides: on the command buffer's error (the
  watchdog), on a buffer that completed without the stop, or after 10 s in which the waiting thread itself ran. A
  wall-clock deadline of 4 s failed passes of a process held by a profiler. A wait over 1 s logs the thread's CPU
  time and page-ins. `TONA_METAL_STOP_SLEEP=f` (default 0.25, 0 off; 0.5 lost decode) sleeps up to f times a stop's expected wait
  before the spin, which otherwise keeps a core busy for the whole pass; `TONA_METAL_STOP_SLEEP_SIGMAS=k` ends the
  sleep k deviations before the mean, `TONA_METAL_STOP_SLEEP_MIN_US` skips short stops (`stop_wait.h`).
- A stop's flag is tested "at or past" its value, never for equality: the GPU writes the prediction word of the next
  stop before that stop's wait, and a CPU that came late spun on an equality that never came true until the
  watchdog aborted the pass.
- The arena's decode passes wait on the CPU inside a command buffer (a shared event per stop): never block the CPU
  thread that must signal the event.
- Another workload may share the machine: keep test runs short, no long runs at full GPU load.
- Every script that waits for something (a server's `/health`, a file, a process) has a deadline (10 min for a
  server start) and then stops with an error message, and checks that the process it waits for still lives. A loop
  without both lost a whole night: the server had exited on a bad option and the script polled until morning
  (`research/measurements/2026-11-12-mac-mtp-server.md`).

## Memory

Matrices stay in the GGUF mapping (no-copy buffers over the whole file); small tensors are dequantized to f32 once.
The n-gram table (`per_layer_token_embd`, 28.8 GB in the Coder) gets no GPU buffer: each pass copies its tokens' 16
rows from the mapping into a staging buffer, so the table costs page cache, not Metal working set.
State per sequence (`TONA_SEQS` slots, default 1): DeltaNet state and conv window and the PLE ring three times (live,
backup for `commit(m < nb)`, save point), f32 KV cache and indexer keys for `max_ctx` positions once. The Metal
working set of the machine is the limit (`MTLDevice.recommendedMaxWorkingSetSize`; 26.8 GB on the M1 Max 32 GB).
A model whose weights exceed the working set gets the expert arena (`arena.h`) by default: the most used experts by
`TONA_EXPERT_PROFILE` (`--profile`) in a buffer of the budget that other programs and the rest of the engine leave
(`TONA_RAM_GB` caps it), a cache of recently read experts and a lookahead; the rest is read from the SSD at each
layer's stop. `TONA_METAL_ARENA_GB=n` sets the budget, `0` turns the arena off. The arena is in blocks of at most 32 MB, the expert
kernels read through a table of GPU addresses; `TONA_METAL_ARENA_GOVERN=1` (off until measured) frees blocks under
memory pressure and reads them again later (`backend/ram_governor.h`), only between passes (`Arena::PassScope`); its
thresholds follow the reserve the budget leaves (`RamGovernorConfig::from_reserve`), free RAM is calibrated at start.
With it the arena also fills in growth steps at start (`backend::fill_step`: each step leaves the growth floor free,
a step that made the OS compress stops the fill) after dropping the page cache of the experts' file ranges
(`msync(MS_INVALIDATE)`): free RAM as the page counts read it includes the file cache, but the OS takes a fill's room
from its inactive pages, other programs' among them, and compresses those
(`research/measurements/2026-11-16-mac-arena-fill-steps.md`).
`TONA_METAL_ARENA_CHURN=n` frees and re-reads a quarter of the arena every n passes: tokens must stay identical.
Opt-in experiments with the governor, none the default (`research/53-mac-purgeable-arena.md`,
`research/54-mac-constants.md`): `TONA_METAL_ARENA_PURGEABLE=1` makes the pin blocks purgeable Metal buffers, volatile
on other programs' compression (sampled every 50 ms) and freed, not re-read, when found purged (give-way);
`TONA_METAL_ARENA_BLOCK_MB=n` sets the block size (default 32). The hold of passes (`TONA_METAL_ARENA_HOLD_Q`) and
the arena without a reserve (`TONA_METAL_ARENA_REACT`) failed their runs and were removed; their code last lived in
b9640fe.
`TONA_METAL_ARENA_CEILING=1` (test) plans the arena up to the working set. `TONA_METAL_ARENA_LOG=n` prints the counters
every n stops. `TONA_METAL_KEEPALIVE_MS` and `TONA_METAL_KEEPALIVE_S` (`keepalive.h`) keep the GPU clocked for a window
after a pass.
`TONA_EXPERT_PROFILE_OUT` counts every routed expert at the arena's layer stops and writes the profile at exit or on
SIGINT/SIGTERM; a cut of the Coder gets a profile from the Coder's by its router rows (`tools/coder_requant/profile_map.py`),
but its own count matched held-out prompts better (85.2 against 75.6 % of uses in the top 104 of 164,
`research/measurements/2026-11-16-mac-dmt-profile.md`).

## Checks that must pass before a change here

1. `tona-check MODEL.gguf "PROMPT" -n 200 --ctx 1024` against `cpu/ref_qwen4exp.cpp`: top-1 equal in every step. For kernel
   changes run it on the mini model (`tools/make_mini_model.py`, all decoders of the Coder's types) and on a real model.
   First cut on the mini model: 206 of 206 steps, worst relative deviation 1.08e-6.
2. Bit-identity of the pass widths: `TONA_DETERMINISTIC=1 tona-check ... --widths SEED`, also with `TONA_SEQS=2`
   (without it the half MMA kernel of 4-token chunks changes their rows, `TONA_METAL_MMA`): passes of random width
   1 to 8 with random commit, a save point and a stash round trip; every logit row bitwise equal to the width-1 run.
   Mat-vec shapes depend on the matrix only, never on the width; keep it that way. Quality and speed as in
   [backends/README.md](../../../docs/backends/README.md#checks) and the blueprint's M1 gate.
3. Cut invariance of the prefill: `TONA_PREFILL_CHUNK=64 tona-check ... --prefill SEED` on the mini model and on a
   model of real shapes: the state after a prefill in one call and in random pieces byte-equal, then the decode steps
   after the prefill against the width-1 run. A GEMM output row is reduced over K alone, never across the tokens of a
   tile; keep it that way (docs/use/prompt-cache.md).
4. MTP head (`--mtp MTP.gguf`, `runner_mtp.mm`, `pass_mtp.mm`): `tona-check ... --mtp MTP.gguf --golden FILE`
   against a golden file of the reference (`--dump` on the PC, same models and prompt). The MTP pass reuses the layer
   encoders with the head's own layer and QSA buffers (the last slot of each SeqState, never stashed); keep those
   encoders free of `m_.layers()`.
5. Kernels and their parameter structs: `kernels/params.h` is the one definition for both sides; change it, never a
   copy.

## Do not

- change shared code (`model/`, `server/`, `cpu/`, `backend/`) without a before/after on the reference;
- copy third-party code without license check and origin note;
- name the programs that ran on the machine in measurement files (background load as sums only).
