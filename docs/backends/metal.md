# Metal backend (Apple silicon)

The Mac backend is in development on the maintainers' M1 Max 32 GB and is the template of
[blueprint.md](blueprint.md). It is linked statically into the macOS programs and chosen through the
registry ([backends/README.md](README.md#selection)). Marks: [V] read, [S] search result only, [M] measured by us (findings of
2026-11-11). Build, checks and safety rules: `engine/backend/metal/AGENTS.md`.

Classes: [mac-chips.md](../hardware/mac-chips.md), [mac-16-24gb.md](../hardware/mac-16-24gb.md),
[mac-32gb.md](../hardware/mac-32gb.md), [mac-64gb-and-larger.md](../hardware/mac-64gb-and-larger.md).

## Where the work is

| Item | File |
|---|---|
| Architecture proposal, decisions D1 to D8, order M0 to M4 | `research/40-mac-backend-proposal.md` |
| M0 reconciliation: resident pass 25 to 32 ms, decode estimates revised | `research/41-mac-reconciliation.md` |
| Papers on MoE with unified memory and SSD | `research/38-moe-on-mac.md` |
| Engines on Mac, watchdog and forward progress (section 4) | `research/39-mac-engines-2026-11.md` |
| Measurements | `research/measurements/2026-11-11-mac-{bandwidth,dispatch,event,timeout,overlap,ssd,expertk,densek}.md` |

## What the Mac work already settled

| Finding | Value | Mark |
|---|---|---|
| GPU read bandwidth, M1 Max | 334 GB/s, 82 % of the peak 409.6 | [M] |
| Dependent dispatch | 1.7 to 2.0 us; command buffer round trip 170 us, pipelined 23 us | [M] |
| CPU signal to a waiting GPU (`MTLSharedEvent`) | 55 us median, p99 about 230 us | [M] |
| Event wait limit | 4.5 s completes, 5 s aborts (`kIOGPUCommandBufferCallbackErrorTimeout`); one timeout leaves the queue usable | [M] |
| GPU clock after idle | 3.3 times slower for about 16 ms after an idle of 30 ms or more; no loss below 10 ms | [M] |
| GPU slowdown while the SSD reads | 11 to 15 % of the read window | [M] |
| Expert kernels, untuned, 48 layers x 10 experts | 10.2 ms (29 % of the bound) | [M] |
| Megakernel | not used: Metal gives no forward-progress guarantee between threadgroups; one paper measured a fused MoE block 1.8 times slower than separate dispatches on an M4 Pro (arXiv 2610.31479) | [S] |

Design of the proposal (check the state of each decision before relying on it): own MSL kernels, one command buffer per
pass, no spinning kernels, per-layer `MTLSharedEvent` wait with a 100 ms cap and a 1 s split of command buffers; static
linking instead of `libtona_metal.dylib` through `dlopen` (D6).

## What a contributor does here

- Mac owners with another chip or memory size: run the "first steps" of your class file and compare to the M1 Max numbers.
- Do not start a second Mac backend; contribute to the existing one through its issue.
- Selection: `engine/backend/registry.h` (`--backend NAME|auto`, `TONA_BACKEND`); the planned `dlopen` interface is in
  [backends/README.md](README.md#selection).
