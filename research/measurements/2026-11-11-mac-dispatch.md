# 2026-11-11: cost of Metal dispatches, command buffers and indirect command buffers, M1 Max (#107)

Phase 0 of the Mac backend, point 3. The reference megakernel synchronizes its steps with ~1,100 grid barriers per
pass; Metal guarantees no progress between threadgroups, so a Mac backend replaces the barriers by separate dispatches
([research/21-backends.md](../21-backends.md), "Megakernel on Apple Metal"). This file measures what one dependent
dispatch costs, so the pass structure can be sized. Speed numbers on this machine are provisional.

## Plan (written before the run)

**Scope.** Tool `bench/mac/dispatch.swift`. The kernel is minimal: one threadgroup of 32 threads; thread 0 reads a
counter, adds 1 and writes it back, so every dispatch depends on the one before (the chain is checked at the end:
counter = number of dispatches). The measured time is therefore launch and dependency cost, not work.

| Block | What runs | Rounds | Duration |
|---|---|---|---|
| D1 dispatches in one encoder | N = 1,000 dependent dispatches in one compute encoder (dispatch type serial), one command buffer; time from the GPU timestamps / N | 20 command buffers per run | < 1 s |
| D2 one encoder per dispatch | the same N = 1,000, each dispatch in its own compute encoder within one command buffer | 20 | < 1 s |
| D3 indirect command buffer | N = 1,000 dispatches encoded once into an `MTLIndirectCommandBuffer`, executed with `executeCommandsInBuffer` (barrier between the commands) | 20 | < 1 s |
| D4 command buffer round trip | one command buffer with one dispatch, commit and `waitUntilCompleted`, CPU wall time | 500 | < 1 s |
| D5 pipelined command buffers | 1,000 command buffers with one dispatch each, all committed, wait for the last; CPU wall time / 1,000 | 20 | < 1 s |

Each block 3 runs, 10 s pause between runs. Per block: median and p99 over the samples, µs per dispatch (D1 to D3,
D5) or per round trip (D4).

**Expectation.**

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| D1 µs per dependent dispatch | 3 µs | 1 to 8 | llama.cpp Metal discussions report a few µs per kernel on M-series (knowledge, unchecked) |
| D2 µs per dispatch with its own encoder | 5 µs | 2 to 15 | encoder switches add cost (knowledge, unchecked) |
| D3 µs per ICB dispatch | 3 µs | 1 to 8 | ICB saves CPU encoding, not GPU launch cost (knowledge, unchecked) |
| D4 round trip | 150 µs | 50 to 400 | commit to completion includes the driver and the completion interrupt (knowledge, unchecked) |
| D5 µs per pipelined command buffer | 15 µs | 5 to 50 | (knowledge, unchecked) |

Consequence to test (not a pass criterion): at 3 µs and ~1,000 dependent steps per token, launch cost is ~3 ms of a
~15 ms pass (20 %).

**Active tooling and conditions.**

- `swiftc -O` (Swift 6.4.2), shader compiled at run time from source; no Metal validation, no GPU capture, no profiler.
- Load and power recorded right before each run.

## Results

Raw data: [2026-11-11-mac-dispatch.jsonl](2026-11-11-mac-dispatch.jsonl). Tool: `bench/mac/dispatch.swift`. The chain
check passed in every sample (counter = number of dispatches).

**Conditions (12:26).** AC power. Load average 15.7 to 16.6 on 10 cores; background load: the two busiest
processes together 64 to 73 % CPU.

| Block | Run 1 | Run 2 | Run 3 | Expected | |
|---|---|---|---|---|---|
| D1 µs per dependent dispatch, one encoder (median / p99 of 20) | 1.79 / 1.93 | 2.00 / 2.05 | 1.96 / 2.08 | 1 to 8 | inside |
| D2 µs per dispatch, one encoder each | 1.73 / 1.84 | 1.90 / 2.01 | 1.99 / 2.08 | 2 to 15 | inside, at the lower end |
| D3 µs per ICB dispatch with barrier | 2.03 / 2.17 | 2.01 / 2.14 | 2.00 / 2.12 | 1 to 8 | inside |
| D4 µs per command buffer round trip (median / p99 of 500; min) | 170 / 325 (96) | 171 / 293 (99) | 172 / 290 (136) | 50 to 400 | inside |
| D5 µs per pipelined command buffer (median / p99 of 20) | 22.9 / 23.3 | 22.8 / 23.0 | 22.6 / 23.2 | 5 to 50 | inside |

- **A dependent dispatch costs ~2 µs of GPU time**, the same in one encoder, with an encoder each, or from an ICB.
  Encoder switches cost no GPU time; their CPU cost is not measured here.
- **1,000 dependent steps per token cost ~2 ms**, 13 % of the 15.3 ms a resident pass of the reference model needs
  ([2026-11-11-mac-bandwidth.md](2026-11-11-mac-bandwidth.md)). A pass from separate dispatches is viable; merging
  steps (fewer, larger kernels per layer) lowers the share further.
- **Never wait on the CPU inside a pass**: a command buffer round trip costs ~170 µs (p99 ~300 µs), 85× a dispatch.
  A pass is one command buffer (or a few, committed back to back: ~23 µs each); the CPU only waits at the end.
  For expert streaming this means: misses are known on the GPU (router) but read by the CPU, so a CPU round trip per
  layer would cost 48 × ~170 µs = ~8 ms per token. The SSD path needs prediction ahead of the pass or an
  asynchronous design (MTLSharedEvent, measured in a later step), not a wait per layer.
