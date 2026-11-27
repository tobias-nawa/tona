# 40: Mac backend (Apple silicon, Metal): architecture proposal (#107)

Result of Phase 0, for decision by the maintainer. Nothing here is implemented. Measurements are from one M1 Max 32 GB
under background load (provisional, see each file); numbers from other projects are their authors' claims; estimates
are marked. Sources: [38-moe-on-mac.md](38-moe-on-mac.md) (papers), [39-mac-engines-2026-11.md](39-mac-engines-2026-11.md)
(engines), [21-backends.md](21-backends.md) (tiers, earlier research).

## Decisions asked for

| # | Decision | Proposal |
|---|---|---|
| D1 | Kernels | own Metal kernels (MSL) in `engine/backend/metal/`; no MLX, no ggml-metal as runtime |
| D2 | Pass structure | one command buffer per pass, one dispatch per step (~2 µs each), no grid barriers, no spinning kernels |
| D3 | Large tier, expert misses | prediction ahead of the layer (overbooked) plus a per-layer `MTLSharedEvent` wait for the rest; CPU compute of misses not pursued; lossy handling only as a measured option for you to decide |
| D4 | Timeout | hard limit per wait and per command buffer with a recovery path (below), so no wait reaches the macOS GPU watchdog |
| D5 | First milestone | a resident qwen4exp model (self-requantized Coder, standard IQ2 formats, ~19 to 20 GB) before streaming |
| D6 | Loading | Changed 2026-11-11, maintainer decision: the Metal backend is linked statically into the macOS programs and chosen through the registry (`engine/backend/registry.h`, `--backend`); macOS has exactly one backend, so `dlopen` would add only risk. The `dlopen` interface is specified in docs/backends/README.md#selection and built with the first backend that is optional within one binary. Originally: `libtona_metal.dylib` via `dlopen`, chosen by detected hardware |
| D7 | Packed expert file | a second copy of the experts on the SSD (~25 GB) in one 2 MB blob per expert, made by the installer |
| D8 | Tiers | stay provisional until our own quality measurement (below); the vendor numbers are unchecked |

## Starting point (measured, M1 Max 32 GB)

| Quantity | Value | File |
|---|---|---|
| GPU read bandwidth | 334 GB/s (82 % of 409.6) | [mac-bandwidth](measurements/2026-11-11-mac-bandwidth.md) |
| CPU read bandwidth | 58 GB/s one thread, 112 to 120 all cores | same |
| Recommended working set / max buffer | 26.8 GB / 20.1 GB | same |
| SSD, packed expert 2.046 MB random, `F_NOCACHE` | 13.0 GB/s at QD 16; 3.6 GB/s at QD 1 (0.57 ms) | [mac-ssd](measurements/2026-11-11-mac-ssd.md) |
| SSD, one expert matrix 0.68 MB (GGUF layout) | ~7 GB/s at QD 16 | same |
| Dependent dispatch | 1.7 to 2.0 µs (encoder, encoder each, ICB alike) | [mac-dispatch](measurements/2026-11-11-mac-dispatch.md) |
| Command buffer round trip / pipelined | 170 µs median (p99 ~300) / 23 µs | same |
| CPU signal to waiting GPU (`MTLSharedEvent`) | 55 µs median, p99 ~230 | [mac-event](measurements/2026-11-11-mac-event.md) |
| Long event wait | 2 s completes, **8 s aborted** (`kIOGPUCommandBufferCallbackErrorTimeout`) | same |

External baseline (not ours): ds4 streams the 512-expert Flash-Next on an M1 Max 32 GB at **8.25 to 8.44 tok/s decode**,
and a further open-source engine with the design proposed here reports rates of the same order (39).

## D1: own kernels

- **MLX** has no IQ formats (GGUF types are cast to fp16 on load), no expert offload and no qwen4exp support (39 §1).
  Our Coder would not fit; MLX would be a second runtime with its own memory manager beside our cache.
- **ggml-metal** (llama.cpp) supports qwen4exp, but IQ formats lack small-batch kernels (#27350 open), so speculation
  with 2 to 8 tokens per pass gains little, and its graph executor decides the dispatch order, not us. Its kernels are
  MIT (Tona is MIT): we may study and adapt them with a license check and an origin note in the file.
- **Own MSL kernels** follow the rule "one call, one whole pass" ([docs/backends/README.md](../docs/backends/README.md)) and let us fuse
  per layer, choose the tile per format for B = 1 to 8 and read experts straight from our own slot arena.
- Host code in Objective-C++ (`.mm`) or metal-cpp (Apple, Apache-2.0); decision with the first commit, both pass the
  gates. Unchecked: whether `tools/gates/tidy.py` handles `.mm` files.

## D2: pass structure

- Metal guarantees no forward progress between threadgroups; Apple documents none, and a paper measured a fused MoE-block
  megakernel 1.8× slower than separate dispatches on an M4 Pro (2610.31479, 38). The reference's grid barriers become
  dispatch boundaries.
- Cost: ~1,000 dependent steps per token at ~2 µs = **~2 ms of a 15.3 ms resident pass (13 %)**; fusing to ~10 dispatches
  per layer (~500 per pass) halves that (estimate).
- **One command buffer per pass**, committed once; the CPU never waits inside the pass with `waitUntilCompleted`
  (170 µs each). Prefill chunks: one command buffer per chunk, pipelined (23 µs).
- **No kernel spins on CPU memory** (the reference's `k_mega` pattern): the watchdog counts executing work, and a spinning
  kernel holds GPU cores. The only GPU-side waits are `encodeWaitForEvent` between dispatches.
- Unmeasured and decisive for the gap to the external baseline: the GPU clock after an idle gap of 0.1 to 3 ms (another
  engine keeps the GPU busy with a spin kernel on a second queue for this), and IQ expert kernel time at B = 1 (another
  engine's profile calls it "latency and ALU bound").

## D3: large tier, expert streaming

**Memory.** Dense weights 4.1 GB, experts 25.15 GB (256 per layer, 2.046 MB each, 48 layers). Arena of 2 MB slots in
shared `MTLBuffer`s (max buffer 20.1 GB, so several buffers), filled by `pread` with `F_NOCACHE` from the packed file at
QD ≥ 16. Planning size 14 to 18 GB (56 to 72 % of the experts) next to the desktop and other programs.

**Condition (MawForge, 38):** a larger cache lowered decode from 13.9 to 2.3 tok/s on a 24 GB M5 Pro through memory
pressure despite more hits. The arena size is chosen by a measurement under realistic background load
(swap, compressor, page-ins recorded), not by the 26.8 GB limit. Unchecked: whether macOS compresses a shared
`MTLBuffer` that the GPU has not touched recently; one engine reports watchdog panics from a compressed arena (39 §4).

**Hit rate:** provisional from the llama.cpp #29887 curve (33 % resident → 85 %, 67 % → 94 %); replaced by the PC
simulation of 154 to 205 slots per layer on our route traces (`tools/cache_sim/`, the coordinator runs it).

**The four ways to handle a miss** (the coordinator's comparison):

| Way | What it costs | Evidence | Proposal |
|---|---|---|---|
| (1) GPU waits on an event per layer | 55 µs median per wait (p99 230), ×48 = ~2.6 ms per token even at 100 % hits; plus the read at QD 1 (0.57 ms) per miss | ours (mac-event); #23324 lost 21 % at 100 % hits | use, but only for misses left after (2); skip the wait when the layer has none |
| (2) prediction ahead of the layer | wasted reads for wrong picks (49 % of prefetched experts used on the PC) | router of l+1 on layer l's input: top-10 hit 65 % with 10, 88 % with 24 candidates (ours, 2026-10-24); papers 91 to 99 % with overbooking (38); +21 % decode on an M4 (DeepInfer-Mac) | use, overbooked (16 to 24 candidates), issued at QD 16 while the GPU computes layer l |
| (3) CPU computes misses | no SSD read saved (unified memory); CPU 58 to 120 GB/s against GPU 334 | no Apple engine does it; SSD-LLaMA measures CPU misses slower (38, 39 §6) | not pursued |
| (4) lossy: drop or substitute a miss | quality, unmeasured for the Coder | top-k₂ renormalization 10 → 5 experts −0.55 MMLU on a related model (2609.04575) | **named only**; your decision after a quality measurement; existing switch `TONA_MISS_DROP` |

**Order within a layer** (ds4's overlap, our E2a shows work before a wait is not held up): router → shared expert and
hit experts → event wait only if misses remain → missed experts → reduction in the fixed order (bit-exact against a
fully resident run).

**Bandwidth sharing, unmeasured:** Flash-MoE claims SSD DMA slows the GPU by 73 % on an M3 Max. A measurement of GPU read bandwidth during 13 GB/s SSD reads decides how much of (2) can be hidden.

## D4: timeout as a hard limit

macOS aborts a command buffer whose event wait lasts between 2 and 8 s (ours); community reports name ~5 s with waits
counted; Apple's `accessRevoked` can take the device away after repeated timeouts (39 §4). So:

1. **Wait cap:** the CPU signals every event at the latest after `TONA_METAL_WAIT_MS` (proposal 100 ms, as
   `TONA_MEGA_YIELD_MS`), read finished or not; it sets an abort flag in shared memory first. Every later dispatch of
   the pass reads the flag and returns at once.
2. **Command buffer cap:** a pass whose total encoded waits could exceed 1 s (e.g. a long prefill chunk with many misses)
   is split into several command buffers.
3. **Recovery:** an aborted pass leaves the state unchanged (KV and DeltaNet state written only after the pass, or
   restored from the checkpoint the generator already keeps for speculation); the generator repeats the pass when the
   reads are done.
4. **After a real `timeout` error:** no retry loop. Recreate the queue once; a second error in the same minute stops the
   backend with a clear message.
5. Before implementation: narrow the limit between 2 and 8 s and test, in a throwaway process, whether a queue accepts
   work after a timeout.

## D5: order of work

| Step | Content | Gate to pass |
|---|---|---|
| M0 | measurements: expert kernel time at B = 1 to 4 for the Coder's types (gate/up IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down IQ4_NL, Q2_0; the file named IQ1_M holds no IQ1_M tensor, [06](06-qwen38-architecture.md)); GPU bandwidth under SSD load; clock after idle gaps; timeout limit and recovery; cache simulation (PC) | each with a file; the reconciliation with the external ~8 tok/s written down |
| M1 | resident qwen4exp: Coder requantized with standard IQ2_XXS experts (~19 to 20 GB, estimate; no such file exists), loader unchanged | `just cross-check` against `cpu/ref.cpp` (top-1, logit deviation); decode within 70 % of the bandwidth bound |
| M2 | streaming of the IQ1_M Coder: packed file, arena, event, prediction, timeout path | bit-exact against M1-style resident run on a short prompt; decode above the external baseline |
| M3 | speculation (MTP drafts, B = 2 to 8) and prefill path | tokens per pass and prefill rate measured |
| M4 | small tier (qwen35moe, new architecture path in every backend) | per the tier decision |

M1 first because it tests all qwen4exp kernels without SSD effects and gives the "all resident" number the streaming
version is compared against.

M1 model file, as decided after Phase 0: requantized from the base Q8_0 by `tools/coder_requant/` on the PC, in three
variants of equal trunk size (~20 GB): A 256 experts IQ2_XXS, B 216 experts IQ2_S, C 192 experts IQ3_XXS (down Q2_0
in all three; experts dropped by gate weight on coding traces). Assumptions:

- The quality of all three is measured the same way with llama.cpp (`llama-perplexity --kl-divergence` against the
  IQ1_M Coder's logits), not with Tona: the reference backend assumes 256 experts and stays so.
- If B or C wins, the Mac backend reads the expert count from the GGUF (`expert_count`) instead of assuming 256.

## Expected rates (estimates)

| Tier / case | Decode | Prefill | Basis |
|---|---|---|---|
| large, streamed (IQ1_M file) | **floor to beat: 8.3** (external); revised after M0: 20 to 28 on demand, 23 to 34 with prediction ([41](41-mac-reconciliation.md); before: 22 to 31, up to 50 to 65) | ~250 to 450 tok/s at 512-token chunks (estimate: 7 to 11 GB of non-resident experts per chunk at 13 GB/s plus compute at 30 to 50 % of ~10 TFLOPS fp16, knowledge unchecked) | mac-ssd; 39 |
| large, upper bound | 66 passes/s | - | 5.1 GB per token at 334 GB/s |
| medium, resident | Gyro-Coder 71 to 82 / Qwen3.8-27B 35 (bounds) | unmeasured | 21-backends tiers |
| small, resident | Ornith Q4_K_M 187 short / 107 at 128K (bound) | unmeasured | same |

Bounds are memory bounds; real rates lie below by the kernel efficiency M0 measures.

## D7, D8: files and tiers

- **Packed expert file:** 25.15 GB extra on the SSD besides the 58.4 GB GGUF (this Mac: ~101 GiB free). Alternative:
  the installer writes the packed file and a GGUF without experts (saves ~25 GB, but the GGUF is no longer the
  upstream file; hash checks change).
- **Tiers** ([21-backends.md](21-backends.md#model-tiers-2026-11-11-107-46)): large = ISTA Coder IQ1_M (SWE-V 75.6 for
  the pinned revision, vendor, unchecked); medium = Gyro-Coder TQ2_0 or Qwen3.8-27B IQ3_XXS; small = Ornith-1.5-35B-A3B
  or Groot-19B-A3B. **Condition:** a tier is fixed only after our own measurement: `scripts/bench-effort.py`
  (23 tasks), `bench/threshold/`, Δ-NLL against the large model, and valid tool-call JSON (2611.12240, 38).

## Risks

- The external baseline (~8 tok/s) may be the real limit of this chip, set by IQ kernel efficiency at B = 1 and not
  by the SSD; M0 answers it before any streaming code.
- Memory pressure from the owner's other programs makes the arena size, and so the hit rate, vary at run time; the
  arena needs to shrink without a restart (unchecked whether purgeable `MTLBuffer` state suffices).
- CPU read bandwidth is below expectation (112 to 120 GB/s against 180 to 260; cause open); it matters only if (3) or a
  CPU dense path is ever reconsidered.
