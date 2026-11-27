# Mac: the Coder with an expert arena of its own (probe `TONA_METAL_ARENA_GB`)

Issue #108, part 2 of the Coder work after the baseline
([2026-11-15-mac-coder-baseline.md](2026-11-15-mac-coder-baseline.md): on the page cache a pass re-reads 23 to 43 GB,
decode about -99 % against DM-T). Design: `engine/backend/metal/arena.h`. The most used experts by the profile
(`toolchain/profiles/coder-iq1_m.bin`, sha256 b5c14ad1…, counted on the PC on 2026-10-23 for shard 1 revision
bba88856, the file measured here; the counting workload is not documented, origin
`research/measurements/2026-11-12-coder-1019.md` and `2026-10-23-pinned-buffers.md`) are read once into one buffer
of the process (shared ranking `backend/expert_tiers.h` `select_pinned`); after each layer's router the pass stops,
the routed experts that are not resident are read with `pread` (F_NOCACHE, no read-ahead) into staging slots, and the
expert kernels read every expert through an offset table (`kernels/moe_table.metal`). The routed experts are left out
of the buffers over the file mapping, so the window and split probes of the baseline are not needed.

Cold read path, measured before on this machine (no new microbenchmark): 2 MB reads with F_NOCACHE 3.5 GB/s at
depth 1, 13 GB/s at depth 16, 0.68 MB reads (one matrix of the GGUF layout) 7 GB/s at depth 16
([2026-11-11-mac-ssd.md](2026-11-11-mac-ssd.md)); a command buffer round trip about 170 µs
([2026-11-11-mac-dispatch.md](2026-11-11-mac-dispatch.md)); the GPU loses 11 to 15 % of its bandwidth while the SSD
reads ([2026-11-11-mac-overlap.md](2026-11-11-mac-overlap.md)).

**Budget against the profile** (the Coder's experts 25.1 GB, one expert 1.5 to 2.7 MB; computed from the GGUF and the
profile): 14 GB of pins hold 56 % of the experts and 91.7 % of the profile's uses, 16 GB 64 % and 95.1 %, 18 GB 72 %
and 97.5 %, 20 GB 80 % and 99.0 %. The PC's floor (pin only from 70 % of the expert bytes) is not applied: it
compares partial pins with the page cache, which on this machine re-reads the model every pass.

## Correctness (before any speed run)

| Check | Model | Result |
|---|---|---|
| `tona-check` 200 steps, arena 0.02 and 0.06 GB (44 % and 98 % of the uses resident), deterministic | mini | 204 of 204 top-1 equal, worst 8.70e-07, the same as without the arena |
| `--widths 7`, 1 and 2 slots | mini | 201 and 404 rows, 0 not bit-identical |
| `--prefill 5`, `TONA_PREFILL_CHUNK=64` | mini | state byte-equal over 3 cuts, 35 steps after the prefill top-1 equal |
| `tona-check` 64 steps, arena 4 GB (46.6 % of the uses resident, 22,375 cold experts read), deterministic | Coder | 71 of 71 top-1 equal against the CPU, worst 3.93e-07 |

The first Coder run failed out of memory: the buffers over the mapping still spanned the left-out experts between
the dense tensors; windows now start anew at a gap of more than a page when the arena is on.

## Run 1: first speed, arena 14 and 16 GB (plan, written before the run)

**Question.** How fast is the Coder with the arena, and where does a pass's time go (cold reads or the stops)? It
decides the next step: the A/B against DM-T if decode is within reach, else the cost of the stops first.

**Run.** The scratch harness of the baseline (one server per budget, Coder + MTP-256, `--ctx 32768`): the 7722-token
request with 100 tokens, then 3 code requests of 400 tokens; per request page-ins (`vm_stat`), prefill and decode,
tokens per pass; at the end the arena's line (cold layer stops, experts, GB read, seconds). Budgets 14 and 16 GB,
about 5 min each. Memory before the run: other programs hold about 6 GB (anonymous 1.3, wired 3.1, compressor 1.6),
so 16 GB of arena plus 4.4 GB of dense weights, 2.8 GB of the head's file and 2.2 GB of KV leave about 3 GB.

**Expectation.** A decode pass (3 to 4 tokens, about 25 distinct experts per layer, 1200 expert reads) misses 8 % of
them at 14 GB and 5 % at 16 GB (the profile's mass, a little worse for the union of a pass): 60 to 100 cold experts,
130 to 210 MB per pass; read 6 at a time (2 experts of 3 matrices per layer) at about 5 GB/s, 26 to 42 ms, plus 48
stops of about 0.25 ms (a command buffer round trip and the encoding), 12 ms. A pass of about 130 to 145 ms against
90 for DM-T: decode 15 to 30 tok/s (DM-T 25 to 53). Prefill: a chunk reads all cold experts of each layer once
(44 % of 12,288 experts at 14 GB, about 11 GB per chunk at 7 GB/s, 1.6 s), 4 chunks of 2048: +6 s on 54 s.
Page-ins near 0 (the cold reads bypass the cache). Decode below 10 tok/s or page-ins above 1 GB per request would be
findings.

**Active tooling.** `TONA_METAL_ARENA_GB`, `TONA_EXPERT_PROFILE`, `TONA_TOKEN_LOG`; defaults otherwise.

## Run 1: result

| Request | Arena 14 GB (89.9 % of the uses) | Arena 16 GB (93.6 %) | DM-T + MTP-164 (baseline file) | Coder on the page cache |
|---|---|---|---|---|
| 7722-token prompt, prefill | 54.5 s | 52.9 s | 53.6 to 54.9 s | 223.6 s |
| same, 100 tokens decode | 10.4 tok/s | 12.6 tok/s | 25.5 tok/s | 0.24 tok/s |
| code requests, decode | 11.9 to 19.0 tok/s | 15.0 to 20.9 tok/s | 38.6 to 47.5 tok/s | 0.49 tok/s |
| page-ins per code request | 64 to 241 MiB | 48 to 403 MiB | | 23 to 43 GB per pass |

Prefill is as fast as DM-T's (the prefill stops are hidden behind the 2048-token layers). Decode is 40 to 60 %
below DM-T: a pass takes about 215 ms against about 104. Outside the expectation (15 to 30 tok/s) at its lower
end; quiet check before: warning (load average 3.4), after: ok. The compressor moved 13 to 15 GB during the long
prefill at 14 and 16 GB (free RAM 4.6 GB at the start of the first request): the arena does press the rest of the
system.

**Where the time goes (diagnosis runs, `TONA_METAL_ARENA_LOG`, short requests).**
1. The first implementation committed and waited on a command buffer at every stop: 2 ms of extra wait per stop
   (wait 4.76 ms against 2.72 ms of GPU time per stop), about 96 ms per pass.
2. One command buffer per pass with `encodeWaitForEvent` and the CPU polling the event's `signaledValue`: worse
   (DM-T with all experts in the arena, no reads at all: 45.5 -> 8.6 to 12.1 tok/s, and the first request after the
   start timed out at the 2 s guard). Reading `signaledValue` sees the GPU's signal milliseconds late.
3. The GPU writes a word into shared memory at the stop (kernel `stop_flag`), the CPU polls it and releases the
   GPU with the event (the direction measured at 55 µs in 2026-11-11-mac-event.md): DM-T with no reads 31 to 35
   tok/s against 43 (128-token requests), 36 to 38 against about 43 (512-token requests).
4. The shared expert and the grouping run between the stop and the wait (the GPU computes while the CPU looks at
   the routing), and the arena is made resident at start (residency set): **DM-T with no reads 43.1 to 46.8 tok/s,
   the same as without the arena** (43.1 to 46.4). The stops cost nothing measurable any more.
5. The Coder at 16 GB with 4: 16.8 to 17.4 tok/s (512-token requests, 6.7 tokens per pass). Per stop 6.9 to 7.6
   cold experts, 14.5 to 16.0 MB, read in 2.6 to 3.0 ms (5.4 to 5.6 GB/s, 0.68 MB reads of the GGUF layout): about
   730 MB and 135 ms of reads per pass, while the GPU waits. A verify pass of 7 to 8 tokens routes to 40 to 50
   distinct experts per layer, so 15 % of them are cold although the arena holds 93.6 % of the profile's uses (the
   plan assumed 3 to 4 tokens and 25 experts).

Next: what a cache of the recently read experts would save (the passes of one answer reuse experts), measured on a
trace of the cold experts per stop before it is built.

## A cache of recently read experts (`TONA_METAL_ARENA_LRU`)

**Simulation first** (the cold-expert trace of the baseline's four requests at 16 GB, `TONA_METAL_ARENA_TRACE`,
10,672 stops; pins by the same ranking, an LRU of K slots per layer filled by decode and verify passes; the slots
come out of the same total): cold reads per stop

| Total | K = 0 | 4 | 8 | 16 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|
| 16 GB | 4.37 | 4.05 | 3.57 | 2.74 | 2.24 | 2.16 | 2.25 | 2.50 |
| 18 GB | 3.07 | 2.70 | 2.23 | 1.63 | 1.34 | 1.30 | 1.38 | 1.65 |

At equal memory 32 to 48 slots per layer halve the reads against pins alone (vLLM's RFC found the same direction:
usage placement plus recency beats usage alone, research notes in the baseline file). K = 32 taken.

**Measured** (16 GB in all: 12.33 GB of pins, 87.0 % of the uses, 3.27 GB of cache, 400 MB staging; four 512-token
requests): **22.9 to 23.3 tok/s** against 16.8 to 17.4 without the cache (+35 %); per stop 4.1 cold experts, 8.7
MB, 1.7 ms of reads (5.1 GB/s), against 7 to 7.6 experts and 2.8 ms. Correct: `tona-check` on the Coder (8 GB, 8
slots per layer, deterministic) 71 of 71 top-1 equal, worst 3.93e-07; mini model with 4 slots: 204 of 204, widths
and prefill cuts bit-identical. DM-T without the arena stays identical at its three references (check, widths,
prefill).

## Lookahead (`TONA_METAL_ARENA_AHEAD`)

After a layer's stop the GPU applies the next layer's router to a prediction of its input; the CPU releases the
GPU, then reads the predicted experts that are neither resident nor cached into the next layer's cache in the
background (the next stop waits for them first). As Pre-gated MoE (Hwang et al., ISCA 2024) and the next-layer
lookahead of `docs/architecture/expert-tiers.md` section 7. Coder, 16 GB, 32 cache slots per layer, two
256-token requests (one run each, quiet check not taken: diagnosis, not a speed result):

| Prediction | Experts per token | Recall (routed non-resident experts that were predicted) | Precision | Reads at the stop | Decode |
|---|---|---|---|---|---|
| none | | | | 4.1 per stop, 1.7 ms | 22.9 to 23.3 tok/s (512-token requests) |
| this layer's gated read (=1) | 10 | 60 % | 55 % | 3.2 to 3.4, 1.5 ms | 23.6 to 24.8 tok/s (512) |
| the next layer's gated read of the residual as it is (=2) | 10 | 69 % | 69 % | 2.55, 1.25 ms | **26.8 to 27.8 tok/s** |
| =2 | 14 | 82 % | 55 % | 2.52, 1.55 ms | 24.2 to 24.4 tok/s |
| =2 | 18 | 89 % | 45 % | 2.81, 2.17 ms | 19.4 to 19.8 tok/s |

More predicted experts raise the recall but evict cached ones and load the SSD (2.7 to 3.4 GB/s at the stop
instead of 4.3): 10 per token (the router's own top-k) is best. Correct with =2: Coder `tona-check` 71 of 71 top-1
(8 GB, 8 slots), widths bit-identical on the Coder and the mini model, prefill cuts byte-equal; DM-T without the
arena identical at its three references.

## Budget from the free memory (`TONA_METAL_ARENA_GB=auto`)

Once the rest of the engine is loaded the arena takes the smallest of: physical memory minus the wired, anonymous
and compressed pages of all processes, our weight mappings and a reserve (0.5 GB slack, 1 GB prompts, 3 GB page
cache, 2 GB for the user: `TONA_METAL_ARENA_FREE_GB`, an estimate); the working set minus our buffers; one buffer's
limit. Page cache counts as free, other programs' anonymous memory does not. `TONA_RAM_GB` caps it as on the PC.

On this machine with the server and MTP-256 (one run, diagnosis): budget 12.12 GB (all processes held 9.8 GB),
8.40 GB of pins (74.9 % of the uses) plus the 3.27 GB cache: **21.1 to 21.2 tok/s** decode, 5.3 cold experts per
stop. The compressor: 1.9 GB during the first request (load), 35 MB during the second. At 16 GB (above) 27 to 28
tok/s: the 4 GB more come out of the user's reserve. `tona-check` with auto: 71 of 71 top-1.

**Defaults (maintainer decision 2026-11-15):** the arena with auto is on when the weights the GPU reads exceed the
working set, with 32 cache slots per layer and lookahead 2; models that fit (DM-T) stay as before, identical at their
three references. With `TONA_RAM_GB` the user's share is outside the limit already, so `TONA_METAL_ARENA_FREE_GB`
(2 GB) applies only without it; the page-cache floor is the PC's 1.5 GB. Without a profile all experts rank equal
(Coder `tona-check` without profile: budget 15.1 GB, 71 of 71 top-1).

## Blocks and the governor (`TONA_METAL_ARENA_GOVERN`, `TONA_METAL_ARENA_CHURN`)

The arena is now blocks of at most 32 MB (one buffer each, all in the residency set); the expert kernels read a
table of GPU addresses (`MTLBuffer.gpuAddress`, a pointer wrapped in a struct, Metal 3 on the M1). A freed block's
experts become cold. Order: the cache's last slots (slot-major, so every layer loses the same slots), then the least
used pins; read again in the reverse order. Blocks change only under the lock a pass holds until its command
buffers completed.

Checks (one run each, diagnosis):

| What | Result |
|---|---|
| Mini model, 1 MB blocks, churn every 3 passes | 204 of 204 top-1; widths (1 and 2 slots) 0 not bit-identical; prefill cuts byte-equal |
| Coder 14 GB, churn every 4 passes (17 events of 3.5 GB) against no churn | tokens identical; 71 of 71 top-1; widths 0 not bit-identical |
| Coder 16 GB, blocks of fixed 32 MB (experts never straddle) | 5527 pins instead of 5791 (3 % gaps): 25.7 to 26.0 tok/s |
| Coder 16 GB, blocks end before the expert that does not fit (no gaps) | 5791 pins, 27.4 to 27.6 tok/s (one buffer: 26.8 to 27.8) |

Two accounting findings with the governor on and no other load:

1. The auto budget missed our own buffers that were not touched yet (KV cache, scratch): the governor found 0 to
   1.4 GB free in the first request and freed 5 to 6.7 GB (decode 10 to 11 tok/s). Our buffers now count whole, and
   with the governor on the budget leaves its target (6 GB) free so that it starts in its dead band.
2. Metal wires the pages of resident buffers, the weight mappings too: wired 3.2 GB idle, 13.2 GB during a pass. So
   the mappings are not among the file-backed pages and must not be subtracted from the free RAM again.

After both: budget 12.07 GB, the governor quiet over two requests, 19.7 to 19.9 tok/s (6 GB left free instead of 5).
Not yet measured: a balloon (another program's memory growing) as in expert-tiers.md section 5: it displaces other
programs' memory and needs the maintainer's consent first.
