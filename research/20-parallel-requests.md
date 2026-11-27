# Several requests at once (research 2026-10-26)

Three agent research runs (Sonnet), plus own numbers. [V] full text or page read (via WebFetch, hence
summarized), [S] search hit/abstract only. For batch 2–8 on bandwidth-bound consumer hardware none of the
three research runs found measurements; almost everything is measured on A100/H100 from batch 16–32.

## Starting point (own measurements)

Pass time by tokens per pass (measurements/2026-10-25-width-8): B = 1 23.4 ms, 2 29.7, 4 39.0, 8 ~54–65 ms. The increase
comes almost entirely from the MoE layers (gate_up 7.5 → 14.3 ms, down 6.3 → 12.0 ms from B = 4 to 8): more tokens touch
more different experts, the missing ones are computed by the CPU.

Calculation for two conversations (draft length per conversation chosen as today, ~2.9 tokens at B = 4):

| | Pass | Tokens | total | per conversation |
|---|---:|---:|---:|---:|
| one after the other | 39 ms | 2.9 | 74 tok/s (decode) | 37 (alternating) or 74 alone |
| together, 4 each (B = 8) | ~60 ms | 5.8 | ~97 tok/s (decode) | ~48 |
| together, 3 each (B = 6, ~47 ms estimated) | ~47 ms | ~4.8 | ~102 tok/s (decode) | ~51 |

True batching with two workers thus brings roughly +30–40 % throughput, but each conversation runs at ~50 instead of 74 tok/s (decode).
The numbers for independent conversations may be worse: draft tokens of the same conversation share more
experts than tokens of different conversations (XShare: 2–3× more overlap).

For Claude Code alone parallelism brings little: the main conversation and the auto-mode checker run one after the other (the
checker blocks the tool call). It pays off for the actual purpose of the engine, several workers of a
coordinator.

## 1. Experts in the batch

| Source | Core | For us |
|---|---|---|
| [V] OEA, arXiv 2511.02237 (2025-11, SGLang, batch ≤ 16) | Each token keeps its best k0 experts, further ones only from the union of the batch. Qwen3-30B: active experts per layer 48.8 → 25.1, MoE time −39 %; LiveCodeBench 62.1 → 61.2 (30B), 68.5 → 66.1 (235B) | most direct lever, training-free; measure code quality ourselves |
| [V] XShare, arXiv 2602.07265 (2026-02) | shared expert set per layer and batch, variant for speculation; +7–13 % at ≤ 1 % loss, aggressive +50 % with AIME −10.8 | like OEA, stronger; warm-up (keep top-k0) decisive |
| [V] LayerScope/PreScope, arXiv 2509.23638 | prediction of the experts of the next layer, scheduler CPU/GPU/transfer; optimizes throughput, latency per request rises with the batch | preloading idea, see research/14 |
| [S] Fiddler 2402.07033, HybriMoE 2504.05897, kTransformers | CPU time grows with tokens per expert, upload is fixed: few tokens to the CPU, upload for many | measure the threshold here; own estimate ~2–4 tokens per expert |
| [S] MoE-Lightning, MoE-Gen, MoE-Lens | giant batches offline | unusable for interactive workers |

Overlap under uniform distribution (N = 512, k = 10): B = 4 ~39, B = 8 ~74 different experts per layer; real routing
lies below (OEA: 48.8 instead of 82 on Qwen3-30B, B = 16). For coding and Qwen3-Next there is no measurement.

## 2. Drafts in the batch

| Source | Core | For us |
|---|---|---|
| [V] vLLM Adaptive Verification (DSpark), PR #47808, merged (blog 2026-08-14) | global top-B over the survival probability (running product of the draft confidences) of all sequences, B maximizes E[tokens]/t(B) with a measured cost table | exactly our problem; our width choice does this today for one sequence |
| [V] SmartSpec, arXiv 2406.14066 | goodput = expected tokens / step time; low load: long drafts, high load: speculation off | same objective function |
| [S] D-cut 2607.14647, EVICT 2605.00342, MoE-Spec 2602.16052 | verification budget by confidence across the batch; MoE cost as a function of width | confirms the approach |
| [S] Batch-SD "Done Right", arXiv 2510.22876 | padded batches cost up to 40 % at B = 8 | a flat token array with sequence IDs avoids this |

## 3. Memory and state per conversation

| Source | Core | For us |
|---|---|---|
| [V] SGLang Mamba radix cache (blog 2025-12, #12867), unified radix cache (2026-08) | fixed state slot per request, KV pool only for full attention; a hit copies the state into the slot (copy-on-write); one slot per draft token | fits MTP; our storage in RAM is the same idea one level deeper |
| [S] vLLM `--mamba-cache-mode align`, issue #40696 | state only on a fixed block grid, experimental | our intermediate-point method is more precise |
| [S] llama.cpp `--parallel` | repeated "full prompt re-processing" (#18497), slots slow things down even when idle (#19523) | no reliable numbers on 16 GB |
| [S] Hydragen 2402.05099, ChunkAttention 2402.15220 | compute the shared prefix once | small with 2–4 sequences and few full-attention layers |
| [S] Autellix 2502.13965, Continuum 2511.02230 | priority by compute time received so far, short calls first; hold KV during tool calls | priority idea adoptable |

Time slicing (switch per pass) brings no throughput and costs 60–250 ms of state exchange per switch; only sensible
coarsely, per answer or tool call, as today through the queue.

Own derivation, not researched: prefill is compute-bound, decode bandwidth-bound. A prefill piece of one
conversation in the same pass as the decode of another (Sarathi-Serve pattern) might be almost free. Worth its own research.

## Path (proposal)

1. **Measure:** record the router selection of two real coding conversations, determine the union per layer, reproduce the pass time
   for two independent conversations with `IE_FAKE_ROUTE`. Decides everything else.
2. **Priority in the queue:** main conversation and short check requests ahead of background workers. Small, usable immediately.
3. **True batching of 2–4 sequences:** megakernel with a sequence ID per row (position, KV, DeltaNet state per
   sequence), state slots in VRAM, global draft width across all sequences (DSpark pattern, our cost table).
   Large: touches the megakernel, prefill cache, MTP head.
4. **Afterwards:** batch-aware routing (OEA) with measurement of code quality; prefill and decode in the same pass.

## Build plan stage 3 (2026-10-26, implemented, result in measurements/2026-10-26-parallel)

State of a conversation today (all allocated via `alloc_state` plus KV): DeltaNet state, conv rings,
intermediate values of the drafts (two halves for committing), PLE history, QSA indexer ring, MTP state: together
**149 MB fixed**; plus per QSA layer KV (~15 KB per token, 45K tokens ≈ 680 MB) and the indexer pool. Measured via
the storing (`Conversation … stored: 909 MB (state 149, KV 683, pool 76)`).

1. **Slots:** every state buffer S times in a row (`alloc_state` lays out [S][size], fixed stride), KV and pool
   an own region per slot. Slot 0 is the current addresses.
2. **Row → conversation:** MegaParams gets per row the slot and per slot position, commit count and half
   (today one value per pass). Only the steps with state compute with the slot offset: dn.conv, dn.recur, dn.norm
   (intermediate values), qsa.prep/score/topk/attn/merge, ple.conv, MTP steps. Matvec, MoE, router, norms stay
   unchanged (they only see nb rows).
3. **One conversation active:** offset 0, same tasks as today. Gate: bench set before/after, no loss beyond
   the noise.
4. **Planner in the generator:** collects the running conversations; drafts per conversation from the MTP head; width across all
   conversations like vLLM DSpark (survival probability per draft, cost table t(B) from the measurement, at most
   8 tokens per pass); verification and acceptance separate per conversation. Prefill alone at first (idea 1 later). Priority as
   in stage 2: the main conversation gets its draft first.
5. **Prompt cache:** store and fetch into a specific slot; intermediate points unchanged.
6. **Memory for slots 1…S−1:** In VRAM only ~0.4 GB are free (14.5 GB engine, 1.4 GB desktop). Ways: (a) when a second conversation
   starts, give up expert slots (149 MB ≈ 4–5 of 88 slots per layer), return them at the end; (b) state in
   pinned RAM (per pass read and write ~300 MB over PCIe, ~5 ms); KV with long context is in RAM anyway. Measure
   first, then decide.
7. **Verification:** IE_DETERMINISTIC: two conversations together must deliver the same tokens as each alone; GPU test with
   warning (new kernel paths).

Goal: two conversations together ≥ +25 % throughput, each ≥ 45 tok/s decode; unchanged alone.

### State after the build (2026-10-26)

- Steps 1–5 and 7 implemented. Memory (6): variant a, fixed at start (measurements/2026-10-26-slot-memory).
- Deviations from the plan: instead of a slot offset at runtime, an own program per slot with the addresses of the slot
  (one conversation alone computes unchanged); in the program of several conversations the host sets per pass the pointers to
  the rows of each slot (one copy of the step list, ~200 KB). DFlash only with one conversation.
- Result: bitwise identical; one conversation unchanged in speed (the second compartment costs ~0.7 %); two together 47–48 tok/s (decode)
  per conversation, total +10 % in the shared passes instead of the hoped-for +25 %. Reason: MTP already fills the pass with
  one conversation, two unrelated conversations share fewer experts.
- Stage 4 (OEA) built and measured; on since 2026-10-27 as k0 = 7 with protection of the experts in VRAM (measurements/2026-10-26-oea). Further levers: MTP steps
  of both conversations in one MTP pass, prefill of one conversation next to the decode of the other.
- Additional ideas, state of the evening of 2026-10-26 (measurements/2026-10-26-prefill-slices):
  - built: prefill in slices, intermediate point at the branch, effort per role.
  - Expert cache following the running conversations: LRU already does this (hit rate 83.7 % with one client, 79.6 % with
    two, bench-parallel); the simulation of 2026-10-25 (measurements/2026-10-25-cache-policy) shows that there is little to gain in
    eviction. No rework.
  - Tool wait time: while one conversation waits for a tool, the other already continues alone; more
    would remain only in storing (50–110 ms per switch), no build of its own.
  - MTP steps of both conversations in one pass: one step costs 1.06 ms (balance over 223 single passes),
    in shared passes 5.4 steps = 6.8 ms; together roughly the max of the chains ≈ 3–4 steps → ~2.6 ms of 64 ms
    (~4 % only in shared passes). Large rework (MTP program per slot like xops), postponed.
