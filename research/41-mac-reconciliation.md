# 41: Large-tier decode on the M1 Max: estimate reconciled with the M0 measurements (#108)

Gate of M0 in the proposal ([40-mac-backend-proposal.md](40-mac-backend-proposal.md#d5-order-of-work)): before M1, the
estimate of 22 to 31 passes/s (up to 50 to 65 with prediction) is set against the external ~8 tok/s decode on the same
machine type ([39-mac-engines-2026-11.md](39-mac-engines-2026-11.md)) using what M0 measured. All rates are decode at
B = 1 (one token per pass) on the 256-expert Coder; every line names whether it is measured or estimated.
**Provisional:** the hit rates come from the llama.cpp #29887 curve; the coordinator's simulation on our route traces
(154 to 205 slots per layer) replaces them.

## Components per token

| Part | Value | Basis |
|---|---|---|
| Expert kernels, 48 layers × 10 experts, all resident | **10.2 ms** (untuned, 29 % of the bound) | measured ([mac-expertk](measurements/2026-11-11-mac-expertk.md), r4) |
| Dense part (4.11 GB: DeltaNet, attention, router, shared expert, LM head) | 14 to 21 ms | estimate: 60 to 85 % of 334 GB/s; dense rows are wide, so better than the experts' 29 %, unmeasured until M1 |
| Dispatch cost beyond the timed expert dispatches | ~1 ms | ~500 dense dispatches × 2 µs ([mac-dispatch](measurements/2026-11-11-mac-dispatch.md)) |
| **Resident pass** | **25 to 32 ms** | sum; the earlier estimate used 15.3 ms (bandwidth bound), the main error |
| Misses per token at 56 to 72 % resident | 29 to 72 (0.6 to 1.5 per layer) | 480 experts × (1 − hit rate 85 to 94 %), provisional curve |
| Cost of one layer with misses, on demand | ~0.45 ms | read at QD 1 to 2 (0.57 ms, [mac-ssd](measurements/2026-11-11-mac-ssd.md)) + event 55 µs ([mac-event](measurements/2026-11-11-mac-event.md)) − ~0.2 ms of hit experts computed before the wait (ds4's order) |
| Layers with at least one miss | 22 to 37 of 48 | Poisson with 0.6 to 1.5 misses per layer (estimate) |
| GPU slowdown while the SSD reads | 11 to 15 % of the read window | measured ([mac-overlap](measurements/2026-11-11-mac-overlap.md)) |
| GPU clock after waits | no loss below 10 ms of idle; 3.3× slower for ~16 ms after ≥ 30 ms | measured (same); per-layer waits stay far below 10 ms |

## Decode estimate, revised

| Case | Per token | Passes/s | Before M0 |
|---|---|---|---|
| on demand: wait per layer with misses | 25 to 32 + 10 to 17 = **35 to 49 ms** | **20 to 28** | 22 to 31 |
| prediction (router of l+1, 16 to 24 candidates, recall 80 to 88 %), residual misses waited for; prefetch costs ~8 % of compute (reads of ~5 MB per layer, 0.38 ms at 13 GB/s, during ~0.6 ms of layer work, at −11 to −15 %) | 27 to 35 + 1.5 to 9 = **29 to 44 ms** | **23 to 34** | 50 to 65 |
| the same with tuned expert kernels (~60 % of the bound, ~5 ms; target, not measured) | 24 to 39 ms | 26 to 42 | - |
| upper bound, all resident, bandwidth bound | 15.3 ms | 66 | 66 |

The on-demand number survives; the prediction number falls from 50 to 65 to 23 to 34, because a resident pass costs
25 to 32 ms with real kernels instead of 15.3 ms. Prediction pays less than assumed: it removes miss waits, but the
pass itself dominates.

## The gap to ds4 (8.25 to 8.44 tok/s, ~120 ms per token)

Applying the same components to the 512-expert Q2 model ds4 runs (dense ~3.4 GB, 1.4 MB experts, a cache of well under
half the experts, hit rates 84 to 91 % at 30 to 50 % residency from the mlx-lm thread in 39) gives ~25 ms of compute
and ~15 to 25 ms of miss waits, i.e. **~40 to 50 ms, 20 to 25 tok/s; ds4 reports 8.3**. A factor of 2 to 3 is not
explained by anything M0 measured. Candidates, none checked:

1. **Memory pressure:** ds4's budget is min(87.5 % of RAM, recommended working set) − 2 GiB, ~26 GB on a 32 GB Mac,
   close to the point where MawForge saw decode collapse from 13.9 to 2.3 tok/s (38). Our design sizes the arena under
   measured pressure instead (proposal, D3).
2. **Per-miss cost:** synchronous reads per layer at low queue depth, or reads into buffers that need a copy; our
   design reads into the arena at QD ≥ 16.
3. **Kernel efficiency on the M1 Max:** ds4's prefill of ~10 tok/s on the M1 Max points to kernels tuned for newer
   chips; Q2_K and IQ2_XXS were not timed here.

Only a profiled run of ds4 on this Mac settles it; that needs its Q2 weight file (download, size to be checked) and
is the maintainer's decision.

## Consequences for M1 and M2

- **M1 measures the dense part.** It decides the resident pass (14 to 21 ms assumed); everything above scales with it.
- **Expert kernel tuning is worth ~5 ms per token** (10.2 → ~5 ms) and more at B ≥ 2 (24.1 ms at B = 4), where
  speculation needs it; it belongs in M1, not after streaming.
- **M2 gate stays "above the external baseline"** (8.3 tok/s decode); the planning range for M2 is 20 to 28 passes/s
  on demand and 23 to 34 with prediction, untuned kernels, provisional hit rates.
- **Warm-up kernel at request arrival** (mac-overlap): the first ~16 ms of GPU work after a pause run up to 3.3× slower;
  one short busy kernel while the request is parsed hides it.
- **Timeout limits** hold with margin: event waits are capped at 100 ms, the measured abort comes at ~5 s
  ([mac-timeout](measurements/2026-11-11-mac-timeout.md)), and a single timeout leaves the queue usable.
