# CECAR offline replay on recorded routing traces (#172), 2026-11-18

Replay only: CPU, no GPU, no engine run, **no change to the engine's routing**. Tool: `tools/cache_sim/cecar_sim.py` (new, with `tools/tests/test_cecar_sim.py`). Raw outputs: `2026-11-18-cecar-offline-replay.txt` (trace A) and `2026-11-18-cecar-offline-replay-cross.txt` (trace B). Note for the record: the first simulator run (5 layers) was started before this plan was written down, as a tool check; the numbers below come from the runs described in the plan, started after the plan.

## Plan

- Question: what hit-rate (and misses per pass) does the reuse-distance predictor of CECAR (arXiv 2611.20119) give over our cache policies at exact routing, and how much would routing have to change for the larger gain of its bonus and pruning?
- Scope: `TONA_ROUTE_TRACE_P` traces (router logits per token, layer and pass; Coder, 256 experts, top-10, MTP verify passes of 4 tokens): **A** `/var/tmp/claude/148/res/s1.tracep` (2026-11-15, 14 bench-set prompts, 1,122 passes, 5,474 token rows); **B** `trp1.bin` (610 passes, fit) and `trp2.bin` (682 passes, test) from 2026-10-25 (8 coding requests each). Nine layers (0, 6, 12, 18, 24, 30, 36, 42, 47) out of 48, 90 slots per layer, hits over the union of the experts of a pass, counted from 20 % of the run. Run A: fit on the first half of the passes, test on the second half. Run B: fit on trp1, test on trp2 (a different run).
- Policies at exact routing, demand only (no preloading, to isolate the eviction policy): LRU, LRFU (half-life 16 passes), a learned reuse predictor, Belady with bypass (oracle). Learned predictor = our reading of the paper: expert-independent, per expert the paper's two inputs (1 / recency, uses / largest use count) plus three decayed frequencies and log recency, logistic regression on the products of these features, target "used again within 8 passes", one model for all layers. It is not the paper's two 32-unit networks (FFN-Hot, FFN-Cold) with the cache-size weights; the paper reports that the scorer's size changes the OOD hit rate by less than 0.5 %.
- Routing changes (CECAR, our reading of its Algorithm 1): bonus `z'_e = z_e + (z[1] - z[m]) * d_e` for resident experts, `d_e` the predictor's reuse probability scaled to 0..1 over the resident experts; top-k on `z'`; then pruning: a selected expert stays if resident or if its rank is within `tau` (0-based rank below tau). Weights are those of the original softmax, normalised over the original top-10; a dropped expert's weight is lost. Configurations: pruning only (tau 4, 6, 8) and bonus with margin index m of 2, 3, 5, 10 and tau 4, 6, 10 (tau 10 = no pruning, bonus only).
- Metrics: hit rate, misses per layer and pass, **replaced**: share of the original top-10 selections that are not in the final selection (counted per token), **weight mass lost**: mean share of the original normalised top-10 weights that those carry. This is a proxy for how far the output moves; it is not a quality measurement.
- Why this size: the question is answered by hit rates, which need no GPU; a trace replay is deterministic, so no repeated rounds; nine of 48 layers (spread over the depth) keeps the run to a few minutes on one thread. The two traces are used to see whether the order of the results depends on the trace.
- Expectation (stated before the final runs, from `2026-10-25-cache-policy.md`): learned over LRU +1 to +3 points; LRFU +0.4 to +1.0; Belady 10 to 12 points over LRU demand-only; the routing configurations give large hit-rate gains whose price is in the two change metrics.
- Active tooling: none; numpy 2.4.6, Python `-I`.

## Test conditions

CPU-only job through the `cpu` queue; no speed claim, so no `quiet_check` gate (the machine had another worker's GPU job running: 14.89 GiB VRAM held by others, 100 % GPU busy before the run). Commit 14cad0f plus the uncommitted new tool.

```
- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM (not used by this job)
- RAM: 60.5 GiB, 23.8 GiB available before the run
- OS: Fedora Linux 44, kernel 7.3.4-200.fc44.x86_64; Driver and toolkit: hip 7.1.52802-9999
- Load average 2.27, 3.74, 2.71; swap in use 1.43 GiB
```

## Results (nine layers, 90 slots, demand only)

Trace A (fit first half, test second half of 1,122 passes) / trace B (fit trp1, test trp2):

| Policy or routing | Hit A | Hit B | Misses per layer-pass A | B | Replaced A | B | Weight mass lost A | B |
|---|---|---|---|---|---|---|---|---|
| LRU, exact | 77.86 % | 76.29 % | 6.35 | 6.38 | 0 | 0 | 0 | 0 |
| LRFU, exact | 78.79 | 78.25 | 6.04 | 5.85 | 0 | 0 | 0 | 0 |
| learned predictor, exact | **79.35** | **78.64** | 5.90 | 5.75 | 0 | 0 | 0 | 0 |
| Belady (oracle), exact | 89.08 | 88.32 | 3.14 | 3.15 | 0 | 0 | 0 | 0 |
| pruning only, tau 8 | 82.67 | 82.55 | 4.45 | 4.25 | 5.4 % | 5.4 % | 2.6 % | 2.7 % |
| pruning only, tau 6 | 86.76 | 87.12 | 3.19 | 2.92 | 10.8 | 10.9 | 5.7 | 5.8 |
| pruning only, tau 4 | 91.08 | 91.75 | 1.98 | 1.71 | 16.8 | 17.1 | 9.9 | 9.9 |
| bonus m 2, tau 10 | 84.01 | 84.44 | 4.24 | 3.85 | 9.0 | 11.1 | 4.8 | 5.9 |
| bonus m 2, tau 6 | 90.14 | 91.03 | 2.37 | 2.01 | 17.4 | 19.0 | 9.9 | 10.8 |
| bonus m 3, tau 6 | 91.97 | 92.90 | 1.93 | 1.57 | 21.4 | 23.9 | 12.6 | 14.0 |
| bonus m 5, tau 6 | 93.93 | 94.91 | 1.44 | 1.10 | 27.5 | 29.3 | 17.1 | 18.1 |
| bonus m 10, tau 6 | 96.69 | 97.12 | 0.75 | 0.59 | 37.9 | 37.6 | 27.0 | 25.7 |
| bonus m 10, tau 4 | 98.12 | 98.37 | 0.41 | 0.33 | 44.3 | 43.4 | 34.3 | 32.2 |

(The full 15-row routing grid is in the two `.txt` files.)

## Reading against the plan

- Within the expectation: learned over LRU +1.5 points (A) and +2.4 (B); LRFU +0.9 and +2.0; Belady +11.2 and +12.0. The order of the policies is the same on both traces.
- **Paper claim reproduced in kind, not in size:** the paper's learned policy alone gives +3 to +4 points over LRU at exact routing (Qwen3-30B-A3B, 24 of 128 experts); we get +1.5 to +2.4 with our own feature set; this is a reimplementation of the idea, not their trained networks, and the cache here holds 35 % of the experts (90 of 256), theirs 19 %.
- The routing configurations reach hit rates the exact policies cannot (above Belady's 89 %), but **every configuration that gets above Belady replaces 17 to 44 % of the original top-10 selections and drops 10 to 34 % of the routing weight** on this model (256 fine-grained experts, top-10, flat router distributions). The paper's Qwen3-30B-A3B result keeps 7.13 of 8 experts on average (about 11 % dropped); our smallest rows with a comparable drop (pruning only, tau 8: 5 % replaced) give +5 points of hit rate, or -30 % misses against LRU.
- Mixing in the bonus with a small margin (m 2, tau 10) gains 6 to 8 points over exact LRU for 9 to 11 % of selections replaced; pruning alone at tau 6 gains 9 to 11 points for 11 % replaced. The bonus on top of pruning at equal tau adds 3 to 5 points for about 6 to 10 more points of replaced selections.
- Caveats, outside what the replay can show: (1) the token stream is the recorded one; a changed routing changes the following tokens and the traces cannot say how; (2) demand only, no preloading: the server's next-layer preloading already brings LRU to about 90 % hits (`2026-10-25-cache-policy.md`), so the same relative change in misses is worth less in the real engine; (3) fit and test come from the same bench-set style prompts (A) or the same kind of coding requests (B); no workload-diverse trace (code against prose, long contexts) was available (`2026-11-15-expert-coactivation.md` plans one); (4) verify passes of 4 tokens, rejected draft tokens included; (5) our predictor is not theirs; (6) nine of 48 layers.
