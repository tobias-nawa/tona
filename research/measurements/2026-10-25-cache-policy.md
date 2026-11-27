# Expert cache: soft eviction and the gap to Belady, 2026-10-25, own measurement

Traces from the real server (speculation on, 8 coding requests of up to 600 tokens, two server runs, 610 and 682
passes): `IE_ROUTE_TRACE_P` writes, per layer and pass, the router logits of each token and the prediction for l + 1.
Simulation (scratchpad `sim/softsim.py`, `sim/gap.py`), 90 slots, counted from 20 % of the run, preloading as in the
server (top-16 per token from the prediction, 8 candidates, at the back of the list, predicted ones protected).

| Hits over the unions | Run 1 | Run 2 |
|---|---:|---:|
| LRU without preloading | 84.1 % | 79.8 % |
| Belady with bypass, without preloading (oracle) | 92.3 % | 90.5 % |
| LRU with preloading (today) | 91.9 % | 90.1 % |
| router probability summed with decay, H = 3 / 10 / 30 passes | 90.7 / 90.2 / 89.2 | 89.4 / 89.0 / 88.1 |
| hits counted with decay (LRFU), H = 3 / 10 / 30 | 92.3 / 92.4 / 91.4 | 90.8 / 91.1 / 90.4 |

- With preloading, LRU is already level with Belady without preloading. The gap of 7–10 points without preloading is
  almost entirely closed by the prediction of the next layer.
- Soft eviction by router probability (idea "probabilistic Belady", SeqMoE arXiv 2609.12978) is
  worse than LRU: the probabilities of the experts that were not chosen say little about the next pass.
- LRFU +0.4–1.0 points in the simulation, earlier without gain in the server (measurements/2026-10-24-misses).
- 70 % of the LRU misses are in the top-16 of the prediction of some token, 30 % never.

Conclusion: there is little left to gain in eviction. More hits only come from more slots.

## Desktop VRAM

At measurement time only 73 instead of 81–91 slots per layer: sysfs 2,832 MiB in use without the engine (normally ~1.4 GB).
Per process (`drm-memory-vram` in fdinfo): the compositor 1,918 MiB, the desktop shell 1,472 MiB, after 26 h of uptime. 10 slots
more earlier brought −0.5 ms per pass (B = 4).
