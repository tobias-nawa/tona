# Mac: the order effect of the server A/B

Issue #108. Follows [2026-11-14-mac-mvg-split.md](2026-11-14-mac-mvg-split.md#server-ab-result): in all six rounds
of that A/B the arm that started second was about 3 % faster (AB rounds B ahead, BA rounds B behind). An effect of
that size hides every kernel change below ~3 % from `scripts/ab.py` on the Mac.

## What the logs of that run already show

Both arms load the same model file, so a page cache of "the other arm's model" cannot be the cause; under macOS
`ab.py` also starts each arm once, discarded, right before its measured start (`--round-warmup auto`). The server
logs (one line per request, 14 requests per start, 25 starts):

- Tokens per pass and passes from the history are equal in all 25 starts (3.155, 121 of 1388): the work is the same,
  only the time per pass differs. Nothing is carried from one start to the next through the history.
- Each request's rate divided by its median over the 25 starts falls into two states: **slow ≈ 0.98, fast ≈ 1.04**
  (6 % apart). Most starts sit in one state from the first to the last request; the state changes mostly at a server
  start, a few times within one (starts 9, 14, 19 to 24).
- By position in the round (warm-up 1, measured 1, warm-up 2, measured 2) the mean rate is 33.8, 33.2, 33.7, 34.3
  tok/s; the second measured start is ahead in 6 of 6 rounds.

## Plan (written before the run)

**Question.** What sets the slow or fast state, and is its link to the position in the round systematic or chance
(6 rounds, p = 1/32 for 6 of 6 by chance one way)?

**Run.** A/A: `scripts/ab.py --a work/build/mtl --b work/build/mtl`, no environment difference, DM-T + MTP-164,
`--workload set`, `--fixed-rounds --max-rounds 6`, round warm-up as before (auto, on). About 57 min. In parallel a
sampler (`work/ab-sampler.py`, every 2 s, sums only): page-ins, swap-ins, (de)compressions and free pages
(`vm_stat`), `kern.memorystatus_level`, GPU device utilization (`ioreg`, includes our own server), the load average,
the CPU share of all programs but `tona-server`, and the thermal and performance warning levels (`pmset -g therm`).
GPU clock and temperature need root (`powermetrics`): not measured.

**Expectation.** A/A has no real difference: the mean of B against A within ±1 % per metric. If the order effect is
systematic, the second measured start is ahead again in at least 5 of 6 rounds by 2 to 4 %; if chance, the sign
follows no order. For the cause:

- page cache: page-ins during a start (above a few hundred MB) in the slow starts and none in the fast ones;
- memory pressure: compressions or swap-ins, or a lower `memorystatus_level`, in the slow starts;
- other load: other CPU share or GPU utilization outside our requests higher in the slow starts;
- none of these: a state of the GPU (clock, power) that we cannot read without root; then the fix is one that does
  not need the cause (e.g. a warm-up long enough that the state settles, or the per-request rates kept as samples).

A slow/fast split that follows none of the sampled values is a finding.

**Active tooling.** The sampler (reads only, 2 s interval, 60 ms per sample: about 3 % of one core of 10);
no `TONA_*` switches beyond the defaults.

## Results

Conditions: before warning (5.7 % other CPU), after "GPU 97 %" a minute after the end (the reading lags our own
runs), ok 20 s later; hw_report and checks in [2026-11-14-mac-ab-order.log](2026-11-14-mac-ab-order.log). A/A report:
[2026-11-14-mac-ab-order-aa.md](2026-11-14-mac-ab-order-aa.md).

**No order effect.** B against A per round, decode code: +1.1, −1.1, 0.0, +1.4, 0.0, +4.7 % (mean +1.0 %
[−1.2, +3.3], just past the expected ±1 %); text −0.4, −1.6, +0.4, −3.8, −0.8, +2.4 % (−0.6 %). The sign follows no
order: the 6 of 6 of the earlier run were chance against a slow drift.

**The two states again, as a block in time.** Starts 18 to 22 (about 12 min) ran fast (rate 1.044 to 1.048 of the
median), all others slow (0.98 to 1.01); start 23 began fast and turned slow. Within a start there is no idle time
between requests (gaps ≤ 1 s, the log resolution), and the state is already set at the first request (first request
1.09 in fast, 1.00 in slow starts): no clock ramp after idle as on the PC.

Per server start, from the sampler (sums; memory level = `kern.memorystatus_level`, percent of memory free):

| State | Starts | Page-ins per start | Compressed per start | Memory level min | CPU other programs |
|---|---|---|---|---|---|
| fast | 18 to 22 | 88 to 301 MB | 0.5 to 1.5 GB | 9 | 15 to 23 % of one core |
| slow | the others | 920 to 5960 MB | 1.2 to 12 GB | 3 to 9 | 16 to 127 % |

**Finding: the slow state goes with page-in traffic** (correction, same day: a balloon that reproduces the
page-ins and the memory level does not slow decode, [2026-11-14-mac-pin.md](2026-11-14-mac-pin.md#calibration-memory-pressure-alone-does-not-slow-decode);
the mechanism below is a hypothesis that the calibration there does not support, and the server A/B with every weight wired still has slow starts, [2026-11-14-mac-pin.md](2026-11-14-mac-pin.md#server-ab-result-pinning-does-not-remove-the-slow-state): refuted). Every slow start read 0.9 to 6 GB back from disk, no fast one more
than 0.3 GB; the CPU load of other programs does not separate the states (starts 12 and 13: 115 and 127 %, slow but
not the slowest). The weights are a read-only file mapping wrapped as Metal buffers without copy
(`engine/model/gguf.cpp` mmap, `engine/backend/metal/weights.mm` `newBufferWithBytesNoCopy`): with the memory level
at 3 to 9 % (model 25 GB on 32 GB) macOS drops clean weight pages and the GPU then waits for their page-in. How much
memory other programs hold decides how often; that drifts over minutes, which made the state come in blocks.

**For ab.py:** the order of the arms is not the problem; a memory-pressure drift is, and the alternation cancels it
only on average. Since this run `ab.py` logs page-ins, compressions and the memory level per server start
(`--load-log`, `scripts/ab_load.py`), so a slow start can be told from a slow arm.

**For the engine (not probed):** weights that macOS may not drop would remove the slow state in production too
(5 to 6 % decode here). Candidates: Metal residency sets (`MTLResidencySet`, macOS 15; llama.cpp ggml-metal keeps its
buffers resident with them) or wiring the mapping (`mlock`, limited for a user process). A probe needs its own plan.
