# Mac: routed experts of a decode pass grouped by expert (#108)

M1 Max 32 GB, Metal backend, target DM, MTP head cut to 164 experts.

## Plan (written before the run)

**Question.** What does the grouped expert mat-vec (`mvg`: one threadgroup row per expert the pass uses, all its
slots decoded together; gate and up in the row-spread shape instead of split-K) bring per pass width and in the
server? Profile behind it: 2026-11-12-mac-verify-profile.md (routed experts half of the time a width-4 pass adds,
~54 GB/s at width 1).

**Arms.** A = commit `e7c99bb` (binaries in `work/build/mtl-base`), B = the same plus the change (`work/build/mtl`).
`scripts/ab.py` starts servers through `systemd-run`, which macOS lacks; the A/B here alternates by hand.

**Scope.**

- `tona-bench DM -p 512 -d 48 -r 3 --ctx 4096 -b W` for W = 1, 2, 3, 4, 8, per width A B A B (four processes, one
  warm-up round each). ~6 min.
- Server: `work/m3/m3.sh` arm B of M3 (DM + MTP-164, ctx 16384, chunk 2048, bench-set), A then B, one round each
  (greedy: rounds of one build gave identical token counts and 0.0 to 0.7 tok/s spread in M3 and its follow-up). Both
  arms use the pass costs committed in `510bc5e` (measured on A). ~8 min.
- Correctness before the run (checks 1 to 4 of the backend AGENTS.md, done): mini 207 of 207 steps (9.4e-7), widths 0
  of 128 rows not bit-identical (1 slot) and 0 of 292 (2 slots), cut invariance byte-equal; DM 39 of 39 (1.4e-6),
  widths 0 of 109, cut invariance byte-equal, prefill against decode the same as A (max |d nll| 1.24e-2, worst
  4.05e-2, also on A); MTP against the reference's golden 241 of 256, median 5.83e-2 (as A).

**Expectation.**

- W = 1: B 0.5 to 2 ms faster than A (45.4): a probe with gate and up in the row-spread shape gave -1.2 ms.
- W = 4: B 85 to 97 ms against A 108: the probe's -6.6 ms plus the shared experts decoded once. The head's routing
  (512 experts) shares 22 % of 40 slots at width 4; the trunk picks from 164, so more (estimate 25 to 35 %) of the
  ~37 ms of experts a width-4 pass adds.
- W = 8: B 140 to 175 ms against A 202.5 (head: 39 % shared at width 8).
- Server code decode: B 27 to 31 tok/s against A ~26.5; text B 20 to 21.

**Active tooling.** None (`TONA_METAL_PROFILE` off, no `TONA_*` variable). `caffeinate -dimsu`. Quiet check and
`tools/hw_report.py` before and after.

## Results of the first set

Quiet check before 08:53:48 ok, after 09:06:06 ok. `tools/hw_report.py` as in 2026-11-12-mac-mtp-server.md (M1 Max,
32 GiB, macOS 27.0.1, no `TONA_*` variable), commit `e7c99bb` plus the change (dirty tree).

Pass, median of 3 rounds per process, ms (A B A B):

| Width | A | B | B against A | Expected B |
|---|---|---|---|---|
| 1 | 45.84 / 45.52 | 46.85 / 46.76 | +1.1 ms (+2.5 %) | 43.4 to 44.9: **finding** |
| 2 | 64.87 / 64.55 | 61.36 / 60.99 | -3.5 ms (-5.5 %) | |
| 3 | 86.74 / 86.63 | 78.71 / 78.25 | -8.2 ms (-9.5 %) | |
| 4 | 107.25 / 107.75 | 94.06 / 94.29 | -13.3 ms (-12.4 %) | 85 to 97 |
| 8 | 203.40 / 203.46 | 173.22 / 172.93 | -30.4 ms (-14.9 %) | 140 to 175 |

Server (bench-set, one round per arm, the same 1,746 passes and 2.67 tokens per pass in both, greedy): code A 26.2,
**B 28.4 tok/s decode** (+8.4 %, expected 27 to 31); text A 19.6, B 20.3 (expected 20 to 21). Raw:
[A](2026-11-12-mac-moe-grouped-a.jsonl), [B](2026-11-12-mac-moe-grouped-b.jsonl).

- Width 1 is slower: the grouping dispatch (`moe_group`, one threadgroup, a serial prefix sum) runs once per layer
  although a single token has one slot per expert anyway. Change before the second set: at width 1 no grouping, the
  row-spread `mv` with one slot per expert (the same arithmetic as `mvg` with one slot). Checks after it: widths 0 of
  109 rows not bit-identical (mini and DM, 1 slot), 0 of 292 (mini, 2 slots); mini 207 of 207; DM 39 of 39 (1.4e-6);
  cut invariance byte-equal; MTP golden 241 of 256, median 5.83e-2.

## Second set (plan written before the run)

Width 1 only, A B A B, same command; ~2 min. Expectation: B 0.5 to 2 ms faster than A (the probe with gate and up in
the row-spread shape: -1.2 ms). Widths 2 to 8 and the server are unchanged by the change (they group as before).

**Results of the second set** (09:12 to 09:14): A 45.44 / 45.48 ms, **B 44.60 / 44.38 ms (-1.0 ms, -2.1 %)**, inside the
expectation. The quiet check before warned of load average 4.17 (the checks just before, lagging a minute), after the
run "GPU 98 % busy" (its own tail); the arms alternated, so both ran under the same conditions.

## Third set: widths 5 to 7 of B for the pass costs (plan written before the run)

`work/build/mtl/tona-bench`, same command, W = 5, 6, 7, one process each; ~1.5 min. Expectation from A's 145.3,
162.2, 184.4 ms and B's -12 to -15 % at widths 4 and 8: 125 to 132, 138 to 145, 157 to 165 ms.

**Results of the third set** (09:15:47 quiet check ok): W = 5 127.4 ms, W = 6 141.6, W = 7 159.4 (spread ≤ 0.4 ms),
inside the expectation. Pass costs of the Metal backend (`metal_pass_costs()`) from B: 44.5, 61.2, 78.5, 94.2, 127.4,
141.6, 159.4, 173.1 ms for widths 1 to 8 (medians of the sets above); MTP step and fill unchanged (not remeasured).

## Reading

- The grouped experts take 12 to 15 % off passes of width 3 to 8 and 2 % off width 1; the server decodes code at
  28.4 tok/s (+8 %). Every token past the first now adds ~16 ms (A: ~21; reference ~6).
- The wider widths gain most (W = 8: -30 ms): their tokens share more experts. Still open toward 40 tok/s: the
  remaining expert decode (latency/ALU-bound, 2026-11-11-mac-expertk.md), the DeltaNet projections (+11.7 ms at
  W = 4) and the gated reads (+8.7 ms).
- Found on the way and fixed in the change: the expert offsets of the grouping (`goff`) held the target's expert count
  plus one; an MTP prefill with the full 512-expert head wrote past them. The cut head (164) was not affected.
