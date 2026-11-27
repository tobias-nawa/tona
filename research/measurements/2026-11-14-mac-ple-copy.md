# Mac: the host copy of the n-gram rows per pass, against the fast and slow server starts

Issue #108. Follows [2026-11-14-mac-pin.md](2026-11-14-mac-pin.md#server-ab-result-pinning-does-not-remove-the-slow-state):
with every weight wired the slow starts stay. Open candidate: the n-gram table (28.8 GB file mapping, part 2 of DM-T)
is only page cache; a token reads 16 rows of 90 bytes on 16 random pages, copied on the CPU before the pass
(`engine/backend/metal/pass.mm` `write_inputs`). A row whose page is not cached stalls the pass on the SSD.

## Plan (written before the runs)

**Residency (done before this plan, `mincore` on the file, read only).** Idle: 0.00 GB of 28.8 GB resident.
`tona-bench DMT -b 1 -p 4096 -d 128 -r 1 --ctx 8192`: afterwards 1.28 GB (+1.06 GB for 4224 tokens, 0.25 MB per
token: 16 pages of 16 KB, almost no reuse).

**Probe switch.** `TONA_PLE_STATS=1` (`engine/backend/ple_stats.h`, shared): times the copy in `write_inputs` and counts
the process's major faults (`getrusage`) around it; one stderr line per 1000 decode passes with the interval's means,
decode and prefill apart. Off: nothing is timed (no behaviour change either way).

**Smoke test (one run, not a result).** `tona-bench DMT -b 1 -p 512 -d 1100 -r 1 --ctx 4096`: first 1000 decode passes
1080 us and 5.7 major faults per pass, next 1000 102 us and 0.5; the measured prefill of 512 tokens 859 ms and 8137
faults (16 per token, 0.1 ms each). Cold rows cost about a third of the prefill and about 2 % of a decode pass.

**Run.** `scripts/ab.py --workload set`, DM-T + MTP-164, A/A (same build, `TONA_PLE_STATS=1` in both arms),
`--fixed-rounds --max-rounds 6`, `--load-log on`; about 60 min. Per server start from its log: the PLE lines (decode
us and faults per pass, prefill us and faults per pass); from the load log page-ins and GPU share; from the report the
decode rate of the measured starts.

**Expectation.** A/A: B against A within ±3 % (the A/A of 2026-11-14-mac-ab-order.md: +1.0 % [−1.2, +3.3]). Slow starts
(decode rate ≥ 3 % below the run's median) show more PLE time per decode pass than fast ones, by at least the time
they lose: at 38 tok/s with MTP a pass takes about 50 ms, 3 % are 1.5 ms per pass, so slow starts at ≥ 1.5 ms of copy
per decode pass against ≤ 0.3 ms in fast ones. If the copy time does not separate the starts, the hypothesis is
refuted and the probe ends with the measurement.

**Active tooling.** `TONA_PLE_STATS=1` in both arms, the load log; defaults otherwise.

## Result: the slow starts are the n-gram rows read back from the SSD

Conditions: before ok, after "do not measure" (GPU 96 % busy: the reading lags 1 to 2 min, it is the run's own
server) and 20 s later ok; log in [2026-11-14-mac-ple-copy.log](2026-11-14-mac-ple-copy.log). Report and load table:
[2026-11-14-mac-ple-copy-server-aa.md](2026-11-14-mac-ple-copy-server-aa.md). A/A B against A: code −0.93 %
[−2.14, +0.28], text +0.12 % [−1.28, +1.51], both within ±3 % as expected.

Each server start printed one PLE line (its first 1000 decode passes, 4.33 tokens per pass; prefill of 55 tokens per
request). Measured starts (round, arm: copy per decode pass, major faults per pass, decode-code and decode-text
tok/s, GPU ours %, machine page-ins, memory level min):

| Round | A | B |
|---|---|---|
| 1 | 5.36 ms, 36.0, 37.0 / 25.4, 89.2 %, 3.2 GB, 5 | 5.43 ms, 36.0, 36.8 / 25.6, 89.1 %, 4.0 GB, 5 |
| 2 | 5.64 ms, 36.1, 36.8 / 25.6, 88.9 %, 2.8 GB, 5 | 5.57 ms, 35.3, 36.6 / 25.3, 89.3 %, 2.3 GB, 5 |
| 3 | 5.58 ms, 36.4, 37.1 / 25.6, 88.4 %, 4.4 GB, 5 | 5.55 ms, 34.6, 37.1 / 25.3, 89.3 %, 1.4 GB, 9 |
| 4 | 0.10 ms, 0.19, 38.3 / 26.3, 94.2 %, 0.6 GB, 9 | 0.25 ms, 1.16, 38.2 / 26.7, 94.7 %, 0.2 GB, 9 |
| 5 | 3.05 ms, 19.5, 37.6 / 25.0, 90.9 %, 1.5 GB, 8 | 5.66 ms, 35.0, 36.6 / 24.9, 89.2 %, 1.5 GB, 7 |
| 6 | 0.27 ms, 0.89, 38.1 / 26.4, 94.3 %, 0.3 GB, 8 | 1.99 ms, 11.4, 37.5 / 26.7, 92.7 %, 0.5 GB, 8 |

Discarded starts follow the same pattern (all 25: 5.2 to 6.3 ms or 0.09 to 0.53 ms, three in between).

**Finding: inside the expectation, the hypothesis holds.** The copy time separates the starts: 8 measured starts at
5.4 to 5.7 ms per decode pass (35 major faults, half of the 69 rows of a pass read from the SSD) run decode-code at
36.6 to 37.1 tok/s, the 3 at 0.1 to 0.3 ms at 38.1 to 38.3 (−3.7 % for the slow ones); the two in between lie in
between. Correlation of copy time and decode rate over the 12 measured starts: code r = −0.96 (−0.25 tok/s per ms),
text r = −0.80. Prefill pays more: 62 to 87 ms and 570 to 730 faults per 55-token request when cold, under 3 ms warm.

This ties the earlier findings together: the slow starts' machine page-ins (1.4 to 4.4 GB) are mostly these rows
(35 faults of 16 KB per pass are 0.6 GB in the first 1000 passes alone), and the lower GPU share for us (88 to 89 %
against 94 to 95 %) is the GPU waiting for the copy before each pass. Pinning the weights could not help and left
the table less page cache (the pin A/B's B −1.5 %). Fast starts are those where the rows of the workload were still
cached from earlier starts.

Open: the extra 5.3 ms per pass is more than the slow starts lose if a pass takes 50 ms (3.7 % are 1.9 ms); either
the pass is longer than estimated or part of the copy overlaps the GPU's previous pass. Not needed for the next step.

**Next step: read the rows in parallel instead of one fault after the other.** The faults are serial (one 16 KB read
at queue depth 1, about 150 us each: 35 x 150 us = 5.3 ms); an SSD serves dozens at once. Bit-identical, only timing
changes: (1) at the copy, touch all pages of the pass's rows concurrently (`madvise(MADV_WILLNEED)` per page, or a
few reader threads) before copying; (2) then, during the running pass, the rows of the next pass's candidates (MTP
drafts), and for prefill the next chunk. In shared code next to `ple.h`, off on Linux until measured on the PC.
