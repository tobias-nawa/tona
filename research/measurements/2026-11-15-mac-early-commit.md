# Mac: the target pass committed early (`TONA_METAL_EARLY_COMMIT`)

Issue #108. Follows [2026-11-14-mac-gpu-gaps.md](2026-11-14-mac-gpu-gaps.md#result-of-the-second-inputs-run-the-target-passs-encoding-is-082-ms-per-cycle):
encoding one target pass costs about 0.82 ms of host time per decode cycle while the GPU waits.

## Plan (written before the run)

**Change under test.** `Runner::run_split`: with `TONA_METAL_EARLY_COMMIT=k` the pass's inputs, its embedding and its
first k layers go into a first command buffer that is committed at once; the remaining layers and the head are encoded
while the GPU runs them, into a second command buffer on the same queue. The same kernels in the same order:
bit-identical. The gap probe counts the GPU's idle between the two halves as "split".

**Prior art.** llama.cpp's Metal backend commits the first `max(64, 10 %)` nodes of a graph from the calling thread and
encodes the rest while they run (`ggml/src/ggml-metal/ggml-metal-context.m`, `n_main`); the reference PC's megakernel
does not encode per pass, so nothing transfers. Indirect command buffers (encoding once, replaying) are the next step
if encoding still shows.

**Checks (done, k = 2).** tona-check DM-T `TONA_DETERMINISTIC=1` identical; widths (seed 7) 0 of 66 rows, with
`TONA_SEQS=2` 0 of 160; prefill cut (chunk 64, seed 5) identical; MTP golden 241 of 256, pick check 0 of 200, hash as
before; 54 of 54 unit tests.

**Run.** `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, DM-T + MTP-164, 6 fixed rounds (about
80 min), `--load-log on`, both arms `TONA_GPU_GAPS=500`. A and B the same build (52e798e merged with main, c3a3f53,
plus this change; no kernel source changed since c3a3f53, so the metallib is the one of A's commit), A without the
switch, B `TONA_METAL_EARLY_COMMIT=2` (two of 40 layers, about 4.6 ms of GPU work, against 0.8 ms of encoding of the
rest).
**Expectation.** encoding target 0.82 → under 0.1 ms per cycle (begin and two layers); split under 0.05 ms; GPU busy
per cycle unchanged within 0.05 ms (one more command buffer); idle −0.7 to −0.8 ms per cycle; decode code, text and
@8000 +0.5 to +1.0 %; prefill@8000 within ±1 % (prefill chunks split too); answers identical.
**Active tooling.** `TONA_GPU_GAPS=500` both arms; the load log; defaults otherwise (GPU pick, read-ahead level 2).

## Result (k = 2, 6 rounds, 78 min)

Full report: [2026-11-15-mac-early-commit-server-ab.md](2026-11-15-mac-early-commit-server-ab.md) (test conditions
in [2026-11-15-mac-early-commit.log](2026-11-15-mac-early-commit.log): quiet check ok before; after: the GPU still
98 % busy at 60 s, ok at 80 s).

| Metric | B vs A, mean [95 % CI] | Expected | Verdict |
|---|---|---|---|
| decode-code | +0.26 % [−0.75, +1.27] | +0.5 to +1 % | open, below the expectation |
| decode-text | +1.25 % [+0.33, +2.18] | +0.5 to +1 % | **B faster** after round 3, above the expectation |
| decode@8000 | +0.50 % [−0.27, +1.26] | +0.5 to +1 % | open, at the low edge |
| prefill@8000 | +0.90 % [−0.99, +2.80] | ±1 % | within |

Answers identical (SHA 727ff658ae742fa7 in both arms, as before the change). decode-code moves in steps of 0.8 %
(38.4 / 38.7 / 39.0 tok/s: the rate of a fixed number of tokens over whole passes), so one round flips by ±0.8 %;
rounds 4 and 5 are its two minuses. Background: a backup ran during starts 16 to 19 (arm A, round 5; CPU system
59 to 65 % against 14 to 20 % otherwise); round 5 is B's worst round in decode@8000 and code, so it did not favour B.

Gap probe, median over the 500-pass reports (A 39, B 36), ms per target pass:

| | busy | idle | inputs | encoding target | encoding other | submit | split |
|---|---|---|---|---|---|---|---|
| A | 93.56 | 3.36 | 1.28 | 0.83 | 0.16 | 0.59 | 0 |
| B | 93.63 | 2.81 | 0.54 | 0.11 | 0.15 | 0.58 | 0.15 |

Read against the plan: encoding target 0.83 → 0.11 ms as expected; busy unchanged (+0.07, within 0.05 + noise);
idle −0.56 ms, short of −0.7 to −0.8 because of the split. **Finding: the split is not under 0.05 ms** and depends
on the report: in every B server the first report (the 8000-token request and the first code requests) shows
0.39 to 0.50, the second (code) 0.14 to 0.16, the third (text) 0.04. Two layers are about 4.6 ms of GPU work
against 0.8 ms of encoding, so the second command buffer should be waiting before the first ends; a gap that
large means either the rest is encoded later than assumed (the host is slower in some passes) or the GPU starts
a second command buffer late even when it is ready. The sweep below separates the two: a host-side cause shrinks
with k, a fixed cost per extra command buffer does not.

## Plan: sweep of k (written before the run)

**Run.** Two `scripts/ab.py` runs of 2 fixed rounds (the minimum), same workload and models as above, same build,
both arms `TONA_GPU_GAPS=500`: first A `TONA_METAL_EARLY_COMMIT=1` against B `=4`, then A `=8` against B `=2`.
About 2 × 27 min. The speed tables of two rounds stay open; the probe is what is read.
**Expectation.** encoding target falls with k (about 0.15 ms at k = 1, 0.1 at 2 and 4, 0.12 at 8: begin plus
k layers of 0.02 ms each); split: if host-side, k = 1 largest (about 0.3 ms) and k = 8 near 0; if a fixed cost,
about 0.15 ms at every k. busy and SHA unchanged.
**Active tooling.** `TONA_GPU_GAPS=500` both arms, the load log, defaults otherwise.

## Result of the sweep (2 × 2 rounds, 56 min)

Reports: [k 1 vs 4](2026-11-15-mac-early-commit-k14-server-ab.md), [k 8 vs 2](2026-11-15-mac-early-commit-k82-server-ab.md)
(quiet check ok before; after: GPU busy at 60 s, ok at 80 s). Answers identical in all four arms (727ff658ae742fa7).
Gap probe, median per report position over the servers of an arm (4 or 5), ms per target pass; the three reports
of a server are the same requests in every arm (1: the 8000-token request and the first code requests, 2: code,
3: text):

| k | encoding target 1 / 2 / 3 | split 1 / 2 / 3 | idle 1 / 2 / 3 |
|---|---|---|---|
| 1 | 0.10 / 0.09 / 0.08 | 0.54 / 0.16 / 0.03 | 3.14 / 2.78 / 1.95 |
| 2 | 0.12 / 0.12 / 0.10 | 0.40 / 0.15 / 0.04 | 3.00 / 2.81 / 1.95 |
| 4 | 0.15 / 0.15 / 0.13 | 0.43 / 0.14 / 0.04 | 3.10 / 2.84 / 2.00 |
| 8 | 0.24 / 0.23 / 0.19 | 0.41 / 0.10 / 0.05 | 3.14 / 2.96 / 2.08 |

Speed (2 rounds, all open except one): k 4 vs 1: code −0.89 %, text −1.27 %, @8000 −0.85 %, prefill −0.19 %;
k 2 vs 8: code +0.13 %, text +0.37 %, @8000 0.00 %, prefill@8000 +1.86 % [+0.53, +3.19] ("B faster").

Read against the plan: encoding target grows with k as expected (about 0.02 ms per layer, k = 8 a little more).
**The split does not shrink with k**: 0.40 to 0.54, 0.10 to 0.16 and 0.03 to 0.05 at every k, so it is not the
second half's encoding arriving late (at k = 8 the first half runs about 18 ms against 0.8 ms of encoding) but a
cost of the GPU starting a second command buffer, larger in the first report's requests. k = 1 and 2 are equal
within the noise; k = 4 and 8 lose by their encoding.

**Decision:** k = 2 by default (the 6-round A/B above: no metric lower, text decided faster, idle −0.56 ms; the
coordinator's rule for this switch: on when nothing loses and idle is measurably shorter). The 6-round run is the
final A/B of this default; no further run. Open: why the second command buffer starts up to 0.5 ms late in the
first requests of a server (the probe would need the host time of the second commit to tell a late commit from a
late start).

## Plan: where the split comes from (written before the run)

**Change.** The probe also takes the host time of the second half's commit: of the split, the part before that
commit is "committed late", the rest the GPU starting a ready command buffer late; and the share of target passes
with more than 0.1 ms of split (a few long stalls or a small cost in every pass).
**Run.** `scripts/ab.py`, same workload and models, A/A of the build with the probe change (k = 2 by default in
both arms), both `TONA_GPU_GAPS=500`, 2 fixed rounds (about 27 min). Only the probe is read.
**Expectation.** split as in the sweep (about 0.4 / 0.15 / 0.04 ms per report position); committed late under
0.02 ms (the second half's encoding, 0.7 ms, is far shorter than the first half's 4.6 ms of GPU work); if the
split is a few stalls, under 20 % of the passes over 0.1 ms in the first report; if a cost in every pass, the share
follows the mean (most passes over 0.1 ms where the mean is 0.4). Speed and answers as A/A: within ±1 %, equal.
**Active tooling.** `TONA_GPU_GAPS=500` both arms, the load log, defaults otherwise.

## Result of the split probe (A/A, 2 rounds, 29 min)

Report: [2026-11-15-mac-early-commit-split-probe-server-ab.md](2026-11-15-mac-early-commit-split-probe-server-ab.md)
(quiet check ok before; after: GPU busy at 60 s, ok at 80 s; answers equal, A/A speed all open). In all 9 servers
and every report: committed late 0.00 ms, **0 % of the target passes over 0.1 ms of split** (printed rounded to
whole percent, so under 0.5 %, at most 2 of 500 passes), while the mean was 0.15 to 0.51 ms in the first report,
0.13 to 0.16 in the second, 0.03 to 0.04 in the third. Read against the plan: the second half is always committed
in time, as expected; neither expectation of the share held: the split is neither a cost in every pass (that would
put most passes over 0.1 ms) nor up to 20 % of them, but one or two stalls of tens of ms per 500 passes, plus
under 0.1 ms in each of the others. A stall that long at the start of a ready command buffer is not the cost of
an extra command buffer; candidates: the GPU's power state or a residency/page-in of mapped model pages at the
start of a command buffer (the first report has the 8000-token request with experts the earlier requests did not
touch). It would then fall into "submit" or "busy" without the split as well.

## Plan: the longest split (written before the run)

**Change.** The probe prints the share to a tenth of a percent and the longest split per report.
**Run.** As above: A/A, 2 fixed rounds, about 27 min.
**Expectation.** longest split 10 to 100 ms in the first report of a server, under 1 ms in the third; share 0.1 to
0.4 %.
**Active tooling.** `TONA_GPU_GAPS=500` both arms, the load log, defaults otherwise.

## Result of the longest split (A/A, 2 rounds, 29 min)

Report: [2026-11-15-mac-early-commit-split-max-server-ab.md](2026-11-15-mac-early-commit-split-max-server-ab.md)
(quiet check ok before; after: GPU busy at 60 s, CPU warning at 80 s, 8.4 % of threads; answers equal, speed all
open). The same in all 9 servers: first report 0.2 % of the passes (one of 500) over 0.1 ms, longest 190 to 246 ms;
second report 0.4 % (two), longest 27 to 37 ms; third none (longest 0.1, once 1.2). One pass's stall makes the whole
mean (200 ms / 500 = 0.4 ms). Within the expectation for the first report (one stall, 10 to 100 ms expected:
longer); the split of every other pass is under 0.1 ms.

## Where the stall sits: the first decode pass after a long prefill

**Plan (written before the runs, a short check).** A scratch harness starts one server (DM-T + MTP-164, ctx 32768,
`TONA_GPU_GAPS=2`: a report per two target passes) and sends the 8000-token request of ab.py's server workload (100
tokens), then three short code requests (400 tokens). The probe now also reports the longest submit of a first
command buffer after a cut (prefill, a pause), whose gap the probe does not count as decode idle. Expectation: the
stall sits in the first passes of the 8000-token request; if it is a system effect after the 45 s prefill and not
the split, k = 0 shows it as that submit. About 2 min per server; k = 0 and k = 2 alternating, twice each.
Conditions and the report lines: [2026-11-15-mac-cold-submit.log](2026-11-15-mac-cold-submit.log) (quiet before;
after: a load average warning, which lags a minute behind our own runs).

| Server | first two passes of the 8000-token request | later passes | short requests |
|---|---|---|---|
| k = 2 (first try) | split 177.5 ms | split ≤ 0.1 | split ≤ 0.1 |
| k = 0 | first command buffer waits 198.7 ms to start | — | — |
| k = 2 | first waits 43.9, then split 183.5 ms | split ≤ 0.1 | — |
| k = 0 | first waits 242.7 ms | — | — |
| k = 2 | first waits 45.2, then split 236.8 ms | split ≤ 0.1 | — |

**Finding.** After the 45 s prefill of the 8000-token prompt the GPU starts the next work 0.2 to 0.28 s late, with
or without the early commit (k = 0: 199 and 243 ms before the first command buffer; k = 2: 44 and 45 ms before the
first plus 184 and 237 ms before the ready second one, committed long before). The early commit does not cause the
stall; it moves most of it into the probe's view; whichever rate counts that pass (prefill or decode, unchecked)
carries it in both arms alike. It costs
about 0.2 s per long prompt, before the first decode token (about 0.5 % of the 8000-token request's 49 s), once per
request; the short requests show none. The earlier rule "every command buffer at most 0.5 s" came from a similar
idle after 2.2 s command buffers (`2026-11-11-mac-prefill-experts.md`); here the prefill's command buffers are short,
so the trigger is the long GPU load as a whole or the switch from prefill to decode. That is a topic of its own
(time to first token), not of the early commit.
