# Mac: what the second pass after a partial commit costs the server (#108)

M1 Max 32 GB, Metal backend, target DM, MTP head cut to 164 experts, build `2b1d282` (grouped experts).

## Plan (written before the run)

**Question.** After a verify pass of k drafts with a < k accepted, the Metal backend restores the recurrent state
from its backup and runs the whole pass again for the m = a + 1 committed tokens (`Runner::commit`, all mat-vecs and
experts). The reference replays only the DeltaNet recurrence of the m tokens at the start of the next pass
(`Mega::commit` sets `pending_commit_`; `k_dn_recur`). How often does this happen in the server, and what does it
cost?

**Scope.** Server as in M3 (`work/m3/m3.sh`, arm B: DM + MTP-164, ctx 16384, chunk 2048, bench-set), one round, with
`TONA_DRAFT_LOG` (diagnostic: one line per verify pass with drafts k, accepted a, forward ms; the forward time
excludes the commit). ~4 min. Rerun cost = sum over the passes with a < k of the pass cost of width a + 1
(2026-11-12-mac-moe-grouped.md), against the round's wall time.

**Expectation.** Tokens per pass 2.67 as before; passes with a < k 50 to 75 % (greedy code accepts ~0.8 per draft,
widths 2 to 4); rerun cost 20 to 35 % of the decode time. The log adds an `fflush` per pass (< 0.1 ms).

**Active tooling.** `TONA_DRAFT_LOG=work/m3/draftlog.txt`. `caffeinate -dimsu`. Quiet check before and after.

## Results

Quiet check before (09:28:06) warning, other programs 10.6 % of all CPU threads (measured anyway: a diagnostic of
counts, not a speed comparison); after (09:31:20) "GPU 96 % busy", the run's own tail. Raw:
[2026-11-12-mac-commit-replay.jsonl](2026-11-12-mac-commit-replay.jsonl); the draft log stays in `work/m3/`.

- Code 28.1 tok/s decode, text 20.4 (the run with the log; 28.4 and 20.3 without it).
- 1,578 logged verify passes (passes from the history drafter and forced closings are not logged): drafts 0 in 164,
  1 in 605, 2 in 207, 3 in 559, 7 in 43 (the probe every 32nd pass); 2.54 tokens per pass.
- **365 passes (23 %) committed fewer tokens than they verified**, below the expected 50 to 75 %: the width choice
  already keeps widths short where acceptance is poor.
- Their second passes cost ~19.4 s at the measured pass costs, **~11 % of the round's 179 s** (expected 20 to 35 %).
  The forwards themselves took 129.4 s (82.0 ms per pass).

Reading: replaying only the recurrence of the committed tokens, as the reference does, would take back most of the
~11 % (code ~28 -> ~31 tok/s at equal widths), and the width choice could go wider once a partial pass costs nothing
extra. The DeltaNet state itself is read once per pass on Metal (`dn_step` keeps the 8 rows of a simdgroup in
registers across the pass's tokens), so the extra time per token in the DeltaNet part (+11.7 ms at width 4) is the
projections' mat-vecs, not the state.
