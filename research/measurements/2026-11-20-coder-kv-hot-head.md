# Coder (reference model): KV hot head by position (#175)

Date: 2026-11-20. Branch `worktree-<worktree>`, base `0abbbed`. Follows the "Coder hot head" proposal of
`2026-11-20-kv-hot-head.md` (block 4: all KV in VRAM at ctx 70000 = decode@64K +7.1 %, but prefill@64K -6.3 %, decode@2K -2.2 %).
Logs in `/var/tmp/claude/ckv/`, ab.py output in `work/ab/ckv-*`.

## Prior art

Placement by position, tiers, FlexGen, InfiniGen, vLLM offloading: see `2026-11-20-kv-hot-head.md` (same questions, same sources).
Specific to the Coder: its decode attention reads only the ~2,048 cells the indexer selects per token (blocks of the pool, see
`engine/gpu/mega_op_topk.h`), so a position split helps only to the extent that the selections fall into the hot region.
Search for "sparse attention KV offload by position, selected-block locality" (arXiv, GitHub) is recorded below at the end of the
histogram block.

## Test conditions

`python3 tools/hw_report.py` (`/var/tmp/claude/ckv/hw.txt`): Ryzen 9 9950X, RX 9060 XT gfx1200 15.9 GiB, 2.57 GiB VRAM held by the desktop, 60.5 GiB RAM
(46.9 available), swap 4.56 GiB in use, load average 5.6 (other contributors' builds in the queue). `quiet_check.py` before block 0: do not measure (CPU use by others
74 %, swap, VRAM held by others). Block 0 counts selections (a deterministic counter, not a rate), so it does not need a quiet machine; the speed
blocks do and get their own check before and after.

## Plan

### Block 0: where do the selected cells lie? (diagnostic, no speed claim)

**Question.** At decode with a 64K and a 128K context, which positions does the QSA selection read: how are the ~2,051 selected cells per token and layer spread over the
absolute position (4096-cell buckets) and over the distance to the current position (4096-cell buckets)? **Decision fixed before.** A hot region (head [0, H) or tail window)
is worth building only if a share of the selected cells inside it clearly exceeds its share of the context: capture share f(H) of the reads and a cost of H cells of VRAM.
The best possible gain is the +7.1 % of block 4 times f; the cost is the all-in-VRAM cost (-2.2 % decode@2K, -6.3 % prefill@64K for 0.8 GB at ctx 70000) scaled by H.
If no region has f above about 50 % at an H of at most 20 % of the context, stop and report: no position placement beats its slot cost.
**Run.** One `exclusive` job: `ab.py` with both arms the same diagnostic build `ckv-diag` (a temporary atomic histogram in `op_topk`, enabled by `TONA_QSA_HIST=<file>`, not committed;
patch kept as `/var/tmp/claude/ckv/qsa-hist.patch`), workload `server`, targets 64000:100,128000:100, ctx 262144, 1 round (two server starts = two histograms, a repeat).
The file gets a cumulative snapshot every 8 decode passes with the position; the segments of the 64K and the 128K request are differences of snapshots. Counted: every
selected cell of every QSA layer (main layers and MTP layer) of every decode token. **Expectation.** Cannot be derived: selected cells are content dependent; guess: recent
cells (small distance) and the first block (attention sink) are over-represented; the two starts agree within 1 %. **Active tooling:** `TONA_QSA_HIST` only.

## Results

### Block 0 result: selected cells of the Coder (job `ckv-b0`, `work/ab/ckv-b0.*`, raw snapshots `work/ab/ckv-b0-hist-a.txt`)

Two rounds, four server starts of the same diagnostic build: the histograms of all four starts are **identical** (deterministic selection; SHAs 997f891fe9d6378b at 64K and d23ee4b14bf8738f at 128K equal in all starts).
Counted: every selected cell of every QSA layer for about 75 decode tokens after a 64K and after a 128K prompt (1.50 M and 1.55 M cells; one code-corpus prompt each, so the shape is content dependent and is one sample).
Share of all selected cells (reads of the decode attention) that fall into a head [0, H) or into a tail window (the H cells before the current position):

| H (cells) | head, ctx 64K | tail window, ctx 64K | head, ctx 128K | tail window, ctx 128K |
|---|---|---|---|---|
| 4,096 | 5.8 % | 6.7 % | **24.0 %** | 4.1 % |
| 8,192 | 11.6 % | 9.2 % | 30.0 % | 4.3 % |
| 16,384 | 24.4 % | 15.6 % | 38.1 % | 5.8 % |
| 32,768 | 70.7 % | 32.6 % | 47.4 % | 10.0 % |
| 65,536 | 100 % (= whole context) | 100 % | 77.4 % | 25.8 % |

Reading. At 64K the selection is spread almost evenly over the context (head share about equal to the share of the context, 5.8 % for 6.4 % of the cells; the head of 32K = 50 % of the context takes 70.7 %).
At 128K one block of the very first cells (4,096 cells = 3.2 % of the context) takes 24 % of the reads (an attention sink), the rest is spread out; the tail (recent cells) is **not** favoured at all (4.1 % in the last 4,096 cells).
No region reaches the fixed threshold (share above 50 % at H of at most 20 % of the context: 13K cells at 64K, 26K at 128K): head 16K = 24 % at 64K, 38 % at 128K.

### Decision (fixed before: no region above 50 % at 20 % of the context = stop)

**No position placement is built for the Coder.** Arithmetic behind it, from block 4 of `2026-11-20-kv-hot-head.md` (all KV in VRAM at ctx 70000, 0.8 GB, i.e. about 11.4 KB per cell over all layers):
the gain of all-in-VRAM is +7.1 % decode at 64K, and a hot head captures at most that times its read share f. A head of 4,096 cells (47 MB, a slot cost below the noise of the 2K decode, about -0.1 %) would give +0.4 % at 64K (f = 5.8 %)
and +1.7 % at 128K (f = 24 %); 16,384 cells (187 MB) +1.7 % and +2.7 %, against a slot cost of about -0.5 % decode@2K (the -2.2 % of 0.8 GB scaled) and about -1.5 % prefill@64K (the -6.3 % scaled). Those costs hit the default
workload (2K) and the long prefill at every context, the gains only the long decode: the margin of the reference rule (no rate worse than 0.5 %) is used up by the cost of the larger heads, and the small head
gains under 0.5 % at 64K, the length the gate names. Bandwidth view: the 2,051 selected cells x 576 B x 12 layers = 14 MB per token read over PCIe at about 20 GB/s is 0.7 ms against a pass of about 14 ms (70 tok/s):
the whole possible gain is 5 to 7 % and a position split reaches a quarter of it at best.
Not searched in the literature beyond the sources of the earlier note (a hot head for *scattered* selections would be a per-block cache by selection frequency, which needs bookkeeping per token and a gather kernel; that is a different design and is not
attempted: with the selection changing at every token the hit rate of a cache of 5 % of the context is not known, **unverified**). Everything the instruction asks of the reference stays as it is: no code change, goldens untouched.

## Summary

- Selection of the Coder (12 QSA layers, 2,051 cells per token) is spread over the whole context at 64K and has a sink in the first 4,096 cells at 128K (24 % of the reads); recent cells are not favoured.
- A hot head by position cannot beat its slot cost on the Coder: stopped as planned. No code change; the diagnostic (atomic histogram in `op_topk`) is kept only as the patch `/var/tmp/claude/ckv/qsa-hist.patch`, not committed.
- Open: a selection-frequency cache (per block, not by position) would need its own design and measurement; a tiny head of 4,096 cells would give +1.7 % at 128K for no measurable cost, but needs the same seven read and write sites (decode, prefill, staging, MTP, stash) for a gain below the 0.5 % margin at 64K.
