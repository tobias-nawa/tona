# Mac: read-ahead of the n-gram rows of a pass (`TONA_PLE_PREFETCH`)

Issue #108. Follows [2026-11-14-mac-ple-copy.md](2026-11-14-mac-ple-copy.md#result-the-slow-starts-are-the-n-gram-rows-read-back-from-the-ssd):
slow server starts copy the n-gram rows of a decode pass in 5.4 to 5.7 ms, because each uncached page is one SSD read
at queue depth 1; fast starts need 0.1 to 0.3 ms and decode 3.7 % faster.

## Plan (written before the runs)

**Change under test.** `engine/backend/ple_prefetch.h` (shared, switch `TONA_PLE_PREFETCH`, off by default):
`write_inputs` first computes the rows of all tokens of the pass, then asks the kernel for their pages
(`madvise(MADV_WILLNEED)` on the page-rounded, merged ranges; page size from `sysconf`), then copies as before. The
copy stays the real read; refused hints are counted, never fatal. `TONA_PLE_STATS` now also prints the measured time
from one decode copy to the next (pass period; gaps over 1 s between requests left out) and the refused hints.

**Micro probe (done before this plan, scratch tool, not in the repository).** DM-T part 2 mapped, 150 passes of 35
random cold 16 KB pages per method, fresh pages each, four seeds; ms per pass (major faults per pass):

| Method | ms per pass |
|---|---|
| serial touch (today) | 3.2 to 5.1 (33) |
| `madvise(MADV_WILLNEED)` per page, then touch | 0.37 (33) |
| `fcntl(F_RDADVISE)` per page, then touch | 0.35 (33) |
| 4 / 8 threads touching | 1.05 / 0.64 (33) |
| 4 / 8 threads with `pread` | 0.99 / 0.64 (0) |

Thread variants start their threads every pass (a pool would be faster, not faster than the hint). `madvise` taken:
level with `F_RDADVISE`, needs no file descriptor or threads, and Linux reads file pages ahead on it the same way, so
the PC path (`engine/gpu/mega_serve.cpp`) can call the same function. **The major-fault count is not the wait:** after
the hint each page still counts as a major fault (33 per pass) although the copy no longer waits for it.

**Prior art.** llama.cpp (`src/llama-mmap.cpp`, master, read 2026-11-14): `POSIX_MADV_WILLNEED` over the file at load
when prefetching, `POSIX_MADV_RANDOM` on lazily loaded ranges, and `llama_prefetch()`, which sorts and merges caller
ranges and calls `madvise(MADV_WILLNEED)` per range (Windows: `PrefetchVirtualMemory`, "prefetch the merged ranges
in one call, so the reads can be issued concurrently"). Same mechanism, used there for whole tensors; here per pass for
the few pages a pass needs. Gemma 3n keeps its per-layer embeddings out of model memory on fast local storage and
feeds them to each layer as it runs ([Gemma 3n overview](https://ai.google.dev/gemma/docs/gemma-3n), "PLE caching"),
the same split of a large, sparsely read table to storage; no read-ahead described.

**Bit identity (done).** DM-T, `TONA_DETERMINISTIC=1`, with and without `TONA_PLE_PREFETCH=1`: `tona-check -n 64`
identical output (71 steps, top-1 71 of 71, 1.40e-06), widths seed 7 identical (0 of 66 rows not bit-identical),
prefill cut (`TONA_PREFILL_CHUNK=64 --prefill 5`) identical (1.24e-02, 13 of 13, 5.60e-03). Unit tests 53 of 53
(new: `test_ple_prefetch`, page rounding, duplicates, the hint on a real file mapping).

**Run.** `scripts/ab.py --workload set`, DM-T + MTP-164, A `TONA_PLE_STATS=1`, B `TONA_PLE_STATS=1
TONA_PLE_PREFETCH=1`, same build, `--fixed-rounds --max-rounds 6`, `--load-log on`; about 60 min. Per start from its
log: copy and faults per decode pass, pass period, refused hints.

**Expectation.**

- Copy per decode pass: A as in the A/A (0.1 to 5.7 ms by state); B at most 0.6 ms in every start, refused hints 0.
- Decode B against A: 0 to +4 % (code and text), depending on how many slow starts A has in the hour (in the A/A 8 of
  12 measured starts were slow at −3.7 %); with no slow starts in A, B within ±1 %.
- Pass period measured: if the 5.3 ms of copy lie fully in the pass, A's slow starts have a period 5.3 ms longer than
  its fast ones; less means part of the copy overlapped the GPU (the open point of the copy probe).

**Active tooling.** `TONA_PLE_STATS=1` in both arms, `TONA_PLE_PREFETCH=1` in B, the load log; defaults otherwise.

## Result: decode +4.2 % (code) and +5.9 % (text), the copy no longer stalls the pass

Conditions: before ok, after "do not measure" (GPU 96 % busy: the reading lags 1 to 2 min, it is the run's own
server) and 20 s later ok; log in [2026-11-14-mac-ple-prefetch.log](2026-11-14-mac-ple-prefetch.log). Report and load
table: [2026-11-14-mac-ple-prefetch-server-ab.md](2026-11-14-mac-ple-prefetch-server-ab.md). `ab.py` stopped after
round 2 of 6 because both metrics were decided.

| Metric | A median (range) | B median (range) | B vs A, mean [CI] | Expected |
|---|---|---|---|---|
| decode-code tok/s | 37.0 (36.9-37.0) | 38.5 (38.3-38.7) | **+4.20 % [+2.77, +5.62]** | +0 to +4 % |
| decode-text tok/s | 25.4 (25.2-25.5) | 26.9 (26.7-27.0) | **+5.92 % [+4.89, +6.95]** | +0 to +4 % |

Every server start (first 1000 decode passes; measured starts are 3, 5, 7 and 9):

| Start | Arm | Copy per decode pass | Major faults per pass | Pass period | Prefill copy per request | GPU ours | Page-ins |
|---|---|---|---|---|---|---|---|
| 1 | A | 5.55 ms | 35.3 | 104.3 ms | 64.6 ms | 88.0 % | 4.1 GB |
| 2 | A | 5.38 ms | 36.3 | 105.8 ms | 109.8 ms | 86.7 % | 3.7 GB |
| 3 | A | 5.63 ms | 35.2 | 104.3 ms | 84.2 ms | 88.9 % | 2.8 GB |
| 4 | B | 0.96 ms | 34.8 | 99.1 ms | 10.9 ms | 93.2 % | 1.3 GB |
| 5 | B | 1.06 ms | 34.8 | 99.3 ms | 9.5 ms | 92.9 % | 1.2 GB |
| 6 | B | 0.88 ms | 28.5 | 101.1 ms | 7.4 ms | 93.9 % | 0.7 GB |
| 7 | B | 0.15 ms | 3.4 | 101.0 ms | 0.9 ms | 94.2 % | 0.7 GB |
| 8 | A | 5.61 ms | 34.1 | 105.2 ms | 60.7 ms | 89.3 % | 1.4 GB |
| 9 | A | 4.75 ms | 30.4 | 104.0 ms | 18.5 ms | 90.0 % | 1.3 GB |

No read-ahead request was refused (0 in all B starts).

**Finding: above the expectation, the gain is real.** Every A start was in the slow state (copy 4.8 to 5.6 ms), so
this run measures the cold case; with fast starts in A the gain shrinks toward 0 (start 7 of B, warm, copies 0.15 ms
as A's fast starts did in the copy A/A). Text gains more than code (+5.9 against +4.2 %): its passes accept fewer
tokens, so the fixed copy per pass weighs more per token. With cold rows, B still copies 0.9 to 1.1 ms per decode pass
(expected ≤ 0.6 ms from the micro probe's 0.37 ms): a pass needs 35 pages read and the hint brings them in parallel,
not for free. The faults stay at 35 per pass in B, as expected: they no longer measure the wait. Prefill copies 7 to
11 ms instead of 61 to 110 ms per cold request.

**The open point of the copy probe is closed:** the measured pass period is 104 to 106 ms in A and 99 to 101 ms in
B. A pass is twice as long as the 50 ms estimated there, so 5.3 ms of copy are 5 % of a pass: the copy lies in the
pass with nothing overlapping it, and the 3.7 % loss of the slow starts is that copy.

**For the default.** The hint takes no memory beyond the page cache the copy would fill anyway (the same pages, read
earlier by microseconds), so the rule "work first" is not touched. Proposal: on by default on macOS, off on Linux until
the PC measures it.

**Decision (coordinator, 2026-11-14):** on by default on macOS, off on Linux until the PC measures it (the PC port
and its cold and warm A/B are the coordinator's).

## Stage 2: the next pass's rows while the current one runs (`TONA_PLE_PREFETCH=2`)

### Plan (written before the run)

**Change under test.** The switch gets levels: 0 off, 1 the pass's own rows just before the copy (the default above),
2 also the rows of the next pass while the current one runs. Decode: the target pass after a draft starts with the
fill's `next` and continues with the drafts; `mtp_fill(next)` asks for the rows of `next` and each `mtp_step(tok)` for
the rows of `tok` before its head pass runs, so only the last draft's rows are left for the copy. Prefill: before each
chunk the next chunk's rows are asked for, so they come from the SSD while the chunk runs on the GPU (a 2048-token
chunk takes about 10 s). Wrong guesses (drafts rejected) cost reads, never bytes: the copy stays the real read.
Hints of level 2 that the kernel refuses are counted in `TONA_PLE_STATS`. `mtp_step_multi` (several conversations)
gets no read-ahead yet.

**Bit identity (done).** DM-T, `TONA_DETERMINISTIC=1`, `TONA_PLE_PREFETCH=2` against the run without read-ahead:
`tona-check -n 64`, widths seed 7 and prefill cut identical; MTP head against the reference golden file (DM, MTP Q8_0)
241 of 256 top-1, median 5.83e-02, as with 0 and 1. Unit tests 53 of 53.

**Run.** `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, DM-T + MTP-164, A `TONA_PLE_STATS=1`
(level 1, the default), B `TONA_PLE_STATS=1 TONA_PLE_PREFETCH=2`, same build, `--fixed-rounds --max-rounds 4`,
`--load-log on`; about 70 min. The 8000-token request is 4 prefill chunks of 2048, three of them read ahead in B.

**Expectation.**

- Copy per decode pass with cold rows: A about 1 ms (level 1, above), B 0.15 to 0.4 ms (the last draft's 16 rows
  only); warm both about 0.15 ms.
- Decode B against A (code, text, decode@8000): 0 to +2 % (cold about +1 %, warm ±0).
- Prefill@8000: −1 to +3 %; the prompt is the same in every start, so its rows are mostly cached after the first
  start and the gain shows only in starts where they were dropped.

**Active tooling.** `TONA_PLE_STATS=1` in both arms, `TONA_PLE_PREFETCH=2` in B, the load log; defaults otherwise.

### Result: decode copy 1.1 → 0.26 ms, decode +0.5 to +1.2 %, prefill read-ahead without a measured effect

Conditions: before ok, after "do not measure" (GPU 97 % busy, the run's own server, the reading lags) and 20 s later
ok; log in [2026-11-14-mac-ple-prefetch.log](2026-11-14-mac-ple-prefetch.log). Report and load table:
[2026-11-14-mac-ple-prefetch2-server-ab.md](2026-11-14-mac-ple-prefetch2-server-ab.md). Answers of the 8000-token
request identical in both arms (SHA).

| Metric | A median (range) | B median (range) | B vs A, mean [CI] | Expected |
|---|---|---|---|---|
| decode@8000 tok/s | 24.6 (24.5-24.7) | 24.9 (24.7-25.2) | **+1.23 % [+0.23, +2.23]**, decided | +0 to +2 % |
| decode-code tok/s | 38.1 (38.1-38.1) | 38.5 (38.2-38.7) | +0.92 % [−0.21, +2.04], open | +0 to +2 % |
| decode-text tok/s | 26.4 (26.4-26.5) | 26.6 (25.8-27.2) | +0.47 % [−2.19, +3.13], open | +0 to +2 % |
| prefill@8000 tok/s | 170.3 (169.7-175.6) | 172.2 (166.0-173.5) | −0.29 % [−2.38, +1.79], open | −1 to +3 % |

Per server start (17, first 1000 decode passes and the first prefill passes):

| | A (9 starts) | B (8 starts) |
|---|---|---|
| copy per decode pass | 1.05 to 1.21 ms | 0.24 to 0.31 ms |
| major faults per decode pass | 33.3 to 34.4 | 33.1 to 35.2 |
| pass period (measured) | 99.9 to 102.4 ms | 100.0 to 102.3 ms |
| prefill copy per pass (1305 tokens on average) | 99 to 395 ms, median 308 | 143 to 593 ms, median 187 |
| refused hints | 0 | 0 |

**Finding: inside the expectation.** Decode: the read-ahead during drafting leaves the copy at 0.24 to 0.31 ms, as
expected (the last draft's rows), and decode gains +0.5 to +1.2 %, decided for the 8000-token request. All starts
were cold for the n-gram rows this time (A at level 1 copies 1.1 ms, its cold value). The pass period does not show the
0.85 ms (the spread of the period between starts is larger).

Prefill: the next chunk's read-ahead does not measurably shorten the copy (B's median lower, the ranges overlap, faults
equal at 11 400 to 12 100 per pass) and prefill@8000 stays open at −0.3 %. A chunk of 2048 tokens touches about 32 000
pages (512 MB); whether macOS reads that many hinted pages ahead, or drops some before the copy, is not measured.
Open, not needed for the decision on decode.

**For the default.** Level 2 is bit-identical and holds no memory beyond the page cache the copy fills anyway (rows of
rejected drafts are read without being used: at most 16 rows of 16 KB per rejected draft). Proposal: level 2 on macOS
for decode; the prefill part costs nothing measurable either way.

**Decision (coordinator, 2026-11-14):** level 2 is the default on macOS (Linux 0). The prefill part is dropped: no
measured effect, so no code (faults equal, the copy ranges of both arms overlap). Open, if the prefill copy counts
later: a chunk asks for about 32 000 single pages, as rows rarely share or neighbour a page; asking for coarser merged
ranges per chunk would be the next attempt.
