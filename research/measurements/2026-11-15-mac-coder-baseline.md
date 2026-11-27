# Mac: the Coder (256 experts) with MTP against DM-T, baseline

Issue #108. The Coder (ISTA IQ1_M, all 256 experts, `toolchain/models.toml` `coder-iq1_m`) beats DM on the PC on agent
tasks; the only Coder number on the Mac is from 2026-11-11 (variant A without MTP, a different file). This file is
the baseline before the Coder path is optimized (placement of experts by usage, `docs/architecture/expert-tiers.md`
section 7); it decides nothing by itself, the optimization follows whatever it shows (maintainer decision).

## Files

| Part | File | Bytes | Note |
|---|---|---|---|
| Coder part 1 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` | 29,608,446,496 | from Hugging Face at the pinned revision, SHA-256 checked by `tools/get_model.py` |
| Coder part 2 | the PLE table, sha256 316b46… | 28,800,138,432 | the same file as DM-T's part 2 (linked); stays out of the GPU's buffers |
| MTP head | the Q8_0 head cut to 256 of its 512 experts | 2,797,873,472 | keep list [2026-11-15-mac-mtp-keep-256.json](2026-11-15-mac-mtp-keep-256.json) |
| DM-T, MTP-164 | as in the A/Bs of 2026-11-15 | 20,548,934,176, 2,316,470,592 | |

**MTP head: 256 of 512 experts by the head's own routing mass.** The masses are those of the 164 cut (145 prompts,
50,043 decode tokens, target DM; 483 experts seen): the top 256 carry 95.4 % of the mass, the 164 cut 88.5 %. The
head's experts are its own (not the Coder's), so the Coder's expert list does not apply. Memory: the head's experts
are about 5.4 MB each, so 256 cost about 1.4 GB resident (164: 0.9 GB, all 512: 2.75 GB); the embedding and output
head in the file (1.35 GB) are never read (the head takes the target's). Every MB of head is a MB less for the Coder's
experts: 0.5 GB more than DM-T's head, about 2 % of the Coder's 26.4 GB of experts.

**Memory budget (estimate).** The GPU's working set is 26.8 GB (`recommendedMaxWorkingSetSize`,
[2026-11-11-mac-bandwidth.md](2026-11-11-mac-bandwidth.md)); the Coder's part 1 alone is 29.6 GB, plus the head's
1.4 GB and the KV and scratch buffers (about 1 GB at `--ctx 32768`). So about 5 GB of what a pass may read do not stay
in memory. The Metal backend has no arena of its own: it wraps the file mapping as buffers
(`newBufferWithBytesNoCopy`), and macOS pages out clean file pages under pressure and reads them back on the next
access (LRU of the page cache). The cost of the missing 5 GB is therefore page-ins during the passes: the number to
watch is page-ins per pass (MiB, `vm_stat`) and per token.

## Plan (written before the runs)

**Question.** How fast is the Coder with MTP against DM-T + MTP-164 on the M1 Max 32 GB, same build (main 7527b3c
with the quiet check fix and the per-arm drafter of `ab.py`)? Decode and prefill are named separately.

**Runs.**
1. Correctness of the Coder path: `tona-check` 64 steps, prompt `def fib(n):\n    return`, `--ctx 1024`, against
   the CPU reference (top-1 of every step, logit deviation); the Metal path has no Coder reference yet, so this is the
   first. Risk: Metal may refuse buffers over the working set or a command buffer may fail with an out-of-memory
   error; then that is the first finding and the A/B waits.
2. Characterization (scratch harness, one server per model): the 7722-token request, then 3 code requests of 400
   tokens; per request the page-ins (`vm_stat`), and with `TONA_TOKEN_LOG` the passes and accepted tokens per pass,
   so page-ins per pass and tokens per pass of each head. About 10 min per model.
3. `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, A DM-T + MTP-164, B the Coder + MTP-256,
   `--a-model/--b-model`, `--a-drafter/--b-drafter`, same build, 2 to 6 rounds, margin 0.5 %, `--load-log on`
   (page-ins per start). 25 to 75 min.

**Expectation (wide, no measurement of this pair exists).** The Coder's experts are byte-equal to DM's (DM is a pure
cut of it), so the work per token is the same except for 1.56 times as many experts to choose from; what differs is
residency. If the page cache keeps the hot experts (the PC profile: the top 128 of 256 per layer carry 88 % of the
reads), a decode pass misses on the order of 1 to 5 % of its expert reads (`docs/architecture/expert-tiers.md`
section 8): with ~20 to 35 experts per layer and pass, 48 layers, about 10 to 80 cold experts of ~2 MB per pass, at
QD 1 page faults (3.3 to 3.7 GB/s, [2026-11-11-mac-ssd.md](2026-11-11-mac-ssd.md)) 6 to 50 ms per pass against a
pass of ~90 ms. Expected B against A: decode -10 to -50 %, prefill@8000 -5 to -40 % (a prefill chunk touches nearly
all experts of a layer, so it reads the cold ones each chunk), tokens per pass of MTP-256 equal to or above
MTP-164's 3.68 (more mass). Page-ins per pass: tens to hundreds of MiB in B, near 0 in A. A result outside these
ranges is a finding; a Coder close to DM would mean the page cache already holds the hot set.

**Active tooling.** Runs 1 and 2: `TONA_DETERMINISTIC=1` in run 1, `TONA_TOKEN_LOG` in run 2; run 3: the load log,
defaults otherwise (keepalive 500 ms, H = 1 attention).

## Run 1: the Coder does not run on today's Metal path

`tona-check` (64 steps, deterministic): the first pass fails with `kIOGPUCommandBufferCallbackErrorOutOfMemory`; the
same with `TONA_METAL_PROFILE=1` (a command buffer per part of the pass). Cause: the weights are wrapped as windows of
at most `maxBufferLength` (20.1 GB), so part 1 of the Coder becomes two buffers of 20.1 and 9.5 GB; a pass binds
both, and a command buffer's buffers must fit the working set (26.8 GB). DM-T (20.5 GB) is two windows as well but
fits in all. The plan's risk, met: the baseline needs the weights in pieces small enough that a command buffer binds
only what its part reads, and passes split into command buffers that each fit.

**Plan change (before runs 2 and 3).** Two probes, off by default, make the Coder run: `TONA_METAL_WINDOW_MB=1024`
(buffers of at most 1 GiB instead of 20.1 GB) and `TONA_METAL_SPLIT_EVERY=8` (a decode or verify pass committed as
6 command buffers of 8 layers, queued without waiting; prefill chunks are already a command buffer per part). With
them DM-T stays identical at the three references; the Coder's first 8 steps agree with the CPU in top-1 (relative
deviation 1.7e-07 to 2.5e-07); waiting on each buffer (`TONA_METAL_SPLIT_WAIT=1`) is not needed. The Coder arm (B in
run 3, the Coder run in run 2) runs with both; DM-T runs as shipped. So the baseline is the Coder on the OS page cache
with no placement of its own: the expectation above stands.

**Run 1 with the probes:** `tona-check` 64 steps (71 positions with the prompt), the Coder on Metal against the CPU
reference: top-1 equal in 71 of 71, worst relative deviation 7.78e-07. The Coder path computes correctly.

## Runs 2 and 3: the Coder thrashes the page cache

| Request | DM-T + MTP-164 | Coder + MTP-256 |
|---|---|---|
| 7722-token prompt, prefill | 53.6 to 54.9 s (141 to 144 tok/s), n=4 | 223.6 s (34.5 tok/s), n=1 |
| same request, 100 tokens decode | 3.9 s (25.5 tok/s, 1.96 per pass) | 425.5 s (0.24 tok/s, 2.61 per pass, 38 passes: 11.2 s per pass) |
| 24-token code prompt, prefill | 0.6 to 0.9 s | 51.2 s |
| code request, decode | 38.6 to 52.9 tok/s (3.9 to 6.5 per pass) | 0.49 tok/s (111 tokens, 17 passes, 6.53 per pass: 13.2 s per pass) |

Run 2 (characterization): DM-T complete; the Coder's 7722-token request did not finish within the harness's 600 s
timeout (29 passes logged). Run 3 (`ab.py`): the three warm-up and discarded starts of A ran; B's discarded start
timed out in `bench-set.py` after its first two requests (the B numbers in the table), so no round completed and the
A/B has no result. Both are far outside the plan's expectation (decode -10 to -50 %): decode is about -99 %,
prefill@8000 about -76 %. A pass of the Coder costs 11 to 13 s against about 90 ms for DM-T; at the measured fault
rate of 3.3 to 3.7 GB/s that is 37 to 48 GB read per pass, more than the whole file: the page cache keeps nearly
nothing between passes. The likely mechanism is the cyclic access of a pass (layer 0 to 47, the same order every
pass) against an LRU cache smaller than the working set, which evicts exactly the pages the next pass reads first
(the textbook worst case of LRU); unverified until the page-ins per pass are counted (run 2b).

Memory during the runs: free RAM 1.6 GB at B's first request (server log), the system compressor active (18.9 GB
compressed during DM-T's first prefill, 22.9 GB page-ins as DM-T loaded); quiet check warnings in all three checks
(other programs 13 to 16 % of the CPU threads, load average 3.5 to 5.8), see the log.

## Run 2b (plan, written before the run)

**Question.** Is the Coder's slowness the page cache re-reading the model every pass (page-ins per pass of the
order of the file, 20 to 30 GB), or something else (page-ins small, then the time goes to faults, wiring or the
split command buffers)? It decides where part 2 starts: an own resident arena (the first case) or a profile of the
pass (the second).

**Run.** The scratch harness, one Coder server with the probes and `TONA_TOKEN_LOG`, the same 32-token code request
twice; per request the `vm_stat` deltas (page-ins, compressions) and the passes. About 10 min.

**Expectation.** Page-ins 15 to 30 GB per pass in both requests (the second no faster than the first); a result
under 5 GB per pass rejects the thrashing explanation.

## Run 2b: result

| Request | Prefill | Decode | Passes | Page-ins | Per pass | Compressed / decompressed |
|---|---|---|---|---|---|---|
| 1st (cold), 22 prompt tokens, 32 tokens | 71.9 s | 56.0 s (0.6 tok/s, 6.20 per pass) | 6 (1 prefill + 5) | 257 GB | 43 GB | 237 / 236 GB |
| 2nd (prompt from the cache) | 0.2 s | 70.2 s (0.5 tok/s, 5.17 per pass) | 6 | 138 GB | 23 GB | 103 / 103 GB |

(`vm_stat` deltas over each request, 16 KiB pages; the whole machine, not only the server.)

The hypothesis holds: 23 to 43 GB of page-ins per pass, against 5 GB as the line below which it would fall. The cache
keeps nothing between passes, and the first request reads more per pass than the whole file (29.6 GB plus the head's
2.8 GB): the read-ahead of the faults brings in pages that are evicted before the pass reaches them, so the cyclic
pass pays more than one read of the file. The second request reads less per pass but is not faster; the time per
pass (9 to 12 s) matches 23 to 43 GB at 2 to 4 GB/s.

The compressor moved about as much as was paged in (237 GB in two minutes): the server's file pages are clean and
are dropped, not compressed, so what was compressed and decompressed is the anonymous memory of the rest of the
system, pushed out by the page-ins and pulled back. On the page cache alone the Coder therefore slows the whole
desktop, not only itself. Swap-ins 1.2 GB. Quiet check before: warning (other programs 12.1 % of the CPU threads);
after: ok.

**Consequence for part 2.** The fix is an arena of the server's own, resident, filled by the usage ranking, with
the cold rest read explicitly (no fault read-ahead, no page cache): the read per pass then falls from 23 to 43 GB to
the cold experts the pass routes to. A bounded arena also bounds what the server takes from the rest of the
system, which the page cache did not.

## The cold read path on macOS: prior art (before part 2 builds it)

- **F_NOCACHE.** An F_NOCACHE `pread` is direct I/O when offset and length are multiples of the device block and
  the buffer is aligned; a misaligned rest goes through the cluster layer and its pages are dumped after the copy
  (xnu `bsd/vfs/vfs_cluster.c`, `cluster_read_direct`, `cluster_read_copy`,
  https://github.com/apple-oss-distributions/xnu). It does not purge pages already cached and is set per descriptor
  right after `open` (https://developer.apple.com/forums/thread/700357). macOS has no `O_DIRECT`; `F_RDAHEAD` turns
  read-ahead off, `F_RDADVISE` is an asynchronous advisory read (`fcntl(2)`).
- **Why the page cache fails here.** A `newBufferWithBytesNoCopy` buffer over a file mapping is wired as a whole
  view when the GPU first uses it: 8 to 10 GiB wired for an 8 GiB mapping of which 2 GiB were touched, against 0.8
  GiB with `pread` into aligned slots (https://github.com/antirez/ds4/issues/638, M4 Max). The llama.cpp proposal
  for expert streaming on Metal states the same reason and keeps a slot pool in shared memory filled by a CPU
  thread with `pread`, synchronized by `MTLSharedEvent`
  (https://github.com/ggml-org/llama.cpp/discussions/23324). On macOS `madvise(MADV_DONTNEED)` does not drop file
  pages, `msync(MS_INVALIDATE)` does (https://github.com/antirez/ds4/pull/637).
- **Measured elsewhere (other machines).** flash-moe (Qwen3.5-397B, 48 GB Mac): parallel `pread` per expert;
  mmap 5 times slower, `dispatch_io` 70 % slower, F_NOCACHE +3 %, `F_RDADVISE` prefetch no gain, the GPU 73 % slower
  while SSD DMA ran beside it, its own Metal LRU removed (+38 % without) (https://github.com/danveloper/flash-moe).
  "LLM in a flash" on an M1 Max: over 6 GiB/s for a long uncached read, sparse reads need 32 KiB and more and many
  threads (https://arxiv.org/abs/2312.11514).
- **Residency.** `MTLResidencySet` (macOS 15 and later): adding one allocation of a heap makes the whole heap
  resident, residency may be deferred under memory pressure of other apps
  (https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets);
  wired memory is never swapped (https://github.com/ml-explore/mlx-lm/issues/883), so the arena's size is taken from
  the rest of the system for as long as it lives.

**What part 2 takes from it.** The hot experts in an arena of the server's own (the shared ranking and budget of
`engine/backend/expert_tiers.h`), not the mapping of the file; the cold experts read with `pread` on a second
descriptor with F_NOCACHE and read-ahead off, page-aligned, by a few threads into slots of a shared staging buffer,
each slot reused only after the GPU signals. Open on this machine, to measure before the design is fixed: the
read rate of 2 MB reads at depth 1, 4, 8 and 16; F_NOCACHE on and off; and how much slower the GPU runs while reads
are in flight (flash-moe's -73 % argues for reading between the GPU's parts, not beside them).
