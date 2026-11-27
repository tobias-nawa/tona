# 36 – llama.cpp expert streaming and cache policy (PRs #31469, #31286, #31233, #29887, #30966) against our slot pool

Research for issue #95, 2026-11-02. Read: the PR pages (WebFetch summaries of the GitHub pages: description, commit
messages, discussion). **Not read:** the diffs (`/files`); the code statements below come from PR text and commit
messages, not from source. #31469's page was longer than one fetch; the author's reply to the regression report
was cut off. Numbers from PRs are the authors' claims and user reports on other hardware (NVIDIA, 12-24 GB VRAM,
PCIe 4.0). Own numbers come from `research/measurements/` (file named in each case); estimates are marked **[estimate]**.
Nothing was run.

## 1. Sources and status (as of 2026-11-02)

| PR | Status | What |
|---|---|---|
| [#29887](https://github.com/ggml-org/llama.cpp/pull/29887) | merged 2026-10-07 (ggerganov; date from the page) | `--moe-cache-mib`: LRU cache of host-resident experts in VRAM; batches up to 32 tokens; only misses are uploaded |
| [#30966](https://github.com/ggml-org/llama.cpp/pull/30966) | open, opened 2026-10-22, 2 approvals of 2 needed pending, last commit 2026-11-01 | `--moe-cache-cpu-share F`: fraction F of the misses is computed on the CPU from host experts; graph builds the expert FFN twice (GPU from slots, CPU with id -1 for GPU-computed), results summed; uploads fill the cache first, CPU share only once the cache is full; `scripts/moe-cache-tune.py` |
| [#31233](https://github.com/ggml-org/llama.cpp/pull/31233) | open, ready for review 2026-10-29, am17an approved 2026-10-30 | ClassLRU: LRFU/MQ-like, 8 classes by decayed use score (CRF, half-life 128 tokens), ~12 bytes per layer and expert, replaces SLRU (#30637) |
| [#31286](https://github.com/ggml-org/llama.cpp/pull/31286) | draft, opened ~2026-10-28 (28 commits, based on #30966) | `--moe-cache-cpu-share auto` plus CRF admission for the CPU misses; stacks #30966 and #31233 |
| [#31469](https://github.com/ggml-org/llama.cpp/pull/31469) | open, opened 2026-10-31, 57 commits, no review yet; bot flags size, multi-backend, AI-written text; author calls it RFC | `--moe-direct-io`: out-of-core streaming, VRAM slot pool per layer, io_uring `O_DIRECT` pager, RAM victim cache |

## 2. What the PRs do

### #31469 (the only one for weights larger than VRAM + RAM)

- Expert tensors stay off the GPU buffers; each MoE layer gets a pool of quantized slots in VRAM plus one zero-filled
  sentinel slot. A slot table maps expert id to slot; a non-resident id points at the sentinel (contributes zero).
- Pager: `O_DIRECT` through io_uring, 4 KiB aligned, into pinned staging buffers, then DMA. Optional packed file
  (`--moe-pack-dir`) with one contiguous blob per expert.
- RAM victim cache: evicted experts stay in host memory as an exclusive tier (no double caching with the page cache).
  Optional extra tier (`MOE_CACHE_MIB`), off by default.
- Eviction: the author traced routing and found **near-uniform routing, little temporal locality, skewed frequency**;
  so round-robin and LRU were replaced by LFU, with per-batch pinning (a miss may not evict an expert the current
  batch still needs).
- Staging split into demand and prefetch pools (prefetch cannot starve demand). Prefetch: recency predictor plus a
  learned router predictor. In-graph KICK/WAIT ops resolve top-k to slot ids on the device; the shared expert runs
  between them to overlap miss handling; decode graph splits 2 -> 1, CUDA graph.
- `--auto-warm` and a checked-in hot-expert profile seed the pool.

Results, with the PR's own contradictions (RTX 3060 12 GB, 16 GB RAM, `MemoryMax=12G`, 68 GB IQ2_XS model):

| Source | Decode tok/s | Note |
|---|---|---|
| description | stock mmap 1.47 / 1.63 (cold / warm), `--moe-direct-io --moe-slots 5000` 16.3 / 22.0 | "byte-exact in 17 configs" |
| later commit messages | 6.6-7.25 (chat, 4,992 slots), 4.3 (longer prompt) | author: the 20 tok/s figure was with boosted GPU, not reproducible cold; split-free decode (16.5) gave degenerate output; long prompts diverge from stock output (prefill MoE path differs, not fixed); live prefetch hit rate 55.7 % and 87 % (20 of 23) |
| zarbis, RTX 5080 + Ryzen 9800X3D | 20.2 -> **14.0** with the flag | GPU utilization 90 % -> 40-50 % |

Reading: a real gain only in the regime "weights do not fit in RAM, stock mmap thrashes". With enough RAM (a 9800X3D box
with a model in RAM) it is a regression. Quality of evidence is low (single author, numbers revised downward).

### #29887 + #30966 + #31233 + #31286 (weights fit in RAM, VRAM is a cache)

- #29887: the cache itself; 10 % of the expert size as a rule of thumb; Qwen3.8-Flash-Next Q4_0 (65 GiB experts):
  RTX 4090 25.0 -> 39.4 tok/s (72 % hits), 5090 30.8 -> 54.5 (77 %) and 67.8 with `-cmoe` (89 %). Third-party
  estimate: break-even near 85 % hits on PCIe 4.0 x8. Commenters asked for frequency-based admission and lagged
  asynchronous cache updates.
- #30966: CPU computes a share of the misses. Gains depend on the ratio RAM bandwidth / PCIe: sanmai (5090, DDR4) 17.3
  -> 28.8 (7.3 GiB cache) and 42-43 tok/s (20.9 GiB cache); MichaelDietzel (2080 Ti) 50.2 -> 64.5 at share 0.95;
  eapache (fast GPU, DDR4-2400) saw **no gain**, best was the largest GPU cache. The auto mode (#31286) came close to
  hand tuning. Uploads fill free slots first, CPU share starts only when the cache is full.
- #31233: ClassLRU. Own simulation: hit rate 87.6 % (ClassLRU), 86.5 % (SLRU), 93.8 % (Belady); +6/+7 % decode over
  SLRU (warm), 8 % fewer uploads; community: 16.65 -> 16.90 tok/s, hit rate 38.6 -> 42.5 %. Weak spot: at cache sizes
  1.0-1.1 x one token's expert set it loses to the ramp-down SLRU (#30848). am17an suggests moving cache management to
  the GPU later to remove the per-layer sync.

Note the baseline: these gains are over SLRU/LRU **without** prefetch. Their hit rates (38-89 %) are far from ours.

## 3. Our state (short)

- VRAM slot pool, 80-91 slots per layer (31-36 % of 256), LRU with a keep set for the current union
  (`mega_select.cpp`, `mega_select_cpu.cpp`); next-layer prediction by the router of l+1 on the x of l, top-16 per
  token, up to `IE_PREFETCH=8` copies by DMA, inserted **at the back** of the LRU list (`mega_prefetch.cpp`,
  `IE_PRE_BACK`, `IE_PRE_KEEP`). Start contents from the expert profile (`mega_experts.cpp`).
- Misses: planner per layer sends each pinned miss to the side that finishes earlier (CPU 60-80 us per expert vs GPU
  reads over PCIe), unpinned ones go to the CPU; `IE_FILL=0` default (no refill of CPU-computed misses into slots
  once the ramp-up is over; "up to 4 per layer while slots are free").
- RAM tier: all experts pinned (25.1 GB) if the budget allows (>= ~31.6 GB limit), otherwise none pinned and the page
  cache (rule "all or none", `measurements/2026-10-27-ram-limit.md`). Reference machine: 60 GiB, everything pinned.

## 4. What they do and we do not, with effect estimates

Cost basis from our logs: misses cost 7.7 ms of a 42.2 ms pass (`2026-10-24-misses`: all in VRAM 34.4-34.6 vs real
42.1-42.3). With union hits ~89-92 % this is ~0.65 ms per point of hit rate, i.e. ~1.5 % decode per point **[estimate]**.
A slot is worth +0.14 % decode (`2026-10-25-slots`: 70/80/91 slots).

### 4.1 Admission / eviction policy (CRF, ClassLRU, LFU)

- Ours measured: LRFU (decayed frequency) in simulation 87.5 / 89.3 % vs LRU 85.1 / 87.2 % **without** prefetch, but
  **no gain in the server** with prefetch (rust 76.1 / 76.2 vs 76.7 / 76.6; `2026-10-24-misses`). With prefetch at the
  back of the list, simulated LRU 91.9 / 90.1 %, LRFU 92.3-92.4 / 90.8-91.1 %, Belady-with-bypass **without**
  prefetch 92.3 / 90.5 % (`2026-10-25-cache-policy`). Conclusion there: "little left in eviction".
- Why their gain is real and ours is not: their baseline is plain SLRU/LRU with no predictor; our back-insertion of
  predicted experts already acts as an admission filter (a good half of the prefetched are never needed). LFU alone
  is worse for us (LFU causal 81.3 % vs LRU 87.2 % on rust, `2026-10-24-misses`), so #31469's switch to LFU is a result
  of its near-uniform-routing trace and of having no recency-based prefetch; it does not transfer.
- Effect of ClassLRU (8 classes, half-life 128 tokens): +0.4 to +1.0 point hits at best (our LRFU simulation range),
  i.e. **0 to +1.5 % decode, most likely within noise** (the server showed 0 for LRFU). Not measured for ClassLRU
  itself; the half-life is in tokens, ours was in passes (3-30), comparable to 128 tokens at ~3.5 tokens per pass.
- CRF admission for refills (#31286): it decides which CPU-computed misses enter the cache. We switched refilling
  off because it cost 5 % (`IE_FILL=0`: 67.9 vs 64.6 tok/s) as refills evict useful experts and use PCIe. Admission
  by score could win back a small part: bounded by the Belady-bypass gap (<= 0.4 point with prefetch) -> **<= +0.6 %**.

### 4.2 Direct-I/O path (io_uring `O_DIRECT`, packed per-expert blobs)

- We use `O_DIRECT` + io_uring only for the PLE rows (`2026-10-24-ple-io`: 78-85 ms); experts are pinned RAM or page
  cache (page faults).
- On the reference machine (60 GiB) no expert is ever read from disk after load: **effect 0**.
- Below ~31.6 GB RAM our own series shows the weak spot: nothing pinned gives 53 tok/s code / 36 text at 29 GB vs
  89 / 65 at 32 GB; partial pinning aborts (cgroup page-fault stalls, `2026-10-27-ram-limit`). An exclusive RAM tier
  filled by direct reads (no page cache double, no page-fault waits, packed 1-read-per-expert layout) is exactly the
  fix for that regime. Possible effect there: part of the 53 -> 89 gap **[estimate: +10 to +30 % on 24-29 GB machines,
  wide uncertainty]**. It would only matter for the installer class "less RAM than 32 GB"; the reference stays
  untouched if it is a separate path (rule 1).
- Evidence from #31469 is weak (numbers revised from 22 to 4-7 tok/s, regression on a fast-RAM box). The idea (not the
  code) is the interesting part. Cost: high (new pager, staging, cgroup behaviour, correctness under memory pressure).

### 4.3 Victim cache (evictions to a RAM tier)

- Ours: the pinned stage already holds every expert (25 GB) when it fits; a victim cache has nothing to hold. In the
  tight-RAM regime it is the same item as 4.2 (what the RAM tier contains). **Effect on the reference: 0.**

### 4.4 CPU-fallback decision

- #30966 fixes a share F (auto in #31286), computed from measured upload and CPU bandwidth, and starts it only after
  the cache is full. We decide per miss and layer from the finish time (CPU 60-80 us per expert vs GD 8-25 us), with
  both sides' costs; per pass 118 CPU / 65 GD at 80 us (`2026-10-25-cpu-experts`). Their finding "depends on RAM/PCIe
  bandwidth ratio" matches our planner; our planner is the finer mechanism. Free CPU experts (`IE_CPU_FAKE`) bound the
  remaining gain at -7 % pass time (39.2 -> 36.4 ms at 20 us), limited by unpacking (~5 GB/s per thread) and by RAM
  contention with GD reads, not by the decision. **Effect of adopting their decision rule: 0 to slightly negative.**
- Their "fill first, then CPU" order: ours is the opposite by measurement (`IE_FILL=0` better by 5 %).

### 4.5 Smaller items

- Demand vs prefetch staging pools: ours uses a separate DMA stream; more prefetch did not help (`IE_PREFETCH` 12: 75.1,
  16: 73.6 vs 8). Effect ~0.
- Learned router predictor for prefetch: 70 % of the LRU misses with prefetch are in the top-16 of the prediction, **30 %
  never** (`2026-10-25-cache-policy`); a learned predictor lifts early layers by up to ~25 points in the literature
  (research/14). If it recovers a third of the 30 % unreachable misses (~1 point of ~9 %): **~+1.5 %** **[estimate]**,
  but needs training data and per-layer cost on the GPU.
- Hot-expert profile at start: done (81.0 -> 82.3 % hits, `2026-10-23-cpu-experts`).
- GPU-side cache management to drop the per-layer sync (am17an): our sync is `host.wait` 23 us per layer (~1.1 ms of a
  ~42 ms pass), but the CPU needs the router result to compute its experts, so it cannot go.
- Per-batch pinning against eviction of needed experts: we have the keep set.
- Long context: expert hits/misses at 512K 65,890 / 10,202 (13.4 % misses) and at 1M 63,776 / 15,081 (19.1 %)
  (`2026-11-02-scale-run`, unit as logged there); misses rise with context. A policy change (4.1) could matter more
  there than at short context, but we have no per-policy trace at that length.

## 5. Ranking

| # | Idea | Gain (decode) | Effort | Where |
|---|---|---|---|---|
| 1 | ClassLRU / CRF admission, checked first by **simulation on our existing route traces** (no GPU) | 0 to +1.5 %, likely ~0 | low (sim), medium if it goes in | all |
| 2 | Direct-I/O exclusive RAM tier with packed expert blobs for machines under ~32 GB | +10 to +30 % on those machines, 0 on the reference | high | installer class, own backend path |
| 3 | Learned predictor for the 30 % of misses the router-of-l+1 never names | ~+1.5 % | medium-high (training) | all (already open in research/14) |

Not worth it: LFU eviction, fixed/auto CPU share, victim cache, fill-before-CPU ordering (all contradicted by our
measurements or equal to what we have).

## 6. Recommendation

- No implementation ticket now. One small `messung` ticket is justified for idea 1 as a **simulation only** on the traces
  from `IE_ROUTE_TRACE_P` (ClassLRU 8 classes with half-lives 32-512 tokens, plus CRF-gated refill, against LRU +
  back-insertion prefetch); go to the server only if the simulation shows >= +1 point over `LRU with preloading`.
- Idea 2 is a `decision item for the maintainer: it targets machines with less than 32 GB, not the reference, and #31469
  is too unstable (results revised downward, open design questions) to copy; revisit when it is reviewed or merged.
- Watch: #30966 / #31233 merge, and the answer to zarbis's regression report in #31469.

## Sources

- llama.cpp #31469, open, opened 2026-10-31: https://github.com/ggml-org/llama.cpp/pull/31469
- llama.cpp #31286, draft, ~2026-10-28: https://github.com/ggml-org/llama.cpp/pull/31286
- llama.cpp #31233, open, ready 2026-10-29: https://github.com/ggml-org/llama.cpp/pull/31233
- llama.cpp #30966, open, opened 2026-10-22: https://github.com/ggml-org/llama.cpp/pull/30966
- llama.cpp #29887, merged 2026-10-07: https://github.com/ggml-org/llama.cpp/pull/29887
- Own: `research/14-expert-prefetch.md`, `research/18-moe-prefill-streaming.md`, `research/measurements/2026-10-24-misses.md`,
  `2026-10-25-cache-policy.md`, `2026-10-25-cpu-experts.md`, `2026-10-25-slots.md`, `2026-10-27-ram-limit.md`,
  `2026-10-24-expert-prediction.md`, `2026-11-02-scale-run.md`
