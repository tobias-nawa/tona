# Tideway (arXiv 2611.14848) against our expert tiers

Read 2026-11-16: the full paper (PDF via pdftotext) and the code at github.com/Afterglow231/Tideway (main branch tarball,
unpacked in its own directory, nothing executed; read: `policy.py`, `elastic_vmm.c/.py`, `elastic_layout.py`, `shift_plan.py`,
`prefill_relocate.py`). Design compared with: `docs/architecture/expert-tiers.md`. Status in `papers.md`: read, transferable.

## What the paper is

Yifei Wang et al. (Duke, 10 authors), cs.OS, preprint under review. A Python/Triton/CuTe engine (11.8K lines of their own code) for
MoE models whose weights spill from VRAM to host RAM to SSD, on one consumer PC.

**Test setup.** RTX 5090 (32 GB, PCIe 5.0: measured 57.6 GB/s H2D, 57.3 D2H, 102 bidirectional), 32-core AMD CPU, 251 GiB RAM capped by
software to 16 to 128 GiB per model, 8 TB WD SN850X NVMe (6.8 GB/s read). Models: Qwen3.6-35B-A3B (BF16, 67 GiB), Gemma4-26B-A4B (BF16,
48 GiB), DeepSeek-V4-Flash (FP4, 155 GiB). Workloads: 270 fixed shapes (4K to 128K input, 128 to 2K output) and replays of eight
multi-turn coding-agent sessions (TraceLab, Zhu et al. 2026). Prefill chunk 8K.
**Baselines:** FreeToken v0.1.2 (extended by the authors with SSD paging), llama.cpp commit 67a17c1, and **re-implementations by the authors** of SSD-LLaMA and
FlashMoE (both closed), built on the FreeToken backend. Speedups are against the fastest baseline per setting.

**Claims.** Average 1.97x end to end (1.13x to 3.18x; 1.13 to 1.58x when everything fits in RAM), TTFT 2.40x (up to 9.04x), decode 1.61x
(1.20 to 2.28x); TraceLab up to 3.14x. Ablation (DeepSeek, 64 GiB, 32K in, 1K out): memory manager 1.23x, async delivery plus prefill pipeline
+1.36x, piecewise CUDA graphs +1.36x, together 2.28x end to end. SSD share of prefill expert traffic 52 % against 86 to 99 % for the baselines.

**Caveats.** The baselines that matter most for the SSD regime are the authors' own re-creations; the hardware (5090, PCIe 5.0 x16,
251 GiB RAM) differs from ours; the models are BF16/FP4, not 1-bit; averages are over settings the authors chose; no external memory
pressure is tested. Decode graphs (a third of the gain) concern CUDA launch overhead that our persistent kernel does not have.

## Their mechanisms

1. **Unified pool, deduplicated.** Experts live in VRAM only, RAM only or both; a residency table; promoting RAM to VRAM keeps the RAM copy
   marked *replaceable* (lazy duplicate reclamation), a few hot experts are exempt; **selective victim writeback**: a VRAM-only victim is
   demoted to RAM over the idle D2H direction instead of dropped (RAM treated as a victim cache of the GPU).
2. **Phase-aware layouts.** Prefill: part of VRAM becomes a double buffer plus activations (borrowed from the expert cache); in RAM the
   resident experts are *anchored* against eviction and a small streaming window cycles the SSD-backed experts. Decode: the borrowed VRAM goes back
   to fixed-address expert slots and RAM is an LRU cache.
3. **Streaming prefill pipeline:** layer l+1's weights are assembled (VRAM copy, PCIe, SSD) while layer l computes, double buffer with two events.
4. **Piecewise CUDA graphs** (attention prefix, router front, expert back) replayed around a host paging handler.
5. Code extras: elastic VMM (stable virtual address, pooled physical 2 MiB handles, unmap and remap **at the same address** with another handle) used
   to trade KV against expert slots inside the process; `policy.py` picks static (GPU core by an independent routing prior plus RAM rows plus
   a decode LRU of at least 16 slots, "hybrid") or an online pager.

## What we do that they do not

- Misses are **computed on the CPU** (AVX-512, planner against the GPU-direct path); they always move the weights to the GPU. At our PCIe
  link and a 1-bit model the CPU path is cheaper than moving.
- The GPU **reads pinned experts directly over PCIe**; our RAM copy is the working copy, not a cache of VRAM.
- A **global usage ranking from a shipped profile** with the 70 % rule and a measured page-cache floor; they have the "prior" variant (policy.py) but no
  measured fraction below which pinning loses.
- **Layer-major prefill over groups of chunks** (one sweep of the experts per group, not per 8K chunk): the number of sweeps that Figure 10 shows driving
  their TTFT is already lower in our design.
- **Yielding to other programs** (RAM governor, VRAM governor with never-remap rule, fake-free tests). They list sharing the machine as future work and
  keep the 32 GB card for themselves.
- A **megakernel** with no per-layer host round trip in the normal path (their decode graphs fix a launch-overhead problem we do not have).

## Verdicts

| Idea | Verdict | Why and next step |
|---|---|---|
| Deduplicated VRAM+RAM pool, lazy reclamation of RAM copies | **IMPROVE (measure first)** | Our hot experts are in a VRAM slot and in pinned RAM at the same time; the ranking key already discounts VRAM-hot experts with `q`, but only at start. First a counter: share of pinned bytes that are also in a slot at a time (the upper bound of what dedup could free is the slot share, about 30 % of the expert bytes at the 58 GB class: **estimate**, 76 slots x 98 MB = 7.5 GB against 25 GB). Pinned memory cannot be reclaimed lazily like page cache (it is pinned, freeing a piece is the governor's release step), so the cheap variant is the existing piece release in reverse score order, guided by the "also in VRAM" bit. Decision after the counter: only if the share is above about 10 % of the pinned bytes. |
| Selective victim writeback (D2H of a VRAM-only victim) | **SKIP for now** | It is only needed once experts can be VRAM-only; our slots are copies of RAM or SSD content. Becomes relevant only if the dedup above lands, and then needs a measurement of the D2H cost against our direct-read path. |
| Anchors against a prefill scan, a streaming window for cold reads | **ADOPT into step 4 (cold reads)** | Our pinned RAM cannot be flushed, but the cold tier goes through the page cache, and a layered prefill scans all cold experts once per group in layer order: the cyclic scan their Figure 8 shows for LRU. Read the cold experts of a prefill sweep with direct reads into a small streaming window (or `posix_fadvise` DONTNEED behind the read) so the page cache keeps the decode working set. Measure: page-cache hit rate of the first 128 decode steps after a long prefill, with and without. |
| Phase-aware layouts | **SKIP (have it)** | The layered prefill borrows the last layers' slot regions and returns them; the elastic generations stay outside. What is new is only the framing that decode gets the borrowed memory back as cache. |
| Streaming prefill pipeline with a double buffer | **SKIP (have it)** | Layer buffers of 2 x E experts in the loan, copy stream, events. |
| Piecewise CUDA graphs | **SKIP** | The megakernel has no launch overhead to remove. |
| Parallel SSD reads at a miss, D2H overlapped with SSD reads | **ADOPT (already step 4)** | Queue depth 8 or more, next-layer prediction. Their number: the SSD read hides behind the PCIe copy. |
| TraceLab-style replay of real coding-agent sessions as the benchmark | **ADOPT (evaluate)** | Our `short,set` and the 2K, 30K, 130K workloads do not repeat prefill, decode and a growing context. Read the TraceLab paper and check whether a trace of our own agents (`tona claude`) can be replayed through `ab.py`; the paper's Figure 9 (decode step latency after each prefill, first 128 steps) is the right metric for "does prefill disturb the working set" and costs nothing to add. |
| Same-address remap of physical handles with a pool (their `elastic_vmm.c`) | **SKIP on AMD** | CUDA allows it; on ROCm it is the stale-translation hazard of ROCm/TheRock #8744, our never-remap rule stays. Tracked in the separate ticket. |
| Trading KV against expert slots inside the process | **SKIP for now** | Our KV lives mostly in RAM; becomes interesting at 1M context when the KV and the slot share compete for the same VRAM. Note only. |

## What this changes in our plan

1. Add the **dedup counter** (share of pinned bytes also resident in a slot) to `TONA_DIAG` of the next measurement run: no engine change, one number.
2. Fold the **cold streaming window** into step 4 and measure the decode-after-prefill page-cache hit rate.
3. Read TraceLab; if its replay format fits, add it as an `ab.py` workload so tiers, the governors and the cold reads are judged on session traces.
4. Their result at 128 GiB (all fits): 1.13 to 1.58x just from removing per-expert dispatch and overlapping PCIe, which is the megakernel's home ground: no action.
