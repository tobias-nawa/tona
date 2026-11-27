# Lever audit per architecture (#175, #120), 2026-11-20

Question of the maintainer: have we got everything out of the models and architectures, and have the papers been checked again?
This file answers the first half. The second half (paper coverage) is in `research/papers.md` (section 2026-11-20, week 47) and
`research/papers-coverage.md`.

Desk work only: no GPU run, no build. Every number is quoted from a file named next to it. **EST** means: computed here
from a measured profile, not measured. Where a lever is estimated, the reasoning is given so it can be checked.

Sources read: `README.md` ("What makes this engine different"), `BENCHMARKS.md`, `research/papers.md` (803 table rows, grouped
by technique in section 4, not row by row), `research/papers-screened.tsv`, the notes `research/11, 15, 30, 55, 56, 59, 65, 66, 67`,
and the measurement files of 2026-11-17 to 2026-11-20 for the three models (named in the tables), plus the older ones that
settle a lever (`2026-10-25-hc-matvec`, `2026-11-03-hc-splitk`, `2026-11-12-expert-kernels`, `2026-11-16-prefill-*`).

## Legend

| Code | Meaning |
|---|---|
| SHIPPED | in the tree and on by default (file or commit named) |
| OPT-IN | in the tree, off by default (maintainer decides) |
| REJECTED | measured, no gain or a loss (file, number) |
| EST | estimated, not built (size, reasoning) |
| OPEN | not yet looked at; estimate from the measured profile, marked as such |
| BE / NC | bit-exact (`TONA_DETERMINISTIC=1` tokens equal) / changes numbers (needs the quality check, maintainer) |

Reference platform: RX 9060 XT (gfx1200, 32 CUs, 315 GB/s VRAM, 93 TFLOPS f16 WMMA, 186 fp8), PCIe 5.0 x16 at 55.7 GB/s,
Ryzen 9 9950X (72 GB/s RAM read), 64 GB RAM.

## Current state of the three architectures (BENCHMARKS.md and the 2026-11-20 measurement files)

| Architecture | Decode tok/s after 2K / 16K / 64K / 128K | Tokens per pass | Prefill tok/s at 2K / 16K / 64K / 128K | Source |
|---|---|---|---|---|
| Coder (`qwen4exp`) | 70.6 / 67.8 / 74.1 / - (MTP; pass 39 ms at width 3.7) | 3.33 / 3.00 / 3.33 | 1,835 to 1,869 (2K to 16K), about 1,900 (64K), 1,780 to 1,930 (130K) | `2026-11-20-coder-decode.md`, `2026-11-20-coder-sparse-attention.md`, `2026-11-16-prefill-breakdown.md` |
| Ornith (`qwen35moe`) | 114.3 / 95.5 / 64.7 / 47.5 (MTP head, clean state) | 2.38 / 2.21 / 2.08 / 2.17 | 2,582 / 4,082 / 2,964 / 2,192 | `2026-11-20-ornith-bolt-clean-state.md` |
| Bolt (`nemotron_h`) | 40.1 / 38.6 / 36.3 / - (no drafter) | 1.00 | 1,415 / 1,578 / 1,429 / - | same |

Ceilings by bandwidth (same files): Coder pass about 21 ms against 39 ms measured at width 3.7; Ornith width-1 pass 7.0 ms
against 11.3 to 12.2 ms; Bolt width-1 pass 15.3 ms against about 24 ms. The Coder's decode profile is flat (no step family
above 15 % that is far from its roofline); Ornith's was dominated by the KV read until the hot head; Bolt's decode is
bandwidth-bound with the remaining gains in single digits each (`2026-11-19-bolt-opt1.md`, handover item 1).

---

## 1. Coder (`qwen4exp`: QSA, DeltaNet, hyper-connections, MoE with experts in RAM, MTP and DFlash2)

### 1.1 Decode (pass 39.2 ms at width 3.7: experts 35 %, dn.in 12 %, gr.* 12 %, dn.out 5.6 %, qsa.attn 5 %, head 4.3 %)

| Lever | Status | Evidence |
|---|---|---|
| Megakernel, one persistent pass, grid barriers 1 ms per pass, imbalance 7.3 to 4.4 ms | SHIPPED | README; `2026-10-25-barriers.md` |
| int8 matvecs with up to 4 token vectors; `k_wide` f16 WMMA from 6 tokens, passes of up to 8 | SHIPPED | `2026-10-25-width-8.md` |
| Expert kernel at 86 % of roofline per expert in VRAM (7.3 us per 2 MB expert) | no lever; REJECTED: three bit-exact kernel variants, none faster | `2026-11-20-coder-decode.md` block 3; `2026-11-12-expert-kernels.md` |
| Router of layer l+1 run on the input of l, 8 preloaded experts per layer | SHIPPED | README "Preloading", 60.5 to 65.5 tok/s |
| Prefetch cap 4 / 6 / 8 / 12 / 16 | REJECTED: -1.2 / -0.2 / 0 / -1.4 / -4.5 %, 8 is the optimum | `2026-11-20-coder-decode.md` blocks 4, 5 |
| Static slot reallocation between layers | REJECTED: 0.3 % less miss mass (simulation); an earlier online test gave +0.6 to +1.4 points, not worth it | same, block 3; `papers.md` (2509.02408) |
| Cache policy (LRU, LRFU, Belady gap, class LRU) | REJECTED: LRU with preloading already equals Belady without it (90 to 92 %) | `2026-10-25-cache-policy.md`, `2026-11-02-classlru.md` |
| Misses over PCIe: 129.5 experts per pass, 5.1 ms (12 % of the pass), 48 % of them in 8 layers (layer 0: 19.0, layer 1: 12.2) | the largest measured lever of the decode | `2026-11-20-coder-decode.md` block 3 |
| - layer-0 prediction from the embedding (layer 0 has no previous layer) | EST, BE (placement): at most 0.3 ms (0.7 %); 19 experts x 39 us = 0.74 ms is the ceiling | same, candidates table |
| - cache-aware routing (CECAR 2611.20119, bonus and pruning) | EST, NC: replay on our traces gives -30 % misses against LRU at 5 % of selections replaced, but demand-only (no preloading), so worth less; estimate 0.5 to 1.5 ms (1 to 3 %); beating Belady needs 17 to 44 % of selections replaced | `2026-11-18-cecar-offline-replay.md` |
| - drop low-weight misses (`IE_MISS_DROP`) | OPT-IN (off): weight 0.1 gives +1 % NLL, +3 to +9 % decode | `papers.md` (Research Square rs-11268552 row) |
| - substitute misses by a buddy or surrogate (BuddyMoE, SPICE, Importance-Driven) | EST, NC, lossy (HumanEval -2.4 to -3.9 in SPICE); not built | `papers.md` MoE section |
| - DraftExpert (trained draft experts) | REJECTED as a build: needs training; the prefetch half exists | `2026-11-20-coder-decode.md` |
| - ECOSPEC-style trimming of the verify pass (`IE_SPEC_CUT`) | REJECTED: one token is worth about 14 ms, its experts about 3 ms | `papers.md` ECOSPEC row |
| - more expert slots: +0.14 % decode per slot; 55 slots at ctx 262144; the MTP head's experts hold 708 MB = 7 slots | OPEN: no cheap source of VRAM found; 7 slots would be about +1 % (EST by the 0.14 % rule) | same; README "Value of an expert slot" |
| RAM tiers: pinned experts, usage profile, CPU experts with AVX-512, 16 threads on the 9950X | SHIPPED | README; `2026-11-17-omp-threads.md` (+1.3 to +4.3 %) |
| Hyper-connection matvecs gr.down / gr.up (4.7 ms at 45 % of roofline, 24 us per step against 11 us of bytes) | | |
| - two sub-blocks per lane in flight, BF16 row in 4 pieces | SHIPPED (pass -2.1 %) | README; `2026-10-25-hc-matvec.md` |
| - split-K over more waves (S = 2, 4) | REJECTED: pass +1.0 to +1.8 % slower at width 1, +2.9 to +4.3 % at width 4; the estimate of 8 us per step rested on an idle grid of 990 waves, the real one has 512 | `2026-11-20-coder-hc-splitk.md`; `2026-11-03-hc-splitk.md` |
| About 500 small steps without weights (about 9 ms: combine, recur, conv, sum, act, router, host.wait) | | |
| - dependencies instead of barriers (`TONA_DEPS`) | SHIPPED, -0.09 ms | README; `2026-10-25-barriers.md` |
| - pre-load of the next step's descriptor and first weight rows (`s_prefetch_data`) | REJECTED: decode -1 to -2 % | `2026-11-18-b7-step-prefetch.md` |
| - LDS-only barriers instead of `__syncthreads` (77 sites) | measured, bit-exact, +0.3 to +1.0 %, mostly inside noise | `2026-11-18-b7-ab.md` |
| - fuse element-wise neighbours (mix, moe.act, moe.sum, dn.conv/norm, qsa.prep/score/merge) | EST: about 4 us per removed step, 3 steps per layer = 0.6 ms (1.5 %); BE only where no rounding moves | `2026-11-17-b1-cycle-breakdown.md` lever 2 |
| Host side (5 ms of the 48 ms cycle is outside the pass: MTP draft 4.0 ms, tail after the last router 2.2 ms, embedding 0.56 ms, selection 0.53 ms) | | `2026-11-20-coder-decode.md` block 2 |
| - CPU selection to GPU top-k | EST, BE: at most 0.53 ms (1.2 %), only on layers without a miss; the handshake stays (Ornith measured the same split: 0.64 ms) | `2026-11-20-ornith-decode.md` block 8 |
| - overlap the first MTP draft step with the tail of the verify pass | EST, BE in principle: at most one of 3.8 draft steps = 1.06 ms (2 %); adds work to the tail | `2026-11-20-coder-decode.md` candidates table |
| - parallel argmax rows and parallel embeddings | up to 0.6 ms expected; the measurement of 2026-11-18 was inconclusive (base alone drifted 2x) | `2026-11-18-b1-host-parallel.md` |
| Dense matvecs below roofline: dn.in 74 %, dn.out 64 %, qsa.in/out 70 % (about 2 ms outside gr.*) | OPEN (estimate): lifting dn.in to Ornith's 88 % gives about 0.65 ms (1.7 %); no kernel idea beyond the loads already tried | profile table here vs `2026-11-19-ornith-opt1.md` D1 |
| Grid-type decode cost (IQ2_S, IQ3_XXS, IQ3_S), x from LDS at nb >= 2, Q2_0 down loads, per-step 10 us | EST, BE: 0.1 to 1.0 ms each; tried twice and failed on registers (192 VGPR, 98 to 141 spills) | `2026-11-16-expert-bandwidth.md` levers 2 to 7 |
| **Speculation** | | |
| - MTP head (truncated vocabulary, rank-128 scorer, adaptive depth, exact speculative sampling, width 8) | SHIPPED | README; `2026-11-12-mtp-dynamic-head.md` (+1.0 %) |
| - DFlash2 drafter (`--dflash`) | SHIPPED as an option; ties MTP (5.26 vs 5.27 tokens per pass on code) | `research/59` |
| - suffix / history drafts (threshold 8), drafter calibration online | SHIPPED | `2026-10-24-suffix-policy.md`, `2026-10-25-drafter-calibration.md` |
| - OEA batch-aware routing for several conversations | SHIPPED (only with 2 or more conversations) | `2026-10-26-oea.md` |
| - better drafter (EAGLE-3, DSpark, DNeo, PARD ...) | REJECTED by desk analysis: a drafter pays only at a step cost near the MTP head's 1.15 ms; the head's conditional acceptance is 0.85 to 0.93 | `research/59`, `research/66` |
| - self-speculation by layer skipping | REJECTED: drafts good (survival 0.75 over 8) but a step costs 4 to 6 times the break-even | `2026-11-15-self-spec-layer-skip.md` |
| - self-distilled MTP head (FastMTP) | EST, NC: 0 to +3 % (acceptance is already high); needs a multi-GPU run; decision for the maintainer | `research/59` |
| - tree verification through DeltaNet (Bole, SGLang #50601, ReplaySSM, SpecLA, KVBuffer) | REJECTED for the state: our chain verify already is ReplaySSM (snapshot bytes 0); `dn.recur` is 3.3 to 3.8 % of the pass, ceiling about 1 ms. A tree drafter itself is not built: each extra token adds about 2.2 ms of experts | `research/65`, `2026-11-18-mtp-net-win-169.md` |

### 1.2 Prefill (about 1,850 tok/s; compute-bound; FP8 dense GEMM, experts in f16 WMMA)

| Lever | Status | Evidence |
|---|---|---|
| Chunked prefill with WMMA, FP8 dense GEMM, FP8 activations converted once | SHIPPED (+3 % against f16; NC, quality checked) | README; `2026-11-10-fp8-quality.md` |
| Layer-by-layer prefill, two layer buffers, loaned slots | SHIPPED (+7 to +11 % at 13K) | README; arXiv 2510.08055, MoE-CORE, FreeToken |
| Radix top-512 for the indexer in prefill and decode | SHIPPED (k_topk 30.5 to 1.3 s at 128K) | `2026-10-29-prefill-topk.md`, `2026-10-29-mega-topk.md` |
| HC fusion, DeltaNet wave kernel, attention under 32 KB LDS | SHIPPED (+12 %, 587 vs 1,161 ms, -28 % attention) | README |
| Sparse attention kernel `k_attn_wmma2` | SHIPPED, byte-equal, +1.8 to +1.9 % at 16K to 128K | `2026-11-20-coder-sparse-attention.md` |
| Weight cache, dense unpack once per group, split rows, split conv, lever 7 | SHIPPED, +0.5 to +1.7 % each | `2026-11-16-prefill-levers-2-3.md`, `-split-rows.md` (+1.0 to +1.1 %), `-split-dn.md` (+0.2 to +0.4 %), `-lever7.md` (+0.5 to +0.6 %) |
| Dense GEMM split into pieces | REJECTED: -0.9 % | `2026-11-16-prefill-split-gemm.md` |
| MTP prompt entries by GEMM, PLE rows over io_uring, prompt cache, warm-up, checkpoints, slices | SHIPPED | README |
| Expert GEMM in fp8, unpack once per layer (`TONA_GEMMQ_FP8=1`) | OPT-IN, NC: +1.9 % at 30K, +2.3 % at 130K; the plan expected +10 to +13 %; shortfall estimated at the in-place conversion (1.5 to 1.8 %), fused conversion in the scatter and SwiGLU kernels is built; no measurable dNLL; decision with the maintainer | `2026-11-17-prefill-fp8-experts.md` |
| Int8 expert GEMM (#100) | REJECTED: 0.76 to 1.2x of the f16 kernel (the per-block scale math eats the gain) | `2026-11-02-prefill-int8.md` |
| Attention and indexer: indexer score 4.1 s + top-k 1.2 s at 5 TFLOPS, attention at 13 TFLOPS (fewer syncs, larger tiles; FP8 QK would be NC) | EST, BE for the tile and sync part: +4 to +5 % at 130K, +1 to +2 % at 30K (part taken by `k_attn_wmma2`) | `2026-11-16-prefill-breakdown.md` lever 5; StreamIndex 2605.02568 |
| Idle gaps of 200 to 500 us (2.6 s of 72 s at 130K) | OPEN, BE: up to +3.6 % at 130K, +4 % at 30K; cause not found (stream priority has no effect, splitting compute-bound kernels loses) | `2026-11-16-prefill-idle-gaps.md` |
| Do not copy loaned slots back between groups (GPU idles 129 to 348 ms at 9 boundaries) | EST, BE: +1.7 to +2.1 % at 130K | `2026-11-16-prefill-breakdown.md` lever 4 (no result file for it) |
| DeltaNet in chunk form (WY on WMMA, FlashQLA idea) | EST, NC (rounding): +1.8 to +2.3 % measured in a prototype; not in the tree | `2026-11-03-dn-chunk.md` |
| Fewer bytes for the fp32 residual and expert outputs (fused combine) | EST: +3 to +4 %; pure fusions BE, fp16 residual NC | `2026-11-16-prefill-breakdown.md` lever 9; `research/16` |
| f16 GEMM kernel itself | no lever: 48 TFLOPS is about 0.6 of the sustained ceiling | same |

### 1.3 Long context and KV

| Lever | Status | Evidence |
|---|---|---|
| KV cache int8 (K and V), in pinned RAM from 0.4 GiB; `just serve` 256K, 1M works (decode 53 tok/s) | SHIPPED | README; `2026-10-23-kv-int8.md` |
| Decode pass at 250K: indexer scoring once per pass, radix, split-K attention (QSA reads 14 MB per token from RAM) | SHIPPED (63.2 to 47.3 ms at 250K) | README; `2026-10-29-mega-topk.md` |
| IndexCache (only every second QSA layer runs its indexer) | SHIPPED, NC, calibrated on 600 questions, no loss down to 4 of 12 layers | `2026-10-30-indexcache-main.md`; arXiv 2603.12201 |
| KV staging into the borrowed slots during layered prefill | SHIPPED (+3 % at 8K, +26 % at 250K) | `2026-10-30-kv-placement.md` |
| KV hot head (first cells in VRAM) | REJECTED for the Coder: no region of the selection reaches a read share of 50 % at 20 % of the context (head 16K = 24 % at 64K); a head of 4K cells would give +0.4 % at 64K, +1.7 % at 128K (a sink block takes 24 % of the reads at 128K) | `2026-11-20-coder-kv-hot-head.md` |
| Selection-frequency cache (per block, not by position) | OPEN: hit rate unknown; upper bound is the +7 % of all-in-VRAM at 64K (0.7 ms of PCIe reads per pass), a quarter of it at best | same |
| Octave KV (vLLM #67844, #59773; three formats, RDNA4 kernels incl. QSA) | not adopted: no decode gain (the selected-block read is constant), NC; revisit as a RAM saver at 1M (14.5 to 6.2 GB) | `research/67` |
| FP8 / NVFP4 KV for the 12 QSA layers (vllm-qwen38-flash-next-kv-quant: fp8 perplexity within CI) | EST, NC: int8 already has the byte count of fp8; NVFP4 halves it, so at most 0.35 ms (1 %) of the 0.7 ms PCIe read | `papers.md` KV quantization section |
| Blind heads (2611.21917): full attention in the few heads where the selector loses the needle, or a 4-bit scorer | OPEN, NC: quality lever for retrieval at 64K to 1M (our `needle --hard` is 51 % at 64K, 39 % at 1M); calibration run on the needle suite proposed; costs about 7 ms at 1M | `research/67` |
| 4-bit indexer pool key | OPEN (estimate): the context-dependent part of the pass is about 3.7 ms at 128K (42.7 vs about 39 ms); halving the pool bytes saves at most 1 ms (2 %) at 128K, 0 at 2K; NC | README numbers |
| Sparse-attention kernel with 22 KB LDS (five blocks per WGP instead of two) | OPEN, BE: the kernel is a small share of the prefill (1.8 % taken); the RAM path is PCIe-bound | `2026-11-20-coder-sparse-attention.md` "open" |
| YaRN, doubled indexer top-k | REJECTED: no measurable retrieval gain | `2026-10-31-long-context-yarn.md`, `-topk.md` |
| KVBuffer / chunk-wise state write | REJECTED: at most 0.6 ms; the paper's own result is 1.00x at one request per second | `2026-11-20-coder-decode.md` |

### 1.4 Expert streaming, PCIe, memory planning

| Lever | Status | Evidence |
|---|---|---|
| GPU-direct reads of pinned experts (13 us per expert), planner for GPU/CPU balance, GTT pinned memory, expert tiers by profile | SHIPPED | README; `2026-11-15-expert-tiers.md` |
| Expert slots allocated last, 640 MB reserve, prefill workspace behind the slots (330 extra slots between prefills) | SHIPPED | README |
| A PCIe-read expert costs +39 us; 5.1 ms per pass | measured (see 1.1) | `2026-11-20-coder-decode.md` block 3 |
| PLE gather: random 160 B rows from a 28.8 GB table; SGLang #51065 found TLB cost dominating with 64 KiB pages | OPEN (estimate): our rows go over io_uring from the page cache (13 ms per chunk warm); in decode the embedding is 0.56 ms per cycle (1 %); huge pages would save at most half of it, BE | README; `2026-11-20-coder-decode.md` block 2 |
| Elastic VRAM governor, desktop reserve, GPU instant give-back | SHIPPED / measured | `research/58, 61`; `2026-11-18-vram-level0.md` |
| Speculation-aware prefetch (SP-MoE, SpecMoE, DraftExpert) | EST: the MTP draft has its own experts, so it cannot predict the target's; the verify width already sets the union | `research/55`: co-activation prefetch not built (needs a measured case where one layer of lead is not enough) |

---

## 2. Ornith (`qwen35moe`: Ornith 1.5 35B-A3B, gated DeltaNet 30 layers + full attention 10 layers, MoE 256 x top 8, MTP head)

Clean state: decode 114.3 / 95.5 / 64.7 / 47.5 tok/s after 2K / 16K / 64K / 128K, prefill 2,582 / 4,082 / 2,964 / 2,192
(`2026-11-20-ornith-bolt-clean-state.md`; the earlier absolute decode numbers after long prompts came from degenerate text and are
not to be reused). Pass at 2K: 17.9 ms at width 2.8, 21.7 ms at 16K (`2026-11-20-ornith-decode.md`).

### 2.1 Decode

| Lever | Status | Evidence |
|---|---|---|
| Megakernel with `Qwen35MoeProgram`, int8 matvecs, CPU experts, pinned stage, tiers | SHIPPED | README "Ornith and Bolt"; `2026-11-15-ornith-o1-o2-cpu.md` |
| Truncated MTP head (85 MB instead of 417 MB; `mtp.head` was 83 % of the 1,669 us draft step) and MTP prompt entries by GEMM | SHIPPED, tokens equal (SHA 2a53319fb7019c40) | `2026-11-19-ornith-opt1.md` L1, L2 |
| Dynamic rank-128 scorer for the head (the Coder's +1.0 %); no `SCORER` code in `engine/gpu/arch/qwen35moe` | OPEN (estimate): +0.5 to +1 % by the Coder's number; the draft is checked by the target, so the output distribution is unchanged | `2026-11-12-mtp-dynamic-head.md` |
| Suffix / history drafts | SHIPPED: 3.5 to 3.75 tokens per pass with history against 2.56 to 2.92 without on the same prompts | `2026-11-19-ornith-opt1.md` D1 |
| KV placement by position (hot head), default budget 13.5 % of free VRAM | SHIPPED: decode at ctx 262144 +52 % / +117 % / +187 % after 16K / 64K / 128K, 2K -0.75 %; stash and restore across the split bit-exact | `2026-11-20-kv-hot-head.md` |
| Per-layer KV split (superseded) | REJECTED: one layer in RAM costs a quarter of the gain (about 3.6 ms per pass at 64K per layer) | `2026-11-20-ornith-decode.md` block 4 |
| Budget rule has one data point (this card); the governor does not know the KV | OPEN, BE: robustness, not speed; a smaller card gets a smaller share | `2026-11-20-kv-hot-head.md` summary |
| Beyond the hot-head budget (128,000 cells of every layer) | EST: contexts above 128K read the tail from RAM again (18 to 24 GB/s over PCIe); extending the budget costs slots (-0.46 % decode at 2K per KV buffer at ctx 262144) | `2026-11-20-ornith-decode.md` blocks 4, 5 |
| `host.wait` 5 to 8 % (CPU round trip per router): CPU selection is 15.9 us per router = 0.64 ms per cycle (3.2 %) | EST, BE: GPU top-k saves at most 0.64 ms and only on layers without a miss; touches the shared megakernel; not built | `2026-11-20-ornith-decode.md` block 8 |
| `router` 6.5 to 7 % (256 rows, latency-bound, 105 GB/s) as an 8-bit matvec | EST, NC (changes expert choices): about 0.5 ms (2.5 %) | `2026-11-19-ornith-opt2.md` item 4 |
| About 15 small steps per layer (res.norm 2.1 %, moe.sum 1.4 %, moe.act 1.2 %, dn.conv 1.4 %): fusion or flags instead of barriers (MonoMoE) | EST: the Coder's figure of 4 us per removed step applies; 3 steps per layer = 120 steps = 0.5 ms (2.7 %); touches the barrier scheme of the shared kernel | same; `2026-11-17-b1-cycle-breakdown.md` |
| VGPR spills 40 / 54 / 70 at widths 1 / 4 / 8 | REJECTED as a lever by profile: the hot steps are at 80 to 100 % of bandwidth | `2026-11-19-ornith-opt1.md` "Open" |
| Expert profile for decode | SHIPPED; decode at 16K -0.6 % (6 of 6 rounds), 2K +1.9 % | `2026-11-19-expert-profiles.md` |
| Prefetch cap, static slot allocation | OPEN (estimate): measured only on the Coder (flat curve, 0.3 %); expect under 1 % | `2026-11-20-coder-decode.md` |
| Better drafter, self-distilled head | EST: no drafter weights for Ornith exist; the Coder analysis (`research/59`) applies in kind: 0 to +3 % | `research/59` |
| Width-1 gap: 11.3 to 12.2 ms against 7.0 ms | explained: host round trip 0.8 to 1.0 ms, moe.copy 0.3, router 0.5 over its bytes, MoE 0.8 over the bytes, 600 small steps 0.8 ms | `2026-11-19-ornith-opt1.md` D1 |

### 2.2 Prefill

| Lever | Status | Evidence |
|---|---|---|
| Wave prefill attention (16-row tile per wave, S^T in registers) | SHIPPED, bit-exact: 64K 1,297 to 2,965 tok/s | `2026-11-19-prefill-attention.md` |
| Rescale skip, permlanex16 | SHIPPED, byte-equal: +4.8 % at 64K | same, round 2 |
| Expert profile for prefill | SHIPPED: +20 % at 2K (fewer PCIe bytes in the prefill) | `2026-11-19-expert-profiles.md`; `2026-11-20-ornith-prefill.md` |
| Short-prompt layered prefill from 640 tokens | SHIPPED: +11.6 to +17.7 % at 959 tokens, +15.3 / +16.4 % at 1,933 | `2026-11-20-short-layered.md` |
| Layered prefill re-norms per chunk (was KL 13) | SHIPPED (fix) | `2026-11-20-bolt-determinism.md` |
| KV hot head in prefill | SHIPPED (prefill +0.1 to +2.0 % at 16K to 128K; 2K -3.9 % against all in RAM) | `2026-11-20-kv-hot-head.md` |
| FP8 dense GEMM (the Coder's route) on Ornith | OPT-IN / measured: +2.7 % at 16K, mean abs dNLL 0.216 against 0.123 for f16 | `2026-11-20-ornith-prefill.md` |
| DeltaNet `k_dn_wave` (sequential per token, 2,500 cycles against about 1,000 at the VALU limit) | EST, BE: up to 0.28 s of 3.6 s at 16K, realistically 0.1 s (3 %) | `2026-11-20-ornith-prefill.md` |
| `k_dn_conv_l2` and `k_qsa_prep` to the memory rate | EST, BE: 0.06 s each at 16K (1.6 % and 1.8 %) | same |
| Dense and expert GEMM to 75 % of the sustained peak (tiling, double-buffered unpack) | EST, BE: 0.2 to 0.3 s at 16K (6 to 8 %); a kernel project | same |
| Host sync per layer for the routing counts (idle gaps 3.8 % at 16K) | EST, BE: needs device-side grouping | same |
| Int8 MMQ route (llama.cpp #17156: block-scaled int8 activations, iu8 WMMA) | REJECTED: the Q4_K/Q6_K int8 expert kernel runs at 0.61 to 0.97x of the f16 kernel at every shape (the per-32 scale epilogue costs more VALU than the halved WMMA saves); the dense GEMMs share that epilogue; the earlier 8 to 11 % estimate assumed a free epilogue | `2026-11-20-ornith-int8-prefill.md` |
| Attention at 64K is 52 % of the pass at about 49 % of the WMMA peak | | `2026-11-19-prefill-attention.md` |
| - second row tile per wave at hd 128 | EST, BE: bounded at about 7 % of the hd 128 kernel (3.5 % of the 64K pass) | same, open (1) |
| - split-KV style (cells over the waves, partial softmax, merge) | EST, NC (summation order); v2 (four scores in flight) was slower | `2026-11-19-ornith-opt2.md` item 2 |
| - one chain over the full head dimension | NC, not measured | `prefill-attention` open (3) |
| - pre-converted f16 K/V | not taken: twice the global bytes, nothing found that does it | same, open (2) |
| Prompts below 640 tokens (layered copies every expert of the next layer: 15.6 GB at 500 tokens against 12.9 GB chunk by chunk) | OPEN, BE: a router-aware copy is impossible ahead of the router; a second copy stream gives +16 % of the copy part at most (47 to 54.5 GB/s measured) | `2026-11-20-ornith-prefill.md`, `2026-11-20-short-layered.md` |
| The 2K rate (2,582) is still 37 % below the 16K rate (4,082); at 2K 45 % of the window had no kernel running before the layered change | partly taken (short layered, profile); remaining copy time about 0.2 s of 0.77 s (EST) | `2026-11-20-ornith-prefill.md` Result 1 |
| Context-parallel DeltaNet warm-up (FlashQLA) | not applicable on one GPU | `papers.md` |

### 2.3 Long context, KV, memory

| Lever | Status | Evidence |
|---|---|---|
| Hot head, staging, 13.5 % rule, 1 GiB reserve | SHIPPED | see 2.1 |
| KV quantization below int8 (NVFP4, FP4, TurboQuant family, KIVI, KVTuner; more than 20 rows in `papers.md`) | OPEN, NC: Ornith has only 10 attention layers, 0.30 GB per buffer at ctx 262144; halving the KV bytes frees about 0.9 GB of the 1.78 GB hot head = 11 slots of 82 MB (EST +1.5 % at 2K by the Coder's 0.14 % per slot, unmeasured here) and halves the RAM read beyond the hot head; K is more sensitive than V (KVTuner); a Hadamard on K is the common recipe; needs the quality check | `papers.md` KV quantization |
| Sparse or hierarchical attention for the 10 full-attention layers | not applicable without a trained indexer (the model has none); lossy | `research/15, 30` |
| DMA of RAM-resident KV into the borrowed slots during decode beyond the hot head | EST, BE: same mechanism as the prefill staging; size not estimated | README "KV_STAGE" |
| Prompt cache and checkpoints for the hybrid (Tail-Replay, DASC, Sparse Prefix Caching) | OPEN: checkpoints are full states (about 460 MB at 16K on the Coder); compression is NC, replay is BE; matters for the 12 GB RAM budget only | `papers.md` KV and prefix cache section |

---

## 3. Bolt (`nemotron_h`: Nemotron 3.5 Bolt 74B-A9B, Mamba-2 + 8 attention layers + latent MoE, 496 experts top 9, NVFP4 dense, IQ2_XXS experts, no MTP in the file)

Clean state: decode 40.1 / 38.6 / 36.3 tok/s after 2K / 16K / 64K, prefill 1,415 / 1,578 / 1,429, one token per pass
(`2026-11-20-ornith-bolt-clean-state.md`). Pass about 24 ms against a 15.3 ms ceiling (`2026-11-19-bolt-opt1.md` handover).

### 3.1 Decode

| Lever | Status | Evidence |
|---|---|---|
| Squared-ReLU CPU experts, pinned expert stage (misses to the CPU instead of waiting for a slot) | SHIPPED: decode +8.1 %, cold prefill 733 / 405 to 1,335 (spread gone) | `2026-11-19-bolt-opt1.md` merge evidence |
| IQ2_XXS matvec `Raw<16>` (bytes in flight per lane) | SHIPPED: +7 % at 1 token (127 to 136 GB/s); the hypothesis was only partly right | same, M1 |
| Hardware counters of the IQ2_XXS loop | not possible: every counter except `SQ_BUSY_CYCLES` reads 0 on this GPU | same, C1 |
| IQ2_XXS matvecs overall: moe.up/down about 3.7 ms of 24 ms | EST: a further +7 % loop gain is 0.25 ms (1 %); BE | same |
| NVFP4 / Q8_0 matvecs (ssm.in 6.1 ms, ssm.out 3.0, head 1.74) | at the bandwidth bound (253 to 327 GB/s; NVFP4 98 % of Q8_0 in the same rig) | `2026-11-19-bolt-b2-b3.md` |
| Host round trip: host.wait 2.8 ms (11 %), moe.copy 1.7, router 1.5, 40 round trips per pass | OPEN: CPU experts removed part; GPU-side routing does not apply (CPU experts need the plan) | `2026-11-19-bolt-opt1.md` handover 1 |
| Expert profile for Bolt | SHIPPED: decode +4 % at 16K, -1.2 % at 2K; prefill +5 % at 2K; not yet offered by the installer (no catalog entry) | `2026-11-19-expert-profiles.md` |
| KV hot head (8 attention layers, 0.6 GB) | SHIPPED: decode +4.2 % / +21.9 % / +36.5 % after 16K / 64K / 128K, prefill +8.0 % / +14.9 % at 64K / 128K | `2026-11-20-kv-hot-head.md` (c) |
| **Speculation: none today.** Suffix history drafts drafted 0 of 399 prose passes; code decode shows 1.00 tokens per pass | OPEN (see below) | `2026-11-19-bolt-opt2.md` |
| - LiLiCorr / DFlash drafter from NVIDIA (block size 8, 6 target taps, top-8 lattice, OpenMDW licence) | EST: decode +9 to +60 % at width 4 for acceptance p of 0.5 to 0.7, up to +97 % at p 0.85 (NVIDIA's tau 4.68 at 7 drafts, not reproduced on IQ2); break-even tau 1.72 at W = 4, 2.46 at W = 8; new architecture, a ticket of its own (B8); first step is an acceptance count on the prose corpus (stop if p below 0.5) | `2026-11-19-bolt-opt2.md` item 5 |
| - suffix drafts on code | OPEN (estimate, unverified): the Coder and Ornith take history drafts at hit length 8 and gain 25 to 30 % tokens per pass on code (Ornith 2.56 to 3.5); Bolt's 1.00 tokens per pass on code suggests they do not fire, or the verify width is not enabled for Bolt; not examined. If they fire at the same rate, the verify pass at W = 4 costs 1.72x, so expect 0 to +10 % | `2026-11-19-ornith-opt1.md`; clean-state table |
| - MTP: the GGUF has no MTP head | n/a | `engine/gpu/arch/nemotron_h/program.h` |
| Verify through Mamba-2 (replay commit) | SHIPPED (widths 1 and 8 tested) | `2026-11-19-bolt-b4.md` |
| Router/latent steps (`quant.lat`, `moe.add` barriers) | part of the profile; no lever recorded | `2026-11-19-bolt-b5.md` |

### 3.2 Prefill

| Lever | Status | Evidence |
|---|---|---|
| Chunked SSD scan on an absolute grid (Mamba-2 on WMMA) | SHIPPED, NC by tolerance (mean KL 0.0013 against the sequential scan); 0.94 us per token and layer against 1.84; grid cuts bit-equal | `2026-11-19-bolt-opt1.md` S1 |
| Dense prefill attention (wave kernel) | SHIPPED: 64K 884 to 1,757 tok/s; rescale skip +1.8 % at 64K | `2026-11-19-prefill-attention.md` |
| Layered prefill fix | SHIPPED | `2026-11-20-bolt-determinism.md` |
| Fusing dequantization into the dense GEMM | REJECTED by the numbers | `2026-11-19-bolt-opt2.md` |
| FP8 dense route (`TONA_PREFILL_FP8=1`) | OPT-IN: prefill +9.5 % cold / +10.2 % warm at 16K; quality pair 4 % over the decode reference on one 64-position text (indication only); expert hits in the following decode fell to 34.8 % (cause unchecked: possible slot theft by the FP8 weight cache) | `2026-11-19-bolt-opt2.md` P2, Q1 |
| Int8 (MMQ-style) route for dense and expert GEMMs, exact on the weights (NVFP4 values are integers) | EST: dense GEMMs run at 26 of 93 TFLOPS, expert GEMMs at 12; bound about 5,000 tok/s against 1,550 warm | `2026-11-19-bolt-opt1.md` handover 3, 4; `bolt-opt2` open (3); note: the Ornith int8 kernel lost to f16 on its scale epilogue (`2026-11-20-ornith-int8-prefill.md`), so a Bolt route needs an epilogue-free design (per-tensor or per-row scales) to have a chance |
| Profile at 2K: dense f16 GEMMs of dequantized weights 34 %, IQ2_XXS expert GEMM 32 %, scan 7.2 % (half after the chunked scan) | measured | same |
| Layered plan unreachable at the default slot count (needs more than about 120 free slots) | OPEN: Bolt stays chunk-by-chunk at 2K; forcing it from one token changed nothing (-1.1 % / -0.3 %, noise) | `2026-11-20-short-layered.md` Result 6 |

### 3.3 Long context, memory

| Lever | Status | Evidence |
|---|---|---|
| Hot head, int8 KV for 8 attention layers | SHIPPED; the KV is small (decode 40.1 to 36.3 from 2K to 64K) | `2026-11-20-kv-hot-head.md` |
| Mamba state: fixed slots, replay commit; no paged state | SHIPPED | `2026-11-19-bolt-b6.md` |
| Expert spread: 496 experts, 145 slots before KV, 70 at ctx 262144, 88.7 % hits on code at 145 slots | measured; prose and code profiles exist | `2026-11-19-bolt-opt1.md` handover 2 |
| Bolt's tokens depend on the prefill plan and the cache state even with `TONA_DETERMINISTIC=1` | OPEN (block 1b of the hot-head file) | `2026-11-20-kv-hot-head.md` |

---

## 4. Techniques in `papers.md` by cluster, and where they land

`papers.md` has 803 table rows (backfills to 2025-01 included). The screening decision of every hit is in
`papers-screened.tsv`. The rows that matter are covered by the tables above; the clusters that no table row names:

| Cluster | Applies to | Verdict |
|---|---|---|
| Megakernels (ForgeMegakernel, Cohere, Mirage, AutoMegaKernel, Ada-MK, No Bubbles) | all three (shared kernel) | already ours; counters, flags and pre-load were tried (`2026-10-25-barriers`; `b7-step-prefetch` rejected) |
| Weight quantization (QTIP-class trellis, Proteus, CPTC, LazyWeights, DynaExq, NVFP4 for GDN) | none: our files are third-party IQ1_M, Q4_K_M, IQ2_XXS/NVFP4; re-quantizing needs training or a calibration set | closed; revisit only with a requant pipeline (`2026-11-11-requant.md`) |
| GDN state quantization (STEPQuant 2609.38169) | Coder, Ornith | EST, NC: the state is read and written once per pass (216 MiB, 0.72 ms floor on the Coder); a 6-bit state saves at most 0.25 ms (0.6 %) |
| Context extension (YaRN, 33 survey rows) | Coder | REJECTED (`2026-10-31-long-context-yarn`) |
| Thinking, sampling, effort levels | all | handled at the server (README sampling) |
| Prefix and prompt cache (LMCache, CacheWise, Tail-Replay, DASC, Sparse Prefix Caching) | all | checkpoints shipped; compression or replay OPEN (see 2.3) |
| Mac and Apple silicon rows | none of the three on this PC | covered by the Mac tickets |
| Training-based methods (speculators, distillation, QAT, FP4 indexers) | none on this PC | out of reach without a rented run |
| Datacenter serving (disaggregation, TP, paged KV scheduling, fairness) | none | not applicable at one GPU |

---

## 5. Ranking of the open levers

Sizes are the gain on the named rate; EST throughout unless a file says measured. Bit-exact first, then number-changing.

### Coder

| # | Lever | Class | Phase | Expected gain | Basis |
|---|---|---|---|---|---|
| 1 | Prefill attention and indexer tiles, syncs | BE | prefill at 130K / 30K | +4 to +5 % / +1 to +2 % (part taken by `k_attn_wmma2`) | `prefill-breakdown` lever 5 |
| 2 | Cause of the 200 to 500 us idle gaps; skip the copy-back of loaned slots | BE | prefill | up to +3.6 % and +1.7 to +2.1 % at 130K; additivity unknown | `prefill-idle-gaps`, lever 4 |
| 3 | Overlap of the first MTP draft step, GPU top-k selection, layer-0 expert predictor, step fusion | BE | decode | 2 %, 1.2 %, 0.7 %, 1.5 %; each small, together about 3 to 5 % | `coder-decode`, `b1-cycle-breakdown` |
| 4 | Expert GEMM in fp8 (built, `TONA_GEMMQ_FP8=1`) | NC | prefill | +1.9 % (30K), +2.3 % (130K), no measurable dNLL | `prefill-fp8-experts` |
| 5 | Cache-aware routing (CECAR-style) | NC | decode, misses | 1 to 3 % | `cecar-offline-replay` |
| 6 | DeltaNet chunk form | NC | prefill | +1.8 to +2.3 % | `dn-chunk` |
| 7 | Blind heads, 4-bit indexer key | NC | retrieval at 64K to 1M; decode at 128K | quality; at most 1 ms (2 %) | `research/67` |

### Ornith

| # | Lever | Class | Phase | Expected gain | Basis |
|---|---|---|---|---|---|
| 1 | Short prompts: second copy stream and scheduling for prompts under about 1,000 tokens | BE | prefill at 0.3 to 1K | 10 to 20 % (EST: the 2K rate is 37 % below the 16K rate; copies are about 0.2 s of 0.77 s) | `ornith-prefill`, `short-layered` |
| 2 | DeltaNet `k_dn_wave`, `k_dn_conv_l2`, `k_qsa_prep`, GEMM tiling | BE | prefill at 16K | 3 % + 1.6 % + 1.8 % + 6 to 8 % | `ornith-prefill` table |
| 3 | Rank-128 scorer for the MTP head; GPU top-k selection; fused small steps | BE | decode | +0.5 to +1 %; up to 3 %; up to 2.7 % | `mtp-dynamic-head`, `ornith-decode` block 8 |
| 4 | ~~Int8 MMQ route (experts and dense)~~ rejected after this audit: 0.61 to 0.97x of f16 in the kernel microbench | NC | prefill | none | `ornith-int8-prefill` |
| 5 | 8-bit router | NC | decode | about 2.5 % | `ornith-opt2` item 4 |
| 6 | KV below int8 for the 10 attention layers | NC | long-context decode, memory | 11 slots = about +1.5 % at 2K; less PCIe beyond the hot head | KV rows of `papers.md` |

### Bolt

| # | Lever | Class | Phase | Expected gain | Basis |
|---|---|---|---|---|---|
| 1 | Find out why suffix drafts do not fire on code (1.00 tokens per pass) and enable them at W = 4 | BE (verify is exact) | decode | 0 to +10 % (unverified; the verify pass costs 1.72x at W = 4) | `bolt-opt2`, `ornith-opt1` |
| 2 | LiLiCorr / DFlash drafter (B8) | BE (verify is exact); new architecture | decode | +9 to +60 % at p 0.5 to 0.7 (estimate; NVIDIA's acceptance not reproduced) | `bolt-opt2` item 5 |
| 3 | Int8 route exact on the weights (NVFP4 dense, IQ2_XXS expert GEMM) | NC for the activations (8-bit) | prefill | bound about 5,000 against 1,550 tok/s; no measurement of the first step | `bolt-opt1` handover 3, 4 |
| 4 | FP8 dense prefill | NC | prefill | +9.5 to +10.2 % at 16K; check the hit-rate drop first | `bolt-opt2` P2 |
| 5 | Layered plan at the default slot count | BE | prefill | unknown; no gain seen when forced | `short-layered` Result 6 |

## 6. The honest answer

Everything that was cheap and exact is taken. For every architecture the profile has been measured down to the step, and the
big blocks sit at 60 to 100 % of their roofline (Coder experts 86 %, dense matvecs 64 to 98 %, Ornith dn.in 88 % and head at the
bound, Bolt NVFP4 at 98 % of Q8_0). What is left per phase:

- **Decode** (Coder, Ornith): no single lever above 3 to 4 %. The gap to the bandwidth ceiling is spread over the host handshake
  (1 to 2 ms), about 500 to 600 small steps (6 to 9 ms) and the expert misses over PCIe (5.1 ms on the Coder). The levers that attack
  the first and the last need a change to the shared megakernel or to the numbers.
- **Prefill**: the largest open areas are Ornith's short prompts (10 to 20 %, estimated), the int8 route was measured slower after
  this audit (`2026-11-20-ornith-int8-prefill.md`) and the Coder's attention and indexer (4 to 5 % at 130K).
- **Bolt decode** is the one place with a big unexamined item: it has no speculation at all. A drafter is worth +9 to +60 % by the
  estimate of `bolt-opt2` and needs a new model architecture; the cheaper step (suffix drafts on code) is not yet looked at.
- Nothing of that size is unexamined on the Coder or Ornith. What could still surprise is the quality side: blind heads and retrieval
  at 1M (Coder), and the unexplained hit-rate drop with FP8 on Bolt.

## 7. Papers and PRs of the 2026-11-20 scan that touch these levers

From `research/papers.md`, section 2026-11-20 (arXiv 2026-11-16 to 17 complete, 11-18 to 19 preview; GitHub complete to 2026-11-19):

| Item | Lever it touches | Effect on the tables above |
|---|---|---|
| SGLang #51385 (O_DIRECT io_uring PLE row reader) | Coder 1.4, PLE gather | new OPEN row: O_DIRECT against our buffered io_uring path; BE; size unknown (cold chunk 105 ms, warm 13 ms today) |
| SGLang #51242 (QSA draft-extend rows share compressed-key reads) | Coder 1.1 and 1.3, `qsa.attn` 2.0 ms | new OPEN row: depends on the overlap of selected cells between neighbouring verify rows, which nobody measured (`coder-sparse-attention` open (2)); no size claimed, the kernel is PCIe-bound in RAM |
| vLLM #68549 (Reuse-Prioritized LRU for sparse KV offloading) | Coder 1.3, selection-frequency cache | gives the open row a concrete policy to simulate on our selection counters (`TONA_QSA_HIST`) |
| Twindex 2611.25078 | Coder 1.3 at 1M | indexer pool 1.7 GB at 1M in VRAM; compact copy plus full rows in RAM would free slots; needs a learned copy |
| FrontierReplay 2611.26020, FlashStates (vLLM #68762) | all three, prompt cache | checkpoint without cutting the prefill; small for us because the prefill is cut-invariant |
| TideCache 2611.26619 | Ornith 2.3 | same mechanism as the open row "DMA of RAM-resident KV into the borrowed slots during decode" |
| G-MTP 2611.25813 | Ornith 2.1 and Coder, MTP head | clustered vocabulary projection to compare with the rank-128 scorer |
| SGLang #51531, vLLM #68337 | GDN state precision | argues against a 16-bit DeltaNet state; supports leaving the STEPQuant row closed |
| DTS 2611.25938 | Ornith 2.2, long prefill attention | training-free block-sparse prefill, NC; Llama-3.1-8B only |
