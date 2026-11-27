# 67 Octave KV, QSA indexer cost and blind heads (#170), 2026-11-18

Research only, no code, no measurement of our own. Sources were read on 2026-11-18; every number below is
either from a cited page (self-reported by its author unless said otherwise) or from our own files, named.

## Summary

| Item | What it is | Transfers to Tona? | Why |
|---|---|---|---|
| Octave KV format (vLLM #59773, #59774, #67844) | 3-bit K/V codes, random sign flip + Walsh-Hadamard rotation, fixed Lloyd-Max codebook, one norm scale per vector; 196 to 248 B per (token, KV head) at head size 256 | **Memory yes, speed not yet** | Our int8 KV is 576 B per (token, KV head) at hd 256. Octave k3v4 is 248 B (-57 %). The decode kernel that would read it is latency-bound today (#149), so fewer bytes alone give nothing until the tile is fixed. |
| vLLM RDNA4 HIP kernels (#67844) | gfx1200/1201 cache store, paged dense and sparse attention, WMMA tiling, D=256 / R=64 only | **Prior art for layout, not for speed** | The PR itself says gfx1200 throughput is still behind its BF16 Triton baseline and no decode curve over context is given. |
| QSA indexer cost on a unified cache (llama.cpp #32247, #32519) | An idle long sequence slows every other sequence's decode because the indexer scores and the attention scans the whole shared cache | **No, we are structurally immune** | Our KV, pool and state are one copy per slot (`SeqBuf`, `mega_seq.h`); steps with per-slot state run once per active slot. |
| Blind heads (arXiv 2611.21917) | A few heads miss the needle under sparse selection; run full attention only in those heads | **Candidate for the 512K/1M retrieval gap**, needs calibration and a full-attention decode path per flagged head | Cost estimate below. |

## 1. What the sources say

### vLLM #59773 (RFC, Octave KV) and PR #59774

- Backend `ROCM_OCTAVE`, full-attention layers, hand-written HIP kernels, no fallback. Validated on gfx1100 and gfx942, not on gfx1200.
- Format: per (token, KV head) one packed slot. Random sign flip, Walsh-Hadamard rotation, 3-bit index into a fixed Lloyd-Max codebook for N(0,1) (no calibration), one scale per vector that preserves the norm.
- Head size 256 with 64 RoPE dims is the only supported shape. Variants: `octave_k3v3_compact` 196 B/slot (everything 3-bit), `octave_k3v3` 216 B (K in 4 blocks of 64, RoPE block 4-bit), `octave_k3v4` 248 B (V 4-bit).
- Quality (author, RDNA3, 7900 XTX): GSM8K about 92 against 91.8 fp16; next-token NLL at 380K +0.0015 nats (compact). For the sparse model (QSA) the same format cost +0.040 nats after position 2048 and +0.071 on short documents, because sparse attention is sharper; an fp16 window of the first 4 and last 64 tokens brought that to +0.016 and +0.009.
- Speed (author, 7900 XTX, 380K): decode 563 us per layer against 2,370 us for fp16, i.e. it wins because it reads fewer bytes from a kernel that is bandwidth-bound there.

### vLLM #67844 (RDNA4 kernels, draft)

- Native HIP kernels for cache store, dense and sparse attention on gfx1200/gfx1201; D=256, R=64 only; wave-level WMMA tiling; no spills reported.
- Capacity: 248 B per token per KV head against 1,024 (BF16) and 512 (FP8), i.e. 2.47x QSA capacity at a matched 3 GiB budget.
- Speed on gfx1201 (27B dense model, 8 concurrent conversations, histories to 20K): Octave 21.3 % slower than BF16 in aggregate output tokens/s. gfx1200 prefill kernel latency at KV=256: 179 to 282 us. **No decode-over-context curve and no indexer discussion.** Unsupported: sliding windows, sinks, soft caps, unequal QK/V dims.
- The author states throughput on gfx1200 is still behind the BF16 Triton baseline.

### llama.cpp #32247 and #32519 (indexer cost, "-37 %")

- Mechanism: QSA selection was an `[n_kv, n_tokens]` mask over the whole **unified** cache; flash attention scanned or read every cell, and the indexer gathered, scored and ranked the pools of every sequence, then masked out the other sequences.
- Numbers on the pages: idle 100K sequence, ms per token master -> PR: RTX PRO 6000 f16 9.64 -> 8.20 (-15 %), q8_0 12.56 -> 11.56 (-8 %); DGX Spark q8_0 51.2 -> 48.9. #32247 (before the split) q8_0: 12.56 -> 7.96 and 51.2 -> 32.8, which is the -37 % of the ticket title (RTX q8_0: 12.56 to 7.96 is -37 %). #32519 alone shows at most -15 %; the attention-side part (row lists instead of mask) is in the other half (#32520) and is what gives the rest. Cache-cell scan on the attention side is deferred there.
- Both PRs are open; HIP is not covered.

### arXiv 2611.21917 (blind heads)

- Sparse attention loses retrieval because a few heads are "blind": the selector under-ranks the needle, the budget is not the problem. A calibration probe (teacher-forced answer; needle attention the selector gives up, and log-probability recovered by fixing the head's selection) flags heads; flagged heads get full attention.
- MiniCPM4.1-8B at 64K multi-key retrieval 54.1 % -> 97.5 %; Qwen3-8B with SeerAttention-R: full attention in 18 of 288 KV heads takes accuracy from 29.8 % to 97.7 %. Four InfLLM-V2 checkpoints and NOSA-8B share the same three blind heads. Full attention in the flagged heads keeps 84 to 89 % of the time-to-first-token savings at 32K to 128K. Alternative: a 4-bit scorer that keeps the shipped budget. The page does not say how it applies to GQA query heads; the head counts are KV heads.

## 2. Our layout (for the comparison)

- Coder QSA: 12 layers, 2 KV heads x 256, int8 KV per position: per KV head K (256 B) + V (256 B) + f32 scales per 32 values (8 for K, 8 for V = 64 B) = **576 B per (position, KV head)**, 1,152 B per position and layer, 13.8 KB per position for 12 layers. At 262,144 this is 3.62 GB, which matches `research/measurements` (KV placement #80). At 1M: 14.5 GB, in pinned RAM. The indexer pool (block of 4 positions, f32 key of 128) is separate and grows with context.
- Selection: the indexer picks 512 blocks (2,048 positions); with IndexCache only 6 of 12 layers run it. Attention reads 2,048 positions per layer regardless of context. Only the indexer pool scan grows with context.
- Measured decode (MTP, server, BENCHMARKS 2026-11-14 scale run): 250K **66.3** tok/s, 512K **57.5**, 1M **54.8** (all decode, with the drafter, one run each). Decode pass 250K 44.3 ms, 128K 42.7 ms (IndexCache on, 2026-10-30).
- Ornith (10 full-attention layers, hd 256, 2 KV heads, 16 Q heads) and Bolt (8 full-attention layers, hd 128, 2 KV heads, 32 Q heads): `research/measurements/2026-11-15-multi-arch-f1-fattn.md`. Decode attention per layer and pass: 262,144 positions hd 256 **2.48 ms** (2.60x the 315 GB/s time), hd 128 **1.94 ms** (4.03x); at 32K 3.5x and 5.1x. Cause unchecked (four barriers per 16 cells). Ticket #149.

## 3. Comparison and what transfers

**KV bytes.** Per (position, KV head) at hd 256: ours 576 B; Octave k3v4 248 B (-57 %), k3v3 216 B (-62 %), compact 196 B (-66 %). For hd 128 (Bolt) the Octave pack for hd 128 is only planned upstream; our row is 288 B.

**Where it matters.**
1. Memory (Coder, 1M): KV 14.5 GB -> 6.2 GB with k3v4 (2 heads x 248 B x 12 layers x 1,048,576). That is RAM, not VRAM, so it frees pinned RAM for the expert tiers (`docs/architecture/expert-tiers.md`) and shortens the PCIe stream of the KV for the selected blocks only marginally (2,048 positions per layer, already small). **Not a decode-speed lever for the Coder.**
2. Speed (Ornith, Bolt, full attention): the bound is bytes read per pass. At 262,144 positions: our bound 0.96 ms/layer (hd 256), measured 2.48; with 248 B the bound is 0.41 ms. But the kernel is not on the bound: it is 2.6x above it, barrier-latency limited. **Fix the tile first (#149: 32/64 cells per barrier, double-buffered staging), then the format shrinks the bound.** Octave also adds work in the staging step (3-bit unpack, codebook lookup) where we already convert int8 to f16 into LDS; the WHT is applied once to q and once to the output, not per position.
3. Quality: our int8 with per-32 scales is close to lossless; Octave 3-bit is lossy (+0.0015 nats at 380K for a dense model, +0.016 to +0.040 for QSA after the fix), so it falls under rule 3 and "lossy levers only after maintainer approval". For the QSA Coder the sparse-sensitivity result is the relevant one.

**What does not transfer.**
- The vLLM kernels (paged, D=256/R=64, no sinks) fit our tile only as a reading reference; we stage into LDS through our own `fattn_tile.h`. No speed prior art on gfx1200 exists yet (their own statement).
- The llama.cpp unified-cache indexer tax does not exist for us: per-slot KV, pool and state, steps run once per active slot (`mega_seq.h`, `SeqExpand`). The idle-sequence effect would only appear if a later design merged slots into one cache. Check item for #149/#156-style cache designs: keep per-slot pools.

## 4. Expected decode at 256K and 1M

**Coder (QSA, measured; Octave would not change it).** 256K about **66 tok/s**, 1M about **55 tok/s** (BENCHMARKS 2026-11-14; one run each, spread unknown; 2026-11-15 R7 gave 55.5 at 1M). The slope is the pool scan of the 6 indexer layers, about 13 ns per position per pass (derived from 250K 44.3 ms and 128K 42.7 ms), against a bandwidth bound of about 2.4 ns per position (6 layers x 128 B per position / 315 GB/s). Gains would come from the indexer, not from the KV format: e.g. a 4-bit pool key (the "4-bit scorer" of the blind-heads paper) would cut the pool bytes by 4 against f32, **upper bound** 6 x 1M x 128 B = 0.8 GB -> 0.2 GB per pass, 2.4 ms -> 0.6 ms at the bound; at today's 5x off-bound efficiency the pool scan at 1M is about 13 ms of the pass, so the realistic win is up to about 5 to 9 ms per pass (about +10 to +17 % tok/s at 1M) **if** the scan stays equally efficient with the narrower key. This is an estimate from our own slope, not measured, and the selection quality of a 4-bit key is unchecked.

**Ornith / Bolt (full attention, attention part only).** The other parts of the pass are not measured on these models, so only the attention time per pass is given (10 and 8 layers):

| | 256K today | 256K with Octave k3v4, tile at 1.5x bound | 1M today (4x of 256K, linear) | 1M with Octave, tile at 1.5x bound |
|---|---|---|---|---|
| Ornith hd 256, 10 layers | 24.8 ms | 10 x 0.41 x 1.5 = **6.2 ms** | 99 ms | **24.6 ms** |
| Bolt hd 128, 8 layers | 15.5 ms | not applicable yet (hd 128 pack not published) | 62 ms | - |

Sources: 2.48 ms and 1.94 ms per layer from the F1 measurement; the Octave line applies the byte ratio 248/576 to the bandwidth bound 0.96 ms and an assumed 1.5x efficiency (the value #149's fix would have to reach; the efficiency is an assumption, not a measurement). Without the tile fix the Octave format would give at best the barrier-limited time unchanged. **Honest reading:** at 1M Ornith's attention alone costs about 99 ms per pass today; either the tile fix (-2.6x to 1.5x) or the format (-2.3x bytes) leaves it at 25 to 40 ms, and only both together reach about 25 ms. Whether Ornith runs 1M at all depends on the model's native context (YaRN is a quality lever, #77).

## 5. Blind heads for the Coder (estimate)

Our open retrieval gap is at 512K and 1M: needle hard 19/40 and 15/40 in both arms (BENCHMARKS 2026-11-15, R7). The paper says the failures sit in few heads. For us a "head" would be a (QSA layer, KV head) pair: 24 KV heads in total (12 layers x 2). Cost of full attention in n flagged heads at 1M: reading 288 B x 1M = 0.30 GB per head and pass (K+V of one KV head), 0.96 ms at the bandwidth bound per head, 2.5 ms at the measured efficiency of the full-attention kernel (hd 256: 2.6x). Three flagged heads: about **+7 ms per pass** now (about -12 to -15 % decode at 1M, from a pass of about 50 ms), **+3 ms** if #149 reaches the bound ratio. Needs: (1) calibration probe over our needle suite (`bench/needle`, teacher-forced answer) to flag heads; (2) a decode path that runs full attention in chosen heads next to the sparse path (the split-K full-attention kernel of F1 exists for Ornith, not wired into the QSA layers); (3) prefill is unchanged because its selection is per token anyway. This is exact for the flagged heads (no loss), a speed-for-retrieval trade. It is the cheapest lever aimed at the measured 1M retrieval gap and does not conflict with C1 (Coder kernel unchanged) if it lives in a separate op.

## 6. Recommendation

1. Do not adopt Octave for the Coder now: no decode gain (selected-block read is constant), lossy, and the only speed evidence on gfx1200 is "behind BF16". Revisit as a RAM-saving option at 1M (14.5 -> 6.2 GB) when expert tiers need the RAM; quality check with the QSA fp16 window needed.
2. For Ornith/Bolt: finish #149 first; then Octave hd 256 is a second step (format 576 -> 248 B), after the upstream hd 128 pack exists for Bolt.
3. Blind heads: worth a calibration run on the existing needle suite (#103/#149 neighbourhood); decide the 7 ms / 1M retrieval trade with the maintainer.
4. Indexer cost: no action for the unified-cache issue; keep per-slot pools. The 4-bit pool key idea is the only transferable indexer lever.

## Prior art

- vLLM #59773, #59774, #67844 (Octave KV, RDNA4 kernels), llama.cpp #32247, #32519, #32520, arXiv 2611.21917 (blind heads); own: `research/measurements/2026-11-15-multi-arch-f1-fattn.md`, BENCHMARKS.md rows 2026-10-29, 2026-10-30, 2026-11-14, 2026-11-15.
- Not read in full: the Octave kernels' source, the blind-heads paper body (abstract-level summary only via page extraction). The "-37 %" is the #32247 q8_0 idle-100K number on one machine, not #32519.
