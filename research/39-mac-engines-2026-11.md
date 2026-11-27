# 39: Mac backend, engines and state of the art, 2026-11-11 (#107)

Update of [21-backends.md](21-backends.md) (sections "Megakernel on Apple Metal", "Large MoE on 32 GB unified
memory", "Addendum: the Mac in detail", "Still open"), [14-expert-prefetch.md](14-expert-prefetch.md) and
[36-llama-cpp-expert-streaming.md](36-llama-cpp-expert-streaming.md). It updates them and does not repeat them. Agent
research (Opus), 2026-11-11; re-checked by hand: llama.cpp #25294 (open), #27742 (merged 2026-08-27), #27350 (open),
discussion #23324. [V] = page, GitHub API or source file read; [S] = search snippet only; [L] = repo file;
"knowledge, unchecked" = training knowledge. All speed numbers from other projects are the authors' or community claims
on their hardware. None of it was run by us. "Decode" and "prefill" are named for every rate.

## Key finding first: open-source engines already stream qwen4exp experts on an M1 Max 32 GB

This was not known on 2026-10-26. It changes the baseline for the large tier.

- **ds4 / DwarfStar** (github.com/antirez/ds4, last push 2026-11-08 [V]; docs/QWEN38_FLASH_NEXT.md, docs/SSD_STREAMING.md
  [V]): Metal SSD streaming for Qwen3.8. The Q2 pack uses IQ2_XXS gate/up and Q2_K down padded to 768. The cache fills on
  demand, with no popularity preload for Qwen.
  - **Miss handling within a layer:** "cache misses overlap with gate/up computation for cached experts and the shared
    expert". Once the reads finish, it computes the missing gate/up, then one down pass and the original ordered
    reduction.
  - **M1 Max 32 GiB, Q2 streamed: 8.25 to 8.44 tok/s decode (median of 3 pairs), prefill ~10 tok/s.** Q4 streamed: 6.07 to
    6.42 tok/s decode, ~7 tok/s prefill, 7.65 tok/s with MTP (2 drafts). Budget: min(87.5 % of RAM, Metal recommended
    working set) − 2 GiB.

**What this means.** For the 512-expert model with an arena of well under half the experts, the M1 Max 32 GB has reached
**about 8 tok/s decode** (ds4) and rates of the same order elsewhere, one token per step, with the per-layer event
handshake and LRU. Our estimate of 22 to 31 passes/s (21-backends, "Large") assumes things these engines do not have:
- the 256-expert Coder (25.1 GB of experts, so a larger arena share; estimate);
- QD-16 reads of packed 2.046 MB experts (13 GB/s measured [L 2026-11-11-mac-ssd]);
- verify width chosen by the expert union.

Before code, reconcile the estimate against these measurements (see "Still open").

## 1. MLX (as of 2026-11-11)

- **Releases** [V, GitHub API]: mlx **v0.32.4 (2026-10-14)**, v0.32.3 (09-29), v0.32.2 (08-25), v0.32.0 (07-07). mlx-lm
  v0.32.2 (2026-10-16).
- **Custom kernels** [V, docs 0.32.4]: `mx.fast.metal_kernel(name, input_names, output_names, source,
  ensure_row_contiguous, atomic_outputs, …)`, called with grid, threadgroup, output shapes/dtypes, template and
  `init_value`. It supports atomic outputs and simdgroup ops. `grid` goes to `dispatchThreads`. Nothing on threadgroup
  barriers across groups, persistent loops or forward progress. Release fixes: use-after-free with several dtypes (#3662,
  v0.32.0), cache collision (#3833, v0.32.1). This stays unchanged from 21-backends: no megakernel path.
- **MoE** [V, release notes]: `gather_qmm` / `gather_mm` improvements on Metal (#4567, #4572, v0.32.3), ragged-K sorted
  `gather_qmm` (#4009), new `gather_qqmm` (#3757), gated DeltaNet kernels (#4020, v0.32.3).
  - Qwen3.8-Flash-Next (`qwen4_exp`) in mlx-lm: **PR #1788 open since 2026-08-26**, last update 10-04 [V]. Its text gives
    17 to 21 tok/s decode, 8-bit, on an M3 Ultra 512 GB [V].
- **SSD/mmap expert offload: not in MLX or mlx-lm.**
  - Issue mlx-lm #1438 (feature request, open since 2026-06-27) [V].
  - PR #1588 "opt-in disk-backed expert offloading" **closed unmerged 2026-08-21**. Maintainer zcbenz: PR count beyond
    review capacity [V].
  - Numbers from that thread (mabaeyens, M5 32 GB, Qwen3.6-35B-A3B-8bit, LRU per layer, 160 tokens) [V]:

    | resident share | decode tok/s | decode hit rate |
    |---|---|---|
    | 0.30 | 8.88 | 0.836 |
    | 0.40 | 9.79 | 0.881 |
    | 0.45 | 10.17 | 0.896 |
    | 0.50 | 10.58 | 0.910 |

    - Marginal cost per miss 240 to 280 µs (cold SSD, ~1.1 MB slices), 54 µs of that for building the MLX array. Cold
      reads are 26 % of the decode token.
    - Coalescing 9 slices into 1 read: +17.6 % decode. On gpt-oss-120b (~4 MB slices) only +9.7 %, because reads there
      are already bandwidth bound.
    - Three independent prefetch attempts gave nothing (reactive fetch only).
  - `lazy=True` load plus a 30 % resident share is the only way to load a model larger than RAM. Eager load OOMs [V].
  - v0.32.1 added zero-copy CPU import of host buffers on unified memory (#3872) [V, release notes].
- **Quantization formats** [V, docs 0.32.4 `mx.quantize`]: affine (1 to 8 bit, group 32/64/128; **1-bit on Metal new in
  v0.32.4, #3161**), mxfp4 (g32), mxfp8 (g32), nvfp4 (g16). There are no codebook or lattice formats.
  - **GGUF:** "unsupported quantization formats … will automatically cast to `mx.float16`" [V, `mx.load` docs]. Native
    GGUF types are Q4_0, Q4_1 and Q8_0 (knowledge, unchecked). **IQ1_M, IQ2_*, IQ3_* and Q2_0 are not supported natively.**
    The Coder's experts would be cast to fp16 on load (~4 to 5× their file size) and do not fit.
- **Decode numbers, MoE on Max chips** [S unless noted], all one token per step:
  - Qwen3.6-35B-A3B 4-bit, M1 Max 64 GB: 54.4 / 42.3 tok/s (lilting.ch).
  - Qwen3-Coder-30B-A3B, M1 Max 64 GB, oMLX: 46 to 48 tok/s, quant not stated.
  - Qwen3-Next-80B-A3B: M1 Max ~43, M3 Max ~60 (already in 21-backends).
  - No MLX number for Qwen3.8-Flash-Next on an M1/M2/M3 Max found. MLX does not stream experts, so a 32 GB Mac cannot run
    it in MLX.

## 2. llama.cpp Metal (as of 2026-11-11, GitHub API [V])

- **Architecture:** qwen4exp merged in **#27742, 2026-08-27** [V]. MTP head for qwen4exp: #27836 open [V]. NextN hidden
  width: #32044 open (already in 21-backends). qwen35moe is upstream (knowledge, unchecked for date).
- **Fusion:**
  - #28948 merged 2026-09-19 [V]: top-k MoE routing fusion (SOFT_MAX+ARGSORT+GET_ROWS), MoE weighted reduce, RMS_NORM+SCALE,
    SSM_CONV+SiLU. M2 Ultra decode tg32: gemma4 26B-A4B Q4_0 107.7 → 115.7 (+7 %), deepseek4 MXFP4 +1 %, qwen35 0.8B +4 %.
  - #30680 (open since 2026-10-16) [V]: qwen4exp SCALE→SIGMOID→SCALE→hc_post chain as one kernel, **+1.1 % decode on M2
    Ultra, Qwen3.8-Flash-Next IQ4_NL**.
  - #31688 merged 2026-11-05 [V]: command buffer splits aligned to fusion groups. Without it, greedy output depended on the
    graph's node count, so Metal was not bit-exact across refactors. Relevant for rule 3 if we ever compare against
    llama.cpp Metal.
- **Concurrency:** #15929 (2025-09) and #19555 (merged 2026-02-13) [V] allow reordered concurrent ops, +0 to 4 % decode.
  Metal encodes the first 10 % of nodes in one command buffer and splits the rest evenly (#31688 text [V]).
- **IQ kernels on Metal:** dequant and mat-vec for every i-quant exist (knowledge, unchecked; type4x4 dequantizers
  mentioned in #27350 [V]).
  - **#27350 (open since 2026-08-19) [V]:** `mul_mv_ext`, the B=2..8 path, admits no i-quants. IQ1/IQ2/IQ3/IQ4_XS
    therefore fall back to the generic mat-vec at verify widths. PR on an M5 Pro: decode +42 to +73 % at batch 4 to 8 for
    Qwen3-1.7B UD-IQ2_XXS (mix with iq1_m). BS8/BS1 throughput 1.08× → 1.86×. Upstream MTP/spec decode on IQ files is
    therefore "a wash" on Metal.
  - #29869 merged 2026-10-05 [V]: few-row MMA kernels (2 to 16 rows) for Q4_0, Q8_0, Q5_K and a generic path for
    F32/F16/Q4_1/Q5_0/Q5_1/Q4_K/Q6_K. No IQ.
  - #30047 (open) [V]: the same for MUL_MAT_ID. On the M3 Ultra Qwen3.6-35B-A3B shapes: 1.4 to 2.4× at 32 to 128 tokens,
    0.95× at 16 with random ids, 1.33× with real ids.
  - No published IQ1_M decode number for Metal found.
- **Speed for our model family:**
  - M1 Max 64 GB, IQ4_XS (unsloth fork before merge): ~18 tok/s decode [S, lilting.ch].
- **Expert streaming on Mac:**
  - #25294 "stream MoE routed experts from disk" open since 2026-07-04 [V]: device-side slot cache, CPU id-remap op,
    `O_DIRECT` I/O pool, wave-partitioned prefill. GB10, GLM-5.2 UD-Q2_K_XL (254 GB): 1.83 / 2.20 tok/s decode at 73 / 79 %
    hits. Comment 2026-10-27 (ImoStocky, **M4 Max 36 GB**): `F_NOCACHE` added for Darwin. **Qwen3.8-Flash-Next UD-Q2_K_XL +
    MTP: ~26 tok/s decode peak at an 18 GB expert cache**, with 16 to 28 GB of swap. Qwen3.6-35B Q8_K_XL 29 to 37 tok/s.
  - #31469 (`--moe-direct-io`) closed unmerged 2026-11-10 (in 21-backends).
  - #29887 (VRAM cache of host experts, merged 2026-10-07) does nothing on unified memory.
  - Discussion #23324 (2026-05-19) [V] is the closest to our plan on Metal: a `kernel_moe_interceptor` copies the expert ids
    to shared memory, a CPU thread maps them to slots by LRU and preads misses, signals an `MTLSharedEvent`, and the GPU
    encoder waits. M3 Pro 36 GB, Qwen3-30B-A3B Q6_K decode: vanilla 38.1, 128 slots 30, 96 slots 30, 64 slots 27.5, 32
    slots 22.1, 8 slots 12 tok/s. So the handshake alone costs ~21 % at 100 % hits (38.1 → 30). No maintainer reply, no
    upstream PR.

## 3. SSD-streaming MoE engines on Apple silicon

| Engine | Machine | Model / quant | Decode tok/s | Cache | Reads | On a miss within the pass |
|---|---|---|---|---|---|---|
| ds4 (above) | M1 Max 32 GiB | Qwen3.8 Q2 (IQ2_XXS/Q2_K) | 8.25 to 8.44 | own cache, on demand | pread into owned buffers | hits and shared expert run while misses load, then missing gate/up + down |
| Flash-MoE (danveloper) [V] | M3 Max 48 GB | Qwen3.5-397B-A17B 4-bit (K=4 of 512, 6.75 MB experts) | 4.36 (2-bit 5.74, breaks JSON) | **OS page cache only** (~35 GB, ~71 % hits); own Metal LRU was 38 % slower | parallel pread, GCD | serial: router → pread K=4 (2.41 ms of a 4.28 ms layer) → expert kernels |
| Flash-MoE, Anemll fork [V] | M5 Max 128 GB | same, Q3 GGUF experts (IQ3_XXS gate/up, IQ4_XS down) | 12.9 (4-bit MLX 9.5, 2-bit 14.5) | page cache | `--cache-io-split 4` | same |
| TurboFieldfare [V] | M5 Pro 24 GB / M2 Air 8 GB | Gemma 4 26B-A4B 4-bit | 31 to 35 / 5.1 to 6.3 | 16-slot LFU per layer | bounded parallel pread | shared expert runs during the reads (cb1 / io / cb2) |
| mlx-lm PR #1588 (closed) [V] | M5 32 GB | Qwen3.6-35B-A3B 8-bit | 8.9 to 10.6 | LRU per layer | byte-range reads | synchronous fetch |
| llama.cpp #23324 [V] | M3 Pro 36 GB | Qwen3-30B-A3B Q6_K | 12 to 30 | LRU slot pool | pread | GPU waits on MTLSharedEvent |
| hypura [S] | M1 Max 32 GB | Mixtral 8x7B | 2.2 | tiered | - | - |

- **Flash-MoE's negative results** [V, README; author's claims, M3 Max]:
  - F_RDADVISE prefetch: net 0 %. "SSD DMA slowed GPU by 73 %".
  - Temporal prediction: 25 % hits, −18 %. MLP router predictor: 31 % accuracy. Speculative early routing: −38 %.
  - mmap of the expert files: 5× slower.
  - Spin-polling the GPU wait: −23 % ("CPU thermal competes with GPU").
  - MTP: break-even.
  - Claim: "SSD DMA and GPU compute share the same memory controller and cannot be profitably overlapped".

  **This contradicts ds4**, which overlaps reads
  with hit-expert compute. Our E2b shows a waiting queue costs a second queue nothing. We have **not** measured GPU
  bandwidth while the SSD reads at 13 GB/s. That measurement decides how much prefetch can hide.
- **Papers:** no 2026 Apple paper succeeding "LLM in a flash" found [S].
  - FlashMoE (arXiv 2601.17063, v2 2026-10-05) [V, abstract]: ML-based replacement mixing recency and frequency, SSD
    offload on a "user-grade desktop", +51 % hit rate over LRU/LFU, up to 2.6× over Fiddler/DAOP. Not Apple.
  - "SSD offloading … considered harmful in energy efficiency" (2508.06978) [S, title only].
  - Survey of SSD-streaming engines (insiderllm.com, 2026-08-01, updated 08-25) [V]: Colibri, ds4, flash-moe (abandoned
    2026-03-19), TurboFieldfare, hypura, rabbit, pulsar. Speeds are not comparable across engines.
- **Read sizes in use:** one read per expert in the engines above, Flash-MoE 6.75 MB per expert. mlx-lm thread: ~1.1
  MB slices are latency bound (2.5 GB/s scattered), ~4 MB slices bandwidth bound (5.8 GB/s), on an M5. All agree on one
  contiguous read per expert. This matches our finding (0.68 MB 7 GB/s against 2.046 MB 13 GB/s at QD 16 [L]).

## 4. Megakernels, forward progress, watchdog on Metal

- **Forward progress between threadgroups:** still no Apple documentation.
  - Forum thread 831017 (2026-06): 0 replies, no Apple answer [V].
  - Thread 771118: `atomic_thread_fence` across threadgroups on M1/A17. The answer points to `coherent(device)` in MSL 3.2
    §4.8 [S].
  - No megakernel or persistent-kernel LLM work on Metal found. BaseRT (arXiv 2607.19438) uses Metal 4 cooperative-tensor
    kernels on M5 [S], not persistent kernels.
  - Conclusion unchanged: no grid barriers. The per-layer event handshake (the engines above, #23324, our E1) is the established
    pattern.
- **Documented by Apple** [V]:
  - `encodeWaitForEvent(_:value:)`: waits only between passes. Says nothing about a time limit.
  - `MTLCommandBufferError.Code.timeout`: "the system interrupted and terminated the command buffer before it finished
    running".
  - **`accessRevoked`: "the system has revoked the Metal device's access because it's responsible for too many timeouts
    or hangs".** So repeated timeouts are not only a lost pass: the process can lose the GPU. A backend must never let a
    wait run into the watchdog in normal operation, and must stop retrying after one timeout.
  - No documented duration found.
- **Observed limits** (community, MLX issues on the upd.dev mirror):
  - #3302 (2026-03-24, M2 Ultra, macOS 26.3.1) [V]: a single SDPA dispatch over ~5 s was killed. "The watchdog fires
    per-command-buffer, not per-dispatch." Fixed by chunking.
  - #3830 (2026-07-10, M2 Ultra / M5 Max / M3 Max, macOS 26.5.1) [V]: reporter: "the macOS ~5 s GPU watchdog counts
    waiting as running and kills the buffer", although the GPU is idle. A GPU spin-wait kernel without a timeout
    deadlocked instead (`MLX_METAL_FAST_SYNCH=1`).
  - #2827 [S]: maintainer remark that Metal has a hard-coded timeout for the GPU waiting on the CPU.
  - **Ours [L, 2026-11-11-mac-event]:** a 2 s event wait completes, an 8 s wait gives `kIOGPUCommandBufferCallbackErrorTimeout`.
    Consistent with ~5 s. Not narrowed down.

## 5. Predicting experts before the pass

### Our data [L]

| Measurement | Value | File |
|---|---|---|
| Router l+1 on int8 x of layer l, true top-10 hit with 10 / 16 / 24 / 32 candidates | 65.4 / 79.6 / 87.8 / 91.9 % | measurements/2026-10-24-expert-prediction.md |
| Union misses inside the top-24 of that prediction | 85.5 % | same |
| Prefetch 8 per layer by DMA: VRAM hits | 74.9 → 88.5 %, decode 60.5 → 65.4 to 65.7 tok/s (+8 %); 49 % of prefetched used | same |
| LRU without / with prefetch, 90 slots (35 %), two server traces | 84.1 / 79.8 → 91.9 / 90.1 %; Belady without prefetch 92.3 / 90.5 | 2026-10-25-cache-policy.md |
| LRU with prefetch at 70 / 110 slots (27 / 43 %) | 86.4 / 85.1 and 94.9 / 93.5 %; ClassLRU +0.1 to +1.8 points | 2026-11-02-cache-admission-sim.md |
| LRU misses never in any token's top-16 prediction | 30 % | 2026-10-25-cache-policy.md |
| Previous tokens: union of b consecutive tokens per layer, b = 1 / 2 / 4 / 8 | 10.0 / 15.7 / 25.0 / 39.5 experts | 2026-10-22-expert-routing.md |
| LRU cold start at 97 of 256 (38 %), CPU reference, 512 tokens | 86.5 % (static ideal 88.3 %) | same |
| Two conversations, 4 tokens each, against one conversation with 8 | 46.7 against 42.3 experts per layer | 2026-10-26-overlap.md |

Reading: "router of l+1 on x of l" is good enough **with overbooking** (16 to 24 candidates). The previous token's set
carries most of the reuse already captured by LRU. No MTP-based prediction is measured.

### Literature, next layer from layer n−1 (numbers as reported)

| Source | Model | Method | Number |
|---|---|---|---|
| ProMoE 2410.22134 v3 [V, HTML] | Qwen2-57B-A14B (64 of 8), DeepSeek-MoE 16B, DS-V2-Lite | previous layer's input into the next gate ("skip-based") | **66.9 %** accuracy on Qwen2-57B; drops in head and tail layers |
| same | same | learned 2-layer MLP (~2M params per layer, ~200 µs on CPU) | **84.7 %** average; one layer further: ~5 points lower |
| same | same | token-based, many layers ahead | 58.3 % |
| Fate 2502.12224 v2 [V, abstract] | not named on the page | cross-layer gate input | "expert hit rate reaches 99 %"; decode up to 4.1× vs load-on-demand. Secondary source (research/14): 78.79 % without training, 97.15 % with more candidates |
| Speculating Experts 2603.19289 [L 14] | Qwen3-30B-A3B | quasi-hidden state | ~90 % recall@k from layer 3 |
| Pre-Attention prediction 2511.10676 [L 14] | Qwen3-30B-A3B | predictor before attention | 94.69 % exact, 98.81 % overbooked |
| SeqMoE 2609.12978 [L 14] | not named | sequence forecast | 96.97 % hits at 45 % residency |
| mlx-lm #1438 thread [V] | Qwen3.6-35B-A3B | **Jaccard of selected sets** of layer l and l+1 (not a predictor) | 0.017, uniform 0.016: the sets are uncorrelated. Not evidence against router-of-l+1, which uses the next router's weights |

### From the previous token (temporal)

- Flash-MoE [V]: temporal prediction 25 % hits (Qwen3.5-397B, K=4 of 512). An MLP router predictor reached 31 %. Both lost
  speed.
- mlx-lm #1438 [V], Qwen3.6-35B-A3B, LRU at 30 % residency: decode hit rate 0.795 (agentic) / 0.831 (code) / 0.871 (prose).
  Belady 0.902 / 0.917 / 0.940. Adjacent-token overlap 10.8× uniform. Pinning a hot set: −6 to −30 points.
- ProMoE token-based 58.3 % [V]. ExpertFlow 2410.17954 reports that predicting from the previous layer's routing (Lina)
  "performs poorly" [S].
- 2412.00099 (TMLR 2025): temporal locality "weak" in four MoEs [V, abstract; no numbers on the page].

### From a speculative / MTP draft

- No paper found that uses an MTP head as an expert predictor and reports hit rates [S].
- DraftExpert 2607.24434: learned draft expert per layer, prefetch hits 73 → 88 % with router-agreement loss; 84 to 87 %
  acceptance [S, in research/14].
- SP-MoE 2510.10302: ~88 % with target gates on draft states (Mixtral/Phi) [L 14].
- MoE-SpeQ 2511.14102: up to 2.34× over offloading frameworks [S].
- AcceptMoE 2608.02989: 2.06× on an RTX 5090, −73.6 % H2D traffic vs EAGLE-3 [S].
- Flash-MoE: MTP break-even [V].

## 6. Computing missed experts on the Apple CPU

- **No Apple-silicon engine found that computes routed-expert misses on the CPU in a streaming path.** ds4,
  Flash-MoE and TurboFieldfare compute every expert on the GPU and use the CPU only for routing, I/O and planning [V].
  - Flash-MoE runs the GatedDeltaNet recurrence on the CPU via Accelerate [V].
  - llama.cpp `--cpu-moe` / `--n-cpu-moe` exist. On a Mac they give CPU-only expert compute, far below the GPU streaming
    engines (reported, not cited). ik_llama.cpp `--cpu-moe` ~10 tok/s on an M4 Pro 24 GB with
    Qwen-Next 80B (#19825, in 21-backends).
  - llama.cpp #30966/#31286 (CPU share of misses) target discrete GPUs (36-llama-cpp).
- **Why it matters less on the Mac.** There is no PCIe link to save. A missed expert has to come from the SSD whichever
  unit computes it. CPU compute only helps if:
  - (a) the GPU does not wait for it (expert deferral, KTransformers, research/14 §5a; lossy), or
  - (b) the expert sits in RAM outside the GPU working set. The page cache above the 26.8 GB limit is ~5 GB minus macOS:
    only a small tier (estimate).
  - CPU and GPU share the 400 GB/s bus. Our CPU read reaches 58 GB/s on one thread and ~115 GB/s on all cores
    [L 2026-11-11-mac-bandwidth].
- **ISA** (knowledge, unchecked):
  - The M1 Max has NEON with dotprod and no i8mm (i8mm from M2) and no SME (SME from M4).
  - The AMX units are reachable only through Accelerate (fp32/fp16 BLAS, no IQ formats).
  - KleidiAI's SME2 path in llama.cpp covers Q4_0/Q8_0 only [S, Arm learning path].
  - SME GEMM on an M4 Pro: 1.23× over Accelerate (arXiv 2512.21473) [S].
  - No numbers for IQ1/IQ2 expert GEMV on M1 NEON found.
  - Reference: our Zen 5 CPU expert costs 60 to 80 µs and is limited by unpacking at ~5 GB/s per thread [L 36].

## Implications for the Metal backend (agent's view, for decision)

1. The architecture is confirmed by shipping engines: one command buffer per pass, a per-layer `MTLSharedEvent`
   handshake, an LRU arena filled by `F_NOCACHE` preads of packed experts, and router-of-l+1 read-ahead. Our E1 (~55 µs
   median per handshake, ~2.6 ms per token at 48 layers) is the fixed cost to beat. #23324 lost 21 % to it at 100 % hits.
   Add ds4's overlap (run the hit experts and the shared expert before the wait, and only the misses after it). Our E2a
   supports this.
2. Our advantages over these engines on the same chip are hypotheses, not numbers:
   - the 256-expert Coder (larger arena share);
   - QD-16 reads (13 GB/s measured);
   - overbooked prediction (16 to 24 candidates);
   - expert-union-aware verify width.
3. Risks found:
   - **watchdog**: ~5 s per command buffer, waits count, `accessRevoked` after repeated timeouts;
   - **GPU clock drop** while the host works;
   - **SSD DMA against GPU bandwidth** (Flash-MoE claims −73 %);
   - **upstream Metal lacks IQ few-row kernels** (#27350 open), so llama.cpp is no fair speed reference for spec decode
     on IQ files.

## Still open (replaces "Still open" in 21-backends for the Mac)

- Reconcile our 22 to 31 tok/s decode estimate for the large tier with the measured ~8 tok/s of ds4 (and rates of the same order
  elsewhere) on the M1 Max 32 GB. Simulate the arena at our real budget with the route traces (`tools/cache_sim/`) at ~180 slots per
  layer, then compute per-layer miss latency at QD 1 (566 µs) against the batch rate at QD 16.
- Measure GPU read bandwidth while the SSD reads at 13 GB/s (does Flash-MoE's −73 % hold on the M1 Max?).
- Measure GPU clock and kernel time after an idle gap of 0.1 to 3 ms (does the GPU need a keep-busy kernel during the
  host's turn?).
- Narrow down the event-wait timeout between 2 and 8 s. Check whether the queue accepts work after a timeout and whether
  repeated timeouts lead to `accessRevoked` (in a throwaway process).
- Time IQ1_M / Coder expert kernels at B=1 and B=2..4 on the M1 Max: ALU bound or bandwidth bound?
- An external engine on our Mac as a yardstick (ds4 with its Q2 pack). This needs a maintainer decision because it
  means downloading weights.
- Recall@k of router-of-l+1 on the Coder in prefill traces, per layer. The 2026-10-24 numbers are teacher text, 300
  tokens.
- MTP head as an expert predictor for the next token: no literature; one trace of our own would answer it.
- CPU expert kernel for IQ formats on M1 NEON: cost per expert, if deferral or a RAM tier is ever considered.
- Unchanged from before: SGLang backend structure; llama.cpp HIP on Windows with gfx1200.

## Sources (accessed 2026-11-11)

- ds4: https://github.com/antirez/ds4 , docs/SSD_STREAMING.md, docs/QWEN38_FLASH_NEXT.md
- Flash-MoE: https://github.com/danveloper/flash-moe ; Anemll fork https://github.com/Anemll/flash-moe
- TurboFieldfare: https://github.com/drumih/turbo-fieldfare
- SSD-streaming survey (2026-08-01/25): https://insiderllm.com/guides/ssd-streaming-moe-engines/
- MLX releases: https://github.com/ml-explore/mlx/releases ; custom kernels
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html ; `mx.load`, `mx.quantize` docs (0.32.4)
- mlx-lm #1788 (qwen4_exp, open), #1438 (expert offload, open), #1588 (closed 2026-08-21): https://github.com/ml-explore/mlx-lm
- MLX issues #3302, #3830, #2827 (mirror): https://upd.dev/ml-explore/mlx/issues/3302 , /3830 , /2827
- llama.cpp PRs #27742, #28948, #30680, #31688, #19555, #27350, #29869, #30047, #25294, #31469; discussion #23324:
  https://github.com/ggml-org/llama.cpp
- Apple: https://developer.apple.com/documentation/metal/mtlcommandbuffer/encodewaitforevent(_:value:) ;
  https://developer.apple.com/documentation/metal/mtlcommandbuffererror-swift.struct/code ;
  forum https://developer.apple.com/forums/thread/831017 , /771118
- ProMoE https://arxiv.org/html/2410.22134v3 ; Fate https://arxiv.org/abs/2502.12224 ; FlashMoE https://arxiv.org/abs/2601.17063 ;
  2412.00099 https://arxiv.org/abs/2412.00099
- Community numbers [S]: lilting.ch (Qwen3.6-35B on M1 Max; Qwen3.8 llama.cpp on M1 Max 64 GB), omlx.ai benchmarks.
