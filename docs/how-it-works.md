# How the engine works

Checked on 2026-11-25 against code, [BENCHMARKS.md](../BENCHMARKS.md) and [research/measurements/](../research/measurements/);
rates are from runs up to 2026-11-20. Settings: [environment.md](use/environment.md); terms: [glossary.md](glossary.md); cache
hits: [prompt-cache.md](use/prompt-cache.md).

## Overview

- Own HIP kernels in C++23, no ggml at run time; GGUF files; llama.cpp is the oracle for parity tests.
- One model module (`engine/model/`), kernel unit (`engine/gpu/arch/`) and CPU oracle (`engine/cpu/ref_*.cpp`) per
  architecture; any other `general.architecture` is rejected at load ([model-architectures.md](architecture/model-architectures.md)).
- The reference backend `hip-gfx1200` (megakernel) serves all three models; Metal serves the Coder only and is
  experimental ([macos.md](use/macos.md)). `tona-server --list-backends` prints the matrix.
- Experts live in VRAM slots and pinned RAM; the CPU computes misses with AVX-512. A persistent megakernel computes
  decode, WMMA GEMMs compute prefill. Prompt cache over VRAM, RAM and SSD.

## The models

| | Coder | Ornith | Bolt |
|---|---|---|---|
| Model | Qwen3.8-Flash-Next Coder, GSQ-RCO IQ1_M | Ornith 1.5 35B-A3B, Q4_K_M | Nemotron 3.5 Bolt 74B-A9B |
| GGUF architecture | `qwen4exp` | `qwen35moe` | `nemotron_h_moe` |
| File | 58.4 GB (2 parts) | 21.7 GB | 22.0 GB |
| Blocks | 48: 36 DeltaNet, 12 QSA (sparse attention) | 40: 30 DeltaNet, 10 gated attention, plus an MTP layer | 88: 40 Mamba-2, 8 attention, 40 latent MoE |
| d_model, vocabulary | 2560, 248,320 | 2048, 248,320 | 4096, 131,072 |
| Residual | 4 gated streams (hyper-connections) | one stream | one stream |
| Experts | 256 per layer, 10 active, plus shared | 256, 8 active, plus shared | 240 to 496, 9 active, latent 1024, plus shared |
| Extras | n-gram embedding table (PLE, 28.8 GB), QSA indexer | none | none |
| Drafter | MTP head (own GGUF) or DFlash2 | MTP layer in the model file | none; history drafts only |
| Expert types | IQ1_M family | Q4_K, Q6_K | IQ2_XXS (Mamba projections NVFP4) |

Sources: `engine/model/{qwen4exp,qwen35moe,nemotron_h_moe}.h`, `engine/backend/builtin.h`, `toolchain/models.toml`. Coder:
experts 25.15 GB (2.05 MB per expert and layer), dense weights 4.11 GB ([reference-platform.md](backends/reference-platform.md)).

## How the engine computes

- **Backend interface** (`engine/backend/backend.h`): one whole pass per call (`forward`, up to `BMAX` = 8 tokens, or
  `forward_multi`, then `commit(m)` per slot), at most `SMAX` = 4 slots. The registry refuses an unsupported
  architecture/weight-type pair at load and names the first tensor without a kernel ([backends/README.md](backends/README.md)).
- **Megakernel:** one persistent kernel per pass, 16 waves per CU with a grid barrier with a time limit; one translation
  unit of `k_mega` per architecture.
- **Prefill:** ordinary kernels in chunks of 2,048 tokens (`TONA_PREFILL_CHUNK`, up to 8,192), WMMA matmuls. Large dense
  weights use an FP8 GEMM (E4M3, factor per row and token): prefill +3 %, not bit-identical, quality cost 0.005-0.009 nats
  ([measurement](../research/measurements/2026-11-03-prefill-fp8-quality.md)); `TONA_PREFILL_FP8=0` returns to f16.
  Output projections, router, indexer, MTP head and hyper-connection matrices stay f16.
- **Layer by layer prefill:** from more than one chunk (`TONA_PF_LAYERED_MIN`) each layer runs for all chunks and its
  experts cross PCIe once; the prefill borrows the slots of the last layers and refills them afterwards. Coder 13K prefill
  +7-11 %, bit-identical; Ornith from 640 tokens; Bolt stays chunk by chunk.
- **Experts:** at every router the kernel reports through a host mailbox; the CPU picks experts and returns the VRAM hits.
  The CPU computes misses (`cpu/experts`, AVX-512 `vpdpbusd`); the GPU waits only before `moe.sum`. `TONA_CPU_EXPERTS=0`
  reads and uploads misses instead.
- **int8 matvecs:** up to 4 token vectors per unpacked weight, int8 activations with a scale per 32 values; from 6 tokens
  matrices of 4,096 rows and more run as f16 WMMA (`TONA_WIDE_MIN`, `TONA_WIDE_ROWS`). `TONA_DETERMINISTIC=1` limits passes
  to 4 tokens and makes everything bit-identical.
- **PLE rows (Coder):** one io_uring read of up to 256 requests per pass (cold 105 instead of 400 ms per chunk).
- **Coder decode techniques:** preloading of the 8 likeliest next-layer experts (`TONA_PREFETCH`, arXiv 2511.05814);
  radix block selection for the QSA indexer; IndexCache (arXiv 2603.12201, not bit-identical, `TONA_IDX_SHARE=all` undoes it);
  hyper-connections fused into GEMM epilogues; fine-grained step dependencies (`TONA_DEPS`). A pass takes 38 to 43 ms: experts 35 %,
  dense matvecs about 40 %, the rest small steps ([measurement](../research/measurements/2026-11-20-coder-decode.md)).

### Speculation and drafts

- **MTP head (Coder, default):** the model's prediction head as a further program in the kernel; GGUF
  `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` from [ggml-org/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF)
  in `work/models/draft/`. Its LM head is truncated (16,000 ids, 4,096 context rows, plus a rank-128 scorer picking 1,024
  more; `TONA_MTP_SCORER=0` disables). Draft depth is adaptive. Net effect: code 96.5 against 41.4 tok/s with
  `TONA_SPEC_OFF=1` (decode).
- **DFlash2 (`--dflash`):** a separate 0.65 GB drafter; up to 3 drafts, verify pass of up to 4; forces `TONA_SEQS=1`.
- **History drafts (all models, `engine/server/suffix.*`, arXiv 2411.04975):** if the last `TONA_SUFFIX_MIN` = 8 tokens
  occurred before, what followed is the draft. Bolt relies on it.
- **Ornith:** the model's own MTP layer (`blk.40`) is the drafter, 2.1 to 2.4 tokens per pass on code.
- **Sampling:** `temperature`, `top_p`, `top_k`, `seed` from the request; default is Qwen's recommendation (thinking T 1.0,
  top_p 0.95, top_k 20; otherwise 0.7 / 0.8 / 20), because greedy makes thinking loop; `TONA_TEMPERATURE=0` makes greedy
  the default. Speculative sampling is exact (accept with min(1, p/q), else draw from max(0, p - q)).
- **Verify passes** of up to 8 tokens (`TONA_DRAFT_N` = 7); with `TONA_DETERMINISTIC=1` 4 tokens and 3 drafts.

### Server (`tona-server`)

- Anthropic (`/v1/messages`) and OpenAI (`/v1/chat/completions`) APIs, both streaming; key via `--api-key-file` (`x-api-key`
  or `Authorization: Bearer`; `/status` and `/health` stay open). `tona start` creates `~/.config/tona/api-key`.
- HTTP listens while the model loads: `GET /status` reports start phases, then request, counters and GPU/RAM; `/v1/*`
  answers 503 until ready.
- The chat template is rebuilt in C++ per family (`ChatFamily`: Qwen3Coder, Qwen35, Nemotron) and equals the GGUF's Jinja
  template (`tests/chat_vs_jinja.py`). Tool calls become `tool_use` or `tool_calls`; a truncated one is dropped
  (`stop_reason` `max_tokens`). Foreign content is encoded without special tokens.
- A too-long prompt answers in the API's wording, which triggers Claude Code's auto-compact. The engine saves its state
  after every prompt, so a resent history computes only the new part.
- **No long waits on the GPU:** the kernel waits at most 1 s for the CPU and a pass has a 1.5 s deadline
  (`TONA_MEGA_BUDGET_MS`), then aborts and the server retries; a router wait over 100 ms pauses the kernel
  (`TONA_MEGA_YIELD_MS`), because amdgpu aborts desktop jobs after 2 s. Driver eviction time is credited from the KFD
  counter (at most 30 s). A copy to or from RAM stalled for 60 s makes the server exit with code 75 and the service restarts it.
- **Several conversations** (`TONA_SEQS`, default 2, up to 4): one slot each (state, KV, indexer pool, checkpoint), computed
  in one bit-identical pass. The second slot costs about 0.7 % decode when alone; MTP chains share a pass; queue priority is
  auto-mode checker, then main conversation, then subagent (project from `X-Tona-Project`).
- **Prefill slices:** a long prompt next to a running conversation computes in slices of 4,096 tokens.
- **Warm-up:** `max_tokens: 0` stores a prompt's checkpoints on the SSD at the lowest priority; a new project's first request
  then takes 2.9 instead of 8.5 s (13.8K prompt). Never unasked.
- **OEA** (arXiv 2511.02237, `TONA_OEA_K0` = 7): in multi-conversation passes each token keeps its best 7 experts plus those in VRAM.

## How the engine uses RAM and VRAM

- **VRAM:** expert slots are allocated last, as many as the free VRAM per sysfs allows minus a 1,024 MB reserve
  (`TONA_VRAM_RESERVE_MB`); `TONA_VRAM_GB` caps it. The prefill workspace (807 MB) lies behind the slots and holds extra
  slots between prefills. Before the slots: further conversations (149 MB state plus 27 MB indexer pool each at 16K), KV,
  the checkpoint. Buffers the GPU copies to the CPU each pass are pinned; ordinary memory made the driver evict all queues.
  16K context gives about 90 of 256 slots per layer; 256K gives 55. `TONA_VRAM_LOG=1` prints the plan;
  `TONA_VRAM_GOVERNOR` (give slots back to other programs) is off ([vram-giveback.md](architecture/vram-giveback.md)).
- **64 GB RAM:** all 25.15 GB of experts pinned as GTT buffers (`HSA_USERPTR_FOR_PAGED_MEM=0`, set by a restart at start).
  The GTT is half the RAM (32.5 GB); the overflow is registered as userptr (`TONA_PIN_OVERFLOW`, `TONA_GTT_MARGIN_MB`).
- **32 GB RAM:** experts stay in the file; the CPU computes them from the page cache or NVMe. Dense weights are dropped from
  the page cache after upload (`TONA_KEEP_CACHE=1` keeps them).
- **Limits (`engine/backend/budget.h`):** `TONA_RAM_GB` is the engine's whole RAM including page cache (the service sets
  `MemoryMax` to it). All experts are pinned if 4.5 GB remain (from about 31.6 GB). With an expert profile
  (`TONA_EXPERT_PROFILE`, shipped in `toolchain/profiles/`) and at least 70 % of experts fitting (`TONA_TIERS`), the most
  used are pinned by one ranking over all layers with 1.5 GB of page cache beside them: at the 26 GB class decode +40 to
  +100 %, prefill +70 to +90 % ([measurement](../research/measurements/2026-11-15-expert-tiers.md)). Below 70 % nothing is
  pinned. Stored conversations get the rest, at most a quarter of the limit (1-12 GB). Pinned tier: `TONA_PIN_GB`, else
  MemTotal minus 34 GiB.
- **Planner:** the GPU reads pinned experts over PCIe (about 13 µs each); a planner splits misses per layer so GPU and CPU
  finish together (`TONA_CPU_US` 60, `TONA_GD_US` 15); unpinned experts go to the CPU with one thread per physical core.
  Expert threads sleep at barriers (`OMP_WAIT_POLICY=passive`); when other processes slow the CPU, the planner moves misses
  to the GPU (`TONA_CPU_ADAPT`).
- **Bit-identical wherever computed:** CPU experts follow the GPU's order and `moe.sum` adds in fixed point, so cache, slot
  count, planner and preloading do not change the output. Passes over 4 tokens round differently in the f16 dense part
  unless `TONA_DETERMINISTIC=1`.

## Assumptions

- **Server defaults:** context 16,384 (`--ctx`; `just serve` uses 256K); thinking off unless the request asks
  (`thinking`, `reasoning_effort`) or `--thinking on`.
- **Prompt cache:** up to `TONA_PROMPT_SLOTS` (16) conversations; the running one in VRAM, the others in RAM as checkpoints
  (about 460 MB at 16K tokens; a switch takes 100-300 ms), together at most `TONA_PROMPT_RAM_GB` (12), and only if 8 GB of
  RAM stay free after the copy (a 1M conversation, 16.8 GB, is never kept at the default). Checkpoints sit at the prompt's
  end and, after 4,096 or more computed tokens, at the start of the last message; this made Claude Code's auto-mode checker
  take 1.7 instead of 48 s. A hit equals a cold run for checkpoints inside a prompt (`TONA_DETERMINISTIC=1`); a turn that
  continues generated tokens is repeatable but not the cold answer ([prompt-cache.md](use/prompt-cache.md)). With
  `TONA_PROMPT_CACHE` (`just serve`: `work/prompt-cache`) checkpoints also go to the SSD (`TONA_PROMPT_CACHE_GB` 16), one
  model per file, dir 0700, files 0600.
- **Context:** int8 KV cache, in pinned RAM from 0.4 GiB (`TONA_KV_HOST`); `just serve` uses 256K positions. Ornith and Bolt
  keep the first 128,000 cells per attention layer in VRAM (`TONA_KV_VRAM_GB`, at most 13.5 % of free VRAM): decode at 256K
  context +52 % (Ornith after 16K) to +187 % (after 128K), Bolt +22 % after 64K. Up to 1M positions run with `TONA_SEQS=1`
  (Coder: decode 56.5, prefill 1,073 tok/s at 1M); retrieval of the latest of several revisions of a fact is 51 % at 64K and
  39 % at 1M, a model limit that YaRN (off) and a larger indexer top-k do not fix. The layer-by-layer prefill stages the KV
  of the running layer in borrowed slots (`TONA_KV_STAGE`).
- **Desktop on the same GPU:** at most 32 KB LDS per block (55 KB hung). `TONA_FREE_CUS` frees compute units: 4 CUs give a
  98 % canary frame rate at decode -4.4 %, prefill -6.0 %.
- **int8 rounding:** with and without speculation the output is equal in float, in int8 not always bit-identical.
- unverified: the cost of the megakernel's time limits on the 9950X was measured on the 9600X.

## Ornith and Bolt

Own kernels where shapes differ; server, expert tiers and queue are shared. Dense prefill attention uses a 16-row tile per
wave (`k_fattn_prefill_wave`; prefill at 64K: Ornith 1,297 → 2,965, Bolt 884 → 1,757 tok/s). Bolt has NVFP4 and IQ2_XXS
kernels (`matvec_q8_nvfp4.h`, `matvec_q8_iq.h`), Mamba-2 decode with replay commit, a chunked SSD prefill, latent MoE and
the Pixtral pre-tokenizer. The layer-by-layer prefill is bit-equal to the chunked one in every plan
(`scripts/bolt-determinism.sh`). Per-model expert profiles: `toolchain/profiles/<model id>.bin`.

## Measured state

Server, 64 GB RAM, Ryzen 9 9950X, RX 9060 XT, ctx 262144, KV cache in RAM, code workload, greedy, queue exclusive, desktop
holding 3.0 to 3.5 GiB VRAM. Spread and setup: [BENCHMARKS.md](../BENCHMARKS.md); the installer shows the same numbers.

| Model, drafter | Context | Decode, tok/s | Prefill, tok/s | Source |
|---|---:|---:|---:|---|
| Coder, MTP | 2K | 70.6 | not measured | 2026-11-20-coder-decode.md, 2 rounds |
| Coder, MTP | 16K | 67.8 | 1,835 | same |
| Coder, MTP | 64K | 74.1 | 1,869 | same |
| Ornith, MTP | 2K | 114.3 | 2,582 | 2026-11-20-ornith-bolt-clean-state.md, 3 rounds, spread under 0.2 % |
| Ornith, MTP | 16K | 95.5 | 4,082 | same |
| Ornith, MTP | 64K | 64.7 | 2,964 | same |
| Ornith, MTP | 128K | 47.5 | 2,192 | same (not in the offer data) |
| Bolt, history drafts | 2K | 40.1 | 1,415 | same, 2 rounds |
| Bolt | 16K | 38.6 | 1,578 | same |
| Bolt | 64K | 36.3 | 1,429 | same |

"Decode after 2K" is the decode rate after a prompt of that length. The Coder's rates sit below its short-prompt rates
because 256K context leaves 55 slots per layer.

Other Coder rates (9950X, one round each, 2026-11-18-scale-run):

| What | Decode, tok/s | Prefill, tok/s |
|---|---:|---:|
| bench-set code / text, short prompts | 94.7 / 68.1 | |
| After 30K / 130K tokens | 75.0 / 64.7 | 1,959 / 1,855 |
| After 250K tokens | | 1,732 |
| After 512K / 1M tokens (`TONA_SEQS=1`) | 60.8 / 56.5 | 1,451 / 1,073 |

llama.cpp-hip (`-t 16`) in the same run: 21.5 / 22.1 / 17.4 tok/s decode (short code, long decode, needle at 30K) against
Tona's 111 / 77.9 / 66.1. The bandwidth ceiling is about 21 ms per pass (3.5 GB dense plus about 2.3 GB of experts at
315 GB/s); a pass takes 38 to 43 ms. Prefill is compute-bound (WMMA peak 93 TFLOPS f16, 186 TFLOPS fp8,
[reference-platform.md](backends/reference-platform.md)).

## Offer data

`data/modeldb/offer.json` feeds the installer ([installer.md](use/installer.md)) and the client ([client.md](use/client.md)):
generated into `engine/offer/` by `tools/gen_offer_data.py`, checked by `tools/gates/offer.py`, records in
`data/modeldb/models/`. It holds tasks, hardware classes, the recommendation per task and class, and per model strengths,
weaknesses, licence and rates at 2K, 16K, 64K, each with a basis word (`measured`, `provisional`, `vendor`,
`independent`, `open`) and a source file. Sizes come from `toolchain/models.toml`. Today one class (PC, 16 GB VRAM, 64 GB
RAM); coding: Coder recommended, Ornith the alternative (better at Rust; the Coder best measured for C#, JavaScript,
Python); knowledge: Bolt; five tasks are "coming". `TONA_PRIORITY` is read by the engine only in a developer start on the
Mac without `TONA_RAM_GB` (`backend/priority.h`).
