# Environment variables

`TONA_*` variables are read through `engine/backend/env.h`, once per process. Unset or empty means the default. A
non-number prints `TONA_X: "abc" is not a number, using <default>` and uses the default. Flags are on when set, non-empty
and not `0`. Measurement switches and test hooks are listed at the end ([Diagnostics and tests](#diagnostics-and-tests)).

## backend

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_BACKEND` | text | `auto` | Backend to create (`hip-gfx1200`, ...); `auto` takes the first that accepts the machine; `--backend` wins; an unknown name stops the start and lists candidates. |
| `TONA_METAL_SOURCES` | text | unset | Metal, development: kernel source directory (`engine/backend/metal/kernels`) compiled at start instead of `tona.metallib`. |
| `TONA_DETERMINISTIC` | flag | off | Every batch width computes bit-identically (passes up to 4 tokens, no wide kernel). |
| `TONA_RAM_GB` | number | unset | RAM the engine may hold (pinned experts, KV in RAM, conversation buffers); unset: unlimited. |
| `TONA_VRAM_GB` | number | unset | VRAM the engine may use instead of "all except the reserve". |
| `TONA_PROMPT_RAM_GB` | number | 12, or derived from `TONA_RAM_GB` | Upper bound of RAM for stored conversations; `0` stores none. |

**Stored conversations (`backend/budget.h`, `server/prompt_cache.cpp`).** A conversation is stored only if its copy
(`stash_bytes`) is at most `TONA_PROMPT_RAM_GB` and, plus an 8 GB reserve (`STASH_RESERVE_GB`), fits the free RAM: the
smaller of MemAvailable and, per cgroup up to the root, `memory.max` minus anonymous, shared and kernel memory (page cache
does not count; a `systemd-run -p MemoryMax=...` scope is honoured). Older conversations are dropped first. Otherwise the
log says "not stashed: needs ... GB, budget ... GB, free RAM ... GB" and the next request computes from scratch; nothing goes
to the SSD either. So **a 1M conversation (16.8 GB) is never stored with the default budget**. The reserve exists because
of a 48-minute stash hang ([measurement](../../research/measurements/2026-11-01-stash-hang.md)).

**Copy time limits (`backend/stall.h`, `gpu/guarded_copy.cpp`).** Stash and restore copies run through pinned bounce
buffers (4 x 16 MB); the engine thread waits at most `2 s + chunk / 250 MB/s` per chunk. On a stall the stash gives its
memory back and waits up to 60 s (`STALL_GRACE_S`); if the copy ends, the stash is dropped (restore: the prompt is computed
anew); otherwise the server logs `FATAL: stalled ...` and exits with code 75, and `Restart=on-failure` restarts it.

## gpu (megakernel)

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_FREE_CUS` | integer | 0 | Compute units left to the desktop (fewer blocks, `HSA_CU_MASK` set to match); costs decode and prefill ([research/63](../../research/63-desktop-latency.md)). |
| `HSA_CU_MASK` | text (ROCm) | unset | Read to detect an existing CU mask before `TONA_FREE_CUS` applies its own. |
| `TONA_EXPERT_SLOTS` | integer | unset | Fixed expert slots per layer where the caller gives none (server; tests). |
| `TONA_VRAM_RESERVE_MB` | integer | 1024 | VRAM kept free for the desktop when slots are sized automatically, on top of the desktop's use at start; costs about 0.6 % decode against 640 ([measurement](../../research/measurements/2026-11-15-expert-tiers.md)). |
| `TONA_KV_VRAM_GB` | number | unset | Fixed VRAM budget of the KV cache; unset: free VRAM minus the reserve. |
| `TONA_SLOTS_CHUNKED` | flag | off | Build the slots' core and prefill workspace from 32 MiB chunks so the kernel can take back a chunk. |
| `TONA_VRAM_GOVERNOR` | flag | off | Last slots are elastic: given back between passes when free VRAM falls below half the reserve, regrown after 30 s of calm ([expert-tiers.md](../architecture/expert-tiers.md), 5a). Needs `TONA_VRAM_RESERVE_MB`. |
| `TONA_VRAM_ELASTIC_MB` | integer | 1536 | With the governor: elastic part of the slots. |
| `TONA_VRAM_STEP_SLOTS` | integer | 4 | With the governor: slots per layer in one generation (1 to 16). |
| `TONA_VRAM_VA_GB` | integer | 512 | With the governor: address space reserved for new generations (never reused, ROCm/TheRock #8744). |
| `TONA_PIN_GB` | number | unset | Pinned expert stage in GB; unset: from `TONA_RAM_GB`, else MemTotal minus 34 GiB. |
| `TONA_GTT_MARGIN_MB` | integer | 1024 | GTT kept free when sizing pins, for the desktop's fallback placements ([research/61](../../research/61-desktop-vram-reserve.md)). |
| `TONA_PIN_OVERFLOW` | text | `register` | Pins beyond the GTT (`mem_info_gtt_total`): `register` maps them as userptr; `cap` lowers the pin budget (prefill -20 to -27 %, decode -9 %). |
| `TONA_QUEUE_SPACERS` | integer | 2 | Idle HIP streams (0 to 8) created at load; with 2, 2,000-token prefill is 7-8 % faster, cause unknown ([measurement](../../research/measurements/2026-11-12-prefill-bimodal.md)). |
| `TONA_USERPTR_THP` | integer | 1 | Align userptr pins to 2 MB and madvise huge pages, so page migration stops fewer queues ([measurement](../../research/measurements/2026-11-11-evict.md)); 0: 4 KB pages. Needs THP `madvise` or `always`; the start log warns below 90 % on huge pages. |
| `TONA_KEEP_CACHE` | flag | off | Keep the page cache and file mappings of the dense weights after upload. |
| `TONA_CTX_AUTO` | flag | 1 | Written and read by the installer and `tona config` only: 1 means `TONA_CTX` was chosen by "auto" and is recomputed when agents or the RAM limit change. The server reads `--ctx`. |
| `TONA_SEQS` | integer | 2 | Sequence slots (1 to the compile-time maximum); forced to 1 with a drafter or `TONA_STATE_HOST`. Metal: default 1, at most 4. |
| `TONA_PREFILL_CHUNK` | integer | 2048 | Prefill chunk in tokens, multiple of 128, 0 to 8192. Metal: multiple of 32, at most 4096; 0 prefills through decode passes of 8. |
| `TONA_ROPE_YARN` | text | off | `off` or a YaRN factor above 1 (e.g. `4`) for the RoPE of the QSA and MTP layers (`backend/rope_yarn.h`); `off` is bit-identical to no switch; anything else is a start error. |
| `TONA_YARN_MSCALE` | number | 0.1 ln(factor) + 1 | Factor on cos and sin of the YaRN rotation; `1.0` switches scaling off. |
| `TONA_IDX_TOPK` | integer | 2048 | Tokens each query attends to through the indexer, rounded down to blocks of 4, at most 4,092; changes the numbers and invalidates stored conversations. |
| `TONA_IDX_SHARE` | text | `0,2,4,6,8,10` | IndexCache (arXiv 2603.12201), decode only: QSA layers (0-11) that run their own indexer; `all` is bit-identical to the engine without it. |
| `TONA_STATE_HOST` | flag | off | Recurrent state in pinned host memory; implies one slot. |
| `TONA_KV_HOST` | integer | -1 | 1 KV cache in pinned RAM, 0 in VRAM, -1 RAM above 0.4 GiB of KV (context of about 31K and up). |
| `TONA_KV_STAGE` | integer | 1 | 0 turns off staging of the running layer's KV in the borrowed slots during the layer-by-layer prefill. |
| `TONA_PF_LCHUNK` | integer | 6144 | Chunk of the layer-major prefill, multiple of 128, up to 8192. |
| `TONA_WIDE_ROWS` | integer | 4096 | Minimum rows of a dense matrix to run on WMMA in wide passes. |
| `TONA_WIDE_MIN` | integer | 6, or 0 with `TONA_DETERMINISTIC` | Tokens per pass from which dense matvecs run on WMMA; 0 disables. |
| `TONA_WIDE_DRAFT` | integer | 1 | 0 keeps the drafter off the WMMA path. |
| `TONA_WIDE_DRAFT_MIN` | integer | `TONA_WIDE_MIN` | Threshold of the drafter program. |
| `TONA_MEGA_BUDGET_MS` | integer | 1500 | Time one pass may run before the kernel yields to check the CPU. |
| `TONA_MEGA_YIELD_MS` | integer | 100 | Router wait before the kernel yields. |
| `TONA_MEGA_FREEZE_US` | integer | 2000 | A poll gap this long means the wave did not run (evicted); it does not count towards wait limit and deadline; 0: count all. |
| `TONA_MEGA_WAIT_CAP_MS` | integer | 5000 | Longest wall time one wait may last. |
| `TONA_EVICT_CREDIT` | flag | on | Queue eviction time (KFD `evicted_ms`, `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu_id>/`) does not count towards the pass deadline (at most the wall time and 30 s); no answer in 50 ms or no counter: the deadline aborts as before. |
| `TONA_MEGA_TIMEOUT_MS` | integer | 1000 | Barrier time limit; on expiry the pass aborts and is reported. |
| `TONA_Q6_K_TILE` | integer | 0 | 1 tiles the 128 Q6_K layer matrices into type 1019 ([tona-types.md](tona-types.md)) at upload; same logits, but the tile-loop kernel makes the 1-token pass 2.0 ms slower. A file with type 1019 uses that kernel anyway. |
| `TONA_MEGA_LANES_EXP` | integer | 8 | Lanes per row for expert matvecs. |
| `TONA_MEGA_LANES_MAX` | integer | 32 | Upper limit of lanes per row. |
| `TONA_MEGA_TSPLIT` | integer | 64 | Maximum split of a row range across tasks. |
| `TONA_FUSE_QUANT` | integer | 1 | 0 stops fusing the activation quantization into the previous step. |
| `TONA_DEPS` | integer | 1 | 0: full barriers instead of fine-grained step dependencies. |
| `TONA_ATTN` | text | unset | `old`: one block per (token, head); `wave`: split-K, a wave per head; otherwise split-K on WMMA. |
| `TONA_DATTN` | text | unset | `split`: drafter attention with a wave per (token, head, chunk) instead of WMMA. |
| `TONA_MEGA_HC_BF16` | flag | off | Hyper-connection streams in bf16 instead of the int8 path. |
| `TONA_CPU_EXPERTS` | integer | 1 | 0: misses are read into VRAM slots instead of computed on the CPU. |
| `TONA_CPU_US` | number | 60 | Planner cost of one CPU expert in µs. |
| `TONA_GD_US` | number | 15 | Planner cost of one expert read directly from pinned RAM by the GPU. |
| `TONA_CPU_ADAPT` | integer | 1 | The CPU expert cost follows the measured time when other processes load the CPU (`gpu/cpu_cost.h`): above 2 x the base, misses go to the GPU (bit-identical), the CPU is re-probed about every 17 layers. 0: fixed; 2: adapts under `TONA_DETERMINISTIC`/`TONA_PLAN` too (test). |
| `TONA_PLAN` | text | unset | `gpu` sends all pinned misses to the GPU, `cpu` to the CPU; otherwise the planner decides. |
| `TONA_FILL` | integer | 0 | Misses per layer refilled into the cache after a pass. |
| `TONA_PREFETCH` | integer | 8 | Experts prefetched per layer from the next layer's predicted router; 0 disables. |
| `TONA_PRE_KEEP` | integer | 1 | 0 lets prefetching evict predicted experts already cached. |
| `TONA_PRE_BACK` | integer | 1 | 0 puts prefetched experts at the front of the LRU list. |
| `TONA_EXPERT_POLICY` | text | `lru` | VRAM cache eviction: `lru` or `classlru` (4 classes by recency and frequency, half-life 128 tokens; `gpu/class_lru.h`). Megakernel only. |
| `TONA_SHARE_WAIT` | integer | 1 | 0 computes the shared expert in moe.gate_up instead of during the host wait. |
| `TONA_EXPERT_PROFILE` | text | unset | File with per-expert use counts; seeds the VRAM cache and pinned set. |
| `TONA_EXPERT_PROFILE_OUT` | text | unset | File for the accumulated counts at exit. Metal: arena only (`TONA_METAL_ARENA_GB=auto`); SIGINT/SIGTERM write it and end the server. |
| `TONA_TIERS` | integer | 1 | With a profile and `TONA_RAM_GB`: pin the most used experts when at least 70 % of expert bytes fit; below that nothing is pinned. 0: off ([expert-tiers.md](../architecture/expert-tiers.md)). |
| `TONA_TIERS_MIN_FRACTION` | number | 0.7 | Smallest share of expert bytes worth pinning in part. |
| `TONA_TIERS_CACHE_GB` | number | 1.5 | Page-cache floor kept next to the pins. |
| `TONA_MTP_VOCAB` | text | unset | `full`: the target's whole LM head for MTP; else a truncated head from this vocabulary file. |
| `TONA_MTP_VOCAB_N` | integer | vocab size | Maximum rows of the truncated MTP head. |
| `TONA_MTP_DYN` | integer | 8192 (4096 with scorer) | Rows of the truncated head that follow the context. |
| `TONA_MTP_HEAD` | text | unset | `iq4`: store the truncated head as IQ4_NL. |
| `TONA_MTP_SCORER` | integer | 128 | Rank of the low-rank scorer for the dynamic rows (multiple of 16, at most 256; 0: off, 41,500-id list and 8,192 context rows). Built from the head at first start (about 6 s); not used with several sequences. |
| `TONA_MTP_SCORER_ROWS` | integer | 1024 | Exact rows per draft step picked by the scorer (multiple of 32). |
| `TONA_MTP_SCORER_IDS` | integer | 98304 | The scorer ranks only ids below this. |
| `TONA_MTP_SCORER_FILE` | text | `<model>.mtp-scorer-r<rank>`, else `~/.cache/tona/mtp-scorer-r<rank>-<fingerprint>.bin` (`$XDG_CACHE_HOME`) | Scorer file; one made for another head is replaced with a log line. |
| `TONA_MTP_EXP` | text | unset | `q8`: keep MTP experts as Q8_0; else converted to Q2_0 on the GPU. |
| `TONA_CLOCK_HOLD` | integer | 0 | Keep the GPU clock up between requests (a one-wave spin for as long as this server's pauses suggest, `server/hold_policy.h`; first prefill about 0.1 s earlier, +27 W while holding). 1: first spare queue (`TONA_QUEUE_SPACERS`); 2: own stream. Needs CPU experts with prefetch. The user's choice: `tona setup` asks, default no ([research/60](../../research/60-gpu-clock-ramp.md)). |
| `TONA_CLOCK_HOLD_S_PER_SAVE` | number | 20 | Seconds of hold the engine may spend per ramp it saves. |
| `TONA_OEA_K0` | integer | 7 | Own experts per token kept by OEA in multi-sequence passes; 0 disables. |
| `TONA_OEA_TAU` | number | 0 | Also keep own experts with router probability at least this. |
| `TONA_OEA_HITS` | integer | 1 | 0 stops keeping own experts already in VRAM. |
| `TONA_OEA_ROW0` | flag | off | Keep the first row of every sequence entirely. |
| `TONA_OEA_EXTRA` | integer | 0 | Add this many experts to the shared base set by summed probability. |
| `TONA_MISS_DROP` | number | 0 | Experimental, lossy: drop misses weighted below this for all tokens. |

### Metal backend

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_METAL_MMA` | integer | 1 | Dense mat-vecs of 4-token chunks on simdgroup matrix units in half precision; changes the numbers of passes of 4+ tokens; 0 or `TONA_DETERMINISTIC=1`: f32 ([quality](../../research/measurements/2026-11-12-mac-mma-quality.md)). |
| `TONA_METAL_MMA_WIDE` | integer | 1 | With MMA: one dispatch for up to 8 tokens; bit-equal to 4-token chunks. 0: chunks. |
| `TONA_METAL_MV5` | integer | 1 | f32 mat-vecs with up to 5 tokens per dispatch; bit-identical per token. 0: 4. |
| `TONA_METAL_GPU_PICK` | integer | 1 | MTP pick on the GPU (`kernels/mtp_pick.metal`); same tokens, q within 1e-6. 0: host scan. |
| `TONA_METAL_EARLY_COMMIT` | integer | 2 | Commit a pass's inputs and first k layers at once, encode the rest while the GPU runs; bit-identical. 0: one command buffer per pass. |
| `TONA_METAL_KEEPALIVE_MS` | integer | 500 | After each pass, for `TONA_METAL_KEEPALIVE_S`, a command buffer every N ms that reads 16 bytes of every weight buffer (avoids a 330-380 ms wake wait); 0: off. |
| `TONA_METAL_KEEPALIVE_S` | integer | 180 | Window after a pass in which the keepalive sends. |
| `TONA_PLE_PREFETCH` | integer | 2 on macOS, 0 elsewhere | Read-ahead of n-gram rows (`madvise(MADV_WILLNEED)`), bit-identical: 1 a pass's own rows, 2 also the next pass's. Linux stays 0 until measured. |

## gpu (megakernel build macros)

The device code reads no environment variable. Macros (`-D...` in `tona_gpu`'s flags) change machine code and need the
checks of [verification.md](../contribute/verification.md).

| Macro | Default | Meaning |
|---|---|---|
| `TONA_MEGA_WAVES` | 16 | Waves per block of `k_mega`; 2 blocks of 8 deadlocked next to the desktop. |
| `CHAIN_WAVES` | 8 (CMake `TONA_CHAIN_WAVES`) | `amdgpu_waves_per_eu` cap of the matvec chain and `k_mega`. |
| `TONA_DN_CW` | 8 | DeltaNet state columns per wave in `op_dn_recur`; 16 spills registers. |
| `MEGA_TIMEOUT_MS` | 1000 | Compile-time default of `TONA_MEGA_TIMEOUT_MS`. |

## gpu (matvec, ops)

Single-kernel path (`engine/gpu/matvec.hip`, `forward*.hip`), not the megakernel.

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_Q8_LANES` | integer | 0 | Lanes per row for the int8 single kernels (power of two up to 32); 0: the largest that divides the row's sub-blocks. |
| `TONA_GPU_LOCK` | text | `work/gpu.lock` of the checkout | Lock file serialising every GPU script and `scripts/ab.py` (`scripts/gpu-lock.sh`); set one path when the GPU is shared beyond worktrees. |

## gpu (prefill, gemm)

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_PF_EXTRA_SLOTS` | integer | 1 | 1: the prefill workspace holds extra expert slots between prefills (`engine/gpu/mega_extra.h`, check `scripts/extra-slots-check.sh`); 0: idle. |
| `TONA_PF_EXTRA_RANKS` | integer | 2 | Blocks per layer of the extra slots (1 to 16). |
| `TONA_PF_LAYERED` | integer | 1 | 0: always chunk by chunk. |
| `TONA_PF_LAYERED_MIN` | integer | unset | Go layer by layer from this many tokens (default: more than one chunk). |
| `TONA_PF_GROUP` | integer | 32768 | Most tokens of one layer-major sweep; halved until buffers fit the borrowed slots. |
| `TONA_PF_LOAN_PCT` | integer | 50 | Percent of the layers (from the back, 1-90) whose slots the layer-by-layer prefill may borrow. |
| `TONA_PF_MIX` | integer | 1 | 0: up GEMM and gated-residual mix as two kernels (not with FP8). |
| `TONA_PF_PAIR` | integer | 1 | 0: no GEMM for two weights with the same input (down, inject). |
| `TONA_PREFILL_FP8` | integer | 1 | Large dense weights (at least 512 rows; not router, indexer, position embeddings, MTP head, hyper-connection matrices) through the FP8 GEMM: prefill +1.6 to +3.2 %, not bit-identical ([quality](../../research/measurements/2026-11-03-prefill-fp8-quality.md)). `0`: f16. `TONA_FP8_V1=1`: first-generation kernel. |
| `TONA_PF_FP8_OUT` | flag | off | The layers' output projections (`ssm_out`, `attn_output`) also through FP8; off because they carry about 70 % of the FP8 loss ([measurement](../../research/measurements/2026-11-10-fp8-quality.md)). |
| `TONA_PLE_IO` | text | unset | `mmap`: read the n-gram table through the file mapping instead of io_uring. |

## cpu

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `OMP_NUM_THREADS` | text (OpenMP) | physical cores | Threads of every OpenMP team. `tona-server` and `tona-gpu-gen` set it at start (6 on the 9600X, 16 on the 9950X) via the one-time restart below (`backend/runtime_env.h`); your value stays. Not set on macOS. |
| `OMP_WAIT_POLICY` | text (OpenMP) | `passive` | Set by the same restart if neither this nor `GOMP_SPINCOUNT` is set; waiting threads sleep, so a pre-empted team member gets its slice (decode at 130K with 8 busy threads 62 instead of 38 tok/s). `active`: old behavior. |
| `HSA_USERPTR_FOR_PAGED_MEM` | text (ROCm) | `0` | Set to `0` by the same restart if unset: `hipHostMalloc` memory becomes a kernel GTT buffer that page migration cannot invalidate, so the driver stops evicting queues under page-cache pressure ([measurement](../../research/measurements/2026-11-01-slow-decode.md)). The GTT is half the RAM (`mem_info_gtt_total`): pins are capped to what is left (log `host memory: pinned experts limited ...`); if the fixed buffers alone do not fit, the start fails naming this variable. `1`: userptr. The start log says `host memory: GTT` or `userptr`. |

## server

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_ALLOWED_ORIGINS` | text | unset | Comma-separated browser origins (e.g. `http://localhost:3000`) accepted besides the loopback origin. Other `Origin`s, and on a loopback bind any `Host` other than `127.0.0.1`, `localhost`, `[::1]` with the server's port, get 403 (DNS rebinding; `engine/server/request_guard.h`). Non-browser clients are unaffected. |
| `TONA_TEMPERATURE` | number | unset | Temperature for requests that give none (0: greedy); unset: Qwen's recommendation (thinking T 1.0, top_p 0.95, top_k 20; else 0.7, 0.8, 20). |
| `TONA_THINK_BUDGET` | text | unset | Thinking tokens per effort level, `low=2048,medium=8192,...`; a missing level is unlimited; malformed entries warn and are ignored. |
| `TONA_ROLE_EFFORT` | text | unset | Effort per request role, `subagent=low,auxiliary=low` (roles: main, subagent, auxiliary). |
| `TONA_THINK_FORCE_MSG` | number | 1 | 0: at budget end force only `</think>` instead of Qwen's closing sentence. |
| `TONA_PROMPT_SLOTS` | integer | 16 | Conversations kept for reuse including the running one (minimum 1). |
| `TONA_PROMPT_CACHE` | text | unset | Directory for SSD checkpoints that survive a restart; unset: RAM only. Created 0700, files 0600; an existing directory is tightened if yours and holding only checkpoints; if others can write into it and it cannot be tightened, the SSD cache is switched off with an error. Warm-up checkpoints go here. |
| `TONA_PROMPT_CACHE_GB` | number | 16 | Size limit of that directory; the longest unused files go first. |
| `TONA_PREFILL_MIN` | integer | 32 | A prompt with more new tokens than this uses the prefill kernels. |
| `TONA_PREFILL_SLICE` | integer | 4096 | Tokens per slice when a long prompt computes next to running conversations. |
| `TONA_PREFILL_YIELD_MS` | number | 1000 | Decode time for other conversations between slices (0: one go). |
| `TONA_SUFFIX_MIN` | integer | 8 | Minimum match length for a history draft (0: off). |
| `TONA_DRAFT_N` | integer | 7 | Most drafts per pass (1-7); 3 with `TONA_DETERMINISTIC`. |
| `TONA_SPEC_OFF` | flag | off | No speculation, one token per pass: the baseline for "what is speculation worth" ([testing.md](../contribute/testing.md#speed-comparison-scriptsabpy)). |
| `TONA_DRAFT_CALIB` | number | 1 | 0: raw DFlash2 probabilities without online calibration. |
| `TONA_DRAFT_CALIB_KEEP` | flag | off | Keep the calibration across requests (not repeatable). |
| `TONA_DRAFT_FULL` | flag | off | DFlash2: always draft at full width. |
| `TONA_DRAFT_BIG_AT` | integer | 0 | DFlash2: large drafter block after this many accepted drafts (0: after all drafts of a pass). |
| `TONA_DRAFT_STAY` | integer | 0 | DFlash2: stay in the large block while at least this many drafts are accepted (0: off). |
| `TONA_DRAFT_ENTER_P` | number | 0 | DFlash2: minimum calibrated probability product to enter the large block. |
| `TONA_DRAFT_BIG_ALWAYS` | flag | off | DFlash2: always the large block. |
| `TONA_MTP_STEP_MS` | number | 1.15 | Assumed cost of one MTP step in ms. |
| `TONA_MTP_FILL_MS` | number | 1.3 | Assumed cost of the MTP catch-up in ms. |
| `TONA_MTP_FIXED` | flag | off | MTP always drafts the full number of tokens. |
| `TONA_MTP_MULTI` | flag | off | MTP chains of several conversations step together (no measured gain). |
| `TONA_MTP_SAMPLE` | number | 1 | 0: with sampling, draft greedily instead of from the MTP head's distribution. |
| `TONA_ERROR_LINGER_S` | integer | 3 | Seconds the server stays up after a failed start so `tona` can read the error. |
| `TONA_DIAG` | flag (`toolchain/env.sh`, `scripts/tona-service.sh`) | off | Developer machines: `toolchain/env.sh` (every build, test and measurement script) and `just start` / `just install-service` set `TONA_ALLOC_TRACE`, `TONA_KFD_SMI_LOG` and `HSA_COREDUMP_PATTERN` under `TONA_DIAG_DIR` (unless set) and lift the core limit, so a GPU fault leaves a trace and a dump (about 45 GB). `scripts/diag-clean.sh` keeps the newest dump and deletes traces older than 7 days. `just serve` does not use it. |
| `TONA_DIAG_DIR` | text | `work/diag` | Directory for `TONA_DIAG` files. |

### Warm-up requests

A request with `max_tokens: 0` (or `max_completion_tokens: 0`) on `/v1/messages` or `/v1/chat/completions` computes the
prompt, stores its checkpoints and samples nothing; the next request starting with the same tokens continues from there.

- **Answer:** Anthropic `content: []`, `stop_reason: "max_tokens"`, `usage` (`input_tokens` computed now,
  `cache_read_input_tokens` found). OpenAI: empty assistant message, `finish_reason: "length"`, `usage`, `timings`. Both
  carry `tona_warm: {done, total, checkpoint, interrupted}` (tokens covered, prompt tokens, last stored position, whether a
  real request interrupted it).
- **Errors (400):** `stream: true`, a negative value, a forced `tool_choice`, a last assistant message. Authentication,
  context limit and body size apply as usual.
- **Checkpoints:** at the head, 16 tokens before the end of the last message's text (margin for differently merged words),
  and at the prompt's end. With `TONA_PROMPT_CACHE` they go to the SSD only, else into the RAM stash. Prompts under 1,024
  tokens store nothing.
- **Class `warm`:** lowest priority, below `subagent`; starts only when nothing else runs, computes in `TONA_PREFILL_SLICE`
  slices, and a real request or closed connection stops it at the next slice (at most about 2.7 s). Progress of at least one
  slice is stored; a later warm-up continues there.
- **Same output:** nothing is generated or drafted; the follow-up answer equals the cold one byte for byte with
  `TONA_DETERMINISTIC=1` ([prompt-cache.md](prompt-cache.md)).

## tests

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_MODEL` | text | unset | GGUF path for tests that need one; without it they skip or use random data. |

## tools

Read by the programs in `engine/tools/`.

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_PORT` | text | 8662 | `tona`: server port (also a settings key; this variable wins), unless `--url` is given. |
| `TONA_USE`, `TONA_PRIORITY`, `TONA_CLOSE` | text | desktop, balanced, ask | Control-centre keys of `tona.conf` ([client.md](client.md)): `tona config set use\|priority\|close`. The engine does not read `TONA_USE`. `TONA_PRIORITY` is read only by the Metal backend in a start without `TONA_RAM_GB`: `tona` (server first) leaves the 1.5 GB `TIERED_CACHE_MIN_GB` out of the arena's reserve (`backend/priority.h`); `work`, `balanced`, unknown text and any start with `TONA_RAM_GB` keep it. `TONA_CLOSE` (ask, keep, stop) is what `q` does with the server. |
| `TONA_START_CMD` | text | `just start` in a checkout, else `systemctl --user start tona-server` | Command `tona` uses to start the server (`s` key, `--start`, agent commands). |

## release scripts

Read by the scripts of [release.md](../maintain/release.md).

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_RELEASE_REPO` | text | `tobias-nawa/tona` | `release-sign.sh`, `release-verify-build.sh`: `OWNER/REPO`. |
| `TONA_JOBS` | integer | 4 | `release-build.sh`: parallel compile jobs. |
| `TONA_KEEP_WORK` | flag | off | `release-build.sh`: keep the scratch directory. |

## updater

Read by `tona` (`engine/update/config.cpp`); see [update.md](update.md).

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_UPDATE_BASE` | text | `https://github.com/tobias-nawa/tona/releases` | Source of manifests and assets: `https://...` or `http://127.0.0.1:<port>/...` (tests), no trailing slash or query; anything else disables the updater and says why. |
| `TONA_INSTALL_DIR` | text | `$XDG_DATA_HOME/tona`, else `~/.local/share/tona` | Install root; also the test whether `tona` runs from a release installation (updates are offered only from `<root>/versions/`). |

## the `tona` command

Read by `tona` when it starts a coding agent ([client.md](client.md#coding-agents)).

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `TONA_TOOLS` | text | unset (all tools) | `lean`: Claude Code with six tools (Bash, Read, Edit, Write, Glob, Grep), no MCP (about 4.8K instead of 20K tokens up front), OpenCode without the skill scan; a list such as `Bash,Read` picks Claude Code tools. |
| `TONA_COMPACT` | text | `128k` | Claude Code compacts from this size; applies only when the server's context is 128K or more. |
| `TONA_MODE` | text | `acceptEdits` | Claude Code `--permission-mode`, e.g. `auto`. |
| `TONA_EFFORT` | text | `medium` (else the settings file's) | Claude Code `--effort`; the Agents page sets it; an explicit `--effort` wins. |
| `TONA_API_KEY_FILE` | text | `~/.config/tona/api-key` (else the settings file's) | API key file the agents send; without it the key is `local`. |
| `PI_CODING_AGENT_DIR` | text | `~/.pi/agent` | Folder whose `models.json` holds the provider `tona` for pi. |

## Diagnostics and tests

Measurement switches, probes and test hooks; none is needed in normal operation. A trailing name in brackets is the tool or
build flag it needs.

Megakernel and GPU:

- `TONA_MEGA_DUMP_PROGRAM` file: dump op programs, task tables, buffer layout. `TONA_MEGA_INFO`: kernel resource use and pinned-expert summary at start. `TONA_VRAM_LOG`: log VRAM and pin budget decisions. `TONA_MEGA_PROF`: time every step. `TONA_MEGA_STALL_MS`: with the profile, log steps and CPU stretches longer than this.
- `TONA_MEGA_ABORT_TRACE`: record barrier arrivals per block for abort reports. `TONA_MEGA_GAP_HIST` (build `-DTONA_WAIT_DIAG=ON`): histogram of poll gaps. `TONA_BAR_PROF` file: barrier arrival times. `TONA_TOKHIST`: tokens per union expert per pass width.
- `TONA_MEGA_K8`: passes of up to 4 tokens on the 8-token kernel. `TONA_MEGA_NOBAR`, `TONA_MEGA_NOBAR_STEPS`: drop barriers (wrong output). `TONA_MEGA_NO_STAGE`: no LDS staging of the activation. `TONA_MEGA_TASKS`, `TONA_DEPS_DUMP`: log task tables, dependency plan.
- `TONA_FAKE_EXPERTS`: skip expert reads (slot = expert modulo slots; meaningless output). `TONA_CPU_FAKE`: skip CPU experts. `TONA_FAKE_ROUTE`: route by hash. `TONA_PREFETCH_CHECK`: compare prefetched slots with RAM. `TONA_WARM_TEST`: expert warm-up test mode. `TONA_VRAM_FAKE_FREE` file: lines "<seconds> <free MiB>" replace the card's free VRAM to test the governor.
- `TONA_ROUTE_LOG`, `TONA_ROUTE_TRACE`, `TONA_ROUTE_TRACE_P` files: chosen experts, expert unions, router logits. `TONA_PREDICT_STATS`: next-layer router prediction quality. `TONA_CPU_COST_LOG` file: one line per CPU-experts call. `TONA_OEA_ALL`: OEA on single-sequence passes too.
- `TONA_MTP_PROF`: time the MTP program. `TONA_MTP_MULTI_CHECK`: recompute each multi-sequence MTP slot alone and compare. `TONA_MTP_TAIL`: cut the last layer short even with MTP. `TONA_MTP_PF_MEGA`: MTP prompt entries through megakernel passes instead of GEMMs. `TONA_HOST_TRACE` file: host-side timestamps of a decode cycle (`tools/host_trace_report.py`). `TONA_HEARTBEAT` file: one-line heartbeat read by `scripts/gpu-watch.sh`.
- `TONA_DECODE_DIAG`: counter differences after each request log line. `TONA_COPY_PROF`: one line per stash and unstash. `TONA_PF_DEBUG`: prefill timing per prompt. `TONA_PF_CHECK`: check MoE experts and scatter rows against the counts (three syncs per step). `TONA_PF_HASH`: FNV-1a hash of each chunk's residual stream per layer (about 3 x slower; the first differing line of two runs names the layer).
- `TONA_PF_UNFUSED`: combine and norm as two kernels. `TONA_PF_FULL`: also compute the last layer's unread output. `TONA_PF_FP8_SKIP` (`rows:in[:type][,...]`): keep these shapes on f16. `TONA_PF_FP8_STATS`: FP8 conversion error stats (synchronises). `TONA_PREFILL_IDX_SCALAR`, `TONA_PREFILL_ATTN_SCALAR`: scalar instead of WMMA kernels. `TONA_DN_KERNEL=old`: block-synchronized DeltaNet kernel. `TONA_GEMM_V1`, `TONA_GEMMQ_V1`: first-generation GEMMs. `TONA_GEMM_LOG`: log every f16 GEMM launch. `TONA_MATVEC_V1`: generic BF16/Q6_K matvec. `TONA_GPU_INT8`: `gpu::Forward` int8 matvecs.
- `TONA_DRAFT_LOG`, `TONA_TOKEN_LOG` files: per-pass draft log; context and accepted tokens per request. `TONA_DUMP_DIR`: write every `/v1/messages` body there.
- `TONA_ALLOC_TRACE` file (`%p` = PID): one line per device or pinned allocation, free and borrowed range (`tools/alloc_trace_lookup.py TRACE ADDRESS` maps a fault address to its buffer). `TONA_KFD_SMI_LOG` file (`%p` = PID): KFD events of this process (eviction with trigger, restore, unmap, migration, GPU reset, VM fault), no root needed; set by `TONA_DIAG` ([research/58](../../research/58-vram-instant-giveback.md)).

Test hooks:

- `TONA_TEST_FREEZE_MS` (build `-DTONA_WAIT_DIAG=ON`): waits at barrier `TONA_TEST_FREEZE_GEN` (300) in launch `TONA_TEST_FREEZE_LAUNCH` (20) stand still this long, at most 1000 (the driver resets the GPU after 2 s); `TONA_TEST_FREEZE_ALL=1` freezes all blocks. `TONA_TEST_EVICT_MS` (same build): the host answers every eviction-credit request with this many ms; the start is refused unless `TONA_MEGA_BUDGET_MS` plus it is below 1000. `TONA_TEST_STALL_MS`: stall the CPU once at layer 10.
- `TONA_TEST_STASH_RESERVE_GB` (8): RAM that must stay free after a stash; 0 reproduces the stash-copy hang. `TONA_TEST_STASH_STALL_MS` with `TONA_TEST_STASH_STALL_DIR` (`stash` or `unstash`): the first copy waits for a host word set after this many ms (`-1`: never). `TONA_TEST_COPY_GRACE_S` (60): shorter grace after a stalled copy. `TONA_TEST_STASH_STALL_SAFE`: with `-1`, set the word right before exit code 75.

Metal probes:

- `TONA_METAL_PROFILE`: GPU time per part of a pass (slows passes; never for speed results). `TONA_METAL_CB_LOG`: per prefill chunk command-buffer count, longest (must stay under 0.5 s), idle and encode times. `TONA_GPU_GAPS` (N): split GPU idle time between decode command buffers, a line per N passes. `TONA_METAL_PICK_CHECK`: run the host scan too and print mismatches. `TONA_METAL_MVG_SPLIT` (S = 1 to 4): spread expert rows over several threadgroup rows; undecided, 1 and 2 slower. `TONA_METAL_PIN` (`rset`, `mlock`, `mlock-dense`): keep mapped weights resident; no gain under memory pressure. `TONA_METAL_PIN_OWN`: backend buffers in a residency set (wires about 0.17 GB). `TONA_PLE_STATS`: time the n-gram row copy and count major faults.

`tona-gpu-gen` and test tools (statistics on stderr, text on stdout):

- `TONA_MEGA_F32` (`--mega`): float matvecs. `TONA_GPU_INT8` (`--teacher`): label the run int8. `TONA_DFLASH_CPU`, `TONA_DFLASH_CHECK`, `TONA_DFLASH_FIXED`, `TONA_DFLASH_EMA`, `TONA_DFLASH_CALIB` (`--dflash`): CPU reference drafter, CPU/GPU draft comparison, fixed draft length, moving-average pass times, calibration print. `TONA_PROMPT_NB` (4), `TONA_DRAFT_PROBE`, `TONA_DRAFT_SMALL`: prompt tokens per pass, extra probe draft, small block after non-full acceptance.
- `--teacher --mega`: `TONA_TEACHER_NB` (1) tokens per pass, `TONA_TEACHER_ACC` accept 1 to 4 in turn, `TONA_TEACHER_TAIL` N tokens teacher-forced through the megakernel, `TONA_TEACHER_SPECIAL` control tokens to ids, `TONA_TEACHER_SPLIT` head checkpoint at a token, `TONA_TEACHER_CUTS` prefill cut points (equal logits under `TONA_DETERMINISTIC=1`), `TONA_TEACHER_NLL` file of "position -log p" (`scripts/nll-pair.py`; `--check N` compares the CPU reference), `TONA_TEACHER_DUMP` -log p and logits per tail token (about 1 MB each), `TONA_TEACHER_REF` compare with such a dump (`TONA_TEACHER_OUT` per-token lines for `scripts/nll-compare.py`).
- `--batch-bench`: `TONA_BENCH_NB` (0: widths 1 to 4). `--batch-check`: `TONA_BATCH_NB` (4), `TONA_BATCH_REF_NB` (1). `--multi-check`: `TONA_MULTI_NB` (2), `TONA_MULTI_B` second-conversation file, `TONA_MULTI_DUMP`, `TONA_MULTI_HALF` (f16 rows, bitwise comparison meaningless), `TONA_MULTI_SWAPS` (2).
- `TONA_CHECK_T` (150): `tona-gemm-check` tokens per expert. `TONA_HC_NC` (64), `TONA_HC_MULT` (1), `TONA_PROBE_BLOCKS_PER_CU` (2): `tona-wide-probe` settings. `tona-expert-check` sets `TONA_Q8_LANES=8` for itself.
