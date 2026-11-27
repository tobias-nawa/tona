# Ornith prefill outside the attention (#175)

Date: 2026-11-20. Branch `worktree-<worktree>`, base `df9e5ef`. Build `work/build/o20`, logs `/var/tmp/claude/o20/`.
Starting point: `2026-11-19-prefill-attention.md` (Ornith prefill 2,270 tok/s at 2K, 4,051 at 16K, 2,829 at 64K after the attention
kernel; attention about 57 % at 64K), `2026-11-19-expert-profiles.md` (prefill@2000 +20 % with the shipped profile, expert counters
unchanged, cause open).

## Test conditions (before the first run, 04:31)

`python3 tools/hw_report.py`: Ryzen 9 9950X (16C/32T), RX 9060 XT gfx1200 15.9 GiB, RAM 60.5 GiB (24.0 GiB available), kernel
7.3.4, hip 7.16, no `TONA_*` set, commit `df9e5ef` plus this branch. `tools/quiet_check.py`: **do not measure** (other programs
71 % of the CPU threads, load average 37, 14.95 GiB of the VRAM held by others, the GPU 99 % busy): other contributors' jobs
run on the machine right now. No number of this file comes from that state; every run below is a queued job and gets its own
quiet check before and after (written next to its result).

## Plan 1: where the Ornith prefill goes (written before the run)

**Question.** Per kernel family, how many milliseconds per prompt token does the Ornith prefill spend at 2K, 16K and 64K, and
how far is each family from its roofline (WMMA 93 TFLOPS f16, 315 GB/s VRAM, PCIe 55 GB/s)? Which family is the next lever?

**Scope.** One `exclusive` job: `scripts/prof_prefill.py` starts `work/build/o20/tona-server` (Ornith Q4_K_M, `--ctx 262144`
as the service, the shipped expert profile as `ab.py`) under `rocprofv3 --kernel-trace --memory-copy-trace` (rocpd database),
asks for a discarded 2K prose prompt (GPU clock ramp), then 64K, 16K, 2K code prompts ("sum", descending so that no prompt
extends a cached one), 4 decode tokens each. One start is enough: the kernel set and its times are deterministic to <1 % in the
same warm state (`2026-11-16-prefill-breakdown.md`). `TONA_PF_DEBUG=1`, `TONA_MEGA_INFO=1`, `TONA_VRAM_LOG=1` give the plan
(chunk, group, layered or not), the pinned experts and the slot count. About 12 minutes (load 2 min, prefill 64K 40 s under the
profiler, conversion a few minutes).

**Expectation (source: the per-token costs of `2026-11-19-prefill-attention.md` Result 2 and the weights per token: experts
8 x 3 x 2048 x 512 per layer, shared expert 3 x 2048 x 512, DeltaNet layer 33.6 M weights, attention layer 29.4 M, 30 DeltaNet and
10 attention layers; 5.0 GFLOP per token, 18,600 tok/s at the f16 peak).** Prefill rate inside the trace within 10 % of the
unprofiled 2,270 / 4,051 / 2,829 tok/s. Shares (+-8 points): at 64K attention 55 %, expert GEMM 15 to 25 %, DeltaNet 8 to 15 %,
dense GEMM 5 to 10 %; at 16K attention 20 to 25 %, expert GEMM 30 to 40 %; at 2K the PCIe copies of the experts that are not in
the VRAM cache set the pace (21 GB of experts, half in VRAM: about 0.2 s at 55 GB/s of 0.87 s) and the idle gaps are large.
A family below 30 % of its roofline is a lever; a family at 60 % or more is not.

**Active tooling.** `rocprofv3 --kernel-trace --memory-copy-trace`, `TONA_PF_DEBUG=1`, `TONA_MEGA_INFO=1`, `TONA_VRAM_LOG=1` (logs
only). No `TONA_DETERMINISTIC`. The rate of a profiled run is not a measurement; the shares are.

**Decision.** Levers are ranked by (time above the roofline) x (feasibility without changing numbers).

## Result 1: the kernel profile (job `o20-prof2`, build `o20` at `df9e5ef`; `tona-bolt-bench` under `rocprofv3 --kernel-trace --memory-copy-trace`, one process per prompt size, 3 prefills each, the last one analysed; quiet check before and after in `/var/tmp/claude/o20/prof2/quiet-*.txt`)

First job (`prof_prefill.py`, the server under the profiler) produced no database: the server has no signal handler and dies on SIGTERM before the
profiler writes. The bench tool exits normally, so it is the profile vehicle (one `Mega::prefill` call of N tokens from position 0, `--ctx 262144`,
shipped expert profile, 83 expert slots per layer of 256, all 21.1 GB of experts pinned). Rates inside the traces (profiled, **prefill**):
2K 2,413, 16K 4,315, 64K 3,207 tok/s (server run of the same builds: 1,934 / 3,877 / 3,187 tok/s).

Per kernel family, ms per prefill (idle gaps are the window minus the union of kernel intervals):

| Family | 2K (0.83 s) | 16K (3.61 s) | 64K (19.85 s) |
|---|---:|---:|---:|
| Expert GEMM (`k_gemm_q2`, Q4_K gate/up, Q6_K down) | 166 (20 %) | 882 (24.4 %) | 3,480 (17.5 %) |
| Dense GEMM (`k_gemm2`, f16 WMMA, weights from the cache) | 117 (14 %) | 849 (23.5 %) | 3,438 (17.3 %) |
| DeltaNet (`k_dn_wave` + `k_dn_conv_l2` + `k_dn_norm_silu`) | 87 (10.5 %) | 698 (19.3 %) | 2,785 (14.0 %) |
| Attention (`k_fattn_prefill_wave`) | 11 (1.3 %) | 475 (13.2 %) | 7,371 (37.1 %) |
| `k_moe_sum` | 22 | 175 | 704 |
| `k_res_norm16` | 15 | 121 | 486 |
| `k_qsa_prep` | 14 | 115 | 462 |
| `k_scatter` + `k_swiglu16` + route + dequant | 31 | 193 | 600 |
| **Idle gaps (no kernel running)** | **372 (44.8 %)** | **136 (3.8 %)** | **506 (2.6 %)** |
| Copies to VRAM (traced) | 12.9 GB, 0.26 s | 15.8 GB | 48 GB (3 rounds) |

Against the rooflines (FLOP from the tensor list: 1.43 G dense and 1.01 G expert weights per token; WMMA 93 TFLOPS):
at 16K the experts need 32 TFLOP = 0.35 s (**39 %** of the peak reached), the dense GEMMs 46 TFLOP = 0.49 s (**58 %**), attention
21 TFLOP = 0.23 s (49 %), DeltaNet 3 TFLOP of FP32 (about 0.35 us per token at the VALU limit against 0.86 us measured, the
recurrence is token by token, see below). `k_moe_sum`, `k_res_norm16`, `k_dn_conv_l2`: memory bound, `k_moe_sum` is at the
315 GB/s rate (47 GB moved at 16K = 0.15 s against 0.175 s).

Findings from the profile:

1. **Short prompts (up to 2,048 tokens) are not compute bound.** At 2K 45 % of the window has no kernel running (0.37 s of 0.83 s); the
   trace shows batches of experts waiting 300 to 630 us each for their weights: the chunk-by-chunk plan copies the missing experts
   (173 of 256 per layer, 2 MB each) batch by batch on a copy stream with two buffers, and a batch computes in 100 to 230 us but its
   copy takes 300 to 600 us at 55 GB/s. The copy of layer l + 1 starts only when layer l + 1 reaches its MoE. The layered plan issues the
   copies of the whole next layer ahead of the routing (`issue_layer`), which the plan takes only above one chunk (2,048 tokens).
2. Above 16K the pass is compute bound (idle 2.6 to 3.8 %). The families above the roofline, in ms at 16K: expert GEMM 530,
   DeltaNet about 400 (bit-exact levers bounded by the sequential recurrence), dense GEMM 360, the memory-bound element-wise kernels
   (`k_moe_sum`, `k_res_norm16`, `k_qsa_prep`, `k_scatter`, `k_swiglu16`) 560 together.
3. The MTP entries of the prompt cost 15 ms at 2K (the `MTP included` counter of the server log is cumulative over the start, not
   per request): not a lever.

## Result 2: the profile finding (job `o20-prof4`, `o20-exp1`; build `o20`)

**Cause of "expert profile: prefill@2000 +20 %, hit counters unchanged": PCIe bytes of the prefill.** The shipped profile fills the 83 VRAM
expert slots per layer (of 256) with the experts the router picks most often. A 2K prompt uses almost every expert of every layer, and the
chunk-by-chunk plan copies each expert that is not in VRAM (1.9 MB) over PCIe: traced copy volume per 2K prefill **12.87 GB with the profile,
18.97 GB without** (window 0.83 s against 1.04 s in the profiled runs, `o20-prof2` against `o20-prof4`). The hit and miss counters of `/status`
count the decode, not these prefill copies, which is why they hardly moved. The effect is the first request only in the sense that the decode of a
request moves the cache towards the decode usage afterwards; the slots are then no longer the profile's. Unprofiled warm rates (`tona-bolt-bench`,
2K, rounds 1 and 2): profile 3,260 / 3,346 tok/s, no profile 2,774 / 2,785 tok/s (+17.5 %). A second, separate effect: round 0 of every process
is 15 to 20 % slower than rounds 1 and 2 in every configuration (2,713 against 3,260 with the profile): the GPU clock ramp of the first prefill
after idle (`2026-11-12-prefill-bimodal.md`), not the profile. Lever for every request: the prefill of up to about 2K tokens is bound by
PCIe bytes (12.9 GB at 50 GB/s = 0.26 s of a 0.61 s prefill), so every expert that stays in VRAM saves 36 us per layer; a cache that follows the
prefill's own usage instead of the decode's is the open idea (not built).

## Result 3: layered plan for short prompts and a larger layered chunk, measured, **not shipped**

Both levers use the layer-by-layer plan. After the measurements the coordinator reported that the layered plan with **more than one chunk** computes a
wrong network (each layer reads its input norm from the `B.mixed16` that another chunk wrote; `2026-11-20-bolt-determinism.md` of the Bolt worker).
My own data show the same symptom on Ornith: at 8,000 tokens (`o20-exp4`, build `o20`) the output hash differs between the two-chunk plan (chunk 4096
or 6144 requested, plan chunk 4096) and the one-chunk plan (chunk 8192), in normal mode **and with `TONA_DETERMINISTIC=1`** (5caaebe6a7dfaa9b against
ab7fd4fd5266e252, each stable over three runs). One-chunk plans agree with the chunk-by-chunk plan (equal hashes at 1,500, 2,000, 3,000 and 4,500 tokens
in `o20-exp3`, where the new plan only changed the chunk size or the plan type). Consequence for the numbers below: **every layered run with more than
one chunk is a run of the faulty path, its speed is not a result.** The one-chunk layered runs are not affected by that data flow but are listed as
layered too. Plan lines: `Prefill N tokens: layer by layer|chunk by chunk, chunk C, ...` in the `.out` files of the jobs.

| Run | Plan | tok/s (**prefill**, warm, mean of rounds 1 and 2) | Valid? |
|---|---|---|---|
| `o20-exp1` a, 2K, profile, default | chunk by chunk | 3,260 / 3,302 | yes |
| `o20-exp1` b, 2K, `TONA_PF_LAYERED_MIN=1`, profile | one chunk, layer by layer | 3,612 / 3,621 (+10 %) | one chunk: data flow not affected; hash equal to chunked at this size in `o20-exp3` |
| `o20-exp1` b, 1K / 500 | one chunk, layer by layer | +3 % / -7 % | as above |
| `o20-exp3` build `o21` (short prompts layered, chunk = prompt), 1.5K / 2K | one chunk, layer by layer | 3,328 against 2,835 (+17 %), 3,765 against 3,311 (+14 %), 3 repetitions, hashes equal to `o20` | one chunk |
| `o20-exp3` 3K, 4.5K | one chunk, layer by layer in both builds (chunk 6144 against prompt size) | +1.6 %, +0.4 % | one chunk |
| `o20-exp2` and `o20-exp3` 8K, 16K, 64K, chunk 4096 / 6144 / 8192 | more than one chunk (16K, 64K), one chunk at 8K with 8064 | 16K 4,328 / 4,408 / 4,462; 64K 3,173 / 3,243 / 3,286 (chunk 8192 +1.2 % and +1.3 %) | **no: faulty path** |

The code that made the short-prompt and chunk-8192 levers (a `PrefillSequence::short_layered_min()` and `layered_chunk()` hook, Ornith 1,024 and
8,192, chunk of the layered plan clamped to the prompt) is **not in the branch**; it is kept as `/var/tmp/claude/o20/layered-hook.patch` (not in the
repository) and is to be re-measured on top of the fix (`PrefillSequence::resume` re-norms per chunk) when the coordinator says it is merged. A third
observation for that re-measurement: the layered copy of every missing expert (`issue_layer`) moves 15.6 GB at 500 tokens against 12.9 GB chunk by
chunk (it copies before the routing is known, also experts no token uses), which is why layered loses at 500 tokens; copying only what the routing
needs is not possible ahead of the router.

Copy mechanics (`/var/tmp/claude/o20/pcie.hip`, job `o20-pcie`, not in the repository): 173 separate 1.9 MB `hipMemcpyAsync` calls on one stream take
7.0 ms (47 GB/s), on two or more streams 6.0 ms (54.5 GB/s), one big copy 5.9 ms (55.7 GB/s), a copy kernel the same 5.9 ms; host enqueue time is
negligible (0.08 ms). So the per-expert copies are within 15 % of the link rate and a second copy stream would give +16 % on the copy part at most.
In the profiled traces the copies appear spaced at 74 us instead of 35 us: the profiler's overhead (unprofiled layered 500 tokens: 0.41 s against
0.65 s profiled), not the engine.

## Estimates for the numbers-changing levers (not built)

At 16K the families above their roofline (profile table): expert GEMM 0.88 s against 0.35 s, dense GEMM 0.85 s against 0.49 s. An int8 MMQ route
(llama.cpp PR 17156 for RDNA4: block-scaled int8 activations, iu8 WMMA at 186 TOPS, scales applied per 32-value sub-block) can at best halve the
WMMA part; its epilogue (int32 to float and a scale FMA per sub-block and output tile) and the nibble unpack stay VALU work, so I estimate
**0.3 to 0.4 s of 3.6 s (8 to 11 %) at 16K and about 6 to 8 % at 64K for the experts, and about the same again for the dense GEMMs**, at the cost of a new
kernel family per quant type (Q4_K, Q6_K) and of the quality check of `docs/backends/README.md#checks` (the decode path already uses int8 activations: mean
|d nll| 0.167 against 0.123 for the f16 prefill, so int8 prefill would land between them; FP8 for the dense GEMMs measured 0.216 on 117 tokens and
gave only +2.7 % at 16K in `2026-11-19-ornith-prefill.md`). Not built: the bit-exact levers are not exhausted, but each is small (below), and this one
is a maintainer decision with a large build. The open bit-exact levers and their sizes at 16K:

| Lever | Bound | Size |
|---|---|---|
| DeltaNet `k_dn_wave` (sequential per token, 2,500 cycles per token against about 1,000 at the VALU limit) | at most 2.5x on 0.47 s | up to 0.28 s, realistically 0.1 s (3 %) |
| `k_dn_conv_l2` to the memory rate | 0.16 s against 0.10 s | 0.06 s (1.6 %) |
| `k_qsa_prep` to the memory rate | 0.115 s against about 0.05 s | 0.06 s (1.8 %) |
| dense and expert GEMM to 75 % of the sustained peak (tiling, double-buffered unpack with TT 64) | 0.85 and 0.88 s | 0.2 to 0.3 s together (6 to 8 %), a kernel project |
| idle gaps at 16K (host sync per layer for the routing counts) | 0.14 s | 3.8 %, needs device-side grouping |

## Status

Branch holds the profile and the explanation only (this file); no engine change is committed. Open: the layered-plan levers after the fix (patch
above), the DeltaNet, conv and prep kernels, the dense/expert GEMM tuning, the int8 route (maintainer). Final A/B with `scripts/ab.py` was not run
because no change to the default exists. Test conditions of the runs: quiet check before `o20-exp1` (04:55): warning, load average 11.7, 2.61 GiB of
VRAM held by others (the desktop); the other jobs ran in exclusive slots of the queue, their quiet checks are in `/var/tmp/claude/o20/*/quiet-*.txt`.
