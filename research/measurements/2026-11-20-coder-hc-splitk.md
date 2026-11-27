# Coder hyper-connection matvecs: split-K across waves (#175)

Date: 2026-11-20. Branch `worktree-<worktree>`, base `9d84a36`. Builds: `work/build/hcs` (this branch), `/var/tmp/claude/hcs/base-src/work/build/base`
(base, a detached worktree of the base commit). Logs `/var/tmp/claude/hcs/`, ab.py output `work/ab/hcs-*`. The maintainer asked to **check** the lever
(measure, do not enable by default): `TONA_HC_SPLITK=S`, default off.

## Starting point

`2026-11-20-coder-decode.md`: gr.down + gr.up are 12 % of a pass at 2K (4.7 ms for 194 steps), at 44 / 46 % of the bandwidth roofline: 24 us per step against 11 us
of bytes. `2026-10-25-hc-matvec.md`: a per-lane chain of sub-block loads (one latency after the other), about 10 us fixed cost per matvec step. Estimate of the profile:
halving the sequential sub-blocks per lane saves about 8 us per step, 1.5 ms per pass, +3.5 % decode (unverified).

Shapes (from the code, `matvec_q8_row.h`, `mega_upload.cpp`): gr.down 320 rows x 10240, Q8_0, 320 sub-blocks per row, 32 lanes per row, one row per wave, 320 units
on the grid of 32 blocks x 16 waves = 512 waves (the log line of block 1 says 32 blocks; an earlier reading of the code suggested about 62, wrong): 320 of 512 waves work, each lane runs 10 sub-blocks in 5 rounds of 2 (`ROWQ8_G`).
gr.up 10240 rows x 320, 10 sub-blocks per row, 2 lanes per row, 16 rows per wave, 640 units, 5 sub-blocks per lane in 3 rounds.

## Design

`DevTask::ksplit = S` (2, 4 or 8): unit = (row, slice), S consecutive waves of a block compute S slices of the sub-blocks of one row, slices 1..S-1 put their sum
into LDS, one block barrier, wave 0 adds in slice order (deterministic) and applies the epilogue (silu). Same per-vector order for every token: width independent.
Only for the gr.down task (first Q8_0 task of its phase, 32 lanes, phase without x in LDS: checked at program upload, `throw` otherwise).
The code exists only in kernel instantiations `Qwen4ExpSplitProfile` (traits `hc_split`, chosen per task array like the Q6_K tile kernels): without the switch no kernel
contains it and the machine code is the base's (checked by `scripts/isa-diff.py`).
gr.up is not split: K = 320 has 10 sub-blocks, a split gives 1280 units on 512 waves (some waves two units, 2 x 2 rounds against 3 now) and its x already sits in LDS.
It is judged by a lanes-per-row sweep instead (`TONA_MEGA_LANES_MAX`), numbers below.

## Prior art

- [Accelerating a Triton Fused Kernel for W4A16 Quantized Inference with SplitK work decomposition, arXiv 2402.00025](https://arxiv.org/abs/2402.00025), status: read (abstract).
  Split-K with atomic reduction for M = 1 to 16 skinny GEMMs where K >> N: +65 % (A100), +124 % (H100). Adopted: the idea of more parallel units along K when N is small.
  Not adopted: atomics (our epilogue silu needs the full sum, and an atomic order would make rounding run-dependent): LDS reduction in a fixed order instead.
- [ROCm blog, Accelerating LLM Inference on AMD GPUs with Low-Latency GEMMs (June 2026)](https://rocm.blogs.amd.com/software-tools-optimization/accelerating-llm-inference-on-amd-gpus-with-low-latency-gemms/README.html),
  status: read (search summary only): intra-CTA K-slice splitting and inter-CTA split-K to fill the grid for small-M GEMMs. Same mechanism as here (intra-block slices).
- [Decompose-K (June 2026 blog)](https://shreyansh26.github.io/post/2026-06-21_decompose-k-triton/): split K into S partial GEMMs and sum; for tiny M, N and very large K.
- [vLLM #34709](https://github.com/vllm-project/vllm/pull/34709) and #34176/#34177 (merged, `research/papers.md` 2026-02-17 row, body read there): the `wvSplitK` skinny GEMM kernels (M = 1..4) enabled on gfx11/gfx12, about 15 % more decode tok/s on a gfx1201 (R9700), BF16/FP16 GEMMs. The same mechanism on the same chip generation; ours is Q8_0 with int8 dot products in a persistent megakernel, so the gain does not transfer one to one (the authors' 15 % is for whole models with large GEMVs). vLLM #62799 (open): wvSplitK reaches about 600 GB/s at 1-5 rows.
- Stream-K (arXiv 2301.03598): even division of K iterations over SMs; judged: our per-row work is too small (320 sub-blocks) for its fixup overhead; a fixed S is simpler.
- llama.cpp mmvq (CUDA/HIP): one block per row or per few rows with several warps along K and a shared-memory reduction (PR 17156 and others in the search), status: merged,
  fixed warp count per architecture; no split-K as a separate mode. Our megakernel has no such per-launch choice, the split is a task property.
- Own earlier study `2026-10-25-hc-matvec.md`: deeper unrolling and preloading per lane rejected: the latency chain stays.

Search: web search (two queries: split-K GEMV / mmvq on RDNA; Stream-K / split-K skinny decode), `research/papers.md` and `papers-screened.tsv` grepped (split-k, stream-k: vLLM wvSplitK rows above, mlx qvm split-k titles), window 2024..2026-11-20;
the topic is a kernel mapping detail, not covered by `tools/papers/topics.toml`; the weekly paper scan is the coordinator's.

## Test conditions

Written into the job logs `/var/tmp/claude/hcs/<job>-hw.txt` (`python3 tools/hw_report.py`) and `<job>-quiet-{before,after}.txt` (`python3 tools/quiet_check.py`); results below name the
state as sums only. The desktop holds some VRAM; the server's slot count is noted per start.

## Build checks

`scripts/isa-diff.py base hcs`: 12 kernels added (the split-K instantiations `Qwen4ExpSplitProfile` and `Qwen4Exp512SplitProfile`: 3 widths x 2 Q6 layouts each; VGPRs 192 like the others, spills
within the range of the base kernels); every existing kernel has the same instructions, register, LDS and spill counts. The only differences are the immediate offsets of
`s_add_co_u32` (relocations of globals that moved because the object grew), checked by normalising those: 0 real differences in 442 kernels. So with the switch off the machine code is the base's.

## Block 1 (plan, written before its run): step time and pass time per S

**Question.** Does `TONA_HC_SPLITK` = 2, 4, 8 lower the time of the gr.down step, and by how much does the pass shorten at widths 1 and 4, with the numbers checked against the single-step reference
(`--batch-check`, bitwise equal across widths is the property to keep)? gr.up is not split (see design); its step time per run is the control.
**Run.** One `exclusive` job: (a) `tona-gpu-gen MODEL PROMPT --mega --batch-check 100` with `TONA_HC_SPLITK=2` (width independence: the verify pass against single steps),
(b) per S in {off, 2, 4, 8}: `tona-gpu-gen MODEL scripts/bench-long-code.txt --mega --batch-bench 300` with `TONA_BENCH_WIDTHS=1,4 TONA_BENCH_REPEAT=3 TONA_MEGA_PROF=1
TONA_EXPERT_PROFILE=work/profile/experts-coder.bin` (as `2026-11-12-verify-pass.md`: real experts from RAM, per-step stamps of block 0, warm-up of 64 tokens per width, 3 rounds
interleaved, spread = min..max). Why this size: the step time is deterministic per pass, 300 passes per width give the mean to well under 1 %; widths 1 and 4 are the bounds of the
decode pass (the server's passes are 3 to 5 wide). Expected duration about 5 min per S after the model load.
**Expectation (source: `2026-11-20-coder-decode.md`, estimate of 8 us per step).** gr.down per step 24 -> 16 us (S = 2: 3 rounds of the sub-block chain instead of 5, plus one LDS barrier of about 1 us;
S = 4: 2 rounds, 320 -> 640 units on 512 waves: 128 waves get two units, so already S = 2 has an imbalance); pass 1 token about -0.8 ms of 24.7 (-3 %), 4 tokens about -0.8 ms of 37 (-2 %); tolerance +-50 % of the saving.
gr.up and everything else: unchanged within 1 %. **Decision fixed before.** The S with the largest pass-time gain beyond the spread goes to the speed block; no S faster than the spread: stop, report the lever as not worth it
(quality block not run, nothing is enabled by default either way). **Active tooling.** `TONA_MEGA_PROF=1` (all arms), `TONA_EXPERT_PROFILE`; no `TONA_DETERMINISTIC`, no rocprof.

### Block 1 result (job `hcs-b1`, `/var/tmp/claude/hcs/b1-*.log`, summary script `/var/tmp/claude/hcs/sum1.py`)

Quiet check before 09:46: warning only for 3.6 GiB VRAM held by others; after: see `b1-quiet-after.txt`. Server-less run: `tona-gpu-gen --mega`, 32 blocks, expert cache 64 per layer, VRAM 11.35 GB.
Correctness: `--batch-check 100` with S = 2: 136 of 136 top-1 equal, worst relative deviation 0.00e+00, **bitwise equal across pass widths in 100 of 100 positions** (the split is per token, width independent).
Mean of 3 interleaved rounds [min..max]; gr.down and gr.up in us per pass (97 steps each), pass in ms:

| S | width | pass ms | gr.down us (per step) | gr.up us |
|---|---:|---|---|---|
| off | 1 | 24.06 [23.92..24.33] | 1,837 [1834..1840] (18.9) | 1,672 |
| 2 | 1 | 24.31 [24.27..24.35] | 2,049 [2047..2051] (21.1) | 1,694 |
| 4 | 1 | 24.49 [24.46..24.55] | 2,194 [2190..2201] (22.6) | 1,716 |
| 8 | 1 | 24.43 [24.40..24.49] | 2,168 [2164..2172] (22.3) | 1,716 |
| off | 4 | 35.80 [35.78..35.83] | 2,358 [2357..2359] (24.3) | 2,265 |
| 2 | 4 | 36.85 [36.84..36.86] | 2,577 [2575..2578] (26.6) | 2,307 |
| 4 | 4 | 37.09 [37.07..37.10] | 2,816 [2815..2817] (29.0) | 2,316 |
| 8 | 4 | 37.33 [37.33..37.34] | 3,057 [3055..3061] (31.5) | 2,328 |

**Result: every split is slower, outside the expectation (-8 us per step expected, +2 to +7 us measured at width 1, +2.3 to +7 us at 4). Pass +1.0 to +1.8 % at width 1, +2.9 to +4.3 % at width 4.** The spread of the rounds is
below 0.5 % of the pass, so the sign is certain. gr.up (not split) rises by 1 to 3 % in the split arms too: a side effect of the extra LDS barrier phase / kernel instantiation (the split kernel is a different
compilation of the same code; its spill counts differ slightly).
Reading (from the code and the numbers, not measured separately): with 512 waves, gr.down already puts 320 rows on 320 waves in one round of 5 sub-block rounds. S = 2 gives 640 units on 512 waves, so 128 waves run two units of 3 rounds each = 6 sequential
rounds against 5, plus one block barrier per round; S = 4 and 8 give 1,280 and 2,560 units (3 and 5 rounds of units per wave) and their per-unit chain is 2 and 1 rounds. The best case in this grid, S = 1.6 with perfect balance, would be 6.25 sub-blocks per lane
= 4 rounds against 5: at most one round (about 3 us) per step, before the barrier cost. The profile's estimate of 8 us rested on an idle grid of about 990 waves; the real grid has 512.
Decision fixed before: no S faster than the spread: **stop, speed and quality blocks (ab.py at 2K/16K/64K, Δ-NLL, KL, code suites) not run**: a change that makes the pass slower has no quality question worth the GPU time (docs/contribute/measurement.md principle 4: cheapest recipe first, stop at the first failure).

## Not taken, with the record (docs/contribute/measurement.md principle 8)

Not taken because it is 1 to 4 % slower at every S. Searched 2026-11-20 on this exact problem (split-K for batch-1/skinny quantized matvec on RDNA; Stream-K for small M; web search plus `research/papers.md` and
`papers-screened.tsv`): the found work (vLLM wvSplitK on gfx1201, W4A16 SplitK Triton, ROCm low-latency GEMMs blog) targets GEMVs with far more work than waves, which our 320-row, 3.3 MB matrices on a 512-wave persistent grid are not;
nothing newer found for a persistent megakernel. Tried: S = 2, 4, 8 (block 1); the cost that remains, the unit imbalance on a 512-wave grid, cannot be removed by a cheaper layout: an S that divides the 16 waves of a block cannot divide 512 waves by 320 rows evenly,
and a split across blocks needs a grid-wide exchange (a barrier step of 5 us or more, the size of the whole gain). gr.up (10 sub-blocks per row) has no K left to split; its lanes-per-row choice (2, 3 rounds) is already the balanced one for 640 units on 512 waves in one round.
The code stays in the branch behind `TONA_HC_SPLITK` (default off, separate kernel instantiations, existing kernels unchanged by `isa-diff`) so the result can be reproduced; the maintainer may also drop it.

## Gates (build `hcs` = this branch, switch off)

`just check` rc 0 (incl. clang-tidy; first attempts failed on CCN 16 of `dev_task`, cognitive complexity and function size of the split-K loop: fixed by splitting, no exceptions); `ctest -L unit` 71 of 71 (own build dir `work/build/hcscpu`);
`just test-py` 681 passed, 1 skipped (outside a queue job); `test_gpu_fattn`, `test_gpu_attn_wmma`, `test_gpu_mamba`, `test_gpu_moe_latent`, `test_gpu_bolt_quant` rc 0;
`scripts/gpu-status.sh --own` (golden gpu, cpu plan, prefill, prefill with dflash, server with MTP): **all identical** to the references of the main checkout. `isa-diff` base against branch: 442 kernels with identical code, registers, LDS and spills (only
relocation offsets of globals differ, normalised), 12 added split-K kernels. Pressure series (C8 note of gpu-status) not run: the change has no host-memory effect with the switch off.
Trap noted: a worktree needs `work/golden` linked to the main checkout; an empty one makes every golden "DIFFERS" against a missing reference.
