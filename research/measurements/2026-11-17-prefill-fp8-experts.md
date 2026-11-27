# Prefill: expert GEMM in fp8 (#159, lever 1): prior art, micro test, plan of the quality check

Change type: technique (not shipped; the experimental kernel lives on branch `159-fp8-micro`, not on `159-lever1`).

## Prior art

- **Our own int8 expert attempt (#100)**, `research/measurements/2026-11-02-prefill-int8.md` (Run 2 and "Finding: the 8-bit
  route does not pay for the experts"), status: stopped, code on branch `100-int8-experts-archive`. Int8 WMMA with the
  weights unpacked into the LDS and the scale applied per K block of 32 in the accumulator ran at **0.76 to 1.20x** the f16
  `k_gemm_q2` (type 18: 0.99 to 1.03x, 20: 1.17 to 1.20x, 22: 0.76x, 23: 1.05 to 1.10x, 42: 1.02 to 1.07x). Ablation of
  the f16 kernel (type 18, 32 x 64 tokens, 0.290 ms): skeleton 0.081 ms, unpacking about 0.10 ms, WMMA about 0.11 ms,
  additive; the scale math of the int8 kernel (30 % of it) cost as much as the 8-bit WMMA saved. Adopted here: the idea that the
  scale must not be applied per K block in the accumulator; the fp8 variant below folds the scale into the unpacked weight
  value instead (fp8 keeps an exponent), so there is no scale math in the K loop.
- **Our dense FP8 prefill (#100, #106)**, `research/measurements/2026-11-10-fp8-quality.md`, status: shipped as the default
  (`TONA_PREFILL_FP8=1`) but **above the rule**: dNLL +0.0150 nat/token (se 0.0027 over the 25 per-turn means), top-1
  agreement 0.939, truncated KL 0.031; the relative error of the FP8 copy of the input is 2.4 to 2.6 % per GEMM (rounding
  noise of 3 mantissa bits). Consequence for this lever: today's baseline already uses the budget of the 0.01 rule, so the
  fp8 experts are judged as the dNLL against today's path, and every added noise source starts from a path that is above the rule.
- [Integer-Exact Microscaling, arXiv 2611.00486](https://arxiv.org/abs/2611.00486), status: abstract only
  (`research/papers.md`): MXFP4 weights as one INT8 GEMM; Qwen3.8-Flash-Next routed experts with INT8 activations KL to the
  FP8 model 0.0557. Not adopted (our weights are IQ grids, not MXFP4); a data point that INT8/FP8 activations on this model's
  experts cost a KL of that order.
- [pat883/w4a8-fp8-wmma](https://huggingface.co/pat883/w4a8-fp8-wmma) (Hugging Face), status: unmeasured reference code (W4A8-FP8
  dense and MoE GEMM for gfx1200). Not read in depth; no benchmarks to compare.
- [GPUOpen WMMA guide RDNA4 part 2](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/) and
  [vLLM forum: FP8 WMMA on RDNA4](https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900):
  FP8 WMMA runs at twice the f16 rate on gfx12; the forum reports up to about 2x prefill up to 10K tokens (single report).

Search: tools/papers/scan.py topics v1; topics: quantization, amd-rocm, moe-offload; terms: "fp8 moe rdna4", "w8a8 prefill",
"e4m3 wmma"; window 2025-11-17..2026-11-16 (12 months); sources: research/papers.md and research/papers-screened.tsv (grep "fp8";
`rdna4-fp8-moe` candidates vllm#36659 and #38086 of 2026-03, titles read only), the GitHub scan of 2026-11-11..15 of this task
(nothing new for RDNA4 grouped fp8 MoE), web search. No fresh `scan.py` run for this note (the screened list already covers
the topic); the gate's search line is met by the book of week 46.

## Micro test (no quality, speed only): `tona-gemm-q-bench`, real expert weights of the model, f16 against the fp8 kernel

Plan (written before the run): `k_gemm_q2f8` is `k_gemm_q2` with the unpack writing E4M3 bytes into the LDS (weight value times a
fixed scale 16 inside `cvt_pk_fp8_f32`), X converted from f16 to E4M3 in the kernel (fixed scale 1) and the fp8 WMMA
(`wmma_f32_16x16x16_fp8_fp8`); same tile, same K step 64, same epilogue. Speed only: the fixed scales are not a usable scheme.
Expectation: 1.2 to 1.4x on the GEMM (WMMA about 38 % of the kernel halved, LDS traffic halved, minus the extra converts); T = 100,
200, 320 tokens per expert (the range where the 128-token tile is used), 32 experts, 10 repetitions after a warm-up, one run
each. Active tooling: none (`TONA_GEMMQ_FP8` the only switch).

Result (ms per call, 64 / 128-token tile as the kernel chooses, f16 -> fp8, ratio f16 / fp8):

| Type | T = 100 | T = 200 | T = 320 |
|---|---|---|---|
| 18 (640 x 2560) | 0.460 -> 0.348 (1.32x) | 0.877 -> 0.654 (1.34x) | 1.148 -> 0.867 (1.32x) |
| 20 (2560 x 640) | 0.476 -> 0.359 (1.33x) | 0.890 -> 0.672 (1.32x) | 1.192 -> 0.904 (1.32x) |
| 21 | 0.479 -> 0.350 (1.37x) | 0.910 -> 0.646 (1.41x) | 1.207 -> 0.887 (1.36x) |
| 22 | 0.512 -> 0.355 (1.44x) | 0.987 -> 0.649 (1.52x) | 1.326 -> 0.917 (1.45x) |
| 23 | 0.433 -> 0.340 (1.27x) | 0.814 -> 0.620 (1.31x) | 1.065 -> 0.839 (1.27x) |
| 42 | 0.440 -> 0.324 (1.36x) | 0.842 -> 0.617 (1.36x) | 1.120 -> 0.827 (1.35x) |

So 1.27 to 1.52x (median 1.35x) on the GEMM, **including** the in-kernel conversion of X, unlike the int8 attempt (0.76 to 1.20x).
The fp8 kernel is at 0.32 to 0.36 ms for the 100-token case, where the skeleton alone was 0.08 ms.

Accuracy of the fixed-scale kernel (same dumps, `TONA_CHECK_DUMP`; X ~ N(0, 1), the first tensor of each type): relative rms
difference to the f16 output after undoing the factor 16: **3.7 to 3.75 %** per GEMM, max absolute error 0.22 to 0.50 of the
output rms, correlation 0.9993. That is the rounding noise of two 3-bit operands, no flush or saturation visible on this sample;
it is above the 2.4 to 2.6 % of the activation-only fp8 of the dense path.

## What this is worth, and the plan of the quality check

- Expert GEMM share of the 130K window: `k_gemm_q2` 16.4 s of 72.0 s = **22.8 %** (profile run5). At the micro ratio of 1.27 to 1.52x
  the GEMM would save 3.4 to 5.5 s of 72.0 s: **+5 to +8 % prefill at best**, before the cost of the real scales (per-token X
  factor needs an extra pass or an in-kernel pre-scan, a weight scale per tensor needs a pass over the weights or a calibration)
  and before any quality cost. This is the largest lever left of #159, the stall splits gave +1.1 % together.
- **Quality prediction (unchecked):** the noise per expert GEMM is about 3.7 % rms; the ten routed experts of a token add independently
  and are weighted, so the noise of the MoE output is about 3.7 % / sqrt(10) to 3.7 % /sqrt(5) of its size, a quarter to half of the
  per-GEMM noise of the dense fp8 (2.4 to 2.6 % per GEMM for activations only), which together gave dNLL +0.015. That puts the
  experts at a few thousandths of a nat, but the dense path is already above the rule, so any addition needs the measured
  dNLL against today's path; no prediction replaces it.
- **Plan (needs about 62 min of GPU, a decision of the maintainer):** (1) quality first, by emulation in the f16 kernel (no speed:
  weights and X rounded to the E4M3 grid with the scales of the real design inside `k_gemm_q2`, behind a default-off switch); (2)
  the 25 stored turns teacher-forced (`TONA_TEACHER_SPECIAL=1`, `TONA_TEACHER_TAIL`, `TONA_DETERMINISTIC=1`) as in
  `2026-11-10-fp8-quality.md` Plan 1, 50 runs of about 75 s: reference arm = today's path (dense fp8 on, experts f16), variant =
  experts emulated in fp8; statistics: dNLL mean with the se over the 25 per-turn means, top-1 agreement, truncated KL; rule: upper
  95 % below 0.01 nat. (3) the needle at 128K. (4) only if (2) and (3) pass: the real scheme (weight scale per tensor from a
  calibration at load, X factor per token) and the speed `ab.py` at 130K. If dNLL fails, the experts stay on the f16 kernel.
- Why the micro test first: it is the smallest run that can stop the lever (a ratio below about 1.15x would have ended it, as in #100);
  it did not, so the quality check is worth its GPU hour.

## Plan of the quality check (written before the runs; coordinator go of 2026-11-17)

- **Question:** (a) incremental dNLL of experts in fp8 against today's path (dense fp8 on, experts f16): this decides the lever, rule
  upper 95 % of the mean dNLL below 0.01 nat; (b) the total against the all-f16 reference (dense fp8 + experts fp8 against
  `TONA_PREFILL_FP8=0`), so that the sum with the dense fp8 default (+0.015, `2026-11-10-fp8-quality.md`) is visible. The dense fp8
  default is not touched.
- **Method:** emulation of the planned scheme inside the f16 path (default off, branch `159-fp8-emu`, `TONA_GEMMQ_FP8_EMU=1`): the
  unpacked weight value of `k_gemm_q2` goes through the E4M3 grid with the fixed scale 16 (`emu_rt`: `cvt_pk_fp8_f32` and back, the
  same rounding as the fp8 WMMA operand; a fixed scale suffices because E4M3 has an exponent range of 2^17: typical weights 0.014
  times 16 sit in the middle of it, values below 1.2e-4 flush), and the rows of the expert inputs (`b.xg` after the scatter, `b.A16`
  after the SwiGLU) go through E4M3 with the factor absmax / 448 per row (`k_fp8_roundtrip`, the same factor as `k_to_fp8` of the
  dense path), back to f16. The f16 products of grid values are exact in the f32 accumulator, so the numerics are those of the
  real fp8 kernel up to the order of the sum; the speed of the emulation is not measured.
- **Turns:** the 25 stored Claude Code turns of `2026-11-10-fp8-quality.md` (prompts 32K to 70K tokens, 21,066 turn tokens; the
  content stays in `/var/tmp/claude/w159/q`, outside the repository), teacher-forced exactly as there: the head through the prefill,
  the turn through the megakernel (`TONA_TEACHER_TAIL`), `TONA_TEACHER_SPECIAL=1`, `TONA_DETERMINISTIC=1`, one process per run (about
  75 s). Order: the 8 turns with the most tokens first (`sub8`: 0064 0069 0070 0054 0086 0095 0058 0082, 12,812 tokens), then the
  rest. Arms: `emu` (new); `on` (today) and `off` (all f16) are taken from the stored runs of 11-10 **after** one turn is rerun
  with today's build and equals the stored `on` bit for bit (if it does not, `on` is rerun for all turns, 25 more runs).
- **Statistic:** dNLL per turn token, mean with the standard error over the per-turn means (tokens of a turn are correlated; the
  honest one), the 95 % interval with the t quantile of n - 1 turns, top-1 agreement, truncated KL(reference || arm) over the top-50
  of the reference.
- **Early stop:** after the 8 turns of `sub8`: if the **lower** 95 % bound of (a) is above 0.01 nat the lever fails and the rest is
  not run; if the **upper** bound is below 0.01 the other 17 turns still run (the rule is on all 25). Needle at 128K only if (a)
  passes on all 25 turns.
- **Expectation (unchecked):** (a) +0.004 nat (tolerance +-0.004): the experts' noise per GEMM is 3.7 % rms but averaged over about
  ten experts; (b) +0.019 (the sum of +0.015 and (a)). A value of (a) above 0.01 would be a finding about how the noise of the expert
  path adds.
- **Why this size:** the 25 turns are the calibrated measure of the rule (`2026-11-10-fp8-quality.md`); a smaller set widens the
  interval beyond what 0.01 can resolve (the per-turn means scatter -0.024 to +0.047); the early stop at 8 turns bounds the cost of a
  clear failure to about 10 min of GPU.
- **Active tooling:** `TONA_GEMMQ_FP8_EMU` (the arm), `TONA_PREFILL_FP8`, `TONA_TEACHER_*`, `TONA_DETERMINISTIC=1`; no profiler. Test
  conditions (`hw_report`, `quiet_check`) before and after the block of runs, desktop VRAM and load named.

## Result of the early check (the 8 turns of `sub8`, 12,812 turn tokens; 00:16 to 01:20 on 2026-11-17)

Statistics as planned (dNLL per turn token; mean; standard error over tokens; the 95 % interval uses the 8 per-turn means with
t = 2.36). Today's build (`work/build/w7`, branch `159-fp8-emu` = `159-lever1` + the emulation), `TONA_DETERMINISTIC=1`:

| Comparison | dNLL mean | se tokens / per-turn means | 95 % over the turns | top-1 agreement |
|---|---|---|---|---|
| **(a) experts fp8 (emulated) minus today (dense fp8 on)** | **-0.0038** | 0.0020 / 0.0015 | **[-0.0069, +0.0001]** | 0.953 |
| (b) experts fp8 + dense fp8 minus all f16 | -0.0038 | 0.0019 / 0.0012 | [-0.0064, -0.0007] | 0.950 |
| dense fp8 alone (today minus all f16), for the record | +0.0000 | 0.0014 / 0.0015 | [-0.0037, +0.0034] | 0.953 |

Per-turn means of (a): see `/var/tmp/claude/w159/q` (not in the repository); the dense-only per-turn values range from -0.008 to +0.005.
Reading: the upper end of (a) is +0.0001, a hundredth of the 0.01 rule: the incremental loss of the experts in fp8 is not measurable
on these 8 turns; it passes, and the early stop does not fire (the lower bound is far below 0.01). The remaining 17 turns run for
the rule on all 25 turns. The truncated KL of these 8 turns is not reported: the emulation arm was extracted against the stored
11-10 reference ids; top-1 agreement and dNLL are valid. Top-1 agreement 0.953 for both comparisons says that the fp8 experts change
the argmax as often as the dense fp8 already does (a measure of rounding noise, not of quality: dNLL is flat).

### Findings that are not about this lever

1. **The stored 11-10 reference arms cannot be reproduced.** The `on`, `off` and `fix` arms stored in `/var/tmp/claude/w106/tf` on
   2026-11-10/11 give a tail NLL of 0.395 to 0.428 for turn 224356-0049 (and 0.74 to 1.03 for the other turns); today's build gives
   0.604 (`on`) and 0.609 (`off`), higher by 0.11 to 0.28 nat on all 8 turns, for **every arm alike**. Builds of 11-11 03:45 (`work/build/merged`),
   11-13 (`main`), 11-15 (`base`) and today's give **bit-identical** NLL (0.608985) for `off`, with every #159 switch off, with
   `TONA_TIERS=0`, and with the model file unchanged since 2026-10-23, so the shift is not in the engine code of the last
   days (the 11-10 build `w106base` is gone) and not in my levers. The targets (`tgt`) are identical, the position-wise
   difference is not noise (19 % of the positions move by more than 0.5 nat, up to 7.3). Cause open: what differs from the
   stored runs is outside the repository (driver or kernel update, toolchain, a setting of the old runs). Consequence: the
   dNLL of this lever and of the dense fp8 are measured against references computed today; the stored +0.0150 of the dense fp8
   (11-10) is not comparable with today's +0.0000 and the two should not be mixed.
2. **Dense fp8 alone is at +0.0000 [-0.0037, +0.0034] on these 8 turns**, against the stored +0.0150 of 11-10: after the merged output-projection
   fix (`TONA_PF_FP8_OUT` off, 2026-11-11) the dense fp8 costs nothing measurable here, which the 11-10 number (before the fix)
   cannot tell.

## Result of the full check (25 turns, 21,066 turn tokens; block 3 ended 01:40 on 2026-11-17)

Method as planned above: teacher-forced, `TONA_DETERMINISTIC=1`, references (`on` = today's path, `off` = all f16) rerun
today on all 25 turns, `emu` = fp8 experts emulation (`TONA_GEMMQ_FP8_EMU=1`: weights through E4M3 with fixed scale 16,
activations per row with absmax/448, in `k_gemm_q2` and for the intermediate of the down GEMM). Mean dNLL over per-turn
means, t-based 95 % interval over 25 turns (t 2.06).

| Comparison | dNLL (nat) | 95 % interval | top-1 agreement |
|---|---|---|---|
| (a) experts fp8 against today's path (`emu` - `on`) | -0.0034 | [-0.0078, +0.0022] | 0.959 |
| (b) experts fp8 against all-f16 (`emu` - `off`) | -0.0027 | [-0.0049, +0.0020] | 0.958 |
| today's path against all-f16 (`on` - `off`), the dense fp8 | +0.0007 | [-0.0034, +0.0061] | 0.959 |

- **Rule (upper 95 % < 0.01 nat): passed for (a), upper bound +0.0022.** The point estimate is slightly negative, that is
  noise, not a gain. Largest single turn: 0083 at -0.052 (better), largest worse: 0053 at +0.012.
- Truncated KL against `on` is 0.157 for `emu`, 0.016 for `off` (values of `emu` on turns 49 to 102 only; the first 8
  turns were extracted against stored ids and are not valid for KL, dNLL and top-1 are).
- Sum: experts fp8 plus dense fp8 together cost -0.0027 [-0.0049, +0.0020] against all-f16, below 0.01.
- Caveat: this measures the emulation (rounding of weights and activations to E4M3 in the f16/WMMA path). The real fp8
  WMMA accumulates in f32 as the emulation's WMMA does in f16-input mode, so the remaining difference is the order of
  summation; the needle test and the real kernel decide the rest.
- Conditions: `hw_report` before and after (files `hw_before1`, `hw_after1`, `hw_after3` in `/var/tmp/claude/w159/q`),
  quiet_check warning in both: 2.6 to 2.9 GiB VRAM held by the desktop, load about 1.1, no other measurement running.

## Drift check of the 11-10 references (coordinator order of 2026-11-17, not finished)

Question: is the +0.11 to +0.28 nat shift of every arm against the stored 11-10 runs caused by our code (rule 1) or by the
environment? Smallest run: turn 224356-0049, all-f16 arm (stored tail NLL 0.3946; today's builds 0.6090), built from the
commit of the stored runs. Outcomes: old build gives the stored number, then a bisect over the builds; old build gives
today's number, then the environment.

- The stored runs came from `w106base` = main ff10b60 plus the diagnostic commit 49a9bec (`TONA_TEACHER_SPECIAL`, `TONA_PREFILL_FP8`
  switch). The tree ff10b60 alone lacks `TONA_TEACHER_SPECIAL`: a first run of it gave other target tokens (424, 1142 where the stored
  runs have 198, 51) and an NLL of 1.9, so that run says nothing and is discarded. Build `work/build/old-ff10b60` is now
  ff10b60 + 49a9bec (cherry-picked, worktree `work/old-ff10b60`), the same as `w106base`.
- Both runs of the corrected build stopped with `out of memory` in `dev_malloc` before the prefill (GTT 10.6 to 19.9 of 32.5 GB
  in use at the failures): the old engine has no GTT budget (`gtt_budget.h` came later) and another worker's server
  (`/var/tmp/claude/161/build/tona-server`, restarting every 15 to 30 s, 23 to 31 GB GTT in use) was running at the same
  time. A run with `HSA_USERPTR_FOR_PAGED_MEM=1` also ran out of memory. No number yet. The check needs the GPU and the
  host memory alone; it is not repeated while another server runs.
- Open until the old build has run: whether the shift is in our code or in the environment (driver, toolchain, kernel).

### Result of the drift check (A1, #163, evening of 2026-11-17): the model file, not our code

Turn 224356-0049, all-f16 arm, `TONA_DETERMINISTIC=1`, tail NLL (165 tokens). Test conditions: Ryzen 9 9950X, kernel 7.3.4
(the 11-17 builds gave 0.608985 on the old 9600X and kernel too, so CPU and kernel do not matter), idle machine (0.1 % CPU of
others, no swap), every run through `gpu_queue --kind exclusive`.

| Build | Model shard 1 | Tail NLL | top-1 |
|---|---|---|---|
| stored 11-10 run (`w106base`) | 2026-09-29 (sha256 e11083ba) | 0.394631 | |
| old build ff10b60 + 49a9bec | shard of 2026-10-19 (bfa3644d, the file in `work/models`) | 0.608985 | 79.39 % |
| today's tree (switches off) | shard of 2026-10-19 | 0.608985 | 79.39 % |
| old build ff10b60 + 49a9bec | shard of 2026-09-29 (e11083ba, the file the other engine keeps, sha256 checked) | **0.394631** | 87.27 % |

- The old build reproduces the stored number to six digits with the old shard and today's number with the new shard; today's
  tree and the old build are bit-identical on the same shard. So there is **no regression in our code** and no bisect is needed;
  the environment (kernel, CPU, driver after the reboot) does not move the result either.
- Cause: on 2026-11-12 `toolchain/models.toml` pinned the shard of 2026-10-19 (commit b6dd58c, 3,063 of 12,288 experts exchanged,
  same size), after the 11-10 runs. The per-turn references of 11-10 were computed with the old shard. The "+0.11 to +0.28 nat for every
  arm" is the difference between the two shards on these turns (all arms alike), not a drift. Rule 1 is not touched.
- The `on - off` and `emu - on` numbers above are differences within one shard (today's), so they stand. Only comparisons with the
  stored 11-10 arms (dense fp8 +0.0150) mix shards.
- The shard of 2026-09-29 does not exist in `work/models` any more; the copy under the other engine's data directory is read only here.
  Raw: `/var/tmp/claude/w159/q/224356-0049-off_{old2,today,oldshard}.nll`.

## Plan: needle at 128K on the emulation (B2 step 1, written 2026-11-18 before the run)

The 25-turn teacher-forced check passed (upper 95 % +0.0022 nat). The coordinator's rule asks for the needle at 128K next
(retrieval at long context is where a lossy prefill shows first, 2026-11-10-fp8-quality.md). Quality run, no timing claim: `gpu_queue --kind gpu`.

- **Question and decision:** does fp8 expert emulation lose needles against today's default path? Pass if emu is not more than 4 hits
  below the default arm on the pooled 80 needles; a loss of more than 4 stops the lever (maintainer decision). A gain is not expected.
- **Scope:** `bench/needle/run.py --hard --require-cache`, 131,072 tokens, texts 0 and 1 (80 needles: single 40, multi 40), arms `on`
  (today's default: dense fp8 as shipped, experts f16, shard 10-19 and its profile) and `emu` (`TONA_GEMMQ_FP8_EMU=1`), same build
  `w7`, server `--ctx 140000`, port 8099, `systemd-run --scope -p MemoryMax=54G -p MemorySwapMax=0`, stopped by PID. One server start per arm.
  Duration about 7 min per text and arm, so about 30 min in all.
- **Pool:** the 11-11 pool is gone (`/var/tmp/claude/w77b`); a new one from this repository (`corpus.py ... engine docs bench`, 549 files,
  3.0 M characters). Absolute hit counts therefore are not comparable with the 23/40 of 11-11 (other pool, other shard); only the pair
  `on` against `emu` on the same documents counts.
- **Expectation:** `on` 18 to 28 of 80 per 40 needles at this depth, i.e. 36 to 56 of 80 (11-11: 23/40 with the fix, range 14 to 26; FP8 as shipped 20/40; f16
  23/40); `emu` minus `on` within +-4 of 80 (binomial spread of two runs on identical needles, p about 0.55: sd of the difference about 4 if
  the runs were independent, less because the documents are identical). A result outside is a finding.
- **Active tooling:** none beyond the emulation switch; no diagnostics; hw_report and quiet_check before and after
  (`/var/tmp/claude/w159/needle/`). The desktop guard of the queue applies (VRAM at 140K context is about 13.6 GB).

## Result: needle at 128K, emulation against today's path (2026-11-18 01:39 to 01:47)

As planned above (texts 0 and 1, `--hard --require-cache`, 131,072 tokens, pool of this repository, build `w7`, one server start per arm,
`gpu_queue --kind gpu`; the queue let it run after other workers' jobs; `hw_before.txt`, `quiet_*` in `/var/tmp/claude/w159/needle/`):

| Arm | all 80 | single 40 | multi 40 |
|---|---|---|---|
| `on` (today's default) | 39/80 = 48.8 % [38.1, 59.5] | 28/40 | 11/40 |
| `emu` (experts fp8 emulation) | 41/80 = 51.2 % [40.5, 61.9] | 27/40 | 14/40 |

Paired, emu minus on: **+2.5 pp [-5.2, +10.2], McNemar p = 0.75** (both right 35, only on 4, only emu 6, both wrong 35); single -2.5 pp
[-11.0, +6.0], multi +7.5 pp [-5.3, +20.3]. Expectation was 36 to 56 of 80 for `on` (39: inside) and emu within +-4 of `on` (+2: inside).
**The experts fp8 emulation loses no needles at 128K**; together with the 25 turns (upper 95 % +0.0022 nat) the quality rule is met for the lever.
Absolute hit counts are not comparable with the 23/40 of 11-11 (other pool, other shard).

## Real kernel and plan of the speed run (2026-11-18)

Branch `159-fp8-real` (from `159-fp8-emu`, main merged), build `w8`. Default off, switch `TONA_GEMMQ_FP8=1`:
- **X:** after the scatter (gate/up input) and after SwiGLU (down input) `k_fp8_rows_inplace` turns every f16 row into E4M3 bytes at the start of
  its own row (the row goes through the LDS first, so no extra buffer; row stride unchanged) with the factor sx[t] = absmax / 448 (the same rounding
  as the emulation and as the dense fp8's `to_fp8_rows`); two small float arrays (`sxg`, `sxa`, 2 x C x ne floats, 0.5 MB) hold the factors.
- **Weights:** `k_gemm_q2f8` unpacks into the LDS as E4M3 (value times the fixed scale 16: the E4M3 exponent range covers the dequantised
  weights of every format in the model, so no per-tensor scale or calibration pass is needed), the fp8 WMMAs (`wmma_f32_16x16x16_fp8_fp8_w32_gfx12`) run
  at twice the f16 rate with half the LDS bytes, the epilogue multiplies sx[t] / 16.
- Needs K a multiple of 64 (the second-generation kernel); otherwise `gemm_q` throws instead of reading fp8 bytes as f16.
- Same rounding as the emulation checked on turn 224356-0049: real against today's path dNLL -0.0271 (se 0.0228), against the emulation -0.0283
  (se 0.0158), top-1 0.958/0.964, KL 0.0214 against 0.0171 of the emulation: the same size, no sign of a bug (gate of the queued job).

**Plan of the speed run (written before it):** `scripts/ab.py --a w8 --b w8 --b-env TONA_GEMMQ_FP8=1` (same build, switch off against on, so that
nothing else of the main merge enters), workloads `long` (130K prompt) then `mid` (30K), `TONA_DETERMINISTIC=1`, exclusive queue, early stopping with
`bench/noise.toml`. Question: what does the real kernel give on prefill? Expectation: prefill +3 to +7 % at 130K (k_gemm_q2 is 22.8 % of the window; the micro
test gave 1.27 to 1.52x on the GEMM, i.e. +5 to +8 % at best; the pre-pass for X and the epilogue take part of it), 30K about the same (expert GEMM share
similar); decode unchanged within +-0.5 % (the switch does not reach decode). Tolerance: the interval of ab.py; outside +3..+7 % is a finding. A gain below
+1.5 % ends the lever (the dense fp8 default was accepted for +3 % at a dNLL cost; this one costs no measurable dNLL, so the bar is lower, but below 1.5 %
the extra code is not worth it). Default change: not by me; the lever goes to the maintainer with the numbers.

## Result: real kernel, quality gate, 8 turns and needle at 128K (2026-11-18 02:29 to 02:48)

Build `w8`, `TONA_GEMMQ_FP8=1`, same methods as the emulation (`gpu_queue --kind gpu`, `hw_before.txt` and `quiet_*` in `/var/tmp/claude/w159/needle/`).
- **8 turns of sub8 (12,812 tokens), teacher-forced, `TONA_DETERMINISTIC=1`:** real minus today's path dNLL **-0.0015**, se 0.0020, 95 % over turns
  [-0.0048, +0.0030] (rule: upper < 0.01: passed), top-1 0.954, KL 0.0136; real minus emulation +0.0023 [-0.0000, +0.0051], top-1 0.954, KL 0.0140.
  So the real kernel agrees with the emulation within noise.
- **Needle 128K, texts 0 and 1, same pool and arms as above:** `real` 41/80 = 51.2 % [40.5, 61.9] (single 28/40, multi 13/40) against `on` 39/80;
  paired real minus on **+2.5 pp [-6.0, +11.0], McNemar p = 0.77** (single +0.0, multi +5.0). No loss.
- So the quality rule is met by the real kernel on the 8 turns (the 25-turn run was done on the emulation) and at 128K; the speed run (plan above) is queued.

## Result: speed of the real kernel, `ab.py` w8 switch off (A) against on (B), exclusive queue, 2026-11-18 03:33 to 03:43

Same build, `TONA_DETERMINISTIC=1`, 2 rounds per workload (AB then BA), stopped early by `ab.py` (`/var/tmp/claude/w159/ab13/`, `hw_before_*`, `quiet_before_*`;
quiet_check warning: the desktop holds VRAM, as in all runs of this night).

| Metric | A (off) | B (on) | B vs A [95 %] | Plan | Verdict |
|---|---|---|---|---|---|
| prefill tok/s after 130K prompt | 1873.5 | 1917.3 | **+2.34 % [+2.29, +2.38]** | +3 to +7 % | below, a finding |
| prefill tok/s after 30K prompt | 1939.6 | 1976.5 | **+1.90 % [+1.82, +1.99]** | +3 to +7 % | below, a finding |
| decode tok/s after 130K | 65.1 | 66.3 | +1.84 % [+0.95, +2.74] | +-0.5 % | text effect (see below) |
| decode tok/s after 30K | 70.2 | 75.2 | +7.16 % [+6.39, +7.92] | +-0.5 % | text effect (see below) |

- Per round: 130K +2.32 and +2.36 %; 30K +1.88 and +1.93 %. Prefill noise is 0.02 % (noise.toml), so both numbers are real and the interval is narrow.
- The gain is above the 1.5 % bar of the plan, but below the +3 to +7 % that the GEMM share (22.8 % of the 130K window) and the micro ratio (1.27 to
  1.52x) predicted (+5 to +8 % at best).
- **Cause of the shortfall (arithmetic, not yet measured):** the in-place conversion reads the f16 rows and writes E4M3 rows of every chunk
  of every MoE layer: gate/up input 6144 x 10 x 2560 x 2 B = 315 MB read and 157 MB written, plus the SwiGLU output 79 MB read: about
  0.55 GB per layer and chunk, 1.1 ms at the memory rate (about 500 GB/s), times 48 layers and 21 chunks at 130K = about 1.1 to 1.3 s of 72 s = 1.5 to 1.8 %; the GEMM
  saves about 4 to 5 %, the difference is the +2.3 % seen. (Unchecked: a profile of `k_fp8_rows_inplace` against `k_gemm_q2f8` would settle it.)
  The conversions can be fused: the scatter writes each source token row to 10 slots, so quantizing in the scatter needs one absmax per source token
  (n rows, not n x 10) and writes 1 byte instead of 2 per element; SwiGLU can write E4M3 and the factor directly. That removes the extra passes.
- **Decode "gain" is a text effect:** the deterministic texts differ between A and B (the arms are not bit-exact, SHA 6eda... against 7d29... at 130K), the expert
  cache counters differ (73,593/9,057 against 75,097/8,854 at 130K), and decode rate follows the generated text (the MTP drafter accepts more or fewer tokens);
  the same effect was seen on 11-11 for the dense fp8 (decode -9.8 % / +8.3 % with a code path that does not touch decode). No decode claim.
