# Coder decode step (#175)

Date: 2026-11-20. Branch `worktree-<worktree>`, base `99e5da2`. Build `work/build/cdec` (base), logs `/var/tmp/claude/cdec/`,
ab.py output `work/ab/cdec-*`. Starting point: README "Measured state" (October): verify pass of 4 tokens about 37 ms, about 4 ms of it expert
misses, 33.6 ms with all experts in VRAM, ceiling about 21 ms (3.52 GB dense + 0.98 GB experts at 315 GB/s). KV placement of the Coder was studied in
`2026-11-20-coder-kv-hot-head.md` (no position placement beats its slot cost): not redone.

## Prior art

- [KVBuffer, arXiv 2605.19049](https://arxiv.org/abs/2605.19049), status: read (abstract; full text pending). Buffers recent K/V and defers the linear-attention state write chunk-wise;
  up to 1.47x batch decode and 5x more concurrent requests with 4 drafts in SGLang (Qwen3.5, Nemotron-3). Judged against the profile below.
- [DraftExpert, arXiv 2607.24434](https://arxiv.org/abs/2607.24434), status: read (abstract). Needs trained draft experts per layer and target-expert prefetch (hits 86-88 %) for
  CPU-to-GPU offload of DeepSeek-V2-Lite / Moonlight; our engine already prefetches the 8 most likely missing experts per layer from the next layer's router
  (`TONA_PREFETCH`, README "Preloading", VRAM hit rate 75 to 88 %). Judged against the profile below.

## Test conditions

`python3 tools/hw_report.py` (06:20, `/var/tmp/claude/cdec/hw.txt`): Ryzen 9 9950X, RX 9060 XT gfx1200 15.9 GiB, 2.95 GiB VRAM held by the desktop,
60.5 GiB RAM (43.8 available), swap 4.4 GiB in use, load average 22 (builds of several contributors, mine included), commit `99e5da2`.
`quiet_check.py` (06:20): **do not measure** (other programs 70 % of the CPU threads, load, VRAM held, swapping). Block 1 reads deterministic
shares of a profile, not rates; every speed block gets its own check before and after, a warning is reported with its result.

## Plan

### Block 1: where the Coder's decode pass goes at 2K, 16K and 64K

**Question.** Per step family: microseconds, bytes, share of the bandwidth roofline of a decode pass after a 2K, 16K and 64K prompt, at the default server
setup (ctx 262144, MTP drafter, shipped expert profile), and which family is furthest from its roofline.
**Run.** One `exclusive` job: `ab.py` A = B = `cdec`, A with `TONA_MEGA_PROF=1 TONA_MEGA_PROF_EVERY=20`, B plain (so the cost of the profile is measured too),
workload `server`, targets `2000:200,16000:100,64000:50`, ctx 262144, 2 rounds (minimum of ab.py). `--why-size`: 2K is the default workload, 16K and 64K show
what grows with the context; 20-pass windows of deterministic steps need no more tokens. Server log lines (slots, hits, misses, tokens per pass) read per context.
**Expectation (source: README "Measured state", `2026-11-20-coder-kv-hot-head.md`).** 2K: pass 30 to 40 ms at width 3 to 4 (about 77 tok/s with 2.5 to 3
tokens per pass); misses about 4 ms; 64K: pass longer by the QSA attention reads of about 0.7 ms (14 MB per token over PCIe) times the width; tolerance +-30 %.
**Decision fixed before.** A family above 15 % of the pass and below 70 % of its bandwidth roofline is a lever candidate; host.wait above 8 % points to
handshake work; `moe.*` growing with misses points to prefetch/placement (DraftExpert); `dn.*` above 15 % with state bytes dominating points to KVBuffer.
**Active tooling.** `TONA_MEGA_PROF`, `TONA_MEGA_PROF_EVERY` in arm A only; no `TONA_DETERMINISTIC`, no rocprof.
**Stop.** Out of memory, GPU fault or queue exit 75: reported.

## Results

### Block 1 (job `cdec-b1`, build `cdec` at `99e5da2`, `work/ab/cdec-b1.{md,json}`, server logs `work/ab/cdec-b1-logs/`)

Quiet check before 06:28 and after 06:40: warning only for 3.0 GiB VRAM held by the desktop. Server: ctx 262144, KV int8 3.62 GB in RAM, VRAM 11.97 GB,
**55 expert slots per layer** (of 256), 29.2 GB pinned, MTP head. A = profile on, B = plain: output SHAs equal at 2K/16K/64K and stable, hits and misses identical,
profile cost decode@2000 +0.07 %, @16000 -0.48 %, @64000 -0.70 % (2 rounds, rounds disagree by up to 1 %: inside the plan of +-3 %), prefill -0.3 to 0 %.
Rates (decode, MTP drafter, tokens per pass 3.33 / 3.00 / 3.33): about 70.6 tok/s at 2K, 67.8 at 16K, 74.1 at 64K; prefill 1,835 to 1,869 tok/s.

Pass time by step family (20-pass windows; the first window of each request has width 3.7, later ones up to 5). Bytes: tensor sizes of the GGUF per step
(`tools/gguf_info.py --tensors`, bits per weight of the type; hc weights as Q8_0 as the engine reads them), roofline = bytes / 315 GB/s:

| Step (count) | us per pass 2K (w 3.7) | 16K | 64K | share | bytes per pass | roofline us | % of roofline |
|---|---:|---:|---:|---:|---:|---:|---:|
| moe.gate_up (48) + moe.down (48) | 7,680 + 6,177 | 7,0xx + 5,586 (window w 3.35) | 8,406 + 6,856 (w 4.25) | 35 % | union experts x 2.0 MB: 30 experts = 3 GB | about 9,500 | about 70 % (union not yet counted, block 2) |
| dn.in (36) | 4,708 | 4,678 | 4,736 | 12 % | 1,140 MB | 3,500 | 74 % |
| gr.down (97) + gr.up (97) | 2,423 + 2,311 | 2,462 + 2,343 | 2,462 + 2,343 | 12 % | 2 x 336 MB (2 x 3.3 MB per step) | 2 x 1,070 | **44 / 46 %** (24 us per step against 11 us of bytes) |
| dn.out (36) | 2,200 | | 2,225 | 5.6 % | 444 MB | 1,409 | 64 % |
| qsa.attn (12) | 1,983 | | 2,338 | 5 to 5.6 % | selected cells 14 MB per token x width | | KV reads from pinned RAM, grows 18 % to 64K |
| head.out (1) | 1,696 | | 1,723 | 4.3 % | 521 MB | 1,655 | 98 % |
| qsa.in (12) / qsa.out (12) | 1,485 / 700 | | 1,493 / 706 | 3.8 / 1.8 % | 330 MB / 151 MB | 1,040 / 479 | 70 / 68 % |
| combine (97) | 1,373 | | 1,421 | 3.5 % | activations only (14 us per step) | | latency |
| dn.recur (36) + dn.conv (36) | 1,181 + 430 | | 1,230 + 417 | 4.1 % | state 3 MB per layer read and written: 226 MB | 720 | 45 % |
| host.wait (48) + router (48) | 1,172 + 936 | | 1,226 + 970 | 5.4 % | gate_inp 63 MB | 200 | handshake |
| moe.sum / shared / act | 485 / 440 / (500 at w 5) | | | 3.5 % | shexp 146 MB | 460 | 100 % |
| **pass** | **39.2 ms (w 3.7)** | 37.9 (w 3.35) | 41.5 (w 4.25) | | dense 3.5 GB + experts | **ceiling about 21 ms** | |

Reading. The profile is flat. **The context hardly matters** (qsa.attn 2.0 ms at 2K, 2.3 ms at 64K: the indexer picks about 2,048 cells whatever the context; KV placement
confirmed as no lever). The 18 ms between the pass (39 ms at width 3.7) and the ceiling (21 ms) is: dense matvecs below roofline about 4.7 ms (of it gr.* 2.7 ms: a latency
bound already studied in `2026-10-25-hc-matvec.md`, about 10 us fixed cost per matvec step), the steps without weights (combine, recur, conv, sum, act, host.wait, router, qsa.attn: about 9 ms in about
500 small steps of 5 to 33 us each), the experts (13.9 ms against about 9.5 ms of bytes at 30 experts per layer; the misses are in it).

### Block 2 (plan, written before its run): experts per pass and the CPU share of a cycle

**Question.** (a) How many experts per layer does a pass compute on the GPU (`nu`), how many of them are read over PCIe (`gd`), so the MoE steps' bytes and their
share of roofline are known; (b) how much of a cycle is the CPU's own work (`TONA_HOST_TRACE`, `tools/host_trace_report.py`). **Run.** One `exclusive` job, `ab.py` A = B = `cdec`, both
`TONA_MEGA_PROF=1 TONA_MEGA_PROF_EVERY=20 TONA_PROF_LAYERS=1`, B also `TONA_HOST_TRACE`, targets `2000:300,16000:100`, ctx 262144, 2 rounds.
**Expectation (source: block 1, README "Measured state").** nu about 25 to 32 experts per layer at width 3.7; gd 2 to 6 per layer (the README's 4 ms of misses per pass = 48 layers x 3 experts x 2 MB /
PCIe 40 GB/s = 7 ms upper end); host trace cycle about 40 ms. Rates -3 to +3 % (host trace). **Decision fixed before.** If MoE bytes at the measured nu are above 85 % of roofline,
the experts are not a lever and the remaining ones are the 500 small steps and the dense latency; if gd times 2 MB over PCIe exceeds 3 ms per pass, prefetch/placement (DraftExpert) is the lever. **Active tooling.** as named.

### Block 2 result (job `cdec-b2`, `work/ab/cdec-b2.*`, trace `/var/tmp/claude/cdec/ht.txt`)

Quiet check files `/var/tmp/claude/cdec/quiet-b2-*.txt` (VRAM held by others only). Rates inside the plan. **nu = 23.8 experts per layer per pass** (mean of the 288 printed rows;
width 3.7 to 5.05), so the experts read per pass are about 23.8 x 2.0 MB x 48 = 2.3 GB = **7.3 ms at 315 GB/s against 13.9 ms measured (about 52 %)**: moe.gate_up about 62 %,
moe.down about 47 % of roofline. `gd` (experts read over PCIe) printed only where above 11.7 experts (1 % of the total, a threshold that hid the per-layer rows): layers 0 and 1
read 12 to 18 experts per pass over PCIe; the mean over layers is not known from this block (block 3 prints every row).
Host trace (115 cycles, mean 48.1 ms): MTP draft 4.0 ms per cycle (3.78 steps of 1.06 ms), tail after the last router 2.2 ms, embedding 0.56 ms, CPU selection 11 us per router
(0.53 ms per cycle), CPU experts 229 us per layer (overlapped with the GPU's 290 us per layer), everything else under 0.1 ms. About 5 ms of the 48 ms cycle (10 %) is outside the pass.
Prior work on the expert kernel (`2026-11-12-expert-kernels.md`, #128): the expert phase is limited inside the CU (decode and dependency chains, VALU 20 to 35 %), not by DRAM; three bit-exact variants measured, none faster;
WMMA int8 for the experts (GPUOpen Double-K, `papers.md`) analysed and not built.

### Block 3 (plan, written before its run): gd and step time per layer

**Question.** Per layer: experts computed (nu), experts read over PCIe (gd) and the time of its moe.gate_up/moe.down steps; regression of step time on nu and gd gives the cost per
expert in VRAM and per expert over PCIe, so the 6.6 ms gap between measured and roofline splits into misses and kernel. **Run.** One `exclusive` job, `ab.py` A = B = `cdec2` (this
branch: `report_profile` prints every row under `TONA_PROF_LAYERS`), `TONA_MEGA_PROF=1 TONA_MEGA_PROF_EVERY=40 TONA_PROF_LAYERS=1`, `2000:400`, ctx 262144, 2 rounds (A/A).
**Expectation.** (source: README "about 4 ms of misses per pass", 2026-11-12-expert-kernels.md: moe.gate_up 150 GB/s at nb 1 in-CU limit) cost per VRAM expert about 7 ms / 1,140 expert-layers = 6 us in gate_up+down; per PCIe expert
about 40 us extra at 50 GB/s; mean gd 2 to 4 per layer. **Decision fixed before.** If misses explain above 3 ms of the gap: placement/prefetch is the lever (DraftExpert-like: predict earlier);
if below 1.5 ms: the kernel (in-CU limit) is, and the lever is the Double-K WMMA design. Active tooling: as named.

### Block 3 result (job `cdec-b3`, A/A of `cdec2`, `work/ab/cdec-b3.*`, logs `work/ab/cdec-b3-logs/`, evaluation `/var/tmp/claude/cdec/reg.py`)

Quiet checks `/var/tmp/claude/cdec/quiet-b3-*.txt` (VRAM held by others only). 8 windows of 40 passes (width 4.4 to 4.6, 42.8 to 43.6 ms per pass) of the 4 server starts, 48 layers each.
Per layer: mean **nu 24.7 experts computed by the GPU, gd 2.7 of them read over PCIe** (hit share 89 %); per pass 129.5 PCIe-read experts, **48 % of them in 8 layers**
(layer 0: 19.0, layer 1: 12.2, layer 2: 6.2, layers 22, 18, 34, 38: 4.7 to 5.9; layers 8, 9, 25, 27, 30, 40 under 0.6). Layer 0 has no previous layer to predict from.
Least squares over the 384 (window, layer) pairs of the time of `moe.gate_up + moe.down` of a layer on nu, gd: **7.3 us per expert in VRAM (2.0 MB: 272 GB/s, 86 % of the
315 GB/s roofline: the kernel is not the problem), +39 us per expert read over PCIe, 58 us fixed per layer** (2.8 ms per pass: step overheads). The MoE steps are 16.5 ms of the 43 ms pass at this width:
8.7 ms experts at bandwidth, **5.1 ms PCIe-read experts (12 % of the pass)**, 2.8 ms fixed.
Decision fixed before (misses above 3 ms: placement / prefetch is the lever; below 1.5 ms: the kernel): **misses are the lever**; the expert kernel at 86 % of roofline is not (consistent with `2026-11-12-expert-kernels.md`).
Static slot reallocation between layers (non-uniform slot counts by the shipped profile): simulation from `toolchain/profiles/coder-iq1_m.bin` (top-55 hit mass per layer, greedy allocation of the same 2,640 slots by marginal frequency)
changes the summed miss mass by 0.3 % (183.2 to 182.7 per token and layer set): **no lever**, not built. The static miss mass correlates with the measured gd per layer at r = 0.67 (layer 0 predicted 0.58).
Prefetch today (`prefetch_n_` = 8 per layer, router of l + 1 run on the input of l; `mega_prefetch.cpp`) is a fixed cap tuned in October at narrower passes; layers 0 and 1 have 19 and 12 misses per pass at width 4.4, above that cap.

### Block 4 (plan, written before its run): prefetch cap sweep (placement only)

**Question.** Does a larger prefetch cap (`TONA_PREFETCH` 12 and 16 per layer instead of 8) lower the PCIe-read experts and raise decode at the passes of today (width 4 to 5)? **Run.** One `exclusive` job, two
`ab.py` calls, A = `cdec2` default, B = `cdec2` with `TONA_PREFETCH=12` (call 1) and `=16` (call 2), workload `server`, targets `2000:300,16000:100`, ctx 262144, 3 rounds each (noise table: decode@2000 sd 0.23 %,
df 4; one round does not decide, rounds alternate AB/BA). `--why-size`: the effect is per pass and shows in a 300-token decode; 16K shows the same mechanism with KV reads added.
**Expectation (source: block 3 and README "Preloading").** Layers 0 and 1 cannot gain (layer 0 has no prediction; the cap acts on the candidates of l + 1 >= 1, so layer 1 can: 12 misses); of the 129 misses per pass at most
the ones the cap currently cuts are recoverable, estimate 20 to 40 of them at 39 us = 0.8 to 1.6 ms = **+2 to +4 % decode** for 12, similar or lower for 16 (the PCIe stream of the prefetch competes with the kernel's direct reads); prefill unchanged (-1 to +1 %).
Output SHAs equal (placement only). **Decision fixed before.** The cap with the best decode@2000 outside the margin (0.5 %) becomes the new default; both inside the margin or worse: the cap is not the lever, the next step is a
layer-0 predictor. **Active tooling.** `TONA_PREFETCH` in B only; no profile.

### Block 4 result: larger prefetch caps are slower (job `cdec-b4`, `work/ab/cdec-b4-{12,16}.*`, 3 rounds each, quiet checks `/var/tmp/claude/cdec/quiet-b4-*.txt`)

SHAs equal at 2K and 16K in both calls (placement only). B against A: **`TONA_PREFETCH=12`: decode@2000 -1.36 % [-1.94, -0.78], decode@16000 -0.73 %; `=16`: decode@2000 -4.46 % [-5.06, -3.87], decode@16000 -2.73 %**; prefill within +-0.2 %
(rounds agree within 0.3 %). Outside the plan (+1 to +5 %): the expectation was wrong in sign. More prefetch candidates cost more than they bring: the DMA of the prefetch competes with the kernel's direct PCIe reads
(the same link) and the extra candidates are mostly never used (README: a good half). The direction suggests the cap of 8 set in October at narrower passes may now be too high.

### Block 5 (plan, written before its run): smaller prefetch caps

**Question.** Is a cap of 6 or 4 better than 8 at today's passes? **Run.** One `exclusive` job, two `ab.py` calls (B = `TONA_PREFETCH=6`, then `=4`), A default 8, same workload and rounds as block 4 (3 rounds, `2000:300,16000:100`, ctx 262144).
**Expectation (source: block 4 slope -0.35 % per candidate between 8 and 12, -0.8 % per candidate between 12 and 16).** If the slope continued: 6 gives +0.5 to +1 %, 4 gives +1 to +2 %; the benefit of the prefetch itself (README: 60.5 to 65.5 tok/s, +8 %)
bends it back at some cap, so a minimum between 0 and 8 is expected, rates of prefill unchanged. **Decision fixed before.** The cap with the best decode@2000 and decode@16000 both not worse than A and at least one better by more than the 0.5 % margin becomes the default
(a change of one constant; output identical, goldens must stay equal). Active tooling: `TONA_PREFETCH` in B only.

### Block 5 result: caps 6 and 4 (job `cdec-b5`, `work/ab/cdec-b5-{6,4}.*`, 3 rounds each)

SHAs equal. `TONA_PREFETCH=6`: decode@2000 -0.21 % [-0.79, +0.38], decode@16000 +0.13 %, prefill within +0.25 % (all inside the 0.5 % margin: equal). `=4`: decode@2000 **-1.22 %** [-1.87, -0.57], decode@16000 -0.57 % (round noise 5 %: one slow round),
prefill@2000 -4.8 % (one round 1251 against 1463: the outlier-prone first request, not the cap). Together with block 4 (12: -1.4 %, 16: -4.5 %): **the cap of 8 is the optimum of the curve 4 / 6 / 8 / 12 / 16 = -1.2 / -0.2 / 0 / -1.4 / -4.5 %**.
Decision fixed before (a better cap by more than the margin becomes the default): none is better, **the default stays**. The prefetch cap is not a lever; the cache is at its hit rate (README: LRU with prefetch equals Belady, 90 to 92 %; here 89 % at 55 slots).

## Candidates and verdicts so far

| Lever | Class | Estimate (source) | Status |
|---|---|---|---|
| KVBuffer (arXiv 2605.19049, full text read) | numbers change (chunkwise, relative error 1e-3 per the paper) | dn.recur + dn.conv are 1.6 ms of a 39 ms pass (4 %), the state is already written once per pass: at most 0.6 ms; the paper reports 1.00x at 1 request/s, gains only at batch 128 and more | rejected, not built |
| DraftExpert (arXiv 2607.24434) | needs trained draft experts | the prefetch half exists (`TONA_PREFETCH`, block 4/5: the cap is at its optimum, 89 % hits); layer 0 has no previous layer, 19 PCIe-read experts per pass x 39 us = 0.74 ms | rejected as a build; layer-0 prediction from the embedding open (at most 0.3 ms) |
| Prefetch cap | placement only | blocks 4 and 5 | **measured, no gain** |
| Static slots by layer | placement only | simulation from the shipped profile: 0.3 % less miss mass | no lever |
| MoE expert kernel | bit-exact | 86 % of roofline per expert in VRAM (block 3); `2026-11-12-expert-kernels.md` three variants no gain | no lever |
| MTP draft overlapped with the tail of the pass | bit-exact in principle | the draft needs the accepted token, i.e. head.out and the host verification; only a batched first draft step for every verify position could start early: saves at most one of 3.8 draft steps (1.06 ms, 2 %) and adds work to the tail; not built | open, small |
| gr.down / gr.up (hc matvecs): split-K over more waves | **changes numbers** (float summation order of a lane's sub-blocks) | 4.7 ms at 45 % of roofline (24 us per step against 11 us of bytes, 194 steps); halving the sequential sub-blocks per lane could save about 8 us per step = 1.5 ms (3.5 %), **unverified** | report only (C6: needs Δ-NLL) |
| Slots: more VRAM for experts (55 per layer, 5.3 GB) | placement only | 88 % to 89 % hit rate; the 512 MTP experts hold 708 MB = 7 slots per layer; no cheap source of VRAM found | open |

Diagnostic kept: `TONA_PROF_LAYERS` now prints every row (per-layer step times, nu, gd) and keeps the counts out of the pass total (`engine/gpu/mega_serve.cpp`); output only, no effect on numbers or speed.

## Gates (build `cdec2` = this branch)

`just check` rc 0 (incl. clang-tidy); `ctest -L unit` 70 tests rc 0 (own build dir; `just test-cpu` uses the shared `work/build/cpu`, which another worktree configured); `just test-py` 681 passed outside a queue job
(inside a queue job two queue-order tests of `test_gpu_queue.py` fail: known trap, not the change). Goldens, all rc 0: `test_gpu_fattn`, `test_gpu_mamba`, `test_gpu_moe_latent`, `test_gpu_bolt_quant`, `test_gpu_attn_wmma`,
`golden.sh compare` (decode, cpu plan, prefill with dflash), `golden-server.sh compare`. The only code change is the print of `TONA_PROF_LAYERS` under `TONA_MEGA_PROF`; Coder speed untouched (A/A of both builds in blocks 1 to 3).
