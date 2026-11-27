# Expert matvec bandwidth per quant type and pass width (#160)

RX 9060 XT (gfx1200), Zen 5, DDR5, Coder IQ1_M shard 1 of 2026-10-19. Branch `160-expert-bw` from main e4a4f41.

## Plan (written before the first run)

**Question.** moe.gate_up and moe.down take 20.9 of 55 ms of a W = 8 verify pass (#126, 2026-11-12-verify-pass.md); the bytes
they must read are 12.9 ms at 315 GB/s. Which expert quant type, at which width, is furthest from the bandwidth limit, and
what in the kernel (occupancy, VGPRs, instruction mix, load width, cache behaviour) causes it? The answer ranks levers
(step 2); no kernel is changed in this part.

**What is not repeated** (read first): #128 (2026-11-12-expert-kernels.md): two rows per lane sharing x, one-ahead raw pipeline for
IQ2_S/IQ3_S/IQ3_XXS, scalar prefetch, WMMA int8 for moe.down and the grid types, the token-count hybrid: all measured, none faster; #130
(Q6_K tile, dense only). #126: IQ4_NL raw pipeline kept (NB <= 4), k_mega<8> spill fix kept.

**Byte accounting (exact).** Per layer the expert types are fixed by the file (48 layers; gate = up type): gate/up IQ3_XXS 17 layers
(98 B per 256 values, row 980 B, one expert 2 x 640 rows = 1.254 MB), IQ2_S 20 (82 B, 820 B/row, 1.050 MB), IQ3_S 10 (110 B, 1.408 MB),
IQ4_XS 1 (136 B, 1.741 MB) -- layer counts are taken by script from the GGUF directory (17 / 20 / 10 / 1); down IQ4_NL 39 layers (18 B per 32
values, row 360 B, one expert 2560 rows = 0.922 MB), Q2_0 9 layers (18 B per 64 values, 180 B/row, 0.461 MB). Bytes per pass and layer
= (GPU-computed union experts n_u of that layer) x bytes per expert. n_u is counted exactly in `publish_plan` (new diagnostic
`TONA_PROF_LAYERS`, summed over the timed passes): experts computed by the CPU are not counted (they are not read by the GPU), GPU-direct
experts are (they are read over PCIe, a lower bandwidth: they are about 3 % of the experts, reported). Block headers and scales are inside
the 98/82/110/136/18/18 byte blocks (no separate arrays in the bytes; the loads of the grid types read qs, signs and scales as
separate streams, which matters for cache lines, not bytes). x vectors (nb x 2560 int8 + scales, LDS) are not counted as bytes.

**Scope.**
1. Engine, per layer: `tona-gpu-gen MODEL PROMPT --mega --batch-bench 400`, `TONA_BENCH_WIDTHS=1,2,4,8 TONA_BENCH_REPEAT=3
   TONA_MEGA_PROF=1 TONA_PROF_LAYERS=1`, the profile of the #126 plan (`scripts/bench-long-code.txt`, real experts, production profile
   `work/profile/experts-coder.bin`). Gives us per layer for moe.gate_up and moe.down, hence per type (layers grouped by type) and width.
   Duration about 8 min GPU after the load. A second run without the profile gives the unperturbed pass time.
2. Probe (`tona-wide-probe --exps`, extended to nb = 1, 2, 4, 8 via `--nbs`): per type the kernel's own bandwidth in the large and in the
   phase form, which separates "kernel" from "pass" effects (barriers, imbalance, misses, host).
3. ISA: `scripts/kernel-regs.py` and an ISA dump of `row_q8<T, NB>` per type (VGPR, VALU/LDS/VMEM counts per loop body); rocprofv3
   counters if the installed one supports gfx1200 (occupancy, L2 hit rate, VALU busy).
**Expectation (with tolerance).** From #126: W = 1 moe.gate_up 3.2 ms and moe.down 2.3 ms (5.5 ms against ~3.1 ms ideal: 56 %); W = 8 11.5
and 9.4 ms against 6.9 and 6.0 ms ideal (60 %, 64 %). So achieved bandwidth 150 to 230 GB/s on the 315 GB/s ceiling, +-25 %; IQ4_NL (light decode)
highest, the grid types lowest (probe, #128: IQ3_XXS 108, IQ2_S 97, IQ3_S 108, IQ4_NL 133, IQ4_XS 172 GB/s at nb = 4, all with
nb-proportional dot work; at nb = 1: 150 / 130 / - / 262 / 240). A type more than 25 % off this order is a finding.
**Active tooling.** `TONA_MEGA_PROF=1`, `TONA_PROF_LAYERS=1` in the profile run only (the read-back per pass adds a hipMemcpy: the profile run
is no speed result); `TONA_EXPERT_PROFILE` as production; no TONA_DETERMINISTIC (it turns the wide path off and caps drafts).
**Why this size.** 400 positions after 64 warm-up tokens give 50 to 340 passes per width and 48 layers each, enough for per-layer means
(spread of the pass time between rounds is below 1 %, #126); three rounds with widths interleaved.


## Test conditions

```
- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.49 GiB held by others before the run, 1 % busy before the run
- RAM: 60.5 GiB, 49.0 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 9.54, 2.89, 1.06
- CPU use by other programs over 0.5 s: 89.3 % of all threads, swap in use 2.1 GiB
- sandbox: no
- Commit: `e4a4f41` (dirty tree)
```
The hw_report above was taken at the start, while the worktree build was still running: its load average and CPU use come from that build, not from the
measurements. The quiet check at the head and foot of each run (in the raw logs) says "warning: 1.49 GiB of 15.9 GiB VRAM are held by others" (desktop) and nothing else.
All runs went through `scripts/gpu_queue.py --kind exclusive`. Commit e4a4f41 plus the diagnostics of this branch (`TONA_PROF_LAYERS`, `TONA_PROBE_*`, `--abl`; no engine device code changed).

## Result 1: bandwidth per type and width in the engine

`scripts/expert_bw.py` on `--batch-bench 400`, widths 1, 2, 4, 8, 3 rounds interleaved, long code prompt, real experts from RAM (12 profile blocks per run, median).
Bytes are the experts read from VRAM by the GPU (union of the layer, minus CPU experts, minus GPU-direct experts) times the exact bytes of one expert; the time is the step time of the layer
(block 0 stamps, so including the imbalance wait at the barrier). Ceiling 315 GB/s (measured, BENCHMARKS.md).

**Run A, `TONA_PLAN=cpu`: no GPU-direct experts, the steps read VRAM only (the clean figure).** Raw: `2026-11-16-expert-bandwidth-prof-plancpu.log`

| step | type | W | layers | experts per pass | MB per pass | ms per pass | GB/s | % of 315 GB/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| moe.gate_up | IQ2_S | 1 | 20 | 189 | 199 | 1.06 | 188 | 60 |
| moe.gate_up | IQ3_S | 1 | 10 | 97 | 137 | 0.65 | 209 | 66 |
| moe.gate_up | IQ3_XXS | 1 | 17 | 159 | 199 | 0.96 | 208 | 66 |
| moe.gate_up | IQ4_XS | 1 | 1 | 10 | 17 | 0.07 | 237 | 75 |
| moe.gate_up | all | 1 | 48 | 455 | 552 | 2.74 | 201 | 64 |
| moe.down | IQ4_NL | 1 | 39 | 371 | 342 | 1.64 | 208 | 66 |
| moe.down | Q2_0 | 1 | 9 | 84 | 39 | 0.29 | 135 | 43 |
| moe.down | all | 1 | 48 | 455 | 381 | 1.93 | 197 | 63 |
| moe.gate_up | IQ2_S | 2 | 20 | 298 | 313 | 1.67 | 187 | 59 |
| moe.gate_up | IQ3_S | 2 | 10 | 154 | 216 | 0.96 | 226 | 72 |
| moe.gate_up | IQ3_XXS | 2 | 17 | 252 | 317 | 1.49 | 213 | 67 |
| moe.gate_up | IQ4_XS | 2 | 1 | 16 | 28 | 0.11 | 253 | 80 |
| moe.gate_up | all | 2 | 48 | 720 | 874 | 4.23 | 207 | 66 |
| moe.down | IQ4_NL | 2 | 39 | 584 | 539 | 2.61 | 207 | 66 |
| moe.down | Q2_0 | 2 | 9 | 136 | 63 | 0.54 | 116 | 37 |
| moe.down | all | 2 | 48 | 720 | 601 | 3.15 | 191 | 61 |
| moe.gate_up | IQ2_S | 4 | 20 | 451 | 473 | 2.43 | 195 | 62 |
| moe.gate_up | IQ3_S | 4 | 10 | 236 | 333 | 1.39 | 240 | 76 |
| moe.gate_up | IQ3_XXS | 4 | 17 | 391 | 490 | 2.20 | 223 | 71 |
| moe.gate_up | IQ4_XS | 4 | 1 | 26 | 45 | 0.17 | 258 | 82 |
| moe.gate_up | all | 4 | 48 | 1104 | 1341 | 6.19 | 217 | 69 |
| moe.down | IQ4_NL | 4 | 39 | 894 | 824 | 3.97 | 208 | 66 |
| moe.down | Q2_0 | 4 | 9 | 210 | 97 | 0.83 | 117 | 37 |
| moe.down | all | 4 | 48 | 1104 | 921 | 4.79 | 192 | 61 |
| moe.gate_up | IQ2_S | 8 | 20 | 653 | 685 | 3.92 | 175 | 55 |
| moe.gate_up | IQ3_S | 8 | 10 | 345 | 485 | 2.43 | 199 | 63 |
| moe.gate_up | IQ3_XXS | 8 | 17 | 569 | 714 | 3.55 | 201 | 64 |
| moe.gate_up | IQ4_XS | 8 | 1 | 38 | 66 | 0.27 | 247 | 78 |
| moe.gate_up | all | 8 | 48 | 1605 | 1951 | 10.18 | 192 | 61 |
| moe.down | IQ4_NL | 8 | 39 | 1297 | 1195 | 6.37 | 188 | 60 |
| moe.down | Q2_0 | 8 | 9 | 308 | 142 | 1.46 | 97 | 31 |
| moe.down | all | 8 | 48 | 1605 | 1337 | 7.84 | 171 | 54 |

**Run B, default plan: 2 to 4 % of the experts are read by the GPU over PCIe inside the step; the table counts only VRAM bytes, so its rates are understated.** Raw: `2026-11-16-expert-bandwidth-prof-default.log`

| step | type | W | layers | experts per pass | MB per pass | ms per pass | GB/s | % of 315 GB/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| moe.gate_up | IQ2_S | 1 | 20 | 189 | 199 | 1.12 | 177 | 56 |
| moe.gate_up | IQ3_S | 1 | 10 | 97 | 137 | 0.66 | 206 | 65 |
| moe.gate_up | IQ3_XXS | 1 | 17 | 159 | 199 | 1.16 | 172 | 55 |
| moe.gate_up | IQ4_XS | 1 | 1 | 10 | 17 | 0.07 | 231 | 73 |
| moe.gate_up | all | 1 | 48 | 455 | 552 | 3.02 | 183 | 58 |
| moe.down | IQ4_NL | 1 | 39 | 371 | 342 | 1.82 | 188 | 60 |
| moe.down | Q2_0 | 1 | 9 | 84 | 39 | 0.33 | 120 | 38 |
| moe.down | all | 1 | 48 | 455 | 381 | 2.15 | 177 | 56 |
| moe.gate_up | IQ2_S | 2 | 20 | 298 | 313 | 1.72 | 182 | 58 |
| moe.gate_up | IQ3_S | 2 | 10 | 154 | 216 | 0.97 | 224 | 71 |
| moe.gate_up | IQ3_XXS | 2 | 17 | 252 | 317 | 1.78 | 178 | 56 |
| moe.gate_up | IQ4_XS | 2 | 1 | 16 | 28 | 0.11 | 250 | 79 |
| moe.gate_up | all | 2 | 48 | 720 | 874 | 4.58 | 191 | 61 |
| moe.down | IQ4_NL | 2 | 39 | 584 | 539 | 2.82 | 191 | 61 |
| moe.down | Q2_0 | 2 | 9 | 136 | 63 | 0.57 | 110 | 35 |
| moe.down | all | 2 | 48 | 720 | 601 | 3.39 | 177 | 56 |
| moe.gate_up | IQ2_S | 4 | 20 | 451 | 473 | 2.55 | 186 | 59 |
| moe.gate_up | IQ3_S | 4 | 10 | 236 | 333 | 1.40 | 237 | 75 |
| moe.gate_up | IQ3_XXS | 4 | 17 | 391 | 490 | 2.58 | 190 | 60 |
| moe.gate_up | IQ4_XS | 4 | 1 | 26 | 45 | 0.17 | 257 | 82 |
| moe.gate_up | all | 4 | 48 | 1104 | 1341 | 6.70 | 200 | 64 |
| moe.down | IQ4_NL | 4 | 39 | 894 | 824 | 4.39 | 188 | 60 |
| moe.down | Q2_0 | 4 | 9 | 210 | 97 | 0.89 | 109 | 35 |
| moe.down | all | 4 | 48 | 1104 | 921 | 5.28 | 174 | 55 |
| moe.gate_up | IQ2_S | 8 | 20 | 653 | 685 | 4.25 | 161 | 51 |
| moe.gate_up | IQ3_S | 8 | 10 | 345 | 485 | 2.48 | 195 | 62 |
| moe.gate_up | IQ3_XXS | 8 | 17 | 569 | 714 | 4.26 | 168 | 53 |
| moe.gate_up | IQ4_XS | 8 | 1 | 38 | 66 | 0.28 | 234 | 74 |
| moe.gate_up | all | 8 | 48 | 1605 | 1951 | 11.28 | 173 | 55 |
| moe.down | IQ4_NL | 8 | 39 | 1297 | 1195 | 7.30 | 164 | 52 |
| moe.down | Q2_0 | 8 | 9 | 308 | 142 | 1.57 | 90 | 29 |
| moe.down | all | 8 | 48 | 1605 | 1337 | 8.86 | 151 | 48 |

Pass times of the profile runs (not a speed result: the profile adds a read-back per pass), default plan: W1 23.3, W2 28.7, W4 35.5, W8 51.7 ms.
Experts per pass (default plan, W = 1 / 2 / 4 / 8): VRAM hits 455 / 720 / 1,104 / 1,605, GPU-direct 10.6 / 13.2 / 26.5 / 63.0, CPU 14.3 / 34.7 / 78.3 / 181.4.

Findings.
1. **The experts read at 55 to 65 % of the ceiling** (plan cpu: gate_up 201 / 207 / 217 / 192 GB/s, down 197 / 191 / 192 / 171 at W = 1 / 2 / 4 / 8). Gap against 315 GB/s at W = 8: gate_up 10.18 against 6.19 ms, down 7.84 against 4.24 ms
   (7.6 ms together); at W = 1 2.74 + 1.93 against 1.75 + 1.21 ms (1.7 ms).
2. **Gap by type at W = 8 (ms, plan cpu):** IQ4_NL down 2.58 (39 layers, 188 GB/s), IQ2_S gate_up 1.75 (175), IQ3_XXS 1.28 (201), Q2_0 down 1.01 (97: the worst rate, 31 % of the ceiling), IQ3_S 0.89 (199), IQ4_XS 0.06 (247).
   At W = 1: IQ4_NL 0.55, IQ2_S 0.43, IQ3_XXS 0.33, IQ3_S 0.22, Q2_0 0.17.
3. **GPU-direct experts (PCIe reads inside the step) cost 0.5 ms at W = 1, 1.0 at W = 4, 2.1 ms at W = 8** (default plan against plan cpu, the same steps): 63 experts at W = 8 cost about 33 us each against 8 us for a VRAM expert.
   This belongs to the misses (prefetch, planner), not to the kernels; plan cpu moves that time into host.wait (pass 24.2 against 23.6 ms at W = 1), so it is not a free gain.
4. **A fixed cost of about 10 us per step** (96 expert steps per pass, about 1 ms per pass at every width): the one IQ4_XS layer (17 MB) takes 70 us at W = 1 against 54 us at 315 GB/s; the IQ4_NL layers 41 us for 8.8 MB, which
   is 31 us at the probe's 286 GB/s plus 10. Barrier imbalance is part of it (#128: 0.5 ms at W = 1, 1.0 ms at W = 8 for both steps).
5. One of the 12 profile blocks of the default run (round 3, W = 8) had 14.5 ms more in moe.down (pass 66.9 against 52.3 ms): a single stall, not seen in the plan cpu run, not analysed; the median excludes it.

## Result 2: where the kernel loses it (`tona-wide-probe --abl`)

256 MB of the real expert rows, L = 8 lanes per row, 32 blocks x 16 waves as k_mega, GB/s of weight bytes. The ablation kernels (`tona_wide_probe_abl.h`) run the engine's row loop with parts left out.
The probe of #128 (`--exps`) launched `cus` = 16 blocks, half of k_mega's 32: fixed here (`TONA_PROBE_BLOCKS_PER_CU`, default 2); the 16-block numbers stay in `...-probe-16blocks.log`, the 32-block ones in `...-probe-32blocks.log`.
Four runs (`...-abl-r1.log`, `-r2`, `-r3`, `-r4`; r3 and r4 are the kernel after the refactor for the code gates, nb = 1 within 3 GB/s except one IQ4_NL mode-2 outlier of 201): the nb = 1 columns agree within 1 GB/s; the nb = 4 and 8 values of the decode-only and dot modes jump by up to 60 GB/s between the runs (clock or compiler state, not understood),
so only nb = 1 is quoted for them, and ranges for the row_q8 column.

| Type (step) | loads only | decode (real, tables in LDS) | decode + dot, x in registers | row_q8 (x in LDS), nb 1 / 4 / 8 | engine, plan cpu, W 1 / 8 |
|---|---:|---:|---:|---|---|
| IQ3_XXS (gate_up) | 289 | 188 | 191 | 168 / 120 / 92 | 208 / 201 |
| IQ2_S (gate_up) | 280 | 161 | 158 | 132 / 79..97 / 54..76 | 188 / 175 |
| IQ3_S (gate_up) | 284 | 194 | 191 | 161 / 100..113 / 66..89 | 209 / 199 |
| IQ4_XS (gate_up) | 294 | 283 | 283 | 282 / 183..194 / 105..133 | 237 / 247 |
| IQ4_NL (down, raw pipeline) | 270 | 261 | 260 | 286 / 162..188 / 111..130 | 208 / 188 |
| Q2_0 (down) | 197 | 155 | 145 | 139 / 83 / 55..61 | 135 / 97 |

Reading it.
- **The loads are not the limit**: 270 to 295 GB/s of 315 for every type but Q2_0, with the 2-byte `ld4_a2` loads (7 to 9 load instructions per sub-block) as they are.
- **Decode is the limit at nb = 1 for the grid types**: it removes 35 to 43 % of the loads-only rate (IQ2_S 280 to 161, IQ3_XXS 289 to 188, IQ3_S 284 to 194); almost nothing for IQ4_NL and IQ4_XS (lookup by v_perm in registers), 21 % for Q2_0.
  The dot products with x in registers cost nothing at nb = 1.
- **Reading x from LDS is the limit from nb = 2**: row_q8 against decode + dot with x in registers: -12 % (IQ3_XXS) at nb = 1, about -40 % at nb = 4, -60 % and more at nb = 8. The engine computes most experts for 1 or 2 tokens
  (56 to 62 % of the experts have one token, #128 part 3), so at W = 8 the effective nb is about 2.
- IQ4_NL with the raw-load pipeline reaches 286 GB/s at nb = 1 (above its loads-only figure: the pipeline hides the next round's latency) but 208 in the engine: the remainder is the per-step cost (finding 4) and the shape of the down matrix
  (20 sub-blocks per row over 8 lanes: three rounds, the third with half the lanes idle). The grid types run faster in the engine than in this probe (208 against 168 at nb = 1): unexplained, so the probe is a relative instrument.
- Sign masks by arithmetic instead of the m8 table in LDS (mode 4, IQ2_S only): 161 to 177 / 181 at nb = 1 (+10 to 12 %, both runs) and up to +30 % at nb = 8 in the decode-only kernel: a small part of the decode, a cheap bit-exact lever.
- Q2_0 is limited by its loads already (5 two-byte loads per 9 bytes; loads-only 197).
- Hardware counters: `rocprofv3 --pmc` runs here, but on gfx1200 every counter I tried except SQ_BUSY_CYCLES reads 0 (VALUBusy, SQ_INSTS_*, LDSBankConflict, TA_BUSY, L2CacheHit, WAVE_*_WAIT; 4 sets, 18 dispatches each; script `pmc.sh`
  was kept out of the repository): no occupancy, VALU or LDS-conflict figure. VGPRs: engine kernels 192 (8 waves per SIMD), the probe's 160 to 176.

## Step 2: ranked levers (nothing built yet)

Gains are ms per pass at W = 1 / 4 / 8 on the default plan; the estimates come from the tables above, they are not measurements of a change.

| # | Lever | Bit-exact | Gain (ms per pass) | Basis | Prior art and what was tried |
|---|---|---|---|---|---|
| 1 | **Fewer GPU-direct reads and misses in the expert steps** (they cost 0.5 / 1.0 / 2.1 ms; better prefetch hit rate or a cheaper way to take a miss: e.g. a CPU expert instead of a PCIe read for the largest imbalance, or planning that keeps GPU-direct experts out of the critical step) | yes | up to 0.5 / 1.0 / 2.1, in practice half | Result 1 finding 3 | #126 optimised the prefetch depth and planner; arXiv 2511.10676 (earlier expert prediction), 2610.34440 (unique experts); not a kernel change |
| 2 | **x from LDS at nb >= 2 for the grid types** (the largest kernel loss: -40 % at nb = 4): fewer LDS reads per vector (one ds_read_b128 pair per sub-block shared by the two g, meta packed with x so one 16-byte read less) or the first token of an expert by the nb = 1 path | yes | 0 / 0.5 / 1.0 (if half of the x cost goes) | row_q8 against x in registers, Result 2 | #128 variant A (two rows per lane sharing x): +0.4 to 1.4 ms, VGPR spills 54 to 90 / 109 to 237; NOX probe +5 to +25 %; no published GPU kernel found that shares x across rows with an LDS budget of 32 KB |
| 3 | **Grid decode cost at nb = 1** (IQ2_S, IQ3_XXS, IQ3_S: 47 layers of gate_up): sign by arithmetic for IQ2_S and IQ3_S (+10 to 12 % decode-only), grid lookups as 32-bit pairs, cheaper scale extraction | yes | 0.1 / 0.2 / 0.4 (IQ2_S sign only); 0.3 / 0.6 / 1.0 for all three types if half the decode loss goes | mode 1 and 4 | #128 "cheaper sign" analysed (7 % of VALU, not built); ik_llama.cpp #2633, #2676 (CPU: sign from parity, 4-row repack); llama.cpp keeps the grid tables in constant memory and builds signs with `__vcmpne4` |
| 4 | **Pipeline the raw loads of the grid types without VGPR cost** (IQ4_NL reaches the loads-only rate in the probe because of it) | yes | 0.3 / 0.6 / 1.0 for the three grid types if they reach 80 % of the loads-only rate | probe: pipelined IQ2_S +28 %, IQ3_S +10 % | tried twice in the engine and failed on registers (#126: +0.4 ms, #128 G = 1: 0); needs fewer live VGPRs elsewhere in k_mega (192 VGPR, 98..141 spills at <4>/<8>) first |
| 5 | **Q2_0 down (9 layers) loads**: 97 to 135 GB/s, loads-only 197: aligned 16-byte load of the 64 weights + separate scale (needs a repack of the file type, a format decision), or one unaligned 4-byte load per word | yes (bytes only move) | 0.1 / 0.3 / 0.6 to 0.9 | Q2_0 rows of the tables | Mac type 1018 (IQ3_XXS aligned repack) is the same idea; probe `TONA_LD4_LIE` on IQ3_XXS gave nothing because loads are not the limit there, for Q2_0 they are |
| 6 | **Per-step cost of about 10 us**: start the loads of the next step's weights (known before the barrier: the experts of the layer) while waiting at the barrier; or one expert-level flag so a block starts its down rows when its gate_up rows are done | yes | 0.3 / 0.5 / 0.8 | finding 4; #128 imbalance wait | #128 bound for the fused step 0.3 to 0.5 ms at W = 4; `s_prefetch_data` had no effect in the steady state (#128 variant C), across the barrier is untested |
| 7 | **IQ4_NL down at nb >= 4 and the NB = 8 instance** (pipeline off above 4 vectors, 39 layers, the largest absolute gap 2.58 ms at W = 8) | yes | 0 / 0.3 / 0.8 | probe nb = 1 286 against engine 188 to 208 | #126 Results 3 (pipeline for NB <= 4 only; +0.06 ms at 8) |

Not a lever: more bytes in flight through scalar prefetch (#128 C), two rows per lane (#128 A), WMMA int8 for the experts (#128 part 2/3: 0.1 to 0.9 ms upper bound, machinery +1 to 2.7 ms).

Recommendation for the coordinator (no larger build started): lever 3 for IQ2_S (sign by arithmetic) is small and bit-exact and can be tried first (about 60 lines in `matvec_q8_iq.h`, then the width curve with hash);
levers 2 and 4 are the large kernel ones but both have failed once on VGPRs and need a spill budget for k_mega first; lever 1 is the largest single item at W = 8 and sits in the host side (miss handling), so it belongs to whoever owns #127;
lever 5 needs a decision on a file type.

## Prior art

Change type: `none` for this measurement file (diagnostics and probe only); the levers above are `technique`.

- [llama.cpp #19478](https://github.com/ggml-org/llama.cpp/pull/19478), [#24386](https://github.com/ggml-org/llama.cpp/pull/24386), [#31804](https://github.com/ggml-org/llama.cpp/pull/31804), status: merged / open (book rows, abstracts read). RDNA4 MMVQ warp-count and row tables; no IQ-grid decode work. Not adopted (batch 1 or Vulkan).
- [ik_llama.cpp #2633, #2676](https://github.com/ikawrakow/ik_llama.cpp), status: merged (book row). CPU kernels for IQ3_XXS / IQ2_S / IQ3_S: scales as integers, sign masks from parity, 4-row repack. Idea for lever 3 and 5.
- [Vec-LUT 2512.06443](https://arxiv.org/abs/2512.06443), [CodeGEMM 2512.17970](https://arxiv.org/abs/2512.17970), [Proteus (ICML 2026)](https://icml.cc/virtual/2026/poster/63948), status: abstracts in the book. Table-free or table-batched decode of codebook formats; relevant to lever 3 (LUT pressure in LDS), no kernel for the IQ formats on RDNA4.
- gfx12 has no asynchronous global-to-LDS copy that I could find (web search 2026-11-16: only `global_load_tr` and the block loads for gfx12; the async-to-LDS patch is for pre-GFX12), so staging weights through LDS would cost registers as before.
- Own: #126 (research/measurements/2026-11-12-verify-pass.md), #128 (2026-11-12-expert-kernels.md), #130 (2026-11-13-pc-q6k-tile.md, 2026-11-15-pc-q6k-tile-spills.md).

Search: tools/papers/scan.py topics v1; topics: quantization, amd-rocm, megakernel; terms: MMVQ, IQ2_S, IQ3_XXS, IQ4_NL, grid lookup, LDS bank conflict, WMMA int8, global_load_lds; window: 2025-11-16..2026-11-16 (12 months);
sources: research/papers.md and papers-screened.tsv (grep, the book covers to 2026-11-11), `gh search prs` on ggml-org/llama.cpp (terms: mmvq iq, RDNA4 MMVQ, mul_mat_vec_q_moe), one web search. `scan.py` itself was not run in this session (ungeprueft for the last five days).

## Step 3: levers built (#160, coordinator order 3, 6, 7, 2)

Method: width curve = `tona-gpu-gen --batch-bench 400`, widths 1/2/4/8, `TONA_MEGA_PROF=1 TONA_PROF_LAYERS=1 TONA_BENCH_HASH=1`, 3 rounds A B A B A B (median pass ms), logits hash per width compared; binaries snapshotted per code state;
ISA: the machinery of `scripts/isa-diff.py` on copies of the `*.hip.o` of the base (e5975ec) and of the lever build.
Why this size: the effect is 0.1 to 0.3 ms of a 24 to 54 ms pass, so one width curve (3 rounds; spread of the pass time between rounds 0.1 to 0.3 ms) with the per-step profile answers whether a lever does anything
before the server A/B; ab.py then runs workload `short` only (decode of the code request included; known noise of decode 0.23 % and 0.6 %), early stopping on, at most 6 rounds.

### Lever 3: sign masks by arithmetic (commit 75bc5b8: IQ2_S and IQ3_S)
- Curve, IQ2_S only against base (`l3a`): pass ms W1/2/4/8 -0.02 / -0.18 / -0.27 / +0.03; hash equal. All three types (`l3c`, IQ3_XXS with the parity bit from popcount) against base: +0.07 / -0.10 / -0.23 / -0.05.
  Per type gate_up step time (ms, base to new, W 1/2/4/8): IQ2_S 1.13 to 1.14 / 1.72 to 1.60 / 2.55 to 2.37 / 4.26 to 4.34; IQ3_S 0.67 to 0.68 / 0.96 to 0.92 / 1.40 to 1.35 / 2.49 to 2.29;
  IQ3_XXS 1.16 to 1.21 / 1.77 to 1.74 / 2.57 to 2.52 / 4.26 to 4.39 (slower at W1 and W8: not kept).
- The probe predicted +10 to 12 % decode for IQ2_S at nb = 1; the engine shows nothing at W1 (the step is not decode-bound there), a gain at W2/W4, +0.08 ms at W8 (k_mega<8>; VGPR and spill counts identical to base).
- ISA: touched kernels are the k_mega instances (code +120 to +290 instructions), k_matvec_q8<22>, k_gemm_q<22>, k_gemm_q2<22> and the probe kernels; the other kernels differ only in one address constant (the table offset; shown for k_matvec_q8<20>).
- Server A/B (`2026-11-16-expert-bandwidth-l3-ab.md`, base against the lever build, workload short, 6 rounds, early stopping on): decode@2000 +0.16 % [-0.19, +0.51] equal; decode@2000:write (code) +0.10 % [-1.12, +1.3] open;
  prefill unchanged; SHAs equal. Kept: bit-exact, curve gain 0.2 to 0.3 ms at W2/W4, no loss at W1; it is not outside the server noise.

### Lever 6: prefetch of each wave's first unit before x is staged (patch `2026-11-16-expert-bandwidth-lever6.patch`, not committed)
- `s_prefetch_data` (31 lines) for the first unit of every expert task, before `matvec_stage_x`. Curve against the lever 3 build (`l6`): pass ms +0.16 / +0.79 / +0.86 / +0.28 at W1/2/4/8, hash equal; slower at every width.
  moe.gate_up +0.07..+0.17 ms, moe.down +0.03..+0.07 ms, and the rest of the pass grows too (code generation of k_mega: SGPR spills of one k_mega<4> instance 121 to 199, VGPR 54 to 59). Not kept.
  A variant with the prefetch in a noinline function (keeps the inlined code of k_mega) (`l6b`): see the result below.

- Result of the noinline variant (`l6b`, same method): pass ms +2.00 / +2.41 / +2.56 / +1.84 at W1/2/4/8 (moe.gate_up +0.17..+0.34, moe.down +0.18..+0.22 per pass, the call costs more than the prefetch brings); hash equal.
  Both forms are slower at every width: **lever 6 dropped.** Not tried: prefetch across the barrier (needs the same unit resolution in `step_end`, the same code generation risk). Principle 8: tried twice here (inline, noinline), `s_prefetch_data` in the steady state
  in #128 variant C; the per-step cost of about 10 us stays.

### Lever 7: IQ4_NL down at nb >= 4 / the NB = 8 instance (analysed, not built)
- The engine data show no headroom that belongs to NB = 8: IQ4_NL down runs at 208 / 191 / 188 / 188 GB/s at W = 1 / 2 / 4 / 8 (plan cpu, `...-prof-plancpu.log`); the rate at W = 8 (k_mega<8>) equals that at W = 4 (k_mega<4>).
  The earlier pipeline for 8 vectors was +0.06 ms (#126).
- The only IQ4_NL-specific loss is the shape of the down matrix (20 sub-blocks over 8 lanes: the third round has half the lanes idle). A bit-exact form exists (4 physical lanes per row, two accumulators for the virtual lanes j and j + 4, local add instead of the first
  butterfly step), but the probe with L = 4 (`...-probe-32blocks.log`, type 20) gives 274 / 217 / 181 / 161 GB/s at nb = 1 / 2 / 4 / 8 against 270 / 263 / 201 / 144 for L = 8: no gain at the nb of 1 to 2 where the engine runs, and #128 measured L = 4 slower
  in the engine (down 2.32 against 2.21 ms). Not built.

### Lever 2: x from LDS at nb >= 2 (analysed, not built; the VGPR budget is zero)
- ISA budget (`scripts/kernel-regs.py`-style listing of the base objects): every k_mega instance is at the 192 VGPR cap (8 waves per SIMD, no more possible: above 192 the two blocks of a WGP are not resident together and the barriers hang, comment in `mega_kernel.h`)
  with VGPR spills 46 to 62 (<1>), 51 to 54 (<4>), 59 to 110 (<8>) and SGPR spills 83 to 163. #128 variant A (x shared by two rows) raised them to 90 (<4>) and 237 (<8>) and cost 0.4 to 1.4 ms.
  Any change that keeps more x words alive across the vector loop adds spills first; the lever 6 attempt shows that even a few extra instructions before the loop move one k_mega<4> instance from 121 to 199 SGPR spills and cost 0.8 ms.
- What the probe gives for it: x from registers instead of LDS (mode 2 against mode 3, `...-abl-r*.log`): +12 % at nb = 1, +40 % at nb = 4 for the grid types. The LDS reads per vector and sub-block are 2 x ds_read_b128 + 1 ds_read_b32 (the meta fields s0 and s1 are unused for the signed grid types);
  they cannot be shared between lanes (each lane of a row has its own sub-block) but between the 4 rows of a wave (already a broadcast). No design that lowers the count without more live registers was found, so nothing was built before the spills of k_mega are reduced (a separate task).
