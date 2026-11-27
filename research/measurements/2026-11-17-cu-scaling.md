# Which steps of the verify pass scale with CUs: 16 against 32 CUs on the reference GPU (#164, #163 C2)

Plan, written before the first run. Results below the line `## Results`. Build `work/build/b164` (main `0fd36c4`), the same
binary in both arms.

## Question and decision

The R9700 estimate (`/var/tmp/claude/r9700/tona-scaling.md`) assumes per-step rules: some steps have fixed latency, some scale
with CUs x clock, some with bandwidth. The reference card cannot change its bandwidth, but it can halve its CUs
(`TONA_FREE_CUS=16` sets `HSA_CU_MASK=0:0-15` and launches 16 blocks instead of 32, `engine/gpu/mega_init.cpp:25-33`; 16 CUs are
8 whole WGPs, the 28-CU mask worked on 2026-10-23, 30 aborted at step 0). With bandwidth at 315 GB/s in both arms, per step:

- time unchanged (and above bytes / 315 GB/s): **fixed** latency; time x2: **CU-bound**; in between: mixed (the share that
  scales is (t16 - t32) / t32 for a step that is fully CU-bound at 1.0).
- Decision it feeds (note section 6): fixed share of the W4 pass >= 30 %: R9700 code **decode** about 135 tok/s; <= 20 %: about 155;
  `host.wait` unchanged at 16 CUs: GPU routing first.

**Confound to measure first:** the premise "bandwidth stays 315 GB/s" holds only if 16 CUs can still pull 315 GB/s. If they
cannot, bandwidth-bound steps also slow down at 16 CUs and look CU-bound. So step A below measures the VRAM read bandwidth at both
masks.

## Scope (every GPU job `--kind exclusive` through `scripts/gpu_queue.py`; no server, no port; runs whenever the queue allows)

| Step | What runs | Rounds | Duration |
|---|---|---|---|
| A bandwidth control | `work/build/gpubench vram` (VRAM read, write, copy; the micro bench of 2026-10-22-micro), once without a mask and once with `HSA_CU_MASK=0:0-15` | 2 per arm (alternating) | 2 min |
| B batch bench 32 CUs | `tona-gpu-gen MODEL "$(cat scripts/bench-long-code.txt)" --mega --batch-bench 400` with `TONA_MEGA_PROF=1 TONA_PROF_LAYERS=1 TONA_PLAN=cpu TONA_BENCH_WIDTHS=1,4,8 TONA_BENCH_REPEAT=3`, `TONA_EXPERT_PROFILE=work/profile/experts-coder.bin`, `OMP_NUM_THREADS=6` | 1 process of 3 rounds | about 6 min incl. 20 s load |
| C batch bench 16 CUs | the same with `TONA_FREE_CUS=16` added; abort counter checked after the first width | 1 process of 3 rounds | about 8 min (slower passes) |
| D repeat | B and C once more in the order C, B (ABBA overall) to see run-to-run and order effects | 1 + 1 | about 14 min |
| E program dump | `scripts/program-dump.sh` is not run; unit counts per step are read from the profile log (`TONA_MEGA_PROF` labels) and the code, only if a step is not clear from its time | | |

`OMP_NUM_THREADS=6` is set on purpose in both arms (the CPU experts do not run: `TONA_PLAN=cpu` keeps the experts of the
profile in VRAM except the CPU-computed ones; the per-layer table needs the exact-count mode of #160). The 9950X thread
question is answered in `2026-11-17-9950x-baseline.md`, not here; the same value in both arms keeps this a CU test.
`TONA_PLAN=cpu` and the per-layer expert counts are those of `2026-11-16-expert-bandwidth.md` run A, so that the 32-CU arm is
checked against that file (and the 9600X against the 9950X: `host.wait` and `host.y` may move).
Evaluation: `scripts/expert_bw.py MODEL LOG...` for the expert steps (GB/s per type and width) and the step table of
`tona-gpu-gen` for all other steps (ms per pass per label at W = 1, 4, 8).

## Why this size

- Widths 1, 4, 8: W1 is the latency floor, W4 and W8 span the production verify pass (W5 to 6; the model weights the mix 0.62 / 0.38
  of W4 / W8). W2 adds nothing to the question (it lies between W1 and W4) and is left out against the 8 min of the #160 run
  with four widths.
- 400 positions after 64 warm-up tokens: 50 to 340 passes per width and 48 layers each, enough for per-layer means; the spread of the
  pass time between rounds is below 1 % (`2026-11-16-expert-bandwidth.md`), so 3 rounds in one process and one repeat of the pair
  (ABBA) give the run-to-run spread. The effect to separate is a factor between 1 (fixed) and 2 (CU-bound), far above 1 %.
- Not long contexts: attention at 130K is not in the verify-pass model of the note; its step times at 2K are in it.

## Expectation (every number with tolerance and source)

Source: the scaling model (`tona-scaling.md` sections 3.2, 3.3, 3.5): per cycle of 48.3 ms (W4 / W8 mix) 11 to 12 ms fixed (does not shrink),
about 22 ms bandwidth, 8.8 ms CU x clock, 3.3 ms wave-count, 2.5 ms vanishing with resident experts. The 28-CU check of 2026-10-23
(fake experts): W1 +1.5 %, W4 +4.8 % at 28 of 32 CUs, where a fully CU-bound pass would be +14 %.

| What | Expectation at 16 against 32 CUs | Reason |
|---|---|---|
| A: VRAM read, 16 CUs | 300 to 315 GB/s (at most -5 %); below 280 means bandwidth-bound steps are confounded and C is read per step against bytes at the measured rate | RDNA4 needs few CUs for GDDR6 at 128 bit; unverified, hence the control |
| whole verify pass | W1 +8 to +25 %, W4 +20 to +45 %, W8 +25 to +55 % (a pass of 23.8 / 35.9 / 54.7 ms: a CU-bound part of 2 / 12 / 20 ms doubling, bandwidth and fixed unchanged) | note 3.3 split, 2026-10-23 check |
| dense large (dn.in, qsa.in, head.out, dn.out, qsa.out, gr.up) | W1 +0 to +10 % (fixed + bytes), W4 and W8 +15 to +60 % (growth above W1 / cu) | note 3.3 |
| experts (moe.gate_up, moe.down) | W1 +10 to +30 %, W8 +40 to +85 % (expert kernels run at 55 to 65 % of the ceiling because of ALU and LDS work, which doubles; the bytes part does not) | `2026-11-16-expert-bandwidth.md`, ablation |
| dn.recur | +0 to +30 % (768 units: 2 rounds on 32 CUs, 4 on 16; latency-bound waves) | note 3.3 |
| attention (2K), element-wise, router, gr.down | +0 to +25 % (small, latency-bound, units below the grid) | note 3.3 |
| host.wait, host.y, moe.copy | unchanged +-5 % (CPU and PCIe; GPU-side waiting may move host.wait) | note 3.3 |
| outcome class | fixed share of W4 pass (steps whose time moves by less than 10 %) between 20 and 35 % | note: ~11.5 of 35.9 |

A result outside a range is a finding: it says which assumption of the model is wrong (written per row in the results).

## Active tooling

`TONA_MEGA_PROF=1` and `TONA_PROF_LAYERS=1` (the profile read-back per pass adds a hipMemcpy: pass times of a profile run are not
speed results, but identical in both arms), `TONA_PLAN=cpu`, `TONA_BENCH_WIDTHS=1,4,8`, `TONA_BENCH_REPEAT=3`,
`TONA_EXPERT_PROFILE=work/profile/experts-coder.bin`, `OMP_NUM_THREADS=6`, no `TONA_DETERMINISTIC` (it turns the wide path off);
`TONA_FREE_CUS=16` in the C arm only. `hw_report` and `quiet_check` before and after every job (the desktop holds about 1.4 GB
VRAM; with 16 CUs left to the desktop its frames are not at risk, the reverse of the 11-15 danger).

## Test conditions

`hw_report.txt` of the job (raw files in `2026-11-17-cu-scaling-logs/`): AMD Ryzen 9 9950X (16 cores / 32 threads), RX 9060 XT
gfx1200 (15.9 GiB), 60.5 GiB RAM, Fedora 44, kernel 7.3.4, rocm 10.1.0, commit `2dec816` (clean; build `b164` is main `0fd36c4`, the
merge since touches only Metal files and the queue). Quiet check before: warning only for 1.27 GiB VRAM of the desktop; load average
2.9 after the queue's own build jobs. After each run `quiet_check` shows 1.3 to 2.8 GiB "held by others" and swapping of 1 to 11 pages in 1 s: that is the
server of the run still shutting down (the 11-14 file saw the same), the desktop before each run held 1.37 GB (`tc.log`).

**What went wrong first, and the deviation from the plan (all recorded).** Two earlier jobs of this file (first with `TONA_EXPERT_SLOTS` unset)
failed or were disturbed: at 20:57 the batch bench saw only 9 expert slots free, and at 21:03 to 21:14 the 16-CU runs died with out of memory (GTT 28.8 and
30.7 of 32.5 GB in use) while another program's `tona-gpu-gen` ran on the GPU during my exclusive slot (a binary under another worker's build
directory, started outside the queue; seen in the process list, not killed). The 32-CU runs of that attempt loaded with 99 slots per layer
(14.79 GB VRAM), too close to the free VRAM at a desktop of 1.6 GB. A third job (killed by the coordinator at 21:16 after the desktop crash; its
data discarded) was replaced by this one: **`TONA_EXPERT_SLOTS=87` in both arms** (not in the plan; 87 slots per layer is what the 2026-11-16 expert
bandwidth file ran with, 13.61 GB VRAM, so the 32-CU arm is comparable to it, and it leaves room for the desktop). Step A (VRAM bandwidth control) is
from the first job, run at 20:57 on a quiet machine, before any disturbance. Step E (program dump) was not needed. The job went through the queue with the desktop
guard (exit 75 would requeue; it did not trigger: `q.log` rc 0 at the first attempt).

## Results

Raw: `2026-11-17-cu-scaling-logs/` (`batch-<CUs>-<n>.log`, `gpubench-vram.log`, `agg.py.txt` evaluates the profile blocks, `cu3.sh.txt` is the job). Order 32, 16, 16, 32 CUs;
3 rounds per width and run, so 6 values per arm and width; "min-max" below is over those 6 rounds. Profile times are the step sums of a `TONA_MEGA_PROF` pass
(profile overhead is the same in both arms). Both arms: 87 slots, `TONA_PLAN=cpu`, `OMP_NUM_THREADS=6`, 9950X.

### A. Does 16 CUs still reach the VRAM bandwidth? Yes.

`gpubench vram`, buffers of 128 MiB and more (random data, 7 repetitions, medians): **316 GB/s with all CUs against 311 to 314 GB/s with `HSA_CU_MASK=0:0-15` (-0.6 to -1.3 %)**; write
277 against 267 GB/s, copy 282 against 270 GB/s. Cache-sized buffers do slow down with the mask (L2 range 3.6 to 4.0 TB/s against 2.7 to 3.2 TB/s; Infinity Cache 1.19 TB/s against 1.0 TB/s),
so the mask is in effect. Expectation was 300 to 315 GB/s: confirmed. **The premise of the test holds: at 16 CUs bandwidth is unchanged, so steps that
slow down are not bandwidth-limited by CU count.**

### B. Pass time (wall, `batch-bench`, ms per pass; median, min-max over 6 rounds)

| W | 32 CUs | 16 CUs | 16 against 32 | expectation (plan) |
|---|---|---|---|---|
| 1 | 23.76 (23.74 to 24.45) | 26.06 (26.04 to 26.62) | **+9.7 %** | +8 to +25 %: within, at the bottom |
| 4 | 36.78 (36.71 to 38.33) | 46.41 (46.31 to 46.46) | **+26.2 %** | +20 to +45 %: within |
| 8 | 55.70 (55.36 to 65.44; one 65.4 outlier round at 32 CUs, the rest 55.4 to 57.6) | 75.31 (75.20 to 75.57) | **+35.2 %** | +25 to +55 %: within |

The 16-CU arm is very stable (spread 0.2 to 0.4 %). The 32-CU first round of each run is 0.4 to 3 % slower than the others (warm-up, W = 1 and W = 4).
Cross-check: the 32-CU steps equal the 9600X run A of `2026-11-16-expert-bandwidth.md` (moe.gate_up all 2.74 ms at W = 1 and 10.18 ms at W = 8, moe.down all 7.84 ms at W = 8; here 2.76, 10.06, 7.87), so the CPU swap does not change these GPU steps.

### C. Per step (profile ms per pass, medians; ratio = 16 CUs / 32 CUs; all 6 rounds in `agg.py.txt` output)

| step | W1 32 / 16 | ratio | W4 32 / 16 | ratio | W8 32 / 16 | ratio |
|---|---|---|---|---|---|---|
| moe.gate_up | 2.76 / 4.00 | 1.45 | 5.95 / 9.68 | 1.63 | 10.05 / 18.02 | 1.79 |
| moe.down | 1.93 / 2.38 | 1.23 | 4.82 / 7.12 | 1.48 | 7.87 / 13.02 | 1.66 |
| dn.in | 3.98 / 4.03 | 1.01 | 4.47 / 5.32 | 1.19 | 5.30 / 6.70 | 1.26 |
| dn.out | 1.83 / 1.98 | 1.08 | 2.13 / 2.75 | 1.29 | 2.63 / 4.02 | 1.53 |
| gr.down | 1.82 / 1.82 | 1.00 | 2.39 / 2.69 | 1.13 | 3.02 / 3.95 | 1.31 |
| gr.up | 1.66 / 1.64 | 0.98 | 2.25 / 2.25 | 1.00 | 2.67 / 2.83 | 1.06 |
| head.out | 1.61 / 1.57 | 0.98 | 1.69 / 1.93 | 1.14 | 1.82 / 2.05 | 1.13 |
| qsa.in | 1.22 / 1.27 | 1.04 | 1.38 / 1.69 | 1.22 | 1.90 / 2.42 | 1.27 |
| qsa.out | 0.61 / 0.64 | 1.06 | 0.72 / 0.89 | 1.23 | 0.89 / 1.31 | 1.47 |
| qsa.attn | 0.14 / 0.20 | 1.47 | 0.25 / 0.51 | 2.05 | 0.43 / 0.94 | 2.20 |
| dn.recur | 0.93 / 0.87 | 0.93 | 1.22 / 1.39 | 1.14 | 1.66 / 2.05 | 1.24 |
| router | 0.62 / 0.62 | 1.00 | 0.93 / 1.05 | 1.14 | 1.31 / 1.62 | 1.23 |
| combine | 0.68 / 0.90 | 1.32 | 1.31 / 1.53 | 1.17 | 2.10 / 2.50 | 1.19 |
| moe.sum, moe.act, moe.shared, mix, dn.conv (small) | 0.27 to 0.42 each, ratios 0.95 to 1.53 | | 0.36 to 0.44, ratios 1.2 to 2.1 | | 0.5 to 1.3, ratios 1.2 to 1.7 | |
| host.wait | 0.72 / 0.66 | 0.92 | 1.22 / 1.18 | 0.97 | 1.73 / 2.42 | 1.40 |
| host.y (waiting for the CPU experts, overlaps the GPU steps) | 0.82 / 0.67 | 0.82 | 3.06 / 2.13 | 0.70 | 7.31 / 4.27 | 0.58 |

Expert kernels (`scripts/expert_bw.py`, bytes read from VRAM per pass over the step time): at 32 CUs moe.gate_up 200 / 226 / 194 GB/s and moe.down 197 / 191 / 170 GB/s at W = 1 / 4 / 8
(the 11-16 values, 55 to 72 % of 315); at 16 CUs gate_up 138 / 138 / 108 and down 160 / 129 / 103 GB/s: the expert kernels are limited by CU work (grid decode, LDS), not by bytes.

Classes by ratio (all steps except host.* and moe.copy, profile sums, t32 -> t16 in ms): W1: unchanged (< 1.10) 15.57 -> 15.66, mixed 5.94 -> 8.05, CU-bound (>= 1.5) 0.37 -> 0.57;
W4: unchanged 2.55 -> 2.55, mixed 22.84 -> 28.75, CU-bound 6.71 -> 11.19; W8: unchanged 2.69 -> 2.86, mixed 20.18 -> 25.26, CU-bound 23.09 -> 39.44.
Many "unchanged" steps at W1 are bandwidth-bound (dn.in 3.98 ms for 1,108 MB = 3.5 ms at 315 GB/s), which this test cannot tell from latency; the split below uses the bytes.

### D. Pass split: bytes, CU-scaling and fixed

Assumption of the split (stated, not measured): the part that scales with CUs is the added time Δ = t16 - t32 (it doubles at half the CUs); bytes time = bytes / 315 GB/s (dense 3.52 GB
with the Q8_0 hyper-connection weights, `docs/backends/reference-platform.md`; experts read from VRAM from the table above); the rest is fixed latency and bandwidth inefficiency, which this test cannot separate. Profile totals:

| | W1 | W4 | W8 |
|---|---|---|---|
| pass (profile, 32 CUs) | 23.5 ms | 36.4 ms | 54.7 ms |
| CU-scaling part (Δ) | 2.3 ms (10 %) | 9.4 ms (26 %) | 19.7 ms (36 %) |
| bytes / 315 GB/s | 4.45 GB: 14.1 ms (60 %) | 5.78 GB: 18.4 ms (50 %) | 6.81 GB: 21.6 ms (40 %) |
| fixed (rest) | 7.1 ms (30 %) | 8.6 ms (24 %) | 13.4 ms (24 %) |

Decision rule of the plan (fixed share of the W4 pass): >= 30 % gives about 135 tok/s, <= 20 % about 155: **24 % at W4 lies between the two, so about 145 tok/s, next to the central 147.**
Direct check of the scaling model (`tona-scaling.md` 3.4) with this split: an R9700 with 2x bandwidth and 2x CUs (clock unchanged) would take
W1 23.5 - 1.2 - 7.1 = **15.3 ms** (model central 15.2), W4 36.4 - 4.7 - 9.2 = **22.5 ms** (22.2), W8 54.7 - 9.9 - 10.8 = **34.0 ms** (32.9; +3 %). The verify-pass central curve is reproduced to within 3 %.

### E. Which assumptions of the model are confirmed, which are broken

Confirmed:
- Dense large steps at W1 are bytes plus a small fixed part, unchanged with CUs (dn.in 1.01, head.out 0.98, gr.* 1.00); their growth above W1 scales with CUs: dn.in W1 to W8 +1.33 ms at 32 CUs, +2.67 ms at 16 (2.0x),
  gr.down +1.2 against +2.1 ms. The "fixed part x k + bytes + growth / cu" rule holds. The fixed part of the dense steps at W1 is about 2.6 ms (measured time 12.7 ms against 10.1 ms of bytes for the dense steps listed), the model's 2.63 ms.
- host.wait at W1 and W4 does not move (0.72 / 1.22 ms against 0.66 / 1.18): the round trip is fixed (the model's rule); at W8 it rises 40 % (the GPU arrives later, not a longer round trip).
- Expert kernels at W4 and W8 scale with CUs (1.63 / 1.79 for gate_up), as an ALU/LDS-limited kernel at 55 to 65 % of the ceiling: the model's "expert kernel loss: CUs x clock".
- host.y falls at 16 CUs (the CPU has more time to finish behind slower GPU steps): consistent with host time that hides behind the GPU, not with GPU time.

Broken or weaker than assumed (the plan's ranges, rows with a different result):
- **Experts at W1: +36 % (4.70 -> 6.38 ms, gate_up +45 %, down +23 %)**, above the plan's +10 to +30 %. The W1 expert step is CU-limited by about 1.7 ms of its 4.7 ms; the model's "fixed f + bytes" treats W1 experts as latency plus bytes. At 64 CUs this means the expert steps gain about 0.9 ms at W1 beyond the bandwidth gain, which the model's `f` (0.7 to 1.2 ms fixed) does not give. Effect on the cycle: about 0.9 ms of 30 (3 %).
- **dn.recur does not scale with CUs at W1 (0.93 -> 0.87, ratio 0.93) and only +14 % / +24 % at W4 / W8.** Doubling 16 -> 32 CUs buys 1.0 / 1.14 / 1.24; the model's dn.recur speed-up on the R9700 (low / central / high 1.3 / 1.6 / 1.9 for 32 -> 64 CUs) is too optimistic; the measured values are an upper bound of 1.0 to 1.25 (latency-bound recurrence, as `k_dn_wave` in prefill). Size: 1.2 to 2.1 ms of the pass at 32 CUs, so a shortfall of about 0.3 to 0.6 ms per pass against the central.
- **Small dense steps (gr.down, router) gain little from more CUs**: 1.00 / 1.13 / 1.31 (gr.down) and 1.00 / 1.14 / 1.23 (router) for 16 -> 32 CUs, against the model's central 1.0 / 1.4 / 1.9 (the model's "high" column is not reachable, "low" is close to the truth). gr.up is flat (1.00 / 1.06 at W4 / W8).
  The verify-pass total is still on the model's central because the model's other pieces (experts at W4/W8, dn.in, dn.out, qsa.out, which scale more than assumed: dn.out 1.29 / 1.53, qsa.out 1.23 / 1.47) compensate. A reader who changes one piece of the model should know that these errors cancel.
- Attention (`qsa.attn`) scales almost fully with CUs at W4 / W8 (2.05 / 2.20); the model's 1.0 / 1.2 / 1.5 is pessimistic for it. It is small (0.25 to 0.43 ms), so no effect.
- Not tested here: the clock, the bandwidth of a card with 640 GB/s (the split above assumes bytes time halves), the draft passes (W1 MTP steps), prefill. Fixed time per step cannot be told from bandwidth inefficiency by this test; the 24 % "fixed" includes both.

### Verdict against the plan

Plan ranges: whole pass within for W1, W4, W8; moe experts W8 within (+73 %), W1 above (+36 %, plan +10 to +30 %); dn.recur within; attention and small steps within; host.wait unchanged
at W1 / W4 and +40 % at W8 (plan: +-5 %; the GPU-side wait grew); outcome class fixed share at W4 24 % (plan 20 to 35 %): within.
The result for the R9700 estimate: **decode about 145 tok/s (code), the central of the model (147) stands; the spread of the W4 fixed share (24 %) narrows to the 20 to 30 % band**, and two single rules of the model
(dn.recur, small dense steps) must be moved toward the low column.
