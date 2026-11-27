# Speed checks for the code-rules change and the hardened release programs, 2026-11-26

Rule 2 plan, written before the first run. Results are read against it below the line `## Results`.

## Plan

**Question 1: does the lint refactor of PR #17 change speed?** `main` against `ci/code-rules` (head `f50cd70c`), both as the
development build (no `TONA_HARDEN`). The refactor touches host code of `engine/gpu`, `engine/cpu`, `engine/model`, no device
code intended. Rule 1 (AGENTS.md): the reference never gets worse. **Decision:** every rate within noise (|difference| inside the
0.5 % margin of `ab.py`, or "equal") -> #17 may merge on speed; a rate slower beyond noise -> find the function (perf) and report.

**Question 2: does hardening cost speed?** `ci/code-rules` dev build against the same tree built with `-DTONA_HARDEN=ON`
(`engine/cmake/hardening.cmake`: `_FORTIFY_SOURCE=3`, stack protector strong, stack clash protection, `-fcf-protection=full`, PIE,
full RELRO; host pass only for HIP sources). Users get the hardened program, every earlier measurement used the unhardened build.
**Decision:** within noise or at most 2 % slower -> keep; more than 2 % -> report, and run one arm per suspect flag.

Builds (`scripts/gpu_queue.py --kind cpu`, before the first measurement, target `tona-server`, no `CMAKE_BUILD_TYPE`, as the
development build): `work/build/n3-main` from `d43a0b3f` (`origin/main`), `work/build/n3-pr` from `f50cd70c`,
`work/build/n3-hard` from `f50cd70c` with `-DTONA_HARDEN=ON`. `origin/main` moved three commits past the merge base of #17
(`0f4299af`: a Metal warning, a macOS client fix, a Mac memory setting, docs). Checked with `git diff --stat 0f4299af origin/main
-- engine installer scripts tools toolchain bench`: no device code, no generator or kernel host file; the only server file is
`engine/server/system_status.cpp` (the `/status` text, not on a per-token path), the rest is Metal, client, installer and tests.
The two arms of q1 therefore differ by #17 plus those files. Nothing is built after the first measurement starts.

### Blocks

| Block | Question | Command (all through `scripts/gpu_queue.py --kind exclusive`, test port 8171; `TONA_DIAG=1`) | Rounds | Expected duration |
|---|---|---|---|---|
| q1 | 1 | `ab.py --a n3-main --b n3-pr --workload short,mid,long` (Coder, MTP head, shipped expert profile) | `ab.py` decides, noise from `bench/noise.toml` (df 9 for 30K and 130K, df 4 for 2K), 1 to 6 | 10 to 30 min |
| q3 | 1 | R7: `ab.py --a n3-main --b n3-pr --workload huge` (512K and 1M) | 1 (a start per arm and about 45 min in total per `docs/contribute/testing.md`) | 50 to 60 min |
| q4 | 1 | R7: `bench/needle/run.py --hard --length 131072 --texts 0,1,2 --require-cache` against a server per arm (`TONA_SEQS=1 TONA_PROMPT_RAM_GB=0`) | 3 texts (the minimum of `bench/needle/README.md`) | 20 min |
| q5 | 1 | R8: `scripts/r8_pressure.py --build n3-pr --pressure-files <the Ornith and Bolt GGUF files, which the Coder server does not use> --port 8173` (20 requests, 10 under pressure, `TONA_DIAG=1`) | 1 (the main arm only if the series misses the expectation) | 25 min |
| q2 | 2 | `ab.py --a n3-pr --b n3-hard --workload short,mid,long` (Coder) | as q1 | 10 to 30 min |
| q2o | 2 | `ab.py --a n3-pr --b n3-hard --workload server --targets 30000:200 --ctx 140000 --no-noise` (Ornith, no drafter file, MTP head in the model) | at least 2 (no noise entry for Ornith), at most 6 | 10 to 25 min |

Why this size: q1 and q2 cover decode at 2K, 30K and 130K and prefill at 30K and 130K, the rates the Coder offer rests on; the
host loop of the decode (per-token host work, where a canary or a CET check could show) is in `short`, the prefill and attention
paths in `mid` and `long`. q3 to q5 are the checks `tools/select_checks.py` names for this diff
(`attn_wmma2.h`, `mega_seq.cpp`, `prefill_plan.cpp`, `prompt_cache.cpp` for long context; `omp_guard.h`, `mega_alloc.cpp`,
`mega_pinned.h`, `mega_prefetch.cpp`, `mega_upload.cpp`, `uring.cpp` for pressure); smallest size that answers the question:
128K is the length where a lint change in the prefill or KV paths would show first for retrieval, `huge` the length past 262K
where 32-bit overflow or plan limits would show (the scale run of the same day found no such problem on the main build).
Ornith decode mid is the second model of question 2 only. Goldens are not repeated: gpu-golden C8 on this head is identical
(ticket), and this file measures speed.

### Expectation (every number with tolerance and source)

| Number | Expectation | Source |
|---|---|---|
| q1: **decode** 2K / 30K / 130K, B (#17) against A (main) | -0.5 to +0.5 % (equal) | `bench/noise.toml` A/A: sd of a paired difference 0.23 % / 0.28 % / 0.42 %; the diff is host lint, no device code (the `ab.py` margin is 0.5 %) |
| q1: **prefill** 30K / 130K | -0.5 to +0.5 % (equal) | same: sd 0.04 % / 0.02 % |
| q1: absolute levels (read against the scale run of the same day) | **decode** after 30K / 130K about 75.5 / 65.3 tok/s, **prefill** 30K / 130K about 2,021 / 1,936 tok/s, **decode** `decode-code` of `short` not measured by it (o-coder 2K: 71.0 tok/s, 16K 67.9) | `2026-11-26-scale-run.md` c3 and o-coder, +-3 % |
| q3: **prefill** 512K / 1M both arms | 1,536 / 1,164 tok/s, 1,430 to 1,540 / 1,000 to 1,180 (slot count decides) | `2026-11-26-scale-run.md` c4 |
| q3: **decode** 512K / 1M | 60.8 / 57.2 tok/s, 58.5 to 63.5 / 54.5 to 59; B against A -1 to +1 % | same |
| q3: answer SHAs | 512K `32e65dd9bc402413`, 1M `dc6c12a50914afe6`, equal in both arms (a difference is recorded) | same |
| q4: needle hard 128K, hits of 120 | equal in both arms within the McNemar noise (about half of the needles on the hard test, 50 % at 128K) | `bench/needle/README.md` (`off`: 128K 50 %) |
| q5: R8 | 0 aborts, 0 `Pass failed`, 0 NaN; at most 2 of 20 requests below 55 tok/s **decode**, none below 50; **decode** 30K 60 to 80, 130K 56 to 70; evicted time per pressured request mean below 2.5 s, max below 5 s | `2026-11-26-scale-run.md` e (one pressured 130K request ran at 36.2 tok/s there on main: a repeat of it here is not caused by #17 until the main arm shows otherwise) |
| q2: **decode** 2K / 30K / 130K and **prefill** 30K / 130K, hardened against dev | -0.5 to +0.5 %; keep up to -2 % | sd as q1; hardening adds a few instructions per function with an array and a CET check per indirect call in host code, while the loop waits on the GPU (**ungeprueft** that no hot host loop exists: perf would tell if it shows) |
| q2o: Ornith **decode** after 30K, **prefill** 30K | -0.5 to +0.5 % | no noise table; the Coder expectation. Absolute: Ornith decode 2K / 16K / 64K 117.1 / 97.8 / 65.7 tok/s in the scale run |
| any block | no GPU reset, hang, oom-kill, `Pass failed`, NaN | AGENTS.md |

A number outside its range is a finding, not noise.

### Active tooling

- `TONA_DIAG=1` in every server (no rate effect). `TONA_SEQS=1` in `huge` (set by the workload) and q4 (with
  `TONA_PROMPT_RAM_GB=0`). `ab.py --load-log on` in every `ab.py` block (sums of CPU, GPU and memory beside the server starts;
  no program names). No profiler, no `TONA_MEGA_PROF`, no `TONA_DETERMINISTIC` (production mode, drafts not capped). No `TONA_*`
  variable in the shell; `ab.py` passes none on.
- Servers in `systemd-run --user --scope` with `MemoryMax` (RAM minus 6 GiB; 54G for q4) and `MemorySwapMax=0`; test ports 8171
  (`ab.py`), 8172 (q4), 8173 (q5); never 8662 or 8097.
- The test conditions (`tools/hw_report.py`, `tools/quiet_check.py`) are in the block logs `<block>.tc.log`, before and after each
  block, copied into `2026-11-26-rules-speed-logs/`.

### Prior art

Not applicable: no engine code changes; this file measures the engine of PR #17 and a build option of it. The method is
`scripts/ab.py` with `bench/noise.toml`; expectations cite the measurement files above.

### Test conditions before the first block (q1, 02:44)

```
(block q1-lint, before) 2026-11-26 02:44:40
### Test conditions

- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.84 GiB held by others before the run, 1 % busy before the run
- RAM: 60.5 GiB, 52.3 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.3.4-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 4.1, 8.2, 6.01
- CPU use by other programs over 0.5 s: 0.4 % of all threads, swap in use 7.99 GiB
- sandbox: no
- Commit: `d43a0b3f` (dirty tree)
- Quiet check (2026-11-26 02:44:41): warning
  - warning: 1.84 GiB of 15.9 GiB VRAM are held by others
quiet_check exit 1
```

### Test conditions after the last block (q3, 06:05)

```
## after q3-huge 2026-11-26 06:05:32
### Test conditions

- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 8.19 GiB held by others before the run, 15 % busy before the run
- RAM: 60.5 GiB, 52.3 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.3.4-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.1, 1.48, 1.68
- CPU use by other programs over 0.5 s: 0.2 % of all threads, swap in use 7.97 GiB
- sandbox: no
- Commit: `d43a0b3f` (dirty tree)
- Quiet check (2026-11-26 06:05:34): warning
  - warning: 1.82 GiB of 15.9 GiB VRAM are held by others
quiet_check exit 1
```


The `hw_report.py` line "Commit" is the checkout the blocks were started from (`d43a0b3f`, the measurement branch), not the
builds: the arms were `n3-main` (`d43a0b3f`), `n3-pr` and `n3-hard` (`f50cd70c`). The "after" report of the last block was written
while its server was still shutting down (8.19 GiB VRAM held; the quiet check two seconds later says 1.82 GiB, the desktop).
Every block has its own `<block>.tc.log` (before and after) in `2026-11-26-rules-speed-logs/`; all of them end with the same single
warning, the desktop's 1.82 to 1.84 GiB of VRAM; others' CPU 0.0 to 0.4 % of all threads, load average at the start of a block
1.3 to 6.9 (the high ones are the end of the builds or of the previous block), swap 8.0 GiB in use (pages of other processes; the
servers have `MemorySwapMax=0`). All blocks ran through `scripts/gpu_queue.py --kind exclusive`.

Hardening proof: `tools/check_hardening.py` says `n3-hard/tona-server` is hardened (PIE, full RELRO, NX, canary, FORTIFY, CET
instrumentation; warning: the GNU property note does not mark IBT/SHSTK because the toolchain's startup objects lack it) and
`n3-pr/tona-server` is not (no BIND_NOW, partial RELRO, no canary, no FORTIFY, no endbr64): the two arms differ by the option.

## Results

### Run record

Builds `n3-main`, `n3-pr`, `n3-hard` (cpu queue, ccache, finished 02:44) before the first measurement; nothing was built after. Slots
(expert slots per layer, desktop at 1.8 GiB): Coder 72 at `--ctx 140000`, 61 at `--ctx 1048576`, Ornith 96. Four queue jobs: q1, q2,
q2o (02:44 to 03:32); q4 and q5 (03:32 to 04:00); q1b and aa (04:00 to 04:37); q3 (04:38 to 06:05).
**Deviations from the plan:** (1) `ab.py` refuses `--min-rounds 1` ("need 2 <= --min-rounds"), so the first start of q3 (inside the
second job) ended at once with no number (`q3-huge-failed-start.tc.log`); q3 was repeated with 2 rounds and without the warm-up start:
88 min instead of the planned 50 to 60. (2) q1b and aa were added after q1 showed "B slower" by 0.12 % at 130K prefill, inside the margin:
a repeat, and an A/A of `n3-main` against itself as control. (3) I edited `blocks.sh` (q1b, aa, q3 changes) while the first job still
ran it; that job had already read its loop, and its three blocks finished with their results (unchecked: whether its shell printed an
error afterwards). No GPU reset, hang, oom-kill, `Pass failed` or NaN in any server log of the `n3-*-logs`, the needle and the R8 logs
(grep: 0; the kernel log was not read).

### Question 1: does the lint refactor of #17 change speed? (main `d43a0b3f` against #17 `f50cd70c`, development builds)

q1: `short,mid,long`, 6 rounds (21.5 min; `ab.py` stops at 6 when metrics stay open), B (#17) against A (main):

| Metric | B against A, mean [95 % interval] | A / B median (tok/s) | Expected | Decision of `ab.py` |
|---|---|---|---|---|
| **prefill** 2K (first request of a start) | +0.26 % [-8.52, +9.03] | 1,395 / 1,396 | -0.5 to +0.5 % | open; per round -6.8 to +15.3 %, finding 1 |
| **decode** 2K | +1.20 % [-1.06, +3.47] | 75.7 / 76.0 | same | open; round 1 +5.1 %, rounds 2 to 6 -0.45 to +1.63 % |
| **prefill** 2K, write prompt | +0.98 % [-1.81, +3.77] | 1,683 / 1,717 | same | open |
| **decode** 2K, write prompt | +0.52 % [-1.18, +2.22] | 128.6 / 128.6 | same | open |
| **prefill** 30K | -0.22 % [-0.52, +0.08] | 1,976 / 1,971 | same | open; inside |
| **decode** 30K | -0.25 % [-0.70, +0.20] | 69.5 / 69.3 | same | **equal** after round 3 |
| **prefill** 130K | -0.12 % [-0.22, -0.02] | 1,887 / 1,885 | same | "B slower, real but inside the margin"; **not reproduced** (q1b, aa below) |
| **decode** 130K | +0.02 % [-0.64, +0.68] | 67.0 / 66.9 | same | open; inside |

Identical in both arms in all 6 rounds: expert cache hits and misses per request (30K 67,822 / 10,547; 130K 73,061 / 9,712), tokens
per pass (3.56, 7.82, 3.12, 3.03) and the answer SHAs of all four requests (`75eff9b8784c7d4c`, `a6f07fd6fb86e448`,
`0a2eac2f1502846f`, `7e79f79e83fade65`; production mode, not deterministic: stable over the rounds and equal between the arms). The
plan verdicts "OUTSIDE the expectation" for the three 2K rows come from intervals wider than the expectation (round 1 of each is
the first request after a start), not from a stable difference; 30K and 130K, where the refactor's work would show, are within +-0.25 %.

q1b, the repeat (`mid,long`, 6 rounds, 22.3 min), B against A: **prefill** 30K +0.25 % [-0.43, +0.92], **decode** 30K +0.08 %
[-0.46, +0.63], **prefill** 130K **+0.06 % [-0.05, +0.17], equal after round 1**, **decode** 130K -0.41 % [-1.26, +0.44] (round 3
-2.24 %, flagged outlier; without it about -0.1 %). The 130K prefill loss of q1 did not come back.

aa, the control (`n3-main` against itself, `mid,long`, 4 fixed rounds, 15.5 min), "B" against "A": **prefill** 30K -0.46 %
[-0.90, -0.02] "slower", **decode** 30K +0.24 % [-0.19, +0.66], **prefill** 130K **+0.19 % [+0.07, +0.32] "faster"**, **decode**
130K +0.44 % [-0.57, +1.44]. Two starts of the same binary differ by -0.46 and +0.19 % in prefill and `ab.py` calls both "real": its
intervals are too narrow for prefill (finding 2). A difference of -0.12 % is smaller than what the control makes of one binary.

q3 (R7, `huge`, 512K and 1M, 2 rounds without warm-up, 87.5 min; `TONA_SEQS=1`, 61 slots):

| Metric | A (main) / B (#17), mean of 2 | B against A [interval] | Expected | Verdict |
|---|---|---|---|---|
| **prefill** 512K | 1,462 / 1,463 tok/s | +0.08 % [-0.95, +1.10] | levels 1,430 to 1,540; -1 to +1 % | within |
| **decode** 512K | 60.1 / 60.3 tok/s | +0.31 % [-0.52, +1.13] | 58.5 to 63.5; -1 to +1 % | within |
| **prefill** 1M | 1,076.5 / 1,077.9 tok/s | +0.13 % [-0.69, +0.94] | 1,000 to 1,180 (61 slots against 69 in the scale run, about 0.8 % per slot: 1,164 becomes about 1,090) | within |
| **decode** 1M | 56.2 / 56.1 tok/s | -0.26 % [-1.10, +0.58] | 54.5 to 59; -1 to +1 % | within |

Answer SHAs `32e65dd9bc402413` (512K) and `dc6c12a50914afe6` (1M), equal in both arms, stable, equal to the scale run. `ab.py` calls all
four "open" (two rounds, intervals of about +-1 %): no difference beyond 1.1 %, none that the expectation did not allow.

q4 (R7 retrieval, `bench/needle/run.py --hard --length 131072 --texts 0,1,2 --require-cache`, `TONA_SEQS=1`, `TONA_PROMPT_RAM_GB=0`,
one server per arm, 233 s per arm): **66 of 120 needles in both arms** (55.0 % [46.1, 63.6]; single 39 of 60, multi 27 of 60). Pairs:
66 both right, 0 only main, 0 only #17, 54 both wrong; McNemar p = 1.000; every depth bin identical (16/24, 18/24, 17/24, 9/24, 6/24).
Expectation "about half" (`off`: 50 % at 128K): within. The arms agree needle by needle.

q5 (R8, `scripts/r8_pressure.py --build n3-pr`, 10 pairs of 30K and 130K, 10 pressured requests, 1 series, page cache pressed with
the Ornith GGUF, 72 slots): the script's four checks all "ok" (no failed request, at most 3 of 10 pressured slow, evicted time 1.3 s in
total below 5 s, longest episode 0.8 s below 3 s). **Decode** 30K unpressured mean 70.7 (69.6 to 73.1), pressured 65.7 (62.7 to 69.5);
130K unpressured 67.4 (61.3 to 70.3), pressured 61.5 (58.1 to 62.9) tok/s; the lowest request is 58.1, none below 55 (expected: at
most 2 below 55, none below 50; 30K 60 to 80, 130K 56 to 70). Evicted time per pressured request mean 0.13 s, max 0.88 s (expected:
mean below 2.5 s, max below 5 s). No abort, no `Pass failed`, no NaN. The 36.2 tok/s request of the scale run on main did not repeat; no main arm was run
(the series met the expectation; the plan says it would only run on a miss).

**Decision question 1:** the speed of #17 against main is **equal within noise** wherever it was measured. 30K and 130K prefill and
decode lie within +-0.25 % in q1 (the one "real" -0.12 % did not reproduce in q1b, +0.06 %, and an A/A of one binary shows -0.46 and
+0.19 %); 512K and 1M within +-0.3 % (interval about +-1 %); answer SHAs, expert cache counts, tokens per pass and needle results are
identical; R8 meets the expectation. **#17 may merge on speed.** The 2K requests (first request after a start) stay undecided in
this form (finding 1): their per-round spread is 6 to 15 %, the same for one binary in the scale run (A/A, 2K prefill -5.8 %).

### Question 2: does hardening cost speed? (#17 dev build `n3-pr` against `n3-hard`, same tree, `-DTONA_HARDEN=ON`)

q2 (Coder, `short,mid,long`, 6 rounds, 21.5 min), B (hardened) against A:

| Metric | B against A, mean [95 % interval] | A / B median (tok/s) | Decision of `ab.py` |
|---|---|---|---|
| **prefill** 2K | -3.35 % [-18.96, +12.27] | 1,402 / 1,262 | open (rounds -16.8 to +16.3 %) |
| **decode** 2K | +0.19 % [-0.25, +0.64] | 75.6 / 75.9 | **equal** after round 5 |
| **prefill** 2K, write | -3.06 % [-7.15, +1.03] | 1,694 / 1,661 | open |
| **decode** 2K, write | -0.21 % [-0.93, +0.52] | 128.5 / 128.4 | open; inside |
| **prefill** 30K | +0.28 % [-0.43, +0.98] | 1,974 / 1,974 | open; inside |
| **decode** 30K | +0.23 % [-0.36, +0.82] | 69.1 / 69.2 | **equal** after round 4 |
| **prefill** 130K | -0.06 % [-0.27, +0.15] | 1,885 / 1,884 | open; inside |
| **decode** 130K | +0.06 % [-0.52, +0.65] | 66.6 / 66.8 | open; inside |

Expert cache counts, tokens per pass and answer SHAs equal in both arms (the SHAs of q1).

q2o (Ornith, workload `server`, 30K prompt, 200 tokens, no drafter file, `--no-noise`, 6 rounds, 3.7 min), B against A: **prefill** 30K
**+0.02 % [-0.51, +0.55], equal** (3,907 / 3,907 tok/s), **decode** 30K -0.08 % [-0.66, +0.49] (88.7 / 88.6 tok/s; open, inside).

**Decision question 2:** hardening costs **no measurable speed**. Decode and prefill at 30K and 130K, decode at 2K and the Ornith
rates are within +-0.3 % (Ornith +-0.1 %), and every interval of these rows excludes a loss of 1 % (lowest lower bound -0.93 %, 2K write
decode), well under the 2 % that the ticket allows. **Keep the hardening on**; no per-flag arms were needed. The two 2K **prefill**
rows (point estimates -3.4 % and -3.1 %) are undecided: rounds swing by +-15 % (finding 1), and the 30K and 130K prefill, which run the
same host code per chunk, are equal within 0.3 %. Ungeprueft whether a dedicated 2K prefill series with several requests per start
would show a small loss; I did not run one.

### Findings

1. **The first 2K request of a start cannot decide a percent.** The paired **prefill** difference at 2K swings between -17 and +16 %
   per round (q1, q2); the same-binary A/A of the scale run gave -5.8 %; `bench/noise.toml` has sd 0.09 % (df 4) for it. Cause not
   examined (ungeprueft; a cold expert cache or GPU clocks at the first request of a start are the obvious suspects). `short` prefill at
   2K should not be a primary metric (`--primary decode`).
2. **`bench/noise.toml` understates the prefill spread at 30K and 130K.** The table has sd 0.04 % and 0.018 % (df 9); the control
   `aa` shows paired differences of -1.26 to -0.06 % (30K) and -0.03 to +0.30 % (130K) for one binary: sd about 0.15 % at 130K, eight
   times the table. `ab.py` therefore called -0.12 % (q1) and +0.19 % (aa) "real". The tables of these runs (`ab/n3-aa-main-noise.toml`
   and the others) were produced but not taken into `bench/noise.toml`: `ab.py` never writes it unasked, and I did not add A/A
   rounds to a shared table without the maintainer.
3. `ab.py` refuses `--min-rounds 1` ("need 2 <= --min-rounds"); a `huge` comparison therefore needs at least 88 min (2 rounds of 44),
   not the 45 min that `docs/contribute/testing.md` names for one round.

### Not run, and why

- Goldens and the deterministic bit-exactness check (R3, R4): identical on #17's head per the ticket (gpu-golden C8). The production-mode
  answer SHAs above are equal and stable, a weaker statement than R4.
- A main arm of R8: the #17 arm met the expectation.
- A perf attribution of the -0.12 %: it did not reproduce.
