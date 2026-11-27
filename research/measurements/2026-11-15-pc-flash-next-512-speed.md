# Flash-Next 512 experts against the Coder: decode and prefill speed on the reference PC (RAM class 58 GB)

Date: 2026-11-15. Build: `work/build/f512` (branch 146-full-512, C1-identical to the base build for the Coder: only the eight kernels of the wide
size class are new), one build for both arms. Raw data and logs: `/var/tmp/claude/146/speed/` (server logs, `bench/run` json and md, hardware
report, quiet checks before and after each job).

## Plan (written before the first run)

**Question.** How fast does the 512-expert Flash-Next (ISTA GSQ-RCO Q2_0, dense Q3_K/Q4_0/Q5_0 repacked losslessly to Q6_K/Q8_0) decode and prefill
on this PC against the Coder (IQ1_M, 256 experts), in the same engine, at the 64 GB class (`TONA_RAM_GB=58`, scope limit 58e9 bytes, no swap)?

**Decision per outcome** (maintainer, 2026-11-15):
- decode at least that of the Coder: the agent-quality test (phase 2b) is next, the model is a candidate for 64 GB;
- 60 to 100 % of the Coder: quality decides;
- below 60 %: the model is not offered and the finding is documented.

**Scope.** Two arms, `f512-tona` (512 experts) and `coder-tona`, alternating A B A B A B (3 rounds), one queue job per arm and round
(`/var/tmp/claude/146/speed_job.sh`, pattern of `bench/polyglot/model_job.sh`: hardware report, quiet check before and after, server in a
systemd scope with `MemoryMax=58e9 MemorySwapMax=0`, port 8099). Cases of `bench/suites/base.toml`: `short-code` (short prompt, about 300 tokens
out, temperature 0), `long-decode` (short prompt, up to 1024 tokens out) and `needle-30k` (about 30K tokens of code in, about 400 words out: the prefill rate at
30K, then decode after it; added by the maintainer's request). Per job one warm-up and one measured run per case (`-n 1`), so 3 values per
arm and case. `--ctx 65536`. MTP drafter of the base model (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, the production setup) on both arms: the server requires exactly one of `--mtp` and
`--dflash`, so a run without a drafter does not exist; the head has 512 experts, and the 512 file answered coherently with it (bring-up, 2a-notes).
Rates: **decode** = tokens generated per s after the first token (`decode_tps_client`); **prefill** = prompt tokens per s to the first token
(`prefill_tps_client` of `needle-30k`; the prompts of the two short cases are about 40 tokens, so their prefill is a latency, not a rate).
Why this size: the decision needs the decode rate with the two caches (expert hit rate) of both models at the same RAM class; two short cases show the
decode rate at small context and 3 rounds give a spread, which the noise file `bench/noise.toml` does not list for this setup. A long-prompt prefill
comparison beyond 30K and the agent quality belong to phase 2b.

**Expectation.** Phase 1 estimate (`/var/tmp/claude/146/notes.md`, unchecked): decode 99 to 108 tok/s for the 512 file against 89 tok/s for the Coder
(measured earlier on this PC with MTP). Tolerance: 512-file decode between 70 and 115 tok/s (the cache share is lower: 144 experts of 512 per layer in
VRAM = 28 % against 56 % of 256, hit rate below the Coder's 83 to 90 %); Coder decode 85 to 92 tok/s (BENCHMARKS.md). Prefill at 30K (`needle-30k`): about 1,300 tok/s for the 512 file against about 1,741 tok/s for the Coder (both phase 1 estimates, unchecked); tolerance 900 to 1,500 and 1,600 to 1,900. Source: notes.md of phase 1 and
BENCHMARKS.md. A result outside the tolerance is a finding, not noise. Expert hit rate and MTP acceptance are read from the server log
(`Experts during generation`) and from `bench/run` (`draft_accept_rate`, `tokens_per_pass`).

**Active tooling.** No `TONA_DETERMINISTIC` (it caps the drafts at 3), no profilers, no `TONA_DIAG`, no test switches; `TONA_RAM_GB=58`,
`OMP_NUM_THREADS=6`, `ROCR_VISIBLE_DEVICES=0`; no expert profile file for either arm (the Coder normally starts with `experts-coder.bin`, which seeds
the cache: the 512 file has none, so the comparison is cold-start for both).

**Prior art.** n/a: a measurement of an existing build; the engine change under test is the size-class split (bit-identical for the Coder, isa-diff).

## Results (15:29 to 15:59, three rounds ABABAB, one measured run per case and round after one warm-up)

Per job one server start (cold: no prompt cache, no expert profile), `--ctx 65536`, MTP on. Values are medians of the three rounds (range in
brackets). **Decode** = `decode_tps_server` (generated tokens per s); **prefill** of `needle-30k` = prompt tokens per s from the server's
"computed in" time (30,909 tokens). Tokens per pass = tokens per verify pass (MTP acceptance and width). Per-round decode, 512 file: short-code
98.9, 104.3, 99.6; long-decode 80.2, 82.2, 80.5; needle-30k 88.0, 87.3, 71.7. Coder: short-code 112.1, 112.8, 101.3; long-decode 79.1, 80.3, 73.8;
needle-30k 64.5, 65.2, 64.1.

| case | metric | 512 experts | Coder | 512 / Coder |
|---|---|---|---|---|
| short-code | decode tok/s | 99.6 (98.9 to 104.3, n=3) | 112.1 (101.3 to 112.8, n=3) | 0.89 |
| short-code | tokens per pass | 6.71 (6.43 to 6.71, n=3) | 6.40 (6.20 to 6.40, n=3) | 1.05 |
| long-decode | decode tok/s | 80.5 (80.2 to 82.2, n=3) | 79.1 (73.8 to 80.3, n=3) | 1.02 |
| long-decode | tokens per pass | 3.15 (3.14 to 3.26, n=3) | 3.45 (3.07 to 3.58, n=3) | 0.91 |
| needle-30k | decode tok/s | 87.3 (71.7 to 88.0, n=3) | 64.5 (64.1 to 65.2, n=3) | 1.35 |
| needle-30k | tokens per pass | 4.04 (2.89 to 4.22, n=3) | 2.74 (2.73 to 2.85, n=3) | 1.47 |
| needle-30k | prefill tok/s | 1969 (1968 to 1969, n=3) | 1932 (1932 to 1932, n=3) | 1.02 |


Expert cache (server log and `/status` of the last job of each arm): 512 file 131 expert slots per layer (25.6 % of 512), 35.1 GB of experts pinned
in RAM, 14.32 GB VRAM; Coder 86 slots per layer (33.6 % of 256), 26.3 GB pinned, 14.35 GB VRAM (58 GB class). Expert hit rate over the whole job
(`totals.expert_hits / (hits + misses)`, warm-ups included): 512 file 88.8 % (1,302,100 hits, 164,922 misses), Coder 89.2 % (1,308,229 hits,
158,988 misses). MTP acceptance is in the tokens-per-pass rows.

### Reading against the plan

- 512-file decode 99.6 / 80.5 / 87.3 tok/s: inside the planned 70 to 115 tok/s on all three cases. The Phase 1 estimate of 99 to 108 tok/s holds for
  the short answer only.
- Coder decode 112.1 / 79.1 / 64.5 tok/s: outside the planned 85 to 92 tok/s on two cases. The plan took the 89 tok/s of BENCHMARKS.md (another
  workload, the bench-set) as the Coder's rate; per case the rate differs by 1.7x, so the single planned band was too coarse. This is a finding about the
  plan, not about the engine.
- Prefill at 30K: 512 file 1,969 tok/s against the planned 900 to 1,500 (above: the estimate of 1,300 was too low); Coder 1,932 tok/s against the
  planned 1,600 to 1,900 (marginally above). The two models prefill alike (ratio 1.02); why is unchecked.
- Expert hit rate: 88.8 % against 89.2 % although the 512 file keeps a smaller share of its experts in VRAM (25.6 % against 33.6 %); why is
  unchecked (the pinned part is larger: 35.1 GB against 26.3 GB). The estimate "hit rate below the Coder's" holds only by 0.4 points.
- `needle-30k` decode (1.35): not a like-for-like speed comparison. The texts differ (768 tokens against 540) and the MTP width differs (4.04 against
  2.74 tokens per pass); the server log shows 21 of 209 passes from the history for the 512 file and 2 of 213 for the Coder; the cause is unchecked. Read
  it as "not slower" rather than "35 % faster".
- Spread: round 3 of the 512 file's `needle-30k` (71.7) and the Coder's `short-code` and `long-decode` (101.3, 73.8) are low by 10 to 18 %; their
  jobs started with about 2.2 GiB of swap in use like the others, and the quiet check reported 1.17 GiB of VRAM held by the desktop in every job;
  no cause was found for the dips, they are left in the medians.

### Decision

Not below 60 % on any case. Decode is at or above the Coder on long answers (1.02) and at 30K context (1.35, see above) and 11 % below on the
short answer (0.89); prefill at 30K is equal (1.02). By the rule of the plan the model is **a candidate for the 64 GB class: the agent-quality test (phase 2b) is
next**; the speed would not rule it out. Caveat for 2b: three rounds with one run each, so the median of three has a range of 5 to 20 %; a
statement below that needs `scripts/ab.py` with more rounds.

### Why every job logged "exit 1"

All six jobs ran to the end (results, `/status` dump and quiet check after the run exist for each). The exit status 1 comes from `speed_job.sh` (copied from
`bench/polyglot/model_job.sh` of #145): the script stops its server explicitly, then its `trap ... EXIT` runs `kill $PID` again on the finished process;
`kill` fails with status 1 and `set -e` is active inside the trap, so the script exits 1. Checked with a five-line shell test (exit 1). The #145 copy has the
same trap; its harmless status is only seen when a caller reads it. The data are not affected.

### Test conditions

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.17 GiB held by others before the run, 1 % busy before the run
- RAM: 60.5 GiB, 54.8 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.52, 2.56, 2.43
- CPU use by other programs over 0.5 s: 0.3 % of all threads, swap in use 2.31 GiB
- sandbox: no
- Commit: `05cb846` (dirty tree)

Quiet check before and after each job: warning, unchanged (1.17 GiB of VRAM held by others); swap in use 2.2 to 2.4 GiB at every start; free RAM 54.8 GiB
before the first run. `bench/run` also warned "Repository has uncommitted changes" (the untracked plan file). Commit of the build: 05cb846.
