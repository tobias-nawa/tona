# IE_DIAG=1 in every development run: cost (#72)

## Plan

Written during round 1, before rounds 2 and 3 and before the bit-exact run; round 1 had already started when the
rule for a plan came in (its results were read against this plan afterwards, none were discarded).

1. Scope. One build (`work/build/wdiag`), only the environment changes. Three rounds, each: server at `--ctx 140000`
   (MTP), prompts of 30K and 130K tokens, 200 tokens, greedy, thinking off (`IE_DIAG=0`, then `IE_DIAG=1`); then
   `scripts/bench-set.py` at `--ctx 32768` (`IE_DIAG=0`, then `IE_DIAG=1`). Finally one deterministic run
   (`IE_DETERMINISTIC=1`, `IE_DIAG=1`, 30K and 130K). 60 s pause between blocks, about 65 min in total.
2. Expectation. `IE_DIAG=1` is as fast as `IE_DIAG=0` within the spread (decode and prefill): the trace costs one
   line per allocation at start-up, the dump is written only at a fault. Reference for the spread: the wrappers
   without variables cost decode -0.1 % (2026-10-31-moe-sum-fault.md). Bit-exact SHAs: `065922c2ff602991` (30K),
   `439aad472c59ae6f` (130K).
3. Active tooling. `IE_DIAG` 0 or 1 (via `toolchain/env.sh`; with 1: `IE_ALLOC_TRACE=<dir>/alloc-trace-%p.txt`,
   `HSA_COREDUMP_PATTERN=<dir>/gpucore-%p.core`, `ulimit -S -c unlimited`), `OMP_NUM_THREADS=6`,
   `IE_EXPERT_PROFILE`, `IE_DETERMINISTIC=1` only in the last run. Nothing else.

## Results

Build `work/build/wdiag` (commit 8402083 plus formatting), server MTP, greedy, thinking off, 200 tokens. Other GPU work
by a second worker ran in the gaps between my blocks, always under `/var/tmp/claude/gpu.lock`. Every rate is named.
Rounds are listed as 1, 2, 3.

| Rate | Setup | `IE_DIAG=0` | `IE_DIAG=1` |
|---|---|---|---|
| Prefill, tok/s | ctx 140000, prompt 30K | 1803, 1800, 1797 (mean 1800) | 1804, 1788, 1796 (mean 1796) |
| Decode, tok/s | ctx 140000, prompt 30K | 71.8, 74.4, 73.1 (mean 73.1) | 73.5, 74.3, 72.9 (mean 73.6) |
| Prefill, tok/s | ctx 140000, prompt 130K | 1705, 1714, 1707 (mean 1709) | 1709, 1712, 1699 (mean 1707) |
| Decode, tok/s | ctx 140000, prompt 130K | 63.2, 63.9, 62.7 (mean 63.3) | 64.2, 63.3, 63.3 (mean 63.6) |
| Decode, tok/s | bench-set code, ctx 32768 | 85.9, 85.7, 85.5 (mean 85.7) | 87.5, 85.7, 86.5 (mean 86.6) |
| Decode, tok/s | bench-set text, ctx 32768 | 62.6, 62.7, 62.2 (mean 62.5) | 63.5, 62.5, 62.9 (mean 63.0) |

Against the plan: no rate with `IE_DIAG=1` is slower outside the spread (the spread of `IE_DIAG=0` alone is 2.6 tok/s
decode at 30K, 1.2 at 130K, 0.4 at bench-set code, 0.4 at text; prefill 6 and 9 tok/s). No deviation.

Bit-exact: `IE_DETERMINISTIC=1`, `IE_DIAG=1`, 30K: SHA `065922c2ff602991`; 130K: `439aad472c59ae6f`; both as
expected. The non-deterministic rounds above gave the same SHAs in all 12 prompt runs. (The first attempt of this
run failed at start with a GPU out-of-memory error because another process still held the GPU; it was repeated
under the lock with identical settings. Rates in that run: prefill 1798 / 1712 tok/s, decode 74.1 / 61.2 tok/s at
30K / 130K; the 130K decode is 2 tok/s under the other rounds, one run, not repeated, deterministic mode.)

Per-PID traces: every server start in `IE_DIAG=1` left its own `alloc-trace-<pid>.txt`; nothing was overwritten
(8 files after the series). Cleaning (`scripts/diag-clean.sh`, tested without GPU): with traces of 10 days and 0 days
and cores of 3, 2 and 0 days old in the directory, only the 0-day trace and the 0-day core remained. The unit test
`trace_path` covers the `%p` expansion (none, one, two, other percent signs).
