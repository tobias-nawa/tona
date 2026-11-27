# Shared expert policy (ranking, OEA, union, CPU split, slot order, profile file), 2026-11-12

Status: **run on 2026-11-12, results below.** Verdict: identical tokens, SHAs and counters, no measurable change in speed.

Step 3 of the shared host code ([research/47](../47-backend-shared-host-code.md)): `engine/backend/` now holds
`class_lru.h`, `route_rank.h`, `cpu_cost.h` (moved from `gpu/`), `expert_order.h` (LRU/ClassLRU order and the slot
assignment of a pass, `LayerCache` derives from it), `expert_profile.h` (file format, order by count),
`expert_route.h` (softmax, top-k, OEA, union with weights) and `expert_split.h` (classify misses, planner, fill
choice). `ExpertSelect` in `gpu/mega_select*.cpp` keeps the application: router logits from the mailbox, cache
lookups, mailbox writes, statistics, traces. This code runs per layer and pass while the kernel waits at the router,
so it is the hot path of decode. Rule 1 and 3 (AGENTS.md): identical tokens, identical goldens, speed not worse.

## Test conditions

`tools/hw_report.py` before each run: Ryzen 5 9600X (6 cores / 12 threads), RX 9060 XT gfx1200 15.9 GiB VRAM with 1.36 GiB
held by the desktop, 1 % GPU busy, RAM 60.5 GiB with 55.5 GiB available, swap in use about 1.8 GiB, rocm 10.1.0, no
`TONA_*` set by hand. A = main `915fbaf` (`tona127b-base`), B = branch `127-shared-host-3` at `f0b5672` (the refactor
commit `8106bee` plus the plan file). Quiet check before and after each run: warning only for the 1.36 GiB of the
desktop; measured anyway. Another helper used the machine through the queue, never at the same time.

## Plan (written before the runs)

### Scope

1. Unit (no GPU): `test_expert_route` (also with `variants`: other OEA settings), `test_expert_host`, the existing
   `test_class_lru`, `test_route_rank`, `test_cpu_cost`, `test_mega_extra`. The new tests replay thousands of random
   inputs (ties, NaN, infinity, 1 to 8 tokens, OEA on and off, pinned and unpinned, ClassLRU and LRU, slot pressure)
   through literal copies of the old code and the new code. Expectation: all equal.
2. Goldens: `golden.sh compare` (gpu, `GOLDEN_PLAN=cpu`, `GOLDEN_MODE=prefill`) and `golden-server.sh compare` on port
   8099. Expectation: identical. The gpu plan uses the slot-assignment path (`assign_slots`), the cpu plan and the server
   use the CPU split.
3. Speed, `scripts/ab.py`, A = main 915fbaf (`tona127b-base`), B = this branch, production mode (no
   `TONA_DETERMINISTIC`), workloads `short,set`, `TONA_DECODE_DIAG=1` on both arms (counters only) so that the select
   time per request (`select ms`) and the hits and misses are compared as well. Why this size: `short` is the
   cheapest workload with decode at 400 tokens; `set` (14 requests) is the production decode with MTP drafts and
   several tokens per pass, where OEA-free union building and the CPU split are exercised at widths 1 to 8.
4. Bit-exact, `ab.py` with `TONA_DETERMINISTIC=1`, `mid,long`, 2 fixed rounds: SHAs equal between the arms and equal to
   `bench/sha.toml`. Why: 2 rounds suffice for a SHA; 6 rounds of speed are not needed here (step 3 speed is measured
   in 3).

### Expectation

- Tokens, SHAs, hits, misses, `fills`, `cpu_experts`, `gpu_direct` per request: identical (the logic is a move).
- Decode and prefill rates B against A within -1 % .. +1 %. Source: `bench/noise.toml` (decode@2000 sd 0.23 % and 0.6 %
  for the write request, `bench-set` decode-code sd 0.065 % with df 2); the work per layer is the same (same
  allocations: three vectors in `ExpertRoute` as before in `ExpertSelect`, one `CpuPlan` per layer as before).
- `select ms` per request equal within the noise (about 62 ms for the 2K summary in the last measurement, 53 ms for the
  write request; `research/measurements/2026-11-12-shared-host-mtp.md`).

### Active tooling

`TONA_DECODE_DIAG=1` (counters printed per request); `TONA_DETERMINISTIC=1` only in run 4. No profiler. `ab.py`
default warm-up.

## Results

1. **Unit** (before and after the last commit): `test_expert_route` (default and `variants`: 4,000 random cases each,
   old code against new: ranking, bad-logit count, OEA keep/extra/fill, union order, weights, masks, `oea_swapped` and
   `oea_short`), `test_expert_host` (5,000 split cases with drop, pinned/unpinned, all `TONA_PLAN` modes and fill limits;
   6 slot-assignment runs of 300 passes with LRU and ClassLRU; profile file bytes, mismatch, missing, truncated file,
   order by count), `test_class_lru`, `test_route_rank`, `test_cpu_cost`, `test_mega_extra`: all ok. `ctest -L unit`
   on the refactor build: all pass.
2. **Goldens on the refactor build** (all identical): `golden.sh compare` gpu (csv, rust, explain), `GOLDEN_PLAN=cpu`
   (3), `GOLDEN_MODE=prefill` (long), `golden-server.sh compare` 36 cases on port 8099.
3. **`short,set`, production mode, `TONA_DECODE_DIAG=1`, 6 rounds** (`work/ab/shared-host-expert-prod.*`; 3.4 min per
   round). B against A, mean [CI]: decode@2000 +0.06 % [-0.34, +0.46] (equal), decode@2000:write +0.06 % [-0.75, +0.88],
   bench-set decode-code -0.17 % [-1.56, +1.22], decode-text -0.14 % [-1.49, +1.21], prefill@2000 -0.05 % (equal),
   prefill@2000:write -0.67 % [-2.20, +0.86]. All within the expected -1 to +1 %; the open ones have wide intervals
   because of one round each (round 1 and 2 with the cold start of the first set pair: decode-code +0.76 % and -2.53 %,
   alternating sign with the arm order). Hits and misses per request identical (133,160 / 19,689 and 86,604 / 23,673),
   SHAs equal. `select ms` per request (mean over all DIAG lines of the six rounds, counter of the per-layer selection
   in the server log): A 37.7 and 49.6, B 37.6 and 49.8 ms for the two first requests of each start, i.e. equal.
4. **`mid,long`, `TONA_DETERMINISTIC=1`, 2 rounds** (`work/ab/shared-host-expert-det.*`): SHAs equal in both arms and
   equal to `bench/sha.toml` (`ee58880d3f5b558f` for 30K, `7d2996546076d1b0` for 130K); expert hits and misses identical
   (71,989 / 7,316 and 76,962 / 6,639). Rates: decode@30000 +0.00 %, decode@130000 +0.19 %, prefill@30000 -0.06 %,
   prefill@130000 +0.01 %.

## The last commit: `hlast_pos` only when h was saved

Question: can the reference reach "h unknown at the end of `mtp_fill`" at all? By code reading: yes.
`mtp_h_of(q)` is unknown when q >= 0, q is not in the window of the last target pass (`trunk_ok_` and
`fw_pos_ .. fw_pos_ + fw_m_`) and q is not `hlast_pos`. `trunk_ok_` of a slot is cleared whenever another slot runs a pass
(`Mega::begin_pass`, `forward_multi` for slots without rows) or a prefill (`prefill_entry.cpp`), and `load_state` clears it.
With `TONA_SEQS=2` (the default) a fill of slot A after a pass or prefill of slot B at A's `pos - 1` therefore finds h
unknown. Before the commit the reference then set `hlast_pos = pos - 1` while `mtp_hlast_` held the h of an older
position, so the next fill took that row as h of `pos - 1` (a wrong h, not counted in `mtp_stale`). Only drafts are
affected, the target verifies every token. Not reproduced at run time (**ungeprueft** how often it happens in a real
two-client session); the unit test of `MtpTrack::after_fill` covers the new rule (`h_saved = false` leaves `hlast_pos`).
The commit makes the reference follow Metal (`after_fill(pos, next, hl != mtp_bad_)`).

On the build of this commit (`956032f`): `ctest -L unit` all pass; `golden.sh compare` gpu (3), cpu plan (3), prefill (1)
and `golden-server.sh compare` (36 cases, port 8099) all identical. No speed run on this commit: it changes one
assignment per MTP fill.
