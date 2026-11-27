# 2026-11-04: rename to Tona, speed and bits unchanged (#43)

## Plan

**Question.** The rename (`ie` → `tona`: namespace, `IE_*` → `TONA_*`, targets, files) changes no computation. Are
decode and prefill speed and the deterministic output SHAs the same as on main 0612252?

**Scope.** One `scripts/ab.py` run, workloads `mid` (30K prompt, 200 decode tokens) and `long` (130K, 200), `--ctx
140000`, MTP, `--env TONA_DETERMINISTIC=1 --env TONA_DIAG=1`.
- Arm A: `work/build/engine` (main 0612252, `ninja: no work to do` before the run) through the wrapper
  `work/build/tona-rename-base/tona-server`, which renames every `TONA_*` variable to `IE_*` and execs `ie-server`
  (the old binary reads only `IE_*`).
- Arm B: `work/build/tona-rename` (branch `43-rename-tona`, 19011b9).
- Rounds from `bench/noise.toml`, ab.py stops when the result is clear. Expected duration 30 to 60 min.

**Why this size.** 30K and 130K are the two prompts with reference SHAs in `bench/sha.toml` (rule 3 check in the same
run); 130K is the long-context case the agents hit. Decode and prefill come from the same requests.

**Expectation.**
- SHAs of both arms equal and equal to `bench/sha.toml` (30000 `6ef803431673370a`, 130000 `453d981764536e4f`).
- Decode and prefill B against A: -0.5 .. +0.5 %. Source: the kernels are the same instruction streams (isa-diff with
  normalised mangled names: 198 of 198 functions equal, `op_route` with a different LDS offset of one `__shared__`
  variable, same total LDS); host code changed only in names.

**Active tooling.** `TONA_DIAG=1`, `TONA_DETERMINISTIC=1`; no profiler. GPU lock taken by ab.py. Nothing else runs on
the machine.

## Result

`scripts/ab.py`, 6 rounds (maximum), 25.2 min; report `work/ab/rename-check.md` / `.json`.

| Metric | A median (range) | B median (range) | B vs A, mean [95 % CI] | Against the plan |
|---|---|---|---|---|
| prefill@30000 | 1903.1 (1830.5-1918.5) tok/s | 1912.2 (1902.3-1918.7) tok/s | +0.94 % [-0.63, +2.52] | outside (+0.5 %) |
| decode@30000 | 65.2 (64.5-65.3) tok/s | 65.2 (65.2-65.3) tok/s | +0.17 % [-0.31, +0.65] | within |
| prefill@130000 | 1784.1 (1761.5-1802.8) tok/s | 1783.2 (1765.7-1803.1) tok/s | +0.03 % [-0.07, +0.13] | within |
| decode@130000 | 63.8 (63.4-65.0) tok/s | 64.2 (63.5-65.0) tok/s | +0.30 % [-0.32, +0.93] | within |

- **Bits:** SHAs of both arms equal, stable over all rounds and equal to `bench/sha.toml` (30000 `6ef803431673370a`,
  130000 `453d981764536e4f`). Expert cache hits and misses identical in every round.
- **The prefill@30000 finding** comes from round 1 alone: A ran first after the warm-up start and gave 1830.5 tok/s
  (B +4.81 %). Rounds 2-6 give -0.65, +0.75, +0.83, -0.03, -0.05 % (mean +0.17 %). Both arms sit on the same two
  levels (about 1903 and 1918 tok/s), so the spread is not a difference between the builds (why the 30K prefill
  switches between two levels is unchecked). The CI contains 0; no slowdown.
- Decode at 30K and 130K and prefill at 130K: equal within 0.5 %.

**Reading.** The rename changes neither tokens nor speed (rule 1 and rule 3 hold). One earlier attempt of this run
was stopped after one round because the machine was needed; it is not counted.
