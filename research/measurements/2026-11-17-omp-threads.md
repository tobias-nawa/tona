# OpenMP thread count on the Ryzen 9 9950X (#164, baseline step 1)

Plan, written before the first run. Results below the line `## Results`. Build: `work/build/b5afe` (main `5afe069`, built through the CPU queue; the same
engine sources as `b164`). Model and pin as of today (Coder shard 1, `work/models/coder-iq1_m/`); the shard decision is open, so the numbers of this file
hold for the current pin and the question (threads) is independent of it.

## Question and decision

Every launcher sets `OMP_NUM_THREADS=6` (the 9600X tuning; list and file:line in `2026-11-17-9950x-baseline.md`). On the 9950X (16 cores / 32 threads, 2 CCDs) that
pins the CPU expert path (`engine/cpu/experts.cpp:351-360`, `expert_threads()`) and all other OpenMP regions to 6 threads. Without the variable the CPU experts take the physical cores (16) and the other
regions all 32 logical threads. Owner rule: "always full performance" on the 9950X. Decision per result:

- B (16) decode or prefill faster than A (6) by more than the margin: the default for the launchers becomes the faster value (then 12 and 32 are tried to find the best);
- equal: keep 6 as a safe default and note that 16 costs nothing, ask the owner (full CPU use for no gain is a power question, not a speed one);
- slower: the 9600X lesson (SMT threads lose) repeats; keep 6 and look at the best count between (8, 12) before concluding.

No default is changed by this file; the coordinator decides after seeing the numbers.

## Scope (every GPU job through `scripts/gpu_queue.py --kind exclusive`, wrapped so that exit 75 of the desktop guard requeues the same job)

`scripts/ab.py --a b5afe --b b5afe --a-env OMP_NUM_THREADS=6 --b-env OMP_NUM_THREADS=16 --workload short,mid,set --max-rounds 4 --min-rounds 2`,
`--load-log on`, port 8099. Decode and prefill are separate metrics (`decode@2000`, `decode@2000:write`, `prefill@2000`, `prefill@2000:write`, `decode@30000`, `prefill@30000`,
decode-code and decode-text of `set`); `--primary` is all of them. Rounds alternate A, B.

| Run | Arms | Next step |
|---|---|---|
| 1 | A 6, B 16 | if any metric differs: run 2 |
| 2 | A 16 (or the better of run 1), B 12 | |
| 3 | A the better so far, B 32 | |
| 4 | A the best, B 8, only if 12 or 16 beat 6 and the best is not clear | |

`hw_report.py` and `quiet_check.py` before and after each run (hw_report must show 16 cores / 32 threads, `/sys/devices/system/cpu/smt/active` = 1 is copied).
Machine quiet is required; the desktop guard decides when the run may start (exit 75: requeue, note the lost round).

## Why this size

- Decode is where the CPU expert path acts (misses and CPU-computed experts, 2.4 ms and more of a 16 to 18 ms pass at 86 % hits): workloads `short` (2K prompts, 400 tokens, 2 requests: code and text)
  and `set` (the 14-prompt production mix, MTP on), the two named in `docs/contribute/measurement.md` for decode without an attention change. `mid` (30K prompt, 200 tokens) is added for **prefill**: the OpenMP regions
  of the prefill path (`prefill_entry.cpp:267,290`, `prefill_mtp.cpp:74`) have no `num_threads` and follow the variable, so a prefill effect cannot be excluded; 30K is the smallest prompt at full prefill groups.
  Not `long` or `huge`: the question is the thread count, not attention or KV.
- Rounds: 2 minimum (the noise table is from the 9600X and not yet proven here: `decode@2000` 0.23 %, `decode@30000` 0.28 %, `prefill@30000` 0.04 %, `decode-code` 0.07 % with df 2),
  4 at most. The expected effect is large against that noise or zero; `ab.py` stops when every rate is decided.

## Expectation (every number with tolerance and source)

Sources: `docs/contribute/measurement.md` (margin 0.5 %), `bench/noise.toml`, `2026-11-14-scale-run.md` (9600X, 6 threads), the CU test (`2026-11-17-cu-scaling.md`: host.y, the CPU expert wait, is 0.8 / 3.1 / 7.3 ms per pass
at W = 1 / 4 / 8 with 6 threads, of 23.5 / 36.4 / 54.7 ms).

| Metric (B 16 against A 6) | Expectation | Reason |
|---|---|---|
| **decode** `short` (code, text), `set` code | -0.5 to +4 % | at most the CPU part that is visible beyond the GPU steps can shrink: host.y 3.1 ms at W4 is partly hidden; 2.7x the cores and the memory path of two CCDs (no longer one fabric link, 58 GB/s on the 9600X) bound the gain to about 1 to 3 ms of 36, i.e. 3 to 8 % of a pass in the best case, less after hiding |
| **decode** `set` text, `mid` decode | -0.5 to +4 % | same, more misses on text |
| **prefill** `mid`, `short` | -0.5 to +1 % | the OpenMP regions of prefill are small host loops (`n >= 16` guard); more threads may add a little synchronization cost |
| tokens, hit and miss counts | identical | thread count does not change the numbers (`TONA_DETERMINISTIC` not set; MTP acceptance is bit-identical per input) |
| run 2 and 3 (12, 32 against the best) | 32 not better than 16 (SMT threads share cores: lost on the 9600X); 12 within 1 % of 16 | `experts.cpp:349` comment (9600X: 12 SMT threads slower than 6) |

A result outside its range is a finding. A slower B is a result, not an error.

## Active tooling

`TONA_DIAG=1` (rate-neutral), `--load-log on` (sums only: our CPU time, system, other; used to see whether 16 threads actually run: own CPU share), `TONA_EXPERT_PROFILE=work/profile/experts-coder.bin`
(set by `ab.py`), `ROCR_VISIBLE_DEVICES=0`, `--expect` bands as in the table, `--expect-source` this file. No other `TONA_*`.

## Test conditions

`python3 tools/hw_report.py` of each run is at the head of `/var/tmp/claude/164/sweep-<run>.log.tc` (copied into `2026-11-17-omp-threads-<run>-logs/` by ab.py's own logs; the tc files are
summarised here): AMD Ryzen 9 9950X, 16 cores / 32 threads, kernel 7.3.4, RX 9060 XT, build `b5afe` (commit `96b4c20` plus the main of the day; engine sources equal to `b164`), 60.5 GiB RAM.
`quiet_check` before and after each run: warning only for 1.27 GiB VRAM held by the desktop; load average 1.3 before run 1. The load log of the server starts shows "CPU other" 2.5 to 5.6 %
except the first start of run 1 (5.6 %) and **the first start of run 2 (45 %: another job's build or tool ran during that start)**; rounds 1 and 2 of run 2 carry that disturbance (round 2 has two outlier rates, marked by `ab.py`).
All jobs went through the queue (exclusive, desktop guard active; exit 75 did not occur).

## Results

Files: `2026-11-17-omp-threads-r1.md/.json` (A 6, B 16), `-r2.md/.json` (A 16, B 12), logs in `-r1-logs/`, `-r2-logs/`. Same build both arms, only `OMP_NUM_THREADS` differs.
Round 1 of each run is AB, round 2 BA, alternating; every round is listed in the `.md` files.

### Run 1: 16 threads against 6 (today's launcher value), 2 rounds, all metrics decided

| Metric (B 16 against A 6) | A 6 | B 16 | B against A [95 % CI] | Plan | Verdict |
|---|---:|---:|---|---|---|
| **decode** `short` code (2K prompt, 400 tokens) | 76.7 | 78.0 tok/s | **+1.78 % [+1.16, +2.40]** | -0.5 to +4 % | within |
| **decode** `short` write | 126.4 | 131.7 tok/s | **+4.25 % [+2.77, +5.73]** | -0.5 to +4 % | **just above** (CI overlaps the range) |
| **decode** after 30K prompt | 70.0 | 70.9 tok/s | **+1.27 % [+0.71, +1.84]** | -0.5 to +4 % | within |
| **decode** bench-set code | 94.8 | 97.0 tok/s | **+2.38 % [+0.95, +3.81]** | -0.5 to +4 % | within |
| **decode** bench-set text | 68.6 | 69.5 tok/s | **+1.31 % [+0.20, +2.42]** | -0.5 to +4 % | within |
| **prefill** 2K (code) | 1,464.7 | 1,463.5 tok/s | -0.08 % [-0.31, +0.14] | -0.5 to +1 % | equal |
| **prefill** 2K (write) | 1,740.5 | 1,740.2 tok/s | -0.02 % [-0.42, +0.38] | -0.5 to +1 % | equal |
| **prefill** 30K | 1,980.7 | 1,980.7 tok/s | +0.00 % [-0.11, +0.11] | -0.5 to +1 % | equal |

Spread: A rounds differ by 0.1 to 1.5 % (decode), B by 0.0 to 0.2 %; tokens (SHAs) identical in A and B for all three prompts; hit/miss counts identical in B (131,540 / 21,309 at 2K), A's first round counted
445 hits fewer (round-1 start effect, not a thread effect). Server CPU time of "ours" at the decode starts: 163.5 % (A) against 226 % (B) of one core: B really runs more threads.
**Result: 16 threads are faster for every decode rate (+1.3 to +4.3 %), prefill is unchanged.** The gain is larger where the CPU expert path matters more (write: more misses, +4.3 %; the long 30K prompt: +1.3 %).

### Run 2: 12 threads against 16, 4 rounds (the first two disturbed by other CPU load, see above)

| Metric (B 12 against A 16) | B against A [95 % CI] | verdict of `ab.py` |
|---|---|---|
| decode `short` code / write | -0.35 % [-1.68, +0.98] / +0.38 % [-0.45, +1.21] | open (undecided at round 4); rounds 1, 3, 4: +0.56, -0.06, +0.03 % and +0.25, +0.45, +0.43 % |
| decode after 30K | -0.38 % [-1.22, +0.45] | open; rounds without the outlier: +0.02, +0.25, +0.00 % |
| decode bench-set code / text | -0.39 % [-1.67, +0.90] / -0.22 % [-1.22, +0.79] | open; rounds 1, 3, 4: -0.10, +0.10, +0.21 % and 0.00, +0.14, +0.14 % |
| prefill 2K (code, write), 30K | -0.13 %, -0.21 % (equal), +0.07 % (equal) | equal |

12 and 16 threads are the same within the noise: the three undisturbed rounds all lie within +-0.5 %, the one outlier round (BA with a start at 45 % foreign CPU load) pulls the means to -0.4 %.
A 16-thread default is not worse than 12 and has no measured reason to be lowered.

### Reading and decision input

- 6 -> 16 threads: **decode +1.3 to +4.3 %** (code +1.8 to +2.4 %, text +1.3 %), prefill 0.0 %. Gain on every decode metric with a CI above zero; the largest on the write workload.
- 12 against 16: equal. So the curve is flat between 12 and 16 and clearly above 6; where it starts to flatten (8 or 10) is not measured (plan run 4: only if the best is unclear; not needed for the "full CPU" rule).
- **Run 3: 32 threads (B) against 16 (A), 4 rounds, run at 01:12 to 01:30 on 11-18 after the queue (desktop guard) let it start; quiet check warning only for the desktop's 1.97 GiB VRAM, "CPU other" 3.2 to 4.6 %.**
  All four rounds agree (AB, BA, AB, BA): B (32) is **slower** on decode, prefill equal.

  | Metric (B 32 against A 16) | A 16 | B 32 | B against A [95 % CI] |
  |---|---:|---:|---|
  | **decode** `short` code | 77.1 | 76.7 tok/s | **-0.46 % [-0.79, -0.14]** (B slower, every round -0.39 to -0.55 %) |
  | **decode** `short` write | 128.6 | 127.0 tok/s | **-1.14 % [-2.00, -0.28]** (B slower, -0.90 to -1.36 %) |
  | **decode** after 30K | 70.3 | 70.1 tok/s | -0.28 % [-0.66, +0.11] (open; every round -0.21 to -0.34 %) |
  | **decode** bench-set code / text | 95.8 / 68.8 | 95.3 / 68.5 tok/s | -0.13 % [-1.22, +0.97] / -0.29 % [-0.99, +0.41] (open) |
  | **prefill** 2K code / write / 30K | 1,452.0 / 1,727.4 / 1,917.3 | 1,452.5 / 1,727.3 / 1,915.7 tok/s | +0.04 % / +0.03 % / -0.03 % (equal; 30K open within +-0.13) |

  Server CPU time of "ours" at the decode starts: 237 % (A, 16 threads) against 319 % (B, 32): the 32 threads run, and they cost a little (SMT siblings share cores and the memory path).
  **The expectation holds: 32 is not better than 16 (decode -0.1 to -1.1 %, prefill equal). No reason to use the logical CPUs; the physical cores are the optimum of the points measured (6, 12, 16, 32).**
  The absolute rates of this run are 0.5 to 1 % under run 1 (e.g. `decode@2000` A 77.1 against 78.0 at 16 threads): the spread between runs on different evenings (ab.py compares inside a run only).
- The default of the launchers (`OMP_NUM_THREADS=6` in `ab_server.py`, `tona-service.sh`, `installer/core/unit.cpp`, the justfile, golden scripts) is **not changed**. Without the variable the engine already picks 16
  for the CPU experts (`experts.cpp:351-360`), but the other OpenMP regions then use all 32 logical threads; run 1 measured B with the variable set to 16 for all of them. Setting `OMP_NUM_THREADS` to the physical core count
  (16) in the launchers is the measured configuration.
- Not covered: a default that depends on the machine (installer: physical cores, as `expert_threads()` computes), the planner constant "60 us at 6 threads" (`experts.cpp:350`), and the CPU-expert microbenchmark
  at 8 to 16 threads (`test_cpu_experts`, 44 to 69 GB/s at 6 threads in `2026-11-15-ornith-o1-o2-cpu.md`).

### Default thread count: the measured configuration is now the engine's default (2026-11-18)

`backend/runtime_env.h` (the one-time restart that already sets `OMP_WAIT_POLICY` and `HSA_USERPTR_FOR_PAGED_MEM`) also sets `OMP_NUM_THREADS` to the physical cores when it is unset: `physical_cores()`, logical CPUs halved with SMT (6 on the 9600X, 16 on the 9950X).
`cpu::expert_threads()` uses the same function; the launchers (`ab_server.py`, `tona-service.sh`, the installer's systemd unit, the justfile on Linux, the golden, profile and bench scripts) no longer set 6; macOS keeps what its launcher sets (the engine sets it on Linux only).
A value the user or a script sets stays. The planner comment (`engine/cpu/experts.cpp`, `backend/cpu_cost.h`) says that the 60 us per CPU expert was measured at 6 threads on the 9600X.

Checks (job `verify.sh`, queue kind gpu, build `b164t` = this branch, 2026-11-18 02:17 to 02:22, all committed code):
- process environment after the restart (`/proc/<pid>/environ` of `tona-gpu-gen` started with `OMP_NUM_THREADS` unset): `OMP_NUM_THREADS=16`, `OMP_WAIT_POLICY=passive`, `HSA_USERPTR_FOR_PAGED_MEM=0`; started with `OMP_NUM_THREADS=6`: stays 6.
- goldens in deterministic mode with the default (16) threads, against the references on disk: `gpu` (csv, rust, explain), `cpu plan` (the CPU expert path, where the thread count acts), `prefill dflash`, and the 36 cases of `golden-server.sh`: **all identical** (43 of 43 "identical", every `rc=0`). Tokens do not depend on the thread count.
- unit tests: `just test-cpu` equivalent (`ctest -L unit`): 65 of 65 pass, including `runtime_env` with the new cases (`physical_cores` 32 -> 16, 12 -> 6, SMT off, unknown; `plan` leaves a set value alone, sets 16 when unset, does not loop after the restart); installer tests 4 of 4 including the unit text (no `OMP_NUM_THREADS` line).
- code gates (`pre-commit run`: clang-format, lizard, ruff, shellcheck, gitleaks, file length): pass. clang-tidy (needs the compile database, `just check`) was not run here.
