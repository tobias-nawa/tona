# Mac: the full MTP head against the 256 cut in the release start (2026-11-26)

## Plan (written before the run)

- **Question**: does the full `mtp-q8_0` head (all 512 experts, 4.1 GB), which the release ships, cost the release
  start on the M1 Max the decode it lost against the earlier runs? The release start decoded 17.9 tok/s at 2K and 16K
  ([2026-11-26-mac-release-rates.md](2026-11-26-mac-release-rates.md)); the runs with the head cut to 256 experts
  decoded 21.5 tok/s with a smaller arena ([2026-11-25-mac-priority-check.md](2026-11-25-mac-priority-check.md), arm
  D-bal, 10.77 GB).
- **Arms** (both exactly the release start of 2026-11-26-mac-release-rates.md: build `work/build/rates` of
  `feat/macos-release` 723b7b1e, Coder IQ1_M part 1, `--ctx 32768`, `OMP_NUM_THREADS=6 TONA_SEQS=1 TONA_RAM_GB=19
  TONA_VRAM_GB=26.8 TONA_OEA_K0=0`, profile `toolchain/profiles/coder-iq1_m.bin`, no prompt cache):
  - A: `--mtp` `mtp-q8_0` (pinned in `toolchain/models.toml`, 4,137,429,312 bytes);
  - B: `--mtp` the 256 cut (`work/models/mtp-256`, 2,797,873,472 bytes, sha256 `53564cf2…`, checked before the run;
    keep list `2026-11-15-mac-mtp-keep-256.json`, 95.4 % of the head's routing mass). Its README says: not to be
    published, Qwen Community License 1.0 of the base model, no redistribution.
- **Run**: `scripts/ab.py` of 723b7b1e, `--a-drafter`/`--b-drafter`, workload `server`, targets `2000:200,16000:100`
  (as the release-rates run), code corpus `work/corpus/code.txt`, 3 rounds fixed (alternating AB/BA/AB), warm-up and
  a discarded start of each arm per round, load log on. About 33 min (the release-rates run: 32.6 min for 3 rounds).
- **Expectation** (B against A):
  - decode@2000 and decode@16000: +5 to +25 %. Source: 17.9 tok/s here against 21.5 tok/s with the 256 cut on
    2026-11-25 (+20 %), with a smaller arena then; the cut reads fewer bytes per draft pass and leaves 1.3 GB more of
    the GPU working set to the target's experts.
  - prefill@2000 and prefill@16000: -2 to +5 % (the head's share of a prefill pass is small).
  - output SHAs: equal in both arms (greedy, the drafter only proposes; the target decides every token).
  - tokens per pass: B at most 5 % below A (the cut keeps 95.4 % of the routing mass).
- **Decision rule** (the coordinator's, fixed before): B decodes at least 8 % faster at 2K and at 16K, with equal
  SHAs and tokens per pass not more than 5 % lower → proposal to the maintainer to publish the 256 cut and recommend
  it on the Mac (the licence note above is part of that proposal). Otherwise `mtp-q8_0` stays and the cause is
  elsewhere: one line of hypothesis, no further run.
- **Active tooling**: ab.py's load log, round warm-up; no `TONA_DIAG`, no `TONA_DETERMINISTIC`, no profiler.
- **Machine**: quiet check before and after; on a warning, quieted once, then the run starts anyway with the load
  reported as sums.

## Result (2026-11-26, 02:06 to 03:01)

Quiet check before (02:06:15) and after (03:01:07): **warning**, other programs 6.2 and 6.0 % of all CPU threads
(load average 7.6 before, 1.7 after). Load as sums from ab.py's load log, per start: other programs 9 to 18 % of one
core, once 60 % (start 13, an A start); the system 12 to 49 %; our server 15 to 19 % of one core. Arena budget
13.53 GB (the RAM limit) in all 17 starts of both arms.

ab.py: `work/ab/2026-11-26-mac-mtp-head-check.{md,json}`, server logs `work/ab/2026-11-26-mac-mtp-head-check-logs/`;
3 rounds (AB, BA, AB), 54.4 min. Output SHAs equal in both arms and every round (2K `34926e2aa8162172`, 16K
`4e89a7a1806666c9`, the same as in the release-rates run).

| Rate | A `mtp-q8_0`, per round | B 256 cut, per round | B against A, mean [CI] | ab.py |
|---|---|---|---|---|
| decode@2000 | 17.9 / 16.9 / 17.5 | 17.8 / 17.8 / 17.8 | +1.87 % [-0.49, +4.23] | undecided |
| decode@16000 | 17.9 / 18.0 / 17.8 | 17.6 / 17.6 / 17.5 | -2.04 % [-2.62, -1.45] | B slower |
| prefill@2000 | 157.9 / 157.0 / 157.9 | 160.8 / 160.9 / 160.6 | +2.03 % [+1.36, +2.71] | B faster |
| prefill@16000 | 137.0 / 136.3 / 135.9 | 138.0 / 137.8 / 137.4 | +0.95 % [+0.35, +1.56] | B faster |

| Request | A tokens per pass (draft steps) | B tokens per pass (draft steps) |
|---|---|---|
| 2000 | 3.08 (3.45) | 2.86 (3.03), -7 % |
| 16000 | 2.70 (3.89) | 2.15 (2.93), -20 % |

Read against the plan:

- **Outside the expectation (a finding): the 256 cut is not faster.** Decode +1.9 % at 2K (undecided) and -2.0 % at
  16K, against +5 to +25 % expected; prefill +1 to +2 %. Its passes are cheaper, but it accepts fewer tokens per pass
  (-7 % at 2K, -20 % at 16K, outside the plan's 5 %), and the two cancel.
- **Decision by the rule fixed before: `mtp-q8_0` stays**; the 256 cut is not proposed. The full head does not cost
  the release start its decode.
- **Hypothesis for the gap to 21.5 tok/s (not measured further)**: the runs of 2026-11-17 and 2026-11-25 decoded other
  prompts (the reserve run's 200-token requests, not the 2K and 16K code prompts of the corpus), so 21.5 against 17.9
  compares different texts, not different setups.

Test conditions before:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 7.6, 3.11, 3.11
- CPU use by other programs over 0.5 s: 7.8 % of all threads
- sandbox: no
- Commit: `723b7b1e` (clean tree)
```

Test conditions after:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.72, 1.34, 1.4
- CPU use by other programs over 0.5 s: 4.0 % of all threads
- sandbox: no
- Commit: `723b7b1e` (clean tree)
```
