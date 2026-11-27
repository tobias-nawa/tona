# Mac: DM-T with and without the expert arena (2026-11-16)

Decision this serves: does the arena stop DM-T from displacing other programs' memory (7.7 to 17.8 GB compressed
per DM-T start in 2026-11-15-mac-coder-arena-server-ab.md, 12 to 49 MB per Coder start with the arena), and at what
cost in speed? If it stops it and costs little, the arena becomes the Mac default for every model, not only above
the working set.

## Plan

- Scope: ab.py, workload set,server (8000:100 bench-set at --ctx 32768) as in the Coder A/B. A: DM-T + MTP-164
  without arena (the default: DM-T's GPU weights fit the working set). B: the same with `TONA_METAL_ARENA_GB=auto`.
  No expert profile exists for DM-T: all experts rank equal (5412 of 7872 experts resident in 11.22 GB at a budget
  of 14.94 GB, tona-check run). Rounds 2 to 6, about 15 min each; 30 to 90 min. Runs right after the third attempt
  of the Coder A/B.
- Expectation (B against A; estimate from the Coder with the arena, recall 69 % at 16 GB, 2026-11-15-mac-coder-arena.md,
  and the 68.8 % of experts resident here): decode-code, decode-text, decode@8000 -40 to -5 %; prefill@8000 -30 to
  0 %. Compressed memory per measured start: A 2 to 18 GB, B under 0.5 GB.
- Identity: `TONA_DETERMINISTIC=1 tona-check` DM-T "def quicksort(items):" -n 64: with and without arena
  byte-identical (done before the plan).
- Active tooling: ab.py load log (on), the compressor watcher (stops at +2 GiB for two samples), the backup wait and
  round repeat of ab.py. TONA_DETERMINISTIC off (server mode as in the A/B before).
- Nothing closed (project rule); load as sums in the load table of the result file.

Test conditions before both runs:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 3.86, 3.35, 2.76
- CPU use by other programs over 0.5 s: 5.1 % of all threads
- sandbox: no
- Commit: `58513e3` (dirty tree)
```

```
- Quiet check (2026-11-16 00:30:14): warning
  - warning: load average 3.86 is high (it lags a minute)
```

Memory before (vm_stat, GB): free 14.8, wired 3.3, file-backed 11.2, anonymous 2.7, compressor 1.5; swap total = 22528.00M  used = 21395.50M  free = 1132.50M  (encrypted); backup     Running = 0;

## Result (result file: 2026-11-16-mac-dmt-arena-server-ab.md; the Coder A/B of the same night: 2026-11-16-mac-coder-arena-server-ab.md)

Two rounds each, every metric decided; rounds with a system backup repeated (Coder A/B: 1, this A/B: 2).

| Metric | DM-T | DM-T + arena | B vs A [CI] | Expected | Reading |
|---|---|---|---|---|---|
| prefill@8000 | 169.9 | 162.6 | -4.3 % [-5.5, -3.1] | -30 to 0 % | in range |
| decode@8000 | 25.1 | 13.3 | -47.2 % [-48.9, -45.4] | -40 to -5 % | **finding: worse** |
| decode-code | 38.7 | 16.5 | -57.3 % [-58.8, -55.8] | -40 to -5 % | **finding: worse** |
| decode-text | 27.0 | 14.2 | -47.4 % [-48.9, -45.9] | -40 to -5 % | **finding: worse** |

Coder (arena, profile) against DM-T in the same night: prefill@8000 -15.9 %, decode@8000 -32.1 %, decode-code
-43.0 %, decode-text -48.2 %, all within that plan's expectation. **DM-T with the arena decodes slower than the
Coder with the arena** (decode-code 16.5 against 22.0 tok/s): without a profile DM-T's resident experts are a
random two thirds; the Coder's are its most used.

Memory displaced (vm_stat compressions per server start, MiB, ab.py's load table; the second start of an arm is
the measured one, the page cache already holds that arm's model):

| Arm | first start after the other arm | second start |
|---|---|---|
| DM-T without arena | 9 988 to 19 354 | 9 987 to 15 790 (every start) |
| DM-T with arena | 30 077 to 35 620 (the arena fills: 15 GB of new memory) | 53 to 1 848 (median 71) |
| Coder with arena (Coder A/B) | 1 774 to 15 911 | 0.6 to 4 439 (median 128) |

**Finding: the arena ends the displacement once it is filled.** DM-T without the arena compresses 10 to 16 GB of
other memory at every start, even with its file in the page cache; with the arena 53 to 71 MB per start (one start
1.8 GB). The fill itself compresses 30 to 36 GB once, after a start of the other arm; in use a model stays loaded
and that cost falls once per server start.

**Finding: without a profile the arena costs half of DM-T's decode.** Next step for the decision: a DM-T profile
(its 164 experts are a subset of the Coder's 256, so the Coder's profile may map onto it), then this A/B again.

Test conditions after both runs:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 3.36, 2.47, 2.32
- CPU use by other programs over 0.5 s: 4.9 % of all threads
- sandbox: no
- Commit: `58513e3` (dirty tree)
```

```
- Quiet check (2026-11-16 03:03:44): warning
  - warning: load average 3.36 is high (it lags a minute)
```

Memory after (vm_stat, GB): free 2.2, wired 3.1, file-backed 23.9, anonymous 2.4, compressor 1.8; swap total = 22528.00M  used = 21283.50M  free = 1244.50M  (encrypted); backup     Running = 0;

## Second A/B: with DM-T's own profile (plan, 2026-11-16)

- Scope: as above; B adds `TONA_EXPERT_PROFILE=work/profiles/dmt-12.bin` (2026-11-16-mac-dmt-profile.md: 85.2 % of
  held-out uses in the top 104 of 164, against 63.9 % for the random choice of the first A/B). Build 6601e28 (profile
  counting off without TONA_EXPERT_PROFILE_OUT). Result file: 2026-11-16-mac-dmt-arena-prof-server-ab.md.
- Expectation (B against A): the cold share of the uses falls from about 36 to 15 %, and the first A/B lost 47 to
  57 % of decode at 36 %; with the cache of recent experts absorbing a part, decode-code, decode-text, decode@8000
  -30 to -5 %; prefill@8000 -10 to 0 %. Compressed memory per measured B start under 2 GB, A 10 to 16 GB as before.
- Decision (coordinator): the arena for every model needs DM-T with profile not slower than without the arena; a
  result in the expected range says no, and how far off.
- Active tooling, conditions: as in the first A/B (ab.py backup wait, compressor watcher, nothing closed).

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
- Load average (1, 5, 15 min): 2.32, 3.14, 2.81
- CPU use by other programs over 0.5 s: 6.0 % of all threads
- sandbox: no
- Commit: `6601e28` (dirty tree)
```

```
- Quiet check (2026-11-16 03:21:40): warning
  - warning: other programs use 8.7 % of all CPU threads
```
Memory before: swap total = 22528.00M  used = 21283.50M  free = 1244.50M  (encrypted); backup     Running = 0;

### Result of the second A/B

Full table: [2026-11-16-mac-dmt-arena-prof-server-ab.md](2026-11-16-mac-dmt-arena-prof-server-ab.md). 4 rounds AB BA
AB BA, plus 2 discarded and repeated because a system backup ran. Answers of A and B identical (SHA 727ff658ae742fa7).

| Metric | A (no arena) | B (arena, DM-T profile) | B vs A [CI] | Expected | First A/B (no profile) |
|---|---|---|---|---|---|
| decode-code | 38.9 | 33.5 | **-14.3 %** [-17.4, -11.3] | -30 to -5 % | -57.3 % |
| decode-text | 27.2 | 22.5 | -18.1 % [-22.0, -14.3] | -30 to -5 % | -47.4 % |
| decode@8000 | 25.2 | 21.0 | -17.3 % [-20.0, -14.6] | -30 to -5 % | -47.2 % |
| prefill@8000 | 169.1 | 164.6 | -2.8 % [-5.2, -0.4] | -10 to 0 % | -4.3 % |

All four within the plan. The profile took about two thirds of the arena's loss away; DM-T with the arena stays 14 to
18 % slower in decode than without. By the coordinator's condition the arena does not become the default for every
model.

Memory per start (ab.py load log): A compressed 11.4 to 27.1 GB of other programs' memory at every start (plan: 10 to
16 GB; more than in the first A/B, a finding of the machine's state, not of A). B compressed 28.9 to 31.2 GB at the
first start of each B block (the arena fills) and 0 to 4.2 GB at the starts after it (median 0.4 GB; plan: under 2 GB,
two of eight above). Findings outside the plan:

- CPU of our processes: B 73 to 84 % of one core, A 6 to 7 %. The cold reads (pread, lookahead) cost one core; on the
  M1 Max that is not the limit (other programs 7 to 26 %), but it is load the user sees.
- Lowest free-memory level (0 to 100, higher is better): A 5 to 7, B 13 to 19.

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
- Load average (1, 5, 15 min): 7.11, 3.75, 2.76
- CPU use by other programs over 0.5 s: 16.7 % of all threads
- sandbox: no
- Commit: `6601e28` (dirty tree)
```

```
- Quiet check (2026-11-16 05:02:18): warning
  - warning: other programs use 19.2 % of all CPU threads
  - warning: load average 7.4 is high (it lags a minute)
```
Memory after: swap used 21259.5 of 22528 MB; no backup running.
