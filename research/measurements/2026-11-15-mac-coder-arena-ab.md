# Mac: the Coder with the arena (auto) against DM-T, A/B

Issue #108, part 2. The baseline A/B (2026-11-15-mac-coder-baseline.md) had no result: the Coder thrashed the page
cache (decode 0.24 to 0.49 tok/s). Since then the arena with its defaults (auto budget, 32 cache slots per layer,
lookahead 2; 2026-11-15-mac-coder-arena.md). This A/B gives the first statistical numbers of the Coder on the Mac.

## Plan (written before the run)

**Question.** Decode and prefill of the Coder + MTP-256 with the arena's defaults (governor off) against DM-T +
MTP-164, same build (108-mac-coder a621804, work/build/mtl), on the machine as it is (maintainer: close nothing,
quiet check before and after, available resources in this file, sums only).

**Run.** `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, A DM-T + MTP-164, B the Coder +
MTP-256 with `TONA_EXPERT_PROFILE=toolchain/profiles/coder-iq1_m.bin`, 2 to 6 rounds, margin 0.5 %, the load log, a
discarded start of each arm per round (macOS default). 30 to 90 min. No build and no other measurement meanwhile.

**Abort.** A watcher samples the compressor every 10 s; if the memory the compressor occupies rises 2 GiB over its
level at the start for two samples in a row, it stops the A/B (maintainer: abort if the compressor starts and
report). The arena's auto budget is meant to take only free memory, so such a rise is a finding either way.

**Expectation (B against A).** From the single diagnosis runs of the arena file (Coder at 12 to 15 GB auto: 20 to 25
tok/s on short requests; DM-T 38 to 53 on code, 25 at 8000 tokens; prefill of the 7722-token prompt with the arena
52.9 to 54.5 s against DM-T 53.6 to 54.9 s):

| Metric | Expected | Why |
|---|---|---|
| decode-code | -40 to -60 % | 2.5 to 3 cold experts per layer stop at 1.2 to 1.5 ms read each, about 60 to 80 ms per pass more |
| decode-text | -30 to -60 % | the same stops; text passes accept fewer tokens, so the fixed cost per pass weighs less or more |
| decode@8000 | -30 to -60 % | as decode-text |
| prefill@8000 | -20 to +5 % | the arena runs measured equal; a chunk reads the cold experts of each layer into staging |

A result outside these ranges is a finding.

**Active tooling.** The load log of ab.py; B: `TONA_EXPERT_PROFILE`; defaults otherwise (arena auto in B, keepalive
500 ms). The compressor watcher reads `vm_stat` only.

## Test conditions before the run

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.62, 1.42, 1.51
- CPU use by other programs over 0.5 s: 8.2 % of all threads
- sandbox: no
- Commit: `a621804` (dirty tree)
```

```
- Quiet check (2026-11-15 22:39:09): warning
  - warning: other programs use 8.2 % of all CPU threads
```

Memory before (vm_stat, GB): free 0.1, wired 3.4, file-backed 24.8, anonymous 3.9, compressor 1.3

## Stopped before round 1

Stopped during A's discarded warm-up start (22:41) for the balloon test, which the maintainer moved before this A/B
(2026-11-15-mac-arena-balloon.md); no result. The compressor watcher saw +0.6 to +1.0 GiB while DM-T (arm A) loaded,
free RAM 0.06 GiB: the page cache was full, the compressor rose before the arena's arm ran. The A/B runs again after
the balloon test.

## Second attempt (plan)

As planned above, with build 108-mac-coder 6cb5c55 (blocks of the arena, governor off by default, the autorelease
fix; tempo-neutral against a621804 in the diagnosis runs: 27.4 to 27.6 tok/s at 16 GB against 26.8 to 27.8). Same
expectation, same abort rule (compressor +2 GiB over its start for two samples).

Test conditions before the second attempt:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.3, 2.26, 2.25
- CPU use by other programs over 0.5 s: 5.5 % of all threads
- sandbox: no
- Commit: `6cb5c55` (dirty tree)
```

```
- Quiet check (2026-11-15 23:08:12): ok
```

Memory before (vm_stat, GB): free 15.4, wired 3.4, file-backed 10.1, anonymous 4.0, compressor 0.5; swap total = 22528.00M  used = 21770.94M  free = 757.06M  (encrypted)

## Second attempt (result: aborted in round 2, one full round)

Result file: 2026-11-15-mac-coder-arena-server-ab.md (written by ab.py, partial). The compressor watcher fired at
23:41 (+2.35 GiB over its start) during the second start of arm A (DM-T) of round 2; the A/B was stopped. Round 1
(AB) is complete:

| Metric | DM-T + MTP-164 | Coder + MTP-256, arena auto | B vs A | Expected | Reading |
|---|---|---|---|---|---|
| decode-code | 38.7 | 23.6 | -39.0 % | -60 to -40 % | just above the range: better than expected |
| decode-text | 27.0 | 14.9 | -44.8 % | -60 to -30 % | in range |
| decode@8000 | 24.9 | 18.4 | -26.4 % | -60 to -30 % | above the range: better than expected |
| prefill@8000 | 172.9 | 144.9 | -16.2 % | -20 to +5 % | in range |

One round: no confidence interval, no decision. What the load table shows (per server start, ab.py's load log):

- **The arm that displaces memory is DM-T, not the Coder.** Each DM-T start paged in 2.4 to 24 GB and compressed
  7.7 to 17.8 GB of other memory (its weights are in the file mapping, the page cache takes what it needs). The Coder
  with the arena compressed 12 to 49 MB per start after its first (the first start: 16.8 GB, while the arena filled
  for the first time after DM-T).
- During starts 8 and 9 (DM-T) a system backup ran (backup column 0.94 and 1.0) and the system's CPU share rose from
  14 to 53 %: the compressor jump falls into that backup, on top of DM-T's own page-ins.

Test conditions after the run:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.73, 2.32, 2.12
- CPU use by other programs over 0.5 s: 6.5 % of all threads
- sandbox: no
- Commit: `6cb5c55` (dirty tree)
```

```
- Quiet check (2026-11-15 23:41:51): warning
  - warning: load average 2.73 is high (it lags a minute)
```

Memory after (vm_stat, GB): free 4.0, wired 3.1, file-backed 23.3, anonymous 1.6, compressor 1.3; swap total = 22528.00M  used = 21771.62M  free = 756.38M  (encrypted)

## Third attempt (plan, 2026-11-16)

As the second, with build 108-mac-coder 58513e3 (governor off by default, so the arena arm is the same code as in
the second attempt) and two changes of the procedure: ab.py waits before each server start until no system backup
runs and repeats a round during which one ran; the compressor watcher stops ab.py and its servers itself. Result
file: 2026-11-16-mac-coder-arena-server-ab.md. Same expectation, from the second attempt's round 1 within it.
Nothing closed (project rule); the test conditions are those of 2026-11-16-mac-dmt-arena-ab.md, taken once for both
runs.

## Third attempt (result)

Complete: 2026-11-16-mac-coder-arena-server-ab.md. Two rounds, every metric decided, one round repeated (a system
backup ran). Coder + arena against DM-T: prefill@8000 -15.9 % [-17.0, -14.8], decode@8000 -32.1 % [-35.3, -28.9],
decode-code -43.0 % [-46.4, -39.7], decode-text -48.2 % [-49.6, -46.7]; all within the expectation. The compressor
watcher did not fire. Test conditions after the run and the memory displaced per start: 2026-11-16-mac-dmt-arena-ab.md.
