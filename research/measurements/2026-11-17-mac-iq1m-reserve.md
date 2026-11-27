# Mac: the arena's reserve on the Coder IQ1_M, 5.0 against 3.5 GB (2026-11-17)

## Plan (written before the run)

The plan of `research/54-mac-constants.md` ("Ready to run"), with these settings:

- **Arms**: the default arena of the Coder IQ1_M (its weights exceed the working set; governor off), profile
  `toolchain/profiles/coder-iq1_m.bin`, MTP head `mtp-256`, `--ctx 32768`. 5.0 GB = today's reserve; 3.5 GB =
  `TONA_METAL_ARENA_FREE_GB=0.5` (the reserve is the sum of its parts, so this takes exactly the 1.5 GB of
  `TIERED_CACHE_MIN_GB` out without a build).
- **Scope**: the `wired_arm` phases (45 s start, 60 s idle, 60 s of 200-token requests back to back), compressions
  sampled every 50 ms, three rounds (BA, AB, BA; A = 5.0 GB), about 25 min. Build `work/build/mtl` of 50fe22a.
- **Guard**: a watcher every 5 s stops the server when the data disk falls under 8 GB free or swap rises 2 GB above
  its level at the start. Deviation from the ready plan: swap is at 18.7 of 19.5 GB (macOS adds swap files as
  needed, so the size is not a limit); the guard bounds the rise instead.
- **Expectation**: B's arena 1.5 GB larger (about 650 more experts of 256 per layer's cut); decode up 2 to 6 %;
  compression during requests 0 to 500 MB in both arms.
- **Decision rule**: the floor goes on the Mac if B compresses no more than A plus the spread of A's own rounds, and
  decodes no worse.
- **Active tooling**: the sampler (one Python process at 20 Hz), the guard (a shell loop every 5 s), stop sleep and
  keepalive defaults.

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
- Load average (1, 5, 15 min): 2.52, 2.7, 2.87
- CPU use by other programs over 0.5 s: 7.9 % of all threads
- sandbox: no
- Commit: `e0ee0cd` (dirty tree)
```

```
- Quiet check (2026-11-17 21:28:16): warning
  - warning: other programs use 5.1 % of all CPU threads
  - warning: load average 2.52 is high (it lags a minute)
```

Possible disturbance: the machine's input idle time (`HIDIdleTime`, IOHIDSystem) was 3 s at 21:29:43, a minute after
the start, while the user had said they were away from the machine; cause not checked at that point (logged every
10 s from 21:35 on, checked after the run).

## Result (21:28 to 21:50)

Guard: never fired (data disk 55 GB free throughout, swap 18654.75 MB at the start and at the end). Input: from 21:35
to 21:50 the input idle time only grew (342 s at the first sample), so the last input was at about 21:29:40: the
user typing their last messages to this session before leaving; no input during the measured phases after that.

| Round | A (5.0 GB): budget / compressed in requests / decode | B (3.5 GB): budget / compressed / decode |
|---|---|---|
| 1 (BA) | 8.83 GB / 0 MB / 16.7 tok/s | 10.27 GB / 397 MB (11 samples) / 20.0 tok/s |
| 2 (AB) | 8.65 GB / 0 MB / 16.6 tok/s | 10.04 GB / 28 MB (4 samples) / 19.6 tok/s |
| 3 (BA) | 8.85 GB / 0 MB / 16.9 tok/s | 10.03 GB / 525 MB (23 samples) / 19.0 tok/s |
| **median** | **8.83 GB / 0 MB / 16.7 tok/s** | **10.04 GB / 397 MB / 19.6 tok/s** |

Decode per request of 200 tokens, 6 to 7 requests per arm and round; idle 0 MB and start 0 MB in all six; wired
during requests A 21.9 to 22.2 GB, B 23.3 to 23.6 GB.

- **Decision (the plan's rule): not met, the floor stays.** B compressed other programs' memory in every round
  (28 to 525 MB per minute), A in none (spread 0); not borderline.
- **The price of the floor is large**: B decodes 17 % faster (19.6 against 16.7 tok/s, the ranges 19.0 to 20.0 and
  16.6 to 16.9 do not overlap) with 1.4 GB more arena (685 more experts, 65.2 against 58.1 % of the profile's uses).
  The 1.5 GB are not a page-cache floor on the Mac (cold reads use `F_NOCACHE`) but the room passes wire beyond
  the idle page counts: at 3.5 GB the passes start to compress other programs. A reserve that is right for this
  machine's load today is about 5.0 GB; it depends on what else runs, so the next step is to measure that room
  instead of fixing it (constant 4 of `research/54-mac-constants.md`).
- On the PC: no change follows; the floor there came from its own sweep (`2026-11-15-expert-tiers.md`).

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
- Load average (1, 5, 15 min): 5.0, 3.29, 2.79
- CPU use by other programs over 0.5 s: 20.0 % of all threads
- sandbox: no
- Commit: `e0ee0cd` (dirty tree)
```

```
- Quiet check (2026-11-17 21:50:10): do not measure
  - do not measure: other programs use 22.5 % of all CPU threads
  - warning: load average 3.81 is high (it lags a minute)
  - warning: the GPU is 6 % busy
```

The quiet check after says "do not measure" (load average 3.81, GPU 6 % busy): the minute after the last server, both
lag our own run.
