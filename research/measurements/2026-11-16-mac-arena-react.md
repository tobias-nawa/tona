# Mac: the arena without a reserve, giving back what other programs lose (2026-11-16)

## Plan (written before the run)

- **Question** (first step of the Mac constants: no proactive reserve): can the arena drop its reserve (5.0 GB on
  this machine: `RAM_SLACK_GB` 0.5, `PROMPT_MIN_GB` 1.0, `TIERED_CACHE_MIN_GB` 1.5, `TONA_METAL_ARENA_FREE_GB`
  2.0) when it reacts to other programs' compression instead?
- **Build** (`TONA_METAL_ARENA_REACT=1`, with `TONA_METAL_ARENA_GOVERN=1`): the arena is planned up to the working set
  and filled in steps; the monitor samples the machine's compressions every 50 ms, and what exceeds our own (our
  compressed pages) is owed: before the next pass (or idle) the arena frees that many bytes, the highest blocks first;
  before a pass it reads back a growth step once the median gap between demand signals passed clean
  (`2026-11-16-mac-purgeable-hold.md`). tona-check: identical (69 steps), 96 give-backs, 17.2 down to 12.65 GB.
- **Scope**: `wired_arm` of `2026-11-16-mac-arena-default-wired.md` (one server per arm, 45 s start, 60 s idle,
  60 s of 200-token requests), arms A no arena, B the default budget (with the reserve), R react; one run each,
  about 12 min.
- **Expectation**: A and B as at 23:20 (123 MB and 0 MB compressed during requests, 34.4 and 29.6 tok/s); R
  compresses during the first requests, then little (under 500 MB in the 60 s), settles between B's 13.65 GB and the
  ceiling 17.05 GB, decode between B and A (30 to 34 tok/s); idle 0 MB in all arms.
- **Decision rule**: the reserve goes (R becomes the arena's governor, still opt-in until the balloon check) if R
  compresses at most a tenth of C's 5.4 GB per minute and decodes no worse than B.
- **Active tooling**: the sampler (one Python process at 20 Hz), stop sleep and keepalive defaults.

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
- Load average (1, 5, 15 min): 4.41, 3.39, 2.72
- CPU use by other programs over 0.5 s: 3.6 % of all threads
- sandbox: no
- Commit: `628767b` (dirty tree)
```

```
- Quiet check (2026-11-16 23:38:45): warning
  - warning: load average 4.41 is high (it lags a minute)
```

## Result, run 1 (23:40 to 23:51)

Quiet check before (23:40:26) ok, after (23:51:24): other programs 5.0 % CPU. Swap 21.2 of 22.5 GB in use.

| Arm | Start (45 s) | Idle (60 s) | Requests (60 s) | Wired during requests | Decode median | Arena at the end |
|---|---|---|---|---|---|---|
| A no arena | 0 MB | 0 MB | 1314 MB (35 samples) | 28.66 GB | 34.6 tok/s (12) | - |
| B default budget | 21 MB | 0 MB | 0 MB | 26.20 GB | 29.9 tok/s (10) | 13.74 GB |
| R react | 5997 MB (4 samples) | 0 MB | 915 MB (7 samples) | 27.99 GB | 30.1 tok/s (10) | 15.63 GB |

- **R reacts**: filled to 17.05 GB (no compression during the fill), the first passes then made the system compress
  5.2 GB of other programs' memory, R gave it back at once (down to 10.99 GB), read back 512 MB steps while clean and
  settled at 15.6 GB; the wait before a step doubled from 19 to 608 samples (30 s) as steps kept failing.
- **Against the decision rule: not met** (915 MB in the requests against at most 540 MB), decode met (30.1 against
  29.9). But A without any arena compressed 1314 MB this time (123 MB at 23:20): what the machine compresses under
  requests varies more than the rule's margin, so one run per arm cannot decide it. Run 2: three rounds per arm.
- The start burst is the cost of having no reserve: the fill meets no compression, the first passes' wiring does.

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
- Load average (1, 5, 15 min): 2.29, 2.35, 2.36
- CPU use by other programs over 0.5 s: 9.1 % of all threads
- sandbox: no
- Commit: `628767b` (dirty tree)
```

## Plan, run 2 (written before the run)

- **Scope**: the same three arms, three rounds in rotated order (ABR, BRA, RAB), about 36 min; commit `970a991`
  (the build of run 1, `monitor` split for the complexity gate).
- **Expectation**: from run 1: A 0.1 to 1.3 GB compressed per 60 s of requests, B under 0.1 GB, R 0.3 to 1.0 GB;
  decode A 34.5, B and R 30 tok/s (plus or minus 1).
- **Decision rule** (replaces run 1's, which compared with a reference the machine's own variation exceeds): the
  reserve goes if R's median compression during requests is no higher than A's (the engine without any arena) and
  R's median decode is no worse than B's by more than 1 %.

## Result, run 2 (23:53 to 00:26)

Quiet check before (23:53:05): other programs 10.5 % CPU, load average 2.62 (our run 1 a minute before); after
(00:26:00) ok.

| Round | A requests / decode | B requests / decode | R requests / decode / arena at the end |
|---|---|---|---|
| 1 (ABR) | 8951 MB / 34.4 | 0 MB / 29.6 | 7869 MB / 30.5 / 16.28 GB |
| 2 (BRA) | 156 MB / 34.1 | 0 MB / 29.8 | 4861 MB / 30.5 / 16.77 GB |
| 3 (RAB) | 0 MB / 34.1 | 0 MB / 29.7 | 4068 MB / 30.5 / 16.67 GB |
| **median** | **156 MB / 34.1** | **0 MB / 29.7** | **4861 MB / 30.5 / 16.67 GB** |

Compressed: other programs' memory during the 60 s of requests, machine-wide; decode in tok/s; idle 0 MB in all nine.

- **Decision (the plan's rule): not met; the reserve stays.** R compresses 4.9 GB per minute of requests (A 0.16, B 0),
  at +2.7 % decode over B.
- Why: R regrows to 16.3 to 17.0 GB of the 17.05 GB ceiling; 4 of 15, 7 of 14 and 7 of 14 growth steps were followed
  by other programs' compression, and the give-back (as many bytes as were compressed, 3 MB to 4.3 GB) is mostly
  smaller than the 512 MB step that caused it. Doubling the wait does not end the cycle within a server's 3 minutes.
- The size B's reserve leads to (13.6 GB) compressed nothing in all three rounds: the reserve stands in for what the
  passes wire beyond the idle page counts. A live measure of that is the arena's size at the first failed step:
  next build, a failed step sets a ceiling that growth passes only when free pages rise by a step above their level
  at the failure (another program left).

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
- Load average (1, 5, 15 min): 1.6, 1.62, 1.67
- CPU use by other programs over 0.5 s: 4.7 % of all threads
- sandbox: no
- Commit: `970a991` (dirty tree)
```

## Plan, run 3 (written before the run)

- **Build** (on `cfb6523`, committed with the result): a give-back or a failed growth step (undone at once) sets a
  ceiling at the arena's size then; growth passes it only when free RAM (free, file and purgeable pages) is a growth
  step above its level at that moment. tona-check: identical, 90 give-backs, 11.71 GB at the end.
- **Scope**: B and R, three rounds alternating (BR, RB, BR), the same phases, about 24 min; A from run 2 (median
  156 MB, 34.1 tok/s).
- **Expectation**: R gives back after the first passes and stays near where it gave way (12 to 15 GB); its
  compression during requests under 500 MB; decode 29.5 to 30.5 tok/s (between B's size and run 2's).
- **Decision rule** (run 2's): the reserve goes if R's median compression during requests is no higher than A's
  (156 MB) and its median decode is no worse than B's by more than 1 %.

## Result, run 3 (00:30 to 00:52)

Quiet check before (00:30:32): other programs 5.2 % CPU, load average 6.04 (the tona-check before); after (00:52:25) ok.

| Round | B requests / decode | R requests / decode / arena at the end | R: give-backs, growth steps, ceilings lifted |
|---|---|---|---|
| 1 (BR) | 0 MB / 29.1 | 5583 MB / 30.5 / 16.28 GB | 31, 13, 4 |
| 2 (RB) | 0 MB / 29.9 | 2061 MB / 30.6 / 16.55 GB | 9, 4, 1 |
| 3 (BR) | 0 MB / 28.6 | 3693 MB / 10.1 (3 requests) / 8.13 GB | 2, 0, 0 |
| **median** | **0 MB / 29.1** | **3693 MB / 30.5** | |

- **Decision: not met, against either bar** (A of run 2: 156 MB and 34.1 tok/s; B: 0 MB and 29.1 tok/s). The ceiling
  was lifted 1 to 4 times per run: free RAM as `free_ram` counts it (free, file and purgeable pages) swings by
  more than a step between passes, so it is no signal that another program left. In round 3 one give-back took the
  arena to 8.1 GB and decode fell to 10.1 tok/s.
- Three runs of the react mode failed; it stays opt-in and is not pursued further. For DM-T the best arm is the
  default (no arena); the reserve is decided on the class above the working set (`research/54-mac-constants.md`).

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
- Load average (1, 5, 15 min): 1.87, 2.09, 2.12
- CPU use by other programs over 0.5 s: 5.7 % of all threads
- sandbox: no
- Commit: `cfb6523` (dirty tree)
```
