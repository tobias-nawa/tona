# Mac: should the arena be the default for a model that fits the working set? (2026-11-16)

## Plan (written before the run)

- **Question** (one decision, the maintainer's open item "arena for all models"): for DM-T, which fits the working
  set, does the arena (profile, governor) displace less of other programs' memory than the model's own file mapping,
  at equal or better decode?
- **Why this measure**: holding memory compresses nothing; passes do, because Metal wires what they bind (4.6 GB wired
  idle, 29.6 GB during requests, 6.5 GB compressed per minute of requests, the arena planned to the working set;
  `2026-11-16-mac-purgeable-hold.md`). Without the arena DM-T's experts are file pages of the mapping that every pass
  touches.
- **Scope**: per arm one tona-server (DM-T + MTP-164, `--ctx 32768`), 20 s after ready, then 60 s idle and 60 s of
  200-token requests back to back (`scripts`-free: curl), the machine's compressions sampled every 50 ms
  (`host_statistics64`), wired memory at the end of each phase, decode per request from the server log. Arms: A no
  arena (`TONA_METAL_ARENA_GB=0`), B the arena with profile and governor (the default budget, not the ceiling), C the
  arena planned to the working set (`TONA_METAL_ARENA_CEILING=1`, for reference). One run each, about 3 min each.
- **Expectation** (from the measurements above and `2026-11-16-mac-arena-stop-cpu.md`): A compresses during the start
  and during requests (its passes page the file in), B and C compress during requests by their wired size, C more
  than B; decode A about 34 tok/s, B and C 32 to 35; idle 0 MB in all arms.
- **Decision rule**: the arena becomes the proposed default for models that fit if B compresses less than A during
  requests at a decode no worse than 1 %; otherwise not.
- **Active tooling**: stop sleep default, keepalive default; the sampler (one Python process at 20 Hz).

## Result (23:20 to 23:32)

Quiet check before (23:20:58) and after (23:31:59): ok. One run per arm; decode from the server's request lines.

| Arm | Arena | Start (45 s) | Idle (60 s) | Requests (60 s) | Wired during requests | Decode median |
|---|---|---|---|---|---|---|
| A | none (the file mapping) | 0 MB | 0 MB | 123 MB (5 of 1200 samples) | 28.72 GB | 34.4 tok/s (12 requests) |
| B | default budget, 13.65 GB (4661 experts, 86.6 % of the uses) | 60 MB | 0 MB | 0 MB | 26.17 GB | 29.6 tok/s (10) |
| C | planned to the working set, 17.05 GB (6271 experts, 96.1 %) | 0 MB | 0 MB | 5399 MB (73 of 1200) | 29.50 GB | 30.5 tok/s (10) |

- **Decision (the plan's rule): the arena does not become the default for models that fit.** B compresses less than
  A (0 against 123 MB per minute of requests) but decodes 14 % slower (cold experts from the SSD); A, the file
  mapping, decodes fastest and compresses little; C wires the most and compresses the most.
- Without the arena the mapped experts are file pages: the system can drop them without compressing anything, which
  is why A displaces little although it wires about as much as C during a pass. The arena's anonymous memory can only
  be compressed or swapped.
- Wired memory while passes run is 26 to 30 GB of 34.4 in every arm: the size that decides what other programs keep.
- One run each and a 60 s window: indicative, not an A/B; the decode gap of B and C to A (10 to 14 %) is larger than
  the spread of the earlier A/Bs on this workload.

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
- Load average (1, 5, 15 min): 1.56, 1.73, 2.16
- CPU use by other programs over 0.5 s: 5.7 % of all threads
- sandbox: no
- Commit: `d94fdd0` (dirty tree)
```
