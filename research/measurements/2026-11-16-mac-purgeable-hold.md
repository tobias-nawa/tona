# Mac: how long the purgeable arena holds a block after a pass (2026-11-16)

## Plan (written before the run)

- **Question** (one decision): the default of `TONA_METAL_ARENA_HOLD_Q`, the quantile of the blocks' reuse gaps (in
  passes) that sets how many passes a block stays non-volatile after its last use. The tona-check runs of the
  purgeable arena showed macOS purging a volatile block between two passes even at pressure level normal (gaps of
  the purged blocks: median 1, p75 2, max 12 passes; 17152 of 18839 wakes found the block purged).
- **Rule for the decision**: the smallest quantile whose wakes (state changes plus the experts read again) cost at
  most 1 % of the decode time; a larger one only holds memory longer.
- **Scope**: `tona-check` on DM-T, `"def quicksort(items):" --ctx 1024 -n 64`, `TONA_DETERMINISTIC=1`, arena auto,
  profile `dmt-12.bin`, governor, `TONA_METAL_ARENA_PURGEABLE=1`; one run each for q = 0.5, 0.9, 0.99, 1.0, about 3 min
  each. A correctness tool, not a speed A/B: the wait per stop and the wake time are read as indications.
- **Logged**: per run the purgeable line (made volatile, non-volatile, found purged, experts read again, wake time),
  the hold line (hold, reuse gaps, non-volatile blocks after a pass's end on average), the wait per stop, identity of
  the 69 steps against the run without the arena.
- **Expectation** (from the gaps above, not measured for all reuses yet): q 0.5 finds most wakes purged and costs
  2 to 5 s of wakes; q 0.9 a few hundred purged; q 0.99 and 1.0 under 100 purged and under 0.4 s of wakes (1 % of 64
  steps of ~0.6 s); non-volatile blocks after a pass's end rise with q towards all 514; every run identical.
- **Active tooling**: as above; stop sleep default (0.25), keepalive default.

## tona-check runs: no answer (21:30 to 21:43)

All four runs identical to the run without the arena (69 steps), and all four alike: made volatile 19871,
non-volatile 18839, found purged 15959 to 17109, experts read again 27962 to 30269, wakes 4.4 to 5.0 s, non-volatile
blocks after a pass's end 273.6 on average, whatever the hold (2, 4, 10, 27 passes). Reuse gaps of all blocks:
median 1, p90 3, p99 9, max 26 passes. Cause: tona-check runs the CPU reference between its steps (about 2.6 s
each, 180 s for 64 tokens), so the engine idles for over a second before every step and the monitor makes every block
volatile; macOS then purged them before the next step, even with 12 GB free. The question needs passes that come back
to back.

Correction for research/53 ("First checks"): the 1 s window's 97 % purged came from the same tona-check pauses, not
from purges between passes; it shows that macOS purges blocks volatile for about 2 s at once, not that it purges
between passes.

## Plan of the server runs (written before them)

- **Scope**: tona-server with the same model and switches (port 8098, `--ctx 32768`), one 200-token coding request
  after another for 150 s (about 19 requests), `TONA_METAL_ARENA_LOG=10000` (the arena's counters and the hold line
  every 10000 stops, then reset); one run each for q = 0.5, 0.9, 0.99, 1.0, about 3 min each. Read: the last full
  interval of each run.
- **Expectation**: with passes back to back the found-purged count falls with q; q 0.99 and 1.0 under 1 % of the
  wakes purged; non-volatile blocks after a pass's end rise with q. Decision as above (1 % of the decode time).

## Result of the server runs (21:48 to 22:01)

Build: this commit's code with the quantile as a switch. The last full interval of 10000 stops per run; all answers
of a run identical, no request failed. Pressure level normal before each run; the q 0.5 run saw a warn event
(other programs' load), during which every block went volatile at each pass's end.

| q | Hold | Reuse gaps median / p99 / max | Found purged | Experts read again | Wakes | Non-volatile after a pass's end | Requests | Decode median |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 2 | 1 / 4 / 7 | 28905 | 117606 | 44.3 s | 0.0 (warn) | 7 | 9.0 tok/s |
| 0.9 | 3 | 1 / 3 / 6 | 6327 | 41183 | 16.7 s | 399.1 of 514 | 11 | 13.2 tok/s |
| 0.99 | 5 | 1 / 4 / 14 | 0 | 0 | 0.02 s | 407.3 of 514 | 26 | 34.3 tok/s |
| 1.0 | 15 | 1 / 4 / 14 | 0 | 0 | 0.02 s | 409.9 of 514 | 26 | 34.3 tok/s |

- **macOS does purge between passes.** With a hold of 3 passes 6327 of 48379 wakes found the block purged: a block
  volatile for a few passes (about 0.1 s) is gone. The reuse gaps are short (p99 4 passes), but a block missed once
  costs a read of its routed experts at the stop, and the decode falls to a third.
- **Decision: q 0.99** (the rule of the plan: the smallest quantile whose wakes cost at most 1 % of the decode time).
  The hold is then 5 passes (about 0.15 s at 34 tok/s); while requests stream about 407 of 514 blocks stay
  non-volatile and about 107 (3.4 GB) are volatile; between requests (idle 1 s) all of them.
- Decode at 34.3 tok/s matches the arena of today in the cost A/B (34.8 median), not measured against it here.

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
- Load average (1, 5, 15 min): 3.39, 3.31, 3.99
- CPU use by other programs over 0.5 s: 7.3 % of all threads
- sandbox: no
- Commit: `108acf5` (dirty tree)
```

```
- Quiet check (2026-11-16 21:30:43, before the tona-check runs): ok
- Quiet check (2026-11-16 22:01, after the server runs): warning
  - warning: other programs use 12.4 % of all CPU threads
  - warning: load average 3.53 is high (it lags a minute)
```

## Plan: one buffer per expert (written before the run)

- **Question**: with `TONA_METAL_ARENA_BLOCK_MB=1` every expert gets its own buffer (the block rounds up to the
  largest expert span, 2 MB). How many bytes stay non-volatile while requests stream, and what do the many buffers
  cost the decode (7872 buffers in the residency set, a state change per routed expert)?
- **Scope**: the server run of above (requests back to back for 150 s, counters every 10000 stops), q 0.99, once
  with 32 MB blocks again and once with one buffer per expert, back to back.
- **Expectation**: non-volatile after a pass's end 1500 to 3000 experts (3 to 6 GB) against 407 blocks (13.7 GB);
  decode within 2 % of 34.3 tok/s; wakes under 0.5 s per 10000 stops; no purged expert, one answer.

## Result: one buffer per expert (22:33 to 22:39)

| Blocks | Hold | Reuse gaps median / p99 / max | Found purged | Wakes per 10000 stops | Non-volatile after a pass's end | Decode median |
|---|---|---|---|---|---|---|
| 32 MB (514) | 5 passes | 1 / 4 / 14 | 0 | 0.02 s | 407 of 514 (13.7 GB) | 34.3 tok/s |
| one per expert (7872, 2 MB) | 49 passes | 2 / 48 / 53 | 9 | 0.05 s | 5630 of 7872 (about 12 GB) | 34.2 tok/s |

- **The buffers cost nothing measurable** (decode 34.2 against 34.3 tok/s, 26 requests each, one answer each).
- **They do not shrink what stays non-volatile:** an expert's own reuse gaps are longer than a block's (a block is
  routed to whenever any of its 15 experts is), so the hold grows to 49 passes and 72 % of the experts stay
  non-volatile. A hold in passes cannot be both fast and small: the working set of 5 s of decoding is most of the
  arena. Quiet check 22:39:35 ok.

## Other programs' compression as the signal (22:42 to 22:49)

Build: the monitor samples the machine's compressions every 50 ms (idle, without Tona, the machine compressed
nothing in 600 samples of 50 ms); whatever exceeds our own compressed bytes and our own re-reads in the sample counts
as other programs' demand, and every block the running pass has not routed to goes volatile at once. Without a
hold: blocks stay non-volatile after use until such a signal. One buffer per expert, server runs as above.

| Run | Signals in 150 s | Compressed in them | Found purged (last interval) | Decode median | Requests |
|---|---|---|---|---|---|
| our re-reads not subtracted | 106 | 3742 MB in the first 10 s (the start), then 6 to 246 MB per 10 s | 3453 | 32.1 tok/s | 23 |
| our re-reads subtracted | 33 | 2085 MB in the first 30 s, then 0 to 302 MB per 30 s | 2871 | 30.6 tok/s | 15 |

- tona-check with the signal on: identical (69 steps) with 66 signals marking blocks in the middle of passes.
- **While Tona holds the full arena (17 GB) the machine keeps compressing other programs' memory**, 0 to 10 MB/s
  after the start (swap 21.8 of 22.5 GB in use, load average 5, other programs 5 % CPU). That is real displacement
  by Tona, and the signal sees it. Making blocks volatile only lends the memory: the next pass routes to them, reads
  them again and takes the memory back, so the displacement goes on and the decode loses 6 to 11 %.
- What would answer it: when other programs' demand continues, a purged block is not read again into the arena
  (its experts come through the staging slots) until the demand has stopped; the arena then shrinks to what the
  machine can spare. Not built.

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
- Load average (1, 5, 15 min): 6.17, 5.58, 3.97
- CPU use by other programs over 0.5 s: 12.9 % of all threads
- sandbox: no
- Commit: `de10478` (dirty tree)
```

```
- Quiet check (2026-11-16 22:49:03): warning
  - warning: other programs use 5.3 % of all CPU threads
  - warning: load average 4.98 is high (it lags a minute)
```

## Plan: giving way under other programs' demand (written before the run)

- **Build**: on a demand signal blocks the running pass has not routed to go volatile (least used first); a stop that
  finds a block purged frees it, and its experts come through the staging slots; before a pass a freed part is read
  back one growth step (512 MB, the governor's) at a time once `wait` samples (50 ms each) came without other
  programs' compression, and a step after which other programs compressed doubles `wait` (a clean one halves it).
  One buffer per expert.
- **Scope**: the server run of above for 300 s (requests back to back), counters every 10000 stops; tona-check before
  it (identical, 25 signals, the arena down to 12.0 GB).
- **Expectation**: other programs' compression stops after the first seconds (signals rare after the first minute);
  the arena settles below the full 17.2 GB (11 to 15 GB); decode between the 30.6 and 32.1 tok/s of the lending runs
  and the 34.3 of a quiet machine, closer to 34; one answer, no failed request.

- Run 1 (22:58 to 23:04): steps judged over one sample: 1183 steps, 92685 buffers given back, 28.7 GB read per 10000 stops, decode 20.4 tok/s; other programs kept compressing (3 to 16 signals per 30 s). Changed before run 2: a step is judged over the median gap between demand signals.
- Run 2 (23:05 to 23:10): 72 steps (48 clean), the arena between 13.4 and 14.5 GB at the end (of 17.2), 16294
  buffers given back, 13.9 GB read per 10000 stops; decode median 32.1 tok/s (last 10 requests 30.8), 47 requests,
  none failed, one answer. Other programs' compression did not stop: 0 to 7 signals and 0 to 540 MB per 30 s.
  Quiet check before ok, after: load average 2.86.

## Where the compression comes from (23:11 to 23:15)

The arena of today planned to the working set (`TONA_METAL_ARENA_CEILING=1`, not purgeable, 17 GB), the machine's
compressions sampled every 50 ms (`host_statistics64`), wired memory at the end of each phase:

| Phase | Compressed | Samples with compression | Wired |
|---|---|---|---|
| server idle, 60 s | 0 MB | 0 of 1200 | 4.62 GB |
| 12 requests of 200 tokens, 60 s | 6515 MB | 66 of 1200 | 29.61 GB |
| the 30 s after | 560 MB | 17 of 600 | 8.68 GB |
| server ended, 30 s | 0 MB | 0 of 600 | 3.24 GB |

- **Holding the arena compresses nothing; running passes does.** While a pass runs Metal wires what it binds (the
  arena, the weight mappings, the caches), and wired memory rises from 4.6 to 29.6 GB of the 34.4 GB: the system makes
  room by compressing other programs' memory. Idle, the same server wires 4.6 GB and compresses nothing.
- So the signal of the purgeable runs saw real displacement, but its cause is the wired size during passes, not
  memory kept between them. Volatile marks cannot help there (a block in use is wired and non-volatile anyway); what
  helps is a smaller set the passes wire, which the step-wise give-way found by probing (13.4 to 14.5 GB) at the
  price of compression bursts at each probe.
- What decides the arena's size is then: wired during passes = arena + weights + caches, against what other
  programs use. That is measurable per machine at the moment it matters (the compression while passes run).

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
- Load average (1, 5, 15 min): 1.9, 1.97, 2.7
- CPU use by other programs over 0.5 s: 5.4 % of all threads
- sandbox: no
- Commit: `d9144ba` (dirty tree)
```
