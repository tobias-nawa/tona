# Mac: RAM governor thresholds from the reserve, shrink and regrowth (2026-11-16)

Decision this serves: do the thresholds of `RamGovernorConfig::from_reserve` (R = the free RAM the auto budget
leaves) let the arena shrink under pressure and return to its full budget afterwards, with identical answers?
Log: 2026-11-16-mac-governor-reserve.log.

## Plan

- Scope: Coder IQ1_M + MTP-256, profile, `TONA_DETERMINISTIC=1`, `TONA_METAL_ARENA_GOVERN=1`, test pressure
  `TONA_METAL_ARENA_GOVERN_TEST_GB=3`, `_S=120` (free RAM reads 3 GB = 0.6 R less for 2 min; nothing displaced).
  Requests of 128 tokens at temperature 0 in a loop: before, during and 5 min after the test pressure, about 8 min
  per run. One run per code state (a test of function, not of speed).
- Expectation: the arena frees about 0.35 R (1.75 GB, from 0.4 R up to the target 0.75 R) and reads it all back
  within 2 min after the pressure ends; every answer equal to the reference; no pass aborted.
- Active tooling: the test pressure, `TONA_DETERMINISTIC=1`; nothing else.
- Test conditions: as in 2026-11-15-mac-coder-arena-ab.md, "Test conditions after the run" (taken 23:41, the
  runs followed directly; nothing closed).

## Results

| Run | Code state | Freed | Read back | Answers | Reading |
|---|---|---|---|---|---|
| bias2 | from_reserve, sample every 250 ms | 3.4 GB | 0 | 37 of 37 equal | no regrowth |
| bias3 | + a log line per change of state | 2.1 GB | 0 | 38 of 38 equal | the state flips calm/band with the passes |
| bias4 | + least free RAM of 10 s, calibration while passes run, growth while passes run | 7.5 GB | 7.5 GB in 15 steps | 50 of 50 equal | regrowth works, release overshoots |
| bias5 | + the window restarts after each step | 1.97 GB in 8 steps | 1.97 GB in 4 steps | 46 of 46 equal | as expected |

Decode in bias5 (deterministic, single requests, budget 8.92 GB as the machine left at start): 14.4 tok/s before,
10.4 to 11.0 during, 13.2 to 14.5 after.

**Finding 1: free RAM swings with the passes.** Metal wires our weight mappings during a pass; idle they count as
file pages, so free RAM reads about 1.8 GB higher between requests (bias3: calm at 6.79, band at 4.98 GB a moment
later without any change of the arena). The governor never saw 30 s of calm while requests ran. It now sees the
least free RAM of the last 10 s (the value in use), and grows only while passes run, at the safe point before a
pass (512 MB in about 0.1 s).

**Finding 2: calibration must wait for the passes.** At start, before the first pass, free RAM read above R
(calibration +0.00); with the passes it read 1.05 to 1.08 GB less. The calibration is now taken once the first 10 s
of passes fill the window.

**Finding 3: the least value must restart after a step.** Otherwise it asks for the same step again until the
window has passed (bias4: 7.5 GB freed for a need of 1.75 GB).
