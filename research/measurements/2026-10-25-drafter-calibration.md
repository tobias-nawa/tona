# Calibration of the drafter probabilities and choice of width, 2026-10-25, own measurement

Server (`scripts/bench-openai.py`, 3 requests of 200 tokens per case), Coder IQ1_M, DFlash2, 80 expert slots (desktop
used 2.1 GB VRAM). Log: `IE_DRAFT_LOG=<file>` (per pass width, accepted, q1 q2 q3, ms of the pass).

## How well do the probabilities fit? (`IE_DRAFT_FULL=1`: always 3 drafts)

| Case | Passes | expected per pass (Σ Π q) | accepted per pass |
|---|---:|---:|---:|
| csv | 153 | 2.69 | 2.82 |
| rust | 177 | 2.39 | 2.25 |
| explain | 241 | 1.78 | 1.42 |
| long (after 13K) | 242 | 1.68 | 1.39 |

Too optimistic for prose, especially at q 0.3–0.85 on positions 2 and 3 (often only half accepted there).

Real pass time per width (ms, without drafter, normal choice): explain B = 2 34.0, B = 3 35.2, B = 4 38.9; long 31.4 / 36.4 /
42.0; csv B = 4 38.7; table in the code 31.3 / 37.0 / 41.7.

## Simulation on the logs (tok/s from the pass-time table, width choice recomputed)

| Table new | csv | rust | explain | long |
|---|---:|---:|---:|---:|
| raw | 84.6 | 72.1 | 54.7 | 54.5 |
| histogram per position | 84.6 | 72.0 | 55.3 | 54.8 |
| logistic online (N 256) | 84.7 | 72.1 | 55.5 | 56.0 |
| oracle (knows the acceptance) | 85.9 | 77.9 | 64.5 | 63.9 |

The probabilities carry little information about the acceptance; calibrating gains only 1–3 % on prose.

## Server (median of 3, mean of 2 runs)

| Decode tok/s | csv | rust | explain | after 13K | mean |
|---|---:|---:|---:|---:|---:|
| raw, fixed table | 87.8 | 76.7 | 57.3 | 52.5 | 68.6 |
| calibrated | 89.2 | 78.0 | 56.3 | 53.4 | 69.2 |
| calibrated + moving pass time | 89.1 | 76.7 | 57.0 | 53.3 | 69.0 |
| moving pass time only | 87.4 | 77.4 | 56.9 | 53.8 | 68.9 |
| new pass times in the table (instead of calibrated) | 89.3 | 77.3 | 55.6 | 52.8 | 68.7 |

First version without step-size control: after csv (almost everything accepted, q ≈ 1) the Newton fit ran away (ω up to
1,400), rust 76.5 → 60.5. With halving of the step, bounds for ω and β and a probe at full width every
32nd pass (otherwise the calibration stays stuck after "no drafts"), stable, ω 0.47–1.62.

Result: calibration on (`IE_DRAFT_CALIB=0` switches it off), moving pass time rejected. The width is no longer a big
lever; the distance to the oracle lies in the information in q, not in the calibration.
