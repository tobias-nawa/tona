# Mac: pass costs of DM-T (W1 to W8, MMA on) for the server's width choice (#108)

M1 Max 32 GB. DM-T (DM with its 128 Q6_K layer matrices in `TONA_Q6_K_T`) is faster than DM at every width since
the 4-row loads (2026-11-12-mac-q6k-tile.md, run 6). The server's width choice takes the pass costs from
`metal_pass_costs()`, measured on DM; coordinator 2026-11-12: DM-T gets its own costs, then a server A/B.

## Plan (written before the run)

- Scope: W1 to W4 from run 6 (same kernels; `dde83c2` only merged host code). New: `tona-bench -b W -p 512 -d 64
  -r 3 --ctx 4096` for W = 5 to 8, A B B A with A = DM, B = DM-T, defaults (MMA on); 16 runs, about 9 minutes.
- Expectation: every pass of W ≥ 5 is a chunk of 4 (MMA, run 6 W4: −2.0 ms) plus 1 to 4 tokens on the R4 path (run
  6: −2.1 / −1.5 / −3.5 / −2.0 ms for 1 / 2 / 3 / 4 tokens): W5 −4.1, W6 −3.5, W7 −5.5, W8 −4.0 ms, each ±1.5.
- Use: the DM-T table is the DM table with MMA (`metal_pass_costs()`) plus B − A per width; the backend picks it
  when the weights hold Q6KT matrices.
- Active tooling: none (defaults).

## Run (22:12 to 22:17)

Conditions: commit `dde83c2` (tree dirty only by this file); quiet check 22:12:21 warning (other programs 9.7 % of
all CPU threads); `tools/hw_report.py` 22:12: load 1.56 / 2.04 / 2.73, other programs 8.3 %; after (22:17:55) "the
GPU is 99 % busy" as the last run ended. Raw: [2026-11-12-mac-dmt-pass-costs.log](2026-11-12-mac-dmt-pass-costs.log);
W1 to W4: [2026-11-12-mac-q6k-tile-run6.log](2026-11-12-mac-q6k-tile-run6.log).

| Width | A = DM (pass, ms) | B = DM-T (pass, ms) | B − A | Expected | Table DM (MMA) / DM-T |
|---|---|---|---|---|---|
| 1 | 45.58 / 45.67 | 43.46 / 43.55 | −2.1 | run 6 | 44.5 / 42.4 |
| 2 | 61.78 / 62.11 | 60.69 / 60.30 | −1.5 | run 6 | 61.2 / 59.7 |
| 3 | 78.68 / 78.84 | 75.28 / 75.23 | −3.5 | run 6 | 78.5 / 75.0 |
| 4 | 86.51 / 86.75 | 84.66 / 84.59 | −2.0 | run 6 | 87.0 / 85.0 |
| 5 | 119.97 / 120.84 | 116.97 / 116.90 | −3.5 | −4.1 ± 1.5 | 120.5 / 117.0 |
| 6 | 134.75 / 135.06 | 131.87 / 131.76 | −3.1 | −3.5 ± 1.5 | 134.9 / 131.8 |
| 7 | 152.33 / 152.85 | 146.55 / 146.21 | −6.2 | −5.5 ± 1.5 | 152.6 / 146.4 |
| 8 | 159.46 / 159.76 | 154.95 / 155.56 | −4.4 | −4.0 ± 1.5 | 161.4 / 157.0 |

W5 to W8 inside the expectation; the sum of a 4-chunk and the rest predicts each within 0.7 ms. Spread between the
two runs of an arm at most 0.9 ms (W5 A). The table holds for DM-T with MMA on (the default); with `TONA_METAL_MMA=0`
a tiled file keeps the DM f32 table (DM-T without MMA not measured at W ≥ 4 since the 4-row loads).
