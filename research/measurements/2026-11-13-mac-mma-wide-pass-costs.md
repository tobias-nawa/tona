# Mac: pass costs W5 to W8 with one MMA dispatch for up to 8 tokens (#108)

M1 Max 32 GB. `TONA_METAL_MMA_WIDE=1` (`cfd7475`, research/measurements/2026-11-12-mac-w8-mma.md) runs every rest of
4 or more tokens in one MMA dispatch, so W5 to W8 get cheaper; W1 to W4 do not change. The server's width choice
needs the costs (`metal_pass_costs()`, `engine/backend/metal/backend.mm`), for DM and DM-T.

## Plan (written before the run)

- Scope: `tona-bench MODEL -b W -p 512 -d 64 -r 3 --ctx 4096` for W = 5 to 8 and MODEL = DM, DM-T; per width and model
  A B B A with A = default (MMA in 4-token chunks), B = `TONA_METAL_MMA_WIDE=1`; 32 runs, about 9 minutes. Build
  `a81c960` (engine as `cfd7475`). Quiet check and `hw_report` before and after.
- Expectation: DM-T as probe 1 of the W8 file (measured with 19 % other CPU load at its end): −12.2, −15.0, −19.0,
  −17.6 ms, each ± 2.5 ms. DM: the same within ± 3 ms (the wide path changes the MMA chunking, not the Q6_K layout).
- Use: wide tables = the MMA tables (`mma`, `mma_tiled`) plus B − A for W5 to W8.
- Active tooling: `TONA_METAL_MMA_WIDE=1` in B only.

## Run (00:23 to 00:32)

Conditions: commit `a81c960` (tree dirty only by this file); quiet check ok at 00:23:20; `hw_report` load 2.11 / 1.81 /
2.23, other programs 3.7 %; after (00:32:08) "the GPU is 99 % busy" (the run's tail), load 0.60 / 1.01 / 1.61, other
programs 4.0 %. Raw: [2026-11-13-mac-mma-wide-pass-costs.log](2026-11-13-mac-mma-wide-pass-costs.log).

| Model | Width | A (pass, ms) | B wide (pass, ms) | B − A (mean of pairs) | Expected | Table MMA → wide |
|---|---|---|---|---|---|---|
| DM | 5 | 120.23 / 121.01 | 106.60 / 106.22 | −14.2 | −12.2 ± 3 | 120.5 → 106.3 |
| DM | 6 | 134.96 / 134.50 | 118.36 / 117.99 | −16.6 | −15.0 ± 3 | 134.9 → 118.3 |
| DM | 7 | 151.79 / 152.20 | 130.74 / 130.76 | −21.2 | −19.0 ± 3 | 152.6 → 131.4 |
| DM | 8 | 159.04 / 159.19 | 141.74 / 142.52 | −17.0 | −17.6 ± 3 | 161.4 → 144.4 |
| DM-T | 5 | 116.85 / 116.88 | 104.08 / 103.82 | −12.9 | −12.2 ± 2.5 | 117.0 → 104.1 |
| DM-T | 6 | 131.22 / 131.79 | 115.51 / 115.37 | −16.1 | −15.0 ± 2.5 | 131.8 → 115.7 |
| DM-T | 7 | 146.04 / 146.18 | 127.10 / 127.11 | −19.0 | −19.0 ± 2.5 | 146.4 → 127.4 |
| DM-T | 8 | 155.11 / 155.41 | 138.10 / 137.98 | −17.2 | −17.6 ± 2.5 | 157.0 → 139.8 |

All within the expectation; DM-T repeats probe 1 within 0.7 ms on a quiet machine. Two runs of an arm differ by at
most 0.8 ms. The tables `wide` and `wide_tiled` in `backend.mm` take the MMA tables plus B − A; the server uses them when
`TONA_METAL_MMA_WIDE` is on.
