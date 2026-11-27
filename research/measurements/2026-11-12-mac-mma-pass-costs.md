# Mac: pass cost per width with the half MMA mat-vec, for the server's width choice (#108)

M1 Max 32 GB. The server picks the draft width from the backend's pass costs (`metal_pass_costs()`,
`engine/backend/metal/backend.mm`). With `TONA_METAL_MMA` the dense mat-vecs of every 4-token chunk run on the MMA
kernel (`MV_BMAX = 4`: a pass of 5 to 7 tokens holds one chunk of 4, a pass of 8 two), so the costs of W = 4 to 8
change. Coordinator 2026-11-12: MMA on DM only (DM-T after its W2 finding), the W ≥ 4 costs from a measurement.

## Plan (written before the run)

- Scope: `tona-bench -b W -p 512 -d 64 -r 3 --ctx 4096` (decode passes after a 512-token prompt, as the
  context of the cost table) for W = 4 to 8; per width A B B A with A = DM, B = DM with `TONA_METAL_MMA=1`; 20 runs,
  about 10 minutes. Build `efeaf66`.
- Expectation (run 4 of 2026-11-12-mac-q6k-tile.md: DM + MMA −6.8 ms at W4): W4 −6.8 ms (−5 to −8.5); W5, W6, W7
  one chunk each, −6.8 ms (−5 to −8.5); W8 two chunks, −13.6 ms (−10 to −16).
- Use: the table gets `pass[W] − (B − A)` for W = 4 to 8 when MMA is on; W1 to W3 are untouched (no 4-token chunk).
- Active tooling: `TONA_METAL_MMA=1` in B only.

## Run (20:58 to 21:03)

Conditions: commit `efeaf66` (tree dirty only by this file); quiet check ok at 20:57:58; `tools/hw_report.py` 20:58:
load 1.42 / 2.23 / 2.13, other programs 6.9 % of all CPU threads; after (21:03:43) "the GPU is 99 % busy" as the
last run ended. Raw: [2026-11-12-mac-mma-pass-costs.log](2026-11-12-mac-mma-pass-costs.log).

| Width | A = DM (pass, ms) | B = DM + MMA (pass, ms) | B − A (mean of pairs) | Expected | Table without / with MMA |
|---|---|---|---|---|---|
| 4 | 92.37 / 92.51 | 85.34 / 85.18 | −7.2 ms | −6.8 (−5 to −8.5) | 94.2 / 87.0 |
| 5 | 126.11 / 126.06 | 119.12 / 119.34 | −6.9 ms | −6.8 (−5 to −8.5) | 127.4 / 120.5 |
| 6 | 140.05 / 141.67 | 133.11 / 135.14 | −6.7 ms | −6.8 (−5 to −8.5) | 141.6 / 134.9 |
| 7 | 159.11 / 158.72 | 152.24 / 151.90 | −6.9 ms | −6.8 (−5 to −8.5) | 159.4 / 152.6 |
| 8 | 171.08 / 172.11 | 159.85 / 159.88 | −11.7 ms | −13.6 (−10 to −16) | 173.1 / 161.4 |

Every width inside the expectation. A chunk of 4 saves 6.7 to 7.2 ms wherever it sits; the second chunk of W8 saves
less (−4.8 ms), unchecked why. The spread between the two runs of an arm is at most 2.0 ms (W6), the effect 3× that
or more. The table keeps the measured costs without MMA (2026-11-12-mac-moe-grouped.md) and subtracts B − A.
