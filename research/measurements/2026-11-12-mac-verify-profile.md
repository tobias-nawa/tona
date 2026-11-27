# Mac: verify pass per width and the MTP head step (#108)

M1 Max 32 GB, Metal backend, target DM, MTP head cut to 164 experts (2026-11-11-mac-mtp.md).

## Plan (written before the run)

**Question.** What does a pass of width 1 to 8 cost on Metal, what does a fill and a chained step of the MTP head cost,
and where does the time of a width-4 pass go? The server's width choice uses the reference's costs (verify width 1 to
8: 24.2 to 58.1 ms, step 1.15 ms, fill 1.3 ms, `engine/server/generator.h`); M3 (2026-11-12-mac-mtp-server.md)
measured ~150 ms per cycle on code against 49 ms for one single-token pass.

**Scope.**

- `tona-bench DM --mtp mtp-164 -k 3 -p 512 -d 48 -r 3 --ctx 4096 -b W` for W = 1 to 8 (one process per width, one
  warm-up round each, not counted). The pass, the fill and the steps are timed apart; tokens are argmaxes, the head
  runs after every pass on the pass's last argmax with 3 chained steps.
- Then `TONA_METAL_PROFILE=1` for W = 1 and W = 4 (`-r 1`): GPU time per part. Profiled runs are no speed results
  (one command buffer per part, ~25 µs each).
- Expected duration ~6 min.

**Expectation.**

- W = 1: pass 47 to 52 ms (49 in 2026-11-11-mac-decode-attn.md at 2K).
- W = 4: 85 to 115 ms: from M3's ~150 ms per cycle minus ~3.3 head passes of ~12 ms. The routed experts dominate: four
  tokens route to up to 32 distinct experts per layer instead of 8, read once each.
- W = 8: 130 to 180 ms.
- MTP step and fill 8 to 15 ms each (one layer of attention and MoE, the truncated LM head, one command buffer and its
  ~0.2 ms round trip); far above the reference's 1.15 ms.
- Profile W = 4: routed experts over 60 % of the GPU time.

**Active tooling.** `TONA_METAL_PROFILE=1` only in the profile runs; otherwise no `TONA_*` variable.
`caffeinate -dimsu`. Quiet check and `tools/hw_report.py` before and after.

## Open points

- Prefill candidate after the verify pass (coordinator 2026-11-12): the gated-delta kernel of mlx-lm PR 454
  (github.com/ml-explore/mlx-lm/pull/454, Sept. 2025): one kernel loops over the T steps of a chunk, one SIMD group
  per value row with `simd_sum`, state in f32; a packed variant holds 8 value rows per SIMD group (4 lanes each,
  shuffle-xor). Prefill 4.2x, decode +12 % on an M3 Ultra (Qwen3-Next-80B).

## Test conditions

Before (08:20:35): quiet check ok; `tools/hw_report.py` as in 2026-11-12-mac-mtp-server.md (M1 Max, 32 GiB, macOS
27.0.1, no `TONA_*` variable), commit `4efe678` plus the `--mtp` option of `tona-bench` (dirty tree). After (08:23:23,
right after the last profiled process): "do not measure: the GPU is 96 % busy" (the run's own tail, as after M3),
other programs 8.0 % of all CPU threads. AC power, `caffeinate -dimsu`. Raw log: `work/m3/vp.log` (not in the repo).

## Results

Medians over 3 rounds of 48 tokens after a prefill of 512 (spread of all rounds within ±0.7 %):

| Width | Pass | Per extra token | Decode with head, all accepted | MTP fill | MTP step |
|---|---|---|---|---|---|
| 1 | 45.4 ms | | 18.3 tok/s | 2.76 ms | 2.13 ms |
| 2 | 64.3 ms | +18.8 | 27.0 | 3.34 | 2.13 |
| 3 | 86.7 ms | +22.4 | 30.9 | 4.00 | 2.13 |
| 4 | 108.0 ms | +21.3 | 33.5 | 4.87 | 2.15 |
| 5 | 145.3 ms | +37.3 | 31.8 | 5.54 | 2.13 |
| 6 | 162.2 ms | +16.9 | 34.3 | 6.27 | 2.13 |
| 7 | 184.4 ms | +22.2 | 35.4 | 7.15 | 2.14 |
| 8 | 202.5 ms | +18.1 | 37.5 | 4.52 | 2.14 |

"Decode with head" is the upper bound of the server path (every draft accepted, 3 head steps per pass).

GPU time per command buffer of a part (`TONA_METAL_PROFILE=1`, one round; the counts mix the 512-token prefills and
the MTP head's layer into the same part names, so only the ratio W = 4 to W = 1 is read):

| Part | W = 1 | W = 4 | Ratio |
|---|---|---|---|
| routed experts | 0.646 ms | 2.335 ms | 3.6 |
| DeltaNet projections (dense) | 0.308 | 0.882 | 2.9 |
| attention projections (dense) | 0.168 | 0.457 | 2.7 |
| shared expert (dense) | 0.150 | 0.291 | 1.9 |
| gated reads | 0.099 | 0.244 | 2.5 |
| attend | 0.182 | 0.605 | 3.3 |

Shares of the GPU time at W = 4: routed experts 40.6 %, DeltaNet projections 21.4 %, gated reads 11.8 %, attention
projections 7.2 %, shared expert 5.1 %.

**Against the plan.**

- W = 1 45.4 ms: below the expected 47 to 52 (shorter context: 512 against 2K).
- W = 4 108.0 ms: inside 85 to 115. W = 8 202.5 ms: **above 130 to 180** (finding).
- Every token past the first adds ~21 ms (W = 5: 37 ms), 46 % of a whole single-token pass; the reference adds ~6 ms
  (22.9 to 67.6 over 1 to 8). **The dense matrices cost 2 to 3 times as much at W = 4 as at W = 1**, although a dense
  matrix is the same bytes for all four tokens: the mat-vecs of width B do not read each weight once, or are bound by
  something other than bytes. This is the verify pass's main cost, ahead of the routed experts (3.6 times: four tokens
  route to more distinct experts, partly inherent).
- **MTP step 2.13 ms and fill 2.8 to 7.2 ms: below the expected 8 to 15** (finding): the head is cheap; M3's ~150 ms
  per cycle is the verify pass of 4 to 5 tokens.
- Width choice with these costs (acceptance 0.8 per draft, illustration): W = 2 gives 1.8 tokens in 69.6 ms
  (25.9 tok/s), W = 3 25.6, W = 4 24.8, W = 1 20.9. A width choice from Metal's costs moves the server from the
  reference's flat costs to W = 2 to 3 but stays near 26 tok/s; above 40 tok/s needs a cheaper extra token (at
  ~6 ms per token, W = 4 would be ~65 ms and 45 tok/s at the same acceptance).
- `TONA_METAL_PROFILE=1` with an MTP head aborts at the end of the process ("Command encoder released without
  endEncoding"): the head's pass opens its own encoder while the profiler splits it. The totals were printed before.

## Follow-up: the server with Metal's own costs

**Plan (written before the run).** Same as M3 (2026-11-12-mac-mtp-server.md, arm B, `work/m3/m3.sh`, 2 rounds),
with `Backend::pass_costs()` of the Metal backend returning the costs above (pass 45.4 to 202.5 ms, step 2.13, fill
4.0) instead of the reference's. Greedy: the same tokens as M3, only the widths change. ~8 min.
Expectation: code 24 to 28 tok/s decode (M3: 23.3; the illustration above caps near 26 at 0.8 acceptance per draft,
code accepts more), text 19 to 21 (M3: 17.4; with the true costs the choice falls back to width 1 to 2 when drafts
are poor, so text should come near the trunk's 20.4 minus the head's fill). Tokens per pass lower than M3's 3.33.
Active tooling as in M3.

**Results.** Raw: [2026-11-12-mac-verify-profile-server.jsonl](2026-11-12-mac-verify-profile-server.jsonl). Quiet
check before (08:27:43) ok; after (08:34:33) "GPU 98 % busy", the run's own tail as in M3.

| Round | Code decode | Text decode | Tokens per pass | MTP passes | Page-ins |
|---|---|---|---|---|---|
| 1 | 26.2 tok/s (per prompt 23 to 35) | 19.6 (19 to 20) | 2.67 (1,746 passes) | 3,604 | 180,731 |
| 2 | 26.9 (24 to 34) | 19.9 (19 to 21) | 2.67 | 3,604 | 19,695 |

Against the plan: code 26.2 to 26.9 inside 24 to 28 (+14 % over M3's 23.3), text 19.6 to 19.9 inside 19 to 21
(+14 % over M3, still 0.5 to 0.8 below the trunk alone), tokens per pass 2.67 against 3.33 as expected. The width
choice now matches the hardware; the rest of the way to 40 tok/s is the cost of each extra token in the verify pass.

## Addendum: clean profile (plan written before the run)

The profile above mixes the two 512-token prefills of each process (warm-up and round) and the MTP head's layer into
the per-part sums; a prefill buffer is far larger than a decode buffer, and at W = 4 there are a quarter as many decode
passes as at W = 1, so even the W = 4 to W = 1 ratios above are unreliable. Also: `TONA_MV_R=8` (eight rows per
simdgroup in the dense mat-vec, a quick probe, not kept) made passes slower (W = 1 49.9 ms against 45.3, W = 4 115.3
against 105.7; 2 rounds each), so the x loads alone do not explain the cost of extra tokens.

Rerun: `TONA_METAL_PROFILE=1 tona-bench DM -p 8 -d 48 -r 1 --ctx 1024 -b W`, W = 1 and 4, no MTP head; ~1 min.
Per pass = part total divided by the passes (warm-up and round: 2 × 48 / W, plus the two 8-token prefills, which are
small). Expectation: sums near the unprofiled passes plus ~25 µs per buffer; routed experts 30 to 40 % at W = 1 and
the largest share of the time added at W = 4 (four tokens route to up to 32 distinct experts per layer).

**Results of the clean profile** (08:36:49 quiet check warning, other programs 6.1 % of all CPU threads; after:
GPU 97 % busy, the run's own tail). Per pass, ms (W = 1: 98 passes, W = 4: 26; profiled passes 56.8 and 119.3 ms
against 45.3 and 105.7 unprofiled):

| Part | W = 1 | W = 4 | Added | Share of the added |
|---|---|---|---|---|
| routed experts | 14.8 | 52.1 | +37.3 | 51 % |
| DeltaNet projections | 11.6 | 23.3 | +11.7 | 16 % |
| gated reads | 9.8 | 18.5 | +8.7 | 12 % |
| attention projections | 3.6 | 7.9 | +4.3 | 6 % |
| indexer | 1.8 | 5.6 | +3.8 | 5 % |
| shared expert | 3.5 | 6.4 | +2.9 | 4 % |
| LM head | 2.6 | 4.9 | +2.3 | 3 % |
| rest (router, scan, attend, PLE) | 4.0 | 6.4 | +2.4 | 3 % |
| total | 51.6 | 125.1 | +73.5 | |

- The expectation holds: routed experts 29 % at W = 1, half of the time added at W = 4. They supersede the dense
  shares of the first profile above.
- **The routed experts run at ~54 GB/s** at W = 1 (8 experts × 48 layers × ~2.08 MB = 0.80 GB in 14.8 ms; 16 % of
  the 334 GB/s measured read bandwidth) and ~62 GB/s at W = 4 (32 slots, one per token and expert, no merge of
  experts two tokens share). At 200 GB/s they would take ~4 ms at W = 1 and ~16 ms at W = 4: the W = 4 pass from
  108 to ~70 ms. This is the next step, ahead of the dense mat-vecs.
