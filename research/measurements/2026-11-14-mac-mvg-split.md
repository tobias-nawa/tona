# Mac: the slots of one routed expert over several threadgroup rows

Issue #108. Follows [2026-11-13-mac-mv8.md](2026-11-13-mac-mv8.md#run-2-results): an expert kernel that holds 5 or 8
slots per weight pass pays for its registers on every slot (W4 +1.8 and +5 ms per pass). The other form named there:
leave the slots per weight pass at 4 (or fewer) and spread an expert's slots over several threadgroup rows of the same
dispatch, which run side by side instead of one after the other.

## Plan (written before the runs)

**Change under test.** `TONA_METAL_MVG_SPLIT=S` (probe switch, 0 = off = today): `moe_group` gives an expert with c
slots ceil(c / S) entries in `glist`, each a threadgroup row of `mvg` that takes at most S slots; S = 1 or 2 use an
instance with at most 2 slots per weight pass (`mvg2`, fewer registers), S = 3 or 4 the 4-slot one. Each slot keeps
its arithmetic (`rows_dot_at` keeps the slots apart), so every S and width must stay bit-identical: check 2 with S =
0, 1, 2, 3, 4.

**Prior art.** llama.cpp `mul_mv_id` (ggml-metal, local checkout d9ee007, `ggml-metal-ops.cpp` the `else` branch of
`ggml_metal_op_mul_mat_id`) gives every (token, expert) slot its own threadgroup (grid z = experts per token ×
tokens) and leaves the reuse of an expert's weights to the cache; that is S = 1 here (with a 2-slot instance). Our
grouped kernel ([2026-11-12-mac-moe-grouped.md](2026-11-12-mac-moe-grouped.md)) was 13 ms per pass faster at W4 than
one slot per row, so the question is only where between the two ends the optimum lies.

**What can change where.** With 8 experts per token an expert has at most W slots in a pass of W tokens. S = 4
changes nothing at W ≤ 4 and only experts chosen by 5 or more tokens at W5 to W8. S = 2 changes every expert with 3 or
more slots from W3 on: it decodes their weights twice instead of once (the decode is ALU-bound,
[2026-11-12-mac-expert-decode-probes.md](2026-11-12-mac-expert-decode-probes.md)) and gains occupancy (the 4-slot
instance halved it, same file).

**Runs.**

1. Profile (`TONA_METAL_PROFILE=1`, `tona-bench DMT -b W -p 64 -d 64 -r 2 --ctx 4096`), S = 0, 4, 2, 1 at W = 4, 5,
   8: the parts `experts` and `experts_down` per pass. One run per arm, ~4 min. Diagnostic only: decides which S go
   to the speed run.
2. Speed run: `tona-bench DMT -b W -p 512 -d 64 -r 3 --ctx 4096`, W = 2 to 8, the kept arms in the order 0 S S 0 per
   width (with two arms 0 a b b a 0), one build, quiet machine, ~10 to 15 min.

**Expectation.**

- S = 4: W2 to W4 ±0.5 ms (same work, one more loop in `moe_group`); W5 to W8 0 to −3 ms per pass (only the experts
  with 5 or more slots; a draft's tokens share experts, the MTP head's routing shared 22 to 39 % of slots at W4 to
  W8).
- S = 2: W2 −0 to −1 ms (no 3- or 4-slot experts, fewer registers); W3 to W8 between −3 and +6 ms: twice the decode of
  shared experts against the higher occupancy. A loss is the likely outcome; a gain would mean occupancy dominates.
- S = 1: W4 +5 to +13 ms (towards the ungrouped kernel); control only.

A result outside these ranges is a finding.

**Active tooling.** The switch; profile mode only in run 1; `TONA_METAL_MMA`, `TONA_METAL_MMA_WIDE`, `TONA_METAL_MV5`
at their defaults (on). No `TONA_METAL_SOURCES`.

## Run 1: profile

Conditions: before warning (other programs 12.1 % of the CPU threads, load average 3.15 after the lint run before),
after warning (5.4 %); hw_report and raw output in [2026-11-14-mac-mvg-split.log](2026-11-14-mac-mvg-split.log).
Routed experts (`experts` + `experts_down`) per pass, ms (51, 39 and 27 passes at W4, W5, W8), and the difference
to S = 0:

| W | S = 0 | S = 4 | S = 2 | S = 1 |
|---|---|---|---|---|
| 4 | 46.4 | −0.2 | +0.5 | +9.4 |
| 5 | 58.3 | −0.6 | +0.3 | +11.1 |
| 8 | 82.7 | −1.4 | +1.2 | +19.9 |

All inside the expectation. S = 2 makes `experts_down` (640-wide input) 0.3 to 0.4 ms per pass cheaper and gate/up
(2560-wide) 0.8 to 1.5 ms dearer: on the long rows the second decode costs more than the occupancy gains. S = 1, one
slot per row as llama.cpp's `mul_mv_id`, costs 9 to 20 ms per pass: the grouping stays. Only S = 4 goes to the speed
run. (S = 2 for the down matrix alone would need a second slot list for 0.3 ms per pass; not probed.)

## Run 2 plan: speed, S = 0 against S = 4

`tona-bench DMT -b W -p 512 -d 64 -r 3 --ctx 4096`, W = 4 to 8, order 0 4 4 0 per width, 20 runs, ~25 min, no build
in between. Expectation from run 1 (profile mode spreads the passes, the real pass shows less): W4 ±0.5 ms, W5 −0.2
to −1.0, W6 and W7 −0.3 to −1.5, W8 −0.5 to −2.0.

## Run 2: results

Conditions: before warning (other programs 9.5 % of the CPU threads), after: "do not measure" 60 s after the last
run (the GPU reading lags our own runs), ok 20 s later. Raw output in the log. Pass median (ms) of three rounds per
run, order 0 4 4 0:

| W | S = 0 | S = 4 | 4 − 0 | Expected |
|---|---|---|---|---|
| 4 | 84.02, 84.13 | 83.22, 83.13 | **−0.9** | ±0.5 |
| 5 | 98.80, 99.06 | 97.46, 98.03 | −1.2 | −0.2 to −1.0 |
| 6 | 113.58, 113.78 | 112.90, 112.48 | −1.0 | −0.3 to −1.5 |
| 7 | 127.06, 127.07 | 125.79, 126.37 | −1.0 | −0.3 to −1.5 |
| 8 | 138.42, 138.12 | 136.83, 136.41 | −1.65 | −0.5 to −2.0 |

**Finding: W4 is 0.9 ms faster** although no expert has more than 4 slots in a pass of 4 tokens, so the same slots
run in the same rows; the profile showed −0.2 ms there. Open; W5 lies just past its range by the same amount. Both
arms ran the same build in alternation, so it is not drift.

## Server A/B plan

`scripts/ab.py --workload set`, DM-T + MTP-164, same build, A defaults, B `TONA_METAL_MVG_SPLIT=4`, both arms with
the same pass-cost tables (the server's width choice stays the same; only the kernel differs), 2 to 6 rounds.
Expectation: passes of 4 to 8 tokens 1.0 to 1.2 % cheaper (run 2), widths 1 to 3 unchanged; at ~3.1 tokens per pass:
decode code +0 to +1.5 %, text −0.5 to +1 %.
Result: [2026-11-14-mac-mvg-split-server-ab.md](2026-11-14-mac-mvg-split-server-ab.md).

## Server A/B result

6 rounds (59 min), conditions: before ok (11:41, after the speed run's GPU reading cleared); after "GPU 96 %" right at
the end (the reading lags our own runs), then warning (10.6 % other CPU, load average 2.7); in the log.

| Metric | A median | B median | B vs A, mean [CI] | Expected | Decision |
|---|---|---|---|---|---|
| decode code | 37.9 | 36.9 | −1.36 % [−4.82, +2.10] | +0 to +1.5 % | open |
| decode text | 26.2 | 25.4 | −2.50 % [−7.70, +2.71] | −0.5 to +1 % | open |

**Finding: an order effect larger than the change.** In every AB round B is ahead (code +2.2, +0.5, +1.4 %), in
every BA round behind (−2.9, −4.5, −4.9 %): the arm that starts second gains about 3 % in either case. The mean over
pairs cancels it and leaves code −0.4, −2.0, −1.8 % per AB/BA pair, text −2.4, −0.5, −4.6 %: below the expectation,
but with a spread four times the effect the pass costs predict (about +0.5 % at ~3.1 tokens per pass). The server
neither confirms the 1 % per pass of run 2 nor shows a loss with confidence.

**State:** the switch stays off by default (`TONA_METAL_MVG_SPLIT=0`, today's kernel; every value bit-identical).
Open: the cause of the order effect (it is larger than any kernel probe of this series can show in the server).
