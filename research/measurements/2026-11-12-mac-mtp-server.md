# Mac: server end to end with the MTP drafter (#108, M3)

M1 Max 32 GB, Metal backend, target model DM (the Coder cut to 164 experts per layer, byte copy), MTP head cut to
164 experts by its own routing on coding load (2026-11-11-mac-mtp-route-stats.md).

## Plan (written before the run)

**Question.** What do decode rate and tokens per pass of the production path (server, MTP drafter) reach on the Mac,
against the same server without drafter?

**Scope.**

- Arms: A = `tona-server DM` without `--mtp`; B = `tona-server DM --mtp mtp-164`. Same build, `--ctx 16384`,
  default `TONA_PREFILL_CHUNK` (2048). The context of 16384 instead of the reference's 32768: DM 20.55 GB + head
  0.98 GB + scratch ~1.65 GB + KV int8 at 32K 3.2 GB would leave no room under the 26.8 GB working set; at 16K
  1.6 GB.
- Workload `set` (`scripts/bench-set.py`: 11 code, 3 text prompts, greedy, up to 400 tokens each), the decode of the
  production path.
- Order A B A B (two rounds per arm, alternating), one server start per round, the first request of a start is
  part of the round (no separate warm-up: the set is long enough and the warm-up shows as a finding if it stands
  out per prompt).
- Per round: page-ins (`vm_stat`) and swap use (`vm.swapusage`) before and after, `/status` (passes, tokens per
  pass, MTP passes, stale entries).
- Expected duration: A ~5 min per round (5,600 tokens at ~20 tok/s), B ~3 to 4 min, loads ~10 s: about 20 min.

**Expectation.**

- A: code and text decode 19 to 21.5 tok/s (20.4 at 2K context, 2026-11-11-mac-decode-attn.md; the set's contexts
  are below 1K). Page-ins below 5K per round once the model is in the page cache.
- B: tokens per pass 3.3 to 4.5 on code (3.9 in the dump on DM with the full head, 4.27 at the server smoke with the
  full head; the cut head may lose some), 2 to 3 on text (unmeasured: a guess). Decode on code 24 to 35 tok/s: the
  server smoke with the full head and a chunk of 512 reached 21.5 to 27.1 tok/s at ~170 ms per pass; the smaller head
  and the larger chunk should not change the pass much, so the rate follows tokens per pass. Text lower.
- A result outside these ranges is a finding, especially B on code below 24 (the drafter pass costs more than its
  tokens bring).

**Active tooling.** No profiler, no `TONA_*` variable in the server's environment; `caffeinate -dimsu` keeps the
machine awake (no CPU load). Quiet check and `tools/hw_report.py` before and after.

**Change before the run (2026-11-12 08:15).** The first attempt (2026-11-11 23:22) measured nothing: `tona-server`
needs exactly one drafter (`--mtp` or `--dflash`; `TONA_DRAFT_N` clamps to 1 to 7), so arm A cannot start. New scope:
arm B only, two rounds (B B), same settings. The baseline without drafter is the trunk decode of
2026-11-11-mac-decode-attn.md (20.4 tok/s at 2K context, tona-check, same build line); the comparison is therefore
server against tona-check, not server against server, and B's rate is read against the expectation above, not as a
measured A/B ratio. Expected duration ~8 min.

## Test conditions

Before (08:10): quiet check ok; `tools/hw_report.py`: Apple M1 Max 10 cores, 32 GiB unified memory, macOS 27.0.1, no
`TONA_*` variable, load 1.4 / 1.61 / 1.62, other programs 5.3 % of all CPU threads, commit `a277ebc` (dirty tree: this
file and the backend AGENTS.md, no code). Swap in use 16.97 GB throughout (load from before the run).
After (08:17:59, seconds after the second server exited): "do not measure: the GPU is 96 % busy", the run's own tail;
again at 08:18:13: warning, other programs 9.0 % of all CPU threads. AC power, `caffeinate -dimsu`.

## Results

Raw: [2026-11-12-mac-mtp-server.jsonl](2026-11-12-mac-mtp-server.jsonl) (one line per round).

| Round | Code decode | Text decode | Tokens per pass (all 14) | MTP passes | Page-ins | Seconds |
|---|---|---|---|---|---|---|
| B1 | 23.3 tok/s (per prompt 20 to 35) | 17.4 (16 to 18) | 3.33 (4,656 tokens, 1,397 passes) | 4,551 | 1,488,192 (cold start) | 236 |
| B2 | 23.3 tok/s (20 to 35) | 17.3 (16 to 18) | 3.33 (identical: greedy) | 4,551 | 135,529 | 227 |

Spread between the rounds: 0.0 code, 0.1 text. Baseline without drafter: 20.4 tok/s (tona-check, 2K context).

**Against the plan.**

- Code 23.3 tok/s: **below the expected 24 to 35** (finding), +14 % over the trunk alone.
- Text 17.3 to 17.4 tok/s: **below the trunk alone** (20.4): the drafter costs more than its tokens bring.
- Tokens per pass 3.33 over both kinds (the plan expected 3.3 to 4.5 on code alone; not split per kind here).
- One cycle (verify pass plus 3.26 MTP passes on average) costs 1,000 / 23.3 × ~3.6 ≈ 150 ms on code against 49 ms for
  a decode pass of one token: the verify pass and the head steps, not the acceptance, hold the rate down.
- The width choice of the server (`engine/server/generator.h`, `t_mpass_`, `t_step_`, `t_fill_`) uses the reference's
  costs (verify width 1 to 8: 24.2 to 58.1 ms, MTP step 1.15 ms); on Metal a head step is a full pass of its own
  (command buffer, MoE of one layer). Unchecked: how far off the Metal costs are and how much a width choice from them
  would give. Next: profile of the verify pass per width and of the head step, then the backend's own costs for the
  width choice (2026-11-12, coordinator: backend delivers its pass times, the reference keeps its constants).
- Page-ins after the cold start 136K per round (29 per generated token): the working
  set (24.30 GB) fits; chunk 2048 with the cut head did not overflow.
