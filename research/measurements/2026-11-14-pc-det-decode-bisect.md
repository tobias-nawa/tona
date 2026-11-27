# Bisect of the deterministic decode regression of the scale run (2026-11-14)

## Plan

Written before the first run (the plan block of every run is in `2026-11-14-pc-det-decode-bisect-logs/`).

- Question: which commits slowed deterministic decode after 30,000 tokens of prompt by -5.65 % against b6dd58c
  (finding 1 of `2026-11-14-scale-run.md`, regression run `ab-regr-b6dd`: decode@30000 -5.65 % [-6.99, -4.32],
  decode@130000 -3.73 % [-4.65, -2.80], prefill +2 %, SHAs equal)? AGENTS.md rule 1: the reference never gets worse.
- Scope: step 0: production mode (no `TONA_DETERMINISTIC`), `engine` (42e1beb) against `b6dd`, workload `mid`, 2 rounds.
  Then bisect over the 27 first-parent merges `b6dd58c..d97de1b -- engine/` (numbered 1 to 27 oldest first), each
  step `ab.py --a b6dd --b bis-<sha>`, `TONA_DETERMINISTIC=1`, `TONA_EXPERT_PROFILE=work/profile/experts-coder.bin`,
  `--workload mid --ctx 140000`, 2 rounds, `--load-log on`, one discarded warm-up start of A (about 7 min a step,
  plus about 2 min build). Then ablation and mechanism on the found commits.
- Why this size: `mid` (30000:200) alone shows the finding at the size of the regression run; 2 paired rounds resolve
  about +-0.8 % here (CI half-width of every step below), enough for a -3 % to -6 % effect; `long` doubles the time
  per step and was already measured in the regression run.
- Expectation per step: good commit decode@30000 within -1 to +1 % of A (A/A noise: `bench/noise.toml`, sd 0.28 %);
  bad commit about -5.6 % (the regression run). Source: `ab-regr-b6dd` above. Prefill@30000 +0 to +9 % (#129).
  Production step 0: -2 to +4 % (#125/#126 decode +1.5 to +2 % each).
- Active tooling: `TONA_EXPERT_PROFILE`, `TONA_DETERMINISTIC=1` (not in step 0), load log (`--load-log on`),
  `--no-noise` (see "Tool finding"), no profiler, no `TONA_DIAG`. Server logs of every step have the line
  "tokens per pass" used for the mechanism.

## Test conditions

Before the first run (`tools/hw_report.py`, `tools/quiet_check.py`, 21:34):

    ### Test conditions

    - CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
    - GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
    - GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.65 GiB held by others before the run, 0 % busy before the run
    - RAM: 60.5 GiB, 57.8 GiB available before the run
    - OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
    - Driver and toolkit: hip 7.16.26385-0000000
    - TONA_* variables: none set
    - cgroup limits of this shell: none set
    - Load average (1, 5, 15 min): 0.36, 1.16, 2.13
    - CPU use by other programs over 0.5 s: 0.3 % of all threads, swap in use 0.96 GiB
    - sandbox: no
    - Commit: `42e1beb` (clean tree)
    - Quiet check (2026-11-14 21:34:10): warning
      - warning: 1.65 GiB of 15.9 GiB VRAM are held by others

After the last run (about 22:30; no run or build followed):

    - RAM: 60.5 GiB, 57.5 GiB available before the run
    - Load average (1, 5, 15 min): 0.73, 1.94, 2.65
    - CPU use by other programs over 0.5 s: 0.5 % of all threads, swap in use 0.95 GiB
    - Commit: `42e1beb` (dirty tree)
    - Quiet check (2026-11-14 22:30:31): warning
      - warning: 2.33 GiB of 15.9 GiB VRAM are held by others

(The "dirty tree" is the new log folder of this file; the engine under test is `work/build/b6dd` and the `bis-*` builds.)

## Results

### Step 0: production mode (what users see)

`engine` (42e1beb) against `b6dd`, no `TONA_DETERMINISTIC`, `mid`, 2 rounds (AB, BA), `ab-step0.md`:

| Metric | B vs A | Verdict |
|---|---|---|
| **decode**@30000 | **+2.23 % [+1.42, +3.04]** (70.5 to 72.0 tok/s) | inside the expectation (-2 to +4 %) |
| **prefill**@30000 | +2.83 % [+1.59, +4.07] | inside (-1 to +9 %) |

**Production decode is not slower, it is 2.2 % faster.** Tokens per pass (server log, normal drafting): A 3.45, B 3.37.
The regression exists only with `TONA_DETERMINISTIC=1`, which caps the drafts at 3 (rule 2 of AGENTS.md). A first
run of this step stopped after 1 round (+2.11 % [+1.40, +2.82], `ab-step0-1round`, see "Tool finding"); the numbers
above are the 2-round run.

### Bisect (deterministic, `mid`, decode@30000 B vs A, 2 rounds, A = `b6dd`)

| # | Commit | Merge of | decode@30000 B vs A [CI] | prefill@30000 | tokens per pass B (A 3.02) | Verdict |
|---|---|---|---|---|---|---|
| 14 | dde83c2 | 108-mac-m2 | **-2.77 % [-3.72, -1.81]** | +0.02 % | 2.80 | bad |
| 7 | bb4a439 | 108-mac-m2 | -0.67 % [-1.49, +0.14] | -0.07 % | 2.93 | open (CI holds 0) |
| 10 | 126e0c7 | 125-dynamic-head | **-2.83 % [-3.65, -2.00]** | -0.06 % | 2.80 | bad |
| 9 | abc55e1 | 126-verify-pass | -0.28 % [-1.18, +0.61] | +0.03 % | 2.93 | good |
| 10, branch | 4585043 | the code commit of #125 (parent d50ae5a: no engine change against #9) | **-3.25 % [-4.11, -2.38]** | -0.01 % | 2.80 | **bad: culprit 1** |
| 27 | 84f7d8c | 108-mac-m2 (tip of the range) | **-6.19 % [-7.00, -5.38]** | +1.20 % | 2.80 | bad (the full regression) |
| 21 | 0228b0a | 130-pc-q6k-tile | **-5.89 % [-6.70, -5.08]** | +1.21 % | 2.80 | bad |
| 18 | 321f249 | 129-prefill-bimodal | -2.81 % [-3.63, -2.00] | +1.20 % | 2.80 | bad only by culprit 1 |
| 20 | 519c1ab | 108-mac-m2 | -2.69 % [-3.50, -1.87] | +1.18 % | 2.80 | bad only by culprit 1 |

Reading: **two separate steps**, each about -3 %, add up to the -6 % of the tip (the regression run's -5.65 % was
measured with more load around it, see the load tables in the step files; the tip here gives -6.19 %).

- **Culprit 1: `4585043` (#125, MTP draft head with the low-rank scorer, default on): -0.28 % -> -3.25 %** (steps 9
  and 4585043; the order of the later merges did not matter: 14, 18, 20 stay at -2.7 to -2.8 %).
- **Culprit 2: `3517701` (#130, `TONA_Q6_K_T` decoder on the HIP backend), merge 0228b0a: -2.69 % -> -5.89 %**
  (steps 20 and 21). The merge brings exactly one engine commit (3517701); the other four are installer, scripts and
  docs (`git log --no-merges 519c1ab..0228b0a`), so no bisect inside the branch was needed.
- The prefill gain of +1.2 % appears between 14 and 18 (not examined; the ticket asks for decode).
- Tokens per pass 3.02 -> 2.93 appears by commit 7 and stays until 9; 2.93 -> 2.80 comes with 4585043 and does not
  change afterwards. The step from 3.02 to 2.93 between 1 and 7 has no speed cost at 9 (-0.28 %, open); its cause is
  unchecked (SHAs are identical in all steps).

### Mechanism

Culprit 1, #125 (verified in part):

- Verified: with `TONA_MTP_SCORER=0` on the build of 4585043 (`ab-abl-scorer0.md`, B = same build, A = `b6dd`),
  decode@30000 is **-0.16 % [-0.99, +0.68]**, i.e. the cost is gone; the scorer on is -3.25 %. Tokens per pass with
  the scorer: 2.80 (71 passes for 200 tokens), without: 2.93 (68 passes), `b6dd`: 3.02 (66 passes).
- Verified: tokens are identical (SHA equal), so only the number of passes and the cost of a pass differ. With the
  scorer on, 4585043 needs 71 instead of 68 passes for the same 200 tokens (+4.4 %) while a pass costs the same
  within the noise (the sum -3.25 % against 4.4 % more passes).
- Read in the diff (`git show 4585043`): the head is cut from 41,500 to 16,000 static ids + 4,096 context rows, the
  scorer adds 1,024 gathered rows chosen per step; with fixed drafts of at most 3 (deterministic mode) the cut list
  lowers the acceptance of the drafts (fewer tokens per pass), and the production mode compensates with deeper drafts
  (3.45 -> 3.37 only). That the candidate set, not the extra ops, causes the lower acceptance in deterministic mode
  is **unchecked**: no run of 16,000 ids with the scorer off (`TONA_MTP_VOCAB_N` against the 41,500 list) was made.
- The commit message measured "set" in production mode (+1.51 % code), where it is a gain; the deterministic mode
  was not part of its check.

Culprit 2, #130 (partly verified):

- Verified: tokens per pass are the same before and after (2.80, 71 passes), SHAs equal, so a pass got slower by
  about 3.3 % (69.9 -> 67.6 tok/s at the same passes). The switch `TONA_Q6_K_TILE` of the later part of #130 does
  not exist at this commit and the plain GGUF does not use the tile type 1019 at runtime, so the new decoder is not
  executed in this measurement.
- Verified: compiling `engine/gpu/mega.hip` with `-Rpass-analysis=kernel-resource-usage` (`res-519c1ab.txt`,
  `res-0228b0a.txt` in the log folder): k_mega<8> SGPR spills 147 -> 163, VGPR spills 109 -> 110, k_mega<4> SGPR
  spills 121 -> 129 (VGPR 54 -> 53), k_mega<1> VGPR 62 -> 61, scratch 496 -> 496 / 496 -> 480 / 528 -> 512, VGPRs 192
  everywhere; the object file grows from 2,721,888 to 2,851,056 bytes (+4.7 %, with `-O2`, no `-g`). The commit adds
  the case `kQ6KTile` to `dispatch_wide` and `dispatch_q8` (`matvec_phase.h`), which are instantiated inside k_mega, and
  routes every row pointer through `row_at<T>` / `decode_row<T>`.
- **Unchecked:** that the larger k_mega (more code in the instruction cache, more SGPR spills in the wide and q8 phases)
  is the cause of the 3.3 % slower pass. The numbers show a codegen change in the kernel without a runtime change in
  the data path; a test would be a build of 0228b0a with the two new `case kQ6KTile` removed, or a pass profile
  (`TONA_MTP_PROF`, the pass table of #126) of the two builds.
- The commit's own check ("verify pass W1-4 within +-0.05 ms, production decode +0.90 %") measured the batch bench
  (`tona-wide-probe`, no spills of k_mega in the wide kernel alone) and production mode with the tile file, never the
  deterministic server run on the plain GGUF.

### Tool finding

`scripts/ab.py --fixed-rounds` does nothing: `Stop(primary, noise, expect, round_warmup=...)` at `ab.py:449` is built
without `fixed`, so `stop.fixed` (line 221) stays False and the run ends as soon as the noise table decides the
metrics (the first step-0 run stopped after round 1 although `--min-rounds 2 --max-rounds 2 --fixed-rounds` were
given). All runs above except `ab-step0-1round` use `--no-noise`, which keeps two rounds. Not fixed here (no code
change in this ticket).

## Hand-over for the fix

Both costs are in deterministic mode (the mode the goldens and the measurements of rule 2 use). They are no loss in
production (step 0, +2.23 %). Options for the coordinator (not decided here): keep both and note that deterministic
decode is the wrong yardstick for #125; or give the deterministic mode the scorer off (`TONA_MTP_SCORER=0` is the
documented bit-identical path) and a build of 0228b0a without the tile cases in k_mega to test the codegen cause of #130.
Principle 8: nothing was dropped here, no fix was tried.
