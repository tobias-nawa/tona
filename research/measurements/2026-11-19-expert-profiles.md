# Expert profiles per model: Bolt and Ornith (#175)

Date: 2026-11-19. Build `work/build/ep175` (HEAD `5034f45` plus the changes of this branch, all in `scripts/`, `tools/`,
`toolchain/profiles/`; no engine code changed). Logs: `/var/tmp/claude/ep175/`.

## Test conditions

`python3 tools/hw_report.py` before the first run (22:48, `/var/tmp/claude/ep175/hw-before.txt`): Ryzen 9 9950X, RX 9060 XT
gfx1200 15.9 GiB, 1.84 GiB VRAM held by the desktop, 60.5 GiB RAM (49.9 GiB available), swap 2.79 GiB in use, commit
`5034f45`. `tools/quiet_check.py` before: **warning** (12.7 % of the CPU threads in use by other programs, 1.84 GiB VRAM held
by others); a second worker's builds run in the `cpu` queue beside our jobs. Our speed jobs are `exclusive`, every
comparison is interleaved (`ab.py`), so a shared CPU load reaches both arms. After-run checks are appended per block.

Output of `tools/hw_report.py` before the first run:

### Test conditions

- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads
- GPU: AMD Ryzen 9 9950X 16-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.84 GiB held by others before the run, 0 % busy before the run
- RAM: 60.5 GiB, 51.9 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.3.4-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 6.84, 5.83, 4.49
- CPU use by other programs over 0.5 s: 12.9 % of all threads, swap in use 2.09 GiB
- sandbox: no
- Commit: `5034f45` (dirty tree)

`tools/quiet_check.py` before:

- Quiet check (2026-11-19 22:48:54): warning
  - warning: other programs use 12.7 % of all CPU threads
  - warning: 1.84 GiB of 15.9 GiB VRAM are held by others

## What exists, what this change adds

The mechanism is complete and model-independent: `TONA_EXPERT_PROFILE` (`Mega::load_profile`, Metal arena) seeds the VRAM
expert cache and ranks the pinned experts (`backend/expert_tiers.h`); the file checks its own layout against the model and
refuses a profile of another model. The installer already ships `toolchain/profiles/<model id>.bin` (embedded into the
signed `tona-install`, `profile_step`, written to `<root>/profiles/<id>.bin` and named in the unit's `TONA_EXPERT_PROFILE`;
in a checkout copied to `work/profile/<id>.bin`, `installer/core/steps.cpp`). The model id is the catalog id of
`toolchain/models.toml` and the folder name under `work/models/`. What was Coder-only:

1. `scripts/ab_server.py` applied `work/profile/experts-coder.bin` when `"coder"` occurred anywhere in the model path.
   Now: `toolchain/profiles/<id>.bin` for the id = folder name of the GGUF, exact match (`profile_of`); an arm can drop it
   with `--a-env TONA_EXPERT_PROFILE=none`. The Coder's id `coder-iq1_m` maps to the same bytes as before
   (`cmp work/profile/experts-coder.bin toolchain/profiles/coder-iq1_m.bin`: identical).
2. No tool made profiles for other models. `scripts/make_expert_profile.py` (new): `tona-tokenize` + `tona-bolt-bench`
   (any model of the generic backend interface) on a code slice and a prose slice, counts accumulated with
   `TONA_EXPERT_PROFILE_OUT`.

## Plan (written before any run)

**Questions and decisions.**

| Question | Outcomes | Decision |
|---|---|---|
| Does a profile of a mixed code and prose slice speed up Bolt (prose 2K/16K, code 2K/16K)? | Prefill +1 % or more or decode +0.5 % or more somewhere and nowhere worse than -0.5 %: ship. Worse than -0.5 % in a rate: report as a cost (small losses count), ship only with the gain that pays for it, maintainer decides | `ab.py` A = no profile, B = profile |
| The same for Ornith (server workload 2K and 16K) | as above | as above |
| Does the Coder profile still apply exactly as before? | `ab_server.child_env` gives the same `TONA_EXPERT_PROFILE` path content; goldens identical; A/A flat | goldens, one short `ab.py` A/A with the new lookup |
| Do outputs stay equal? | SHA of every request equal in A and B (a profile changes placement, not numbers); a difference is a bug | stops the block |

**Training data (no train-on-test).** `scripts/make_expert_profile.py`, `--offset 1500000`: 20,000 prompt tokens plus 300
decode tokens from each of the code corpus and the prose corpus, from character 1,500,000 on. The A/B prompts read: `prose`
the start of the prose corpus (16,000 tokens at most, below 100,000 characters), `longdoc` from 90,000 on (not run here),
`sum` the start of the code corpus, `write` from 1,000,000 on (2,000 or 16,000 tokens, below 1,100,000 characters). The slices
are disjoint from all of them; the tool refuses a smaller offset. (Bolt's E2 profile used 540,000 on; this one is a fresh
build, not that file.)

**Runs (queue).**

1. Build `ep175` (`cpu`). Profiles for Bolt (`nemotron-bolt-74b-a9b-24gb`) and Ornith (`ornith-1.5-35b-a3b`): two `gpu` jobs,
   each about 40,000 prefilled tokens plus 600 decode tokens, a few minutes each. Check with `tools/profile_compare.py`
   (concentration, cross coverage code and prose).
2. Bolt prose: `ab.py --a ep175 --b ep175`, arm A with `--a-env TONA_EXPERT_PROFILE=none`, workload `prose` (2K and 16K
   prompts, 400 decode tokens, ctx 32768), `--drafter none`, 4 rounds AB/BA (no noise entry for `prose`; the Bolt B0 run saw
   spread below 0.25 %, `2026-11-19-bolt-opt2.md`).
3. Bolt code: workload `server`, targets `2000:300,16000:300`, ctx 32768, same arms, 4 rounds.
4. Ornith: workload `server`, targets `2000:300,16000:100`, ctx 32768, `--drafter none` (MTP head in the file), 4 rounds. The
   decode after a 16K prompt is the one rate that Ornith's ctx pressure moves most; 100 tokens suffice for a rate.
5. Coder: `golden.sh compare`, `golden-server.sh compare` on `ep175`, then `ab.py` A/A (`--a ep175 --b ep175`, workload `short`,
   2 rounds, `--max-rounds 2`): both arms get the profile by the new lookup; flat within the known spread
   (`bench/noise.toml`, `decode@2000` 0.23 %, `prefill@2000` 0.09 %).

Why these sizes: the profile acts on the first requests of a server (VRAM cache start, pinned set) and on the steady hit
rate; 2K shows the start, 16K the steady state and the prefill with the experts streaming. 64K adds attention, not experts.

**Expectation (with tolerance, source).**

- Bolt prose: prefill@2000 +5.8 % (+/- 3 points), decode@2000 -0.3 % (+/- 0.7), prefill@16000 +0.1 % (+/- 2), decode@16000
  +3.3 % (+/- 1.5). Source: E3 of `2026-11-19-bolt-opt2.md`, a profile of 20,000 tokens of one text kind. A profile from
  both kinds should be as good or better on code: code prefill@2000 +6 % (+/- 3), decode +1.1 % (+/- 1.5). Code at 16K: not
  measured before, expected like prose at 16K.
- Ornith: no data. The VRAM holds about half of the 20.9 GB of routed experts (`2026-11-19-ornith-opt1.md`), so the start
  by usage matters more than for Bolt (whose VRAM cache holds 145 of up to 496): decode +3 to +10 %, prefill@2000 +2 to +8 %,
  prefill@16000 0 to +4 %. Source: the Bolt result for the direction, the Coder's tier measurements
  (`docs/architecture/expert-tiers.md`) for the size; a result outside is a finding.
- Coder A/A: flat, all rates within 0.5 %.
- SHA equal everywhere.

**Active tooling.** None in the speed jobs: only `TONA_EXPERT_PROFILE` in arm B, `TONA_EXPERT_PROFILE=none` marks arm A
(removed from its environment by `child_env`; the report's tooling line shows it). The profile jobs set only
`TONA_EXPERT_PROFILE_OUT`/`TONA_EXPERT_PROFILE`.

**Stop rules.** Exit 75 (desktop guard): requeue, note the lost round. A GPU fault or an out-of-memory stops everything for
the coordinator. A SHA difference stops the block.

## Plan addendum: second Ornith run and the Coder guards (written after the first three jobs, before these runs)

The first Ornith run (4 rounds) left three metrics outside the expectation and prefill@2000 with a large spread (B against A
+9 to +41 % per round, the A values 2290 to 2531, the B values 2757 to 3220 tok/s), while the expert counters of A and B are
nearly the same (hits 83,699 against 83,647 at 2K). A rate that moves without the counters moving is not explained by the
profile's hit rate; the second run repeats it with 6 rounds (`ep175-ornith2`, same targets and arms, `--fixed-rounds`)
to see whether the prefill@2000 gain and the decode@16000 loss hold. Expectation: the same direction as run 1 (prefill@2000
+10 to +40 %, decode@2000 +1 to +3 %, prefill@16000 +-0.5 %, decode@16000 -1.2 to -0.3 %); a different sign is a finding about
noise, not about the profile. Duration about 4 minutes. Coder: goldens (`golden.sh compare`, `golden-server.sh compare`, build
`ep175`), then `ab.py` A/A `--workload short --max-rounds 2 --min-rounds 2 --fixed-rounds`, both arms the default lookup
(expect all rates within +-0.5 %, SHA equal). Active tooling: as before.

## Results

Reports (the `ab.py` output, one file per run): `2026-11-19-expert-profiles-bolt-prose.{md,json}`, `-bolt-code`, `-ornith`
(run 1), `-ornith2` (run 2), `-coder-aa`. Test conditions after the runs (23:20): quiet check **warning**, 2.64 GiB VRAM
held by others (3.3 GiB during the runs, 1.8 GiB before); `/var/tmp/claude/ep175/hw-after.txt`. All jobs `exclusive` or `gpu`
through the queue; no server on port 8662.

### The profiles (job `ep175-prof-bolt`, `ep175-prof-ornith`; `scripts/make_expert_profile.py`, offset 1,500,000, 20,000 + 300 tokens per kind)

| Model id | File | Uses counted | Concentration (`tools/profile_compare.py`) |
|---|---|---|---|
| `nemotron-bolt-74b-a9b-24gb` | `toolchain/profiles/nemotron-bolt-74b-a9b-24gb.bin` (55,472 bytes) | 14,615,280 | top-145 per layer 98.9 % of uses (min layer 70.6 %), top-345 100.0 % |
| `ornith-1.5-35b-a3b` | `toolchain/profiles/ornith-1.5-35b-a3b.bin` (40,972 bytes) | 12,991,360 | top-128 of 256 per layer 81.3 % (min layer 64.5 %), top-192 94.3 % |

Bolt matches the E2 numbers (98.9 % against 98.8 to 99.2 %). Ornith's router is much flatter than Bolt's: half of its experts
carry 81 % of the uses.

### Speed (B = profile, A = `TONA_EXPERT_PROFILE=none`; same build `ep175` in both arms; SHA of every request equal in A and B)

| Model, workload | Rate | A (median) | B (median) | B against A [95 % CI], rounds | Plan |
|---|---|---|---|---|---|
| Bolt prose | prefill@2000 | 1357 tok/s | 1427 | **+5.13 % [+4.68, +5.58]**, 4 | +2.8 to +8.8: in |
| | decode@2000 | 42.1 tok/s | 41.5 | **-1.20 % [-1.72, -0.69]**, 4 | -1.0 to +0.4: **outside** (small loss) |
| | prefill@16000 | 1849 | 1851 | +0.14 % [-0.31, +0.59] (open) | in |
| | decode@16000 | 42.4 | 44.2 | **+4.13 % [+3.67, +4.58]** | +1.8 to +4.8: in |
| Bolt code | prefill@2000 | 1384 | 1453 | **+4.99 % [+4.54, +5.44]**, 4 | in |
| | decode@2000 | 41.8 | 41.7 | -0.38 % [-0.84, +0.07] (open) | in |
| | prefill@16000 | 1855 | 1859 | +0.21 % [-0.24, +0.66] (open) | in |
| | decode@16000 | 42.5 | 44.2 | **+3.92 % [+3.47, +4.37]** | in |
| Ornith, run 2 (6 rounds) | prefill@2000 | 2291 | 2753 | **+20.21 % [+19.87, +20.55]** | +10 to +40: in |
| | decode@2000 | 121.8 | 124.2 | **+1.94 % [+1.61, +2.27]** | in |
| | prefill@16000 | 4089.5 | 4089.5 | -0.02 % [-0.33, +0.29], equal | in |
| | decode@16000 | 148.9 | 148.0 | **-0.59 % [-0.94, -0.24]** | in |

Ornith run 1 (4 rounds, plan: decode +3 to +10 %, prefill@2000 +2 to +8 %, prefill@16000 0 to +4 %) gave prefill@2000 +27.6 %
[+15.5, +39.7] with per-round values from +9 to +41 % (B values 2757 to 3220 against A 2290 to 2531 tok/s), decode@2000 +2.32 %,
decode@16000 -0.78 %. Run 2 repeated it with the same arms and the result is steady (all six rounds +19.9 to +20.5 %), so
run 1's spread was a disturbed run (other programs' load, 3.3 GiB of VRAM held beside us), not an effect of the profile; the
plan addendum's expectation was met. Both runs fall **outside the first plan's expectation** for Ornith (decode +3 to +10 %:
measured +2 % and -0.6 %; prefill@2000 +2 to +8 %: measured +20 %), which was a guess from Bolt, and the finding is: the
profile matters for the first request's prefill (+20 %), not for the decode hit rate.

Expert counters (prefill plus decode of one request): Bolt 2K prose 107,906 hits / 36,094 misses (A) against 86,058 / 57,942
(B), 16K prose 111,821 / 32,179 against 138,972 / 5,028 (B: 84 % fewer misses); code 16K 85,562 / 22,438 against 105,719 /
2,281. As in E3: the profile fills the VRAM slots with the experts of the whole mix, a 2K request sees a different mix, so
its start has more misses (and decode@2000 is 0.4 to 1.2 % lower) while the 16K request has far fewer. Ornith's counters
hardly move (2K: 83,699 / 3,842 against 83,647 / 3,894; 16K: 21,832 / 1,580 against 21,740 / 1,672): its cache already holds
about 96 % of the reads without a profile, and with one it holds the same share; the prefill@2000 gain therefore does not
come from hits and misses of the decode cache (cause not measured: the pinned set by usage in the first request's layered
prefill is the candidate; a `TONA_DIAG` run of the prefill would show it). Not guessed further.

### Coder guard

- `scripts/golden.sh compare work/build/ep175`: all prompts identical (csv, rust, explain); `scripts/golden-server.sh compare
  work/build/ep175`: all identical (`/var/tmp/claude/ep175/golden*.log`).
- `ab.py` A/A, workload `short`, 2 rounds, both arms the default lookup (`toolchain/profiles/coder-iq1_m.bin`, byte-equal to
  `work/profile/experts-coder.bin`): prefill@2000 +0.07 %, decode@2000 +0.27 % (open, 2 rounds), prefill@2000:write -0.02 %,
  decode@2000:write +0.09 %, all inside the plan's +-0.5 %; expert hits and misses identical in A and B per request
  (128,937 / 23,912 and 82,220 / 28,057), SHA equal. The profile is applied as before.
- `test_gpu_mamba`, `test_gpu_moe_latent`, `test_gpu_bolt_quant` on `ep175`: all passed.

### Decision

Ship both profiles. Bolt: prefill of the first request +5 % (prose and code), decode at 16K +4 %, decode at 2K -0.4 to -1.2 %
(the one cost, reported; it comes from the average-mix start, which is the price of one profile for all text). Ornith: first
request prefill@2000 +20 %, decode@2000 +1.9 %, decode@16000 -0.6 % (a small, consistent loss in 6 of 6 rounds, cause not
measured), prefill@16000 equal. Both are net gains on the first request, which is what a user's session starts with. The
maintainer decides the Ornith decode@16000 cost.

### How profiles reach users

`toolchain/profiles/<model id>.bin` is embedded into `tona-install` (`tools/gen_installer_data.py` takes every `*.bin` of the
folder), written to `<root>/profiles/<id>.bin` by the "Expert profile" step and named in the systemd unit
(`TONA_EXPERT_PROFILE`); in a checkout it is copied to `work/profile/<id>.bin`. This path existed for the Coder and works for any
id. What is missing for Bolt and Ornith: `toolchain/models.toml` has no entry for either model yet (not published, not
offered by the installer), so no user reaches the profile until the entry is pinned; the profile file already waits under the
id the folder name gives (`nemotron-bolt-74b-a9b-24gb`, `ornith-1.5-35b-a3b`). If the catalog ids differ from these folder
names, rename the two files to the catalog ids. A profile built for a different GGUF revision is refused at start (layout
check), a profile for the same layout but other weights would only rank wrongly: rebuild with the tool when the file changes.
Alternative not taken: building the profile on the user's first run (needs 40,000 tokens of text and GPU time at install, and
the text slices; shipping 40 to 55 KB is cheaper and reproducible).
