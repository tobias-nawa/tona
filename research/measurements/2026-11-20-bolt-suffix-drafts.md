# 2026-11-20: Bolt drafts from the history (suffix drafts) on code (#175)

Branch `worktree-<worktree>`, base `7415cd8`. Logs and scripts: `/var/tmp/claude/sfx175/`. Builds
`work/build/sfxA` (main, arm A) and `work/build/sfxB` (the change, arm B). Every GPU job goes through
`scripts/gpu_queue.py --kind exclusive`.

## Why 1.00 tokens per pass (evidence)

`Generator::draft` (`engine/server/draft.cpp`) returned right after `spec_off || !has_drafter()` and so before the
history draft (`sfx_[s.slot].draft(...)`, `TONA_SUFFIX_MIN` default 8) was even asked. `has_drafter()` is
`dfl_ || mg_.has_mtp()` (`generator.h`): Bolt has no MTP head and no DFlash drafter, so it never drafted, whatever the
text. The comment named "a model without a drafter (Ornith until its MTP head is wired)": the exit was meant for the
time before Ornith had an MTP head and was never narrowed to "no drafter *and* no history". The history draft itself
needs no drafter state (`draft_from_history` only calls `catch_up_drafter`, which is a no-op without a drafter), and
the verify pass is the same one the Coder and Ornith use (`forward_and_verify`, `post`, `mg_.commit(a + 1)`; for the
Mamba-2 layers the replay commit of B4, `2026-11-19-bolt-b4.md`, widths 1 to 8). So no rollback work is missing: the
server log line "0 of M passes from the history" of the earlier prose requests (`bolt-opt2`) is this early exit, not
a miss rate.

## Change

`draft.cpp`: the early exit now covers only `TONA_SPEC_OFF`; a model without a drafter still asks the history, and
returns a block of `last` alone when the history has no match (`!has_drafter()` after the history branch). The thinking
close (`forcing`) stays off for such a model as before.

## Test conditions

Written at each job start into `/var/tmp/claude/sfx175/hw-*.txt` and `quiet-before-*.txt` / `quiet-after-*.txt`
(`tools/hw_report.py`, `tools/quiet_check.py`); results are quoted below.

## Plan

### Question and decision

Does enabling the history drafts speed up Bolt's decode on code, and does it change any token? Decision: if decode is
equal or faster at every context point and the SHAs under `TONA_DETERMINISTIC=1` equal main's, the change goes in
(BENCHMARKS.md row); a decode loss on prose or at some point means a width rule (draft only when the history match is
long) before merging.

### Job 1: identity (`job-id.sh`, one exclusive job)

`scripts/ab.py --a sfxA --b sfxB --env TONA_DETERMINISTIC=1`, Bolt 24 GB file, `--drafter none`, workload `server`
targets `2000:200,16000:200` plus `prose` (2K, 16K prose, 400 tokens), 2 rounds (the ab.py minimum; SHAs are a yes/no, the second round also shows A against A;
ab.py reports A/B SHAs per request). Deterministic mode caps the width at 4. Expectation: every SHA of B equals the SHA
of A. The A start itself may differ from another A start (Bolt's tokens depend on the cache state,
`2026-11-20-kv-hot-head.md`): if B differs, a second round with A against A tells whether the cause is the change.
Duration about 25 min.

### Job 2: speed (`job-speed.sh`, one exclusive job)

`scripts/ab.py --a sfxA --b sfxB --drafter none --workload server --targets 2000:200,16000:200,64000:100 --ctx 262144`
and then `--workload prose` (2K and 16K, 400 tokens), default environment, 2 rounds minimum, noise.toml decides the
rest (at most 4). Size: the three context points of the README row plus the prose pair (where the history match is
rare: B must not lose). Duration about 40 to 60 min.

Expectation (source: `2026-11-19-bolt-opt2.md` D1: pass(2) 1.26x, pass(4) 1.51x, pass(8) 2.25x of pass(1) = 24.35 ms;
`2026-11-19-ornith-opt1.md`: history passes lift Ornith tokens per pass by 25 to 30 % on code): the summary prompts
quote the code in bullet lists, so history matches of 8 tokens occur in some passes. A drafted pass of width 4 pays when
it yields more than 1.51 tokens.

| Rate | Expected B against A | Tolerance |
|---|---|---|
| decode, code 2K / 16K / 64K | +3 % | -1 to +10 % (0 to +10 % of the audit) |
| decode, prose 2K / 16K | 0 % | -1 to +2 % |
| tokens per pass, code | 1.00 to 1.20 | (A: 1.00 exactly) |
| prefill, all | 0 % | -1 to +1 % (no prefill path touched) |

A result outside the tolerance is a finding, read against this plan.

### Active tooling

Defaults only, except `TONA_DETERMINISTIC=1` in job 1. No profiler. `--load-log` off. Exit 75 of the desktop guard:
requeue and note the lost round here.

### Plan addition after job 1 (written before job 2 started)

Job 1 showed 1.00 tokens per pass on the code summaries at 2K and 16K (no history match of 8 tokens in bullet-list
summaries) and 1.01 to 1.02 on prose. The `server` workload asks only for summaries, so job 2 gets a third run, the
`short` workload (summary plus "write a unit test" at 2K, 400 tokens, ctx 140000), where the answer quotes the code.
Expectation: tokens per pass 1.05 to 1.3 on the write request, decode +2 percent (-1 to +10 percent).

## Results

Test conditions: `/var/tmp/claude/sfx175/hw-*.txt` (Ryzen 9 9950X, RX 9060 XT, 4.57 GiB VRAM held by others, 54 GiB RAM
available, load 0.5, commit `7415cd8` plus the change). `quiet_check.py` before and after job 2 (14:21, 15:16): "do not
measure" only because the desktop holds 4.6 GiB VRAM; CPU and load quiet. Jobs: identity 13:57 to 14:20 (2 rounds per
workload), speed 14:20 to 15:16 (code 4 rounds, prose 4 rounds, short 3 rounds). No desktop-guard exit.

### Identity (`TONA_DETERMINISTIC=1`, width cap 4)

Output SHAs of B equal those of A for every request, in both rounds and stable: code summaries 2K / 16K, prose 2K / 16K.
In the default mode (job 2, widths up to 8) code SHAs at 2K / 16K / 64K are equal as well, the prose 16K SHA differs (A `4094a16a`, B
`e46208bf`) and 2K prose A was unstable between rounds: widths of 6 and more round differently (`k_wide`, see
`generator_env.h`), the same property as the Coder; the `short` workload SHAs (summary, write) are equal.

### Speed (B = history drafts, A = main; decode and prefill named; tokens per pass is of the whole request)

| Request | decode A | decode B | B against A [CI] | tokens per pass B (A 1.00) | passes from the history |
|---|---|---|---|---|---|
| code summary 2K | 39.5 | 39.5 | +0.00 % [-0.69, +0.69] | 1.00 | 0 of 199 |
| code summary 16K | 37.9 | 37.9 | -0.07 % [-0.75, +0.62] | 1.00 | 0 of 199 |
| code summary 64K | 35.8 | 35.4 | **-1.11 % [-1.80, -0.42]** | 1.00 | 1 of 99 (one rejected draft) |
| code write-a-test 2K (195 tokens) | 40.2 | 41.9 | **+4.38 % [+3.33, +5.44]** | 1.06 | 5 of 184 (3.2 per pass) |
| code summary 2K (short, 347 tokens) | 39.4 | 39.3 | -0.09 % [-0.49, +0.32] | 1.00 | 0 of 347 |
| prose 2K | 39.7 | 39.9 | +0.38 % [-0.40, +1.15] | 1.01 | 3 of 396 |
| prose 16K | 38.6 | 39.2 | **+1.35 % [+0.66, +2.04]** | 1.02 | 5 of 391 |

Prefill (all requests): -0.19 to +0.97 %, every CI contains 0 (no prefill path touched). Against the plan: all within
the expectation except decode at 64K (-1.11 %, expected -1 to +10 %: the one history pass of 99 was rejected, a
width-4 pass costs about 1.5 x of a width-1 pass, about 0.5 % of 99 passes, the rest is the extra expert reads).

### Reading

- The reason for 1.00 was the early exit, not the text: with it removed the history drafts fire, exactly as for the
  Coder, and verification is exact. The firing rate is low on Bolt's outputs: summaries of code are bullet lists that
  do not repeat 8-token runs (0 of 199 to 347 passes); a request that quotes the code (write a test) drafts in 5 of 184
  passes at 3.2 tokens per pass and gains +4.4 % decode.
- Where drafts do fire the gain is the tokens saved minus the 1.26 to 1.51 x pass cost of width 2 to 4 (`bolt-opt2` D1):
  a rejected draft costs, an accepted one of 3 gains. At the observed rates the net is -1 to +4 %. The audit's "0 to
  +10 %" holds for copy-heavy code (refactors, tests, edits), not for summaries.
- Open: the hit length of 8 (`TONA_SUFFIX_MIN`) was tuned for the Coder/Ornith, whose passes are cheaper to widen
  (MTP); Bolt's width cost is higher, so a longer minimum (rejections cost more) or a shorter one (more accepts) could be
  better; not measured. The real gain needs the DFlash/LiLiCorr drafter (B8).

### Gates

`just check` (incl. clang-tidy) green; ctest unit 71 of 71; `just test-py` 681 passed; Coder goldens (csv, rust, explain), prefill dflash golden and server goldens identical; test_gpu_fattn, test_gpu_attn_wmma, test_gpu_mamba, test_gpu_moe_latent, test_gpu_bolt_quant, test_prefill_order ok. `just test-cpu` was not run: its build directory `work/build/cpu` belongs to another worktree (shared-machine rule), the unit label ran in this worktree's build instead.
