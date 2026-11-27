# Golden checks: one model load for several prompts (#90)

Date: 2026-11-02. Branch `90-golden-loads` (from main 4d4e52f). Build `work/build/wgold` (this branch).

## Plan

Scope: no speed or quality claim about the engine, so no tok/s rates. The question is about the golden checks
themselves: do they check the same thing after the change, and how long do they take? All GPU runs under
`/var/tmp/claude/gpu.lock`, 60 s between blocks, no build while a block runs, one block at a time.

1. **Same result.** Every golden check against the existing references in `work/golden/`, once with the old
   `golden.sh` (`git show main:scripts/golden.sh`, one process per prompt) and once with the new one (one process per
   variant, `ie-gpu-gen --prompts-file`), both with the binary of this branch (its single-prompt path is unchanged).
   The references are not re-recorded: `sha256sum` of every file in `work/golden/` before and after must be equal.
   Checks: golden gpu (3 prompts), golden cpu plan (3), golden prefill dflash (1), golden prefill (1), golden server
   (20 cases; it does not use `ie-gpu-gen`, so it runs once as the fifth check, not old against new).
2. **Warm expert cache.** The second and third prompt of a process run with an expert cache that the first one
   filled. Expectation: tokens identical to the references (with `IE_DETERMINISTIC=1` the cache holds the same bytes,
   it only decides where a matvec reads them). Shown by the new `golden.sh` itself (prompt 2 and 3 of the process
   compare equal) and by a second order of the prompts (explain, rust, csv: another cache history) for the gpu and
   the cpu plan. A difference would be a finding to report, not a reason to re-record.
3. **Duration per check, old and new.** One run each; the checks are deterministic to within seconds. When old and
   new differ by less than 10 s, a second pair of runs.
4. **Golden prefill (362 s), investigation only.** One run of the 13K-token prompt with `IE_MEGA_PROF=1` and the
   stderr log of `ie-gpu-gen` (load time, prompt time, per-token time, step profile). Not part of the comparison
   in 3 because the profile changes the timing. No change to the check.
5. **Order in `gpu-status.sh`**: tested without GPU with stub scripts (`tools/tests/test_select_checks.py`).

Expectation (from `2026-11-02-select-checks.md`, medians of 40 to 80 runs): old gpu 67 s, cpu plan 65 s, prefill
dflash 30 s, prefill 362 s, server 107 s. The load takes 18 s per process, so three prompts in one process save two
loads: new gpu and cpu plan about 67 - 36 = 31 s each (tolerance 10 s, the resets and the freopen cost nothing
measurable); prefill dflash and prefill unchanged within 5 s (one prompt, one load either way). Source: 18 s per load
from the ticket and the `loaded in` lines of the logs.

Active tooling: `IE_DIAG=1` (allocation trace and GPU dump, `toolchain/env.sh`, as always on this machine; costs no
rate outside the spread, measured in `2026-11-01-diag-always.md`). `IE_DETERMINISTIC=1`, `IE_PLAN`,
`IE_EXPERT_PROFILE` and `OMP_NUM_THREADS=6` are set by `golden.sh`. `IE_MEGA_PROF=1` only in block 4.

## Results

All runs on the reference machine, idle GPU under the lock, `IE_DIAG=1`. Times are wall clock of the whole script
(model load included), to the second; first value of a pair from the first block, second from the repeat where one
was made.

| Check | Old s (process per prompt) | New s (one process) | Expected new | Result against `work/golden/` |
|---|---|---|---|---|
| golden gpu (3 prompts) | 68.2 | 28.9, 28.7 | 31 +- 10 | 3 of 3 identical, old and new |
| golden cpu plan (3 prompts) | 68.0 | 28.3, 28.2 | 31 +- 10 | 3 of 3 identical, old and new |
| golden prefill dflash (1 prompt) | 27.9, 27.5 | 27.6, 27.3 | unchanged +- 5 | identical |
| golden prefill (1 prompt) | 359.6 | 359.1 | unchanged +- 5 | identical |
| golden server (20 cases; no `ie-gpu-gen`) | - | 105.4 | 107 | 20 of 20 identical |
| all five | 631 (expected) | 548 | | |

The prefill pair differs by 0.5 s and the dflash pairs by 0.3 s and 0.2 s; one prompt means one load in both ways, so no
difference is expected. A second pair of the 360 s check was not run: the spread of 45 earlier runs was 355 to 370 s
(`2026-11-02-select-checks.md`), twenty times the difference. Server: the script and binary path are unchanged, one
run as the fifth check.

Reading against the plan: the gpu and cpu plan checks are 2 s faster than expected (28.9 against 31 s): the load is
closer to 17 s than 18 s (`loaded in 16.9 s` in the profile run). Everything else is inside the tolerance.

**Warm expert cache.** Prompts 2 and 3 of each process run with an expert cache that prompt 1 filled, and the
tokens are byte-identical to the references of a fresh process. A second order (explain, rust, csv: another cache
history) for the gpu and the cpu plan is identical as well: 6 of 6. Tokens do not depend on the cache contents; no
reference was re-recorded. `sha256sum` of every file that was in `work/golden/` before the runs is unchanged
(`mini/`, new from #91, was added meanwhile by another worker).

Not covered: the state reset is checked through the golden prompts only (fresh state against the references). The
GPU drafter ring (`dring_bufs_`, `mega_build_draft.cpp:91`) is not cleared by `Mega::reset()` (`mega_state.cpp:190`)
and needs no reset: the drafter attention reads only the positions `[max(0, p - window + 1), pos + nb)` of the current
sequence (`mega_op_dattn.h:14-19`, `:220`), and each of them is written in this prompt before it is read (the prompt
feed injects every position). Stale slots of an earlier prompt are never read. The golden runs agree. `--prompts-file`
with `--check` or `--teacher` is refused in `parse_options` (their CPU reference and teacher state are not reset).

## Golden prefill, where the time goes (investigation, nothing changed)

One run with `IE_MEGA_PROF=1` (profile changes timing slightly; the unprofiled check takes 359 s):

| Part | Time | Share |
|---|---|---|
| model load | 16.9 s | 5 % |
| 13,328 prompt tokens, one per megakernel pass | 338.4 s = 25.4 ms per pass (profiled mean 24.9 ms) | 94 % |
| 64 decode tokens | 1.7 s = 37.6 tok/s decode (26.6 ms per token) | 0.5 % |

The prompt goes token by token through the decode kernel because that is what the check is for (decode at depth). A
pass is dominated by the same steps as in plain decode: `dn.in` 16 %, `moe.gate_up` 15 %, `moe.down` 10 %, `dn.out` 8 %,
dense MLP (`gr.up`, `gr.down`) 14 %, `head.out` 7 %; attention (`qsa.*`) 9 % even at 13K, `host.wait` 3 %. The cost is
the number of passes times a pass of about 25 ms, nothing else.

Two ways to shorten it, each with what it stops checking (not implemented):

1. **Prompt in verify passes of width 4** (the path of `--dflash` with `IE_PROMPT_NB=4`; a width-4 pass takes 37 ms
   against 4 x 22.9 ms, `kSeedPassMs` in `ie_gpu_gen_spec.cpp`): about 3,300 passes of about 37 ms = 125 s, check about
   145 s instead of 359 s. Not checked afterwards: the width-1 kernel path for the 13K prompt positions (KV and DeltaNet
   state come from the width-4 program; the 64 decoded tokens still run width 1 on that state). Whether width-4 and
   width-1 write bit-identical KV is not measured here (`--batch-check` exists for that); if not, the reference must be
   re-recorded once, which needs your decision.
2. **Shorter prompt** (for example 6.5K tokens instead of 13.3K): 170 s. Not checked afterwards: the second half of the
   context range, 6.5K to 13K, where IndexCache and top-k selection run at their deepest; the #87 number change showed at
   13K and it is not known whether it would at 6.5K. Needs a new reference and a test that the old defect still shows.

Recommendation: way 1 if width 4 turns out bit-identical; the full 359 s check stays in the scale run and before a release.

## Order in gpu-status.sh

`prefill dflash`, `gpu`, `cpu plan`, `server`, `prefill` (cheapest first by the new times); the first red check ends
the block and the status description names the checks that did not run; `--keep-going` runs all selected ones.
Tested without GPU in `tools/tests/test_select_checks.py` (order, stop, keep going).
