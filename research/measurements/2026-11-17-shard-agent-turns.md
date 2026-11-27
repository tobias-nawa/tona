# Model shard 2026-09-29 against 2026-10-19 on agent turns (teacher-forced), 2026-11-17

Question (coordinator, after #163): the pin of the shard of 2026-10-19 (commit b6dd58c, decision of 2026-11-12) rested on
32K tokens of public code and prose (`2026-11-12-coder-1019.md`), not on agent turns. On one agent turn (224356-0049, 165 tail
tokens) the new shard is 0.21 nat worse (0.609 against 0.395, top-1 79 % against 87 %). Is the new shard worse on agent
turns in general? Decision this feeds: keep the pin, or revert it (a revert changes the reference SHAs, so #164 waits).

## Plan (written before the runs)

- **Scope:** the 25 stored Claude Code turns of `2026-11-10-fp8-quality.md` (21,066 turn tokens), teacher-forced exactly as in
  `2026-11-17-prefill-fp8-experts.md` (`tona-gpu-gen --mega --teacher`, `TONA_TEACHER_TAIL/SPECIAL`, `TONA_DETERMINISTIC=1`),
  today's tree (build `w7`, all #159 switches off = main's behaviour, bit-identical to main, checked), **default path** as users run
  it (dense fp8 on, experts f16).
- **Arms:** A = shard of 2026-09-29 (sha256 e11083ba), read in place from the other engine's data directory, read only, not copied (the file
  is mapped by the engine; shard 2 of both revisions is the same). B = shard of 2026-10-19 (bfa3644d, `work/models`). Arm B
  already exists for all 25 turns (`on` of the fp8-experts check, computed earlier today with the same build and switches, one
  turn bit-identical before and after the reboot to the 9950X), so only arm A runs: 25 runs of about 1 to 2.5 min.
- **Estimator:** per-turn mean of dNLL = B - A (positive = B worse), mean over turns, t-based 95 % interval over turns; top-1
  agreement of A and B against the teacher targets is read from the per-turn files (fraction of positions where the greedy
  token equals the target), plus the agreement of A with B.
- **Order and early stop:** the 8 turns of `sub8` first (the early check of the fp8-experts run). After 8: if the 95 % interval
  excludes 0 clearly (here: lower bound above +0.03 or upper below -0.03 nat) stop and report, else the other 17.
  Why this size: the shift is 0.11 to 0.28 nat on every one of those 8 turns in the stored comparison, with a per-turn spread of
  about 0.07; 8 turns decide a difference of 0.1 nat, 25 turns resolve about 0.04.
- **Expectation:** B - A = +0.15 nat (range +0.05 to +0.30), top-1 of B 3 to 8 points below A, from the 8 stored turns of 11-10
  (all arms alike, same shards). Outside the range is a finding. A difference in favour of B would contradict the stored runs.
- **Active tooling:** `TONA_TEACHER_DUMP` / `TONA_TEACHER_NLL`; no profilers, no diagnostics beyond the teacher switches.
- **Conditions:** `hw_report` and `quiet_check` before and after each block, in `/var/tmp/claude/w159/q`; every run through
  `gpu_queue --kind exclusive`.

## Limitation (coordinator, 2026-11-17)

Teacher-forced NLL on stored Claude Code turns measures how probable Claude's own replies are under a shard. For two numeric
variants of one model (fp8 against f16) that is the right quantity; between two model versions it partly measures closeness
to Claude's style, not task success. It is one signal next to the task outcomes (bench-effort Python tasks, tool-call JSON
validity, loops) and the code/prose NLL of `2026-11-12-coder-1019.md`. For a mixed variant the mix is chosen on a subset of
turns and reported on held-out turns separately.

## Result of the early check (6 of the first 8 turns, 8,012 turn tokens, 20:50 to 20:58 on 2026-11-17)

Turns 224649-0064 and 224751-0069 did not run: both stopped in `host_malloc` (pinned mapped host memory) at start, within one
second after the previous process, not a result. Not repeated yet. Per turn dNLL = new shard (10-19) minus old shard (09-29), nat:

| 0070 | 0054 | 0086 | 0095 | 0058 | 0082 |
|---|---|---|---|---|---|
| +0.172 | +0.292 | +0.205 | +0.181 | +0.262 | +0.125 |

- Mean dNLL +0.209 nat, 95 % over turns (t 2.57, 5 dof) **[+0.142, +0.271]**: the old shard is better on every one of the 6 turns.
  The stop rule (lower bound above +0.03) is met, so the other 19 turns were not run yet. Expectation was +0.15 (+0.05 to +0.30):
  inside it.
- Mean NLL over the 8,012 tokens: new 1.054, old 0.845. Top-1 against the teacher target: new 0.681, old 0.726. Greedy
  agreement between the two shards 0.811.
- Both arms use today's tree, default path (dense fp8 on), `TONA_DETERMINISTIC=1`; arm B (new shard) from the earlier runs of today.
- Conditions: Ryzen 9 9950X, kernel 7.3.4, `hw_beforeA.txt` and `quiet_*A.txt` in `/var/tmp/claude/w159/q`, exclusive queue.
- **Reading, with the limitation above:** on the stored agent turns the 10-19 shard assigns Claude's replies clearly lower
  probability. Whether the agent *works* worse is not decided by this; the task outcomes decide that.

## Shard diff (CPU, `/var/tmp/claude/w159/shard/{hashes,analyze}.py`: blake2b per tensor and per expert slot of both files)

Same result as the 11-12 compare (`2026-11-12-coder-1019.md`), recomputed from the files on 2026-11-17 and made usable for a mix:

- 192 of 1223 tensors differ: for each of the 48 layers `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` and the router `ffn_gate_inp`;
  nothing else (attention, DeltaNet, shared expert, embeddings, norms, hyper-connection). Header, directory and offsets are identical.
- Per layer 35 to 86 experts (mean 63.8, 3,063 in total) of the new shard are not in the old one; the other 75 % are byte-identical in
  gate, up, down and router row. In each layer the three expert tensors differ in exactly the same slots.
- Slots are stacked in the order of the original expert index, so a changed mask shifts slots: 200 to 256 of 256 slots differ in
  bytes per layer although only 35 to 86 experts are new. A single expert is therefore swapped as a slot of the three expert
  tensors plus the router row (the router row travels with its expert), not by position in the original order.
- Quant types per layer are the same in both shards: (gate, up, down) ggml type ids per layer are one of 7 combinations: (22,22,20) 15 layers, (18,18,20) 14, (21,21,20) 9,
  (22,22,42) 5, (18,18,42) 3, (21,21,42) 1, (23,23,20) 1; the expert byte size per slot is identical, so a slot copy is a fixed-size splice.
- Consequence: swaps are possible per layer (whole layer from either shard) and per expert slot, with no re-quantisation.
  Whether the mix works is a quality question: the router of a layer is calibrated for its own expert set.
- Tool: `tools/shard_mix.py` (sha-pinned inputs, layer or slot swaps, writes OUT and OUT.mix.json with the output sha256).

## Result of the full 25 turns (21,066 turn tokens; old-shard runs 20:50 to 21:36 on 2026-11-17)

Run notes: the queue block was killed at 21:14 by the desktop incident (turn 0083 had no output and was rerun); turns 0064 and 0069
stopped with `host_malloc` out of memory at start with the old shard (GTT 33.3 of 32.5 GB: the 10-19 shard with its profile fits, the
old shard with the same profile does not), so those two ran with `HSA_USERPTR_FOR_PAGED_MEM=1` (memory placement only, no numerical
effect). The wait before each start now checks the previous process by name and `mem_info_gtt_used` below 1 GiB instead of a sleep.

- dNLL new (10-19) minus old (09-29): token-weighted +0.2079 nat (se 0.0057 over tokens); per-turn mean +0.186, 95 % over the 25 turns
  **[+0.151, +0.221]**, **all 25 turns positive** (range +0.028 to +0.441; 0077 at +0.441, 0099 at +0.028).
- Mean NLL over the 21,066 tokens: new 1.012, old 0.804. Top-1 against the teacher target: new 0.697, old 0.742 (+4.5 points);
  greedy agreement between the shards 0.840.
- The first 8 turns alone gave +0.2258 [+0.169, +0.267], the 6 valid of them +0.209: the early check was representative.

## Caveat that goes with every number above (coordinator, 2026-11-17)

On the public texts of 2026-11-12 (`2026-11-12-coder-1019.md`, section 3) the new shard was equal on code (+0.004 [-0.002, +0.010])
and 0.023 nat better on prose; on Claude's agent turns it is 0.21 nat worse. Between two model versions, teacher-forced NLL on Claude
turns also measures closeness to Claude's style; on its own it cannot decide which shard solves agent tasks better. The decision needs
(1) eval-code and eval-text NLL at 32K of every candidate (old, new, best mix), same method as 11-12, and (2) task outcomes with real
generation (bench-effort, 23 tasks; 32 tool-call JSON prompts) on the same three. Mixes are chosen on a subset of the turns and
reported on held-out turns separately.

## Localisation (3 decisive turns 0054, 0058, 0086: chosen as the three with the largest dNLL, so selected on this set; 4,065 turn tokens)

Mixes are the new shard with the old shard's layers spliced in (`tools/shard_mix.py --old-layers`, diagnostic, no output sha), default
path, `TONA_DETERMINISTIC=1`. dNLL = mix minus full old shard; the full new shard is +0.254 on these three turns (that is the
distance to recover). Recovered = 0.254 minus the value. Intervals over 3 turns are wide (t 4.3), so read the means as indications.

| Mix (old layers) | dNLL against old | recovered | per layer |
|---|---|---|---|
| none (= new shard) | +0.254 | 0 | |
| 0-23 (lower half) | +0.173 | 0.081 | 0.0034 |
| 24-47 (upper half) | +0.160 | 0.094 | 0.0039 |
| 0-11 | +0.224 | 0.030 | 0.0025 |
| 12-23 | +0.221 | 0.033 | 0.0028 |
| 24-35 | +0.216 | 0.038 | 0.0032 |
| 36-47 | +0.197 | 0.057 | 0.0048 |

- The loss is **diffuse**: every quarter recovers 0.03 to 0.06, the quarters sum to 0.158 and the halves to 0.175 (the two sums are
  nearly equal, so the recovery is roughly additive), and the last quarter gives about twice per layer what the first does. There is no
  small group of layers that carries it, so no cheap partial swap restores the old quality; a swap gives back about 0.003 to 0.005 nat per layer.
- Both new-shard parts of a half mix are not independent either: 0.254 minus 0.175 leaves 0.08 that only the full old set removes
  (router and experts of different layers interact), so the mix is not simply the sum of its parts.
- Reading: the 10-19 mask is worse on Claude's agent turns throughout the depth, not at a few layers. Whether the new experts carry
  gains elsewhere (code, tool calls, loops) is not shown by NLL; that is what the task suite below must say, because a layer mix that
  keeps new layers for those gains pays about 0.004 nat per new layer on these turns.
- Not done: slot-level (single expert) localisation. The experts only the new shard has are 3,063; a slot swap needs the new-only
  experts to be the carriers of the loss, which this table cannot show (a layer swap exchanges experts and router together).

## Plan: public-text NLL and task outcomes on both shards (written before the runs; coordinator order of 2026-11-17, evening)

Question: does the 10-19 shard (new) solve tasks better or worse than the 09-29 shard (old) with real generation, and do the public
texts of 2026-11-12 still show "equal on code, new slightly better on prose" with today's engine? NLL on Claude turns says old is better
by 0.2 nat; this plan adds the signals that do not depend on closeness to Claude's style. No mix is a candidate (the loss is diffuse), no slot swaps.

Same build for both shards (`work/build/w7`: main behaviour, #159 switches off), each shard with its own expert profile
(`experts-coder-0929.bin` for old, `experts-coder-1019.bin` for new; `experts-coder.bin` is the 10-19 one since 11-12).
Every run through `gpu_queue --kind exclusive`; hw_report and quiet_check before and after each job.

| Job | What | Size |
|---|---|---|
| N | `tona-gpu-gen --teacher eval-{code,text}.txt --mega --prefill --max 32000 --ctx 33000`, `TONA_DETERMINISTIC=1`, 4 runs (2 texts x 2 shards), paired block bootstrap (`scripts/nll-pair.py` + `nll-compare.py`, 64-token blocks, 4,000 draws) | 4 x ~2 min |
| T | 32 tool-call JSON prompts (`toolcall32.py`: three tools, greedy, 200 tokens, valid = tool call with parseable arguments and a known tool name), `tona-server` with MTP, 2 server starts per shard | 4 x ~3 min |
| E | `scripts/bench-effort.py`, 23 own Python tasks with tests, `high` greedy (`SAMPLING=0`) and `medium` with the server's default sampling, `MAX_TOKENS` 24000, `--ctx 16384`, adaptive thinking, one run per shard and level | about 30 min (high) + 10 min (medium) per shard |

- **Expectation (all from 2026-11-12, older engine, unchecked for today's):** code NLL difference new minus old +0.004 [-0.002, +0.010],
  prose -0.023 [-0.032, -0.014]; tolerance +-0.01 on each mean. Tool-call JSON 32/32 on both. bench-effort `high`: old 22/23, new 21/23;
  `medium`: 22/23 each; no loops either side; the sum of thinking tokens at `high` about 10 % higher for new. Outside these is a finding.
- **Resolution (what would count at all):** with 23 tasks per arm a pass-rate difference is resolved only above about 30 percentage
  points (Fisher exact, e.g. 22/23 against 15/23); paired by task, a sign test needs at least 6 tasks flipping one way and none the
  other (or 8 against 2) for p < 0.05, so a difference of one to three tasks is reported as not resolved. For the 32 JSON prompts the
  test is saturated at 32/32; a loss counts only at 6 or more invalid prompts (Fisher p about 0.05 against 32/32). The thinking-token
  sum and the number of loops (answer cut at the limit with a repetitive thinking tail, `work/coder-1019/loops.py`) are reported as
  descriptive numbers. The tasks are small, single-function problems, not agentic repair; the result says "no gross failure", not "better".
- **Decision use:** if neither shard is resolved apart on tasks and the public texts are equal, the only resolved difference left is
  the 0.2 nat on Claude's agent turns (a style-closeness signal); the maintainer decides with that stated.
- **Active tooling:** `TONA_DETERMINISTIC=1` for N only; servers in T and E run the production path (MTP drafter, no diagnostics);
  `TONA_TEACHER_*` only in N; servers under `systemd-run --scope -p MemoryMax=54G -p MemorySwapMax=0`, port 8099 (never 8662, 8097
  or the unit `tona-server`), stopped by PID.

## Result: public-text NLL and task outcomes, old (09-29) against new (10-19), 2026-11-17 22:41 to 00:34

Build `w7` (main behaviour), each shard with its own profile, exclusive queue, Ryzen 9 9950X; hw_report and quiet_check before and
after in `/var/tmp/claude/w159/suite/` (`hw_beforeN.txt`, `hw_beforeTE.txt`, `quiet_*`). Raw files in the same directory.

**N: NLL at 32K, prefill path, `TONA_DETERMINISTIC=1`** (paired block bootstrap, 64-token blocks, 4,000 draws):

| Text | old | new | new minus old [95 %] | 11-12 |
|---|---|---|---|---|
| eval-code | 0.272582 | 0.276565 | +0.0040 [-0.0017, +0.0095] | same, to the digit |
| eval-text | 0.908216 | 0.885243 | -0.0230 [-0.0320, -0.0141] | same, to the digit |

The 11-12 numbers are reproduced bit for bit with today's engine, so the public-text result is stable: equal on code, new 0.023 nat better on prose.

**T: tool-call JSON, 32 prompts, greedy, 2 server starts per shard:** old 32/32 and 32/32, new 32/32 and 32/32 (saturated, no resolving power).

**E: bench-effort, 23 tasks with tests, `--ctx 16384`, `MAX_TOKENS` 24000, one run per shard and level:**

| Level | Shard | Passed | Hit limit / loop (limit + repetitive tail) | Thinking tokens median / sum | Time |
|---|---|---|---|---|---|
| high, greedy | old | 22/23 (fails: roman) | 0 / 0 | 3,678 / 166,418 | 1955 s |
| high, greedy | new | 21/23 (fails: roman, expr) | 1 / 1 (roman) | 2,795 / 167,149 | 2116 s |
| medium, default sampling | old | 21/23 (fails: csv, weekday) | 0 / 0 | 980 / 39,293 | 591 s |
| medium, default sampling | new | 21/23 (fails: csv, spiral) | 0 / 0 | 1,299 / 38,045 | 591 s |

- Flips: high `expr` old ok, new fails; medium `spiral` old ok, new fails, `weekday` old fails, new ok. One task either way; Fisher p = 1.000 on
  every row. Against the plan (resolved only above about 30 points, sign test needs 6:0): **no resolved difference** on passes or loops.
- Against the expectation (old 22/23, new 21/23 at high; 22/23 each at medium; no loops): high as expected; medium 21/23 each (one task lower than
  on 11-12 for both, `csv` fails with both); one loop at high with the new shard (`roman`, cut at the limit, repetitive tail, also failed with old). 11-12 had no loops.
  The sum of thinking tokens at high is equal today (+0.4 %, 11-12: +10 %), the median is lower for new (2,795 against 3,678; 11-12 higher):
  per-task thinking counts of greedy decoding are chaotic, as written there.
- Reading: the tasks show no gross failure on either shard and cannot separate them; the 23 small single-function tasks are the weakest evidence
  here. The signals side by side: public code equal, public prose new 0.023 nat better (stable), JSON equal, tasks equal within noise (one extra loop on new at high), Claude agent turns old 0.21 nat better on all 25 turns (a style-closeness signal as well as quality).
