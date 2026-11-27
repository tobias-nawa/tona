# KV place threshold (#82)

## Plan (written before the first run)

**Question.** From which KV size (ctx) is the KV cache in pinned RAM better than or equal to the KV in VRAM, for decode
and for prefill? Today the rule is "pinned RAM from 1 GiB of KV in total" (`engine/gpu/mega_build_bufs.cpp`,
`ProgramBuilder::setup_kv`). The measurement of 2026-10-30 (`2026-10-30-kv-placement.md`) showed at ctx 70000
(KV 0.97 GB) that RAM with staging (85 slots) beats VRAM (63 slots) by 1.5-3.4 % in prefill at 16K-64K with equal
decode; below 32K ctx and prompts under 6,144 new tokens (not staged, the attention reads the RAM directly) were not
measured. Since then #36 (extra expert slots in the prefill workspace) is merged.

**Build.** `work/build/w82` = main 0620741 (no code change in the measuring arms). Arm A = `IE_KV_HOST=0` (VRAM),
arm B = `IE_KV_HOST=1` (pinned RAM), same binary. All differences are therefore B against A: positive = RAM faster.

**Scope.** One `scripts/ab.py` run per ctx (16384, 32768, 70000), workload `server` + `bench-set` (one server start per
arm and round, so the decode and the set share the start), decode 600 tokens per request (at least 500, so the decode
rate is per pass over many passes). Prompt sizes per ctx:

| ctx | KV (int8) | prompts (new tokens, cold) |
|---|---:|---|
| 16384 | 0.23 GB | 2000 (not staged), 8000, 15000 |
| 32768 | 0.46 GB | 2000, 8000, 16000, 31000 |
| 70000 | 0.97 GB | 2000, 8000, 16000, 32000, 64000 |

Why this size: the KV place changes (a) the expert slots (VRAM freed by the KV becomes slots: the ctx decides how
many), (b) the prefill attention (staged from 6,144 new tokens, direct RAM read below), (c) the decode attention reads
(over PCIe from RAM). 2000 covers the unstaged path, 8000 the first staged size, the largest prompt per ctx the
longest read. `bench-set` gives the production decode (MTP drafter). 600 decode tokens as the ticket requires.
Rounds: no entry in `bench/noise.toml` for these targets, so at least 2 rounds; at most 4 (GPU time: the answer is a
threshold, a margin of 0.5 % is not needed for it; undecided after 4 is reported as undecided).
Counters first: slots per arm from the server log (`grep -i slot`), expert hits and misses per request from `ab.py`.

**Bit-exactness.** The KV place changes no number. One `IE_DETERMINISTIC=1` run per ctx (1 round, decode 200, the same
prompts): the SHAs of A and B must be equal for every prompt.

**Expectation (source: `2026-10-30-kv-placement.md`, ctx 70000 table and Step 1).**

- ctx 70000, prefill 8K-64K, RAM against VRAM: +1.5 to +3.4 % (tolerance -1..+5 %); decode equal (-2..+2 %; the
  measured range was +-1.5 ms of 44 ms).
- ctx 70000, prefill 2000 (unstaged): no earlier value; guess -5..+3 %.
- ctx 32768 and 16384: no earlier value. The extra slots are fewer (the KV is smaller: about 22 slots for 0.97 GB, so
  about 10 at 32768 and 5 at 16384), the RAM read costs the same per token. Guess: prefill -4..+3 %, decode -3..+2 %,
  the advantage shrinking with the ctx.
- Plan check `--expect`: `prefill` -5..+5 %, `decode` -3..+3 % for all (wide, these are guesses); a result outside is a
  finding.

**Active tooling.** `IE_DIAG=1` on both arms; `OMP_NUM_THREADS=6` and the memory limit are set by `ab.py`; no profiler;
no other `IE_*` variable. Server port 8097 (default of `ab.py`), never 8080; GPU lock by `ab.py`; no build during a
run; 60 s pause between blocks. Expected duration: about 15-40 min per ctx for 2-4 rounds (decided after round 1).

### Plan addendum (written after the three runs above, before the fourth)

The three runs (results below) put the break-even between ctx 32768 (KV 0.45 GB: RAM decode +0.9 to +1.2 %, prefill
-1.7 to -2.1 %) and ctx 70000 (0.97 GB: decode +2.3 to +3.7 %, prefill -1.3 to +4.0 %). One more point decides where
inside this gap the rule goes: **ctx 49152** (KV 0.68 GB), same recipe, prompts 2000, 8000, 16000, 32000, 45000,
600 decode tokens, `bench-set`, 2 to 4 rounds. Expectation (linear between the two neighbours): decode +1.5 to +3 %,
prefill@8000 -2 to -1 %, larger prompts -1 to +3 %; slots A about 70, B about 86. The `--expect` ranges are the same
as above.

### Plan addendum 2 (before the deterministic runs)

`IE_DETERMINISTIC=1` on both arms, two rounds (the minimum of `ab.py`) without warm-up per ctx (16384, 32768, 70000), prompts 2000, 8000 and the
largest of the speed run (15000, 31000, 64000), 200 decode tokens, workload `server` only. Expectation: the SHAs of A
and B are equal for every prompt (and equal to the SHAs of the speed runs for 2000 and 8000, which were the same
prompts). The rates of these runs are not evaluated (deterministic mode, one round); duration about 4 min per ctx.

## Results

Reports of `scripts/ab.py` (plan block, every round, expert counts, SHAs): `work/ab/w82-ctx<N>.md` and `.json` with the
server logs in `work/ab/w82-ctx<N>-logs/`, console copies in `/var/tmp/claude/w82/ab-<N>.out`. Server 8097, `--ctx` as
named, one server start per arm and round, order alternating AB/BA, one discarded warm-up start of A first.
**Prefill** = prompt tokens computed per second of the prefill; **decode** = tokens per second of the request (MTP
drafter). The SHAs are equal in A and B (below), so both arms decode the same tokens in the same number of passes: the
ratio of decode tok/s equals the ratio per pass. The follow-up chain matters for reading the prefill rows: the prompts
of one run are prefixes of the same corpus text, so each request after the first reuses the conversation before it
(prompt cache) and computes only the tokens after it: 8000 is cold (about 7.7K new), 15000 / 16000 are follow-ups with
about 7K / 8K new tokens, 31000 / 32000 about 16K new, 45000 about 13K, 64000 about 32K. All of them are at or above
6,144 new tokens (staged) except 2000, which is not staged.

## Counters: slots

| ctx | KV | A (VRAM) slots | B (RAM) slots | A VRAM GB | B VRAM GB | B pinned GB |
|---|---:|---:|---:|---:|---:|---:|
| 16384 | 0.23 GB | 82 | 88 | 14.34 | 14.43 | 25.5 |
| 32768 | 0.45 GB | 77 | 87 | 14.39 | 14.39 | 25.8 |
| 49152 | 0.68 GB | 71 | 87 | 14.35 | 14.45 | 26.0 |
| 70000 | 0.97 GB | 65 | 86 | 14.45 | 14.42 | 26.3 |

Expert cache per request (hits / misses, prefill and decode together, identical in every round, so the counters are
deterministic): B has fewer misses at every ctx, for example the 2000 request: 16384: 18,139 against 16,168; 32768:
19,912 against 16,474; 49152: 22,256 against 16,474; 70000: 24,990 against 16,815. The misses of A grow with the ctx
(fewer slots), those of B stay flat.

## Speed, B (KV in RAM) against A (KV in VRAM), mean of the paired differences in percent

Rounds: 4, 4, 4, 2 (70000 decided after round 2). `+` = RAM faster. Bold = the interval excludes 0 (decided); the others
are open (undecided after 4 rounds, interval contains 0). The prefill rate is of the prompt named; its new tokens are in
the paragraph above.

| ctx | 2000 prefill | 2000 decode | 8000 prefill | 8000 decode | 15-16K prefill | 15-16K decode | 31-32K prefill | 31-32K decode | 45K / 64K prefill | 45K / 64K decode | set code | set text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16384 | +0.65 | +0.39 | -0.33 | +0.45 | -1.51 | +0.33 | | | | | +0.40 | +0.16 |
| 32768 | **+0.73** | **+0.92** | **-2.14** | **+0.90** | **-1.71** | +0.06 | **-2.09** | +0.01 | | | **+1.15** | **+1.25** |
| 49152 | **+1.11** | **+1.39** | **-1.49** | **+0.99** | **-1.89** | **+1.09** | **+3.51** | **+1.64** | +0.16 | **+1.49** | **+1.78** | **+1.96** |
| 70000 | **+1.73** | **+2.30** | **-0.97** | **+2.78** | **-1.26** | **+3.04** | **+3.97** | **+3.70** | **+1.26** | **+3.30** | **+3.12** | **+3.27** |

(Absolute values, A to B: prefill@2000 1,281 to 1,289 (16384), 1,285 to 1,294 (32768),
1,316 to 1,331 (49152), 1,304 to 1,326 (70000) tok/s; decode@2000 66.1 to 66.4, 66.4 to 67.0, 65.1 to 66.1, 64.8
to 66.3 tok/s; set code 88.9 to 89.2, 89.0 to 90.1, 86.9 to 88.6, 86.6 to 89.3 tok/s; set text 64.5 to 64.4,
64.2 to 65.0, 62.5 to 63.8, 62.7 to 64.8 tok/s. More in the reports.)

Against the plan: all values lie inside the (wide) `--expect` ranges except the decode rows of ctx 70000 at 16K, 32K
and 64K and the set (+3.0 to +3.7 % against the upper bound +3 %): findings, in the direction of RAM faster. The
expectation from 2026-10-30 (ctx 70000: prefill 16K-64K +1.5 to +3.4 %, decode equal) is **partly not reproduced**:

- Decode: RAM is not equal but faster, +0.9 to +1.2 % at 32768, +1 to +2 % at 49152, +2.3 to +3.7 % at 70000. At 16384
  equal. The reason is in the counters: RAM keeps 86-88 slots, VRAM falls from 82 to 65 with the ctx.
- Prefill: cold 8000 and the follow-ups with 7K-8K new tokens are **slower** in RAM, -1 to -2 % at 32768, 49152, 70000
  (-0.3 and -1.5 % at 16384, open); on 2026-10-30 these sizes were faster in RAM by 1.5 to 3.4 % at 70000. The change
  since then that touches the VRAM arm is #36: the prefill workspace holds extra expert slots between prefills, which
  gives the VRAM arm slots too. Prefill with 13K and more new tokens is faster in RAM (+3.5 to +4.0 % at 31-32K, +1.3 %
  at 64K), at 13K (45000) equal.
- The unstaged path (2000 tokens, KV read directly from RAM) is not slower: +0.7 to +1.7 % prefill, with the decode
  gain as above. The fear of the ticket (direct read costs) does not materialize.

## Bit-exactness

SHAs of the speed runs (greedy, not the deterministic switch): equal in A and B for every prompt in every round and
stable over the rounds: 2000 f6573f396c15300f, 8000 a97a64af564c274a (all ctx), 15000 eb0e16c310c3b6e7 (16384),
16000 31bb727bc5cc2e5f (32768, 49152, 70000), 31000 9a832e75cac51dec, 32000 bbcaa30e7e0c3771, 45000 a1d510500e0979df,
64000 6abc6c35a27b313f. Deterministic runs (`IE_DETERMINISTIC=1`):

| ctx | prompt | SHA A (VRAM) = SHA B (RAM) | stable over 2 rounds |
|---|---|---|---|
| 16384, 32768, 70000 | 2000 | a5999d5dc73db7da | yes |
| 16384, 32768, 70000 | 8000 | 1aaf29af01af6f0d | yes |
| 16384 | 15000 | 16020266c5955c3f | yes |
| 32768 | 31000 | 6675bd378f51bf7c | yes |
| 70000 | 64000 | 45da463fd94d6a7a | yes |

The KV place changes no number, in the staged and the unstaged path.

## Decision

The threshold **stays at 1 GiB**; the code changes only in its comment. Reason, by the rule "nowhere worse" (AGENTS.md
rule 1, `small-gains-count`): there is no ctx below 1 GiB where RAM is equal or better in decode **and** in prefill
for every prompt size. Decode wins from ctx 32768 (+1 % rising to +3 % at 70000), but the staged prefill of prompts
with 6-16K new tokens loses 1 to 2 % at 32768, 49152 and 70000, and 70000 itself (0.97 GB) is already inside the old
rule. Open for the maintainer (decision): a threshold of about 0.6 GiB (ctx about 45K) would trade -1 to -2 % prefill on
6-16K-new-token prompts (cold or follow-up; a typical agent follow-up has fewer than 6,144 new tokens and gains
+1 % prefill, +1.4 % decode) for +1 to +3 % decode everywhere and +3.5 % prefill on 13K+ new tokens.
Idea to remove the loss instead: the staged prefill in RAM costs more than the slots bring at 6-16K new tokens
(a measurement of the stage-in copy per layer at that size would show it); not part of this ticket.
