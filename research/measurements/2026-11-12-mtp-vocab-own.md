# MTP draft vocabulary from own data only, 2026-11-12

Status: **run on 2026-11-12, result below.** The A/B report of part 2 is in `2026-11-12-mtp-vocab-own-ab.md`.

## Test conditions

Output of `tools/hw_report.py` (before the run, load already settled): Ryzen 5 9600X (6 cores / 12 threads), RX 9060 XT
gfx1200 15.9 GiB VRAM, 1.17 GiB held by others (the desktop), 0-1 % GPU busy, RAM 60.5 GiB with 55.5 GiB available, swap
in use 2.02 GiB, load average 0.71 / 2.0 / 1.84, Fedora 44, kernel 7.2.15, HIP 7.16.26385, no `TONA_*` set, commit
`bdef57f` (merge of `main`, dirty tree: plan file). `quiet_check.py` before the run: warning only for the 1.17 GiB
VRAM of the desktop (not quietable without a session change; measured anyway); the first check 3 minutes earlier
also warned for a load average of 4.73 (the build), so it was repeated after the load settled. The saved file
`work/quiet-before.json` was overwritten by a second, late-running copy of the same command while the first server
was loading, so the comparison `quiet_check.py --compare` after the run is not usable ("DISTURBED" is an artefact of
that); after the run the check found 74 % CPU use by other programs and load 3.04 with no server of ours running.
Within the run the rounds of `ab.py` are stable for decode (spread below 1 %), only prefill shows outliers (below).

## Plan (written before the run)

### What changes

`engine/model/mtp_vocab.inc` is the token list of the truncated LM head of the MTP drafter (rows that can be drafted;
8,192 more rows come from the context at run time). It steers drafts only: the target verifies every token, so the
output text must not change.

- Old list (A): 41,489 ids = 24,576 ids of our own count over code in 8 languages + the `code32` list of another
  project (32,768 ids, no licence), ids in `work/mtp-vocab-old.txt` (taken from `main`).
- New list (B): 41,500 ids from `scripts/build-mtp-vocab.py`: 400 special ids + the best ids by summed relative
  frequency over 9 own corpora (Python, Rust, JS/TS, C/C++, Go, C#, shell, Markdown, web and data formats; 8 MB each,
  duplicates skipped; tokenized with `tona-tokenize` on the CPU). 30,123 ids are shared with A, 11,377 are new, 11,366
  of A are gone.
- No model output of ours exists on disk in useful size (`work/golden` holds about 3 KB of answers), so the list is a
  count over real source and text, not over generated answers. The offline check against the files we do have
  (golden texts, `work/golden/prefill/long.txt`, `scripts/bench-set.py`; 15,118 tokens, partly drawn from the same
  repositories as the corpus, so optimistic): ids missing from the list 435 (2.9 %) with A, 158 (1.0 %) with B. This is
  not a result of the run.

### Scope

Same server build for both arms (the list is selectable at run time: A = `TONA_MTP_VOCAB=work/mtp-vocab-old.txt`,
B = built-in list), so no other difference exists. Model `work/models/coder-iq1_m/` (big), drafter
`work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, `--ctx 32768`, greedy, `TONA_DETERMINISTIC=1`.

1. **Tokens per pass** (`scripts/bench-set.py`: 11 code and 3 text prompts, 400 tokens each; it prints tokens per
   verify pass per group; this is the rate asked for, tokens per verify pass, not tok/s). 3 runs per arm, alternating
   ABABAB. One server start per run. Duration: 6 starts x (about 1 min start + 3.5 min set) = about 27 min. Why 3: in
   deterministic mode the text is fixed, so tokens per pass is expected to repeat exactly; 3 runs show it (a spread
   would be a finding).
2. **Decode speed and bit identity** (`scripts/ab.py`, workloads `short,set`, A against B, `--env TONA_DETERMINISTIC=1`).
   Rounds: `bench/noise.toml` has `decode-code` with df 2 only (not used, df < 4), so ab.py takes its prior; allow up
   to 6 rounds, it stops when decided. Duration: about 5 min per round, 5-30 min. `short` gives the SHA of the text
   per request for the identity check (A against B); `set` gives decode tok/s of code and text.
   Decode tok/s here is the end-to-end rate with MTP (verify passes and drafts together).

### Expectation (B against A), revised 2026-11-12 after merging `main`

`main` switched the references to Coder shard 1 of 2026-10-19 (`bench/sha.toml`, new goldens, new
`work/profile/experts-coder.bin`). The old numbers (5.27 code, 3.00 text) come from the shard of 2026-09-29 and are
**not known for this shard**; they are not used as a bound any more. Arm A (old list, same build, same shard) is the
reference and the tolerance is relative to A. Source of the comparison: `research/measurements/2026-10-26-mtp.md`
(old shard: the three lists differ by at most 0.1 tokens per pass for code) and the offline coverage above.

| Quantity | A (old list) | B (new list), tolerance |
|---|---|---|
| Code tokens per pass | reference (value measured; expected near 5.3, not binding) | A - 0.10 or better; 3 runs per arm |
| Text tokens per pass | reference (expected near 3.0, not binding) | A +- 0.10 |
| Decode tok/s, code and text (`set`) | reference | -2 % to +2 % (MTP time per pass equal: 41,500 against 41,489 rows) |
| Output SHA (`short`, `TONA_DETERMINISTIC=1`) | reference | identical in every request and round |

Decision rule (from the coordinator): B >= A - 0.10 tokens per pass on code and identical SHAs: "fits". Below that
the file proposes how to make the list larger or weight it differently (no rollback to code32, no iteration inside
this run). A result above A is a finding, not noise. A result outside the table in either direction is a finding.
Each wait for `/status` is limited to 10 minutes, then the run aborts with a message.

### Active tooling

`TONA_DETERMINISTIC=1`; `TONA_MTP_VOCAB=<repo>/work/mtp-vocab-old.txt` on A only;
`TONA_EXPERT_PROFILE=work/profile/experts-coder.bin` and `OMP_NUM_THREADS=6`, `ROCR_VISIBLE_DEVICES=0` as set by
`scripts/ab_server.py`; no `TONA_DIAG`, no profiler, no `TONA_MTP_VOCAB_N`, `TONA_MTP_DYN` at its default 8192.
No build runs during the measurement.

### Start command

Build first (not under the lock; do not build while a measurement runs), in the worktree `<worktree>`
(its `work/` is the shared one):

```sh
cd <worktree> && source toolchain/env.sh
cmake -S engine -B work/build/tona124 -G Ninja >/dev/null && cmake --build work/build/tona124 --target tona-server
python3 tools/hw_report.py && python3 tools/quiet_check.py --json > work/quiet-before.json
```

Run (both parts under one lock; the lock is held by `flock`, so `--no-lock`):

```sh
cd <worktree> && flock work/gpu.lock bash -c '
set -euo pipefail
OLD=<repo>/work/mtp-vocab-old.txt
B=work/build/tona124/tona-server
M=$(ls work/models/coder-iq1_m/*-00001-of-*.gguf); D=work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
mkdir -p work/ab/mtp-vocab-own
for order in "A B" "B A" "A B"; do
  run=$((${run:-0} + 1))
  for arm in $order; do
    list=(); [ $arm = A ] && list=(TONA_MTP_VOCAB=$OLD)
    env -u TONA_MTP_VOCAB TONA_DETERMINISTIC=1 OMP_NUM_THREADS=6 ROCR_VISIBLE_DEVICES=0 \
        TONA_EXPERT_PROFILE=work/profile/experts-coder.bin "${list[@]}" \
        systemd-run --user --scope -q -p MemoryMax=54G -p MemorySwapMax=0 $B "$M" --mtp "$D" --port 8099 --ctx 32768 \
        > work/ab/mtp-vocab-own/server-$run-$arm.log 2>&1 &
    pid=$!
    for _ in $(seq 200); do curl -sf http://127.0.0.1:8099/status | grep -q "\"ready\": *true" && break; sleep 3; done
    python3 scripts/bench-set.py http://127.0.0.1:8099 | tee work/ab/mtp-vocab-own/set-$run-$arm.txt
    kill $pid; wait $pid || true
    sleep 5
  done
done
python3 scripts/ab.py --a tona124 --b tona124 --a-env TONA_MTP_VOCAB=$OLD --env TONA_DETERMINISTIC=1 \
  --workload short,set --max-rounds 6 --port 8099 --no-lock --out work/ab/mtp-vocab-own/ab \
  --why-size "short: SHA identity of the text per request; set: decode tok/s of code and text, the list steers MTP drafts only" \
  --expect-source "research/measurements/2026-10-26-mtp.md and the plan of research/measurements/2026-11-12-mtp-vocab-own.md" \
  --expect decode-code:-2..+2% --expect decode-text:-2..+2%
'
```

Syntax of the loop was checked with `bash -n`. The arm is named in each log file name (A = old list). After the run:
`python3 tools/quiet_check.py --compare work/quiet-before.json`, then copy the section "Result" below from
`work/ab/mtp-vocab-own/` and add a row to `BENCHMARKS.md`.

## Result

Deviations from the plan: (1) `bench-set.py` prints no tokens per pass when the server reports no draft counters, so
tokens per pass come from the per-request line of the server log (`... tokens per pass`, two decimals), aggregated as
sum of tokens / sum of passes over the first 11 requests (code) and the last 3 (text) of each run
(`work/ab/mtp-vocab-own/tpp.py`). (2) `ab.py` demanded an `--expect` for every metric; part 2 was started a second
time with `--expect decode:-2..+2% --expect prefill:-2..+2%` (the first start aborted at once, no GPU time). (3)
Waiting for `/status` was limited to 10 minutes (it was never close).

### Part 1: tokens per verify pass (bench-set, 3 runs per arm, alternating ABABAB... as A B / B A / A B)

| Group | A (old list) | B (new list) | B - A |
|---|---|---|---|
| Code, tokens per pass | 3.379, 3.379, 3.379 (spread 0.000) | 3.368, 3.368, 3.368 (spread 0.000) | **-0.011** |
| Text, tokens per pass | 2.561, 2.561, 2.561 | 2.529, 2.529, 2.529 | -0.032 |

In deterministic mode the runs repeat exactly, as expected. The plan expected about 5.3 (code) and 3.0 (text) from the
old shard; on the Coder shard 1 of 2026-10-19 and this prompt set the level is lower (3.38 and 2.56), so the old numbers
do not carry over; that is why A is the reference. The difference B - A is within 0.10 on code.

### Part 2: `ab.py`, `TONA_DETERMINISTIC=1`, 6 rounds (full report: `2026-11-12-mtp-vocab-own-ab.md`)

| Quantity | Result | Plan |
|---|---|---|
| SHA of the text (`short`, 2 requests, 6 rounds) | identical A = B and stable in all rounds: `11eacf1ee10017b4`, `a6f07fd6fb86e448` | identical: **met** |
| Decode, bench-set code | A 85.8, B 85.5 tok/s (median of 6), B vs A -0.21 % [-0.82, +0.40] | -2 to +2 %: met |
| Decode, bench-set text | A 69.3, B 68.8 tok/s, -0.87 % [-1.46, -0.27] ("B slower", inside the margin of the plan, consistent with -0.03 tokens per pass = -1.2 %) | -2 to +2 %: met |
| Decode after a 2K prompt (`short`) | A 72.2, B 73.2 tok/s, +1.32 % [+0.89, +1.76] (B faster) | met |
| Decode after a 2K write prompt | +0.14 % [-0.63, +0.91] | met |
| Prefill after 2K tokens | +6.61 %, driven by 3 of 6 rounds in which A was at 1360 against 1515 tok/s | outside: finding |
| Prefill after 2K write prompt | +1.35 % (round 1 +7.8 %), rounds 2-6 within +-0.25 % | met |

The prefill outliers are in arm A (three rounds at about 1365 tok/s, the others at 1515), in different positions of the
order (AB, AB, BA...), not tied to the list: the list steers only the MTP drafter and cannot touch prefill. They are
reported as a finding, not explained (no profile taken). The expert-cache hit counts of the request `2000` differ
between A and B (142,122 against 140,360 hits, misses equal): different drafts, different experts touched; the text is
identical.

### Verdict

**Fits.** Code tokens per pass B - A = -0.011 (rule: at least -0.10) and the SHAs are identical in every request and
round. The new list loses 0.03 tokens per pass on text (-1.2 %), and about 0.9 % decode tok/s on the three text
prompts; code decode is unchanged within the noise. Two notes for the owner: the text loss is small but consistent
(3 of 3 runs, 6 of 6 rounds below A); if text matters, a Markdown/prose weight in the next list version is the first
lever (the head costs about 7 us per 1,000 rows per step). The prefill outliers of arm A are worth a look on their
own, independent of this change.

## Note added later the same day: which mode these numbers describe

The runs behind the tokens-per-pass and decode numbers above ran with `TONA_DETERMINISTIC=1`, which caps the drafts at
3 (`engine/server/generator_env.h`, passes up to 4 tokens); users run up to 7. They compare the two lists fairly at that
depth, but they are not the production rates. Tokens per pass here are pooled (tokens / passes); the scale runs report
the mean of the per-request values, which is higher (2026-11-10 logs: mean 4.244, pooled 3.629). At fixed depth 7 on
bench-set (work of the dynamic draft head, same day) the own list gives 5.199 code / 3.153 text tokens per pass.
