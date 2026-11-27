# needle: retrieval at long context, greedy, chat format

A needle test that tells two variants apart (for example `TONA_IDX_TOPK` 2048 against 4092, or `TONA_ROPE_YARN` off
against 4). It replaces the teacher-forced block of phase 1 (`research/measurements/2026-10-31-long-context-yarn.md`,
section 4), whose hit rate moved by +-5 of 24 from text to text.

## What it measures

A document of source files (a pool, shuffled per text) holds 40 needles `# NOTE: the access code for vault <name> is
<6 digits>.`, eight in each of five depth bins: 1-10 %, 22-28 %, 47-53 %, 72-78 % and 90-99 %. The first user message
holds the document and the first question. Greedy, thinking off:

- **single**: 20 needles, one question each as further turns of the same conversation ("What is the access code for
  vault X? Answer with the code only."). The document is computed once per text.
- **multi**: the other 20 needles in one last answer (`name: code`, one line per vault), the case where Qwen models fall
  off beyond 256K (research/33). The sets are disjoint.

A hit is the code as a whole number in the answer (single) or on the line of its name (multi). Names, codes, depths and
the single/multi split follow from the text number only, so two variants see the same questions and compare pairwise.

## Easy and hard

The plain test (**easy**) is saturated: Qwen finds a uniquely named vault line at every depth (64K 159/160, 128K 160/160,
250K 160/160, 512K 40/40, 1M 39/40; `off`), so it cannot show a gain from a larger top-k or a loss from YaRN. `--hard`
adds what a retrieval-limited model fails on:

- **Redefinition.** Each needle is revision 3 of its vault; revisions 1 and 2 with other codes stand at random places.
  The question asks for the code of the highest revision.
- **Lookalike key.** `vault <name>-b` has a revision 3 with another code.
- **Many needles** (40 real, 120 decoy lines per text), asked one by one and as a block of 20.

With `off` (4 texts up to 250K, 3 above): 64K 51 %, 128K 50 %, 250K 43 %, 512K 39 %, 1M 39 % of 160 / 160 / 160 / 120 /
120 needles. The hit rate falls with depth (250K: 91 % in the first 10 %, 3 % in the last 10 %; not examined).

89 to 100 % of the missed single questions answer with the code of a decoy line of the asked vault (stale revision, less
often the lookalike); none answers without a code. The vault is found and the wrong revision picked, so the hard test
measures disambiguation more than finding. It does not replace the easy test as a check that a needle is found (99 to
100 % up to 512K, 97.5 % at 1M).

## How many needles

McNemar's exact test on paired outcomes. With a share psi of discordant pairs, a difference d needs
`n = (1.96 sqrt(psi) + 0.84 sqrt(psi - d^2))^2 / d^2` pairs for 80 % power (`report.py plan`):

| psi | d = 10 pp | 15 pp | 20 pp |
|---|---|---|---|
| 0.15 | 116 | 50 | 27 |
| 0.25 | 194 | 85 | 47 |
| 0.35 | 273 | 120 | 67 |
| 0.50 | 391 | 173 | 96 |

Variants that differ by a setting like top-k mostly agree: psi 0.25 to 0.35, so **120 pairs find 15 points**. A text
gives 40 pairs, so **3 texts are the minimum at every length**, 4 where the prefill is short. Cost: one prefill per text
(1.0 M tokens about 17 min, 512K 6 min, 250K 2.5 min, 128K 75 s) plus about 1 s per question.

## Use

```sh
# a pool of source files, once (about 6 MB of code are 1.9 M tokens)
bench/needle/corpus.py work/needle-pool.txt engine docs bench --max-mb 8

# server with --ctx for the length, then per variant and length
bench/needle/run.py --pool work/needle-pool.txt --length 131072 --texts 0,1,2,3 --label topk2048 --out work/needle/
bench/needle/run.py --pool work/needle-pool.txt --length 131072 --texts 0,1,2,3 --label topk4092 --out work/needle/

bench/needle/report.py summary work/needle/*.jsonl
bench/needle/report.py compare --a topk2048 --b topk4092 work/needle/*.jsonl
bench/needle/report.py plan --pairs 120
```

`--length` is the token count of the first message, found with `/v1/messages/count_tokens` to within 0.3 %.
`--require-cache` stops when a later turn computes the document again. A finished text is skipped on a second call, so an
interrupted run resumes.

Output: one JSONL per variant, length and text (`<label>-<length>-t<text>.jsonl`), one line per needle with `hit`,
`mode`, `bin`, `frac` (depth), the answer, `prompt_n`, `cache_n`, `prompt_ms`; the last line of a finished text has
`"done": true`. `summary` prints hits with the Wilson interval per variant and length, by mode and depth bin; `compare`
prints the pairs (both right, only a, only b, both wrong), the difference with a Wald interval and the exact McNemar p.

## Limits

- **The needles of one text are not independent** (shared document and history). Intervals and p are too narrow; the
  number of texts is the honest count. Check per-text rates before trusting a small p.
- **Run long contexts with `TONA_SEQS=1 TONA_PROMPT_RAM_GB=0`.** Follow-up questions continue the running conversation
  (28 tokens computed per turn at 1M, 1,000,000 reused). Every new text would stash the previous one (9 GB at 512K, 16.8 GB
  at 1M); with two slots at 512K that copy once stood 48 min in `hipMemcpy` (docs/use/environment.md, budget rule). Use the
  same flags for every compared variant.
- **Pools overlap between texts**: different shuffles of one pool; a 1M text uses more than half of a 6 MB pool.
- **Haystack is source code**; prose or another needle type gives other rates.
- **Greedy, one sample per question**: differences below the table above are noise.
- The pool is built from the caller's own trees; the phase 1 pool (this repository plus five other projects) is not in
  the repository.

Unit tests (no server): `python3 -m unittest discover -s bench/needle`.
