# Paper scan

Complete, repeatable search for new papers and upstream work. Only the standard library; GitHub needs an
authenticated `gh`.

```sh
python3 tools/papers/scan.py coverage                                  # days without complete coverage, up to yesterday
python3 tools/papers/scan.py arxiv  --from 2026-10-24 --to 2026-11-10  # arXiv API, one query per topic and 7-day window
python3 tools/papers/scan.py github --from 2026-10-24 --to 2026-11-10  # all PRs, plus issues by topic term
```

| File | Role |
|---|---|
| `topics.toml` | the topics: arXiv expression, categories, GitHub terms; `version` rises with every change |
| `scan.py` | command line (`arxiv`, `github`, `screen`, `watch`, `coverage`) |
| `papers_arxiv.py`, `papers_github.py` | the two sources: complete pagination, retries, window splitting |
| `papers_book.py` | reads and writes the coverage book, finds gaps |
| `../../research/papers-coverage.md` | the coverage book: one row per run |

How it stays complete:

- **arXiv**: `submittedDate:[...]` window AND topic AND categories, `start`/`max_results` until `totalResults` is
  reached, 3 s between requests, 8 attempts per page (waiting 30 s times the attempt number after a failure). A short page or a failed page makes the run incomplete.
  The 3 s rule of the arXiv API terms counts per client, not per process: never run two scans at the same time
  (https://info.arxiv.org/help/api/tou.html).
- **GitHub**: `created:FROM..TO`; the search API returns at most 1000 hits per query, so a window with more hits is
  halved until every part has at most 1000. Pacing 2.5 s (limit 30 requests per minute), rate-limit errors wait 65 s.
  PRs: every PR of the seven repositories. Issues: one query per repository and topic term (not every issue).
- **Known entries**: arXiv IDs and GitHub items that already stand in `research/papers.md` get `"known": true`
  in the candidates; they are marked, not dropped.
- **Output**: `work/papers/<run date>/<source>-<from>_<to>.jsonl` (ignored by git). Reading the candidates and writing
  the relevant ones into `research/papers.md` is the manual step.
- **Coverage book**: every run adds a row (source, topic version, scope, window, requests, hits, new, complete). A day
  counts only for runs with scope `all`, `Complete` = `yes` and a topic version at least the current one.
  `--topic`/`--repo` runs and `--no-book` trial runs never count. `--to` must not be later than yesterday.

## Screening (every hit gets a decision)

`research/papers-screened.tsv` has one row for every hit, dropped ones too: `id  date  source  topics  decision
reason  method`. `decision` is `drop`, `note` or `candidate`; `reason` is a short code (`off-topic`, `datacenter-only`,
`needs-training`, `other-modality`, `lossy`, `dup`, ...; for a candidate its relevance). `method` says how deep the hit
was looked at: `title-read`, `abstract-read`, `fulltext-read`, or `keyword-filter` (a filter over the titles, nobody
read the title; reasons `title-filter` and `bulk-no-keyword`). `screen --method` sets the default, a fourth column in
the decisions file sets it per id. The book proves coverage and
prevents screening twice: `scan.py` marks every ID in it as `known`. GitHub rows with decision `drop` are the bulk
(tens of thousands for a long backfill) and go to the local, ignored file `work/papers/screened-github-drops.tsv`
instead; arXiv rows and GitHub `note`/`candidate` rows stay in the book. `scan.py` reads both files as known.

1. Stage 1, all hits: title and abstract, then a decision. `scan.py screen <candidates.jsonl> --decisions <tsv>
   --default drop:off-topic` writes the rows (`--decisions` lists the exceptions: `id<TAB>decision<TAB>reason`).
2. Stage 2, `candidate`: read the abstract page or HTML, then one row in `research/papers.md` with relevance and status
   `read`. For `high` the full text is mandatory, with test conditions (hardware, model, batch) and whether code exists.
3. Every row in `papers.md` says: source (venue, preprint, PR, blog), authors' numbers or own measurement,
   abstract-only or read, code link, hardware of the evaluation. Status chain: `new -> read -> transferable -> tested`,
   or `rejected (reason, measurement link)`.
4. Open PRs worth following go to `research/papers-watch.tsv` (`screen` adds open PRs screened as `candidate`).
   `scan.py watch` re-checks them (open, merged, closed); run it in every weekly search.

After a change to `topics.toml`, raise `version`; `scan.py coverage` then shows every window that was searched with
older topics. Tests: `tools/tests/test_papers.py` (recorded responses, no network).
