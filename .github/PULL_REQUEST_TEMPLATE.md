<!-- One topic per PR. Fill in every section; write "n/a" where a section does not apply.
     The title of this pull request is the commit on main and sets the version: `type(scope): subject (#n)`, types
     feat fix perf docs test refactor build ci chore revert, `!` or a `BREAKING CHANGE:` paragraph below for a breaking
     change (CONTRIBUTING.md, tools/gates/pr_title.py). -->

Closes #

<!-- feat, fix, perf and breaking changes: the number of the issue in the title. A feat or perf needs the label
     `accepted` on it (CONTRIBUTING.md). docs, test, ci, build, chore, refactor, revert: delete this line. -->

## What and why

<!-- What changes, and why this way. One topic per pull request; commit subjects end with `(#n)`. -->

Change type: <!-- exactly one word: bugfix | technique | backend | refactor | none (none only without engine code; refactor needs the bit-identical proof in the text; CI checks it) -->

## Prior art

<!-- Required for engine code, written by the skill `research-before-building` (tools/agent_files/). One bullet per
     paper, PR or issue: link, status (read, open, merged, closed, rejected), what you adopt and why not the rest.
     Or `nothing found`. Then one line:
     Search: tools/papers/scan.py topics v<N>; topics: <ids>; terms: <words>; window: <from>..<to> (<n> months)
     Windows: bugfix 3 months, technique and backend 12. A backend also names docs/backends/<name>.md.
     Change type none: write n/a. Proposed rows for research/papers.md go below the bullets. -->

## Platform

- Backend: <!-- e.g. hip-gfx1200 (reference), or engine/backend/<name> -->
- [ ] hardware class and concept file checked (docs/hardware/)

## Test conditions

<!-- Required. Paste the output of `python3 tools/hw_report.py` (add `--model FILE` for the model) from the machine
     that measured, unedited, plus the result of `python3 tools/quiet_check.py` before and after the run.
     The same block opens every measurement file. Without it the numbers cannot be compared. -->

```text
<output of tools/hw_report.py and tools/quiet_check.py>
```

## Checks

- [ ] `just lint` is green (gates: file/function size, format, lint, scripts, secrets)
- [ ] `just test` and `just test-gpu` are green on the platform above (`just test` alone when you have no GPU; say which
      checks only the maintainer can run: `python3 tools/select_checks.py`)
- [ ] a bug fix has a test that fails without it; no test, gate or golden reference was weakened
- [ ] no new dependency, or it has an accepted issue, a pin and a row in docs/contribute/licenses.md

```text
<last lines of `just lint` and `just test`>
```

## Correctness

<!-- Refactor: identical tokens with TONA_DETERMINISTIC=1 (command and result).
     Tokens changed on purpose: re-record every affected golden reference in this PR and say so in the measurement file.
     Numerical change: quality check from docs/backends/README.md (−log p interval, KL, top-1). -->

## Models

<!-- One row per model you implemented or measured: at least one, more are welcome. Id as in data/modeldb/models/.
     List every model you did not measure under `Not measured:`; others can pick them up in the issue. "n/a" for a
     change without measurements. -->

| Model id | Quantization / file | Measured (golden/quality, decode, prefill) |
|---|---|---|
| | | |

Not measured: <!-- ids, or none; others can pick them up in the issue -->

## Measurements

<!-- Before and after, several runs, spread. Name every rate (decode / prefill). Add a row to BENCHMARKS.md
     and a file under research/measurements/ that opens with the plan written before the run: scope, expectation
     with tolerance, active tooling (AGENTS.md rule 2). -->

| Setup | Before | After | Runs |
|---|---|---|---|
| | | | |

## Effect on the reference backend

- [ ] Reference code untouched
- [ ] Shared code changed: reference measured before/after (bench-set, same tokens, speed within spread)

## AI use

<!-- Exactly one line `AI use:` below. Write `AI use: none`, or `AI use: <agent and model, what it did>`. With tools,
     end this text with one line `Assisted-by: <agent>:<model>` per agent (example:
     `Assisted-by: Claude Code:claude-sonnet-5-5`); the squash merge keeps this text, not the commits' trailers.
     With `none`, no `Assisted-by:` line. Provenance, not a judgement. -->

AI use:

By opening this pull request I confirm that I have read every line of this change and can explain it without the
tool, that I may contribute it under the MIT licence of this repository, and that it reproduces no third-party code
that I cannot license that way (GitHub Terms of Service, section D.6: inbound = outbound;
[CONTRIBUTING.md](../CONTRIBUTING.md#rights-and-dependencies)).
