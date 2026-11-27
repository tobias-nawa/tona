# bench/agent: paired agentic comparison of two server variants

`run.py compare` plays the same tasks with the same seeds against two variants (arms A and B: another model file, another
build or other `TONA_*` variables) and writes a measurement table with paired statistics (exact McNemar, Wilson, bootstrap
over tasks). Every arm-repeat is one `gpu_queue` job of kind `gpu` (quality, no timing claim). Details: `run.py --help`.

Suites (reported separately, never summed into one number without the per-suite rows):

| Suite | What the model does | Harness |
|---|---|---|
| `swe` | 17 SWE-bench Verified tasks (`bench/swe/subset.txt`), a shell agent works in a container until it submits a patch, graded by the official tests | mini-swe-agent, rootless Podman |
| `polyglot` | Exercism exercises in Python, Go, Rust, JavaScript: write the files, run the tests, read the output, fix; up to 4 attempts in one conversation | `bench/polyglot/` |
| `csharp` | the same method for the C# (.NET) practice exercises of exercism/csharp, own suite so the polyglot order stays fixed | `bench/polyglot/` with `--langs csharp` |
| `java`, `cpp`, `php`, `typescript`, `dart`, `bash` | one suite per language, the same method; the report has a row per suite and a "new languages (sum)" row. Data of `php`, `typescript`, `dart`, `bash` prepared by `bench/polyglot/prepare.py` (`java` and `cpp` use the Aider copy as it is), valid lists `valid-<lang>.txt` under `AGENT_LANGS_VALID` (default `/var/tmp/claude/poly-langs`) | `bench/polyglot/` with `--langs <lang>` |
| `effort` | 23 own Python tasks with tests | `scripts/bench-effort.py` |
| `json` | 32 tool-call prompts; valid JSON arguments | `bench/agent/toolcall32.py` |

Result files hold task ids and pass/fail only, no transcripts.

## Fetching the data

No third-party task data is stored in this repository. Each run fetches it into the directories named by the `AGENT_*`
variables in `bench/agent/suites.py`. Pin what you fetch (git commit or dataset revision) and write the pin into the
measurement file.

| Data | Source | Licence status |
|---|---|---|
| Exercism tracks | `git clone https://github.com/exercism/<track>` (`python`, `go`, `rust`, `javascript`, `cpp`, `java`, `csharp`, `php`, `typescript`, `dart`, `bash`) | MIT, see the table below |
| Aider polyglot exercise list | `git clone https://github.com/Aider-AI/polyglot-benchmark` | no licence file in the repository; used to read the exercise list and the Java and C++ exercises (derived from Exercism, MIT) |
| SWE-bench Verified | `princeton-nlp/SWE-bench_Verified` on Hugging Face, fetched by `mini-extra swebench --subset verified` | the dataset card names no licence; only instance ids are written into result files |

Adding a downloader to the repository needs a new pinned dependency, so the commands stay documentation until an issue
accepts one.

## Licences of what the suites use (checked 2026-11-18, GitHub repository metadata)

Nothing below is stored in this repository; the data is fetched into `/var/tmp/claude/139/` at run time.

| Source | Licence | Used for |
|---|---|---|
| exercism/cpp, go, java, javascript, python, rust | MIT (all six repositories) | exercises of `polyglot` |
| exercism/php, typescript, dart, bash | MIT (GitHub metadata; `LICENSE` read), commits in `research/measurements/2026-11-19-polyglot-languages.md` | exercises of the five new suites |
| bats-core (test runner of `bash`) | MIT | container only |
| exercism/csharp | MIT (`LICENSE` of the repository, read 2026-11-18, commit f4e2411) | exercises of `csharp` |
| Aider-AI/polyglot-benchmark (the curated copy of those exercises) | no licence in the repository metadata; its README names the Exercism tracks as the source, so the MIT terms of the tracks apply to the exercises. Only the JavaScript exercises carry their own `LICENSE` file (MIT) in the copy | exercise files |
| Aider-AI/aider (method: two attempts, test output fed back) | Apache-2.0 | method only, no code copied |
| SWE-bench/SWE-bench (grading code `swebench`) | MIT | `swe` grading |
| SWE-agent/mini-swe-agent | MIT | `swe` agent |
| SWE-bench Verified dataset (princeton-nlp/SWE-bench_Verified on Hugging Face) | the dataset card names no licence (no `license` field, no licence text, read 2026-11-18); the content is issues, tests and patches of the task repositories, which keep their own licences (next table) | `swe`; only instance ids are written into files |

Licences of the five task repositories of the 17 `swe` tasks (11 django, 2 xarray, 2 sphinx, 1 pytest, 1 sympy;
`bench/swe/subset.txt`): django BSD-3-Clause and pytest MIT, xarray Apache-2.0 (GitHub metadata); sphinx two-clause BSD and
sympy BSD (the GitHub metadata says `NOASSERTION`; the first lines of `LICENSE.rst` and `LICENSE` were read, sympy's
clause count was not). The images the harness pulls (`swebench/sweb.eval.*`) are third-party images, not redistributed here.

## Provenance of the polyglot exercises

144 valid exercises of 175 in the copy form the `polyglot` suite (the reference solution passes its own tests in the sandbox; C++ and Java
are separate suites since then, below). Per Exercism track, each track repository MIT:

| Track | Repository | Exercises in the copy | Used (valid) |
|---|---|---|---|
| Go | exercism/go | 39 | 39 |
| JavaScript | exercism/javascript | 49 | 49 (each has its own MIT `LICENSE` file in the copy) |
| Python | exercism/python | 34 | 34 |
| Rust | exercism/rust | 30 | 22 |
| C++ | exercism/cpp | 26 | 26 (own suite `cpp`) |
| Java | exercism/java | 47 | 47 (own suite `java`) |

The curated copy is `Aider-AI/polyglot-benchmark` (no licence field; its README lists the six tracks above as the source
and these are the exercises of those tracks, so the MIT terms of the tracks apply). Open: an exercise may name another
upstream source in its `.meta/config.json`; those were not read one by one.

The `csharp` suite uses exercism/csharp directly (not the Aider copy, which has no C#): 144 practice exercises, 143 valid
(the reference solution `.meta/Example.cs` passes in the sandbox; `ledger` does not), fetched to
`/var/tmp/claude/139/exercism-csharp` (commit f4e2411, 2026-11-15) and never stored in this repository. Each exercise
may name another upstream source in its `.meta/config.json`; those were not read one by one.

## The six languages added after the first set

Java (47) and C++ (26) come from the Aider copy as it is; PHP (61), TypeScript (58), Dart (35, no Flutter) and Bash (52)
from the Exercism tracks, selected by the rule "slug occurs in Aider's polyglot copy, not deprecated". Reference
solutions that fail with every skipped test enabled are dropped (counts, pins, toolchain versions: `research/measurements/2026-11-19-polyglot-languages.md`).
Each language is its own suite so the order of the first four stays fixed. Run `python3 bench/polyglot/prepare.py --aider DIR --track DIR --lang LANG --out DIR` for
the track data, build `bench/polyglot/Containerfile.<lang>` (C++: `Containerfile.c-plus-plus`, the clang-format hook takes `.cpp` for C++) as `localhost/tona-poly-<lang>`, then
`run.py reference` writes the valid list. Flutter UI tasks need their own benchmark (possible later item).
