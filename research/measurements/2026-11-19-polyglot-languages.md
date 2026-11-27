# 2026-11-19: Six more languages for the polyglot coding eval (Java, C++, PHP, TypeScript, Dart, Bash)

Status: finished. The plan below was written before the reference runs; the sixth language (Bash) was added by the maintainer during the work (same plan). Results and deviations follow the plan.
Ticket #165. Harness change plus reference check only: no model server was started, no eval was run, no GPU was used.

## Plan

**Question.** Can the polyglot harness (`bench/polyglot`, called by `bench/agent`) carry five more languages so the big model
comparison (Coder, Ornith, Bolt) covers them, and how many exercises of each survive the reference check?

**Scope.**
- Java and C++ (and later Bash, see Results): Aider's polyglot copy as it is (47 and 26 exercises, Aider's own selection), commands as Aider runs them.
- PHP, TypeScript, Dart: the official Exercism tracks at a pinned commit. Selection rule: the practice exercises of the
  track whose slug occurs in Aider's polyglot copy (any of its six languages, 100 distinct slugs), without exercises the
  track marks `"status": "deprecated"`. Same slugs across languages keep the languages comparable. Dart only, no Flutter.
- One reference check per language: the exercise's reference solution (`.meta` example) must pass its tests in the sandbox
  (`run.py reference`, as for the first four languages); failures are dropped for every model and listed with the reason.
- Skipped tests are enabled the way Aider does it (Java: its `@Disabled` regex; JavaScript `xtest`; here TypeScript `xit`,
  `xtest`, `xdescribe`, Dart `skip: true`, PHP `markTestSkipped`). The tests are always the pristine files; the model can
  only edit the exercise's solution files, never tests or build and package files (`tasks.PROTECTED`).
- Container builds and reference runs are CPU work: `gpu_queue.py --kind cpu`, rootless Podman, `--network=none` at test time.

**Expectation (with tolerance).** The reference solutions are the track's own CI fixtures, so most pass: at least 90 % per
language, i.e. valid counts of Java 42 to 47, C++ 22 to 26, PHP 54 to 61, TypeScript 54 to 60, Dart 31 to 35. Sources: the
first four languages passed 22/30 (Rust, no network for crates), 39/39, 49/49, 34/34; C#, 143/144. A language far below
that is a finding about the sandbox, not noise; I look at the failing output before dropping anything.
Run time: about 4 to 10 s per exercise (measured in the smoke test: Java 4 to 5 s, C++ 5 s, PHP under 1 s, TypeScript 1 s,
Dart 3 s), so each language takes minutes.

**Active tooling.** None beyond the harness: no diagnostics, profilers or `TONA_*` switches (no engine involved).

**Stop rules.**
- A reference failure whose cause is the sandbox (missing library, path, permissions) is fixed in the image or the
  harness and the language is re-run; a failure whose cause is the exercise (reference solution wrong, test known-broken,
  test too slow) is recorded and the exercise dropped.
- A language below 80 % valid after the fixes is reported to the maintainer instead of run through.
- Nothing from a downloaded repository is executed outside the sandbox; each source sits in its own new directory under
  `/var/tmp/claude/poly-langs/`.
- The existing five languages (Rust, Go, JavaScript, Python, C#) must produce byte-identical prompts and test commands.

## Test conditions

(`python3 tools/hw_report.py`, 2026-11-19, abridged; the machine was in use by others, which does not matter for pass/fail
work: quiet check before the first reference run: warning, 1.28 GiB VRAM held by others, light swapping.)

- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads; RAM 60.5 GiB; Fedora Linux 44, kernel 7.3.4
- GPU: RX 9060 XT (not used by this work); other jobs shared the machine, CPU jobs ran through `gpu_queue.py --kind cpu`
- Podman rootless, 2 CPUs and 6 GB memory per test container, 240 s per exercise; engine commit `ce18c48` plus the working tree of this change

## Results

Counts. "Track" is the number of practice exercises in the source, "selected" what the rule keeps, "valid" what passed the
reference check. The valid lists are `/var/tmp/claude/poly-langs/valid-<lang>.txt` (outside the repository, like the data).

| Language | Source (licence), pinned commit | Track | Selected | Valid | Toolchain in the image | Image size |
|---|---|---|---|---|---|---|
| Java | Aider polyglot copy, Aider's own selection (MIT tracks; the copy has no licence field); upstream HEAD 7e0611e of 2024-12-22, the local copy is an unpacked zip without hash | 47 | 47 | **47** | Temurin JDK 21.0.13, Gradle 8.7 (sha256 checked), JUnit 5.10.0, AssertJ 3.25.1, rxjava 2.2.12, org.json 20190722; `FROM eclipse-temurin:21-jdk@sha256:c775c384...` | 655 MB |
| C++ | same Aider copy | 26 | 26 | **26** | Debian trixie GCC 14.2.0, CMake 3.31.6, Boost 1.83 (all), Catch2 inside each exercise; `FROM debian:trixie-slim@sha256:7378260c...` | 1.35 GB |
| PHP | exercism/php, MIT, `9443d5cac755` (2026-11-17) | 121 | 61 (66 in Aider's slugs, 5 deprecated) | **61** | PHP 8.4.27, PHPUnit 11.5.58 phar (sha256 checked); `FROM php:8.4-cli@sha256:37d80ce5...` | 555 MB |
| TypeScript | exercism/typescript, MIT, `2de55fcd08eb` (2026-11-07) | 111 | 60 (62, 2 deprecated) | **58** | Node 22.23.5, jest 29.7.0, babel-jest 29.7.0, @exercism/babel-preset-typescript 0.6.0, typescript 5.6.3 (lockfile `bench/polyglot/ts/package-lock.json`); `FROM node:22-slim@sha256:47725aca...` | 325 MB |
| Dart | exercism/dart, MIT, `c6bb6de05` (2026-11-15) | 80 | 35 (36, 1 deprecated) | **35** | Dart SDK 3.10.9, `test` 1.31.1 and 48 more packages in the baked pub cache; `FROM dart:3.10@sha256:7eed31e0...` | 855 MB |
| Bash | exercism/bash, MIT, `86938160670a` (2026-11-19) | 100 | 53 (54, 1 deprecated) | **52** | bats-core 1.10.0 (the version the track's CI installs, sha256 checked), Debian trixie bash 5.2, bc, gawk; `FROM debian:trixie-slim@sha256:7378260c...` | 90 MB |
| **Sum** | | | 282 | **280** | | |

Java, C++, PHP and Dart: every reference solution passes. TypeScript loses 2 and Bash 1 (below). All counts are within the expectation (Java 42 to 47, C++ 22 to 26, PHP 54 to 61, TypeScript 54 to 60, Dart 31 to 35; Bash was added later and not in the plan, 52 of 53 is 98 %).

Selection rule, recorded: slug of the track's practice exercises is in Aider's polyglot copy (100 distinct slugs over its six
languages) and the track does not mark it `deprecated` (`bench/polyglot/prepare.py`; deprecated hit: accumulate, beer-song,
binary, scale-generator, trinary). Flutter is not covered: Flutter UI tasks need their own benchmark (possible later item).

### Failures of the reference check on the first run, and what was done

Only the first round had failures; each was a harness or sandbox cause, none an exercise defect that we hide:

- `php/word-search`: the track's single example file holds three classes while the stub is split in three files; the
  reference check put it over one stub and the other stubs redeclared the classes. Fix in `tasks.reference_files`: the
  example goes into the solution file named like the exercise and the other stubs are emptied. Valid after the fix.
- `cpp/meetup`, `cpp/gigasecond`: CMake `find_package(Boost date_time)`; the image had `libboost-dev` only. Aider's image
  installs `libboost-all-dev`; the image now does too. Valid after rebuild.
- `typescript/crypto-square` (reference fails the track's `it.skip('54 character plaintext ...')`) and
  `typescript/pythagorean-triplet` (`it.skip('triplets for large number')`, timeout at 240 s): all skipped tests are enabled
  (maintainer's review: a model must pass the full test file), so these two reference solutions fail and the exercises
  are **dropped** (TypeScript valid 58 of 60). The third `it.skip` (robot-name) passes enabled. An earlier run of this
  file kept `it.skip` and had 60 valid; that was reversed on review.
- `bash/pythagorean-triplet`: with `RUN_HARD_TEST=true` (the track's switch for its one slow test) the 7th test does not
  finish within 240 s with the reference solution, so the exercise is **dropped** (Bash valid 52 of 53).
- C++ first needed the mount point `/work/<slug>`: CMake names the target after the directory, so mounting at `/work` made
  the target `work` (found with the smoke test, before the full run).
- Dart: the pub cache must be writable (`active_roots`); the command copies the baked cache to `/tmp` first.

Negative controls (stub without a solution must fail): Java 31 of 31 tests failed, C++ compile errors, PHP 16 errors,
TypeScript 31 of 31 failed, Dart compile errors, Bash (bowling) `not ok` for the stub; so the tests really run and skipped tests are enabled.

### Deviations from Aider's practice

- Aider has no PHP, TypeScript, Dart or Bash; those use the official tracks' own test commands (`phpunit`, jest, `dart
  test`, `bats`) and the track's own skip switches. TypeScript: implementation tests only (jest, types stripped by babel,
  no `tsc` and no `tstyche` type tests, which the track's online runner also adds).
- Java: `gradle --offline --no-daemon test` with the installed Gradle 8.7 instead of `./gradlew` (the wrapper downloads
  Gradle); same version the wrappers ask for. `@Disabled` removal is Aider's regex, applied on the staged copy.
- C++: Aider's `cpp-test.sh` command (cmake `-DEXERCISM_RUN_ALL_TESTS=1`, make); GCC 14 instead of Ubuntu 22.04's 11.
- Base images are pinned by digest (Aider's is not); the first four languages' images are still unpinned tags.
- The tests are pristine copies each time and the model can edit only the
  solution files (tests, `build.gradle`, `CMakeLists.txt`, `package.json`, `pubspec.yaml`, `composer.json`,
  `tsconfig.json` are protected by `tasks.PROTECTED`).
- Test output to the model is the last 6000 characters as before. Gradle prints every passed test, so a Java failure report
  can lose its first lines; not measured, open question below.
- Bash: `BATS_RUN_SKIPPED=true`, the track's own switch (every `skip` line is `[[ $BATS_RUN_SKIPPED == "true" ]] || skip`),
  and `RUN_HARD_TEST=true` for the slow test of pythagorean-triplet.

### Every language runs all tests

Skipped tests are enabled in the staged copy of the test files in the sandbox (`sandbox.enable_skipped_tests`, the bats
switches via environment), never in the files the model sees (the prompt shows only the solution stubs). Per language:
Java `@Disabled` (Aider's regex); TypeScript `xit`, `xtest`, `xdescribe`, `it.skip`, `test.skip`, `describe.skip`;
Dart `skip: true` (1063 occurrences in the track, no other `skip:` form exists); PHP `markTestSkipped` (1 file; no other
skip mechanism found by grep); Bash `BATS_RUN_SKIPPED` and `RUN_HARD_TEST`; JavaScript `xtest`, `xit`, `xdescribe` as before.
One exception: `bash/tournament` has one test guarded by `(( HAS_TTY )) || skip` ("ignore this test in CI": it needs an
interactive terminal on stdin); the containers run without a TTY, so that single test stays skipped, as in the track's CI.

### Harness change does not alter the first five languages

For all 296 exercises of Python, Go, Rust, JavaScript (Aider copy) and C# (exercism/csharp), a script monkeypatched
`subprocess.run`, called `sandbox.run_tests` and `sandbox.reference` and recorded the SHA-256 of `first_prompt` plus the full
`podman run` command (staged path normalised) before and after the change: the two dumps are byte-identical
(`cmp` equal, sha256 14d8b29e...). `tools/tests/test_polyglot_langs.py` freezes the old commands and the hash of the prompt
and retry texts. The default `LANGS` is unchanged, so the shuffled order of the polyglot suite is unchanged; each new
language is its own suite in `bench/agent` (`java`, `cpp`, `php`, `typescript`, `dart`, `bash`), like `csharp`.

### Not done / open

- No model run (as ordered). Per-language run size and decision rule belong in the plan of the model comparison.
- `bench/polyglot/model_job.sh` is not extended; the suites run through `bench/agent` or `run.py --langs <lang> --data
  /var/tmp/claude/poly-langs/data --valid /var/tmp/claude/poly-langs/valid-<lang>.txt`.
- Exercises name other upstream sources in `.meta/config.json`; not read one by one.
