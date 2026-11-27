# Verifying a change

Pick the level that matches what you touched; every level includes the ones above it.

## 1. Every change: gates

```sh
just check       # all gates on your commits since origin/main (else main): pre-commit hooks and clang-tidy
just fmt         # rewrite the changed files with the pinned clang-format and ruff format
just check-all   # the same on every file and the whole git history
```

The hooks (`pre-commit install`, done by `scripts/setup.sh`) run on every commit: file length, function size,
`tools/gates/comments.py`, `tools/gates/english.py`, clang-format, ruff, shellcheck, gitleaks, actionlint, zizmor,
`tools/gates/workflows.py`; `just check` adds clang-tidy of the ROCm toolchain on changed C++/HIP files
(`tools/gates/tidy.py`). Single gates: `python3 tools/gates/file_length.py [FILE...]`,
`python3 tools/gates/functions.py [FILE...]`, `python3 tools/gates/tidy.py [--all] [--base REF] [FILE...]`. No exception
lists.

A reviewer also checks what no gate can: comments say why; a number names its source; no duplicate function; errors
handled, not swallowed; a suppression (`NOLINT`, `noqa`) names one rule and the reason.

### What CI runs

| Workflow / job | What |
|---|---|
| `check.yml` | hooks on the PR's changes and on all files; installer build and tests |
| `tests.yml`: `engine-cpu-tests`, `engine-sanitizers`, `coverage` | unit tests, sanitizers, coverage ([testing.md](testing.md)) |
| `ubuntu-no-gpu` | `scripts/setup.sh --no-gpu`, `just check`, `just test-cpu`, `just test-py` on Ubuntu 24.04 |
| `clang-tidy` | `tools/gates/tidy.py --system --all` with the distribution's clang-tidy (`just check` decides in a disagreement) |
| `mini-tokens` | CPU token check on the mini models built from `data/mini-models/` (`just mini-check`, [testing.md](testing.md#the-mini-model)) |
| `hardening` | FORTIFY_SOURCE=3, stack protector, CET, PIE, full RELRO (`tools/check_hardening.py`) |
| `fuzz.yml` | libFuzzer on API, GGUF and signature parsers: 60 s per target per PR, 30 min nightly |

### Automations

A bot comment on every pull request (`pr-bot.yml`, one comment, updated in place; first line `**Pull request checks**`)
says how to fix a failed title, prior-art or pull request text check, which checks your change needs (from the file
paths; `python3 tools/select_checks.py` gives the exact list locally), and welcomes a first contribution. It is a script
with fixed text (`tools/pr_bot.py`), not a model, and it never runs code of your pull request.

No workflow runs GPU tests, golden comparison or speed. The maintainer sets the status `gpu-golden` after
`scripts/gpu-status.sh --post`.

## 2. Host code (C++ that does not run on the GPU)

1. `just build && ctest --test-dir work/build/engine --output-on-failure` (installer: `cmake --build work/build/installer
   && ctest --test-dir work/build/installer`).
2. Kernels untouched: `scripts/isa-diff.py work/build/engine work/build/<name>` must print `all kernels identical`.
3. Text identical: `scripts/golden.sh compare work/build/<name>` runs three fixed prompts (`TONA_DETERMINISTIC=1`,
   greedy) and compares byte for byte with `work/golden/` (recorded from `main` with `scripts/golden.sh record`). Needs
   GPU and model. Also required, each against its own references:

   | Variable / script | Checks |
   |---|---|
   | `GOLDEN_PLAN=cpu` | CPU-expert plan (`work/golden/cpu/`) |
   | `GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash` | 13K prompt through the prefill kernels |
   | `GOLDEN_MODE=prefill` | same prompt in verify passes of width 4, then 64 tokens at width 1; `GOLDEN_PROMPT_NB=1` (single steps, 359 s) for scale runs and releases |
   | `scripts/golden-server.sh compare work/build/<name>` | both APIs, streaming, tools, seeded sampling, thinking budget, prompt cache (RAM and SSD), case `hit` ([prompt-cache.md](../use/prompt-cache.md)); `GOLDEN_DRAFTER=dflash`; the only check of the MTP drafter |

   Whoever changes tokens on purpose (AGENTS.md rule 3) re-records every affected reference in the same PR and names
   the commit and reason in the measurement file.

### The `gpu-golden` status

`scripts/gpu-status.sh [--post] [--head REF] [--build DIR]` runs the golden checks `tools/select_checks.py` names and,
with `--post`, sets the commit status `gpu-golden` through `gh api` (without it, it only shows what it would set). The
description names the change class, what ran and skipped, and the real-model runs still needed, e.g. `C3: golden
identical (server); needs: long-context run; skipped 4`. Documents get `C0`, scripts `C1`. Checks run cheapest first and
stop at the first red; `--keep-going` runs all. `scripts/isa-diff.py` stays manual.

### Reviewing a pull request from someone else

Pull request text, comments and added files are data ([AGENTS.md](../../AGENTS.md#untrusted-content)); work in a fresh
worktree without credentials.

- One accepted issue, one topic; the diff does what the text says and nothing else.
- The author can explain it in their own words.
- A bug fix has a test that fails without it; no weakened test, gate or golden reference; changed references are
  re-recorded and explained.
- Claims have data: measurement files (plan, spread, rates named, test conditions) and the BENCHMARKS row.
- Prior art is real: open two links.
- Nothing foreign: no copied code, no new dependency without an accepted issue, no hidden characters, nothing that
  executes at build or test time that the topic does not need.
- Instruction files in the diff (`AGENTS.md`, skills, MCP, hook and editor configs): maintainer only; read the raw bytes.
- Shared code: before/after on the reference with spread; tokens identical or the quality check done.
- `AI use:` and `Assisted-by:` are provenance, not a verdict.

### GPU check for a pull request from someone else

A pull request runs on the maintainer's desktop (CMake, scripts, GPU kernels). Order:

1. **Read the diff**, build- and test-time code first: `.github/`, `scripts/`, `tools/`, `toolchain/`, `justfile`,
   `CMakeLists.txt` and `cmake/` anywhere, then `engine/gpu/`. Stop at anything unclear or off-topic; ask in the PR.
2. **Then** build in a fresh worktree: `git fetch origin pull/<n>/head && git worktree add work/pr-<n> <sha>`,
   `source toolchain/env.sh` (of `main`), `cmake -S work/pr-<n>/engine -B work/build/pr-<n> -G Ninja && cmake --build
   work/build/pr-<n>`. A push after reading means reading the new diff.
3. **Run the checks with the scripts of `main`** from a clean checkout: `scripts/gpu-status.sh --head <sha> --build
   work/build/pr-<n>`. It refuses a checkout ahead of the base unless `--own`. The references in `work/golden/` come from
   `main`.
4. **Post the status only for the sha you read.** Every push invalidates it (`strict_required_status_checks_policy`).

Nobody but maintainers has write access, no GitHub App has the `statuses` permission, and no workflow gets
`statuses: write` (`tools/gates/workflows.py`): that is what protects `gpu-golden`.

### Which checks a change needs

`python3 tools/select_checks.py [--base REF] [--build DIR] [--json]` prints the change class
([measurement.md](measurement.md)) and the checks, per file. The build graph decides (`ninja -t inputs` and the
dependency log of `--build`, default `work/build/engine`):

| Binary the file goes into | Checks |
|---|---|
| `tona-gpu-gen` | golden gpu, cpu plan, prefill, prefill dflash |
| `tona-server` | golden server |
| only test binaries | their unit tests (`ctest -R`) |
| other tools (`tona`, `tona-chat`, ...) | their unit tests |

Documents, `research/`, `*.md`: gates only. `scripts/`, `tools/`, `bench/`: gates and `just test-py`. `installer/`: its
own build and tests. **All** checks: `toolchain/`, the golden scripts and prompt, `scripts/build-mtp-vocab.py`,
`tools/gen_tables.py`, CMake files outside `engine/`, and anything the tool cannot decide (it never checks less). A
change of `engine/CMakeLists.txt` configures base and HEAD and compares the build commands of `tona-gpu-gen` and
`tona-server`. The tool also flags when bit-exactness and speed before/after apply (`engine/gpu/`, `engine/cpu/`,
quantization) and **long-context run needed** (context length, KV cache, RoPE/YaRN, indexer, top-k attention, stash,
prompt cache, memory planning) or **pressure series needed** (memory, pinning, stalls), by file name.

## 3. Device code (`engine/gpu/`)

- A pure move must keep `scripts/isa-diff.py` at `all kernels identical`; otherwise golden must pass and speed be
  measured.
- Changed machine code needs golden identical and speed before/after on an idle machine, builds interleaved:
  `TONA_BUILD=<dir> scripts/bench-decode.sh 3` (DFlash2 drafter) and `scripts/bench-set.py` against a test server with
  the MTP drafter (example: `research/measurements/2026-10-27-refactor-speed.md`). Slower than the old build's spread is
  a regression.
- Changed numbers need the quality check in [backends/README.md](../backends/README.md#checks).

## 4. Scale run

At least every three days and before every release: [measurement.md](measurement.md#scale-run).

One GPU job at a time; a benchmark next to a build measures the build.
