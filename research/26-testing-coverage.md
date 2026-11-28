# Test audit and coverage gates (2026-10-28)

Agent research (web) plus an audit of the repository at this date, not re-checked.


## Part 1: audit (local, read only)

### 1.1 Inventory

| Test / check | Covers | Model (~55 GB) | GPU | CI today | Could run on hosted runner |
|---|---|---|---|---|---|
| ctest `gguf` (test_gguf, 207 l) | GGUF loader with hand-written files: values, tensors, split files, broken files | no | no | NO (engine not built in CI) | yes |
| ctest `suffix` (test_suffix, 95 l) | spec/suffix.cpp | no | no | no | yes |
| ctest `suffix_index` (81 l) | server/suffix.cpp | no | no | no | yes |
| ctest `draft_calib` (70 l) | spec/draft_calib.h | no | no | no | yes |
| ctest `chat` (test_chat, 187 l) | server chat render/parser (thinking, tool calls, truncation, Claude-Code request, tool_choice, prefill) | no | no | no | yes |
| ctest `install` (installer, 121 l) | installer core: recommendation for invented machines, config roundtrip | no | no | YES (job `installer`) | yes (already) |
| ctest `model_shapes` (ie-model) | tensor shapes of the real GGUF | yes | no | no | no (model) |
| ctest `cpu_experts` | cpu/experts vs reference on real expert rows | yes | no | no | no (could be done on a tiny synthetic tensor) |
| ctest `chat_vs_jinja` (py) | C++ chat render vs jinja2 template, uses model's template | yes (template read from GGUF) | no | no | with a small GGUF fixture holding the template: yes |
| ctest `chat_tokens` | chat tokenization | yes | no | no | needs small tokenizer fixture |
| ctest `dequant_ggml` | quant.cpp vs ggml (random blocks) | no | no | no | needs llama.cpp ggml build (CPU only, feasible, cached) |
| ctest `dequant_ggml_model`, `tokenizer_llamacpp` | same with real tensors / tokenizer vs llama-tokenize on corpus | yes | no | no | no (model + llama.cpp) |
| ctest `expert_bitexact` | CPU vs GPU experts bit exact | yes | yes | no | no |
| `scripts/golden.sh` (record/compare; variants cpu plan, prefill, dflash) | byte-exact greedy text of 3 prompts via ie-gpu-gen | yes | yes | no | no |
| `scripts/golden-server.sh` | both APIs, streaming, tools, sampling seed, thinking budget, prompt cache RAM/SSD | yes | yes | no | no |
| `scripts/isa-diff.py` | GPU machine code identical between builds | no (needs 2 builds) | no GPU but needs ROCm toolchain | no | no (multi-GB wheels; possible with cache but heavy) |
| `scripts/fuzz.sh` / tests/fuzz/fuzz_api.cpp (4 seeds) | libFuzzer+ASan+UBSan on API parsing, chat render, output parser | no | no | no (uses toolchain clang) | yes with system clang; short run (60 s) + seeds as regression |
| `tools/server_smoke.py`, `just test-gpu` | agent round against running server | yes | yes | no | no |
| `scripts/check-ram.sh` | memtester on hardware | n/a | n/a | no | no (hardware only, not a test of code) |
| pre-commit gates (file length, lizard, clang-format, ruff, shellcheck, gitleaks, actionlint) | style / size | no | no | YES (`gates` job) | already |
| clang-tidy (tools/gates/tidy.py) | lint | no | no | no (needs ROCm clang) | possible with apt clang-tidy, but version differs from pinned |
| bench-*.py, nll-compare.py, effort_tasks.py | measurements / quality | yes | yes | no | no |

Total unit-test code: ~1,000 lines C++ for ~40k lines of host+device source. CI executes exactly 1 ctest suite (installer).

Important blocker for hosted CI: `engine/CMakeLists.txt` sets `-march=znver5 -Wall -Wextra -Werror` globally, requires OpenMP, uses `io_uring` syscalls directly (linux headers only, fine), CMake >= 3.28, C++23. A hosted runner CPU is not Zen 5 (unverified which CPU the runner gets; Intel/AMD Zen 3/4 typical) so znver5 binaries (AVX-512 VNNI/BF16 etc.) will likely die with SIGILL. Compiling works with GCC 14 (znver5 supported since GCC 14), running does not. A CI build needs a cache variable (e.g. `IE_MARCH=x86-64-v3` for the CI/unit-test configuration) - needs a maintainer decision, as the reference build must not change. Whether code paths under test use AVX-512 intrinsics unconditionally (cpu/experts.cpp, detmath.h) is unverified; if so, those tests must stay local or use a portable fallback.

### 1.2 Map of source areas vs tests (lines are `cat dir/*`, approximate)

| Area | Lines | Tests | Gap |
|---|---|---|---|
| engine/model (gguf, quant, tokenizer, qwen4exp, progress) | 5.1k | gguf (good); quant only vs ggml (needs llama.cpp build for random-block test); tokenizer only vs llama-tokenize with model; qwen4exp only via model_shapes | tokenizer has no model-free test (BPE, unicode tables, special tokens, invalid UTF-8); quant has no self-contained golden-vector test; progress.cpp none |
| engine/cpu (ref, dflash, experts) | 1.7k | cpu_experts (model); ref.cpp, dflash.cpp none | ref/dflash untested except through GPU cross-check |
| engine/server | 4.2k | chat/render/parser (test_chat + fuzz + jinja), suffix | api.cpp/api_openai/api_anthropic request parsing only fuzzed (no assertions on valid/invalid requests, error JSON, error codes); http_util, routes_*, request_options, sampling, job, prompt_cache, system_status, engine*, generator, prefill, draft, verify: no unit test. Only exercised by golden-server (GPU+model) |
| engine/spec | 0.35k | suffix, draft_calib | fine |
| engine/backend (budget.h, env.h) | 0.25k | none | env parsing/budget logic is pure and testable |
| engine/io (uring) | 0.17k | none | error paths (kernel without io_uring, EAGAIN) untested |
| engine/tools | 4.8k | none (tools are executables, but gpu-gen/teacher/spec report logic) | low priority; some are really library code |
| engine/gpu | 15.7k (93 files) | expert_bitexact; golden; isa-diff | no kernel-level tests (ie-gemm-check, ie-expert-check are tools not ctest); everything local-only |
| installer | ~? | test_install (one file, 121 l) | disk.cpp, hw.cpp parsing (lspci/sysfs/proc parsing), pkgs, steps, report, config error paths: partly covered, unmeasured; installer already in CI |
| Python scripts/tools | ~2.1k + gates | none (no pytest, no unittest); gate scripts have no tests | tools/gates/*.py (file_length, functions, tidy) are enforcement code with zero tests; gen_tables.py, get_model.py, gguf_info.py parse files; isa-diff.py parses ISA text |
| Bash | fuzz.sh, golden*.sh, setup.sh, ie.sh, ie-service.sh | shellcheck only | no bats tests; low priority |
| Error paths overall | | broken GGUF files tested; truncated model output tested | HTTP malformed/oversize/slow clients, OOM / RAM limit, partial reads, CLI argument errors: none |

### 1.3 What can run on hosted CI without GPU and without the model
Immediately (after the -march issue is solved and a CPU-only configure target exists, e.g. `-DIE_CPU_ONLY=ON` that skips `ie-ref`/tools needing the model): `gguf`, `suffix`, `suffix_index`, `draft_calib`, `chat`, `install` (already). With small fixtures (to create): tokenizer (tiny BPE vocab GGUF, few KB, generated by a test writer like test_gguf's Writer), `chat_vs_jinja` with a template-only GGUF, quant golden vectors, cpu_experts on a synthetic tensor, API request-parsing tests, http_util, request_options, sampling (seeded, deterministic), prompt_cache serialization (RAM/SSD stash on temp dir), budget/env. Sanitizer job (ASan+UBSan, GCC) over all of these. Short libFuzzer run (system clang) for fuzz_api with seeds as regression. Python: pytest for tools/gates, gen_tables.py (it has a "header matches committed tables" property: easy CI check), gguf_info.py on the synthetic GGUF, isa-diff.py parser on a saved text sample. The `dequant_ggml` random-block test could use a cached ggml-base build (CPU only) - optional.
Cannot: golden*, expert_bitexact, anything with the real model/GPU, isa-diff (toolchain), clang-tidy (pinned ROCm clang).

## Part 2: coverage tooling (web research; "(unverified)" marks what I could not confirm)

### 2.1 C++ host coverage
- gcov/lcov/gcovr (GCC): matches the project compiler, works with the CMake/Ninja build. gcovr's branch numbers are CFG arcs from `-fprofile-arcs`, not source decisions; its FAQ warns decision coverage is buggy with optimization -> build coverage at -O0/-O1 (https://gcovr.com/en/stable/faq.html, fetched via search 2026-10-28). gcovr outputs Cobertura/LCOV/JSON/HTML/summary and has `--fail-under-line/--fail-under-branch`.
- LLVM source-based (`-fprofile-instr-generate -fcoverage-mapping`, `llvm-cov`): exact region/branch coverage tied to source constructs, needs clang (the ROCm toolchain's clang or apt clang), export to LCOV via `llvm-cov export -format=lcov`. A 2025 differential-testing paper (DebCovDiff, https://mir.cs.illinois.edu/marinov/publications/ZhangETAL25DebCovDiff.pdf) found gcov and llvm-cov disagree, including branch counts, so numbers from the two are not comparable; pick one and keep it.
- Line vs branch: with exceptions/templates/`-Werror` code, branch coverage in GCC is noisy (std:: internals). Use LINE coverage as the gated metric, report branch informationally (use `--exclude-throw-branches --exclude-unreachable-branches` in gcovr).
- HIP DEVICE code: previously effectively not measurable; however LLVM added a documented source-based workflow for HIP device code on 2026-06-30 (compile with `-fprofile-instr-generate -fcoverage-mapping`, extract device ELF from `.hip_fatbin`, `clang-offload-bundler`, `llvm-profdata`, `llvm-cov`) - https://llvm.googlesource.com/llvm-project/+/fb83b9b663e02a6db0e104faa6d14642a8d95e7f (fetched 2026-10-28). Whether ROCm 10.1 / LLVM 24 in the repo's wheels contains it, and whether it works with the megakernel (persistent, spin-waits, 100 ms yield, heavy instrumentation perturbing registers/occupancy and the GPU-hang risk in CLAUDE memory) is UNVERIFIED. It also needs the GPU and model. Decision: scope = HOST code only for gates; device coverage at most an experiment on the reference machine, never a gate, and never in the golden/benchmark build.
- Scope definition: `engine/{model,cpu,server,spec,backend,io}`, `engine/tools/*.cpp` (excluded or separate), `installer/core`, exclude `engine/gpu`, `engine/third_party`, `tests/`, `*.hip`, FTXUI, `tui*.cpp` (UI), generated tables.

### 2.2 Python
coverage.py + pytest-cov (https://pytest-cov.readthedocs.io) is standard; `coverage xml` -> Cobertura; `--fail-under`. Scope: tools/gates, tools/*.py, scripts/*.py that are importable. diff-cover (https://github.com/Bachmann1234/diff_cover) accepts Cobertura/LCOV XML and gives patch coverage plus `--fail-under`, no external service (my knowledge; not re-fetched).

### 2.3 Services
- Codecov: public repos free (Codecov blog/third-party trackers; paid tier prices disagree - check https://about.codecov.io/pricing, unverified). Fork PRs: tokenless upload works for public repos on unprotected (fork) branches; codecov-action v4+ adds the `fork:` prefix automatically; a token is required for protected branches (main) unless the org disables "token required" in Global Upload Token (option needs codecov-action v5+). Existing orgs default to "Required", new orgs "Not required" (https://docs.codecov.com/docs/codecov-tokens, fetched 2026-10-28). So: PR uploads from forks work without secrets; main uploads use a repo secret from a push workflow. Statuses: `codecov/patch` (changed lines only) and `codecov/project` (overall vs base, `target: auto` + `threshold`), https://docs.codecov.com/docs/commit-status. Needs the Codecov GitHub App (third party with repo access) and its upload path must be allowed in the workflow (a third-party action; pin by SHA like the existing workflow). vLLM's `.github/codecov.yml` only sets `require_ci_to_pass: false` and path `fixes` - no thresholds (fetched 2026-10-28).
- Coveralls: similar, free for public; no stronger case. SonarQube Cloud: imports coverage reports, quality gate on "new code" coverage, free for public projects (unverified details); it is mainly a second analysis tool; skip unless wanted for static analysis anyway.
- No service: gcovr/diff-cover in the workflow: no third-party app, nothing leaves GitHub, works identically for fork PRs (plain `pull_request`, read-only token), report in `$GITHUB_STEP_SUMMARY`, artifact with HTML, a failing step = failing check. Downside: no history graphs, no PR comment (a comment from a fork PR needs write token -> `workflow_run` split; avoid, use summary).

### 2.4 Thresholds that work in PRs
- Patch/diff coverage (new/changed lines) is the enforceable one: it does not punish legacy code and prevents erosion. Global floor alone is a blunt tool (a big unrelated deletion changes it); ratchet (never drop more than e.g. 0.5 pp vs main) needs a stored baseline (Codecov `target: auto` does this; self-made: commit `coverage-baseline.json` updated by a maintainer-only workflow, or compare against a main-branch artifact).
- Per-directory: only for a few areas where it matters (server/api*, model/gguf, spec), via path-specific flags/components (Codecov components) or separate gcovr invocations.
- Starting numbers (my judgment, not sourced): measure first; set global floor = measured value minus 1 pp (ratchet), patch >= 80 % lines on changed host-code lines, with an escape through PR label/justification only for lines in untestable-by-design code (I/O syscalls, hardware reads) using explicit `// GCOVR_EXCL_LINE`-style markers that the reviewer sees. Given AGENTS.md says "no exception lists", prefer excluding by directory scope (documented in one config file) over per-line pragmas.
- How others handle it: LLVM publishes a nightly coverage report (https://llvm.org/reports/coverage/) with no PR gate; vLLM uses Codecov without thresholds (above); llama.cpp: I found no coverage tracking and it uses its own ggml-org/ci on dedicated machines (search result, weak/unverified); Ollama, ROCm libraries: not verified (no data found). So gating on patch coverage would be stricter than the LLM-engine peers; that is fine but keep the first phase informational.

### 2.5 Enforcement on a free account
- Rulesets (and required status checks) are available for public repos on GitHub Free (https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets, per search 2026-10-28; private repos need Pro/Team). Required checks are matched by check NAME, not workflow/event, so keep job names unique and stable. Rulesets support "evaluate" mode (status reported, not enforced): use that for the rollout. Team plan is not needed for a public repo.
- Fork PRs: `pull_request` workflows from forks get a read-only GITHUB_TOKEN and no secrets; first-time contributors need maintainer approval to run workflows (repo setting "Require approval for all outside collaborators" - recommended). Coverage via plain `pull_request` with gcovr needs neither.
- GPU-less trust problem: the CPU gate says nothing about kernels or end-to-end. Make that explicit: name the check `cpu-coverage (host code, no GPU)`; PR template has a checklist "golden / isa-diff run by maintainer"; a maintainer-run local script posts a commit status `gpu-golden` (via `gh api` with the maintainer token) which is a required check in the ruleset for merge (the maintainer sets it after running `just lint`+golden). That keeps GPU checks outside GitHub's runners and unforgeable by forks (only users with write access can set statuses). Optional: label `gpu-verified`.
- Self-hosted GPU runner on a PUBLIC repo: GitHub says self-hosted runners should "almost never be used for public repositories" because anyone can open a PR that runs code on it; environment approvals do not isolate the runner; destroying runners after jobs is not guaranteed; JIT runners (one job) are stronger (https://docs.github.com/en/actions/reference/security/secure-use, fetched 2026-10-28). `pull_request_target` runs with secrets/write token in the base context; checking out the PR head there is the "pwn request" (https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/): label gates ("safe to test") have a TOCTOU race (push after approval); use the two-workflow split (unprivileged build, privileged `workflow_run`) and never execute PR content in the privileged one. Specific for this project: the runner is the maintainer's desktop with the daily-driver GPU, 55 GB model, SSH keys, and the documented fact that a GPU hang resets the desktop. Recommendation: NOT advisable on fork PRs, and not on the desktop at all. If ever: separate machine/VM, JIT runner, only `push` to main and `workflow_dispatch` by maintainers, running code already merged; PRs tested by the maintainer locally on the PR branch after reading the diff.

### 2.6 Mutation testing
Mull (C++, LLVM based, v0.34.0 listed 2026-05-12 per a search listing; incremental on git diff) and mutmut (Python; not verified in search). Cost is high (rebuild + rerun per mutant; needs clang build and a stable fast test suite). Not worth it now: first get line coverage with real assertions. Reconsider later, narrowly, for pure-logic modules with fast tests (gguf parser, chat parser, suffix, draft_calib, installer plan) as an occasional maintainer-run job, never a PR gate. mutmut on tools/gates: only if those scripts grow.

## Recommendation

Tool: gcovr (GCC, -O0, `--coverage`) for C++ host code + coverage.py/pytest-cov for Python + diff-cover for patch coverage; all inside GitHub Actions, summary in the job summary, no external service in phase 1. Optional phase 3: add Codecov for history/badge only (tokenless fork upload works; main upload with secret) - it is additive, not required.
Thresholds: patch (changed host lines) >= 80 % line coverage, enforced from phase 2; global line floor = measured baseline minus 1 pp, ratchet upward by hand when it rises by >= 2 pp; branch coverage reported only. Per-dir floors later for server/api*, model/gguf*, spec. GPU/device code excluded from all numbers and stated so in docs/contribute/verification.md.
Enforced: required checks via ruleset on main (Free plan OK): `gates`, `installer`, `engine-cpu-tests` (unit tests + sanitizers), `cpu-coverage` (patch gate), plus maintainer-set status `gpu-golden`. Evaluate mode first.
Phased rollout:
1. (prereq) CPU-only CMake configuration: `IE_CPU_ONLY`/`IE_MARCH` option for CI that avoids znver5 and ROCm, builds `ie_model ie_spec ie_chat_lib` and the 5 model-free tests; add job `engine-cpu-tests` (GCC 14, ctest, then ASan+UBSan variant). No coverage yet.
2. Add the coverage job (gcovr + job summary, informational, record the baseline numbers per directory; Python pytest for tools/gates).
3. Turn on patch gate 80 % and the global ratchet; ruleset from evaluate to active; PR template checkbox for GPU verification and the `gpu-golden` status.
4. Short fuzz job (60 s, system clang, seeds regression), nightly longer fuzz on a schedule; optional Codecov; optional mull experiment locally.
Not recommended: GPU self-hosted runner for PRs; device-code coverage as a gate.

First tests to add (largest gaps, all model-free, CPU only):
1. API request parsing (api.cpp, api_openai.cpp, api_anthropic.cpp): valid/invalid bodies, error JSON and status, tool schemas, limits (reuse fuzz seeds as unit cases).
2. Tokenizer with a tiny synthetic GGUF vocab (encode/decode round trip, specials, invalid UTF-8, merges) - closes the biggest model-bound gap; same fixture for `chat_vs_jinja` and `chat_tokens`.
3. http_util + request_options + sampling (seeded determinism, top-p/top-k edge cases, temperature 0) + job lifecycle/cancellation.
4. prompt_cache RAM/SSD stash round trip and corruption handling on a temp dir.
5. quant.cpp golden vectors (hard-coded blocks -> expected floats) so it needs no llama.cpp; cpu_experts on a synthetic tensor.
6. backend/env.h and budget.h (env parsing, VRAM budget arithmetic), io/uring error paths (skip if kernel lacks io_uring).
7. Installer: hw.cpp/disk.cpp parsers fed with fixture text (lspci, /proc/meminfo, sysfs files), steps/report/config error paths.
8. Python: pytest for tools/gates/{file_length,functions,tidy parsing}, gen_tables.py consistency check, gguf_info.py on the synthetic GGUF, isa-diff.py parser on a saved sample.
9. Promote existing fuzz entry to a CI smoke run; add a second fuzz target for the GGUF loader (model/gguf.cpp parses untrusted files - high value).

Unverified/open: runner CPU vs znver5 SIGILL; AVX-512 unconditional use in cpu/; whether ROCm 10.1 clang has HIP device coverage; Ollama / ROCm-library coverage practice; Codecov current paid prices; SonarQube Cloud free tier details; diff-cover LCOV acceptance details (believed fine).
