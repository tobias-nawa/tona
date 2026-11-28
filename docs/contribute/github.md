# GitHub setup (maintainers)

What protects the repository on GitHub and what to check by hand.

## Files in the repository

| File | What it does |
|---|---|
| `.github/workflows/check.yml` | code gates (pre-commit incl. actionlint, zizmor) on changes and all files, installer build |
| `.github/workflows/pr-bot.yml`, `tools/pr_bot.py` | one comment per pull request, updated in place: how to fix the failed text gates, which checks the change needs, a welcome for first contributions. Fixed text, no model |
| `.github/workflows/codeql.yml`, `.github/codeql/codeql-config.yml` | CodeQL for C/C++, Python, workflows; every PR, `main`, weekly; `engine/third_party/` ignored |
| `.github/workflows/scorecard.yml` | OpenSSF Scorecard (SARIF, public score) |
| `.github/workflows/release.yml` | tag `v*`: builds, attests, creates a **draft**; no secrets ([docs/maintain/release.md](../maintain/release.md)) |
| `.github/dependabot.yml` | weekly grouped updates for `github-actions`, `pip`, `pre-commit`, 7-day cooldown |
| `.github/CODEOWNERS` | maintainer owns instruction files, CI, gates, updater, release scripts, reference backend; a backend owner owns the folder but not its `AGENTS.md` |
| `SECURITY.md` | private reporting first, e-mail second; latest release only; reply within 7 days |
| `scripts/github-setup.sh` | settings that exist only in the API |

Every workflow (zizmor): `permissions: {}` at the top, least privilege per job; `persist-credentials: false`; actions
pinned to a full SHA with the version as a comment; no `workflow_run` with fork code; `pull_request_target` only in `pr-bot.yml`: it must comment on fork pull requests, so it checks out the base branch
only, reads title, text and file list through `gh api` as data, and has `pull-requests: write` and `contents: read` and nothing else
([Preventing pwn requests](https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/); zizmor `dangerous-triggers` is ignored there with the reason in the file;
`tools/gates/workflows.py` refuses the head of the pull request in any such workflow).
`tools/gates/workflows.py` also refuses `statuses: write`, `self-hosted` and `github.event.*` in `run:` (use `env:`).
Dependabot does not see the tools in `additional_dependencies` of `.pre-commit-config.yaml` (clang-format, lizard,
gitleaks): update them by hand with the files they reformat.

## Protecting an agent-first repository

Each gate is a hook (so also in `just lint` and `gates`) or a job of `check.yml`; its header says what it misses.

| Gate | Where | Refuses |
|---|---|---|
| `tools/gates/hidden_chars.py` | hook | invisible characters, HTML comments, hidden markup in `AGENTS.md`, `CLAUDE.md`, skills, templates, `data/modeldb/`, bench prompts |
| `tools/gates/agent_config.py` | job `agent-config` | changes to `.mcp.json`, `.claude/`, `.cursor/`, `.vscode/`, `.devcontainer/`, hooks, workflows, `AGENTS.md`, skills by anyone whose `author_association` is not `OWNER` (Dependabot may bump pins) |
| `tools/gates/codeowners.py` | hook | a protected path not owned by the maintainer alone |
| `tools/gates/commit_trailers.py` | hook `commit-msg`, job `commits` | `Assisted-by:` not `<agent>:<model>`; `Co-Authored-By`/`Signed-off-by` naming a tool; `Claude-Session:`, "Generated with", `Generated-by:` |
| `tools/gates/pr_text.py` | job `pr-text` | a PR text without exactly one `AI use:` line (`none`, or tools with an `Assisted-by:` line each) |
| `tools/gates/dependencies.py` | hook | unpinned requirement (no `==` and hash), hook `rev` that is a branch, action without SHA, `FetchContent` without SHA or hash, model without revision and `sha256`, any name missing from [licenses.md](licenses.md) |
| `tools/gates/prior_art.py --online` | job `prior-art` | impossible arXiv numbers and GitHub links; 404/410 links in CI |
| `tools/gates/workflows.py` | hook | also: actions, API hosts, key variables and tools that send repository or PR text to an LLM service |

With one maintainer `REQUIRED_APPROVALS` stays 0 and `agent-config` holds; with a second, set `REQUIRED_APPROVALS=1`
(turns on `require_code_owner_review`).

**Rights statement:** no DCO; one paragraph of the PR template. Only squash merges; the repository copies PR title and
text into the commit (`squash_merge_commit_title: PR_TITLE`, `squash_merge_commit_message: PR_BODY`), so the statement
and `Assisted-by:` lines stay in history.

**Open PR limit (by hand):** Settings, Moderation options, Interaction limits, "Limit open pull requests from users
without write access" = 1 ([CONTRIBUTING.md](../../CONTRIBUTING.md)), trusted contributors on the bypass list; no API field
known (unchecked), so `scripts/github-setup.sh` cannot set it.

## Dependencies

`tools/gates/dependencies.py` accepts a new package, hook, action, `FetchContent` entry or model only when pinned and
listed in [licenses.md](licenses.md). Pins: hash for pip files (`pip-compile --generate-hashes`, header of
`tools/gates/requirements.in`), full SHA for actions and fetched sources, release tag for pre-commit `rev`, SHA and
`sha256` for models. A human looks up every package name an agent proposes.

## `scripts/github-setup.sh OWNER/REPO [--apply]`

Prints the calls; `--apply` needs `gh auth login` as admin and can rerun any time. `REQUIRED_APPROVALS=1` with a second
maintainer. Required checks are listed at the top of the script (`tools/tests/test_required_checks.py` fails on a name
that is no workflow job).

Required checks: `check-ok` and `tests-ok`, green when every job of their workflow passed or was skipped. Heavy jobs
run only when `scripts/ci-changes.sh` finds C++, Python or CI files; a README change runs the gates only. Push to
`main`, nightly and manual runs run everything.

| Setting | Value |
|---|---|
| Ruleset `main` | PR only, squash only, linear history, no force push or deletion, threads resolved, checks `check-ok`, `tests-ok`, status `gpu-golden`, no CodeQL alerts of severity high or above (rule `code_scanning`) |
| Ruleset `release tags` | `v*`: only admins create, nobody moves or deletes |
| Secret scanning, push protection, Dependabot alerts and security updates, private vulnerability reporting | on |
| Immutable releases | on (draft, attach assets, publish) |
| Actions | only SHA-pinned actions; read-only token; workflows cannot approve PRs; maintainer approves every outside contributor's runs |

`gpu-golden` is a status a maintainer posts after `scripts/golden.sh compare`. Branches must be up to date, so after
every rebase the new head needs it again.

## First day

1. **Create** the empty public repository `tobias-nawa/tona` (no README, licence or `.gitignore`). Turn on Discussions.
2. **Push** `main`: `git remote add origin git@github.com:tobias-nawa/tona.git && git push -u origin main` (starts `check`,
   `tests`, `codeql`, `scorecard`).
3. **Dry run** `scripts/github-setup.sh tobias-nawa/tona`, then `--apply` with `ENFORCEMENT=evaluate` (may need a paid
   plan, unchecked; if refused apply `active` and test with step 5 before pushing anything else).
4. **Wait for the first runs** on `main`: `gates`, `installer`, `engine-cpu-tests`, `engine-sanitizers`, `coverage`,
   `Analyze (c-cpp)`, `Analyze (python)`, `Analyze (actions)`, Scorecard (`prior-art`, `pr-text`, `commits`,
   `agent-config` only on PRs). A required check that never reported blocks PRs.
5. **Test PR** from a second account or fork: look at Insights, Rules (every rule satisfiable); post `gpu-golden` with
   `scripts/gpu-status.sh`; workflows must wait for approval and run without secrets.
6. **Switch both rulesets to Active** (rerun without `ENFORCEMENT`).
7. **Notifications** (account settings, github.com/settings/notifications, Actions): "Send notifications for failed
   workflows only" (wording unchecked).

## Check by hand after the first run

- Settings, Rules: both rulesets active; admin role id 5 in the tag ruleset.
- Settings, Code security: secret scanning, push protection, Dependabot alerts and security updates, private reporting.
- Settings, Actions, General: approval for all external contributors; read-only workflow permissions; no PR approval by
  Actions; actions pinned to a full SHA. Settings, Releases: immutable.
- Settings, General, Pull Requests: only squash merging, default message "Pull request title and description".
- Security, Code scanning: CodeQL for all three languages and Scorecard results (fix what it lists); the `code_scanning`
  rule appears as required.
- `securityscorecards.dev/viewer/?uri=github.com/OWNER/REPO` shows a score; then add the badge to `README.md`:
  `[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/OWNER/REPO/badge)](https://securityscorecards.dev/viewer/?uri=github.com/OWNER/REPO)`.

## What CodeQL does not see

GPU code (`.hip`, `.cu` are no supported extensions, so `engine/gpu/*.hip` is unanalysed; device code in `.h` is read as
C++), generated code, `engine/third_party/`, logic and numerics (golden and quality checks,
[verification.md](verification.md)). clang-tidy and the fuzzers (`scripts/fuzz.sh`) overlap with it. Not used: Semgrep,
cppcheck, Codacy, DeepSource, GitHub Advanced Security (`research/25-security-scanners.md`); gitleaks replaces paid
secret-scanning extras.
