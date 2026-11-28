# Maintainer runbooks

Fixed procedures for one maintainer. Contributors: [CONTRIBUTING.md](../../CONTRIBUTING.md). Settings:
[github.md](../contribute/github.md); checks: [verification.md](../contribute/verification.md), [measurement.md](../contribute/measurement.md). Time budgets are
estimates.

No bot reads the text of issues and pull requests from outsiders (prompt injection:
[Comment and Control](https://labs.cloudsecurityalliance.org/research/csa-research-note-comment-control-github-prompt-injection-20/)):
no AI reviewer, vouch system or SBOM.

**Automations** (`.github/workflows/`; each reads outsider text only through a parser in the repository, never in a shell
line):

- `labels.yml`, job `author`: from the AI-use box of an issue form it sets `author-verified` (ticked) or
  `author-unverified` (unticked) when an issue is opened or edited, and switches back when the box changes. The first
  time it sets `author-unverified` it writes one comment. Forms without the box (backend proposal, blank issue) get
  neither label. Parser: `tools/author_verified.py`.
- `labels.yml`, job `sync`: on push to main that touches `.github/labels.tsv`, and by hand, it creates or updates the
  labels from that file (`scripts/sync-labels.sh`, also used by `scripts/github-setup.sh`). It is additive: a label that
  is not in the file is never deleted, so removing one from the file means deleting it in the GitHub UI.
- `issue-check.yml` (`tools/issue_bot.py`): an empty or `_No response_` required section of the form sets `needs-info`
  and lists the sections in one comment; filled in later, it updates the comment and removes the label only if it set
  it. No model reads the text. The AI-use box is left to the author labels.
- `issue-reminder.yml`: `needs-info` and no comment from the author for 14 days: one reminder; after 28 days: closed as
  not planned, reopenable. Daily; run by hand with `dry-run` to see the plan. Issues only, not pull requests.
- `pr-bot.yml`: keeps one comment per pull request (text gates, needed checks, welcome); it is a script with fixed
  text, not a model ([github.md](../contribute/github.md)).

Labels (`.github/labels.tsv`, at most 16): `accepted`, `needs-info`, `needs-repro`, `confirmed`, `cannot-repro`,
`owner-needed`, `help-wanted`, `wontfix`, `orphaned`, `bug`, `backend`, `docs`, `model`, `feature`, `author-verified`,
`author-unverified`, plus one label per accepted backend. The two author labels are set by
the bot, not by hand. The next step of an issue is exactly one of `needs-info`, `needs-repro`, `confirmed`,
`owner-needed`, `wontfix`, `help-wanted`. Replies S1 to S7 are at the end.

## 1. Issue triage

Weekly, 45 minutes. Start with the filter `is:open no:label`.

1. Security problem in public text: edit the text out, open a private advisory yourself, tell the reporter (S7).
2. Template incomplete: `needs-info` and S1.
3. Duplicate: close with the link to the first issue.
4. Question or idea: move it to Discussions (S2). A feature is built only after a proposal is accepted (S3, `accepted`).
5. Bug on the reference: reproduce it on the reference machine, then `confirmed`, or `cannot-repro` with what you ran. No
   repro from the reporter: `needs-repro`.
6. Bug on another backend: `backend` plus that backend's label, mention its owner, do not reproduce it. No owner:
   `owner-needed` and S6.
7. A speed or quality claim needs the format of AGENTS.md rule 2 (S4), else `needs-info`.
8. Inactive for 30 days in `needs-info` or `needs-repro`: close with S5; it reopens with the missing answer.

Done when `no:label` is empty; a label and a template reply are enough.

## 2. Pull request review

Cheapest check first; stop at the first failure and say which.

1. An accepted issue (`accepted`) and the author's only open PR? No: close with S3.
2. Template complete, with the AI disclosure and the rights checkbox? No: ask once, then close.
3. One topic, a diff you can read. Unrelated changes: ask to split.
4. Public CI green (`check-ok`, `tests-ok`, code scanning). Workflows of a first-time contributor wait for your
   approval: read the diff for `.github/` changes first.
5. Read everything that executes at build or test time (list in
   [verification.md](../contribute/verification.md#gpu-check-for-a-pull-request-from-someone-else)).
6. `python3 tools/select_checks.py --base origin/main --head <sha>` names the change class and the checks. For anything
   that reaches a golden check, do the GPU check as written there, with the scripts of `main`, batched with other pull
   requests (runbook 6).
7. Reference rule: shared code needs before/after on the reference with spread; slower or different tokens: reject
   (AGENTS.md rules 1 and 3). A bug fix needs a test. A speed claim needs the files and the BENCHMARKS.md row.
8. A backend pull request touches only `engine/backend/<name>/`, its docs and a build option, and has a named owner
   (runbook 4).
9. Merge: squash, subject with the issue number (`#n`), the author's `Assisted-by:` line kept. Post `gpu-golden` for the
   sha you merge.

Close without a full review: no accepted issue, author cannot explain it, opened by an autonomous agent, unrelated to the
issue, bug fix without test, benchmark claim without data, or a weakened test, gate or golden reference.

## 3. Security report

Read daily, 15 minutes: the **Security and quality** tab, advisories in `Triage`.

1. Acknowledge within 7 days. Never discuss in public.
2. No reproduction the reporter ran, or an unverified AI-written text: close with S7.
3. Reproduce it yourself on the latest release build, not on `main`.
4. Rate it by the model in [SECURITY.md](../../SECURITY.md) (the server binds to `127.0.0.1`; a remote attack needs an
   exposed server or a malicious model file), not by the reporter's score.
5. Fix in the advisory's temporary private fork, with a regression test.
6. Disclosure date with the reporter: 90 days by default, sooner when the fix is ready, earlier when the flaw is
   exploited.
7. Request the CVE through the advisory (GitHub is a CNA), credit the reporter if wanted.
8. Release with `--severity security` ([release.md](release.md)); publish the advisory with it and name the fix in the
   notes.

Only the latest release is supported; an older one gets "update".

## 4. Backend contribution

1. **Proposal** (issue form "Backend proposal"): hardware, a named owner who has it, evidence, a build option off by
   default. Accept with `accepted`, or `wontfix` with the reason.
2. **Owner:** named in `engine/backend/<name>/AGENTS.md`, `.github/CODEOWNERS` (created with the first backend) and the
   backend list of the README.
3. **Pull request:** only `engine/backend/<name>/`, docs and the build option. Shared code only if the reference stays
   bit-exact and as fast ([backends/README.md](../backends/README.md#ground-rules)).
4. **Tier 2:** after 3 weeks stable, the owner's public CI, the checks of [backends/README.md](../backends/README.md#checks) and either a
   second person who can build it or a public CI job that builds and tests the backend from the documented steps on a
   GitHub-hosted runner of that platform. Otherwise it stays tier 3.
5. **Monthly health check (15 minutes):** per backend, did the owner act in the last 90 days, is their CI green?
6. **Orphaned:** no reply to a mention within 14 days and no activity for 90 days: label `orphaned`, announce it in
   Discussions and the next release notes, mention the owner.
7. 60 days later without a new owner: demote or remove in one pull request, mentioning the owner; the code stays in the
   history.

A broken backend never blocks a release of the reference ([backends/README.md](../backends/README.md#time-limits)).

## 5. Release

Building and signing: [release.md](release.md). The order:

1. `just lint-all` and `just test` green on `main`; no CodeQL alert of severity high.
2. The scale run ([measurement.md](../contribute/measurement.md#before-a-release)) on the release candidate, with spread and plan; the
   row in [BENCHMARKS.md](../../BENCHMARKS.md); every number the README publishes, three runs. Worse than the last release
   without a reason: stop.
3. `scripts/gpu-status.sh` on the release commit (from `main`, so `--own` is not needed); golden identical to the
   references or re-recorded on purpose and named.
4. Tier 2 owners confirm their backend or are dropped from the notes.
5. Release notes: what changed, fixed issues, known problems, backends and their tier.
6. Tag, wait, rebuild locally, test, sign, publish ([release.md](release.md#release-steps)). The signature check of
   `engine/update/` passes on a second machine, a clean-machine install works and the updater switches to the new
   version.
7. Announce in Discussions.

A hotfix contains only the fix; steps 2 and 3 shrink to the checks of its change class.

## 6. Weekly routine

Target 5 hours a week, cap 8.

| What | Time |
|---|---|
| Security inbox (runbook 3), daily | 15 min a day |
| Triage (runbook 1), weekly | 45 min |
| First look at each pull request | 10 min each; deep review only after `accepted` |
| GPU checks of reviewed pull requests, batched in one session | 1 h |
| Backend health check (runbook 4), monthly | 15 min |
| The rest | own work |

When the queue exceeds the budget, pull these levers in order and announce each in Discussions:

1. Bugs without a reproduction go to Discussions.
2. The `accepted` rule holds strictly; nothing is reviewed without it.
3. Lower the cap of open pull requests for contributors without write access (Settings, Moderation, Interaction limits;
   draft pull requests can count; exemptions by account age or merged pull requests).
4. Interaction limit "prior contributors" for 1 week to 1 month.
5. Pull requests for collaborators only (Settings, pull request access).

Conduct ([CODE_OF_CONDUCT.md](../../CODE_OF_CONDUCT.md)): a first incident gets a written correction, hostile or repeated
behavior a ban; reports by e-mail only.

## Saved replies

**S1 needs info.** Thank you for the report. To look at it I need the missing parts of the form: [list]. I set
`needs-info`; with the answer I continue. Without one, this closes after 28 days (the issue bot does it).

**S2 moved to Discussions.** This is a question or an idea rather than a defect, so Discussions is the better place. I
moved it. A feature becomes an issue once a maintainer accepts a proposal.

**S3 feature process.** Thank you for the work. Changes here start with an issue that a maintainer has accepted (label
`accepted`), see CONTRIBUTING.md. There is none for this change, so I close the pull request. If you open a proposal and
it is accepted, a pull request is welcome.

**S4 benchmark format.** A speed or quality claim needs several runs with their spread, the rate named (decode or
prefill), a plan written before the run and one file under `research/measurements/` (AGENTS.md rule 2,
docs/contribute/measurement.md). Until then I label this `needs-info`.

**S5 closing for inactivity.** I close this because the information asked for did not arrive within 30 days. Reopen it
with the answer and I will continue.

**S6 hardware bug without an owner.** This happens on hardware the maintainers do not own and for which no backend has a
named owner, so I cannot reproduce or fix it. `owner-needed` is set. If you can maintain the backend, see
docs/backends/README.md; with logs and measurements it can move forward.

**S7 security report not verified.** Thank you for the report. We act on reports with a reproduction that the reporter
ran (SECURITY.md). This one has none, or we could not confirm it, so we close it without further analysis. If you add a
minimal reproduction, we look again.
