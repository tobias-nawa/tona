# Code quality and security scanners (2026-10-28)

Agent research (two agents, web), not re-checked; claims carry their source and date. Part 1: static analysis and
quality scanners for this stack. Part 2: GitHub-native and supply-chain security.

## Part 1: static analysis and quality

Baseline already in place: ruff, clang-format, shellcheck, actionlint, gitleaks, lizard gates, clang-tidy (bugprone/clang-analyzer/performance, ROCm clang), libFuzzer + ASan/UBSan harness (local). CI today runs only gates + installer build.

### Findings per tool
**CodeQL (C/C++)**
- Build-free C/C++ scanning (`build-mode: none`) GA since 2025-10-14, CLI >= 2.23.3; default setup picks `none` for C/C++. https://github.blog/changelog/2025-10-14-codeql-scanning-rust-and-c-c-without-builds-is-now-generally-available (preview 2025-06-03)
- Accuracy caveat per docs: less accurate than autobuild/manual with heavy custom macros or many external deps; docs say nothing about CUDA/HIP. https://docs.github.com/en/code-security/reference/code-scanning/codeql/build-options-for-compiled-languages (fetched 2026-10-28). HIP: UNVERIFIED. Expectation (inference): `.hip` extension is not a CodeQL C++ extension, so HIP kernels are likely skipped or mis-parsed; host `.cpp`/`.h` fine. Test it.
- Cost: free on public repos (only third-party source: toolradar; GitHub's own page not verified). Private repos need Code Security, $30/active committer/month (https://github.blog/changelog/2025-03-04-introducing-github-secret-protection-and-github-code-security/). So: do NOT buy Enterprise for this; going public makes it free.
- Adds over clang-tidy: interprocedural taint/dataflow (server JSON/chat/HTTP input -> sinks), CWE-mapped security queries, SARIF in Security tab, PR annotations. Overlaps with clang-analyzer on plain memory bugs. FP reputation: lowish for the default suite (security-extended is noisier). Gate: yes (PR check, via code scanning rulesets) or dashboard. Setup: default setup = 1 click; with `none`, scans only what it can infer; no ROCm needed. Effort: low.

**SonarQube Cloud**
- Free OSS plan announced 2026-01-27: public projects, unlimited members, unlimited branch + PR analysis, custom quality gates; any public project qualifies (Sonar rep 2026-03-09). https://community.sonarsource.com/t/introducing-the-new-free-oss-plan-in-sonarqube-cloud/176923 ; plan table https://docs.sonarsource.com/sonarqube-cloud/administering-sonarcloud/managing-subscription/subscription-plans (OSS plan lists "Advanced SAST", all Team-plan languages). Old Free plan: PR analysis only for main-targeting PRs.
- C/C++: "one-click automatic analysis" for GitHub-hosted projects, any compiler/build system, always free for OSS. https://www.sonarsource.com/products/sonarqube/cloud/features/auto-analysis-for-c-and-cpp/ . Page does not mention CUDA/HIP/compile_commands: HIP handling UNVERIFIED (inference: .hip likely not indexed; to get real accuracy use CI analysis with compile_commands.json, which needs the ROCm clang -> self-hosted or CPU-only compile DB).
- Adds: maintainability/code-smell rules (cognitive complexity, duplication, dead code), hotspots, coverage + duplication tracking, quality gate with "new code" focus, dashboard. Overlaps lizard/clang-tidy heavily. FP reputation for C++: moderate; noisy on style rules; repo notes "Sonar: 15 reported too much" (research/24). Gate: yes (status check on PR). Effort: low-medium.
- SonarQube Server Community Build: C/C++ is Developer Edition+ (paid) per forum posts; community plugin sonar-cxx is third party. No 2025/26 news that Community Build gained C++ found: UNVERIFIED. https://community.sonarsource.com/t/cfamily-code-quality-and-security-for-community-edition-sonarqube/45729 (old). Self-hosting is not worth it given Cloud OSS plan.

**Coverity Scan**
- Still operated by Black Duck, free for registered OSS projects (freely redistributable; corporate-started projects at discretion). https://scan.coverity.com/faq (2026-10-28). Upload limits per project: <100K LOC 28/week, 4/day; 100-500K 21/wk, 3/day; 500K-1M 14/wk; >1M 7/wk. No announcement of shutdown found (negative search, third-party only).
- Needs a real build (cov-build) of the code -> ROCm toolchain required, so not on a hosted runner for the GPU part; CPU part feasible. HIP support: UNVERIFIED (Coverity has CUDA support in commercial; Scan with hipcc/clang unknown).
- Adds: best-in-class interprocedural resource/null/use-after-free/concurrency defects, low FP rate (reputation). Dashboard only, not PR gate (no PR integration; scheduled upload). Effort: medium (registration approval, build capture, token). Value: good, periodic (weekly/release).

**Semgrep**
- CE engine LGPL-2.1; Semgrep-maintained rules under Semgrep Rules License (internal/non-SaaS use limits); no cross-file taint in CE (paid Code). https://semgrep.dev/docs/faq/overview , https://semgrep.dev/docs/licensing . C/C++ coverage in CE is thin (pattern-matching, mostly security-audit style); the AppSec Platform free tier limits not verified. Opengrep fork exists (LGPL).
- Adds little over clang-tidy + CodeQL for C++; useful for custom project rules (e.g. banned APIs) and for Python/YAML/Actions workflow rules (GitHub Actions injection). FP: moderate (pattern-based). Gate: yes. Verdict: skip for C++; optional for workflows (zizmor is better, see below, not researched in depth).

**cppcheck**: free, GPL, no build needed, mature, FP low-moderate; overlaps clang-tidy/analyzer, finds some things (uninit members, container misuse, some UB patterns). HIP: parses `.hip` poorly (needs -D / cfg for __global__, <<<>>>) UNVERIFIED. Premium/MISRA not needed. Verdict: nice-to-have, marginal. (No source fetched; from general knowledge, mark unverified.)

**clang static analyzer / scan-build**: clang-tidy `clang-analyzer-*` already runs the same engine on the TU. scan-build adds nothing except CTU (cross-translation-unit) analysis, which is available via CodeChecker (Ericsson, free/OSS) with clang `--ctu` and a server dashboard. Value: low; consider CodeChecker CTU only if clang-tidy results prove thin. Not verified in this session.

**Infer (Meta)**: MIT, C/C++ support (pulse, lifetime, races); current maintenance status UNVERIFIED (search returned no activity data). Needs build capture; weak C++20 support historically. Skip.

**PVS-Studio**: free license for OSS exists (README/commit/PR mention requirement; excludes commercial/corporate projects and forks; April 2026 post: older "comment-in-source" free mode removed, OSS licensing terms unchanged). https://pvs-studio.com/en/order/open-source-license/ , https://pvs-studio.com/en/blog/posts/1362/ . Strong C++ checks, low FP, but proprietary, needs license key in CI secrets (public repo PR from forks cannot use secrets), eligibility ambiguous (page also wording "personal creative projects"). HIP: UNVERIFIED. Verdict: skip for now; maybe one-off audits.

**Codacy / DeepSource / CodeFactor**: free for public repos (DeepSource "Open Source" plan since Feb 2026; Codacy gives Team features for OSS; CodeFactor free public) but C++ support: DeepSource C/C++ reported beta; Codacy/CodeFactor C++ unclear. Mostly wrap cppcheck/clang-tidy/etc. (aggregator sources: aicodereview.cc, agentdeals.dev, 2026; low reliability). Adds nothing over Sonar + CodeQL. Skip.

**Sanitizers on CPU tests in CI**
- ASan/UBSan: free, high value; run unit tests/CPU-path tests + the fuzz corpus replay on GCC/clang on ubuntu runner. Only useful if there are CPU-only tests (installer tests, parser tests, tokenizer, JSON, chat template, sampling on CPU). Gate: yes. Effort: low.
- TSan: useful for the server (threads, queues). On Ubuntu 24.04 runners TSan can die with "unexpected memory mapping" because vm.mmap_rnd_bits=32; workaround `sudo sysctl vm.mmap_rnd_bits=28` as a workflow step (PX4 docs https://docs.px4.io/main/en/test_and_ci/sanitizers ; bitcoin issue 30674). Can't be combined with ASan. Effort: low-medium.
- GPU code: ASan/TSan do not cover device code (not a CI matter anyway).

**ClusterFuzzLite**
- GitHub Actions modes: PR fuzzing (default 600 s, gates PR, reports only novel crashes via baseline from main build), batch (cron, 3600 s, needs prune), prune, coverage. Sanitizers address (default), undefined, memory. Needs `.clusterfuzzlite/Dockerfile`+`project.yaml`, targets built with its clang and `$LIB_FUZZING_ENGINE`. Corpus in a storage repo (PAT) or artifacts. https://google.github.io/clusterfuzzlite/running-clusterfuzzlite/github-actions/ (2026-10-28)
- Fits existing libFuzzer harness; the fuzz target for JSON/chat parsing must build without ROCm (only the parsing TU) - plausible. Effort: medium (Docker build recipe). Value: turns local fuzzing into PR gate + continuous corpus.
- OSS-Fuzz: requires "significant user base and/or critical to global IT infrastructure"; PR adding projects/<name>/project.yaml; contact must be an established committer with Google account. https://google.github.io/oss-fuzz/getting-started/accepting-new-projects/ . A new niche engine will most likely not qualify now (my assessment); revisit after adoption. Criticality-score detail is from a secondary source.

### Gaps in what the combination covers
Most value not covered by existing stack: (1) interprocedural taint on network-facing server input (CodeQL), (2) CI-enforced sanitizer + fuzz regression (ASan/UBSan tests, ClusterFuzzLite), (3) deep defect analysis (Coverity), (4) trend dashboard/PR quality gate (Sonar). Workflow-injection linting (zizmor) is outside the question and not researched.
Common blind spot: none of CodeQL/Sonar/Coverity is verified to parse `.hip` kernels; expect host code coverage only. GPU kernels remain covered by clang-tidy with ROCm clang, local only.

### Ranked recommendation
Must-have
1. CodeQL default setup (build-mode none, free for public): zero cost, PR annotations, taint analysis on server input; verify what it does with `.hip`/headers after first run.
2. ASan+UBSan (and later TSan) CI job on CPU-buildable tests incl. fuzz-corpus replay: closes the gap that sanitizers only run locally.
Nice
3. ClusterFuzzLite PR fuzzing (code-change mode, ASan, 600 s) on the JSON/chat parser target: continuous, gates PRs; needs Docker recipe.
4. SonarQube Cloud OSS plan (auto analysis): free dashboard + PR quality gate/new-code metrics; overlaps lizard/clang-tidy, noise risk; use as dashboard first, gate later; HIP blind.
5. Coverity Scan weekly/release upload from the maintainer machine (has ROCm): deep defects, free; manual process, dashboard only.
Skip
- Semgrep (thin C++ in CE, rules license), cppcheck (marginal; maybe later), Infer (status unverified), PVS-Studio (proprietary key, eligibility unclear), Codacy/DeepSource/CodeFactor (C++ weak, wrappers), SonarQube Server self-host (C++ paid), OSS-Fuzz (acceptance criteria unlikely to be met now), GitHub Enterprise upgrade (not needed: all useful scanners are free for public repos).

## Part 2: GitHub and supply chain


Agent research (web only, Sonnet). "UNVERIFIED" = not confirmed against a primary source.
Existing in repo: gitleaks, actionlint, shellcheck, ruff, clang-tidy, lizard (pre-commit), SHA-pinned actions, `permissions: contents: read`, single workflow check.yml.

### 1. Plans and prices (verified)
- Secret Protection $19 and Code Security $30 per active committer/month; GHAS split into these two on 2025-04-01
  (https://github.blog/changelog/2025-03-25-introducing-github-secret-protection-and-github-code-security/ ; https://github.com/security/plans, fetched 2026-10-28). Purchasable on Team without Enterprise. No 2026 price change found.
- Team $4/user/mo and Enterprise from $21/user/mo "for the first 12 months" (https://github.com/pricing, 2026-10-28; list price after year 1 UNVERIFIED).
- Public repos: GitHub's feature matrix states these are "available for public repositories by default"
  (https://docs.github.com/en/code-security/getting-started/github-security-features, 2026-10-28).

### 2. Feature by feature

### CodeQL code scanning
- Free on public repos, any plan (features page above). Default setup uses build-mode `none` for C/C++, C#, Java, Rust; advanced setup is needed for autobuild/manual
  (https://docs.github.com/en/code-security/code-scanning/creating-an-advanced-setup-for-code-scanning/codeql-code-scanning-for-compiled-languages, 2026-10-28).
  Caveats of `none`: generated code not analysed; accuracy "Good" (manual = "Best"); dependency guesses may be wrong. GA/preview label of C++ `none`: UNVERIFIED (page does not say).
- For this repo: HIP (`.hip`/`__global__`, hipcc) probably not understood by CodeQL's extractor under `none`; host C++ (HTTP server, JSON, GGUF parser) is what matters. Expect partial coverage of device code. Manual build with hipcc: UNVERIFIED whether the extractor traces it. Test in a PR.
- Copilot Autofix for CodeQL alerts: free on public repos since 2024-09 (https://github.blog/changelog/2024-09-17-now-available-for-free-on-all-public-repositories-copilot-autofix-for-codeql-code-scanning-alerts), no Copilot subscription needed (docs: autofix concepts page).
  NEW 2026: "agentic autofix" GA 2026-10-15 (https://github.blog/changelog/2026-10-15-agentic-autofix-for-code-scanning-alerts-general-availability): Enterprise Cloud, consumes AI credits from the Code Security budget -> paid, not the free one.
- SARIF upload of third-party tools (e.g. clang-tidy, cppcheck, semgrep) into code scanning: free on public (plans page).

### Secret scanning / push protection
- Free on public: secret scanning alerts, push protection, provider patterns, partner notification (https://github.com/security/plans).
- NOT free (Secret Protection / Team+Ent plans): validity checks, Copilot generic-secret scanning, generic patterns, bypass controls (delegated bypass), scan history API, security overview insights (same page). Custom patterns: features page lists them under "requires Secret Protection" (no free public option stated).
- Net: gitleaks in pre-commit + CI already covers generic/custom needs; the free native part adds provider-pattern push protection for contributors. Turn on.

### Dependabot / dependency graph
- Alerts, security updates (grouped), version updates, malware alerts: all plans (features page; plans page).
- Ecosystems: `github-actions`, `pip`, `gitsubmodule`, `pre-commit`, `vcpkg` (baseline SHA only) are supported for updates; Conan and CMake FetchContent are NOT listed
  (https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories, 2026-10-28; only first ~150k chars read, rest read via offset).
- Vendored C++ in `engine/third_party/`: no evidence Dependabot or the dependency graph detects vendored headers (docs mention vendoring only for yarn). Options: (a) make each vendored lib a git submodule (then `gitsubmodule` updates work), or (b) submit them via the dependency submission API (https://docs.github.com/en/rest/dependency-graph/dependency-submission) with a hand-written snapshot or SPDX/CycloneDX SBOM (actions like "SBOM dependency submission" exist; none C++-specific found). Submitted deps then get Dependabot alerts only if a matching package-URL exists in the Advisory DB (purl type for C/C++: generic/`pkg:github/...` matches rarely: UNVERIFIED). Pragmatic: track version + upstream commit in a manifest and a scheduled "new upstream release" check.
- Dependency review action: public repos free (https://docs.github.com/en/code-security/concepts/supply-chain-security/about-dependency-review: "available for all public repositories"; private needs Code Security). NOTE: the plans page lists it as not free, contradicting the docs; docs wording is more specific, treat as free on public (conflict noted).
  Value here is low: only pip + Actions manifests are visible to the graph.

### OpenSSF
- Scorecard action: free for public repos; badge needs `publish_results: true` + `id-token: write`; forced off for private (https://github.com/ossf/scorecard-action README / marketplace https://github.com/marketplace/actions/ossf-scorecard-action; latest v2.4.4 as seen). Cheap, gives a prioritised hardening list (branch protection, pinned deps, token permissions, SAST, fuzzing, signed releases, security policy). Check list from memory, UNVERIFIED here.
- Best Practices badge (bestpractices.dev): free self-attestation, passing/silver/gold; a newer "Baseline" ladder exists (UNVERIFIED). One-evening form; useful as a checklist.
- Allstar: GitHub App, free, opens issues on policy violations, runs per org (mainly for orgs; low value for a single repo; no deprecation found, 2026 status UNVERIFIED; https://openssf.org / https://opensource.google/documentation/reference/github/allstar).

### Attestations / SLSA / signing
- Artifact attestations: all plans for public repos (Sigstore public-good instance, public transparency log); private/internal repos need Enterprise Cloud (features page). `actions/attest-build-provenance` alone = SLSA v1.0 Build L2; L3 needs a reusable workflow (https://docs.github.com/en/actions/concepts/security/artifact-attestations). Verify with `gh attestation verify`.
- Immutable releases: GA 2025-10-28, all repos/orgs, locks tag + assets, adds signed release attestation (https://github.blog/changelog/2025-10-28-immutable-releases-are-now-generally-available/). Create release as draft, attach assets, then publish. Reported tag-reuse friction.
- Sigstore/cosign signing of binaries: free, but attestations already use Sigstore; extra cosign only if you need `.sig` files outside GitHub. Scorecard Signed-Releases accepts attestations/`.sig`/`.asc` (UNVERIFIED which).

### Actions hardening
- zizmor: free OSS static analyser (template injection, dangerous triggers like pull_request_target/workflow_run, excessive permissions, unpinned uses, YAML-anchor evasion hardened 2026) (https://blog.trailofbits.com/2026/05/22/we-hardened-zizmors-github-actions-static-analyzer/ ; https://pypi.org/p/zizmor). Complements actionlint (syntax). Can emit SARIF into code scanning. Run as pre-commit hook.
- pull_request_target: Mar 2026 trivy-action compromise via misconfigured pull_request_target (Trail of Bits, above). actions/checkout v7 (GA 2026-06-18) refuses fork-PR checkout in `pull_request_target`/`workflow_run` by default; backport to all supported majors announced for 2026-07-16 (https://github.blog/changelog/2026-06-18-safer-pull_request_target-defaults-for-github-actions-checkout; backport landing not confirmed). Rule: use plain `pull_request`, never check out fork code with secrets.
- Self-hosted runner warning (project-specific, from GitHub docs knowledge, UNVERIFIED in this run): a GPU self-hosted runner on a public repo must never run fork-PR code; GPU tests should run on maintainer-triggered (label/`workflow_dispatch`/merge-queue) events only.
- Actions policy "require SHA pinning" and block-list: all-plan setting at repo/org level (https://github.blog/changelog/2025-08-15-github-actions-policy-now-supports-blocking-and-sha-pinning-actions/; plan availability for personal accounts UNVERIFIED). Cheap, enable it.
- Fork-PR defaults: "require approval for all outside contributors" in Settings > Actions; set default `GITHUB_TOKEN` to read-only.
- harden-runner (StepSecurity): Community tier free, public repos + GitHub-hosted only, 10,000 runs/week then enforcement silently stops; private repos need paid (https://docs.stepsecurity.io/harden-runner/installation/community-license/). Use audit mode first. Nice-to-have, not needed with one small workflow. StepSecurity maintained actions free for public repos since 2026-06-29 (their blog).
- 2026 Actions roadmap (2026-03-26): workflow `dependencies:` lockfile (preview 3-6 mo), ruleset-based workflow execution rules (e.g. block pull_request_target), scoped secrets, native egress firewall (preview 6-9 mo), data stream; plans not stated (https://github.blog/news-insights/product-news/whats-coming-to-our-github-actions-2026-security-roadmap/). Current status of these: UNVERIFIED (not searched for shipped versions).
- Rulesets / branch protection: available on public repos with Free (user and org) (https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets). Free personal account on a public repo: rulesets, required status checks, required reviews, CODEOWNERS all OK. "Restrict who can push" only for orgs (Free org public ok / Team) -> use a rulesets-based approach in an org if that is needed (personal-account limit: UNVERIFIED). Required workflows (org-level) are Enterprise-ish: UNVERIFIED in this run (about-rulesets page says org rulesets are Enterprise-described). Push rulesets: private/internal only.

### Vulnerability reporting
- Private vulnerability reporting + repository security advisories (GHSA, CVE request) + SECURITY.md: all plans, public repos (docs link: configuring-private-vulnerability-reporting-for-a-repository; features page). Enable PVR, write SECURITY.md pointing to it. Temp private forks / CVE flow: not covered by fetched page, known from GitHub docs (UNVERIFIED here).
- Security campaigns: Code Security (paid), not free on public (features page).

### 3. Does Enterprise/GHAS buy anything for a PUBLIC repo?
Effectively no for this project. Everything that matters (CodeQL, SARIF upload, Autofix, secret scanning + push protection, dependency review, Dependabot, rulesets, attestations, PVR, advisories) is free on public repos on a Free account.
What paid adds on public repos: secret-scanning validity checks, Copilot/generic secret detection, custom patterns, delegated bypass; Dependabot custom auto-triage rules; security campaigns; security overview/org-level insights; agentic autofix (AI credits); AI Scan; org-wide required workflows/enterprise policies. None blocks a one-maintainer public C++ engine; gitleaks already covers custom/generic secrets. Enterprise Cloud is only needed for attestations on a private repo (not applicable once public) or org governance.
Only worth considering if: multiple maintainers need org-level rulesets with restricted push (an Free/Team org suffices, $4/user), or private mirrors need the same scanning.

### 4. Ranking
Must-have (all free on public):
1. Branch ruleset on main: required checks, PR review, no force-push, block deletions (Scorecard Branch-Protection; fork-PR model).
2. Actions hardening: `permissions` least privilege (done), zizmor in pre-commit/CI, read-only default token, approve-all-outside-contributors, never pull_request_target + fork checkout, GPU runner never for fork PRs, require-SHA-pinning policy.
3. Dependabot version updates for `github-actions` + `pip` (+ `gitsubmodule`/`pre-commit` if used) and security updates/alerts.
4. Secret scanning + push protection (native, free) next to gitleaks.
5. SECURITY.md + private vulnerability reporting enabled (HTTP server and GGUF parser are the attack surface).
6. CodeQL default setup for C/C++ (verify what it sees of HIP code); keep clang-tidy security checks.
7. Release path: attest-build-provenance + immutable releases (draft -> publish).
Nice:
- OpenSSF Scorecard workflow + badge; Best Practices badge (passing).
- Dependency review action (small value: pip/Actions only).
- Submit vendored C++ libs via dependency submission API or convert to submodules.
- harden-runner audit mode (community tier).
- Fuzzing of GGUF parser/JSON (OSS-Fuzz / ClusterFuzzLite; Scorecard Fuzzing check) - not in scope of this run, only mentioned: UNVERIFIED.
Skip:
- GHAS / Secret Protection / Code Security licences, Enterprise, agentic autofix, security campaigns, Allstar (single repo), cosign on top of attestations (until needed), required org workflows.

### 5. Free on public repos vs needs paid
Free on public (Free account): CodeQL + SARIF upload + Copilot Autofix (non-agentic), secret scanning alerts + push protection + provider patterns, dependency graph + SBOM export, Dependabot alerts/security/version updates, dependency review action, rulesets/branch protection (repo-level), CODEOWNERS, artifact attestations, immutable releases, PVR, advisories/GHSA, SECURITY.md, Scorecard action/badge, Best Practices badge, Allstar, zizmor, Actions SHA-pinning policy (personal-account availability UNVERIFIED), harden-runner community tier (public, 10k runs/wk cap).
Needs paid plan: secret validity checks, generic/Copilot secret detection, custom secret patterns, delegated bypass, scan history API (Secret Protection $19 or Team/Ent); security campaigns, Dependabot auto-triage rules, agentic autofix/AI Scan, security overview insights (Code Security $30); attestations and dependency review on private repos (Enterprise Cloud / Code Security); push restrictions by user (org Team/Ent); org-wide required workflows (Enterprise, UNVERIFIED); harden-runner on private repos or self-hosted runners.

### Not verified / gaps
- Whether CodeQL extracts HIP kernels; whether Dependabot/dependency graph sees any vendored C/C++; purl matching for C++ advisories.
- Exact Team/Enterprise list price after first year; Allstar 2026 status; Scorecard check list; whether the checkout v7 backport (2026-07-16) landed; shipped state of the 2026 Actions roadmap items.
- github.com/pricing page and docs disagree on dependency-review-action being free (see above).
