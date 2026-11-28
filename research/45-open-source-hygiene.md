# 45: Open-source hygiene: model rights for DM, DCO, AI notice in the README

Research of 2026-11-11 for #122, points 1, 3 and 4 (point 2, code origin, is a separate file). **Not legal advice.** Where
the law is open it says so and names whom to ask. Evidence tags: [V] = primary text read directly, retrieved
2026-11-11 (UTC); [S] = secondary source; "unchecked" = not verified. Verbatim quotes are marked with quotation marks.

## 1. Model rights: may DM be published on Hugging Face?

DM = byte copy of 164 of 256 experts per layer of the ISTA Coder GGUF, plus the unchanged non-expert part
(`research/measurements/2026-11-11-requant.md`). It is therefore a derivative of two things at once: the ISTA release
and, through it, Qwen3.8-Flash-Next.

### 1.1 The ISTA repository [V]

Repository `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`, read through the Hugging Face API and raw files
(`huggingface.co/api/models/<repo>`, `/commits/main`, `/raw/<rev>/README.md`, `/discussions`).

- Created 2026-09-26, last modified 2026-10-19, 775 452 downloads, not gated. HF tag `license:apache-2.0`; card YAML
  `license: apache-2.0`, `base_model: Qwen/Qwen3.8-Flash-Next`, `base_model_relation: quantized`.
- The card's License section: "These weights inherit the license of the base model (**Qwen3.8-Flash-Next**). The
  GSQ-RCO tooling is released by the Deep Algorithms and Systems Lab under its repository license."
  The page badge links `license-apache--2.0` to `#license`, i.e. to the sentence above.
- **Files in the repository:** `.gitattributes`, two IQ1_M shards, `README.md`, `mmproj-...-BF16.gguf`, `assets/*`,
  `tensor-allocation/*.txt`. **No `LICENSE` file, no `NOTICE`.**
- **Card history:** 17 commits. The License section is identical in the first card commit (`a1be57df`, 2026-09-26), in
  our pinned revision (`5348543e`, 2026-09-29) and in `main` (`bba88856`, 2026-10-19), checked by diff: the
  contradiction (Apache tag, "inherit" text) existed from day one and was never resolved or discussed.
- **Shard replaced:** on 2026-10-19 ISTA replaced shard 1 (`d1d94679`, "Replace first shard with updated expert-pruned
  model"). The card says (update of 2026-10-17): SWE-bench Verified 75.60 to 79.20, LiveCodeBench v6 86.28 to 87.43.
  **DM and every Tona measurement derive from the superseded shard (pinned `5348543e`, 2026-09-29).** Consequence: a
  DM card must say which ISTA revision it was cut from, and ISTA's current benchmark numbers do not apply to it.
- **What ISTA did:** "50% expert-pruned" (256 of 512 experts per layer), retained weights at 3.5 bpw, selection by RCO
  on calibration data. In discussion #22 (2026-10-06/07) `anm2211` named the calibration sources
  (`sci-datasets/sci-code`, `gaia-benchmark/GAIA`, `HuggingFaceM4/the_cauldron`, `open-thoughts/OpenThoughts3-1.2M`);
  "we don't have access to RCO scores anymore". So DM is a second pruning on top of ISTA's pruning; ISTA's selection
  is part of what we copy.
- **Discussions:** 30 (27 open). A full-text scan of all 30 threads for "licen", "Apache", "Qwen Community",
  "redistribut", "derivative", "requant" found **no thread about the license.** Staff answer there: `anm2211` replied on
  2026-10-06, 10-07, 10-27 and 10-28 (discussions #11, #22). That ISTA is the author's organisation is inferred from
  wording ("we used", "we are currently working on"); not checked against an affiliation page.

### 1.2 Other ISTA material [V]

`huggingface.co/api/models?author=ISTA-DASLab`: the GSQ-RCO releases carry `apache-2.0` (Qwen3.8-27B, Qwen3.8-Flash-Next
unpruned) or `mit` (DeepSeek-V4.1-Flash, a different base); Kimi-K3 MoESQ is `other`. The unpruned
`Qwen3.8-Flash-Next-GSQ-RCO-GGUF` has the same License text ("These quantized weights inherit the license of the base
model"). So the pattern is systematic: YAML tag as a default, text defers to the base model. Code:
`github.com/IST-DASLab/GSQ` and `/RCO` are **Apache-2.0** (GitHub API `license.spdx_id`, `LICENSE` file in both).
Papers: arXiv 2604.18556 (GSQ), 2605.00649 (RCO), cited by the card; arXiv licenses unchecked (not needed, we copy no
text). Citation requested in the card's BibTeX block (not a license condition).

### 1.3 Qwen Community License 1.0 [V]

Source: `huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE` (repo state `812d96e1`, 2026-11-07). The Qwen README
has only YAML (`license: other`, `license_name: qwen-community-1.0`, `license_link: LICENSE`), no extra text.
`ggml-org/Qwen3.8-Flash-Next-GGUF` and Unsloth use `license: other`.

Grant: permission "to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, sell, deploy, host, fine-tune, and create derivative works from", where
"Software" is defined as "the model weights, parameters, configuration files, inference code and associated
documentation files".

- **Derivative:** the term "derivative works" is used but not defined. Quantization and expert removal are covered by
  "modify" and "create derivative works" in the grant itself, so for DM it does not matter whether a cut "is" a
  derivative: the grant names both, and condition 1 applies to "all copies or substantial portions".
- **Condition 1 (notice):** "The above copyright notice and this permission notice shall be included in all copies or
  substantial portions of the Software." Plus: model name shown prominently if the licensee's commercial product or
  service has more than 100 000 000 monthly active users or US$ 20 000 000 monthly revenue ("or any derivative works").
- **Condition 2:** "If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant
  business, the licensee shall obtain a separate license from Qwen before Using the Software or its derivative works
  for any commercial purpose." Exempt: internal use that does not make the Software or its outputs "available to any
  third party". "AI Work Assistant" is "an independent AI-powered product primarily designed for AI-assisted coding or
  office productivity (e.g., Qoder and QwenWork)"; it excludes "an AI assistant that is a feature of a product whose
  primary purpose is not AI-assisted coding or office productivity".
- **Not in the text:** no naming rule for derivatives, no "Built with", no copyleft, no acceptable-use policy, no
  clause on who may host. Contact for questions: model-business@notice.qwencloud.com.
- **Applies to Tona?** Tona is free (MIT, no revenue, memory `tona-stays-free`); condition 2 needs a "business" and a
  "commercial purpose". An AI-coding launcher (`tona claude`) inside an engine is arguably "AI-assisted coding" in
  spirit, but Tona is an inference engine, the user runs it, and nothing is sold. **Open and not decidable by us:** how
  Qwen reads "AI Work Assistant business" for a free tool if that ever changes. Ask Qwen at the address above before
  any commercial offer.

### 1.4 The conflict inside the ISTA card, and what each reading demands

Reading A (text wins): the weights are Qwen-licensed; Qwen condition 1 applies. Reading B (YAML wins): Apache-2.0
applies to ISTA's weights; section 4 of the license [V, apache.org/licenses/LICENSE-2.0.txt] requires: "(a) You must
give any other recipients of the Work or Derivative Works a copy of this License; and (b) You must cause any modified
files to carry prominent notices stating that You changed the files". Reading C: both apply in layers (Qwen to what
comes from Qwen, Apache to ISTA's own contribution: the expert choice and quantization). Hosting both license texts
and a change note satisfies A, B and C at once; nothing in either license forbids the combination (Apache-2.0 is
permissive, the Qwen license has no copyleft). ISTA cannot grant more than Qwen granted (it received the weights under
the Qwen terms), so B alone can never be less restrictive than Qwen for the Qwen-derived part.

### 1.5 Result

- **Likely permitted** (Qwen grant is explicit about modify, derivative works and distribute; ISTA's text defers to
  it; the Apache tag would also allow it), **under these duties:** (1) verbatim Qwen `LICENSE` in the DM repository;
  (2) Apache-2.0 text and a change note ("expert subset, byte copy, no requantization, cut from ISTA revision
  `5348543e`") because of the tag; (3) card names Qwen3.8-Flash-Next as base, ISTA as source with links to both papers and
  both code repositories, and says what we changed; (4) YAML `license: other`, `license_name: qwen-community-1.0`,
  `base_model_relation` per HF schema (quantized/finetune: pick the closest, unchecked which the Hub accepts for a cut);
  (5) a note for users on condition 2; (6) a name that does not suggest an official Qwen or ISTA release.
- **Open, with the person to ask:**
  - What ISTA's own license for its contribution is (YAML or text): **ISTA**.
  - Whether ISTA objects to a further pruning of its selection: **ISTA**; the licenses do not require permission, it
    is courtesy and risk reduction.
  - Whether weights are copyrightable at all and what a byte-copy cut adds: legally unsettled; unchecked (no
    authoritative source gathered); a lawyer, if this ever carries commercial weight.
  - Condition 2 for a future commercial offer: **Qwen**, model-business@notice.qwencloud.com.
- **Recommendation:** publish DM only with notices complete. If ISTA's licence stays unclear, the licence chain still
  permits it (both readings), but then say so on the card and keep the "derived from an unreviewed ISTA revision" note.

## 2. DCO in 2026

### 2.1 The text and what a sign-off means [V]

Developer Certificate of Origin 1.1 (`developercertificate.org`): the contributor certifies "(a) The contribution was
created in whole or in part by me and I have the right to submit it under the open source license indicated in the
file; or (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an
appropriate open source license and I have the right under that license to submit that work with modifications ... or
(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not
modified it. (d) I understand and agree that this project and the contribution are public and that a record of the
contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely".
Marker: a `Signed-off-by: Name <email>` trailer, added by `git commit -s`. Linux `submitting-patches.rst`: "using a
known identity (sorry, no anonymous contributions.)". QEMU: the name "does not need to be your legal name ... but
should not be anonymous, nor misrepresent whom you are", and should match the commit author.

### 2.2 Who uses it how [V unless marked]

| Project | Position | Source |
|---|---|---|
| Linux kernel | DCO via `Signed-off-by`; AI: allowed, tools follow the normal process; "AI agents MUST NOT add Signed-off-by tags. Only humans can legally certify the Developer Certificate of Origin (DCO)"; the human must review all AI code, add their own sign-off and take "full responsibility"; transparency about tools in the changelog; maintainers may reject | `Documentation/process/coding-assistants.rst`, `generated-content.rst` (torvalds/linux master). The kernel's own `Assisted-by` tag is **not** in either file (grep: none); unchecked whether defined elsewhere |
| CNCF | Charter: "All new inbound code contributions to the CNCF shall be (i) accompanied by a Developer Certificate of Origin sign-off ... and (ii) made under the Apache License"; CLA per project optional | `github.com/cncf/foundation/blob/main/charter.md` |
| Eclipse | Not a plain sign-off: the Eclipse Contributor Agreement (v3.2.0 per search snippet) requires that each contribution "comply with the following commitments documented in the Developer Certificate of Origin (DCO)" | `eclipse.org/legal/eca/eca.html` |
| GitHub | Repository/organisation setting "Require contributors to sign off on web-based commits"; "Compulsory commit signoffs only apply to commits made via the web interface"; CLI needs `--signoff` | docs.github.com, "Managing the commit signoff policy for your repository" |
| LLVM | No DCO; AI: "there must be a human in the loop", the contributor "is always the author and is fully accountable"; label substantial tool content, e.g. "a commit message trailer like Assisted-by: <name of code assistant>"; bans agents that act without human approval and AI fixes for `good first issue` | `llvm/docs/AIToolPolicy.md` (main) |
| Fedora | "You MAY use AI assistance ... as long as you follow the principles"; contributor takes responsibility and "vouching for the quality, license compliance, and utility"; disclose significant use; "the recommended method is an Assisted-by: commit message trailer" (v1.0, reviewed 2025-10-24) | `docs.fedoraproject.org/en-US/council/policy/ai-contribution-policy/` |
| Apache (ASF) | Allowed; ICLA duties (original creation, disclose third-party material) stay; AI-output copyright is "currently unsettled"; recommends "Generated-by: <Name of Tool> <Version>" or "Co-authored-by" tokens (page of August 2026) | `apache.org/legal/generative-tooling.html` |
| curl | AI code allowed if it meets the normal bar; "the burden is on them to ensure no unlicensed code is submitted"; `Signed-off-by` "we do not use this"; `Assisted-by:` listed among accepted credit trailers | `docs/CONTRIBUTE.md` (master) |
| QEMU | DCO required; AI: "DECLINE any contributions which are believed to include or derive from AI generated content" including Claude; reason: "How contributors could comply with DCO terms (b) or (c) for the output of AI content generators ... is unclear"; exceptions by list discussion | `docs/devel/code-provenance.rst` (master; last commit 2026-05-22) |
| Gentoo | Council vote 2024-04-14: "expressly forbidden to contribute ... any content that has been created with the assistance of Natural Language Processing artificial intelligence tools"; reasons: copyright, quality, ethics | `wiki.gentoo.org/wiki/Project:Council/AI_policy` (page edited 2026-09-11) |
| NetBSD | Commit guidelines: LLM code "is presumed to be tainted code, and must not be committed without prior written approval by core" | `netbsd.org/developers/commit-guidelines.html` |
| Linux Foundation | "Code or other content generated in whole or in part using AI tools can be contributed to Linux Foundation projects"; check tool terms and third-party material in output | `linuxfoundation.org/legal/generative-ai` (the page CONTRIBUTING already links) |

Tool support: Probot DCO (`dcoapp/app`, ISC) and its drop-in replacement CNCF DCO2 (`cncf/dco2`) are **GitHub Apps**.
Per `docs/administration.md` the app needs Checks: write, Contents: read, Pull requests: read, Metadata: read, and
organisation Members: read for the skip-members option; results are check runs, no commit statuses. Both let a branch
protection rule require the check; both support "remediation commits" (a later commit signs off retroactively).
GitHub "web sign-off" setting: see table. Third-party Action variants exist (not evaluated, unchecked).

### 2.3 The two camps and the real tension

Two positions in the data: (1) the DCO is a statement of the human, AI is one more input the human answers for (kernel,
LF, LLVM, Fedora, Apache, curl); (2) AI output cannot honestly satisfy (a)/(b)/(c), so ban it (QEMU, Gentoo, NetBSD).
Tona sits in camp 1; camp 2 is a legal-risk argument, not a quality one, and its premise (unsettled copyright and
possible regurgitation of licensed code) is real. Apache says outright that relying on non-copyrightability alone
"ties their contribution to an answer nobody knows yet". US Copyright Office guidance on AI output: not fetched,
unchecked. Residual risk stays with the contributor who signs; it is not removed by any process.

### 2.4 Does DCO fit `Assisted-by` and "no Co-Authored-By for tools"? Yes, with one rule

- `Signed-off-by` = human certification; `Assisted-by` = provenance disclosure. Different questions, different
  trailers; the kernel, LLVM, Fedora and curl use them together (kernel: human sign-off, tools disclosed in the changelog).
- "No Co-Authored-By for tools" is **consistent** with the kernel: authorship and certification stay with humans. Apache
  suggests `Co-authored-by` for tools; that is a different choice, not a conflict with DCO. Tona's rule (CONTRIBUTING:
  "a tool never adds Signed-off-by") matches the kernel text.
- **The trap:** `git commit -s` takes name and email from git config. An agent that runs `git commit -s` in the
  maintainer's checkout writes the maintainer's sign-off before any review. That makes the sign-off meaningless.
  Rule: the human adds the sign-off **after** reading the change (`git rebase --exec 'git commit --amend --no-edit -s'
  <base>` or at merge); agents commit without `-s`. A sign-off is only a text line and is not authenticated; signed
  commits (SSH/GPG) are a separate, optional layer, not needed for DCO.

### 2.5 Recommendation for Tona (MIT, AI-first, one maintainer)

1. **Adopt the DCO for outside contributors; it replaces the checkbox in the PR template as the durable record.**
   Benefits: sign-off travels with the commit (the checkbox does not survive forks or history), no CLA, matches
   CNCF/kernel practice, one line to explain. Cost: friction for agent-driven flows (see rule above) and one more thing to
   fail in CI. maintainer decision (process change).
2. **Wording in CONTRIBUTING (draft, not applied):** "Every commit carries `Signed-off-by: Name <email>` (git commit -s),
   which certifies the Developer Certificate of Origin 1.1. You add it yourself after reviewing the change; a tool never
   adds it. AI-assisted work is allowed: `Assisted-by` names the tool, the sign-off is your statement that you reviewed
   the change and have the right to submit it under the MIT license."
3. **Gate, forge-neutral, no secrets, minimal rights.** Do not use a GitHub App (it needs repository read and check
   write permissions and is a third party). Build on what exists:
   - `tools/gates/dco.py` (stdlib only): for each non-merge commit in `from..to`, require a `Signed-off-by:` trailer
     whose email equals the author email (as dcoapp does) or the committer email; skip bot authors (e.g.
     `dependabot[bot]`); print each failing commit with the fix command. Add `tests` in `just test`.
   - `.pre-commit-config.yaml`: a `repo: local` hook, `language: system`, `pass_filenames: false`, `always_run: true`,
     `stages: [pre-push]`. At pre-push, **not** `commit-msg`: agents commit unsigned, the human signs before push.
     pre-commit passes the range as `PRE_COMMIT_FROM_REF` / `PRE_COMMIT_TO_REF` (so CI can reuse it with `--from-ref`).
   - `.github/workflows/check.yml`: the `gates` job already runs `pre-commit run --from-ref "$from" --to-ref HEAD` with
     `permissions: contents: read`, `fetch-depth: 0`, SHA-pinned actions, no secrets, base SHA passed through `env`.
     Add one step `pre-commit run dco --hook-stage pre-push --from-ref "$from" --to-ref HEAD`. No new trigger, no
     `pull_request_target`, no new permission; zizmor, actionlint and `tools/gates/workflows.py` already cover the file.
   - `just signoff`: the rebase-exec one-liner for the maintainer's own review step.
   Status: **design, not run**; nothing of this is in the repository.
4. **Squash and release.** With squash-merges the individual sign-offs survive only if they end up in the squash
   message: unchecked for GitHub; prefer rebase-merge for outside PRs, or let the maintainer sign the merge
   commit. The one-commit release (README section, point 3) gets `Signed-off-by` of the maintainer; contributors'
   earlier sign-offs are not carried into it, which the AI notice and a `CONTRIBUTORS`-style note should state.
5. **Residual risk, said openly:** QEMU's argument (DCO (b)/(c) for AI output is unclear) is not answered by the gate.
   Mitigations already in CONTRIBUTING (review, explain every line, no output that reproduces someone else's code, LF
   guidance). Whom to ask if this must be settled: a lawyer with open-source experience, or the LF/SPDX legal community.

## 3. AI notice in the README

### 3.1 How projects say it [V, retrieved 2026-11-11]

- **DwarfStar / ds4** (`github.com/antirez/ds4`, README, main): "This software is developed with **strong assistance from
  AI coding agents** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how
  the project was built. If you are not happy with AI-developed code, this software is not for you." Plus an
  acknowledgement section crediting llama.cpp and GGML.
- **Ghostty** (`ghostty-org/ghostty`, `AI_POLICY.md`): "Ghostty is written with plenty of AI assistance, and many
  maintainers embrace AI tools as a productive tool in their workflow." Contributors must disclose tool and extent;
  "The human-in-the-loop must fully understand all code."
- **llama.cpp** (`CONTRIBUTING.md`, master): "AI-generated code is allowed. You are 100% responsible for every line,
  however it was produced." Disclosure of the manner of use is required; "Undisclosed AI usage may result in your
  account being permanently banned". Policy for contributors, not a statement about the code base.
- A wording choice seen in the wild [S: issue btclib-org/.github #1994, 2026-11-02]: say "LLM coding agents" unless a
  statement is true only for one tool.
- Gaps: I found **no** README of a comparable project that explains that a squashed release commit drops per-commit
  disclosure. Tona's note is novel; keep it factual. No search for more examples beyond the three above (unchecked
  whether other engines have a similar paragraph).

What the examples share: first person, plain statement of fact, one sentence on human role, no apology, no claim of
quality. What to avoid: "100% AI-free", "fully autonomous", "production-grade" (unmeasured claims).

### 3.2 Draft for Tona (English, README, short)

> ## How Tona is developed
>
> Tona is developed with AI coding agents (mainly Claude Code, by Anthropic). A human, the maintainer, sets the
> direction, reviews every change before it is merged and answers for it. Agents do not open pull requests or issues on
> their own. What backs the code is not trust in the tool: unit and golden tests, bit-exact checks, quality checks for
> numerics, the gates in [AGENTS.md](../AGENTS.md), and measurements with their test conditions under
> `research/measurements/`.
>
> Every commit of the development history that an agent wrote or substantially shaped ends with an `Assisted-by:`
> line. The first public release is published as a single commit, so that history, with its per-commit disclosures,
> is not part of it; this paragraph is the disclosure for the release. Contributions follow
> [CONTRIBUTING.md](../CONTRIBUTING.md) (AI use is allowed and must be disclosed per commit).

Notes: (1) "a human reviews every change" must be true as written; say "the maintainer or a reviewing agent under the
maintainer's rules" only if that is how it is, otherwise keep the sentence. This is the claim most likely to be challenged. (2) The model name list in `Assisted-by` shows several
models; the paragraph says "mainly Claude Code" to stay true. (3) Do not say "no code was copied": point 2 (code
origin, llama.cpp and others) decides what the README may claim about origins.

## Sources (all retrieved 2026-11-11)

- ISTA repo and API: `huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` (card, `/commits/main`,
  `/discussions`), `github.com/IST-DASLab/GSQ`, `/RCO`
- Qwen: `huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE`
- Apache-2.0: `apache.org/licenses/LICENSE-2.0.txt`; ASF: `apache.org/legal/generative-tooling.html`
- DCO: `developercertificate.org`; `github.com/dcoapp/app`; `github.com/cncf/dco2`; `github.com/cncf/foundation` charter;
  `eclipse.org/legal/eca/eca.html`; GitHub Docs sign-off policy
- Kernel, LLVM, QEMU, curl docs in their repositories (paths in the table); Gentoo wiki; NetBSD commit guidelines;
  Fedora Council policy; Linux Foundation generative AI policy
- README examples: `github.com/antirez/ds4`, `github.com/ghostty-org/ghostty`, `github.com/ggml-org/llama.cpp`
