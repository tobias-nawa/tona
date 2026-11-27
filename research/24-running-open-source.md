# Open source in the AI era: contributions, platforms, rules (release preparation)

2026-10-27. **Agent research** (Sonnet, web only) and an own measurement of the code (lizard). UNVERIFIED marks what was not checked against a primary source.

## AGENTS.md and CLAUDE.md

- AGENTS.md: free Markdown, the file nearest to the edited path applies, large repos get one per subproject
  ([agents.md](https://agents.md/), 2026-10-27). Read by Codex, Cursor, Copilot agent, Gemini CLI (with
  `context.fileName`), Jules, Aider, Zed and others; steward: Agentic AI Foundation (Linux Foundation).
- Claude Code ([memory](https://code.claude.com/docs/en/memory), 2026-10-27): reads AGENTS.md itself, but only if
  no CLAUDE.md exists. Recommended for repos with AGENTS.md: CLAUDE.md imports it (`@AGENTS.md`), below it only
  Claude-specific content. Import instead of symlink (Windows). CLAUDE.md is context, not enforcement: hard rules via hook
  or gate.
- Study July 2026: agents mostly ignore a repo's published AI rules (arXiv 2605.16706, UNVERIFIED). So
  state rules explicitly in AGENTS.md **and** as a gate.

## AI rules of other projects (state 2025/2026)

| Project | Rule |
|---|---|
| llama.cpp | allowed; disclosure mandatory, "100 % responsible for every line", must be able to explain every line ([CONTRIBUTING](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)) |
| LLVM | a human reviews everything before review, labeling (e.g. `Assisted-by:`), no autonomous agents, "a contribution must be worth more than its review" ([AIToolPolicy](https://llvm.org/docs/AIToolPolicy.html)) |
| Linux | disclose tools, the contributor must be able to defend it; maintainers may reject without review ([generated-content](https://docs.kernel.org/process/generated-content.html)) |
| Mesa | allowed, trailer `Assisted-by:` / `Generated-by:` ([submittingpatches](https://docs.mesa3d.org/submittingpatches.html)) |
| Ghostty | disclosure for outsiders, maintainers exempt ([AI_POLICY](https://github.com/ghostty-org/ghostty/blob/main/AI_POLICY.md)) |
| curl | same standards; disclose AI security reports and check them yourself ([CONTRIBUTE](https://github.com/curl/curl/blob/master/docs/CONTRIBUTE.md)) |
| Gentoo, NetBSD | ban or only with approval (license reasons) |

Majority: allow, disclose, a human is liable. 78 % of 118 rules allow AI, 51 % require disclosure (UNVERIFIED).

## Platforms the maintainers do not own

- Rust ([target-tier-policy](https://doc.rust-lang.org/rustc/target-tier-policy.html)): tier 3 = named
  maintainer, docs, no CI, no guarantee; tier 2 = at least 2 maintainers, builds in CI; removal when a target is
  orphaned.
- LLVM ([DeveloperPolicy](https://llvm.org/docs/DeveloperPolicy.html)): new backends "experimental", off by default,
  maintainer in the first merge, promotion after 3 months stable with public buildbots.
- llama.cpp: CODEOWNERS per backend, the contributor provides CI including hardware, operator tests + perplexity + bench mandatory.

## Code gates

- clang-tidy `readability-function-size`: by itself only 800 statements; lines, nesting, parameters off.
  `readability-function-cognitive-complexity` default 25 (Sonar: 15 reported too much).
  OpenBMC: 60 lines, 6 parameters. Google: "think about splitting from ~40 lines", no hard limit.
- `clang-tidy-diff.py`: only warnings on changed lines, needs `compile_commands.json`.
- Format, lint and security checks need no GPU. Whether clang-tidy parses `.hip` cleanly: UNVERIFIED, try it out.
- Hardly any large project publishes file limits; the usual solution: a script of its own.

### Gate design

- File gate 500 lines, method gate (lizard) 80 lines, 6 parameters, nesting 5, CCN 15.
- Rationale: an agent reads a file up to 500 lines entirely, above that only samples and overlooks wiring.
- Gates as pre-commit hooks and in `just lint`; CI only as a supplement.

## Own measurement (lizard, without third_party, 2026-10-27)

888 functions in 82 files. Over 80 lines: 41, over 200: 9. CCN over 15: 97. Largest: `Mega::build` 1,100 lines
(CCN 274), `ie_gpu_gen main` 663, `prefill_chunk_run` 616, `ie_server main` 491. Files over 500 lines: mega.hip
5,188, prefill.hip 1,976, ie_server.cpp 1,096, matvec_dev.h 1,039, generator.cpp 949, gemm.hip 795, ie_gpu_gen.cpp
720, mega.h 676, ie_top.cpp 656, experts.cpp 560, gpubench.hip 502.

## Release (UNVERIFIED, own draft)

No primary source found. Draft: the release depends only on the reference (tests, BENCHMARKS.md re-run, quality);
other backends "experimental, checked by the maintainer" with the version the maintainer last checked; orphaned ones drop out.
