# 43: Coding-agent harnesses: one source of truth for "optimize Tona for my hardware"

Research of 2026-11-11 for the AI-first contributor flow (#111). Question: contributors use different agents; how do we
cover as many as possible with one source of truth and the least duplication? [V] = vendor documentation page fetched
and read on 2026-11-11, [S] = secondary source (search result, blog, issue thread, forum) or a vendor page seen only as a
search snippet. A cell marked "unverified" has no primary source yet.

## Result in short

1. **`AGENTS.md` is the de-facto standard.** agents.md (stewarded by the Agentic AI Foundation under the Linux
   Foundation, 23 tools listed) names Codex, Jules, Factory, Aider, goose, opencode, Zed, Warp, VS Code, Devin, Junie,
   Amp, Cursor, RooCode, Gemini CLI, Kilo Code, Copilot coding agent, Windsurf, Augment and others [V agents.md]. Every
   harness in the task list reads it natively except **Claude Code** (reads it only when no `CLAUDE.md` exists, so the
   `@AGENTS.md` import stays) and **Aider** (needs one config line). Continue is unresolved.
2. **Reusable commands converged on Agent Skills** (`SKILL.md` in a folder, [agentskills.io](https://agentskills.io)).
   Slash-command files are being merged into skills or deprecated: Claude Code merged commands into skills, Cursor
   converts commands to skills (`/migrate-to-skills`), Codex deprecated custom prompts, Copilot prompt files run only in
   the Local harness that is scheduled for removal in 02/2027, Gemini CLI itself is retired. The portable unit is one
   `SKILL.md`. The cross-client directory is `.agents/skills/`; Claude Code and Copilot do not read it and need their
   own directory.
3. **Recommendation:** keep `AGENTS.md` as the only instruction text, keep `CLAUDE.md` = `@AGENTS.md`, add **one tiny
   skill** `optimize-for-hardware` whose body only points to `docs/contribute/optimize-for-your-hardware.md`, and publish it to the
   three directories (`.agents/skills/`, `.claude/skills/`, `.github/skills/`) as **generated copies checked by a gate**,
   not symlinks. Do not add `GEMINI.md`, `.cursor/rules`, `.github/copilot-instructions.md`, `.clinerules`, `.junie/`
   and similar stubs: they add nothing and some shadow `AGENTS.md` (see "Priority traps").
4. **The playbook must start with an environment check**: cloud sandboxes (Codex cloud, Jules, Copilot cloud agent,
   Claude Code on the web) have no access to the contributor's GPU and cannot produce measurements for this project.

## 1. Harness table

Columns: instruction files read, reads `AGENTS.md`, imports, directory scoping, command/skill format, runs on the
contributor's machine.

| Harness | Instruction files (auto) | `AGENTS.md`? | Imports | Scoping | Commands / skills | Local? |
|---|---|---|---|---|---|---|
| **Claude Code** [V] | `CLAUDE.md`, `.claude/CLAUDE.md`, `CLAUDE.local.md`, `.claude/rules/*.md` (`paths:`) | Only if no `CLAUDE.md` in cwd or above (v2.1.277+); setting `claude-md-and-agents-md` reads both | `@path`, relative to the importing file, depth 4, external imports need approval | Parents at launch; subdirectory files on demand | Skills `.claude/skills/<n>/SKILL.md`; `.claude/commands/*.md` merged into skills; `/name`. Does not read `.agents/skills` [S] | CLI/IDE local; web/cloud sessions separate |
| **OpenAI Codex** CLI/IDE [V] | `AGENTS.override.md`, `AGENTS.md`, `project_doc_fallback_filenames`; global `~/.codex/` | Native | none | Root to cwd, concatenated, nearest last; one file per directory; **32 KiB total** (`project_doc_max_bytes`) | Skills in `.agents/skills` (cwd up to repo root), `$name` or `/skills`; follows symlinked skill folders. Custom prompts deprecated [S] | Local |
| **Codex cloud** [S] | `AGENTS.md` (lint/test commands) | Native | none | as above | Skills: not stated | **No**, container; agent-phase internet off by default |
| **GitHub Copilot** (VS Code agent mode, CLI, cloud agent, review) [V] | `.github/copilot-instructions.md`; `.github/instructions/*.instructions.md` (`applyTo`); `AGENTS.md`; or single root `CLAUDE.md`/`GEMINI.md` | Yes (cloud agent and code review documented; "nearest wins") | none documented | `applyTo` globs; nearest `AGENTS.md` | Skills `.github/skills`, `.claude/skills` (Codex integration `.agents/skills`); `.github/prompts/*.prompt.md` run only in the Local harness, removal planned 02/2027 | VS Code/CLI local; cloud agent **no** |
| **Cursor** [V] | `.cursor/rules/*.mdc` (always / intelligent / globs / manual), `AGENTS.md`; legacy `.cursorrules` | Yes, root and nested | none; "reference files instead of copying" | Nested `AGENTS.md` for its directory | Skills `.agents/skills`, `.cursor/skills`, also reads `.claude/skills`, `.codex/skills`; `/name`; commands convert to skills. Rules: keep under 500 lines | Local (cloud agents separate) |
| **Windsurf** (Cognition) / Devin desktop [V] | `AGENTS.md`, `AGENTS.local.md`, `.devin/rules/*.md`, imports `.windsurf/rules`, `.windsurfrules` | Yes, root and each directory down to cwd | none documented | Directory chain | Skills `.devin/skills`, **`.agents/skills`**, `.windsurf/skills`; `/name` | Local |
| **Gemini CLI** [V] | `GEMINI.md` hierarchy, `context.fileName` accepts a list (example contains `AGENTS.md`) | Via setting | `@file.md` | Workspace, parents, just-in-time on tool access | Custom commands `.gemini/commands/*.toml` (not read, link only) | Page banner: replaced by Antigravity CLI on 2026-06-18 |
| **Antigravity CLI** (`agy`) [S] | `AGENTS.md`, legacy `GEMINI.md` still read; `AGENTS.md` wins [S] | Yes | not stated | not stated | `.agents/skills/`; workflows `.agents/workflows` (codelab only, unverified) | Local |
| **Google Jules** [V/S] | `AGENTS.md` in repo root | Yes | n/a | root | none (task prompt) | **No**, short-lived cloud VM with internet [S] |
| **JetBrains Junie** [V] | `.junie/AGENTS.md`, else `AGENTS.md` (+ `.junie/playbook.md`, `.junie/rules/*.md`), legacy `.junie/guidelines.md`; global `~/.junie/AGENTS.md` | Yes (priority 1-2) | none | project root | custom slash commands exist (not read) | Local |
| **Aider** [V] | `CONVENTIONS.md` only by `--read` / `.aider.conf.yml` `read:` | **No** by default; `read: AGENTS.md` per agents.md | none | none | none; `/read`, `/run` | Local; no autonomous agent loop |
| **Cline** [V/S] | `.clinerules/` or `.cline/rules/`, single `.clinerules`, `.cursorrules`, `.windsurfrules`, `AGENTS.md` | Yes (fallback, root and subdirectories [S]) | none | `paths:` frontmatter | Workflows `.clinerules/workflows/*.md`, skills `.cline/skills` [S]; `.agents/skills` unverified | Local |
| **Roo Code** [V/S] | `.roo/rules/`, `.roorules`, mode rules, `AGENTS.md`/`AGENT.md` root | Yes (`useAgentRules`) | none | root | `.roo/commands/*.md`, `.roo/skills`. **Repository archived 2026-05-15 [S]**: legacy | Local |
| **Continue** [V/S] | `.continue/rules/*.md` (`globs`, `alwaysApply`) | **Unresolved**: one source says merged 2025-09 (PR #7717), another says only `.continue/rules` [S] | none | globs | `invokable: true` markdown prompts [S] | Local |
| **Zed** [S] | One project file, first match of `.rules`, `.cursorrules`, `.windsurfrules`, `.clinerules`, `.github/copilot-instructions.md`, `AGENT.md`, `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`; global `~/.config/zed/AGENTS.md` | Yes, but only if no earlier file exists | none | root only (unverified) | slash commands; no skills documented | Local |
| **opencode** [V] | `AGENTS.md`; fallback `CLAUDE.md`, `~/.claude/CLAUDE.md`; `instructions` in `opencode.json` (paths, globs, URLs) | Native, first | **`@file` not parsed**; use `instructions` | Per directory | `.opencode/commands/*.md` (frontmatter `description`, `agent`, `model`; `$ARGUMENTS`, `$1`); reads `~/.claude/skills` | Local |
| **Amp** [V] | `AGENTS.md` up to `$HOME`, `~/.config/amp/AGENTS.md`; fallback `AGENT.md`, `CLAUDE.md` | Native | `@path` incl. globs, `globs:` frontmatter | Subtree files load when a file there is read | Skills `.agents/skills`, then `.claude/skills` (project and parents) | Local |
| **Kiro** [V] | `.kiro/steering/*.md` (`always`, `fileMatch`, `manual`, `auto`), `AGENTS.md` | Yes, global, root, subdirectories (always on) | `#[[file:path]]` | per steering file | steering as slash command (`manual`); skills not stated | Local; Kiro Web cloud |
| **Devin** (cloud) [V/S] | `AGENTS.md`, Knowledge, Playbooks (platform side) | Yes | n/a | nested files [V] | Playbooks (platform) | **No**, cloud VM |
| **Goose** [S] | `.goosehints`, `AGENTS.md`; `CONTEXT_FILE_NAMES` overrides | Yes (docs conflict on default) | n/a | nested | skills `.agents/skills`, `.goose/skills`, `.claude/skills` | Local |
| **Warp** [S] | `AGENTS.md`, `WARP.md` (WARP.md wins in the same directory) | Yes (default) | n/a | root and cwd; others best-effort | skills `~/.agents/skills`; `/skills` | Local |
| **Augment** [S] | `AGENTS.md`, `CLAUDE.md` (AGENTS.md used if both), `.augment/rules/` (root only) | Yes | n/a | subdirectories for AGENTS/CLAUDE | custom commands unverified | Local |
| **Factory Droid** [S] | `AGENTS.md` (nested ok) | Yes | n/a | nested | skills `.factory/skills`; commands `.factory/commands` merged into skills; import from `.agents/commands`, `.claude/commands` | Local |

Sources (all 2026-11-11): [agents.md](https://agents.md); [Claude Code memory](https://code.claude.com/docs/en/memory);
[Claude Code skills](https://code.claude.com/docs/en/skills); [Codex AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md);
[Codex skills](https://learn.chatgpt.com/docs/build-skills); [Codex custom prompts](https://developers.openai.com/codex/custom-prompts);
[Codex cloud environments](https://developers.openai.com/codex/cloud/environments);
[GitHub: repository instructions](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions);
[VS Code skills](https://code.visualstudio.com/docs/copilot/customization/agent-skills);
[VS Code prompt files](https://code.visualstudio.com/docs/copilot/customization/prompt-files);
[Cursor rules](https://cursor.com/docs/context/rules); [Cursor skills](https://cursor.com/docs/context/skills);
[Windsurf/Devin AGENTS.md](https://docs.devin.ai/desktop/cascade/agents-md);
[Gemini CLI GEMINI.md](https://geminicli.com/docs/cli/gemini-md/);
[Flutter docs on Antigravity CLI](https://docs.flutter.dev/ai/antigravity-cli.md); [Jules docs](https://jules.google/docs);
[Junie guidelines](https://junie.jetbrains.com/docs/guidelines-and-memory.html);
[Aider conventions](https://aider.chat/docs/usage/conventions.html); [Cline rules](https://docs.cline.bot/customization/cline-rules);
[Roo custom instructions](https://roocodeinc.github.io/Roo-Code/features/custom-instructions);
[Continue rules](https://docs.continue.dev/customize/deep-dives/rules); [opencode rules](https://opencode.ai/docs/rules/),
[commands](https://opencode.ai/docs/commands/); [Amp AGENTS.md](https://ampcode.com/docs/customize/agents-md),
[skills](https://ampcode.com/docs/customize/skills); [Kiro steering](https://kiro.dev/docs/steering/);
[Agent Skills client guide](https://agentskills.io/client-implementation/adding-skills-support).
[S] only: Zed (rules page 404 on fetch; priority list from the Zed repository commits and guides), Goose, Warp,
Augment, Factory, Devin limits, Jules limits, Antigravity details, Continue, Roo shutdown.

### Findings that change the design

- **Claude Code ignores `AGENTS.md` when any `CLAUDE.md` or `CLAUDE.local.md` exists** in cwd or above [V]. Our
  `CLAUDE.md` = `@AGENTS.md` is the documented pattern ("keeping the import never makes Claude read `AGENTS.md` twice").
  A maintainer's `CLAUDE.local.md` also switches the direct read off; harmless because of the import.
  Claude Code before v2.1.281 reads only `CLAUDE.md` in some sessions (Bedrock, telemetry off) [V], another reason to
  keep the import.
- **Nested `AGENTS.md` in Claude Code** (`engine/backend/<name>/AGENTS.md`): the doc says a subdirectory `AGENTS.md`
  loads on demand "when that subdirectory has none of the three `CLAUDE.md` files of its own", but the same list sits
  under the case "no `CLAUDE.md` above". Whether the nested file loads when the root has a `CLAUDE.md` is **not
  clear from the text; test with `/context all`** before relying on it. Safe variant: a generated one-line
  `CLAUDE.md` (`@AGENTS.md`) next to each nested `AGENTS.md`.
- **Nested scoping is not universal.** Codex (root to cwd, concatenated), Copilot, Cursor, Amp, Kiro, Cline, Junie
  (root only per doc), Devin, Factory handle it; Zed reads one root file (unverified), opencode scopes per directory.
  Portable fallback: one sentence in the root `AGENTS.md`: "before changing files under `engine/backend/<name>/`, read
  that directory's `AGENTS.md`". Cheap and works in every harness that can read a file.
- **Size.** Our `AGENTS.md` is 149 lines and 10.9 KB. Limits found: Codex 32 KiB total of all chained files
  (`project_doc_max_bytes`) [V], Claude Code "under 200 lines per file, skip above 4 MiB" [V], Cursor "rules under 500
  lines" [V], Windsurf/Antigravity rule-file limits in the 6k to 12k character range [S, unverified]. We fit all of
  them; the budget for nested backend files is roughly 20 KB for Codex. Keep the root file below 200 lines and below
  ~12 KB.
- **Links are not followed automatically.** Only Claude Code (`@`), Gemini (`@`), Amp (`@`), Kiro (`#[[file]]`) and
  opencode (`instructions`) expand references; opencode explicitly does not parse `@file` in `AGENTS.md` [V]. All
  others see a Markdown link as text and open it only if the model decides to. Therefore the trigger sentence that
  points to the playbook must stay in `AGENTS.md` itself (it is already there, line 6) and name the exact path.

### Priority traps for stub files

Some harnesses pick the **first** file from a list, so a pointer stub can shadow `AGENTS.md`:

- **Zed** order: `.rules`, `.cursorrules`, `.windsurfrules`, `.clinerules`, `.github/copilot-instructions.md`,
  `AGENT.md`, `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, first match only [S]. A `.github/copilot-instructions.md` stub
  would be read **instead of** `AGENTS.md`.
- **Warp**: `WARP.md` beats `AGENTS.md` [S]. **Gemini CLI**: `GEMINI.md` unless `context.fileName` lists `AGENTS.md`.
- **Cursor** merges all rules ("earlier sources take precedence") [V]; a `.cursor/rules` stub adds a second source.
- **Junie**: `.junie/AGENTS.md` beats root `AGENTS.md` [V].

Rule: no stub whose content is only "read AGENTS.md" in any harness that already reads `AGENTS.md`. The harness that
needs one is Claude Code (`CLAUDE.md`, exists) and Aider (config line, no instruction file).

## 2. Reusable "optimize Tona for my hardware"

What exists per harness (all [V] unless marked):

| Harness | Mechanism | File | Invocation |
|---|---|---|---|
| Claude Code | Skill; commands merged | `.claude/skills/<n>/SKILL.md` (or `.claude/commands/<n>.md`) | `/optimize-for-hardware`, or description match |
| Codex | Skill (prompts deprecated) | `.agents/skills/<n>/SKILL.md` | `$optimize-for-hardware`, `/skills`; no bare `/name` yet (issue openai/codex#13893 [S]) |
| Copilot | Skill; prompt files legacy | `.github/skills/<n>/SKILL.md` | `/optimize-for-hardware` in chat |
| Cursor | Skill; commands convert | `.agents/skills/<n>/SKILL.md` | `/optimize-for-hardware` |
| Windsurf, Devin desktop | Skill | `.agents/skills/` | `/name` |
| Amp, Goose, Warp, Antigravity | Skill | `.agents/skills/` | `/name`, `/skills` |
| opencode | Command file; reads `.claude/skills` | `.opencode/commands/<n>.md` or `.claude/skills` | `/optimize-for-hardware` |
| Cline | Workflow | `.clinerules/workflows/<n>.md` | `/<n>` |
| Gemini CLI (retired) | TOML command | `.gemini/commands/<n>.toml` | `/<n>` |
| Aider, Zed, Junie, Jules, Codex cloud | none useful | n/a | the sentence in `AGENTS.md` |

**Portable format: one `SKILL.md`** (YAML frontmatter `name`, `description`, Markdown body). Claude Code accepts extra
keys; outside Claude Code only `name`, `description`, `license`, `compatibility`, `metadata`, `allowed-tools` are
valid [V]. So the file uses only the portable keys. `disable-model-invocation` is Claude/Cursor/Copilot specific and
is omitted so the file stays valid everywhere; a description that names the exact user phrase triggers it.

Proposed body (about 12 lines, no content of its own):

```markdown
---
name: optimize-for-hardware
description: Optimize Tona for the user's hardware and end with a finished pull request text. Use when the user says
  "optimize Tona for my hardware" or asks to port or tune Tona for their GPU, CPU or Mac.
---
Read AGENTS.md completely, then follow docs/contribute/optimize-for-your-hardware.md from step 0 to the end. That page is the only
source of the steps; this file only points to it. Stop at every ASK marker and wait for the user.
```

Without the skill the flow still works: `AGENTS.md` line 6 triggers on the phrase in every harness that loads it. The
skill adds discoverability (`/` menu) and implicit invocation, not new content.

Not recommended: Copilot prompt files (Local harness removal 02/2027 [V]), Cursor `.cursor/commands` (converted to
skills by Cursor itself), Codex `~/.codex/prompts` (deprecated, user-level only), Gemini TOML (tool retired),
opencode/Cline command files (covered by skills or by the `AGENTS.md` sentence; add only on demand).

## 3. Capability differences the playbook must handle

| Capability | Differs how | Playbook consequence |
|---|---|---|
| **GPU / user hardware** | Codex cloud, Jules, Copilot cloud agent (Actions runner or cloud sandbox, "doesn't mention GPU" [V]), Claude Code on the web (Anthropic VM), Devin: none see the contributor's GPU. Copilot Actions sessions can use **self-hosted runners** (a runner on the contributor's own PC would be real hardware; 59 min hard limit [V]); cloud sandboxes cannot [V] | Step 0 "where am I running": detect sandbox, stop measuring, see below |
| **Long jobs (20+ min)** | Tool calls have timeouts and the session may end: Claude Code cloud sessions stop after inactivity and drop running shell commands [V]; Copilot Actions session 59 min [V]; local harnesses have per-command timeouts and some have no background jobs (Aider, plain chat IDEs) | Start every long run detached (`nohup`/`systemd-run --user`), logging to `work/` and writing a result file; poll with short commands; the measurement file's plan plus the log are the resume state; never wait inside one tool call |
| **Questions mid-work** | Interactive terminals and IDE chats can ask; non-interactive modes (`codex exec`, `claude -p`, Jules, cloud agents, scheduled runs) cannot, an unanswered question ends or stalls the run | At each ASK marker: if the session is not interactive, do not guess "yes"; write the open ASK into the PR text under "Open questions" and stop that branch of work |
| **Shell access** | Aider has no agent loop; Zed/Continue/IDE chats may need per-command approval; some harnesses sandbox the network | If no shell: print the exact commands for the human to run and wait for pasted output. All steps are commands (`tools/hw_report.py`, `tools/quiet_check.py`, `just check`) |
| **Network** | Codex cloud: agent phase off by default, setup scripts online [S]; Claude cloud: "Limited" default allowlist [V]; Jules has internet [S] | Downloads (model, toolchain) happen in setup or on the user's machine; in a restricted sandbox say what could not be fetched |
| **GPU safety, desktop hang** | Any local harness can start the kernel | The existing ASK for risky GPU tests stays; additionally: never run it from an agent that cannot be interrupted by the human (headless, scheduled) |
| **Sandbox write limits / approvals** | Codex sandbox, Cursor, Copilot ask per command | Playbook lists commands in groups so a human can approve once (build, measure, gates) |
| **Context loss** | All harnesses compact long sessions; Claude Code cloud loses background work after VM reclaim [V] | State lives in files: measurement file plan, `work/quiet-before.json`, a `work/optimize-state.md` checklist of the 12 steps ticked |
| **Attribution** | Each harness has its own default "Co-authored-by"/"Generated with" trailer | AGENTS.md rule (Assisted-by only) already says this; the playbook repeats it at step 12 because harness defaults override quietly |

### Suggested additions to `docs/contribute/optimize-for-your-hardware.md`

**Step 0, before step 1: where does this run?** Four checks, in this order:

1. Is this the machine that will measure? A cloud or CI sandbox is recognizable by: no GPU in `tools/hw_report.py`,
   `CI=true`, `GITHUB_ACTIONS`, `/.dockerenv`, hostname/user typical for the harness, or the user saying so. Then the
   agent says: "This session runs in a cloud sandbox without your hardware. I can read, plan, write code and run the
   CPU gates (`just test-cpu`, `just check`), but **no measurement counts**. For a measurement, run the same prompt in
   a local agent on your machine." It must not write measurement files with sandbox numbers and the PR text must say
   "not measured on target hardware".
2. Is the session interactive? If not, the ASK rule above applies.
3. Can this session survive 20+ minutes? If it cannot run detached jobs, hand the long run to the human as a command.
4. Does the shell exist? Else switch to "human runs, agent reads".

Also: `tools/hw_report.py` should print a `sandbox:` line (container/CI detection) so the Test conditions block shows
it; maintainers reject measurements from a sandbox on sight.

**State file**: tell the agent to keep `work/optimize-state.md` (ignored by git, numbers only, no program names) with
the 12 steps ticked, so a compacted or restarted session resumes.

**Headless contributor**: a short paragraph "What if I only have a cloud agent?" in `CONTRIBUTING.md`: it can prepare
the change and CPU gates, but the measurement needs one local run.

## 4. Symlink vs stub vs generation

| Option | Works | Breaks |
|---|---|---|
| **Symlink** (`CLAUDE.md -> AGENTS.md`, `.claude/skills -> ../.agents/skills`) | Claude Code reads through it; Codex follows symlinked skill folders [V]; Claude docs allow symlinked skill directories [S] | **Windows**: git checks a committed symlink out as a text file with the target path unless `core.symlinks` is on (needs admin or Developer Mode), "which leaves that clone with a one-line `CLAUDE.md` in place of your instructions" [V Claude docs]; Claude's Edit/Write refuse to write through a symlink [V]; Cursor has open bugs on symlinked rule folders, skills after restart, symlinks outside the project root [S forum.cursor.com]; Cowork skips symlinked `~/.claude/CLAUDE.md` [V]; tar/zip downloads of the repo (GitHub "Download ZIP") store links badly |
| **Stub with pointer** ("read AGENTS.md") | Any tool that loads the stub | Relies on the model opening the file; shadows `AGENTS.md` in first-match harnesses (Zed, Warp, Junie); drifts if it carries content |
| **Import** (`@AGENTS.md`) | Claude Code, Gemini, Amp: real file, loaded at start, Windows safe | Not parsed by opencode, Codex, Copilot; only for files that those tools do not need |
| **Generated copies + gate** | Plain files everywhere, no symlink risk, content identical by construction | Needs a small script and a pre-commit hook; copies show up in diffs |

**Recommendation**

- `AGENTS.md`: the single text. Exists. No stubs for harnesses that read it.
- `CLAUDE.md`: keep `@AGENTS.md` (import; Windows-safe; documented). No symlink.
- Skill: one template `tools/agent_files/optimize-for-hardware/SKILL.md`, a script `tools/gates/agent_files.py` writes
  it to `.agents/skills/`, `.claude/skills/` and `.github/skills/`, and the same script with `--check` runs as a
  pre-commit hook (same pattern as the other gates: generated files must equal the template). Three 12-line files,
  no symlink. If the cross-client convention `.agents/skills/` is later read by Claude Code and Copilot
  ([claude-code#31005](https://github.com/anthropics/claude-code/issues/31005) is the open request [S]), delete the
  two extra copies.
- Aider: `.aider.conf.yml` with `read: AGENTS.md` (documented on agents.md [V]); Aider reads it from the repo root
  [S: aider docs, not fetched].
- Nested backend files: per `engine/backend/<name>/` a real `AGENTS.md`; plus, after testing with `/context all`, a
  generated `CLAUDE.md` (`@AGENTS.md`) beside it if Claude Code does not load the nested file on its own.
- Files not to create: `GEMINI.md` (Antigravity reads `AGENTS.md` [S]), `.github/copilot-instructions.md` (Copilot
  reads `AGENTS.md` [V]; Zed would pick the stub first), `.cursor/rules`, `.clinerules`, `.junie/`, `.windsurf/`,
  `.kiro/steering`, `WARP.md`. Continue is the one open case: if its maintainers confirm no `AGENTS.md` support,
  a `.continue/rules/tona.md` generated stub is the only candidate; Continue's rule files load in lexical order and
  are additive [V], so no shadowing.

## 5. Open points (unverified)

- Claude Code: does a nested `engine/backend/<name>/AGENTS.md` load when the root has `CLAUDE.md`? Test with
  `/context all` after reading a file in that directory.
- Copilot Chat on github.com and older VS Code versions: documented only for `copilot-instructions.md`; the agent and
  cloud agent read `AGENTS.md`. Whether to add a stub for Chat depends on the Zed shadowing trade-off above.
- Cursor: reads `CLAUDE.md`? (not in the pages read; irrelevant for us since `AGENTS.md` works.)
- Windsurf/Antigravity rule-file size limits (6k to 12k characters) from third-party sources only.
- Cline and Roo: whether `.agents/skills/` is scanned; Cline `.cline/skills` [S].
- Codex cloud task time limit: no official figure found (a blog says 24 h [S]); agent-phase internet default off [S].
- Jules limits and network: FAQ says internet access, quotas vary by plan; no figures verified.
- Roo Code shutdown (2026-05-15) from a repository page and a blog [S], no official announcement found.
- Gemini CLI retirement date 2026-06-18 [V banner on geminicli.com, S Google migration guides]; the Antigravity
  context-file behavior is from the Flutter docs mirror [S].
- `AGENTS.md` in Aider: confirm that `.aider.conf.yml` at the repo root is picked up when Aider starts in a
  subdirectory.
