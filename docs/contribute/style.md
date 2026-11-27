# Writing style

For every text in this repository: docs, README, screens of `tona`, pull requests, commits,
release notes. Readers give a text a few seconds. Write for that.

## The rules

1. **Action first.** The first lines are what the reader does: a command, a table, a choice.
2. **One sentence per point.** One fact per sentence, 1 to 3 sentences per paragraph.
3. **A reason only where the reader needs it** to act or to avoid breaking something. Never two reasons for one rule.
   Never the same reason on two pages: link instead.
4. **Cut, do not add.** If a sentence can go without the reader doing anything wrong, it goes.
5. **Numbers, not adjectives.** "68 tok/s decode", not "fast". Name every rate: decode or prefill.
6. **Plain English.** Short words. Sentence-case headings.

## Pages

| Part | Rule |
|---|---|
| Title line | one sentence, e.g. "Your local AI for any task." |
| Start | the command or table, no introduction ("This page describes ...") |
| Facts | tables and lists over paragraphs; commands in code blocks |
| Details | on their own page, linked once |
| History, decisions, measurements | in `research/` or a linked measurement file, not in user docs |
| Anchors | keep every heading another file links to (`tools/gates/md_links.py`) |

The README follows [ollama](https://github.com/ollama/ollama) and [llama.cpp](https://github.com/ggml-org/llama.cpp):
title, one sentence, install commands, a few usage commands, short tables, links. Under 100 lines.

## Screens (tona)

- Show what the user decides or acts on. Nothing else.
- A label only where rows differ. `[x]` already marks the choice: no "Recommended" on every row.
- No word that is always the same ("measured" on every model).
- Speeds: one line per page that says for which system they hold.
- No explanations of words on the main screen ("Token = word piece"); they go in help (`?`).
- No hints, nags or status the user did not ask for ("You are working: ..."). An update or a running job: one line.
- Short keywords, never a cut-off sentence: "strong: browsing, agents, 256K context".
- Screens are English only.

## Words to drop

| Drop | Use |
|---|---|
| robust, seamless, powerful, comprehensive, crucial, key, leverage, delve, showcase, boasts | a number or nothing |
| ensure, enables, serves as, allows you to | is, has, does, can |
| it is worth noting, note that, simply, just, in order to, it is important | nothing |
| Moreover, Furthermore, Overall, In short | nothing |
| "not X but Y", "not just", "no X, no Y, just Z", three adjectives for rhythm | the plain fact |
| "..., ensuring ...", "..., making ..." | two sentences |
| em dashes, semicolon chains, parenthetical asides | a new sentence, or drop it |
| "vendor:", "basis:", source labels on screens | the fact; the source goes in details |

Bold only for a real warning.

## Pull requests, commits, release notes

- Title: Conventional Commit, the change in under 70 characters ([CONTRIBUTING.md](../../CONTRIBUTING.md)).
- Body: what changed and the result, as a short list. Numbers with their file. No story of how it was found.
- Release notes: what a user notices, one line per change.

## Examples

| Before | After |
|---|---|
| "Local AI instead of one frontier model: the right model for each task, tuned for your hardware, with the tools to use it." | "Your local AI for any task." |
| "Coming, not built yet (no model for them in Tona today, no date promised)" | "Coming:" |
| "[x] Coder  Recommended  decode ≈ 68 tok/s at 16K  58 GB  measured" | "[x] Coder  68 tok/s  58 GB" plus one line "Speed on this PC" |
| "strong: vendor: strong on agent and browsing tasks, 256K context  weak…" | "strong: browsing, agents, 256K context" |
| "You are working: Tona gives your screen priority." | nothing |
