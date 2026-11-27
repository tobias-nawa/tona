# Coding-agent harnesses against ie-server

Agent research, as of 2026-10-23. The exception is the section "Own captures": we measured that one
ourselves on 2026-10-23, and it is marked accordingly.

Goal: tune `ie-server` for the common coding harnesses. The server has an Anthropic and an
OpenAI chat API, one model, one request at a time and one checkpoint at the end of the latest prompt.
Prefill runs at ~40 tok/s, decode at 25–50 tok/s, context 16K.

The basis is four research agents. They read source code at the named tags, plus docs, issues
and PRs (via `gh` and the web). Without a measured source, token counts are estimated from character
lengths (chars/4) and marked as **estimate**. Measured numbers come either from our own captures or from the
source named in each case. What has no source is marked **not substantiated**.

## Summary

1. **One checkpoint is not enough in any harness.** All of them send side requests with their own prompt:
   - Title: Claude Code, OpenCode, Crush, Goose, Zed, Kilo, Continue, Codex TUI, OpenHands server
   - Suggestions, memory, summary, commit message, compaction

   Each of these requests overwrites the checkpoint. The next main request is then recomputed in full:
   ~5–8 min at 12–20K tokens.
2. **A prefill without a single byte breaks the client timeouts.** ie-server sends `message_start` or the
   first chunk only after the prefill. At 40 tok/s, a 12K prompt is done after ~300 s, and that is exactly where
   most default timeouts sit: Claude Code, Codex, OpenCode, Kilo, Cline < 4.1.25, Roo/Zoo and
   OpenHands. The client then repeats the whole request.
3. **The startup prompts are mostly too large for 16K.** Measured or substantiated:
   - Claude Code interactive ~20K
   - Qwen Code ~14–15K
   - Crush ~12K
   - Codex ~11K
   - OpenHands ~10–12K (estimate)

   In a lean configuration:
   - Goose ~1.4K
   - Claude Code `--bare` ~1.0K, with 6 tools ~4.8K
   - OpenCode V2 ~3K (estimate)
   - Cline v4 ~5K (estimate)
4. **Prefix stability within a session is mostly good.** The main requests only append:
   - measured for Claude Code, Codex, OpenCode and Crush
   - per the agent's measurement also for Goose

   Exceptions:
   - Qwen Code: a wandering memory reminder breaks every round
   - Aider: file block behind the history, repo map, summary
   - Harnesses that do not send old thinking back or serialize it differently: OpenHands, Roo/Zoo, Zed
     (default), Aider
5. **Codex speaks only the Responses API now.** `wire_api = "chat"` is rejected (checked ourselves with
   0.164.0). Without `/v1/responses`, Codex does not run.
6. **Ranking of the changes** (section "Conclusions"):
   1. Keepalive during the prefill
   2. Several checkpoints, selected by longest prefix, in RAM
   3. Detect side requests
   4. Render the history token-identically
   5. `/v1/responses`
   6. Context to 32K
   7. Ship client profiles

## Overview

Tokens before the actual task: system, tools and preamble, measured with the Qwen tokenizer (**M**)
or estimated (S).

| Harness, version | API to our own endpoint | Tokens upfront, default → lean | Prefix over rounds | Side requests (prefix killers) | Notes |
|---|---|---|---|---|---|
| **Claude Code** 2.1.309 | Anthropic Messages | 20.1K (M, interactive) / 11.2K (M, `-p`) → 4.8K (M) / 1.0K (M, `--bare`) | stable, append only (M) | title per prompt, suggestion per turn, recap, compaction | `role:"system"` in the middle of the history; header `x-claude-code-request-class` (M); 300 s watchdog |
| **Codex CLI** 0.164.0 | **Responses only** | 10.8K (M) → smaller via `model_instructions_file` (size not measured) | stable (M); environment diffs are appended | title (TUI only, parallel), compaction | `/models` in Codex format; 300 s idle, 5 retries; without `model_context_window` 272K is assumed |
| **OpenCode** V1 1.18/1.19, **V2** 2.1.4 | Chat Completions, Responses, Anthropic (per package) | V1 7.1K (M) + skills; V2 ~3K (S) | V1 stable (M), date at the front; V2 deltas at the back | title parallel (can be turned off), compaction | V2 detects :8000/:1234 automatically; header and chunk timeout 5 min each |
| **Kilo Code** 7.8.17 (OpenCode fork) | like OpenCode V1 | ~8–10K (S) | date at the front, daily (#12666) | title at turn end (4 attempts), prompt enhancement, memory | no compaction without `limit` |
| **Cline** 4.1.25 | Chat Completions (AI SDK), Anthropic | ~5K (S) | stable; Plan↔Act breaks from 0; date at the front | compaction (81%), commit message on button press | undici 300 s abort only fixed in 4.1.25 (PR #15024) |
| **Roo Code** 3.54 (discontinued 2026-05-15) / **Zoo Code** 3.90 | Chat Completions, Anthropic | ~9–11K (S) | `environment_details` at the back, stable; reasoning is removed | condensing with its own system prompt | cache fields only `cache_read_input_tokens`; 300 s abort (Zoo #1407) |
| **Continue** 2.0/2.1 (no longer maintained) | Chat Completions | ~3–5K (S) | rules in the system prompt each round; truncation at the front | title (16 tokens), **autocomplete on every keystroke** | Qwen tools only with `capabilities: [tool_use]` |
| **Aider** 0.86.2 (barely maintained) | Chat Completions (LiteLLM), **no tool calls** | 0.4–1.6K + repo map 2K (S) | **breaks every round** behind the history | commit message, summary (parallel, from 1K history) | `--map-refresh files`, `--max-chat-history-tokens` |
| **Qwen Code** 0.26.0 | Chat Completions, Responses, Anthropic | 15.4K (M) / 14.0K (#12028) | **breaks every round** (memory reminder) (M) | memory extraction per turn (M), follow-up suggestions | name `qwen3.x…` → 1M context assumed; side requests with `stream:false` |
| **Goose** 1.55.0 | Chat Completions, Anthropic, Ollama | 4.1K → 1.35K (M, agent) | stable, no date (M, agent) | session name on the first 3 prompts (can be turned off) | reads `meta.n_ctx` from `/v1/models`; MOIM off below 32K |
| **Zed Agent** 1.25.2 | Chat Completions, Responses, Anthropic, llama.cpp native | ~10–15K (S) | system prompt rebuilt per request, date; old thinking as text | title asynchronous | `/props` (llama.cpp) for `supports_preserve_reasoning`; compaction only from an 80K window |
| **OpenHands** 1.32 / SDK 1.58.1 | Chat Completions (LiteLLM) | ~10–12K (S) | reasoning is not sent back for Qwen → break at every step | title (server), condenser | 300 s hard, 5 retries; itself recommends ≥22–32K |
| **Crush** 0.98.1 | Chat Completions, Anthropic | 11.6K (M) | stable (M) | title parallel (**cannot be turned off**), auto-summarize from 80% | 26 tools incl. LSP; 5 min to the first byte |
| **pi** 1.4.0 | Chat, Responses, Anthropic | small (not measured) | changes are appended | compaction (`reserveTokens` 16384!) | 117K stars |

## Own captures (2026-10-23)

**Own measurement, not agent research.**

Setup:
- Each harness ran against a fake server on 127.0.0.1. It records every request and answers with a fixed
  script: first two tool calls, then text.
- Task in an empty git repo: "Read notiz.txt twice and summarize it".
- Tokens are counted by the model's Qwen tokenizer (`ie-tokenize`) on the raw text, without the template
  frame. For tools, the Qwen template adds another ~250 tokens of instructions.
- "Prefix stable" means: every request begins character-exactly with the previous one, across tools, system
  prompt and all messages; `cache_control` does not count.

The scripts live only in the scratchpad and are not in the repo.

| Harness, version, mode | System | Tools (count) | Preamble 1st message | **Total before the task** | Prefill at 40 tok/s | Prefix over tool rounds |
|---|---:|---:|---:|---:|---:|---|
| Claude Code 2.1.309 interactive, empty config | 1,844 | 15,375 (15) | 2,872 | **~20,100** | ~8 min | stable; but title and suggestion requests in between |
| Claude Code 2.1.309 `-p`, empty config | 1,523 | 7,278 (12) | 2,415 | **~11,200** | ~4.7 min | stable |
| Claude Code `-p`, personal config (hooks, rules, CLAUDE.md) | 1,476 | 7,278 (12) | 8,172 | **~16,900** | > context 16K | stable |
| Claude Code interactive, `--tools Bash,Read,Edit,Write,Glob,Grep` + env switches (see below) | 1,402 | 2,849 (7) | 537 | **~4,800** | ~2 min | stable, no side requests (with all switches incl. title) |
| Claude Code `--bare` | 123 | 908 (3) | 11 | **~1,040** | ~26 s | stable (date in the system prompt) |
| OpenCode 1.18.35 (`run`, openai-compatible) | 2,041 + 4,183 skills¹ | 5,088 (10) | – | **~7,100** (+4,200¹) | ~3 min | stable; title request first |
| Codex CLI 0.164.0 (`exec`, Responses API) | 6,702² | 4,085 (9) | (in ²) | **~10,800** | ~4.5 min | stable |
| Qwen Code 0.26.0 (`--yolo`, OpenAI) | 7,624 | 4,804 (10) | 2,930 | **~15,400** | ~6.4 min | **breaks every round** (wandering reminder) |
| Crush 0.98.1 (`run`, openai-compat) | 5,167 | 6,406 (26) | 91 | **~11,600** | ~4.8 min | stable; title request first |

¹ Skills from `~/.agents/skills` and `~/.claude/skills` on this machine. OpenCode reads them even though
`XDG_CONFIG_HOME` was redirected.

² Codex sends no `instructions` text. Everything goes as `developer` and `user` items in `input`:
system prompt 17 KB, skills 8 KB, `<environment_context>` 1.6 KB.

On the deviations from the agent research:
- With Claude Code 2.1.207, Systima measured ~32.8K tokens in the first request, 24K of them for 27 tools
  ([report](https://systima.ai/blog/claude-code-vs-opencode-token-overhead)). In 2.1.309 there are 12–15 tools
  with an empty config. The rest sit behind `ToolSearch` as deferred tools.
- For `--bare` the agent had estimated 5–9K. Measured it is 1.04K.

### Claude Code 2.1.309 in detail

**Requests of an interactive session** (empty config, two user prompts):
- For each prompt, a **title request** comes first:
  - model `claude-haiku-5-5`, no tools
  - own system prompt of ~3 KB
  - `output_config.format` as JSON schema `{title}`
  - Measured, it came for **both** prompts, not only the first.
- Then the main requests come.
- After every turn a **suggestion request** comes. It goes to the main model, with the same prefix and
  "[SUGGESTION MODE: …]" as an additional user message.
- A single checkpoint is thus **lost twice at every user prompt**:
  - The title request overwrites it.
  - The next real request deviates from the suggestion prompt at the last user message.

**Turning off (tested):**
- `CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION=false`: the suggestion request is dropped.
- `CLAUDE_CODE_DISABLE_TERMINAL_TITLE=1` or `-n <name>`: the title request is dropped.
- `ANTHROPIC_DEFAULT_HAIKU_MODEL=<name>`: otherwise redirects the title request to another model.
- `CLAUDE_CODE_ATTRIBUTION_HEADER=0`: removes the line `x-anthropic-billing-header:
  cc_version=2.1.309.<hash>; cc_entrypoint=…` at the start of the system prompt. The hash changed between the
  title requests (`.b8b`, `.cc3`); the main requests stayed at `.8a3`.
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`: removes `HEAD /api/hello`. Without the fetched feature flags,
  no suggestion request came either.

**Telling them apart by header (tested):** With `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`, every request carries
`x-claude-code-request-class`. The title request has `auxiliary`, the main requests have `main`. In addition
come `x-claude-code-prompt-id` (the same within one user prompt) and
`x-claude-code-prev-tool-durations`.

**Prefix:**
- The main requests only append, across tool rounds and user prompts.
- The system prompt is rendered once per conversation and reused verbatim. `--system-prompt-snapshot`
  controls this: default `on`, under `--bare` `off` (per `claude --help`).
- Nothing dynamic is in the system prompt. Environment, date, skills list and deferred tools come in a
  **`role: "system"` message after the first user message** (2,119–2,580 tokens).
- Git status and attribution hint stand as `<system-reminder>` in the first user message.
- With an output style, Claude Code appends another `system` message after every tool result ("… output
  style is active …"). It stays in the history.

**Change of form without a change of text:** The same system message comes once as a list
`[{type: text, text, cache_control}]` and in the next request as a string. The renderer has to output both
the same way.

**The same `message.id` is merged:** When the fake server gave all responses the same `id`, Claude
Code merged the assistant messages into one and shifted older blocks around; the prefix broke. ie-server
generates random IDs (`rand_id("msg_")`), which is correct.

**Fields:**
- `stream: true`, always.
- `max_tokens`: 128,000 for a known model name, **32,000 for an unknown one**.
- `thinking: {type: "adaptive", display: "omitted"|"updates"}`.
- `output_config.effort`: `high`, with an empty config `medium`.
- `context_management.edits = [{type: "clear_thinking_20251015", keep: "all"}]`. So thinking stays in the
  history.
- `metadata.user_id` as a JSON string.
- In the first request of a session `safeguards: [{type: "dangerous_tool_use", classifier_context: …}]`.
- Query string `?beta=true`.
- Headers:
  - `anthropic-beta` with up to 11 values, among them `context-1m-2025-08-07`,
    `interleaved-thinking-2025-05-14`, `context-management-2025-06-27`, `effort-2025-11-24`,
    `advanced-tool-use-2025-11-20` and `inline-tools-2026-09-15`
  - `X-Stainless-Timeout: 600`
  - `X-Claude-Code-Session-Id`

**Also observed:**
- **No** call to `/v1/models` or `/v1/messages/count_tokens` in these sessions.
- **Auto mode:** Two writing Bash commands ran without an additional classifier request to our
  server. Whether the check is dropped entirely for third-party endpoints is **not substantiated**.
- **Context window of unknown models:** The binary contains the hint "Claude Code doesn't recognize the
  model … so it assumes a context window of … tokens". It points to `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and
  to [model-config#correct-the-window-for-a-gateway-or-custom-model-id](https://code.claude.com/docs/en/model-config).

### Codex CLI 0.164.0 in detail

**Responses API only:**
- `wire_api = "chat"` is rejected: "`wire_api = "chat"` is no longer supported … set `wire_api =
  "responses"`", with a pointer to [discussion #7782](https://github.com/openai/codex/discussions/7782).
- **So Codex needs `/v1/responses`.**

**Request:**
- Fields:
  - `stream: true`, `store: false`, `parallel_tool_calls: true`, `tool_choice: "auto"`
  - `reasoning: {summary: "auto"}`, `include: ["reasoning.encrypted_content"]`
  - `prompt_cache_key` = session ID
  - `client_metadata`: changes every turn, but is not prompt content
- Headers: `session-id`, `thread-id`, `x-codex-beta-features: remote_compaction_v2`.

**Unknown model name:**
- Codex reports "Model metadata for `qwen-local` not found. Defaulting to fallback metadata".
- With this metadata 9 tools came: `exec_command`, `write_stdin`, `request_user_input`, `view_image`,
  the namespace `multi_agent_v1`, `get_goal`, `create_goal`, `update_goal` and `web_search`.
  - `multi_agent_v1` alone takes 2,293 tokens.
  - `web_search` is of type `web_search`, not a function tool.
- **No `apply_patch`** in this mode.

**History as items, pure appending:**
- `message` (developer/user/assistant, content as `input_text`/`output_text`)
- `function_call`, with our `id` and `call_id`
- `function_call_output`, with client `id` `fco_…`

**Our minimal SSE stream was enough**, in this order:
1. `response.created`
2. `response.output_item.added`
3. `response.content_part.added`
4. `response.output_text.delta`
5. `response.output_text.done`
6. `response.content_part.done`
7. `response.function_call_arguments.delta` and `.done`
8. `response.output_item.done`
9. `response.completed`, with `usage.input_tokens_details.cached_tokens`

### OpenCode, Qwen Code, Crush in detail

**OpenCode 1.18.35** (`@ai-sdk/openai-compatible`):
- The first request is the **title request**: `small_model`, system prompt "You are a title generator…" (2 KB),
  no tools.
- Then the main request follows:
  - `max_tokens` = configured `limit.output`
  - `stream_options.include_usage: true`
  - `tool_choice: "auto"`
- The system prompt contains an `<env>` block with `Today's date`. It is stable within the session, but
  changes daily.

**Qwen Code 0.26.0** (`OPENAI_BASE_URL`/`OPENAI_MODEL`):
- `max_tokens` 32,768.
- The reminder "Current auto-memory catalog" always attaches to the **latest** user or tool message and
  disappears from its previous position. The prefix therefore breaks **in every round**
  shortly before the end.
- After the turn comes a **side request "managed memory extraction subagent"**: own system prompt
  (13.7 KB), 5 tools, the whole history. It overwrites a single checkpoint.

**Crush 0.98.1** (`type: openai-compat`):
- First comes the **title request**: `max_tokens: 40`. Crush itself appends
  `<think>\n\n</think>` to the user text to turn off Qwen's thinking.
- Then the main requests follow with `max_tokens` = `default_max_tokens`.
- 26 tools, 6 of them LSP tools.
- A reminder about the empty todo list stands as its own user message before the task.
- The system prompt contains the date and a git status snapshot; it is stable within the session.

## Per harness

This is the agent research. The source references apply to the paragraph they stand in.

### Claude Code (2.1.309, 2026-10-22)

**1. API and configuration**
- Speaks only Anthropic Messages, no OpenAI format.
- Endpoints per [llm-gateway-protocol](https://code.claude.com/docs/en/llm-gateway-protocol):
  - `POST /v1/messages?beta=true`.
  - `POST /v1/messages/count_tokens` is optional; if it is missing, Claude Code estimates by characters.
  - `HEAD /api/hello` serves as a connection probe and may be rejected.
  - `GET /v1/models` comes only with `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`. Only IDs containing
    `claude` or `anthropic` are accepted.
- Configuration via environment variables ([env-vars](https://code.claude.com/docs/en/env-vars)), also in the
  `env` block of `~/.claude/settings.json`:
  - `ANTHROPIC_BASE_URL`
  - `ANTHROPIC_AUTH_TOKEN` (sent as Bearer) or `ANTHROPIC_API_KEY` (sent as `x-api-key`)
  - `ANTHROPIC_MODEL`
- `--bare` accepts only `ANTHROPIC_API_KEY` or `apiKeyHelper`
  ([headless](https://code.claude.com/docs/en/headless#start-faster-with-bare-mode)).

**2. Size and lean mode**
- Measured, see above: 20K interactive, 11K with `-p`, 4.8K with 6 tools, 1K with `--bare`. Personal hooks,
  rules and CLAUDE.md alone added ~7K tokens on a developer machine.
- Switches:
  - `--tools`: allowlist of the built-in tools.
  - `--disallowedTools`: a bare tool name removes the tool from the context.
  - `--system-prompt[-file]`: replaces the system prompt.
  - `--strict-mcp-config`: loads no MCP servers.
  - `CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS=1`: drops the commit and PR instructions (~2.3K) and the
    git status.
  - `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1` and `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`.
  - Source: [env-vars](https://code.claude.com/docs/en/env-vars), Piebald-AI/claude-code-system-prompts
    v2.1.308.
- For unknown model IDs, task and todo tools are dropped since 2.1.268
  ([tools-reference](https://code.claude.com/docs/en/tools-reference#task-tool-availability)).
- The switches `CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS`, `CLAUDE_CODE_ENABLE_AWAY_SUMMARY` and
  `CLAUDE_STREAM_IDLE_TIMEOUT_MS` are in the 2.1.309 binary (checked ourselves).

**3. Prefix**
- Measured, see above.
- According to [prompt-caching](https://code.claude.com/docs/en/prompt-caching) and the
  [blog of 2026-04-30](https://claude.dev/blog/lessons-from-building-claude-code-prompt-caching-is-everything/):
  - The tool order is fixed.
  - Stale things such as the date or changed files arrive as `<system-reminder>` on the next message.
- The cache is invalidated by:
  - model change and effort change
  - an MCP server that connects without tool search being active
  - a tool that removes a deny rule
  - compaction
  - an update of Claude Code
- Attribution block (`x-anthropic-billing-header`):
  - Before 2.1.181 it changed per request ([#68900](https://github.com/anthropics/claude-code/issues/68900)).
  - Since 2.1.181 it is stable with a custom base URL.
  - Even so, Unsloth, LM Studio and field reports recommend `CLAUDE_CODE_ATTRIBUTION_HEADER=0`
    ([Unsloth](https://unsloth.ai/docs/basics/claude-code),
    [LM Studio](https://lmstudio.ai/docs/developer/anthropic-compat),
    [llama.cpp report](https://www.mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix/)).
- `role:"system"` in the middle of the history: if the server rejects it with 400, Claude Code retries once
  without these messages and puts the text into user messages. It stays that way until `/clear` or compaction
  (gateway docs). Since 1f51077, ie-server converts them to user text itself.
- Thinking:
  - Old `thinking` blocks go back together with their `signature`.
  - If the server rejects the signature with 400, Claude Code permanently removes all old thinking blocks.
  - After an hour of idle time, `clear_thinking` with `keep:1` can come
    ([postmortem 04-23](https://www.anthropic.com/engineering/april-23-postmortem)).
  - `DISABLE_INTERLEAVED_THINKING=1` turns interleaving off.

**4. Side requests** (agent; [env-vars](https://code.claude.com/docs/en/env-vars),
[costs](https://code.claude.com/docs/en/costs#background-token-usage); what is measured is stated above)
- Title:
  - Runs on the main model with a custom base URL.
  - Exception: `ANTHROPIC_API_KEY` is set and `ANTHROPIC_AUTH_TOKEN` is missing. Then the request goes to
    the default Haiku; that is also how we measured it.
- Prompt suggestions: after every answer.
- Session recap: after ≥3 min idle and ≥3 turns. Turn off with `CLAUDE_CODE_ENABLE_AWAY_SUMMARY=0`.
- WebFetch: own model call per fetch.
- Subagents: own prefix.
- Auto-mode classifier.
- The Bash prefix check has been removed since 2.1.228 (Piebald changelog).
- Compaction ([model-config](https://code.claude.com/docs/en/model-config#context-window-and-auto-compaction)):
  - Flow: same prefix plus a summary instruction, then rebuild.
  - Unknown ID: Claude Code assumes 200K.
  - `CLAUDE_CODE_MAX_CONTEXT_TOKENS=16384` works for IDs that do not start with `claude-`. That fits
    `qwen3.8-flash-next-coder`.
  - `CLAUDE_CODE_AUTO_COMPACT_WINDOW` applies only from 100,000.
  - `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (default 32,000) shrinks the usable window and has to go to ~4096 at
    16K.
- `DISABLE_NON_ESSENTIAL_MODEL_CALLS` is no longer in the docs; presumably outdated, **not substantiated**.

**5. Fields**
- Streaming is mandatory, with the complete sequence up to `message_stop`.
- Several `tool_use` in one response are possible; the arguments arrive as `input_json_delta`.
- Context usage = `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`
  ([statusline](https://code.claude.com/docs/en/statusline)). So the sum has to yield the real prompt.
- Overflow: 400 with "Prompt is too long" triggers reactive compaction
  ([errors](https://code.claude.com/docs/en/errors#prompt-is-too-long)).
- Timeouts with a custom base URL
  ([network-config](https://code.claude.com/docs/en/network-config#streaming-idle-watchdogs)):
  - `API_TIMEOUT_MS` 600 s until the headers.
  - Byte watchdog: 300 s without a byte.
  - Event watchdog: 300 s without a parsed event.
  - Adjustable via `CLAUDE_STREAM_IDLE_TIMEOUT_MS` (300,000–1,800,000) and `API_FORCE_IDLE_TIMEOUT=0`.
- `MAX_THINKING_TOKENS=0` sends `thinking: disabled`; `CLAUDE_CODE_DISABLE_THINKING=1` omits the parameter.
- `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` removes beta fields such as `context_management` and
  `output_config.format`.

**6. Problems and other servers**
- [#32636](https://github.com/anthropics/claude-code/issues/32636): since 2.1.36, llama-server recomputes
  everything on every request, for hybrid or recurrent models. Title request and billing header were
  suspected as the cause. Exactly our case.
- Workaround for llama.cpp: `--ctx-checkpoints 128 --cache-ram … -np 1`.
- Ollama ([Anthropic compatibility](https://docs.ollama.com/api/anthropic-compatibility)):
  - no `count_tokens`, no `cache_control`
  - recommends ≥64K context ([Claude Code integration](https://docs.ollama.com/integrations/claude-code))

### OpenAI Codex CLI (0.164.0, 2026-10-23)

**1. API and configuration**
- Responses only now:
  - [PR #10157](https://github.com/openai/codex/pull/10157) "nuke chat/completions API", merged
    2026-02-03.
  - Announced in [#7782](https://github.com/openai/codex/discussions/7782).
- Built-in providers `ollama` and `lmstudio`, plus `--oss`. Ollama must be at least 0.13.4.
- `model_providers` belongs in `~/.codex/config.toml`; a project-local file ignores the field
  ([config-reference](https://learn.chatgpt.com/docs/config-file/config-reference)).
- Proposal of the agent. The keys are substantiated, the combination is untested:
  ```toml
  model = "qwen3.8-flash-next-coder"
  model_provider = "ie"
  model_context_window = 16384           # otherwise 272,000
  model_auto_compact_token_limit = 13000
  model_supports_reasoning_summaries = false
  model_instructions_file = "~/.codex/ie-base.md"
  web_search = "disabled"
  approval_policy = "never"
  [features]
  multi_agent = false
  [tools]
  view_image = false
  [model_providers.ie]
  name = "ie-server"
  base_url = "http://127.0.0.1:8080/v1"
  wire_api = "responses"
  stream_idle_timeout_ms = 900000        # default 300,000
  request_max_retries = 1                # default 4
  stream_max_retries = 1                 # default 5
  ```

**2. Size**
- Measured: 10.8K.
- With fallback metadata:
  - Base prompt `models-manager/prompt.md`, 20.9 KB.
  - **No apply_patch tool**; the prompt still demands apply_patch via the shell
    ([#33405](https://github.com/openai/codex/issues/33405)).
  - Source: `codex-rs/models-manager/src/model_info.rs`.
- A custom `model_catalog_json` with ModelInfo can set the base prompt, `apply_patch_tool_type: "freeform"`
  and `context_window`.

**3. Prefix**
- Since [PR #51156](https://github.com/openai/codex/pull/51156) (2026-10-05), the base instructions come as
  the first `developer` item, with a deterministic ID.
- Later changes are **appended in the middle of the history** as diff fragments: date, cwd
  (`core/src/context/world_state/`).
- Reasoning items stay in the history and go back, with `summary`, `content` and `encrypted_content`.
- `previous_response_id` exists only over WebSocket; for custom providers it is off.

**4. Side requests**
- Title:
  - Since [PR #42749](https://github.com/openai/codex/pull/42749) (2026-09-04), the TUI generates the
    thread title in parallel with the first turn, with the current model.
  - No tools, prompt ≤960 bytes, `text.format` as json_schema `{title}`.
  - **No off switch substantiated.** `codex exec` is not affected.
- Compaction:
  - Locally it is a normal `/responses` request: history plus compact prompt.
  - Afterwards the history consists of base context, up to 20,000 tokens of the latest user messages and
    the summary (`core/src/compact.rs`).
  - Threshold: min(`model_auto_compact_token_limit`, 90% of the window).
  - `/responses/compact` has been removed since PR #44273.
- Model catalog:
  - Codex queries `GET /models?client_version=…` and expects `{"models":[ModelInfo…]}`.
  - LM Studio's `{"data":[…]}` fails on it ([#35758](https://github.com/openai/codex/issues/35758),
    open).
  - `ollama launch codex` works around this with `model_catalog_json`
    ([docs](https://docs.ollama.com/integrations/codex)).

**5. Fields**
- Codex takes output items **only from `response.output_item.done`**. `function_call_arguments.*` is
  ignored.
- `usage` in `response.completed` needs `input_tokens`, `output_tokens` and `total_tokens`. If the
  details objects are present, they need `cached_tokens` or `reasoning_tokens`. **Codex treats incomplete
  `usage` as a stream error.** Source: `codex-api/src/sse/responses.rs`, `protocol/src/models.rs`.
- If `response.completed` is missing, Codex retries the request up to 5 times.
- The idle timeout measures between parsed events. Whether an SSE comment counts is **not substantiated**.
  `response.in_progress` surely resets it.
- Tool types:
  - `function`
  - `namespace` (MCP)
  - `web_search`
  - `custom` (freeform `apply_patch` with Lark grammar, only with `apply_patch_tool_type`)
  - `tool_search`

**6. Other servers**
- llama.cpp `/v1/responses` since [PR #18486](https://github.com/ggml-org/llama.cpp/pull/18486) (2026-01-21,
  "partial", translated to Chat Completions, namespace tools are discarded).
  - Follow-ups open: #19720, #23235, #27751, #29412.
  - #21174 "Codex compatibility" was closed without a merge.
- vLLM docs with `--tool-call-parser qwen3_coder`
  ([docs](https://docs.vllm.ai/en/stable/serving/integrations/codex/)).
- Ollama `/v1/responses` since 0.13.3, stateless.
- LM Studio since 0.3.29.

### OpenCode (V1 1.19.4 / V2 2.1.4, 2026-10-23)

Two lines. V2 (`@opencode/cli`, since 2026-09-11) is the main line, V1 (`opencode-ai`) uses the Vercel
AI SDK. Locally we measured V1 1.18.35.

**1. API**
- V1: `opencode.json`, `provider.<id>.npm`:
  - `@ai-sdk/openai-compatible` = Chat Completions
  - `@ai-sdk/openai` = Responses
  - `@ai-sdk/anthropic` with `options.baseURL`
  - plus `models.<id>.limit.context/output`
  - [docs](https://opencode.ai/docs/providers)
- V2: `providers.<id>.package` (`@opencode/ai/providers/openai-compatible`, `…/responses`,
  `…/anthropic-compatible`), plus `compatibility.reasoningField` etc.
  ([V2 docs](https://opencode.ai/v2/docs/providers)).
- V2 looks for local servers on :11434, :1234 and :8000 automatically. **So do not put ie-server on 8000 or
  1234.**

**2. Size**
- V1: measured 7.1K plus skills.
  - For Qwen and unknown IDs, V1 takes `default.txt` (8.5 KB).
  - V1 selects prompt and tools by substrings of the model ID. Avoid names with `gpt` or `claude`.
- V2: base prompt 756 characters, short tool texts, **~2.5–4K (estimate)**.

**3. Prefix**
- V1: one system block with date, AGENTS.md, skills and MCP. It breaks at the front on a change of date,
  skills or MCP.
- V1: tool parts are serialized differently byte by byte, the KV cache misses by 343–1009 tokens (#37886,
  #24840).
- V2: instructions "most to least shared", changes come as deltas in the middle of the history
  ([session.md](https://github.com/anomalyco/opencode/blob/v2.1.4/specs/v2/session.md)). On the chat route
  they become user text.
- Both lines send back the `reasoning_content` of all earlier responses.

**4. Side requests**
- Title in parallel with the first request:
  - V2 can be turned off with `agents.title.disabled: true`, V1 with `agent.title.disable: true`.
  - #54401: a title request to a local thinking model ran 501 s.
- Compaction:
  - V2: appends to the history, threshold at 16K ~13.9K, lower `keep` (default 15,000).
  - V1: own agent, complete restart. Without `limit.output`, V1 compacts after every step at 16K
    (#49965).
- Warming (V2, only with `warming:true`): one request every 4 min (#56636).

**5. Fields**
- Streaming with `include_usage`; `prompt_tokens_details.cached_tokens` is read.
- V2 sends `max_completion_tokens`, V1 `max_tokens`.
- Timeouts: 5 min until the headers and 5 min between chunks. In V2, an SSE comment resets the
  chunk timeout (`packages/ai/src/route/transport/http.ts`). Can be turned off with
  `headerTimeout`/`chunkTimeout`.

**6. Problems**
- System messages in the middle of the history break Qwen templates (#42909, #45055).
- Ollama recommends ≥64K ([docs](https://docs.ollama.com/integrations/opencode)).

### Kilo Code (7.8.17, 2026-10-23)

An OpenCode fork since v7 (`packages/opencode`).
- **Configuration:** `kilo.jsonc` like OpenCode V1. Without `limit`, context = 0 applies: no compaction,
  output 32,000 ([custom-models](https://github.com/Kilo-Org/kilocode/blob/main/packages/kilo-docs/pages/code-with-ai/agents/custom-models.md)).
- **Size:** `default.txt` + `soul.txt` + env + skills, **~8–10K (estimate)**.
- **Date at the front:** `toDateString()` in the system prompt, so it breaks daily from position 0
  ([#12666](https://github.com/Kilo-Org/kilocode/issues/12666), open).
- **`environment_details`:** generated once per message and cached, so it is stable.
- **Plan agent:** inserts its instructions transiently each round; presumably a break, **not measured**.
- **Side requests:**
  - Title at turn end (up to 4 attempts, `agent.title.model` → `small_model` → session model)
  - Prompt enhancement
  - Memory auto-save (memory is off by default)
  - Autocomplete always goes to the Kilo gateway, not to us
- **Timeouts:** 300 s until header and first byte, adjustable via `options.timeout`, `chunkTimeout` and
  `headerTimeout` (`packages/opencode/src/kilocode/provider/provider.ts`).

### Cline (4.1.25, 2026-10-16; CLI 3.0.73)

Since v4.0.0 (2026-06-26), Cline runs on the "Cline SDK" (Vercel AI SDK).
- **API:** "OpenAI Compatible" (Chat Completions, `includeUsage`), LM Studio, Ollama (`num_ctx` 32768 as
  default) and Anthropic with a custom base URL. Since 4.1.25 the base URL is taken exactly as
  entered ([release](https://github.com/cline/cline/releases/tag/v4.1.25)).
- **Size:** `act.ts` 3.7 KB plus tools, **~5K (estimate)**.
  - Native tool calls only.
  - "Use Compact Prompt" has been removed (`state.proto`).
- **Prefix:**
  - No `environment_details` any more.
  - `<mode_notice>` stands before the next user message.
  - **Plan↔Act swaps system prompt and tools**, so it breaks from position 0.
  - Date (`toLocaleDateString`) in the system prompt.
- **Compaction:** "agentic" at 81% of the window, target 70%, summary up to 8192 tokens
  (`compaction-shared.ts`). A title request via LLM is **not substantiated**.
- **Timeout:** until 4.1.24, undici cut off after 300 s without a byte. PR
  [#15024](https://github.com/cline/cline/pull/15024) (2026-10-13) turns this off for local providers
  (issue [#13371](https://github.com/cline/cline/issues/13371)).

### Roo Code (3.54.0, discontinued) / Zoo Code (3.90.0)

Roo Code was shut down on 2026-05-15, the repo is read-only. The README points to the fork Zoo Code
and to Cline ([README](https://github.com/RooCodeInc/Roo-Code)).
- **API:** OpenAI Compatible (`parallel_tool_calls: true`, `include_usage`), LM Studio, Ollama and
  Anthropic. Default for custom models: **contextWindow 128,000**; it has to be set to 16K.
- **Size:** 21 native tools, XML mode removed (PR #10841). Code mode **~9–11K (estimate)**.
  `.roo/system-prompt-<mode>` removed (PR #11387).
- **Prefix:**
  - `environment_details` is appended to every new user message; old blocks stay.
  - According to Zoo [#1891](https://github.com/Zoo-Code-Org/Zoo-Code/issues/1891) (open), the prefix is broken by:
    - **Reasoning is removed** when `preserveReasoning` is off (default for OpenAI Compatible).
    - Tool arguments are re-serialized.
    - Condensing uses its own system prompt.
    - A mode switch changes the tool list.
- **Condensing:** threshold `contextWindow·0.9 − maxTokens`. Fallback is a sliding window that hides 50%.
- **Fields:**
  - `<think>` and `reasoning_content` are recognized.
  - Roo reads cache hits only from `cache_read_input_tokens`.
  - `apiRequestTimeout` 600 s, yet abort after 300 s (Zoo
    [#1407](https://github.com/Zoo-Code-Org/Zoo-Code/issues/1407)).
  - Roo detects overflow by 400 with "context length".

### Continue (2.0.0 / 2.1.0-vscode, no longer maintained)

Per [README](https://github.com/continuedev/continue) "no longer actively maintained".
- **API and size:**
  - `config.yaml`, `provider: openai`, `apiBase`.
  - The agent system prompt has ~150 tokens, with tools **~3–5K (estimate)**.
  - Native tools exist only for a recognized model name; for Qwen set `capabilities: [tool_use]`.
- **Prefix:** glob rules change the system prompt each round. In VS Code it is truncated at the front when
  the limit is hit (`countTokens.ts`).
- **Side requests:**
  - Title with `maxTokens: 16`
  - **Autocomplete on every keystroke** if a model has the role `autocomplete`. This destroys the
    checkpoint continuously.

### Aider (0.86.2, 2026-02-12)

49.7K stars, barely maintained any more.
- **API:** LiteLLM only, Chat Completions. Call with `OPENAI_API_BASE` and `--model openai/<name>`
  ([docs](https://aider.chat/docs/llms/openai-compat.html)). **No tool calls**, edits are parsed from the
  text.
- **Size:** depending on the edit format 0.4–1.6K (estimate).
  - For unknown models the format is `whole`.
  - Repo map = `max_input/8`, limited to 1024–4096; at 16K that is 2048.
- **Prefix:**
  - Order: system, examples, readonly, repo map, history, **chat files**, current message,
    reminder (`chat_chunks.py`).
  - The file block stands behind the history and changes after every edit.
  - The repo map can change per message (`--map-refresh auto`).
  - From `max_chat_history_tokens` (at 16K: 1024), Aider summarizes the history and rewrites it from
    its start.
  - **A checkpoint at the prompt end practically never hits.**
- **Side requests:** commit message after every edit, summary **in parallel in its own thread**.
- **Fields:** `reasoning_tag: think`; reasoning is not sent back. Timeout 600 s.
- **Tips:** `--map-refresh files`, `--max-chat-history-tokens 6000`, `--no-auto-commits`, `--timeout 1800`.

### Qwen Code (0.26.0, 2026-10-21)

- **API:** `openai` (Chat), `openai-responses` and `anthropic`. For local servers `openai` is the right
  way. Configuration via `~/.qwen/settings.json` → `modelProviders.openai[]` with `generationConfig`
  (`timeout`, `streamIdleTimeoutMs`, `contextWindowSize`, `extra_body`)
  ([docs](https://qwenlm.github.io/qwen-code-docs/en/users/configuration/model-providers/)).
- **Context window by name**, if `contextWindowSize` is missing:
  - `^qwen3\.\d` → **1M** ([#13415](https://github.com/QwenLM/qwen-code/issues/13415))
  - `^qwen` → 256K
  - otherwise 200K

  **Our name `qwen3.8-flash-next-coder` falls under 1M.** So `contextWindowSize: 16384` is mandatory.
- **Size:** measured 15.4K; [#12028](https://github.com/QwenLM/qwen-code/issues/12028) reports 13,956.
  The project itself calls 16K "right at the edge" (#4657). It can be made leaner with `tools.eager` as an
  allowlist, `QWEN_SYSTEM_MD` or `--system-prompt`, and `model.skipStartupContext`.
- **Prefix:** The memory catalog wanders to the end. With llama.cpp this led to a full re-prefill in 86 of 92
  requests ([#14203](https://github.com/QwenLM/qwen-code/issues/14203), open). Remedy: memory off.
- **Side requests** (without streaming!):
  - Follow-up suggestions per turn (`ui.enableFollowupSuggestions`)
  - Memory extraction per turn (`memory.enableManagedAutoMemory`)
  - Dream every 24 h
  - Compaction at 0.85 × window (`context.autoCompactThreshold`)
- **Timeouts:** 120 s until the first response, 240 s between chunks, 3 retries.
  - **SSE comments do not count**, only real chunks.
  - Consequence: a 14K prefill was sent three times ([#14215](https://github.com/QwenLM/qwen-code/issues/14215)).
- **Quirks:**
  - Some tools come **without `parameters`**; Ollama and vLLM answer that with 400 (#12878, #14112).
  - Side requests use `tool_choice:"required"` with `respond_in_schema` as well as
    `chat_template_kwargs.enable_thinking:false`.

### Goose (1.55.0, 2026-10-16; repo now aaif-goose/goose)

The agent captured Goose itself against a mock.
- **API:** provider `openai` (Chat; Responses only for o3- and gpt-5-style names), `anthropic` and `ollama`.
  Custom provider under `~/.config/goose/custom_providers/<name>.json`, measured to work
  ([docs](https://goose-docs.ai/docs/getting-started/providers)):
  ```json
  {"name":"custom_ie","engine":"openai","base_url":"http://127.0.0.1:8080/v1/chat/completions",
   "models":[{"name":"qwen3.8-flash-next-coder","context_limit":16384}],"dynamic_models":false,
   "supports_streaming":true,"requires_auth":false,"timeout_seconds":1800}
  ```
- **Size:**
  - Default: 889 tokens system + 3,239 tokens for 17 tools.
  - Lean: 588 + 759 tokens with 5 tools (measured by the agent).
  - Plus: `GOOSE_DISABLE_SESSION_NAMING: true`, extensions off, `developer` with `available_tools`.
- **Prefix:**
  - System and tools byte-identical, no date.
  - MOIM is skipped below 32K context.
  - A full prefill occurs only when extensions are switched on or off or hints from subdirectories are
    loaded later.
  - `reasoning_content` of all rounds goes back.
- **Side requests:**
  - Session name in parallel and before the main request, on the first 3 prompts.
  - Compaction at 0.8: writes the history into the system prompt, so a new prefix. The cache-friendly
    variant is [PR #11257](https://github.com/aaif-goose/goose/pull/11257), open.
  - Lead/worker and toolshim are removed or unsuitable for us.
- **Context limit:** `GOOSE_CONTEXT_LIMIT` → `context_limit` → **`GET /v1/models` `data[].meta.n_ctx`** →
  models.dev → 128K.
- **Timeouts:** 600 s; with the Ollama provider 120 s per line.

### Zed Agent (1.25.2)

- **API:** `language_models.openai_compatible.<Name>` with `available_models[].capabilities`
  (`chat_completions:false` = Responses), `anthropic_compatible` and native providers for llama.cpp
  (`/props`, `/v1/models`), Ollama and LM Studio
  ([default.json](https://github.com/zed-industries/zed/blob/main/assets/settings/default.json)).
- **Size:** `system_prompt.hbs` 19.8 KB plus ~16 tools, **~10–15K (estimate)**. Profile `minimal` or
  `ask`, or custom profiles.
- **Prefix:**
  - The system prompt is rendered anew per request, with the date.
  - With `interleaved_reasoning:false` (default), old thinking goes back **as normal assistant text**
    (`crates/open_ai/src/completion.rs`).
  - The llama.cpp provider asks `/props` for `supports_preserve_reasoning`.
- **Side requests:** title asynchronous (whole history + "Generate a concise 3-7 word title"). Compaction
  only from an 80,000 window (PR #58883), so none at 16K.
- **Fields:** `temperature: 1.0` if nothing is set.

### OpenHands (1.32.0 / SDK 1.58.1 / CLI 1.16.0)

- **API:** LiteLLM, `openai/<model>` with `base_url`. The docs recommend ≥22K, better 32K context
  ([local-llms](https://docs.openhands.dev/openhands/usage/llms/local-llms)).
- **Size:**
  - System prompt ~5K + tools ~5–6K, **~10–12K (estimate)**.
  - Native tool calls are the default.
  - Every tool gets the parameters `security_risk` and `summary`.
- **Prefix:** `reasoning_content` goes back only for Kimi, DeepSeek and MiniMax, **not for Qwen**
  (`model_features.py`). So the prefix breaks at every step behind `<|im_start|>assistant`.
- **Side requests:** title (agent server, parallel), condenser from 80 events (the same LLM).
- **Timeouts:** 300 s hard, 5 retries. Already the first request runs into the timeout at 40 tok/s.

### Crush (0.98.1)

- **API:**
  - `openai-compat` (Chat) and `anthropic`; Responses only for Copilot and Zen (#3937).
  - Since 0.88.0 the configuration is the Bash file `.crushrc`, `crush.json` is considered outdated
    ([release](https://github.com/charmbracelet/crush/releases/tag/v0.88.0)). In our measurement it still ran
    with `crush.json`.
- **Size:** measured 11.6K. `coder.md.tpl` has 21.5 KB; plus 26–27 tools, 8 of them LSP tools
  (`auto_lsp`).
- **Prefix:**
  - Stable (measured).
  - `<mcp-instructions>` are appended per request and change when an MCP server is added.
  - `TrimSpace` on the assistant text sent back (`internal/message/content.go`).
- **Side requests:**
  - Title (cannot be turned off, also for sub-agents, #4083)
  - Auto-summarize from 80% (at 16K that is ~12.8K) with its own system prompt
  - can be turned off with `disable_auto_summarize`
- **Timeouts:** 60 s idle between parts; for the first part Crush waits max(timeout, 5 min) (#3658).

### Other harnesses common in 2026

| Harness | Stars (10-23) | Local | Note |
|---|---:|---|---|
| **pi** (earendil-works/pi, ex badlogic/pi-mono) 1.4.0 | 117.4K | `models.json`, `api`: `openai-completions`, `openai-responses` or `anthropic-messages` | no date in the system prompt, changes are appended; `reserveTokens=16384` makes 16K unusable (`packages/coding-agent/docs/settings.md`) |
| **Mistral Vibe** 3.0.2 | 5.2K | default provider `llamacpp` = `http://127.0.0.1:8080/v1` | `reasoning_field_name: reasoning_content`; compaction appends to the prefix (README) |
| **GitHub Copilot BYOK** (VS Code) | – | "Custom Endpoint", `apiType` Chat, Responses or Messages | `chat.utilityModel` for titles and commit messages ([docs](https://code.visualstudio.com/docs/copilot/customization/language-models), 2026-10-21) |
| **Gemini CLI** | 107.6K | Gemini API only, locally via a proxy | the forks (Qwen Code) are the actual way |
| **Factory Droid** | closed source | `customModels`, `generic-chat-completion-api` / `anthropic` | [docs](https://docs.factory.ai/cli/byok/) |

Cursor integrates local models only by tunnel through its own servers. Kimi CLI and Void are archived.

## How other servers solve this

State of the agent research, with PR numbers.

### llama.cpp (master f5027fb, 2026-10-23)

**Endpoints:**
- Anthropic `/v1/messages`: [#17570](https://github.com/ggml-org/llama.cpp/pull/17570) (2025-11-28),
  `count_tokens`, thinking blocks with [#18551](https://github.com/ggml-org/llama.cpp/pull/18551)
  (2026-01-06).
- `/v1/responses`: [#18486](https://github.com/ggml-org/llama.cpp/pull/18486), partial.

**Reasoning:** `--reasoning-preserve` has been the default since [#28174](https://github.com/ggml-org/llama.cpp/pull/28174)
(2026-09-02). Old thinking thus stays in the history if the template can handle it.

**Prompt cache:**
- `--cache-reuse` (KV shifting) is turned off for recurrent and hybrid models.
- Host-RAM prompt cache `--cache-ram` (default 8 GiB) since
  [#16391](https://github.com/ggml-org/llama.cpp/pull/16391):
  - Stored prompts are extra slots.
  - The entry with the longest common prefix is chosen, **together with its checkpoints**.

**Context checkpoints for hybrid and SWA:**
- Introduced with [#15293](https://github.com/ggml-org/llama.cpp/pull/15293).
- Today `--ctx-checkpoints` 32 per slot, `--checkpoint-min-step` 8192.
- They sit at these places:
  1. **before the last user message**, via the `message_spans` from the template
     ([#22929](https://github.com/ggml-org/llama.cpp/pull/22929), 2026-05-25)
  2. at earlier user starts, at the earliest every 8192 tokens
  3. **two shortly before the prompt end**, at n−4−n_ubatch and n−4
     ([#20288](https://github.com/ggml-org/llama.cpp/pull/20288))
- Open is [#30745](https://github.com/ggml-org/llama.cpp/pull/30745): reuse 91% → 94%,
  memory 3.6 → 0.8 GB for Qwen3.8-27B.

### vLLM, SGLang, Ollama, LM Studio

**vLLM** v0.32.0:
- Anthropic endpoint (#22627, #27882) and Responses (#20504).
- Prefix caching by block hash; any number of prefixes work.
- For hybrid models `--mamba-cache-mode align` (#30877), plus prefill checkpoints for GDN (#57329,
  #60050).

**SGLang:**
- RadixAttention.
- For hybrid models `MambaRadixCache` (#11214) after Marconi (arXiv 2411.19379): SSM states only at
  selected nodes, eviction by FLOPs saved.

**Ollama** v0.40.9:
- Anthropic API (#13600) without `count_tokens`.
- `/v1/responses` (#13351).
- `ollama launch claude|codex|opencode` (#13871).
- The default context depends on VRAM; below 24 GiB it is 4K. For agents the docs recommend
  ≥64K ([docs](https://docs.ollama.com/context-length)).

**LM Studio:** `/v1/responses` since 0.3.29, Anthropic since 0.4.1 ([blog](https://lmstudio.ai/blog/claudecode)).

## Conclusions for ie-server

Priority by effect per effort. The prefill times assume 40 tok/s.

### 1. Send bytes and events during the prefill

This is mandatory.

**Problem today:** `message_start` goes out only after `wait_started`, i.e. after the prefill (`ie_server.cpp`
~lines 363–375).

**New:**
- Send the header and `message_start` immediately. `input_tokens` and the reusable part are known to the
  server before the prefill: prompt length and common prefix with the checkpoint.
- Then keepalives, depending on the protocol:

  | Protocol | Keepalive |
  |---|---|
  | Anthropic | `event: ping` every ≤10 s |
  | OpenAI Chat | immediately `delta:{role:"assistant"}`, then empty delta chunks; Qwen Code ignores SSE comments |
  | Responses | `response.created` immediately, then `response.in_progress` |

**If the client aborts, finish the prefill anyway and save it.** Claude Code, Codex, Qwen Code,
OpenHands and Goose repeat the same request, which then only appends. To check: today `j.cancel` is checked
only before `prefill` (`run_once`).

**Effect:** Without this item, every prompt above ~12K trips the 300 s watchdogs, and the retry recomputes
everything.

### 2. Several checkpoints, selected by longest common prefix

**Model:** llama.cpp with `--cache-ram`, `--ctx-checkpoints` and the positions from #22929/#20288.

**One entry** consists of:
- token sequence
- recurrent state: 112.6 MiB in F32 per 06 §9
- logits

The KV cache is positional, truncating costs nothing. If a request deviates at the front (title), the old
branch additionally needs a KV copy in RAM. At 16K that is ~0.45 GB in FP16 or 0.9 GB in F32, copied over
PCIe in ~10–20 ms.

**Where the points belong, per request:**
- (a) End of the static part: system + tools + first message. Saves compaction, sub-agents, new
  sessions and Codex after PR #51156.
- (b) **Before the last user or tool message.** Saves Qwen Code's wandering reminder, Claude Code's
  suggestion request and Aider.
- (c) Shortly before the end, at n−4. Saves deviations in the last tokens.
- (d) End of the prompt, as today.

**Memory:** 8–16 states in RAM are 0.9–1.8 GB. That is cheap at 64 GB. Eviction by LRU,
weighted by prefill seconds saved (the Marconi idea).

**Effect:** Every side request then costs only its own prefill. The next main request finds its
branch again instead of recomputing for 5–8 min.

### 3. Detect side requests and treat them separately

**Markers:**
- `x-claude-code-request-class: auxiliary|compaction|subagent` (with `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`,
  tested)
- no `tools`
- `output_config.format` or `text.format` with `title`
- `max_tokens` ≤ 64 (Crush 40, Continue 16)
- `stream:false` with `tool_choice:"required"` (Qwen Code)
- known system starts: "You are a title generator", "You name conversations", "You will generate a
  short title", "Generate a concise 3-7 word title", "[SUGGESTION MODE"

**Treatment:**
- Thinking off, cap `max_tokens`.
- Never evict the main branch (see item 2).
- Optional `--side-requests=cheap`: answer titles without the model from the first words. Drawback: the
  title is worse, and the behavior deviates from "real" model output.

**Queue:** Titles come in OpenCode, Crush, Goose, Codex TUI and Zed **in parallel** with the main request.
With one request at a time they belong behind the main request, not in front of it.

### 4. Render the history token-identically to what was generated

**Remember generated tokens.** The server should remember the generated tokens per response, keyed by
normalized text plus tool calls, or `tool_use.id`/`call_id`. If it finds the response again in the history,
it inserts these tokens instead of re-rendering.

**This saves:**
- Harnesses that omit thinking: OpenHands, Roo/Zoo, Aider
- Harnesses that return it as text: Zed (default)
- Harnesses that serialize differently: argument order, `TrimSpace` in Crush, OpenCode V1

This is also the comment in `generator.h` ("the response is often tokenized differently in the process").

**Drawback:** The model sees its old thinking although the harness removed it. With `preserve_thinking`
in the template, this is the intended form anyway.

**Normalize when comparing prefixes:**
- ignore `cache_control`
- treat string and `[{type:text}]` the same
- remove the `x-anthropic-billing-header` line, as api.anthropic.com does (agent, not checked ourselves)
- output signatures stably

**Effort line:** The effort line stands **at the very front** of the system prompt today (`chat.cpp` lines 100–112). A
change of `effort` therefore breaks from token ~5. Claude Code changes effort with `/effort` and `--effort`.
Check whether the template demands it at the front. Otherwise accept it, since it happens rarely.

### 5. `/v1/responses` for Codex

Minimum scope according to the agent (source `codex-rs`, confirmed by our own capture):

**Input:**
- `input` with these items:
  - `message`: developer, user, assistant; contents `input_text`, `output_text`
  - `function_call` and `function_call_output`
  - `custom_tool_call` and `custom_tool_call_output`
  - `reasoning`
- `tools` with the types function, custom, namespace and web_search
- `reasoning.effort`
- `text.format`

**Output:**
- `response.created`
- `response.output_item.added` and `.done` with the complete item
- `response.output_text.delta`
- `response.reasoning_text.delta`
- `response.completed` with complete `usage` including `cached_tokens`

**Implementation:**
- The first `developer` item becomes the system prompt.
- Later developer items become user text, as with Claude Code.
- Offer `namespace` tools flat as a function with a prefix.
- Offer freeform `apply_patch` as a tool with a string parameter and answer it as `custom_tool_call`.

**Catalog:** serve `GET /v1/models` with `data` (OpenAI) **and** `models` (Codex ModelInfo with `context_window:
16384`). Alternatively ship a `model_catalog_json`.

### 6. Context to 32K, metadata and error forms

**32K context:**
- **Why:** The defaults of Claude Code (20K), Qwen Code (15K), Crush (12K + summarize at 12.8K) and
  OpenHands (recommends ≥22–32K) do not fit into 16K. The KV cache costs ~28.4 KB per token (FP16, 06 §9),
  so ~0.9 GB VRAM at 32K.
- **But:** A 32K prompt takes 13 min at 40 tok/s. More context replaces neither lean profiles nor
  caching. The real lever is prefill speed.

**Metadata:**
- `/v1/models` with `meta.n_ctx` and `context_length`; Goose, Crush (llamacpp type) and Zed read that.
- `/props` in llama.cpp style with `n_ctx`, `supports_tool_calls` and `supports_preserve_reasoning` (Zed).

**Error forms:**
- Anthropic: 400 with "prompt is too long: N tokens > M maximum".
- OpenAI: 400 with `code: context_length_exceeded`, text "This model's maximum context length is …" and the
  fields `n_prompt_tokens`/`n_ctx`.
- Always cap `max_tokens`, never reject. ie-server already does this (`room`).

**Accept robustly:**
- tools without `parameters`
- `stream:false`
- `max_completion_tokens`
- `tool_choice:"required"`
- unknown beta fields (`defer_loading`, `strict`, `eager_input_streaming`)
- `chat_template_kwargs.enable_thinking`
- `HEAD /api/hello`

**Report usage:**
- Chat: `prompt_tokens_details.cached_tokens` and additionally `cache_read_input_tokens` (Roo/Zoo reads that).
- Anthropic: `input_tokens + cache_read_input_tokens` = real prompt.

### 7. Ship client profiles

For example in the `justfile` next to `just claude`.

**Claude Code:**
```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=ie \
ANTHROPIC_MODEL=qwen3.8-flash-next-coder ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-flash-next-coder \
CLAUDE_CODE_ATTRIBUTION_HEADER=0 CLAUDE_CODE_DISABLE_TERMINAL_TITLE=1 \
CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION=false CLAUDE_CODE_ENABLE_AWAY_SUMMARY=0 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 CLAUDE_CODE_GATEWAY_HINT_HEADERS=1 \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=16384 CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096 \
CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000 \
claude --tools Bash,Read,Edit,Write,Glob,Grep --strict-mcp-config
```
- Measured, this gives ~4.8K tokens upfront and no side requests. The title request dropped with the title switch.
- `CLAUDE_CODE_ENABLE_AWAY_SUMMARY`, `CLAUDE_CODE_MAX_*` and `CLAUDE_STREAM_IDLE_TIMEOUT_MS` were not tested
  individually by us.
- Personal hooks and CLAUDE.md cost extra: ~7K.

**For the other harnesses:**

| Harness | Profile |
|---|---|
| Codex | `config.toml` from the Codex section; needs item 5 |
| Goose | custom provider file, extensions off, `GOOSE_DISABLE_SESSION_NAMING`; at ~1.4K the smallest |
| OpenCode | V2, title off, set `limit`, `headerTimeout:false` |
| Qwen Code | `contextWindowSize: 16384`, memory and follow-up suggestions off, `timeout: 0`, `streamIdleTimeoutMs: 0` |

**Model name:**
- The name `qwen3.8-flash-next-coder` is good for Claude Code because it does not start with `claude-`.
- For Qwen Code it is bad (1M assumed); only the explicit window helps there.
- Avoid `gpt` and `claude` in the name (OpenCode V1).

**Port:** not 8000 or 1234 (OpenCode V2 looks for local servers there automatically).

## Open points

- No harness calls `count_tokens` in our captures. When Claude Code uses it is **not
  substantiated**.
- Whether Claude Code's auto mode classifies commands for third-party endpoints is **not substantiated**.
- Whether Codex title threads send the base instructions along is **not substantiated**.
- Whether SSE comments reset Codex's idle timer is **not substantiated**.
- Prompt sizes of Cline v4, OpenCode V2, Kilo, Zed, Roo/Zoo, Continue and OpenHands are only estimated.
  They could be measured with the fake server; the scripts exist but are not in the repo.
- The figure "40 tok/s prefill" is the premise of the task, not newly measured.
