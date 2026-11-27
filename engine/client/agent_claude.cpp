#include <string>

#include "client/agent.h"

namespace tona::client {

namespace {

// Six tools without MCP: about 4.8K instead of about 20K tokens up front (research/07-harnesses.md)
constexpr const char* kLeanTools = "Bash,Read,Edit,Write,Glob,Grep";

// From a context of 128K on Claude Code compacts at TONA_COMPACT; below it the window would exceed the context
constexpr int64_t kCompactMinContext = 131072;

}  // namespace

// Claude Code over the Anthropic API. Everything the agent has (tools, MCP servers, skills) stays; only the title and
// suggestion side requests and the terminal title are off, because every extra request costs the one local server.
std::string claude_command(const AgentContext& c, const std::vector<std::string>& user_args, AgentCommand& out) {
    const AgentSettings& s = c.settings;
    out.program = "claude";
    auto set = [&out](const char* k, const std::string& v) { out.env.emplace_back(k, v); };
    if (c.n_ctx >= kCompactMinContext) set("CLAUDE_CODE_AUTO_COMPACT_WINDOW", s.compact);
    set("ANTHROPIC_BASE_URL", "http://127.0.0.1:" + std::to_string(c.port));
    set("ANTHROPIC_AUTH_TOKEN", c.key);
    set("ANTHROPIC_MODEL", c.model_id);
    set("ANTHROPIC_DEFAULT_HAIKU_MODEL", c.model_id);
    set("CLAUDE_CODE_MAX_CONTEXT_TOKENS", std::to_string(c.n_ctx));
    set("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "8192");
    set("CLAUDE_STREAM_IDLE_TIMEOUT_MS", "1800000");  // a long prefill sends nothing for minutes
    set("CLAUDE_CODE_ATTRIBUTION_HEADER", "0");
    set("CLAUDE_CODE_DISABLE_TERMINAL_TITLE", "1");
    set("CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION", "false");
    set("CLAUDE_CODE_ENABLE_AWAY_SUMMARY", "0");
    set("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1");
    set("CLAUDE_CODE_GATEWAY_HINT_HEADERS", "1");
    set("ANTHROPIC_CUSTOM_HEADERS", "X-Tona-Project: " + c.project);

    out.args = {"--permission-mode", s.mode, "--append-system-prompt", c.hint_text};
    // Thinking level medium unless the call names one: 21/23 in 8 min against high 20/23 in 38 min
    // (research/measurements/2026-10-26-effort.md); /effort switches it inside the session
    if (!has_arg_with_prefix(user_args, "--effort")) out.args.insert(out.args.end(), {"--effort", s.effort});
    if (!s.tools.empty())
        out.args.insert(out.args.end(), {"--strict-mcp-config", "--tools", s.tools == "lean" ? kLeanTools : s.tools});
    out.args.insert(out.args.end(), user_args.begin(), user_args.end());
    return {};
}

}  // namespace tona::client
