#include "client/agent.h"
#include "third_party/nlohmann/json.hpp"

namespace tona::client {

namespace {

using nlohmann::json;

// The provider as inline config; OpenCode merges it last, over the user's own config. The key and the project header
// come from the environment of the agent, so neither stands in the config text.
json opencode_config(const AgentContext& c) {
    // A long prefill sends no chunk for minutes and OpenCode aborts after 5 minutes by default: timeouts off
    const json options = {{"baseURL", "http://127.0.0.1:" + std::to_string(c.port) + "/v1"},
                          {"apiKey", "{env:TONA_API_KEY}"},
                          {"headers", {{"X-Tona-Project", "{env:TONA_PROJECT}"}}},
                          {"timeout", false},
                          {"headerTimeout", false},
                          {"chunkTimeout", false}};
    const json model = {{"name", c.model_name},
                        {"tool_call", true},
                        {"reasoning", true},
                        {"interleaved", {{"field", "reasoning_content"}}},
                        {"limit", {{"context", c.n_ctx}, {"output", 8192}}}};
    const json provider = {{"npm", "@ai-sdk/openai-compatible"},
                           {"name", "tona-server"},
                           {"options", options},
                           {"models", {{c.model_id, model}}}};
    return {{"$schema", "https://opencode.ai/config.json"},
            {"instructions", json::array({c.hint_file})},
            {"provider", {{"tona", provider}}},
            {"model", "tona/" + c.model_id},
            {"small_model", "tona/" + c.model_id},
            // The title request runs before the first answer and costs time (research/07-harnesses.md)
            {"agent", {{"title", {{"disable", true}}}}}};
}

}  // namespace

// OpenCode over Chat Completions
std::string opencode_command(const AgentContext& c, const std::vector<std::string>& user_args, AgentCommand& out) {
    out.program = "opencode";
    out.env.emplace_back("TONA_API_KEY", c.key);
    out.env.emplace_back("TONA_PROJECT", c.project);
    // The lean profile skips the scan of ~/.claude and ~/.agents for skills (about 4.2K tokens)
    if (c.settings.tools == "lean") {
        out.env.emplace_back("OPENCODE_DISABLE_EXTERNAL_SKILLS", "1");
        out.env.emplace_back("OPENCODE_DISABLE_CLAUDE_CODE_SKILLS", "1");
    }
    out.env.emplace_back("OPENCODE_CONFIG_CONTENT", opencode_config(c).dump());
    out.args = user_args;
    return {};
}

}  // namespace tona::client
