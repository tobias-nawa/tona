#include "client/agent.h"
#include "third_party/nlohmann/json.hpp"

namespace tona::client {

// Junie with a custom model profile, written per call because port and context come from the running server. Junie
// sends no max_tokens, so an answer could run to the end of the context: 8192 as for Claude Code.
std::string junie_command(const AgentContext& c, const std::vector<std::string>& user_args, AgentCommand& out) {
    using nlohmann::json;
    const std::string dir = c.state_dir + "/junie-models";
    const json profile = {{"id", c.model_id},
                          {"baseUrl", "http://127.0.0.1:" + std::to_string(c.port) + "/v1/chat/completions"},
                          {"apiType", "OpenAICompletion"},
                          {"displayName", c.model_name},
                          {"providerName", "tona-server"},
                          {"apiKey", "${TONA_API_KEY}"},
                          {"extraHeaders", {{"X-Tona-Project", "${TONA_PROJECT}"}}},
                          {"extraBody", {{"max_tokens", 8192}}},
                          {"maxContextLength", c.n_ctx}};
    if (const std::string err = write_file_atomic(dir + "/tona.json", profile.dump(2) + "\n"); !err.empty()) return err;
    out.program = "junie";
    out.env.emplace_back("TONA_API_KEY", c.key);
    out.env.emplace_back("TONA_PROJECT", c.project);
    out.args = {"--model-location", dir, "--model", "custom:tona", "--ide-guidelines", c.hint_text};
    out.args.insert(out.args.end(), user_args.begin(), user_args.end());
    return {};
}

}  // namespace tona::client
