#include <fstream>
#include <sstream>

#include "client/agent.h"
#include "third_party/nlohmann/json.hpp"

namespace tona::client {

namespace {

using nlohmann::json;

json pi_provider(const AgentContext& c) {
    const json model = {
        {"id", c.model_id},   {"name", c.model_name},
        {"reasoning", true},  {"contextWindow", c.n_ctx},
        {"maxTokens", 16384}, {"cost", {{"input", 0}, {"output", 0}, {"cacheRead", 0}, {"cacheWrite", 0}}}};
    return {{"name", "tona-server"},
            {"baseUrl", "http://127.0.0.1:" + std::to_string(c.port)},
            {"api", "anthropic-messages"},
            {"apiKey", "$TONA_API_KEY"},
            {"headers", {{"X-Tona-Project", "$TONA_PROJECT"}}},
            {"models", json::array({model})}};
}

// The models file as it is, or {} when it is missing or empty. False when it is not a JSON object: pi's other
// providers must not be lost, so a file that cannot be read is left alone.
bool read_models(const std::string& path, json& doc) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    if (ss.str().find_first_not_of(" \t\r\n") == std::string::npos) return doc = json::object(), true;
    doc = json::parse(ss.str(), nullptr, false);
    return doc.is_object() && (!doc.contains("providers") || doc["providers"].is_object());
}

}  // namespace

// pi over the Anthropic API. pi reads custom providers only from <agent-dir>/models.json, so tona keeps its own
// provider "tona" in there (written per call: port and context come from the running server); every other entry of
// the file stays as it is.
std::string pi_command(const AgentContext& c, const std::vector<std::string>& user_args, AgentCommand& out) {
    const std::string file = c.settings.pi_dir + "/models.json";
    json doc;
    if (!read_models(file, doc)) return "could not update " + file;
    doc["providers"]["tona"] = pi_provider(c);
    if (!write_file_atomic(file, doc.dump(2) + "\n").empty()) return "could not update " + file;
    out.program = "pi";
    out.env.emplace_back("TONA_API_KEY", c.key);
    out.env.emplace_back("TONA_PROJECT", c.project);
    out.args = {"--model", "tona/" + c.model_id, "--append-system-prompt", c.hint_text};
    if (!has_arg_with_prefix(user_args, "--thinking")) out.args.insert(out.args.end(), {"--thinking", "medium"});
    out.args.insert(out.args.end(), user_args.begin(), user_args.end());
    return {};
}

}  // namespace tona::client
