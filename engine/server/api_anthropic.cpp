// Anthropic /v1/messages request -> Request
#include "server/api.h"
#include "server/api_internal.h"

namespace tona::server {

namespace {

// Adds one content block to cur; tool results become their own "tool" messages
void add_block(Request& q, Message& cur, const json& b) {
    const std::string t = b.value("type", "");
    if (t == "tool_result") {
        std::string s = b.contains("content") ? text_of(b["content"]) : "";
        if (b.value("is_error", false)) s = "Error: " + s;
        q.msgs.push_back({"tool", s, "", {}});
    } else if (t == "text") {
        cur.content += (cur.content.empty() ? "" : "\n") + b.value("text", "");
    } else if (t == "thinking") {
        cur.reasoning += b.value("thinking", "");
    } else if (t == "tool_use") {
        cur.calls.push_back({b.value("id", ""), b.value("name", ""), b.value("input", json::object())});
    } else if (t == "image" || t == "document") {
        cur.content += (cur.content.empty() ? "" : "\n") + std::string("[attachment omitted]");
    }
}

void add_blocks(Request& q, const std::string& role, const json& c) {
    Message cur{role, "", "", {}};
    for (const auto& b : c) add_block(q, cur, b);
    if ((!cur.content.empty() || !cur.reasoning.empty() || !cur.calls.empty() || role == "assistant") &&
        (!(role == "user" && cur.content.empty())))
        q.msgs.push_back(std::move(cur));
}

// Claude Code prepends a text block whose cch= value changes every request; left in, the prompt differs from the first
// token on and the prefix cache never hits. Only this exact header prefix is dropped.
bool is_billing_header(const json& b) {
    return b.is_object() && b.value("type", "") == "text" && b.contains("text") && b["text"].is_string() &&
           b["text"].get<std::string>().starts_with("x-anthropic-billing-header:");
}

void read_system(Request& q, const json& r) {
    if (!r.contains("system")) return;
    const json& sys = r["system"];
    if (!sys.is_array()) {
        q.msgs.push_back({"system", text_of(sys), "", {}});
        return;
    }
    json kept = json::array();
    for (const auto& b : sys)
        if (!is_billing_header(b)) kept.push_back(b);
    if (kept.empty() && !sys.empty()) return;  // only the header: same as no system prompt
    q.msgs.push_back({"system", text_of(kept), "", {}});
}

void read_messages(Request& q, const json& r) {
    read_system(q, r);
    for (const auto& m : r.at("messages")) {
        std::string role = m.at("role");
        const json& c = m.at("content");
        // The template has no system messages in the middle of the history (Claude Code sends them)
        if (role == "system") {
            add_system_message(q.msgs, text_of(c));
            continue;
        }
        if (c.is_string())
            q.msgs.push_back({role, c.get<std::string>(), "", {}});
        else
            add_blocks(q, role, c);
    }
    merge_users(q.msgs);
}

void read_tools(Request& q, const json& r) {
    if (!r.contains("tools")) return;
    for (const auto& t : r["tools"])
        if (t.contains("input_schema"))  // skip server tools (web_search ...) without a schema
            q.tools.push_back({{"type", "function"},
                               {"function",
                                {{"name", t.value("name", "")},
                                 {"description", t.value("description", "")},
                                 {"parameters", t["input_schema"]}}}});
}

void read_thinking(Request& q, const json& r, bool think_default) {
    q.thinking = think_default;
    if (r.contains("thinking") && r["thinking"].is_object()) {
        const std::string ty = r["thinking"].value("type", "");
        q.thinking = ty == "enabled" || ty == "adaptive";
        if (ty == "enabled" && r["thinking"].contains("budget_tokens")) {
            q.think_budget = std::max(1, opt(r["thinking"], "budget_tokens", 32000));
            q.effort = q.think_budget < 4000 ? "low" : q.think_budget < 16000 ? "medium" : "high";
        }
    }
    if (r.contains("output_config") && r["output_config"].contains("effort") &&
        r["output_config"]["effort"].is_string())
        q.effort = effort_level(r["output_config"]["effort"]);
}

}  // namespace

Request from_anthropic(const json& r, bool think_default) {
    Request q;
    read_messages(q, r);
    read_tools(q, r);
    read_thinking(q, r, think_default);
    q.max_tokens = opt(r, "max_tokens", 4096);
    read_sampling(q, r);
    if (r.contains("stop_sequences") && r["stop_sequences"].is_array())
        for (const auto& s : r["stop_sequences"]) q.stops.push_back(s);
    std::string choice = "auto", name;
    if (r.contains("tool_choice") && r["tool_choice"].is_object())
        choice = r["tool_choice"].value("type", "auto"), name = r["tool_choice"].value("name", "");
    finish_request(q, choice, name);
    check_max_tokens(q);
    return q;
}

}  // namespace tona::server
