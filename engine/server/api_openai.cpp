// OpenAI /v1/chat/completions request -> Request
#include "server/api.h"
#include "server/api_internal.h"

namespace tona::server {

namespace {

// Tool call arguments arrive as a JSON string; unparsable ones become an empty object
ToolCall read_call(const json& t) {
    const json& f = t.at("function");
    json args = f.contains("arguments") ? f["arguments"] : json::object();
    if (args.is_string()) {
        try {
            args = json::parse(args.get<std::string>());
        } catch (const std::exception&) {
            args = json::object();
        }
    }
    return {t.value("id", ""), f.value("name", ""), args};
}

Message read_message(const json& m) {
    Message x;
    x.role = m.at("role");
    if (x.role == "developer") x.role = "system";
    if (m.contains("content") && !m["content"].is_null()) x.content = text_of(m["content"]);
    if (m.contains("reasoning_content") && m["reasoning_content"].is_string())
        x.reasoning = m["reasoning_content"];
    else if (m.contains("reasoning") && m["reasoning"].is_string())
        x.reasoning = m["reasoning"];
    if (m.contains("tool_calls") && m["tool_calls"].is_array())
        for (const auto& t : m["tool_calls"]) x.calls.push_back(read_call(t));
    return x;
}

void read_thinking(Request& q, const json& r, bool think_default) {
    q.thinking = think_default;
    if (r.contains("reasoning_effort") && r["reasoning_effort"].is_string()) {
        const std::string e = r["reasoning_effort"];
        q.thinking = e != "none" && e != "minimal";
        q.effort = effort_level(e);
    }
    if (r.contains("chat_template_kwargs") && r["chat_template_kwargs"].contains("enable_thinking"))
        q.thinking = r["chat_template_kwargs"]["enable_thinking"].get<bool>();
}

void read_stops(Request& q, const json& r) {
    if (!r.contains("stop")) return;
    if (r["stop"].is_string())
        q.stops.push_back(r["stop"]);
    else if (r["stop"].is_array())
        for (const auto& s : r["stop"]) q.stops.push_back(s);
}

}  // namespace

Request from_openai(const json& r, bool think_default) {
    Request q;
    for (const auto& m : r.at("messages")) {
        Message x = read_message(m);
        if (x.role == "system")
            add_system_message(q.msgs, std::move(x.content));
        else
            q.msgs.push_back(std::move(x));
    }
    merge_users(q.msgs);
    if (r.contains("tools") && r["tools"].is_array()) q.tools = r["tools"];
    read_thinking(q, r, think_default);
    // Without a limit, up to the end of the context as with OpenAI (the server caps it to the free space)
    q.max_tokens = opt(r, "max_completion_tokens", opt(r, "max_tokens", std::numeric_limits<int>::max()));
    read_sampling(q, r);
    read_stops(q, r);
    std::string choice = "auto", name;
    if (r.contains("tool_choice")) {
        const json& c = r["tool_choice"];
        if (c.is_string())
            choice = c.get<std::string>();
        else if (c.is_object() && c.contains("function") && c["function"].is_object())
            choice = "function", name = c["function"].value("name", "");
    }
    finish_request(q, choice, name);
    check_max_tokens(q);
    return q;
}

}  // namespace tona::server
