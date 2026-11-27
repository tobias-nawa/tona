#include "server/api.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "server/api_internal.h"

namespace tona::server {

std::string text_of(const json& c) {
    if (c.is_string()) return c.get<std::string>();
    std::string s;
    if (c.is_array())
        for (const auto& b : c) {
            const std::string t = b.value("type", "");
            if (t == "text")
                s += (s.empty() ? "" : "\n") + b.value("text", "");
            else if (t == "image" || t == "image_url")
                s += (s.empty() ? "" : "\n") + std::string("[image omitted]");
        }
    return s;
}

// A system message after the start of the history (Claude Code sends "output style" and similar reminders after every
// tool result) is appended to the user or tool message before it, not turned into a user turn of its own: the prompt
// "user: <tool_response> / user: reminder / assistant" is not the format the model was trained on, and with it the
// probability of <|im_end|> at the places where a turn could stop was 4 to 7 times higher than with the reminder inside
// the tool response (research/measurements/2026-11-10-fp8-stop.md, Result 8). Appending keeps the content, and the
// prompt of request N is still a prefix of request N + 1, which the prompt cache needs. Before the first
// non-system message it is one more system message (the template joins them); after an assistant message it stays a
// user turn.
void add_system_message(std::vector<Message>& msgs, std::string text) {
    if (text.empty()) return;
    const bool leading = std::all_of(msgs.begin(), msgs.end(), [](const Message& m) { return m.role == "system"; });
    if (leading) {
        msgs.push_back({"system", std::move(text), "", {}});
    } else if (msgs.back().role == "user" || msgs.back().role == "tool") {
        msgs.back().content += (msgs.back().content.empty() ? "" : "\n\n") + text;
    } else {
        msgs.push_back({"user", std::move(text), "", {}});
    }
}

// The template would turn each of them into its own turn
void merge_users(std::vector<Message>& msgs) {
    std::vector<Message> out;
    for (auto& m : msgs) {
        if (m.role == "user" && !out.empty() && out.back().role == "user" && out.back().calls.empty())
            out.back().content += "\n\n" + m.content;
        else
            out.push_back(std::move(m));
    }
    msgs = std::move(out);
}

std::vector<ChatPiece> request_pieces(const ChatFormat& fmt, const Request& q) {
    auto pieces = fmt.render_pieces(q.msgs, q.tools, q.thinking, q.effort);
    if (!q.forced.empty()) pieces.push_back({q.forced, false});
    if (!q.prefill.empty()) pieces.push_back({q.prefill, true});
    return pieces;
}

void prime(OutputParser& p, const Request& q) {
    if (!q.forced.empty() || !q.prefill.empty()) (void)p.feed(q.forced + q.prefill);
}

namespace {

// A last assistant message without tool calls becomes the prefill
void take_prefill(Request& q) {
    if (q.msgs.empty() || q.msgs.back().role != "assistant" || !q.msgs.back().calls.empty()) return;
    std::string s = q.msgs.back().content;
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();  // Anthropic: no trailing whitespace
    q.msgs.pop_back();
    q.prefill = s, q.thinking = false;
}

bool has_tool(const json& tools, const std::string& name) {
    for (const auto& t : tools)
        if (t.contains("function") && t["function"].value("name", "") == name) return true;
    return false;
}

}  // namespace

// A last assistant message without tool calls becomes the start of the answer (Anthropic prefill; OpenAI as in
// llama.cpp). tool_choice none drops the tools; a required tool makes the answer start with the tool call. Both
// disable thinking: the template closes <think> before the answer, and Anthropic allows neither with thinking.
void finish_request(Request& q, const std::string& choice, const std::string& name) {
    take_prefill(q);
    if (choice == "none") q.tools = json::array();
    if (q.tools.empty() || q.tools.is_null()) return;
    if (choice == "any" || choice == "required") q.forced = "<tool_call>\n<function=";
    if (choice == "tool" || choice == "function") {
        if (!has_tool(q.tools, name)) throw std::invalid_argument("tool_choice: unknown tool " + name);
        q.forced = "<tool_call>\n<function=" + name + ">\n";
    }
    if (!q.forced.empty()) q.thinking = false, q.prefill.clear();
}

void check_max_tokens(Request& q) {
    if (q.max_tokens < 0) throw std::invalid_argument("max_tokens: must not be negative");
    if (q.max_tokens > 0) return;
    if (q.stream) throw std::invalid_argument("max_tokens 0 (warm-up) does not support stream: true");
    if (!q.forced.empty() || !q.prefill.empty())
        throw std::invalid_argument(
            "max_tokens 0 (warm-up) does not support a forced tool_choice or a last assistant message");
    q.warm = true;
}

void read_sampling(Request& q, const json& r) {
    q.temperature = opt(r, "temperature", -1.0f), q.top_p = opt(r, "top_p", 1.0f), q.top_k = opt(r, "top_k", 0);
    q.seed = opt(r, "seed", (uint64_t)0);  // not in the Anthropic API, but useful for reproducible runs
    q.stream = opt(r, "stream", false);
}

// Unknown levels (also "auto") give high, the default of Claude Code
std::string effort_level(const std::string& e) {
    for (const char* k : {"low", "medium", "high", "xhigh", "max"})
        if (e == k) return e;
    return "high";
}

}  // namespace tona::server
