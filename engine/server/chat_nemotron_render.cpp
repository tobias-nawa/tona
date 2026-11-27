// Chat template rendering of the Nemotron family: the GGUF's template reimplemented as pieces. Every deviation from the
// template is a bug (tests/chat_vs_jinja.py compares byte for byte), so the quirks are kept on purpose: the system
// block is always written, content is never trimmed (except where the template trims), a tool message at the very
// start opens no user turn, and values render as Python would print them (True, None, repr of lists).
#include <algorithm>
#include <stdexcept>
#include <string_view>

#include "server/chat_nemotron.h"
#include "server/chat_qwen3.h"
#include "server/chat_util.h"

namespace tona::server {

namespace {

// The special tokens the scaffold writes. Foreign content is encoded without special tokens, so a piece boundary is
// only needed at these; everything between two of them is one piece, which keeps the token ids equal to those of the
// whole text (a word never crosses a special token, but it does cross a boundary between scaffold and content).
constexpr std::string_view kSpecials[] = {"<|im_start|>", "<|im_end|>",   "<think>",         "</think>",
                                          "<tool_call>",  "</tool_call>", "<tool_response>", "</tool_response>"};

struct Segment {
    std::string text;
    bool value;  // foreign: encoded without special tokens
};

class Segments {
public:
    void scaffold(const std::string& x) { add(x, false); }
    void value(const std::string& x) { add(x, true); }

    std::vector<ChatPiece> pieces() const {
        std::vector<ChatPiece> out;
        ChatPiece cur{std::string(), false};
        auto flush = [&] {
            if (!cur.text.empty()) out.push_back(cur);
            cur = {std::string(), false};
        };
        for (const Segment& s : segs_) {
            if (s.value) {
                cur.text += s.text, cur.content = true;
                continue;
            }
            std::string_view rest = s.text;
            while (!rest.empty()) {
                size_t at = std::string_view::npos, len = 0;
                for (std::string_view sp : kSpecials)
                    if (const size_t p = rest.find(sp); p < at) at = p, len = sp.size();
                cur.text += rest.substr(0, at == std::string_view::npos ? rest.size() : at);
                if (at == std::string_view::npos) break;
                flush();
                out.push_back({std::string(rest.substr(at, len)), false});
                rest.remove_prefix(at + len);
            }
        }
        flush();
        return out;
    }

private:
    void add(const std::string& x, bool value) {
        if (!x.empty()) segs_.push_back({x, value});
    }
    std::vector<Segment> segs_;
};

// Python repr of a string: single quotes unless the text has a ' and no "
std::string py_quote(const std::string& s) {
    const char q = s.contains('\'') && !s.contains('"') ? '"' : '\'';
    std::string o(1, q);
    for (const unsigned char c : s) {
        if (c == '\\' || c == (unsigned char)q) {
            o += '\\', o += (char)c;
        } else if (c == '\n') {
            o += "\\n";
        } else if (c == '\r') {
            o += "\\r";
        } else if (c == '\t') {
            o += "\\t";
        } else if (c < 0x20 || c == 0x7f) {
            static const char* hex = "0123456789abcdef";
            o += "\\x", o += hex[c >> 4], o += hex[c & 15];
        } else {
            o += (char)c;
        }
    }
    return o + q;
}

// Python repr of a JSON value (what str() prints for a list, a dict or inside them)
std::string py_repr(const json& j) {
    if (j.is_null()) return "None";
    if (j.is_boolean()) return j.get<bool>() ? "True" : "False";
    if (j.is_string()) return py_quote(j.get<std::string>());
    if (j.is_array()) {
        std::string o = "[";
        for (size_t i = 0; i < j.size(); i++) o += (i ? ", " : "") + py_repr(j[i]);
        return o + "]";
    }
    if (j.is_object()) {
        std::string o = "{";
        for (auto it = j.begin(); it != j.end(); ++it)
            o += std::string(o.size() > 1 ? ", " : "") + py_quote(it.key()) + ": " + py_repr(it.value());
        return o + "}";
    }
    return j.dump();
}

// Jinja's `string` filter
std::string py_str(const json& j) { return j.is_string() ? j.get<std::string>() : py_repr(j); }

// `x | tojson if x is mapping or (x is sequence and x is not string) else x | string`
std::string json_or_str(const json& j) { return j.is_object() || j.is_array() ? to_pyjson(j) : py_str(j); }

// The macro render_extra_keys: every key of a mapping that is not handled, as its own tag
void extra_keys(Segments& b, const json& dict, std::initializer_list<const char*> handled) {
    if (!dict.is_object()) return;
    for (auto it = dict.begin(); it != dict.end(); ++it) {
        if (std::any_of(handled.begin(), handled.end(), [&](const char* h) { return it.key() == h; })) continue;
        b.scaffold("\n<"), b.value(it.key()), b.scaffold(">");
        b.value(json_or_str(it.value()));
        b.scaffold("</"), b.value(it.key()), b.scaffold(">");
    }
}

// Text of a field that the template prints as `field | trim` / `field | string`
void tagged(Segments& b, const char* tag, const std::string& text) {
    b.scaffold(std::string("\n<") + tag + ">"), b.value(text), b.scaffold(std::string("</") + tag + ">");
}

void render_parameters(Segments& b, const json& params) {
    const bool props = params.is_object() && params.contains("properties") && params["properties"].is_object();
    b.scaffold("\n<parameters>");
    if (props) {
        for (auto it = params["properties"].begin(); it != params["properties"].end(); ++it) {
            const json& f = it.value();
            b.scaffold("\n<parameter>");
            tagged(b, "name", it.key());
            if (f.is_object() && f.contains("type")) tagged(b, "type", py_str(f["type"]));
            if (f.is_object() && f.contains("description")) tagged(b, "description", trim(py_str(f["description"])));
            if (f.is_object() && f.contains("enum")) tagged(b, "enum", to_pyjson(f["enum"]));
            extra_keys(b, f, {"name", "type", "description", "enum"});
            b.scaffold("\n</parameter>");
        }
    }
    extra_keys(b, params, {"type", "properties", "required"});
    if (params.is_object() && params.contains("required")) tagged(b, "required", to_pyjson(params["required"]));
    b.scaffold("\n</parameters>");
}

void render_tool(Segments& b, const json& entry) {
    const json& tool = entry.is_object() && entry.contains("function") ? entry["function"] : entry;
    b.scaffold("\n<function>");
    tagged(b, "name", tool.is_object() && tool.contains("name") ? py_str(tool["name"]) : "");
    if (tool.is_object() && tool.contains("description")) tagged(b, "description", trim(py_str(tool["description"])));
    render_parameters(b, tool.is_object() && tool.contains("parameters") ? tool["parameters"] : json());
    extra_keys(b, tool, {"type", "name", "description", "parameters"});
    b.scaffold("\n</function>");
}

constexpr const char* kToolsFormat =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple "
    "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow "
    "the specified format: an inner <function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional "
    "reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If "
    "there is no function call available, answer the question like normal with your current knowledge and "
    "do not tell the user about function calls\n</IMPORTANT>";

void render_system(Segments& b, const std::string& system, const json& tools) {
    b.scaffold("<|im_start|>system\n"), b.value(system);
    if (tools.is_array() && !tools.empty()) {
        if (!system.empty()) b.scaffold("\n\n");
        b.scaffold("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (const json& t : tools) render_tool(b, t);
        b.scaffold(kToolsFormat);
    }
    b.scaffold("<|im_end|>\n");
}

// Text after the last </think>, before the first <think>
std::string after_last(const std::string& s, const std::string& tag) {
    const size_t p = s.rfind(tag);
    return p == std::string::npos ? s : s.substr(p + tag.size());
}

bool has(const std::string& s, const char* what) { return s.contains(what); }

// The history before the last user message keeps no reasoning (truncate_history_thinking, on by default)
std::string assistant_content(const Message& m, bool truncated) {
    std::string c = m.content;
    if (!trim(m.reasoning).empty())
        c = "<think>\n" + m.reasoning + "</think>" + c;
    else if (!has(c, "<think>") && !has(c, "</think>"))
        c = "<think></think>" + c;
    if (!truncated) return trim(c);
    if (m.calls.empty()) {
        if (has(c, "<think>") && has(c, "</think>")) c = "<think></think>" + after_last(c, "</think>");
        return trim(c);
    }
    if (has(c, "</think>"))
        c = after_last(c, "</think>");
    else if (has(c, "<think>"))
        c = c.substr(0, c.find("<think>"));
    return "<think></think>" + c;
}

std::string render_calls(const std::vector<ToolCall>& calls) {
    std::string a;
    for (const ToolCall& c : calls) {
        a += "<tool_call>\n<function=" + c.name + ">\n";
        if (c.args.is_object())
            for (auto it = c.args.begin(); it != c.args.end(); ++it)
                a += "<parameter=" + it.key() + ">\n" + json_or_str(it.value()) + "\n</parameter>\n";
        a += "</function>\n</tool_call>\n";
    }
    return a;
}

// Model answers are scaffold, so they tokenize exactly as when generated (prompt cache)
std::string render_assistant(const Message& m, bool truncated) {
    const std::string c = assistant_content(m, truncated);
    if (m.calls.empty()) return "<|im_start|>assistant\n" + c + "<|im_end|>\n";
    return "<|im_start|>assistant\n" + c + "\n" + render_calls(m.calls) + "<|im_end|>\n";
}

struct Turn {
    const std::vector<Message>& msgs;
    int64_t last_user;  // index of the last user message, -1 if there is none
    bool medium;
};

void render_message(Segments& b, const Turn& t, size_t i) {
    const Message& m = t.msgs[i];
    const int64_t at = (int64_t)i;
    if (m.role == "assistant") {
        b.scaffold(render_assistant(m, at < t.last_user));
    } else if (m.role == "tool") {
        if (i > 0 && t.msgs[i - 1].role != "tool") b.scaffold("<|im_start|>user\n");
        b.scaffold("<tool_response>\n"), b.value(m.content), b.scaffold("\n</tool_response>\n");
        if (i + 1 == t.msgs.size() || t.msgs[i + 1].role != "tool") b.scaffold("<|im_end|>\n");
    } else {
        b.scaffold("<|im_start|>"), b.value(m.role + "\n" + m.content);
        if (m.role == "user" && at == t.last_user && t.medium) b.value("\n\n{reasoning effort: efficient}");
        b.scaffold("<|im_end|>\n");
    }
}

}  // namespace

std::vector<ChatPiece> NemotronFormat::render_pieces(const std::vector<Message>& msgs, const json& tools, bool thinking,
                                                     const std::string& effort) const {
    if (msgs.empty()) throw std::runtime_error("no messages");
    const bool has_system = msgs[0].role == "system";
    const std::vector<Message> loop(msgs.begin() + (has_system ? 1 : 0), msgs.end());
    // The template knows one reduced level (medium_effort); low is the nearest to it
    const bool medium = effort == "medium" || effort == "low";
    if (!medium && !effort.empty() && effort != "xhigh" && effort != "high" && effort != "max")
        throw std::runtime_error("effort " + effort + ": allowed are xhigh, medium, low");
    int64_t last_user = -1;
    for (size_t i = 0; i < loop.size(); i++)
        if (loop[i].role == "user") last_user = (int64_t)i;

    Segments b;
    render_system(b, has_system ? msgs[0].content : std::string(), tools);
    const Turn turn{loop, last_user, medium};
    for (size_t i = 0; i < loop.size(); i++) render_message(b, turn, i);
    b.scaffold(thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think></think>");
    return b.pieces();
}

std::unique_ptr<OutputParser> NemotronFormat::make_parser(bool thinking, const json& tools) const {
    return std::make_unique<Qwen3CoderParser>(thinking, tools);
}

std::string NemotronFormat::think_close(bool) const { return "</think>"; }

}  // namespace tona::server
