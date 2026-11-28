// Qwen3CoderParser: splits the model output into thinking, text and tool calls while it streams
#include <cctype>
#include <string_view>
#include <utility>

#include "server/chat_qwen3.h"
#include "server/chat_util.h"

namespace tona::server {

namespace {

// Properties of the named tool's parameter schema, or null
const json* find_props(const json& tools, const std::string& name) {
    const json* props = nullptr;
    if (!tools.is_array()) return props;
    for (const auto& t : tools) {
        const json& fn = t.contains("function") ? t["function"] : t;
        if (fn.value("name", "") == name && fn.contains("parameters") && fn["parameters"].contains("properties"))
            props = &fn["parameters"]["properties"];
    }
    return props;
}

std::string param_type(const json* props, const std::string& name) {
    if (!props || !props->contains(name)) return "string";
    if (!(*props)[name].contains("type")) return "any";
    const json& t = (*props)[name]["type"];
    if (t.is_string()) return t.get<std::string>();
    return t.is_array() && !t.empty() && t[0].is_string() ? t[0].get<std::string>() : "string";
}

// Older Qwen form: {"name": ..., "arguments": {...}}
ToolCall parse_json_call(const std::string& b) {
    ToolCall c;
    try {
        const json j = json::parse(b);
        c.name = j.value("name", "");
        if (j.contains("arguments"))
            c.args = j["arguments"].is_string() ? json::parse(j["arguments"].get<std::string>()) : j["arguments"];
    } catch (const std::exception&) {
        c.args = json::object();  // unparsable arguments stay empty
    }
    return c;
}

// Value of the parameter whose tag ends at pe; ve is where the value ends, closed tells if </parameter> was seen
std::string param_value(const std::string& b, size_t pe, size_t& ve, bool& closed) {
    ve = b.find("</parameter>", pe);
    closed = ve != std::string::npos;
    if (!closed) {
        ve = b.find("<parameter=", pe);
        if (ve == std::string::npos) ve = b.find("</function>", pe);
        if (ve == std::string::npos) ve = b.size();
    }
    std::string v = b.substr(pe + 1, ve - pe - 1);
    if (starts_with(v, "\n")) v.erase(0, 1);
    if (ends_with(v, "\n")) v.pop_back();
    return v;
}

bool word_char(char ch) { return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'; }

// Length of the Python literal True, False or None that starts at i as a whole word, with its JSON spelling
size_t py_literal_at(const std::string& v, size_t i, const char*& js) {
    static const std::pair<std::string_view, const char*> kWords[] = {
        {"True", "true"}, {"False", "false"}, {"None", "null"}};
    if (i > 0 && word_char(v[i - 1])) return 0;
    for (const auto& [py, j] : kWords) {
        const size_t n = py.size();
        if (v.compare(i, n, py) == 0 && (i + n == v.size() || !word_char(v[i + n]))) {
            js = j;
            return n;
        }
    }
    return 0;
}

// v with True, False and None outside double-quoted strings replaced by their JSON spelling
std::string py_literals_to_json(const std::string& v) {
    std::string out;
    bool in_str = false;
    for (size_t i = 0; i < v.size(); i++) {
        const char ch = v[i];
        const char* js = nullptr;
        const size_t n = in_str ? 0 : py_literal_at(v, i, js);
        if (n) {
            out += js;
            i += n - 1;
            continue;
        }
        out += ch;
        if (in_str && ch == '\\' && i + 1 < v.size()) {
            out += v[++i];
        } else if (ch == '"') {
            in_str = !in_str;
        }
    }
    return out;
}

// The Coder writes Python literals (True, False, None) in non-string parameters, also inside arrays and objects
// (256 "unlock": "False" in the BFCL run, research/measurements/2026-11-15-pc-bfcl-tool-path.md); JSON parsing alone
// would leave "False" a truthy string. vLLM's tool parsers fall back the same way (json.loads, then
// safe_literal_eval, vllm/tool_parsers/hy_v3_tool_parser.py).
json parse_value(const std::string& v) {
    try {
        return json::parse(v);
    } catch (const std::exception&) {
        try {
            return json::parse(py_literals_to_json(trim(v)));
        } catch (const std::exception&) {
            return v;
        }
    }
}

void parse_params(const std::string& b, size_t pos, const json* props, ToolCall& c) {
    while (true) {
        const size_t ps = b.find("<parameter=", pos);
        if (ps == std::string::npos) break;
        const size_t pe = b.find('>', ps);
        if (pe == std::string::npos) break;
        const std::string name = trim(b.substr(ps + 11, pe - ps - 11));
        size_t ve = 0;
        bool closed = false;
        const std::string v = param_value(b, pe, ve, closed);
        if (param_type(props, name) == "string") {
            c.args[name] = v;
        } else {
            c.args[name] = parse_value(v);
        }
        pos = closed ? ve + 12 : ve;
    }
}

// Quantized Qwen3.8 models can emit <|im_start|> where the '<' of <function= belongs inside <tool_call>
// (vllm-project/vllm#69190). Accept that slip inside an already-open <tool_call> block.
size_t find_function_start(const std::string& b, size_t& prefix_len) {
    if (const size_t f = b.find("<function="); f != std::string::npos) {
        prefix_len = 10;
        return f;
    }
    if (const size_t f = b.find("<|im_start|>function="); f != std::string::npos) {
        prefix_len = 21;
        return f;
    }
    return std::string::npos;
}

}  // namespace

Qwen3CoderParser::Qwen3CoderParser(bool thinking, const json& tools) : st_(thinking ? think : text), tools_(tools) {}

// Emits text up to the end but holds back trailing whitespace and a possible start of the closing tag
void Qwen3CoderParser::flush_text(std::vector<Event>& ev, bool all) {
    if (at_start_) {
        size_t a = 0;
        while (a < buf_.size() && is_ws(buf_[a])) a++;
        buf_.erase(0, a);
        if (buf_.empty()) return;
        at_start_ = false;
    }
    size_t keep = 0;
    if (!all) {
        const std::string tag = st_ == think ? "</think>" : "<tool_call>";
        for (size_t k = std::min(tag.size() - 1, buf_.size()); k > 0; k--)
            if (ends_with(buf_, tag.substr(0, k))) {
                keep = k;
                break;
            }
        size_t e = buf_.size() - keep;
        while (e > 0 && is_ws(buf_[e - 1])) e--;
        keep = buf_.size() - e;
    } else {
        size_t e = buf_.size();
        while (e > 0 && is_ws(buf_[e - 1])) e--;
        buf_.resize(e);
    }
    const size_t n = buf_.size() - keep;
    if (n == 0) return;
    ev.push_back({st_ == think ? Event::thinking : Event::text, buf_.substr(0, n), {}});
    buf_.erase(0, n);
}

std::vector<OutputParser::Event> Qwen3CoderParser::feed(const std::string& piece) {
    std::vector<Event> ev;
    buf_ += piece;
    while (true) {
        if (st_ == think || st_ == text) {
            const std::string tag = st_ == think ? "</think>" : "<tool_call>";
            const size_t p = buf_.find(tag);
            if (p == std::string::npos) {
                flush_text(ev, false);
                return ev;
            }
            std::string rest = buf_.substr(p + tag.size());
            buf_.resize(p);
            flush_text(ev, true);
            buf_ = rest;
            st_ = st_ == think ? text : call;
            at_start_ = true;
        } else {
            const size_t p = buf_.find("</tool_call>");
            if (p == std::string::npos) return ev;
            ev.push_back({Event::tool_call, {}, parse_call(buf_.substr(0, p))});
            buf_.erase(0, p + 12);
            st_ = text;
            at_start_ = true;
        }
    }
}

std::vector<OutputParser::Event> Qwen3CoderParser::finish() {
    std::vector<Event> ev;
    if (st_ == call) {
        // Only a complete call counts at the end without </tool_call> (</function> seen, every parameter closed).
        // A truncated one (max_tokens, context full) would be written as a half file or half executed.
        const size_t fe = buf_.find("</function>");
        const std::string body = fe == std::string::npos ? std::string() : buf_.substr(0, fe);
        size_t open = 0, closed = 0;
        for (size_t p = 0; (p = body.find("<parameter=", p)) != std::string::npos; p++) open++;
        for (size_t p = 0; (p = body.find("</parameter>", p)) != std::string::npos; p++) closed++;
        ToolCall c = fe == std::string::npos || open != closed ? ToolCall{} : parse_call(body + "</function>");
        if (!c.name.empty())
            ev.push_back({Event::tool_call, {}, c});
        else
            truncated_ = true;
        buf_.clear();
    } else {
        flush_text(ev, true);
    }
    return ev;
}

ToolCall Qwen3CoderParser::parse_call(const std::string& body) const {
    const std::string b = trim(body);
    if (starts_with(b, "{")) return parse_json_call(b);
    ToolCall c;
    size_t prefix_len = 0;
    const size_t f = find_function_start(b, prefix_len);
    if (f == std::string::npos) return c;
    const size_t fe = b.find('>', f + prefix_len);
    if (fe == std::string::npos) return c;
    c.name = trim(b.substr(f + prefix_len, fe - f - prefix_len));
    parse_params(b, fe + 1, find_props(tools_, c.name), c);
    return c;
}

}  // namespace tona::server
