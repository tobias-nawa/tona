// Chat template rendering of the Qwen3 family: the layout (system block with tools, user/assistant/tool turns) as
// pieces
#include <stdexcept>

#include "server/chat_qwen3.h"
#include "server/chat_util.h"

namespace tona::server {

namespace {

// Collects pieces: template scaffold (special tokens apply) and foreign content (no special tokens)
struct PieceBuilder {
    std::vector<ChatPiece> out;
    void put(const std::string& x, bool content) {
        if (x.empty()) return;
        if (out.empty() || out.back().content != content) out.push_back({std::string(), content});
        out.back().text += x;
    }
    void scaffold(const std::string& x) { put(x, false); }
    void content(const std::string& x) { put(x, true); }
};

constexpr const char* kToolsIntro = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
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

// The template knows three levels: xhigh (also high and max), medium (no line), low
std::string reasoning_line(bool thinking, const std::string& effort) {
    if (!thinking) return "";
    if (effort == "xhigh" || effort == "high" || effort == "max" || effort.empty())
        return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
               "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
               "clarity in the final answer.";
    if (effort == "low")
        return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
               "conclusion without unnecessary elaboration.";
    if (effort != "medium") throw std::runtime_error("effort " + effort + ": allowed are xhigh, medium, low");
    return "";
}

// Leading system messages joined into one; num_sys is how many were consumed
std::string merge_system(const std::vector<Message>& msgs, size_t& num_sys) {
    std::string merged;
    num_sys = 0;
    for (const auto& m : msgs) {
        if (m.role != "system" && m.role != "developer") break;
        const std::string c = trim(m.content);
        if (!c.empty()) merged += (merged.empty() ? "" : "\n") + c;
        num_sys++;
    }
    return merged;
}

void render_system(PieceBuilder& b, const std::string& merged, const std::string& reasoning, const json& tools) {
    if (tools.is_array() && !tools.empty()) {
        b.scaffold("<|im_start|>system\n");
        if (!reasoning.empty()) b.scaffold(reasoning + "\n\n");
        b.scaffold(kToolsIntro);
        for (const auto& t : tools) b.scaffold("\n"), b.content(to_pyjson(t));
        b.scaffold(kToolsFormat);
        if (!merged.empty()) b.scaffold("\n\n"), b.content(merged);
        b.scaffold("<|im_end|>\n");
    } else if (!merged.empty()) {
        b.scaffold("<|im_start|>system\n" + (reasoning.empty() ? "" : reasoning + "\n\n"));
        b.content(merged);
        b.scaffold("<|im_end|>\n");
    } else if (!reasoning.empty()) {
        b.scaffold("<|im_start|>system\n" + reasoning + "<|im_end|>\n");
    }
}

// The template keeps the reasoning of every earlier answer in the history. Model answers are scaffold, so they
// tokenize exactly as when generated, which lets the prompt cache reuse them.
std::string render_assistant(const Message& m, const std::string& content) {
    std::string a = "<|im_start|>assistant\n<think>\n" + trim(m.reasoning) + "\n</think>\n\n" + content;
    for (size_t k = 0; k < m.calls.size(); k++) {
        const ToolCall& c = m.calls[k];
        if (k == 0)
            a += content.empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=";
        else
            a += "\n<tool_call>\n<function=";
        a += c.name + ">\n";
        if (c.args.is_object())
            for (auto it = c.args.begin(); it != c.args.end(); ++it)
                a += "<parameter=" + it.key() + ">\n" +
                     (it.value().is_string() ? it.value().get<std::string>() : to_pyjson(it.value())) +
                     "\n</parameter>\n";
        a += "</function>\n</tool_call>";
    }
    return a + "<|im_end|>\n";
}

void render_message(PieceBuilder& b, const std::vector<Message>& msgs, size_t i) {
    const Message& m = msgs[i];
    const std::string content = trim(m.content);
    if (m.role == "system" || m.role == "developer") {
        throw std::runtime_error("system message only at the start");
    }
    if (m.role == "user") {
        b.scaffold("<|im_start|>user\n"), b.content(content), b.scaffold("<|im_end|>\n");
    } else if (m.role == "assistant") {
        b.scaffold(render_assistant(m, content));
    } else if (m.role == "tool") {
        if (i == 0 || msgs[i - 1].role != "tool") b.scaffold("<|im_start|>user");
        b.scaffold("\n<tool_response>\n"), b.content(content), b.scaffold("\n</tool_response>");
        if (i + 1 == msgs.size() || msgs[i + 1].role != "tool") b.scaffold("<|im_end|>\n");
    } else {
        throw std::runtime_error("unknown role " + m.role);
    }
}

}  // namespace

std::vector<ChatPiece> Qwen3CoderFormat::render_pieces(const std::vector<Message>& msgs, const json& tools,
                                                       bool thinking, const std::string& effort) const {
    if (msgs.empty()) throw std::runtime_error("no messages");
    PieceBuilder b;
    size_t num_sys = 0;
    const std::string merged = merge_system(msgs, num_sys);
    render_system(b, merged, effort_lines_ ? reasoning_line(thinking, effort) : "", tools);
    for (size_t i = num_sys; i < msgs.size(); i++) render_message(b, msgs, i);
    b.scaffold("<|im_start|>assistant\n");
    b.scaffold(thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
    return std::move(b.out);
}

std::unique_ptr<OutputParser> Qwen3CoderFormat::make_parser(bool thinking, const json& tools) const {
    return std::make_unique<Qwen3CoderParser>(thinking, tools);
}

// Closing text when the thinking budget is reached: the sentence from the Qwen docs (quickstart, thinking budget);
// without it only </think> (llama.cpp issue 20297: HumanEval ~89 % with the sentence, ~79 % without)
std::string Qwen3CoderFormat::think_close(bool with_message) const {
    return with_message ? "\n\nConsidering the limited time by the user, I have to give the solution based on the "
                          "thinking directly now.\n</think>\n\n"
                        : "\n</think>\n\n";
}

}  // namespace tona::server
