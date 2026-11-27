// Types of the chat layer: messages, rendered pieces and the parser of the model's output. The format of a model family
// (the GGUF's tokenizer.chat_template, reimplemented in C++ instead of running Jinja) is a ChatFormat (chat_format.h);
// tests/chat_vs_jinja.py compares each against its template via jinja2.
#pragma once

#include <string>
#include <vector>

#include "third_party/nlohmann/json.hpp"

namespace tona::server {

using json = nlohmann::ordered_json;

struct ToolCall {
    std::string id, name;
    json args = json::object();
};

struct Message {
    std::string role;  // system, user, assistant, tool
    std::string content;
    std::string reasoning;  // assistant only
    std::vector<ToolCall> calls;
};

// Piece of the rendered chat: template scaffold (special tokens apply) or foreign content (system prompt, user text,
// tool results, tool descriptions). Content is encoded without special tokens so that a "<|im_end|>" in a file the
// model read does not end a role. The pieces concatenated are render_chat.
struct ChatPiece {
    std::string text;
    bool content;
};
// json like Jinja's tojson (Python json.dumps: ", " and ": ", non-ASCII unchanged)
std::string to_pyjson(const json& j);

// Splits the model output piece by piece into thinking, text and tool calls. Parts of tags and trailing whitespace
// are held back until it is clear where they belong. One implementation per chat family (chat_format.h).
class OutputParser {
public:
    struct Event {
        enum Kind { thinking, text, tool_call } kind;
        std::string delta;
        ToolCall call;
    };
    virtual ~OutputParser() = default;
    virtual std::vector<Event> feed(const std::string& piece) = 0;
    virtual std::vector<Event> finish() = 0;
    virtual bool truncated() const = 0;  // after finish: a truncated tool call was dropped
};

}  // namespace tona::server
