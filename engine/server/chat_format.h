// What the server needs from the chat format of a model family: render the conversation, parse the output, and the
// texts of the special tokens the engine looks up in the vocabulary. One implementation per ChatFamily
// (model/arch.h), chosen once at load from ModelBase::chat_family(); shared server code never tests the family itself.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "model/arch.h"
#include "server/chat.h"

namespace tona::server {

class ChatFormat {
public:
    virtual ~ChatFormat() = default;

    // tools: list in OpenAI format [{"type": "function", "function": {name, description, parameters}}]
    // thinking: on (the template's default; effort is low, medium, high, xhigh or max, as far as the template knows
    // levels) or off
    virtual std::vector<ChatPiece> render_pieces(const std::vector<Message>& msgs, const json& tools, bool thinking,
                                                 const std::string& effort) const = 0;
    // Parser for one answer; it keeps a reference to tools, which must outlive it
    virtual std::unique_ptr<OutputParser> make_parser(bool thinking, const json& tools) const = 0;

    // Token texts of the format (looked up in the vocabulary once, at load)
    virtual std::string_view turn_start() const = 0;  // opens a message; the checkpoint seams sit before it
    virtual std::vector<std::string_view> stop_texts() const = 0;  // end of the model's turn
    virtual std::string_view think_end() const = 0;                // closes the thinking block
    // Text that ends thinking when the budget is reached; with_message: the sentence that asks the model to answer now
    virtual std::string think_close(bool with_message) const = 0;
};

// The format of a family; throws if the build has none for it yet
const ChatFormat& chat_format(ChatFamily family);

// The pieces of the format concatenated
std::string render_chat(const ChatFormat& fmt, const std::vector<Message>& msgs, const json& tools, bool thinking,
                        const std::string& effort = "xhigh");

}  // namespace tona::server
