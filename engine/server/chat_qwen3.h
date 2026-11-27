// Chat format of the Qwen3 family (Coder, and Ornith: ChatML turns, <think>, XML tool calls). Part of chat_format.h.
#pragma once

#include "server/chat_format.h"

namespace tona::server {

class Qwen3CoderFormat final : public ChatFormat {
public:
    // effort_lines: the template puts "Reasoning effort is set to ..." into the system message when thinking is on
    explicit Qwen3CoderFormat(bool effort_lines = true) : effort_lines_(effort_lines) {}
    std::vector<ChatPiece> render_pieces(const std::vector<Message>& msgs, const json& tools, bool thinking,
                                         const std::string& effort) const override;
    std::unique_ptr<OutputParser> make_parser(bool thinking, const json& tools) const override;
    std::string_view turn_start() const override { return "<|im_start|>"; }
    std::vector<std::string_view> stop_texts() const override { return {"<|im_end|>", "<|endoftext|>"}; }
    std::string_view think_end() const override { return "</think>"; }
    std::string think_close(bool with_message) const override;

private:
    bool effort_lines_;
};

// Output of the Qwen3 family: <think> block, text, <tool_call> blocks in the XML form (and the older JSON form)
class Qwen3CoderParser final : public OutputParser {
public:
    Qwen3CoderParser(bool thinking, const json& tools);
    std::vector<Event> feed(const std::string& piece) override;
    std::vector<Event> finish() override;
    bool truncated() const override { return truncated_; }

private:
    enum State { think, text, call } st_;
    bool at_start_ = true;  // drop whitespace at the start of a section
    bool truncated_ = false;
    std::string buf_;
    const json& tools_;
    void flush_text(std::vector<Event>& ev, bool all);
    ToolCall parse_call(const std::string& body) const;
};

}  // namespace tona::server
