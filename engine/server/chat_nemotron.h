// Chat format of the Nemotron family (Nemotron 3.5 Bolt): ChatML turns with an always present system block, tools as
// XML (<tools><function><name>..), <think> blocks, and tool calls in the same XML form as the Qwen3 family. Part of
// chat_format.h; the GGUF's own template is the reference (tests/chat_vs_jinja.py, byte-equal).
#pragma once

#include "server/chat_format.h"

namespace tona::server {

class NemotronFormat final : public ChatFormat {
public:
    std::vector<ChatPiece> render_pieces(const std::vector<Message>& msgs, const json& tools, bool thinking,
                                         const std::string& effort) const override;
    // The model writes tool calls as <tool_call><function=..><parameter=..> exactly like the Qwen3 family, so the
    // parser is that one
    std::unique_ptr<OutputParser> make_parser(bool thinking, const json& tools) const override;
    std::string_view turn_start() const override { return "<|im_start|>"; }
    std::vector<std::string_view> stop_texts() const override { return {"<|im_end|>"}; }
    std::string_view think_end() const override { return "</think>"; }
    // Nemotron 2 and 3 are trained for budget control by appending </think> (arXiv 2512.20856); no trained sentence
    // like the Qwen docs' exists, so with_message makes no difference
    std::string think_close(bool with_message) const override;
};

}  // namespace tona::server
