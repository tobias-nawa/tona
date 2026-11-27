#include "server/chat_format.h"

#include <stdexcept>

#include "server/chat_nemotron.h"
#include "server/chat_qwen3.h"

namespace tona::server {

const ChatFormat& chat_format(ChatFamily family) {
    static const Qwen3CoderFormat qwen3;
    static const NemotronFormat nemotron;
    static const Qwen3CoderFormat qwen35(false);
    switch (family) {
        case ChatFamily::Qwen3Coder:
            return qwen3;
        case ChatFamily::Qwen35:
            return qwen35;
        case ChatFamily::Nemotron:
            return nemotron;
    }
    throw std::logic_error("unhandled ChatFamily");
}

std::string render_chat(const ChatFormat& fmt, const std::vector<Message>& msgs, const json& tools, bool thinking,
                        const std::string& effort) {
    std::string s;
    for (const ChatPiece& p : fmt.render_pieces(msgs, tools, thinking, effort)) s += p.text;
    return s;
}

}  // namespace tona::server
