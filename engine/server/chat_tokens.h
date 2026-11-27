// Encode a rendered chat: scaffold with special tokens, foreign content without (server/chat.h, ChatPiece)
#pragma once

#include "model/tokenizer.h"
#include "server/chat.h"

namespace tona::server {

inline std::vector<int32_t> encode_chat(const Tokenizer& tok, const std::vector<ChatPiece>& pieces) {
    std::vector<int32_t> ids;
    for (const ChatPiece& p : pieces) {
        const auto t = tok.encode(p.text, p.content ? Tokenizer::Special::none : Tokenizer::Special::all);
        ids.insert(ids.end(), t.begin(), t.end());
    }
    return ids;
}

}  // namespace tona::server
