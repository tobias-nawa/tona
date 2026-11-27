// Requests of the Anthropic and OpenAI APIs -> messages in the chat template's format (server/chat)
#pragma once

#include <string>
#include <vector>

#include "server/chat_format.h"

namespace tona::server {

struct Request {
    std::vector<Message> msgs;
    json tools = json::array();
    bool thinking = false;
    // Effort as in Anthropic: low, medium, high, xhigh, max (OpenAI: reasoning_effort). It acts through the line in
    // the template (chat_qwen3_render.cpp) and the thinking budget per level (tona-server, TONA_THINK_BUDGET)
    std::string effort = "high";
    int think_budget = -1;  // explicit: thinking {type: enabled, budget_tokens} (-1: from the effort level)
    int max_tokens = 4096;
    // max_tokens 0: warm-up. Compute the prompt and store its checkpoints, sample nothing (set by check_max_tokens)
    bool warm = false;
    float temperature = -1, top_p = 1;  // -1: not given (server default TONA_TEMPERATURE)
    int top_k = 0;
    uint64_t seed = 0;  // 0: random
    std::vector<std::string> stops;
    bool stream = false;
    // Start of the answer that is already given: forced from tool_choice (scaffold, "<tool_call>\n<function=..."),
    // prefill from a last assistant message (content). The answer contains neither; the parser is told (prime).
    std::string forced, prefill;
};

// Prompt of the request in pieces: chat template, then forced and prefill
std::vector<ChatPiece> request_pieces(const ChatFormat& fmt, const Request& q);
// Feed the given start to the parser (its events are not part of the answer)
void prime(OutputParser& p, const Request& q);

// Text of a content field: string or list of blocks ({type: text, text}); images are omitted
std::string text_of(const json& c);
// Merge consecutive user messages into one
void merge_users(std::vector<Message>& msgs);
// Effort level of a request value; unknown values give "high"
std::string effort_level(const std::string& e);
// think_default: whether to think when the request does not say
Request from_anthropic(const json& r, bool think_default);
Request from_openai(const json& r, bool think_default);

}  // namespace tona::server
