// libFuzzer: requests of both APIs → Request → rendered chat, and model output → OutputParser.
// The first byte picks the path. Errors as std::exception are allowed (the server answers 400), anything else
// (crash, ASan, UBSan, other exceptions) is a finding. Build and run: scripts/fuzz.sh
#include <cstdint>
#include <exception>
#include <string>

#include "server/api.h"
#include "server/chat.h"
#include "server/chat_format.h"

using tona::server::json;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 1) return 0;
    const tona::server::ChatFormat& fmt = tona::server::chat_format(tona::ChatFamily::Qwen3Coder);
    const int mode = data[0] % 4;
    const std::string s(reinterpret_cast<const char*>(data + 1), size - 1);
    if (mode < 2) {
        json j;
        try {
            j = json::parse(s);
        } catch (const std::exception&) {
            return 0;
        }
        try {
            const auto q = mode == 0 ? tona::server::from_anthropic(j, true) : tona::server::from_openai(j, false);
            const auto pieces = fmt.render_pieces(q.msgs, q.tools, q.thinking, q.effort);
            std::string a;
            for (const auto& p : pieces) a += p.text;
            if (a != tona::server::render_chat(fmt, q.msgs, q.tools, q.thinking, q.effort)) __builtin_trap();
            (void)tona::server::request_pieces(fmt, q);
            const auto p = fmt.make_parser(q.thinking, q.tools);
            tona::server::prime(*p, q);
        } catch (const std::exception&) {
            return 0;  // allowed: the server answers 400
        }
        return 0;
    }
    // Model output in pieces (streaming): step size from the second byte, tools from a fixed schema
    static const json tools = json::parse(R"([{"type": "function", "function": {"name": "f", "parameters":
        {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}, "c": {"type": ["array"]}}}}}])");
    const size_t step = s.empty() ? 1 : 1 + (uint8_t)s[0] % 17;
    const auto p = fmt.make_parser(mode == 2, tools);
    try {
        for (size_t i = 0; i < s.size(); i += step) (void)p->feed(s.substr(i, step));
        (void)p->finish();
    } catch (const std::exception&) {
        __builtin_trap();  // the parser must not throw: the server catches nothing there
    }
    return 0;
}
