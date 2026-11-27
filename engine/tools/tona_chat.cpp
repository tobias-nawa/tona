// Chat template without a model: cases (JSON file, messages in OpenAI format) -> rendered prompts as a JSON list.
// For tests/chat_vs_jinja.py. Usage: tona-chat CASES.json
// Render a recorded request: tona-chat --anthropic|--openai REQUEST.json [MODEL.gguf] -> prompt on stdout, with a
// model the token count on stderr.
// tona-chat --tokens CASES.json MODEL.gguf: encoded in pieces (as the server does) equals encoded whole as long as
// the content has no special tokens; with special tokens in the content the role boundaries stay those of the
// scaffold.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "model/gguf.h"
#include "model/model_registry.h"
#include "model/tokenizer.h"
#include "server/api.h"
#include "server/chat.h"
#include "server/chat_format.h"
#include "server/chat_tokens.h"

using tona::server::json;

namespace {

// This tool is the gate of the formats (tests/chat_vs_jinja.py); --model FILE picks the family of that model, else the
// Coder's
tona::ChatFamily g_family = tona::ChatFamily::Qwen3Coder;
const tona::server::ChatFormat& fmt() { return tona::server::chat_format(g_family); }

std::vector<tona::server::Message> messages_of(const json& c) {
    std::vector<tona::server::Message> msgs;
    for (const auto& m : c["messages"]) {
        tona::server::Message x;
        x.role = m["role"];
        if (m.contains("content") && m["content"].is_string()) x.content = m["content"];
        if (m.contains("reasoning_content")) x.reasoning = m["reasoning_content"];
        if (m.contains("tool_calls"))
            for (const auto& t : m["tool_calls"])
                x.calls.push_back({"", t["function"]["name"], t["function"]["arguments"]});
        msgs.push_back(std::move(x));
    }
    return msgs;
}

int render_request(const std::string& api, const char* request_path, const char* model_path) {
    std::ifstream f(request_path);
    const json r = json::parse(f);
    const auto q = api == "--anthropic" ? tona::server::from_anthropic(r, false) : tona::server::from_openai(r, false);
    const std::string p = tona::server::render_chat(fmt(), q.msgs, q.tools, q.thinking, q.effort);
    std::cout << p;
    std::fprintf(stderr, "%zu messages, %zu tools, thinking %s (%s), %zu characters\n", q.msgs.size(), q.tools.size(),
                 q.thinking ? "on" : "off", q.effort.c_str(), p.size());
    if (model_path) {
        tona::Gguf g(model_path);
        tona::Tokenizer tok(g);
        std::fprintf(stderr, "%zu tokens\n", tok.encode(p).size());
    }
    return 0;
}

bool has_special_in_content(const tona::Tokenizer& tok, const std::vector<tona::server::ChatPiece>& pieces) {
    using Special = tona::Tokenizer::Special;
    for (const auto& p : pieces)
        if (p.content && tok.encode(p.text, Special::all) != tok.encode(p.text, Special::none)) return true;
    return false;
}

// Number of <|im_start|> and <|im_end|> the scaffold itself sets
std::pair<long, long> scaffold_boundaries(const tona::Tokenizer& tok,
                                          const std::vector<tona::server::ChatPiece>& pieces, int32_t im_start,
                                          int32_t im_end) {
    long starts = 0, ends = 0;
    for (const auto& p : pieces)
        if (!p.content) {
            const auto t = tok.encode(p.text, tona::Tokenizer::Special::all);
            starts += std::count(t.begin(), t.end(), im_start), ends += std::count(t.begin(), t.end(), im_end);
        }
    return {starts, ends};
}

// Number of deviations found in one case
int check_case(const tona::Tokenizer& tok, const json& c, int number) {
    const auto msgs = messages_of(c);
    const json tools = c.value("tools", json::array());
    const bool thinking = c.value("thinking", true);
    const std::string effort = c.value("effort", "xhigh");
    const auto pieces = fmt().render_pieces(msgs, tools, thinking, effort);
    const auto split = tona::server::encode_chat(tok, pieces);
    const auto whole = tok.encode(tona::server::render_chat(fmt(), msgs, tools, thinking, effort));
    int bad = 0;
    if (!has_special_in_content(tok, pieces) && split != whole) {
        std::fprintf(stderr, "Case %d: in pieces %zu tokens, whole %zu\n", number, split.size(), whole.size());
        bad++;
    }
    const int32_t im_start = tok.find(std::string(fmt().turn_start())),
                  im_end = tok.find(std::string(fmt().stop_texts()[0]));
    const auto [starts, ends] = scaffold_boundaries(tok, pieces, im_start, im_end);
    if (std::count(split.begin(), split.end(), im_start) != starts ||
        std::count(split.begin(), split.end(), im_end) != ends) {
        std::fprintf(stderr, "Case %d: role boundaries in content\n", number);
        bad++;
    }
    return bad;
}

int check_tokens(const char* cases_path, const char* model_path) {
    std::ifstream f(cases_path);
    const json cases = json::parse(f);
    tona::Gguf g(model_path);
    tona::Tokenizer tok(g);
    int bad = 0, n = 0;
    for (const auto& c : cases) bad += check_case(tok, c, ++n);
    std::printf("%d cases, %d deviations\n", n, bad);
    return bad ? 1 : 0;
}

int render_cases(const char* cases_path) {
    std::ifstream f(cases_path);
    const json cases = json::parse(f);
    json out = json::array();
    for (const auto& c : cases)
        out.push_back(tona::server::render_chat(fmt(), messages_of(c), c.value("tools", json::array()),
                                                c.value("thinking", true), c.value("effort", "xhigh")));
    std::cout << out.dump() << "\n";
    return 0;
}

int dispatch(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        g_family = tona::load_model(argv[2])->chat_family();
        argc -= 2, argv += 2;
    }
    const std::string first = argc >= 2 ? argv[1] : "";
    if (argc >= 3 && (first == "--anthropic" || first == "--openai"))
        return render_request(first, argv[2], argc >= 4 ? argv[3] : nullptr);
    if (argc == 4 && first == "--tokens") return check_tokens(argv[2], argv[3]);
    if (argc != 2) {
        std::fprintf(stderr,
                     "Usage: [--model MODEL.gguf] tona-chat CASES.json | tona-chat --anthropic|--openai REQUEST.json "
                     "[MODEL.gguf]\n");
        return 2;
    }
    return render_cases(argv[1]);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return dispatch(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
