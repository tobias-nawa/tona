// Helpers shared by both request parsers: content text, merging, effort levels, the prompt pieces of a request.
#include <string>

#include "server/api.h"
#include "tests/test_support.h"

using tona::server::json;
using tona::server::Message;
using tona::server::OutputParser;
using tona::server::Request;

// The format under test: the Qwen3 family (Coder and Ornith)
static const tona::server::ChatFormat& fmt() { return tona::server::chat_format(tona::ChatFamily::Qwen3Coder); }

namespace {

void test_text_of() {
    EXPECT(tona::server::text_of(json("plain")) == "plain");
    EXPECT(tona::server::text_of(json::parse(R"([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])")) ==
           "a\nb");
    EXPECT(tona::server::text_of(json::parse(R"([{"type": "image_url"}, {"type": "text", "text": "a"}])")) ==
           "[image omitted]\na");
    EXPECT(tona::server::text_of(json::parse(R"([{"type": "unknown", "text": "x"}, {"text": "y"}])")).empty());
    EXPECT(tona::server::text_of(json(5)).empty() && tona::server::text_of(json(nullptr)).empty());
}

void test_merge_users() {
    std::vector<Message> m{{"user", "a", "", {}},
                           {"user", "b", "", {}},
                           {"assistant", "c", "", {}},
                           {"assistant", "d", "", {}},
                           {"user", "e", "", {}}};
    tona::server::merge_users(m);
    EXPECT(m.size() == 4 && m[0].content == "a\n\nb" && m[1].content == "c" && m[2].content == "d" &&
           m[3].content == "e");
}

void test_effort_level() {
    for (const char* e : {"low", "medium", "high", "xhigh", "max"}) EXPECT(tona::server::effort_level(e) == e);
    EXPECT(tona::server::effort_level("auto") == "high");
    EXPECT(tona::server::effort_level("") == "high");
}

void test_request_pieces() {
    Request q;
    q.msgs = {{"user", "hello", "", {}}};
    q.forced = "<tool_call>\n", q.prefill = "pre";
    const auto pieces = tona::server::request_pieces(fmt(), q);
    EXPECT(pieces.size() >= 3);
    EXPECT(pieces[pieces.size() - 2].text == q.forced && !pieces[pieces.size() - 2].content);
    EXPECT(pieces.back().text == q.prefill && pieces.back().content);
    std::string all;
    for (const auto& p : pieces) all += p.text;
    EXPECT(all == tona::server::render_chat(fmt(), q.msgs, q.tools, q.thinking, q.effort) + q.forced + q.prefill);
}

// The forced start is part of the prompt: the parser must not report it again
void test_prime_hides_the_forced_start() {
    const json tools = json::parse(R"([{"type": "function", "function": {"name": "read", "parameters":
        {"type": "object", "properties": {"path": {"type": "string"}}}}}])");
    Request q;
    q.tools = tools;
    q.forced = "<tool_call>\n<function=read>\n";
    const auto parser = fmt().make_parser(false, q.tools);
    OutputParser& p = *parser;
    tona::server::prime(p, q);
    auto ev = p.feed("<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>");
    for (auto& e : p.finish()) ev.push_back(e);
    EXPECT(ev.size() == 1 && ev[0].kind == OutputParser::Event::tool_call);
    EXPECT(ev[0].call.name == "read" && ev[0].call.args["path"] == ".");

    const auto plain_parser = fmt().make_parser(false, q.tools);
    OutputParser& plain = *plain_parser;
    Request none;
    tona::server::prime(plain, none);  // nothing forced: nothing is fed
    std::string text;
    for (const auto& e : plain.feed("hi there")) text += e.delta;
    for (const auto& e : plain.finish()) text += e.delta;
    EXPECT(text == "hi there");
}

}  // namespace

int main() {
    return tona::test::run("test_api_common", [] {
        test_text_of();
        test_merge_users();
        test_effort_level();
        test_request_pieces();
        test_prime_hides_the_forced_start();
    });
}
