// Anthropic /v1/messages requests -> Request: valid bodies and the errors a malformed one must raise.
#include <climits>
#include <stdexcept>
#include <string>

#include "server/api.h"
#include "tests/test_support.h"

using tona::server::from_anthropic;
using tona::server::json;
using tona::server::Request;
using tona::test::throws;

namespace {

json J(const char* text) { return json::parse(text); }

void test_plain_conversation() {
    const Request q = from_anthropic(J(R"({"system": "Be brief.", "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
                     {"role": "user", "content": "Bye"}]})"),
                                     false);
    EXPECT(q.msgs.size() == 4);
    EXPECT(q.msgs[0].role == "system" && q.msgs[0].content == "Be brief.");
    EXPECT(q.msgs[1].role == "user" && q.msgs[1].content == "Hi");
    EXPECT(q.msgs[2].role == "assistant" && q.msgs[2].content == "Hello");
    EXPECT(q.max_tokens == 100 && !q.thinking && !q.stream);
    EXPECT(q.temperature == -1 && q.top_p == 1 && q.top_k == 0 && q.seed == 0);
    EXPECT(q.prefill.empty() && q.forced.empty() && q.tools.empty());
}

void test_defaults_and_sampling_fields() {
    const Request d = from_anthropic(J(R"({"messages": [{"role": "user", "content": "x"}]})"), true);
    EXPECT(d.max_tokens == 4096 && d.thinking && d.effort == "high" && d.think_budget == -1);
    const Request q = from_anthropic(J(R"({"messages": [{"role": "user", "content": "x"}], "temperature": 0.25,
        "top_p": 0.5, "top_k": 7, "seed": 42, "stream": true, "stop_sequences": ["END", "STOP"]})"),
                                     false);
    EXPECT(q.temperature == 0.25f && q.top_p == 0.5f && q.top_k == 7 && q.seed == 42 && q.stream);
    EXPECT((q.stops == std::vector<std::string>{"END", "STOP"}));
    // Some proxies send null instead of leaving a field out
    const Request n = from_anthropic(J(R"({"messages": [{"role": "user", "content": "x"}], "max_tokens": null,
        "temperature": null, "stream": null})"),
                                     false);
    EXPECT(n.max_tokens == 4096 && n.temperature == -1 && !n.stream);
}

void test_numbers_are_clamped() {
    const Request big =
        from_anthropic(J(R"({"messages": [{"role": "user", "content": "x"}], "max_tokens": 1e91})"), false);
    EXPECT(big.max_tokens == INT_MAX);
    const Request frac =
        from_anthropic(J(R"({"messages": [{"role": "user", "content": "x"}], "max_tokens": 12.9})"), false);
    EXPECT(frac.max_tokens == 12);
}

// max_tokens 0 is the warm-up request; a negative value, a stream and a forced start are errors
void test_warm_up_request() {
    auto with = [](const char* extra) {
        return from_anthropic(
            json::parse(std::string(R"({"messages": [{"role": "user", "content": "x"}],)") + extra + "}"), false);
    };
    EXPECT(with(R"("max_tokens": 0)").warm && with(R"("max_tokens": 0, "stream": false)").warm);
    EXPECT(!with(R"("max_tokens": 1)").warm && !with(R"("max_tokens": null)").warm);
    EXPECT(throws([&] { with(R"("max_tokens": 0, "stream": true)"); }));
    EXPECT(throws([&] { with(R"("max_tokens": -1)"); }));
    EXPECT(throws([&] { with(R"("max_tokens": -1e91)"); }));
    EXPECT(throws([&] {
        with(R"("max_tokens": 0, "tools": [{"name": "f", "input_schema": {}}], "tool_choice": {"type": "any"})");
    }));
    EXPECT(throws([&] {
        from_anthropic(J(R"({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "Sure"}],
            "max_tokens": 0})"),
                       false);
    }));
}

void test_content_blocks() {
    const Request q = from_anthropic(J(R"({"system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "image", "source": {}},
                                                 {"type": "text", "text": "two"}]}]})"),
                                     false);
    EXPECT(q.msgs.size() == 2);
    EXPECT(q.msgs[0].content == "A\nB");
    EXPECT(q.msgs[1].content == "one\n[attachment omitted]\ntwo");
}

void test_tool_use_and_result() {
    const Request q = from_anthropic(J(R"({"messages": [
        {"role": "user", "content": "list"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "t1", "name": "ls", "input": {"path": "."}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.txt"},
                                    {"type": "text", "text": "thanks"}]},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "is_error": true,
                                      "content": [{"type": "text", "text": "denied"}]}]}]})"),
                                     false);
    EXPECT(q.msgs.size() == 6);
    const auto& a = q.msgs[1];
    EXPECT(a.role == "assistant" && a.reasoning == "hmm" && a.content == "ok" && a.calls.size() == 1);
    EXPECT(a.calls[0].id == "t1" && a.calls[0].name == "ls" && a.calls[0].args["path"] == ".");
    EXPECT(q.msgs[2].role == "tool" && q.msgs[2].content == "a.txt");
    EXPECT(q.msgs[3].role == "user" && q.msgs[3].content == "thanks");
    EXPECT(q.msgs[5].role == "tool" && q.msgs[5].content == "Error: denied");
}

void test_user_messages_merge_and_system_in_the_middle() {
    const Request q = from_anthropic(J(R"({"messages": [{"role": "user", "content": "a"},
        {"role": "user", "content": "b"}, {"role": "system", "content": "c"}, {"role": "assistant", "content": "d"}]})"),
                                     false);
    // The assistant message is the prefill, so what is left is one merged user message
    EXPECT(q.msgs.size() == 1 && q.msgs[0].role == "user" && q.msgs[0].content == "a\n\nb\n\nc");
    EXPECT(q.prefill == "d");
}

// Claude Code sends a system message after every tool result: it belongs into the tool response, not into a user turn
// of its own (research/measurements/2026-11-10-fp8-stop.md, Result 8)
void test_system_message_joins_the_message_before_it() {
    const Request q = from_anthropic(J(R"({"system": "S", "messages": [{"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ls", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.txt"}]},
        {"role": "system", "content": "r1"}, {"role": "system", "content": [{"type": "text", "text": "r2"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "ls", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "b"}]},
        {"role": "system", "content": "r3"}]})"),
                                     false);
    EXPECT(q.msgs.size() == 6);  // system, user, assistant, tool, assistant, tool
    EXPECT(q.msgs[3].role == "tool" && q.msgs[3].content == "a.txt\n\nr1\n\nr2");
    EXPECT(q.msgs[5].role == "tool" && q.msgs[5].content == "b\n\nr3");
    // The same prefix gives the same messages whatever follows (the prompt cache depends on it)
    const Request p = from_anthropic(J(R"({"system": "S", "messages": [{"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ls", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.txt"}]},
        {"role": "system", "content": "r1"}, {"role": "system", "content": [{"type": "text", "text": "r2"}]}]})"),
                                     false);
    EXPECT(p.msgs.size() == 4 && p.msgs[3].content == q.msgs[3].content);
}

void test_system_message_at_the_start_and_after_an_assistant_message() {
    const Request q = from_anthropic(J(R"({"messages": [{"role": "system", "content": "first"},
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "system", "content": "late"},
        {"role": "user", "content": "c"}]})"),
                                     false);
    EXPECT(q.msgs.size() == 4);
    EXPECT(q.msgs[0].role == "system" && q.msgs[0].content == "first");
    // Nothing to append to: it stays a user turn, which merges with the next one
    EXPECT(q.msgs[3].role == "user" && q.msgs[3].content == "late\n\nc");
}

void test_tools_need_a_schema() {
    const Request q = from_anthropic(J(R"({"messages": [{"role": "user", "content": "x"}], "tools": [
        {"name": "read", "description": "Read a file", "input_schema": {"type": "object"}},
        {"type": "web_search_20250305", "name": "web_search"}]})"),
                                     false);
    EXPECT(q.tools.size() == 1);
    EXPECT(q.tools[0]["type"] == "function" && q.tools[0]["function"]["name"] == "read");
    EXPECT(q.tools[0]["function"]["description"] == "Read a file");
    EXPECT(q.tools[0]["function"]["parameters"]["type"] == "object");
}

void test_thinking_settings() {
    const char* base = R"({"messages": [{"role": "user", "content": "x"}], "thinking": %s})";
    auto make = [&](const char* thinking, bool dflt) {
        std::string body = base;
        body.replace(body.find("%s"), 2, thinking);
        return from_anthropic(json::parse(body), dflt);
    };
    EXPECT(!make(R"({"type": "disabled"})", true).thinking);
    EXPECT(make(R"({"type": "adaptive"})", false).thinking);
    const Request low = make(R"({"type": "enabled", "budget_tokens": 2000})", false);
    EXPECT(low.thinking && low.think_budget == 2000 && low.effort == "low");
    EXPECT(make(R"({"type": "enabled", "budget_tokens": 8000})", false).effort == "medium");
    EXPECT(make(R"({"type": "enabled", "budget_tokens": 20000})", false).effort == "high");
    EXPECT(make(R"({"type": "enabled", "budget_tokens": 0})", false).think_budget == 1);
    EXPECT(make(R"({"type": "enabled"})", false).think_budget == -1);
    const Request o = from_anthropic(
        J(R"({"messages": [{"role": "user", "content": "x"}], "output_config": {"effort": "max"}})"), false);
    EXPECT(o.effort == "max");
    const Request u = from_anthropic(
        J(R"({"messages": [{"role": "user", "content": "x"}], "output_config": {"effort": "auto"}})"), false);
    EXPECT(u.effort == "high");
}

void test_tool_choice_and_prefill() {
    const char* msgs = R"("messages": [{"role": "user", "content": "x"}])";
    const std::string tool = R"("tools": [{"name": "read", "input_schema": {}}])";
    auto make = [&](const std::string& choice) {
        return from_anthropic(json::parse("{" + std::string(msgs) + "," + tool + "," + choice + "}"), true);
    };
    const Request any = make(R"("tool_choice": {"type": "any"})");
    EXPECT(any.forced == "<tool_call>\n<function=" && !any.thinking);
    const Request named = make(R"("tool_choice": {"type": "tool", "name": "read"})");
    EXPECT(named.forced == "<tool_call>\n<function=read>\n" && !named.thinking);
    EXPECT(make(R"("tool_choice": {"type": "none"})").tools.empty());
    const Request autoc = make(R"("tool_choice": {"type": "auto"})");
    EXPECT(autoc.forced.empty() && autoc.thinking && autoc.tools.size() == 1);
    EXPECT(throws([&] { make(R"("tool_choice": {"type": "tool", "name": "write"})"); }));

    const Request pre = from_anthropic(J(R"({"messages": [{"role": "user", "content": "x"},
        {"role": "assistant", "content": "The answer is  \n"}]})"),
                                       true);
    EXPECT(pre.prefill == "The answer is" && !pre.thinking && pre.msgs.size() == 1);
}

void test_malformed_requests_throw() {
    EXPECT(throws([] { from_anthropic(J(R"({})"), false); }));
    EXPECT(throws([] { from_anthropic(J(R"({"messages": [{"content": "x"}]})"), false); }));
    EXPECT(throws([] { from_anthropic(J(R"({"messages": [{"role": "user"}]})"), false); }));
    EXPECT(throws([] { from_anthropic(J(R"({"messages": 5})"), false); }));
    EXPECT(throws([] { from_anthropic(J(R"([1, 2])"), false); }));
    EXPECT(throws([] { from_anthropic(J(R"({"messages": [{"role": 3, "content": "x"}]})"), false); }));
    EXPECT(throws(
        [] { from_anthropic(J(R"({"messages": [{"role": "user", "content": "x"}], "max_tokens": "many"})"), false); }));
    EXPECT(throws(
        [] { from_anthropic(J(R"({"messages": [{"role": "user", "content": "x"}], "stream": "yes"})"), false); }));
}

}  // namespace

int main() {
    return tona::test::run("test_api_anthropic", [] {
        test_plain_conversation();
        test_defaults_and_sampling_fields();
        test_numbers_are_clamped();
        test_warm_up_request();
        test_content_blocks();
        test_tool_use_and_result();
        test_user_messages_merge_and_system_in_the_middle();
        test_system_message_joins_the_message_before_it();
        test_system_message_at_the_start_and_after_an_assistant_message();
        test_tools_need_a_schema();
        test_thinking_settings();
        test_tool_choice_and_prefill();
        test_malformed_requests_throw();
    });
}
