// OpenAI /v1/chat/completions requests -> Request: valid bodies and the errors a malformed one must raise.
#include <climits>
#include <string>

#include "server/api.h"
#include "tests/test_support.h"

using tona::server::from_openai;
using tona::server::json;
using tona::server::Request;
using tona::test::throws;

namespace {

json J(const char* text) { return json::parse(text); }

// A request body with one user message plus the given extra members ("\"stream\": true, ...")
Request with(const std::string& extra, bool think_default = false) {
    return from_openai(json::parse(R"({"messages": [{"role": "user", "content": "x"}],)" + extra + "}"), think_default);
}

void test_plain_conversation() {
    const Request q = from_openai(J(R"({"messages": [{"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Bye"}]})"),
                                  false);
    EXPECT(q.msgs.size() == 4 && q.msgs[0].role == "system" && q.msgs[0].content == "Be brief.");
    EXPECT(q.msgs[1].content == "Hi" && q.msgs[2].role == "assistant" && q.msgs[3].content == "Bye");
    // Without a limit: up to the end of the context
    EXPECT(q.max_tokens == INT_MAX && !q.thinking && q.effort == "high");
}

void test_roles() {
    const Request q = from_openai(J(R"({"messages": [{"role": "developer", "content": "rules"},
        {"role": "user", "content": "a"}, {"role": "system", "content": "late"}, {"role": "user", "content": "b"}]})"),
                                  false);
    EXPECT(q.msgs.size() == 2 && q.msgs[0].role == "system" && q.msgs[0].content == "rules");
    // A system message after the first turn would be its own template turn: it becomes user text and merges
    EXPECT(q.msgs[1].role == "user" && q.msgs[1].content == "a\n\nlate\n\nb");
}

void test_system_message_after_a_tool_result_joins_it() {
    const Request q = from_openai(J(R"({"messages": [{"role": "user", "content": "go"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a b"}, {"role": "system", "content": "r1"},
        {"role": "developer", "content": "r2"}]})"),
                                  false);
    EXPECT(q.msgs.size() == 3 && q.msgs[2].role == "tool" && q.msgs[2].content == "a b\n\nr1\n\nr2");
}

void test_content_parts_and_null_content() {
    const Request q = from_openai(J(R"({"messages": [{"role": "user", "content": [{"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:..."}}]},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "ls", "arguments": "{\"path\": \"/\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a b"}]})"),
                                  false);
    EXPECT(q.msgs.size() == 3);
    EXPECT(q.msgs[0].content == "look\n[image omitted]");
    EXPECT(q.msgs[1].content.empty() && q.msgs[1].calls.size() == 1);
    EXPECT(q.msgs[1].calls[0].id == "c1" && q.msgs[1].calls[0].name == "ls" && q.msgs[1].calls[0].args["path"] == "/");
    EXPECT(q.msgs[2].role == "tool" && q.msgs[2].content == "a b");
}

void test_tool_call_arguments() {
    auto call = [](const char* arguments) {
        const std::string body = R"({"messages": [{"role": "user", "content": "x"}, {"role": "assistant",
            "tool_calls": [{"function": {"name": "f", )" +
                                 std::string(arguments) + R"(}}]}, {"role": "tool", "content": "r"}]})";
        return from_openai(json::parse(body), false).msgs[1].calls.at(0);
    };
    EXPECT(call(R"("arguments": {"k": 1})").args["k"] == 1);  // an object instead of a string is accepted
    EXPECT(call(R"("arguments": "{not json")").args.is_object() && call(R"("arguments": "{not json")").args.empty());
    EXPECT(call(R"("x": 0)").args.is_object());  // no arguments at all
    EXPECT(
        throws([] { from_openai(J(R"({"messages": [{"role": "assistant", "tool_calls": [{"id": "c"}]}]})"), false); }));
}

void test_reasoning_is_kept() {
    const Request q = from_openai(J(R"({"messages": [{"role": "user", "content": "x"},
        {"role": "assistant", "content": "y", "reasoning_content": "because"},
        {"role": "user", "content": "z"}, {"role": "assistant", "content": "w", "reasoning": "since"},
        {"role": "user", "content": "v"}]})"),
                                  false);
    EXPECT(q.msgs[1].reasoning == "because" && q.msgs[3].reasoning == "since");
}

// max_tokens 0 or max_completion_tokens 0 is the warm-up request; a negative value, a stream and a forced start are
// errors
void test_warm_up_request() {
    EXPECT(with(R"("max_tokens": 0)").warm && with(R"("max_completion_tokens": 0)").warm);
    EXPECT(with(R"("max_tokens": 5, "max_completion_tokens": 0)").warm);
    EXPECT(!with(R"("max_tokens": 0, "max_completion_tokens": 5)").warm && !with(R"("n": 1)").warm);
    EXPECT(throws([] { with(R"("max_tokens": 0, "stream": true)"); }));
    EXPECT(throws([] { with(R"("max_tokens": -1)"); }));
    EXPECT(throws([] { with(R"("max_completion_tokens": -5)"); }));
    EXPECT(throws([] {
        with(
            R"("max_tokens": 0, "tools": [{"type": "function", "function": {"name": "f"}}], "tool_choice": "required")");
    }));
}

void test_limits_and_sampling() {
    EXPECT(with(R"("max_tokens": 50)").max_tokens == 50);
    EXPECT(with(R"("max_tokens": 50, "max_completion_tokens": 70)").max_tokens == 70);
    EXPECT(with(R"("max_tokens": 1e91)").max_tokens == INT_MAX);
    const Request q = with(R"("temperature": 0, "top_p": 0.9, "top_k": 5, "seed": 7, "stream": true)");
    EXPECT(q.temperature == 0 && q.top_p == 0.9f && q.top_k == 5 && q.seed == 7 && q.stream);
    const Request n = with(R"("temperature": null, "top_p": null, "seed": null)");
    EXPECT(n.temperature == -1 && n.top_p == 1 && n.seed == 0);
    EXPECT((with(R"("stop": "END")").stops == std::vector<std::string>{"END"}));
    EXPECT((with(R"("stop": ["a", "b"])").stops == std::vector<std::string>{"a", "b"}));
    EXPECT(with(R"("stop": 5)").stops.empty());
}

void test_thinking_settings() {
    EXPECT(with("\"stream\": false", true).thinking);
    EXPECT(!with(R"("reasoning_effort": "none")", true).thinking);
    EXPECT(!with(R"("reasoning_effort": "minimal")", true).thinking);
    const Request low = with(R"("reasoning_effort": "low")", false);
    EXPECT(low.thinking && low.effort == "low");
    EXPECT(with(R"("reasoning_effort": "xhigh")", false).effort == "xhigh");
    EXPECT(with(R"("reasoning_effort": "auto")", false).effort == "high");  // unknown levels give high
    EXPECT(with(R"("reasoning_effort": 3)", true).thinking);                // not a string: ignored
    EXPECT(!with(R"("chat_template_kwargs": {"enable_thinking": false})", true).thinking);
    EXPECT(with(R"("chat_template_kwargs": {"enable_thinking": true})", false).thinking);
}

void test_tool_choice() {
    const std::string tools = R"("tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],)";
    EXPECT(with(tools + R"("tool_choice": "auto")", true).forced.empty());
    const Request req = with(tools + R"("tool_choice": "required")", true);
    EXPECT(req.forced == "<tool_call>\n<function=" && !req.thinking && req.tools.size() == 1);
    const Request named = with(tools + R"("tool_choice": {"type": "function", "function": {"name": "read"}})", true);
    EXPECT(named.forced == "<tool_call>\n<function=read>\n");
    EXPECT(with(tools + R"("tool_choice": "none")", true).tools.empty());
    EXPECT(throws([&] { with(tools + R"("tool_choice": {"type": "function", "function": {"name": "nope"}})"); }));
    EXPECT(with(R"("tool_choice": "required")", true).forced.empty());  // no tools: nothing to force
}

void test_prefill_from_last_assistant_message() {
    const Request q = from_openai(J(R"({"messages": [{"role": "user", "content": "x"},
        {"role": "assistant", "content": "Sure, "}]})"),
                                  true);
    EXPECT(q.prefill == "Sure," && q.msgs.size() == 1 && !q.thinking);
}

void test_malformed_requests_throw() {
    EXPECT(throws([] { from_openai(J(R"({})"), false); }));
    EXPECT(throws([] { from_openai(J(R"({"messages": [{"content": "x"}]})"), false); }));
    EXPECT(throws([] { from_openai(J(R"({"messages": [{"role": null}]})"), false); }));
    EXPECT(throws([] { from_openai(J(R"("just a string")"), false); }));
    EXPECT(throws([] { with(R"("max_tokens": "many")"); }));
    EXPECT(throws([] { with(R"("stream": 3)"); }));
    EXPECT(throws([] { with(R"("chat_template_kwargs": {"enable_thinking": "no"})"); }));
}

}  // namespace

int main() {
    return tona::test::run("test_api_openai", [] {
        test_plain_conversation();
        test_roles();
        test_system_message_after_a_tool_result_joins_it();
        test_content_parts_and_null_content();
        test_tool_call_arguments();
        test_reasoning_is_kept();
        test_limits_and_sampling();
        test_warm_up_request();
        test_thinking_settings();
        test_tool_choice();
        test_prefill_from_last_assistant_message();
        test_malformed_requests_throw();
    });
}
