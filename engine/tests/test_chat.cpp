// OutputParser: output fed character by character (as when streaming) and in one piece must give the same result
#include <cstdio>
#include <string>

#include "server/api.h"
#include "server/chat.h"
#include "server/chat_format.h"

using tona::server::json;
using tona::server::OutputParser;

// The format under test: the Qwen3 family (Coder and Ornith)
static const tona::server::ChatFormat& fmt() { return tona::server::chat_format(tona::ChatFamily::Qwen3Coder); }
static auto make_parser(bool thinking, const json& tools) { return fmt().make_parser(thinking, tools); }

static int g_fail = 0;

struct Result {
    std::string thinking, text;
    std::vector<tona::server::ToolCall> calls;
};

static Result run(const std::string& out, bool thinking, const json& tools, size_t step) {
    const auto parser = make_parser(thinking, tools);
    OutputParser& p = *parser;
    Result r;
    auto take = [&](const std::vector<OutputParser::Event>& ev) {
        for (const auto& e : ev) {
            if (e.kind == OutputParser::Event::thinking)
                r.thinking += e.delta;
            else if (e.kind == OutputParser::Event::text)
                r.text += e.delta;
            else
                r.calls.push_back(e.call);
        }
    };
    for (size_t i = 0; i < out.size(); i += step) take(p.feed(out.substr(i, step)));
    take(p.finish());
    return r;
}

static void expect(const char* what, bool ok) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail++;
    }
}

static size_t count_calls(const std::vector<OutputParser::Event>& ev) {
    size_t n = 0;
    for (const auto& e : ev) n += e.kind == OutputParser::Event::tool_call;
    return n;
}

static void test_thinking_and_call() {
    const json tools = json::parse(R"([{"type": "function", "function": {"name": "read_file", "parameters":
        {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"},
         "opts": {"type": "object"}}}}}])");
    const std::string out =
        "I should read the file.\n</think>\n\nReading it now.\n\n<tool_call>\n<function=read_file>\n"
        "<parameter=path>\n/tmp/a <b>.txt\n</parameter>\n<parameter=limit>\n10\n</parameter>\n"
        "<parameter=opts>\n{\"x\": [1, 2]}\n</parameter>\n</function>\n</tool_call>";
    for (size_t step : {1, 2, 3, 7, 1000}) {
        const Result r = run(out, true, tools, step);
        expect("thinking", r.thinking == "I should read the file.");
        expect("text", r.text == "Reading it now.");
        expect("one tool call", r.calls.size() == 1);
        if (r.calls.size() != 1) continue;
        const auto& c = r.calls[0];
        expect("name", c.name == "read_file");
        expect("path (string)", c.args["path"] == "/tmp/a <b>.txt");
        expect("limit (integer)", c.args["limit"] == 10);
        expect("opts (object)", c.args["opts"] == json::parse(R"({"x": [1, 2]})"));
    }
    // no thinking, text with angle brackets, two tool calls
    const std::string out2 =
        "a < b and <tool at the end\n<tool_call>\n<function=f>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=g>\n<parameter=x>\nmore\nlines\n</parameter>\n</function>\n</tool_call>";
    for (size_t step : {1, 5, 1000}) {
        const Result r = run(out2, false, json::array(), step);
        expect("text 2", r.text == "a < b and <tool at the end");
        expect("two calls", r.calls.size() == 2 && r.calls[0].name == "f" && r.calls[1].name == "g" &&
                                r.calls[1].args["x"] == "more\nlines");
    }
}

static void test_python_literals() {
    // From the BFCL run: lockDoors(unlock=False) arrived as the string "False" and left the doors unlocked
    const json tools = json::parse(R"([{"type": "function", "function": {"name": "lockDoors", "parameters":
        {"type": "dict", "properties": {"unlock": {"type": "boolean"}, "door": {"type": "array"},
         "mode": {"type": "string"}, "n": {"type": "integer"}}}}}])");
    auto call = [](const char* unlock) {
        return std::string("<tool_call>\n<function=lockDoors>\n<parameter=unlock>\n") + unlock +
               "\n</parameter>\n<parameter=door>\n[\"driver\"]\n</parameter>\n<parameter=mode>\nFalse\n"
               "</parameter>\n<parameter=n>\nNone\n</parameter>\n</function>\n</tool_call>";
    };
    const Result r = run(call("False"), false, tools, 1000);
    expect("one call", r.calls.size() == 1);
    if (r.calls.size() != 1) return;
    const auto& a = r.calls[0].args;
    expect("False is boolean", a["unlock"].is_boolean() && a["unlock"] == false);
    expect("array", a["door"] == json::parse(R"(["driver"])"));
    expect("string stays string", a["mode"] == "False");
    expect("None is null", a["n"].is_null());
    const Result r2 = run(call("True"), false, tools, 3);
    expect("True", r2.calls.size() == 1 && r2.calls[0].args["unlock"] == true);
    const Result r3 = run(call("false"), false, tools, 1000);
    expect("json false", r3.calls.size() == 1 && r3.calls[0].args["unlock"] == false);
}

static void test_nested_python_literals() {
    // Inside arrays and objects; True inside a string and a word that only starts with True stay as they are
    const json tools = json::parse(R"([{"type": "function", "function": {"name": "f", "parameters":
        {"type": "dict", "properties": {"x": {"type": "array"}}}}}])");
    auto call = [](const char* x) {
        return std::string("<tool_call>\n<function=f>\n<parameter=x>\n") + x +
               "\n</parameter>\n</function>\n</tool_call>";
    };
    const Result r = run(call(R"([True, {"a": None, "b": "True \" False"}])"), false, tools, 1000);
    expect("nested literals",
           r.calls.size() == 1 && r.calls[0].args["x"] == json::parse(R"([true, {"a": null, "b": "True \" False"}])"));
    const Result r2 = run(call("[Trueish, False]"), false, tools, 1000);
    expect("unknown word keeps the raw text", r2.calls.size() == 1 && r2.calls[0].args["x"] == "[Trueish, False]");
}

static void test_truncated() {
    // A truncated tool call (max_tokens) is dropped instead of delivered half; a complete one without </tool_call>
    // counts
    for (size_t step : {1, 1000}) {
        const std::string cut =
            "x\n<tool_call>\n<function=write>\n<parameter=path>\na.txt\n</parameter>\n"
            "<parameter=content>\nfirst half";
        const json no_tools = json::array();  // the parser keeps a reference: it must outlive the parser
        const auto parser = make_parser(false, no_tools);
        OutputParser& p = *parser;
        size_t calls = 0;
        for (size_t i = 0; i < cut.size(); i += step) calls += count_calls(p.feed(cut.substr(i, step)));
        calls += count_calls(p.finish());
        expect("truncated call dropped", calls == 0 && p.truncated());
        const Result r =
            run("<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n", false, json::array(), step);
        expect("complete call without </tool_call>", r.calls.size() == 1 && r.calls[0].args["a"] == "1");
        const Result r2 = run("<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\nhal", false,
                              json::array(), step);
        expect("call without </function> dropped", r2.calls.empty());
    }
    const Result r3 = run("half done </thi", true, json::array(), 1);
    expect("interrupted thinking", r3.thinking == "half done </thi" && r3.text.empty());
}

static void test_claude_code_request() {
    const json req = json::parse(R"({"system": [{"type": "text", "text": "S1"}, {"type": "text", "text": "S2"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "a"}]},
                     {"role": "system", "content": [{"type": "text", "text": "Context"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
                     {"role": "user", "content": "c"}],
        "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}, "max_tokens": 10})");
    const auto q = tona::server::from_anthropic(req, false);
    expect("system prompt from blocks",
           q.msgs.size() == 4 && q.msgs[0].role == "system" && q.msgs[0].content == "S1\nS2");
    expect("mid-history system message as user text", q.msgs[1].role == "user" && q.msgs[1].content == "a\n\nContext");
    expect("adaptive thinking, effort medium", q.thinking && q.effort == "medium");
    expect("renders", !tona::server::render_chat(fmt(), q.msgs, q.tools, q.thinking, q.effort).empty());
}

static void test_billing_header() {
    // The per-request billing block must not reach the prompt; other blocks stay untouched
    const auto render = [](const char* sys) {
        const std::string field = *sys ? std::string(R"("system": )") + sys + "," : "";
        const json req = json::parse("{" + field + R"("messages": [{"role": "user", "content": "c"}]})");
        const auto q = tona::server::from_anthropic(req, false);
        return tona::server::render_chat(fmt(), q.msgs, q.tools, q.thinking, q.effort);
    };
    const std::string hdr = R"({"type": "text", "text": "x-anthropic-billing-header: cc_version=1; cch=abc;"})";
    const std::string s1 = R"({"type": "text", "text": "S1"})";
    const std::string plain = render(("[" + s1 + "]").c_str());
    expect("billing header dropped", render(("[" + hdr + "," + s1 + "]").c_str()) == plain);
    expect("billing header value does not matter",
           render(("[" + hdr + "," + s1 + "]").c_str()) ==
               render((R"([{"type": "text", "text": "x-anthropic-billing-header: cch=zzz;"},)" + s1 + "]").c_str()));
    expect("header-only system equals no system", render(("[" + hdr + "]").c_str()) == render(""));
    const std::string other = R"({"type": "text", "text": "x-anthropic-other: keep"})";
    expect("other x-anthropic text kept", render(("[" + other + "]").c_str()) != render(""));
    expect("string system untouched", render(R"("S1")") == plain);
}

static void test_prefill(json& pr) {
    // The last assistant message is the start of the answer, without thinking; the answer excludes that start
    pr = {{"messages", json::parse(R"([{"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "text", "text": "{\"a\": "}]}])")},
          {"max_tokens", 10},
          {"thinking", {{"type", "enabled"}, {"budget_tokens", 2000}}}};
    const auto qp = tona::server::from_anthropic(pr, true);
    std::string all;
    for (const auto& c : tona::server::request_pieces(fmt(), qp)) all += c.text;
    expect("prefill at the end, no thinking",
           qp.prefill == "{\"a\":" && !qp.thinking && qp.msgs.size() == 1 && all.ends_with("</think>\n\n{\"a\":"));
    const auto parser = make_parser(qp.thinking, qp.tools);
    OutputParser& p = *parser;
    tona::server::prime(p, qp);
    std::string t;
    for (const auto& e : p.feed(" 1}")) t += e.delta;
    for (const auto& e : p.finish()) t += e.delta;
    expect("prefill not in the answer, space kept", t == " 1}");
}

static void test_tool_choice(json& pr) {
    // A required tool starts the answer, the parser sees the whole call; none drops the tools
    const json tl = json::parse(R"([{"name": "f", "input_schema": {"type": "object"}}])");
    pr["messages"] = json::parse(R"([{"role": "user", "content": "q"}])");
    pr["tools"] = tl;
    pr["tool_choice"] = {{"type", "tool"}, {"name", "f"}};
    const auto qt = tona::server::from_anthropic(pr, true);
    expect("tool_choice tool", qt.forced == "<tool_call>\n<function=f>\n" && !qt.thinking);
    const auto parser = make_parser(qt.thinking, qt.tools);
    OutputParser& p = *parser;
    tona::server::prime(p, qt);
    size_t calls = count_calls(p.feed("<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>"));
    calls += count_calls(p.finish());
    expect("forced call parsed", calls == 1);
    pr["tool_choice"] = {{"type", "any"}};
    expect("tool_choice any", tona::server::from_anthropic(pr, true).forced == "<tool_call>\n<function=");
    pr["tool_choice"] = {{"type", "none"}};
    expect("tool_choice none", tona::server::from_anthropic(pr, true).tools.empty());
    pr["tool_choice"] = {{"type", "tool"}, {"name", "g"}};
    bool threw = false;
    try {
        (void)tona::server::from_anthropic(pr, true);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect("tool_choice unknown tool", threw);
    const json oa = json::parse(R"({"messages": [{"role": "user", "content": "q"}], "tool_choice": "required",
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]})");
    expect("OpenAI required", tona::server::from_openai(oa, false).forced == "<tool_call>\n<function=");
}

static void test_format_facts() {
    // What the engine looks up in the vocabulary, and the refusal of a family without a format in this build
    const auto& f = fmt();
    expect("turn start", f.turn_start() == "<|im_start|>");
    expect("stop texts", f.stop_texts().size() == 2 && f.stop_texts()[0] == "<|im_end|>");
    expect("think end", f.think_end() == "</think>");
    expect("think close without the sentence", f.think_close(false) == "\n</think>\n\n");
    expect("think close with the sentence", f.think_close(true).starts_with("\n\nConsidering the limited time") &&
                                                f.think_close(true).ends_with("</think>\n\n"));
    const auto& n = tona::server::chat_format(tona::ChatFamily::Nemotron);
    expect("nemotron facts", n.turn_start() == "<|im_start|>" && n.stop_texts().size() == 1 &&
                                 n.stop_texts()[0] == "<|im_end|>" && n.think_end() == "</think>" &&
                                 n.think_close(true) == "</think>" && n.think_close(false) == "</think>");
}

// The Nemotron format by hand (the byte-equal check against the GGUF's template is tests/chat_vs_jinja.py): the system
// block is always there, content is not trimmed, and the pieces split at special tokens only
static void test_nemotron_render() {
    const auto& n = tona::server::chat_format(tona::ChatFamily::Nemotron);
    tona::server::Message u;
    u.role = "user", u.content = " hi ";
    const auto pieces = n.render_pieces({u}, json::array(), true, "xhigh");
    expect("nemotron prompt",
           tona::server::render_chat(n, {u}, json::array(), true) ==
               "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n hi <|im_end|>\n<|im_start|>assistant\n<think>\n");
    // The role name travels with the content: no boundary inside a word; the neighbours are special tokens
    size_t at = 0;
    while (at < pieces.size() && !pieces[at].content) at++;
    expect("nemotron pieces", at >= 1 && at + 1 < pieces.size() && pieces[at].text == "user\n hi " &&
                                  pieces[at - 1].text == "<|im_start|>" && pieces[at + 1].text == "<|im_end|>");
    const auto nothink = tona::server::render_chat(n, {u}, json::array(), false);
    expect("nemotron thinking off", nothink.ends_with("<|im_start|>assistant\n<think></think>"));
    expect("nemotron medium effort", tona::server::render_chat(n, {u}, json::array(), true, "medium")
                                         .contains(" hi \n\n{reasoning effort: efficient}<|im_end|>"));
    bool threw = false;
    try {
        (void)n.render_pieces({u}, json::array(), true, "bogus");
    } catch (const std::exception&) {
        threw = true;
    }
    expect("nemotron unknown effort", threw);
    const json no_tools = json::array();  // the parser keeps a reference: it must outlive the parser
    const auto p = n.make_parser(true, no_tools);
    auto ev =
        p->feed("think</think>\n<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>");
    expect("nemotron parser",
           ev.size() == 2 && ev[1].kind == tona::server::OutputParser::Event::tool_call && ev[1].call.name == "f");
}

static void test_stray_im_start_in_tool_call() {
    // Quantized Qwen3.8 models can emit <|im_start|> instead of '<' before function= inside <tool_call>
    // (vllm-project/vllm#69190); streaming and single-shot must still deliver the tool call.
    const json tools = json::parse(R"([{"type": "function", "function": {"name": "get_weather", "parameters":
        {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}])");
    const std::string out =
        "Checking.\n</think>\n\n<tool_call>\n<|im_start|>function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
    for (size_t step : {1, 3, 1000}) {
        const Result r = run(out, true, tools, step);
        expect("im_start thinking", r.thinking == "Checking.");
        expect("im_start one call", r.calls.size() == 1);
        if (r.calls.size() != 1) continue;
        expect("im_start name", r.calls[0].name == "get_weather");
        expect("im_start city", r.calls[0].args["city"] == "Paris");
        expect("im_start days", r.calls[0].args["days"] == 3);
    }
    const std::string no_close =
        "<tool_call>\n<|im_start|>function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n";
    const Result r2 = run(no_close, false, tools, 1000);
    expect("im_start call without </tool_call>",
           r2.calls.size() == 1 && r2.calls[0].name == "get_weather" && r2.calls[0].args["city"] == "Paris");
}

static void run_all() {
    test_format_facts();
    test_nemotron_render();
    test_thinking_and_call();
    test_stray_im_start_in_tool_call();
    test_python_literals();
    test_nested_python_literals();
    test_truncated();
    test_claude_code_request();
    test_billing_header();
    json pr;
    test_prefill(pr);
    test_tool_choice(pr);
}

int main() {
    try {
        run_all();
    } catch (const std::exception& e) {
        fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }
    if (g_fail) return 1;
    printf("parser: all cases equal\n");
    return 0;
}
