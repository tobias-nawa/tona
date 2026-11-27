// OpenAI API: POST /v1/chat/completions (plain and streamed)
#include <chrono>
#include <format>

#include "server/http_util.h"
#include "server/request_options.h"
#include "server/server_context.h"

namespace tona::server {

namespace {

void openai_error(httplib::Response& res, int status, const std::string& msg) {
    send_error(res, status, {{"error", {{"message", msg}, {"type", "invalid_request_error"}}}});
}

std::string finish_reason(const Job& j, bool tools) {
    if (j.why == Stop::length && j.stop_seq.empty()) return "length";  // a cut-off call is missing
    return tools ? "tool_calls" : "stop";
}

json usage_of(const Job& j) {
    return {{"prompt_tokens", j.prompt.size()},
            {"completion_tokens", j.out_tokens},
            {"total_tokens", j.prompt.size() + j.out_tokens},
            {"prompt_tokens_details", {{"cached_tokens", j.info.reused}}}};
}

// Timings in the format of llama.cpp (scripts/bench-openai.py reads predicted_per_second)
json timings_of(const Job& j) {
    return {{"cache_n", j.info.reused},          {"prompt_n", j.info.computed},
            {"prompt_ms", j.info.seconds * 1e3}, {"predicted_n", j.out_tokens},
            {"predicted_ms", j.gen_s * 1e3},     {"predicted_per_second", j.out_tokens / std::max(j.gen_s, 1e-9)}};
}

json call_json(const ToolCall& c, int index, bool with_index) {
    json t = {
        {"id", rand_id("call_")}, {"type", "function"}, {"function", {{"name", c.name}, {"arguments", c.args.dump()}}}};
    if (with_index) t["index"] = index;
    return t;
}

bool parse_request(const ServerContext& ctx, const httplib::Request& req, httplib::Response& res, Request& q,
                   Job& job) {
    try {
        const json body = json::parse(req.body);
        q = from_openai(body, ctx.opt.think_default);
        tag_origin(job, req, body);
        q.effort = role_effort(job.role, q.effort);
        job.prompt = prompt_of(ctx, q);
        return true;
    } catch (const std::exception& e) {
        openai_error(res, 400, e.what());
        return false;
    }
}

struct Identity {
    std::string id;
    int64_t created;
};

void respond_plain(Job& job, const Request& q, const Identity& who, const ChatFormat& fmt, const httplib::Request& req,
                   httplib::Response& res) {
    const auto parser = fmt.make_parser(q.thinking, q.tools);
    OutputParser& p = *parser;
    prime(p, q);
    std::string thinking, text;
    json calls = json::array();
    auto take = [&](const std::vector<OutputParser::Event>& ev) {
        for (const auto& e : ev) {
            if (e.kind == OutputParser::Event::thinking)
                thinking += e.delta;
            else if (e.kind == OutputParser::Event::text)
                text += e.delta;
            else
                calls.push_back(call_json(e.call, (int)calls.size(), false));
        }
    };
    drain(
        job,
        [&](const std::string& s) {
            take(p.feed(s));
            return true;
        },
        [&] { return !req.is_connection_closed(); });
    take(p.finish());
    if (!job.error.empty()) return openai_error(res, 500, job.error);
    json msg = {{"role", "assistant"}, {"content", text.empty() && !calls.empty() ? json(nullptr) : json(text)}};
    if (!thinking.empty()) msg["reasoning_content"] = thinking;
    if (!calls.empty()) msg["tool_calls"] = calls;
    json out = {{"id", who.id},
                {"object", "chat.completion"},
                {"created", who.created},
                {"model", model_id()},
                {"choices", {{{"index", 0}, {"message", msg}, {"finish_reason", finish_reason(job, !calls.empty())}}}},
                {"usage", usage_of(job)},
                {"timings", timings_of(job)}};
    if (job.warm) out["tona_warm"] = warm_json(job);
    res.set_content(out.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
}

struct ChunkStream {
    ChunkStream(httplib::DataSink& s, const Identity& w) : sink(s), who(w) {}
    httplib::DataSink& sink;
    const Identity& who;
    bool ok = true;
    int ncalls = 0;

    bool chunk(const json& delta, const json& fin) {
        const json c = {{"id", who.id},
                        {"object", "chat.completion.chunk"},
                        {"created", who.created},
                        {"model", model_id()},
                        {"choices", {{{"index", 0}, {"delta", delta}, {"finish_reason", fin}}}}};
        const std::string s = sse("", c);
        return sink.write(s.data(), s.size());
    }
    void emit(const std::vector<OutputParser::Event>& ev) {
        for (const auto& e : ev) {
            if (e.kind == OutputParser::Event::thinking)
                ok = ok && chunk({{"reasoning_content", e.delta}}, nullptr);
            else if (e.kind == OutputParser::Event::text)
                ok = ok && chunk({{"content", e.delta}}, nullptr);
            else
                ok = ok && chunk({{"tool_calls", {call_json(e.call, ncalls++, true)}}}, nullptr);
        }
    }
};

void stream_completion(Job& job, const Request& q, const Identity& who, const ChatFormat& fmt,
                       httplib::DataSink& sink) {
    ChunkStream cs(sink, who);
    cs.ok = cs.chunk({{"role", "assistant"}, {"content", ""}}, nullptr);
    // Empty chunks during the prefill (SSE comments are ignored by some clients, e.g. Qwen Code)
    if (!cs.ok || !wait_started(job, [&] { return cs.chunk({{"content", ""}}, nullptr); })) {
        job.cancel = true;
        sink.done();
        return;
    }
    const auto parser = fmt.make_parser(q.thinking, q.tools);
    OutputParser& p = *parser;
    prime(p, q);
    drain(
        job,
        [&](const std::string& s) {
            cs.emit(p.feed(s));
            return cs.ok;
        },
        [&] { return sink.is_writable(); });
    cs.emit(p.finish());
    if (!job.error.empty()) {
        const std::string s = sse("", {{"error", {{"message", job.error}, {"type", "server_error"}}}});
        sink.write(s.data(), s.size());
    }
    const json c = {
        {"id", who.id},
        {"object", "chat.completion.chunk"},
        {"created", who.created},
        {"model", model_id()},
        {"choices", {{{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason(job, cs.ncalls > 0)}}}},
        {"usage", usage_of(job)},
        {"timings", timings_of(job)}};
    const std::string s = sse("", c) + "data: [DONE]\n\n";
    sink.write(s.data(), s.size());
    sink.done();
}

void handle_completions(ServerContext& ctx, const httplib::Request& req, httplib::Response& res) {
    Request q;
    auto job = std::make_shared<Job>();
    if (!parse_request(ctx, req, res, q, *job)) return;
    if ((int64_t)job->prompt.size() + 8 > ctx.engine->max_ctx())
        // OpenAI's wording (context_length_exceeded)
        return openai_error(res, 400,
                            std::format("This model's maximum context length is {} tokens. However, your "
                                        "messages resulted in {} tokens.",
                                        ctx.engine->max_ctx(), job->prompt.size() + 8));
    configure_job(*job, q);
    ctx.engine->submit(job);
    const Identity who{
        rand_id("chatcmpl-"),
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()};
    if (!q.stream) return respond_plain(*job, q, who, *ctx.chat, req, res);
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream",
                                     [job, q = q, who = who, fmt = ctx.chat](size_t, httplib::DataSink& sink) {
                                         stream_completion(*job, q, who, *fmt, sink);
                                         return true;
                                     });
}

}  // namespace

void register_openai_routes(ServerContext& ctx) {
    ctx.svr.Post("/v1/chat/completions",
                 [&ctx](const httplib::Request& req, httplib::Response& res) { handle_completions(ctx, req, res); });
}

}  // namespace tona::server
