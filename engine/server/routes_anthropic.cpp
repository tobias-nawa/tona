// Anthropic API: POST /v1/messages (plain and streamed) and /v1/messages/count_tokens
#include <atomic>
#include <format>
#include <fstream>

#include "backend/env.h"
#include "server/chat_tokens.h"
#include "server/http_util.h"
#include "server/request_options.h"
#include "server/server_context.h"

namespace tona::server {

namespace {

json stop_reason(const Job& j, bool tools) {
    if (!j.stop_seq.empty()) return "stop_sequence";
    if (j.why == Stop::length) return "max_tokens";  // also with tool calls: the last one was cut off
    return tools ? "tool_use" : "end_turn";
}

json stop_sequence(const Job& j) { return j.stop_seq.empty() ? json(nullptr) : json(j.stop_seq); }

json tool_use_block(const OutputParser::Event& e) {
    return {{"type", "tool_use"}, {"id", rand_id("toolu_")}, {"name", e.call.name}, {"input", e.call.args}};
}

// TONA_DUMP_DIR: every request as a file (time, running number), to replay and analyze
void dump_request(const httplib::Request& req) {
    const char* dd = tona::env::text("TONA_DUMP_DIR");
    if (!dd) return;
    static std::atomic<int> nreq{0};
    std::ofstream(std::format("{}/{}-{:04d}.json", dd, hms(), nreq++)) << req.body;
}

// Parse the body into q and the job's origin and prompt; on an invalid request answers 400 and returns false
bool parse_request(const ServerContext& ctx, const httplib::Request& req, httplib::Response& res, Request& q,
                   Job& job) {
    try {
        const json body = json::parse(req.body);
        q = from_anthropic(body, ctx.opt.think_default);
        tag_origin(job, req, body);
        q.effort = role_effort(job.role, q.effort);
        job.prompt = prompt_of(ctx, q);
        return true;
    } catch (const std::exception& e) {
        send_error(res, 400, anthropic_error("invalid_request_error", e.what()));
        return false;
    }
}

// Non-streaming answer: content blocks in the order thinking, text / tool calls
struct Collector {
    std::unique_ptr<OutputParser> parser;
    json content = json::array();
    std::string thinking, text;

    Collector(const ChatFormat& fmt, const Request& q) : parser(fmt.make_parser(q.thinking, q.tools)) {
        prime(*parser, q);
    }

    void take(const std::vector<OutputParser::Event>& ev) {
        for (const auto& e : ev) {
            if (e.kind == OutputParser::Event::thinking)
                thinking += e.delta;
            else if (e.kind == OutputParser::Event::text)
                text += e.delta;
            else {
                if (!text.empty()) content.push_back({{"type", "text"}, {"text", text}}), text.clear();
                content.push_back(tool_use_block(e));
            }
        }
    }
};

void respond_plain(Job& job, const Request& q, const std::string& id, const ChatFormat& fmt,
                   const httplib::Request& req, httplib::Response& res) {
    Collector col(fmt, q);
    drain(
        job,
        [&](const std::string& s) {
            col.take(col.parser->feed(s));
            return true;
        },
        [&] { return !req.is_connection_closed(); });
    col.take(col.parser->finish());
    if (!job.error.empty()) return send_error(res, 500, anthropic_error("api_error", job.error));
    json& content = col.content;
    if (!col.text.empty()) content.push_back({{"type", "text"}, {"text", col.text}});
    if (!col.thinking.empty())
        content.insert(content.begin(), json{{"type", "thinking"}, {"thinking", col.thinking}, {"signature", "tona"}});
    bool tools = false;
    for (const auto& b : content) tools = tools || b["type"] == "tool_use";
    json out = {{"id", id},
                {"type", "message"},
                {"role", "assistant"},
                {"model", model_id()},
                {"content", content},
                {"stop_reason", stop_reason(job, tools)},
                {"stop_sequence", stop_sequence(job)},
                {"usage",
                 {{"input_tokens", job.info.computed},
                  {"cache_read_input_tokens", job.info.reused},
                  {"cache_creation_input_tokens", 0},
                  {"output_tokens", job.out_tokens}}}};
    if (job.warm) out["tona_warm"] = warm_json(job);
    res.set_content(out.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
}

// Streaming answer as server-sent events: content blocks opened and closed as the parser delivers events
struct EventStream {
    explicit EventStream(httplib::DataSink& s) : sink(s) {}
    httplib::DataSink& sink;
    int idx = -1;
    std::string open;  // kind of the open block
    bool tools = false, ok = true;

    bool put(const std::string& ev, const json& d) const {
        const std::string s = sse(ev, d);
        return sink.write(s.data(), s.size());
    }
    void block_stop() { ok = ok && put("content_block_stop", {{"type", "content_block_stop"}, {"index", idx}}); }
    void close() {
        if (open.empty()) return;
        if (open == "thinking")
            ok = ok && put("content_block_delta", {{"type", "content_block_delta"},
                                                   {"index", idx},
                                                   {"delta", {{"type", "signature_delta"}, {"signature", "tona"}}}});
        block_stop();
        open.clear();
    }
    void tool_call(const OutputParser::Event& e) {
        close();
        tools = true;
        idx++;
        ok = ok && put("content_block_start", {{"type", "content_block_start"},
                                               {"index", idx},
                                               {"content_block",
                                                {{"type", "tool_use"},
                                                 {"id", rand_id("toolu_")},
                                                 {"name", e.call.name},
                                                 {"input", json::object()}}}});
        ok = ok && put("content_block_delta",
                       {{"type", "content_block_delta"},
                        {"index", idx},
                        {"delta", {{"type", "input_json_delta"}, {"partial_json", e.call.args.dump()}}}});
        block_stop();
    }
    void text_delta(const OutputParser::Event& e) {
        const std::string kind = e.kind == OutputParser::Event::thinking ? "thinking" : "text";
        if (open != kind) {
            close();
            idx++;
            open = kind;
            ok = ok && put("content_block_start",
                           {{"type", "content_block_start"},
                            {"index", idx},
                            {"content_block", kind == "thinking" ? json{{"type", "thinking"}, {"thinking", ""}}
                                                                 : json{{"type", "text"}, {"text", ""}}}});
        }
        ok = ok && put("content_block_delta",
                       {{"type", "content_block_delta"},
                        {"index", idx},
                        {"delta", kind == "thinking" ? json{{"type", "thinking_delta"}, {"thinking", e.delta}}
                                                     : json{{"type", "text_delta"}, {"text", e.delta}}}});
    }
    void emit(const std::vector<OutputParser::Event>& ev) {
        for (const auto& e : ev) {
            if (e.kind == OutputParser::Event::tool_call)
                tool_call(e);
            else
                text_delta(e);
        }
    }
};

void stream_message(Job& job, const Request& q, const std::string& id, const ChatFormat& fmt, httplib::DataSink& sink) {
    EventStream es(sink);
    // message_start right away, so the client sees bytes during the prefill; the split into computed and reused
    // tokens follows in message_delta
    es.put("message_start", {{"type", "message_start"},
                             {"message",
                              {{"id", id},
                               {"type", "message"},
                               {"role", "assistant"},
                               {"model", model_id()},
                               {"content", json::array()},
                               {"stop_reason", nullptr},
                               {"stop_sequence", nullptr},
                               {"usage",
                                {{"input_tokens", job.prompt.size()},
                                 {"cache_read_input_tokens", 0},
                                 {"cache_creation_input_tokens", 0},
                                 {"output_tokens", 0}}}}}});
    if (!wait_started(job, [&] { return es.put("ping", {{"type", "ping"}}); })) {
        sink.done();
        return;
    }
    std::string early;  // error before the first token; job.error is only final once done, and read under mu
    {
        std::lock_guard lk(job.mu);
        if (job.done) early = job.error;
    }
    if (!early.empty()) {
        es.put("error", anthropic_error("api_error", early));
        sink.done();
        return;
    }
    const auto parser = fmt.make_parser(q.thinking, q.tools);
    OutputParser& p = *parser;
    prime(p, q);
    drain(
        job,
        [&](const std::string& s) {
            es.emit(p.feed(s));
            return es.ok;
        },
        [&] { return sink.is_writable(); });
    es.emit(p.finish());
    es.close();
    if (!job.error.empty()) es.put("error", anthropic_error("api_error", job.error));
    es.put("message_delta",
           {{"type", "message_delta"},
            {"delta", {{"stop_reason", stop_reason(job, es.tools)}, {"stop_sequence", stop_sequence(job)}}},
            {"usage",
             {{"input_tokens", job.info.computed},
              {"cache_read_input_tokens", job.info.reused},
              {"output_tokens", job.out_tokens}}}});
    es.put("message_stop", {{"type", "message_stop"}});
    sink.done();
}

void handle_messages(ServerContext& ctx, const httplib::Request& req, httplib::Response& res) {
    dump_request(req);
    Request q;
    auto job = std::make_shared<Job>();
    if (!parse_request(ctx, req, res, q, *job)) return;
    // The Anthropic API's wording: Claude Code recognizes the overflow by it (auto-compact)
    if ((int64_t)job->prompt.size() + 8 > ctx.engine->max_ctx())
        return send_error(
            res, 400,
            anthropic_error("invalid_request_error", std::format("prompt is too long: {} tokens > {} maximum",
                                                                 job->prompt.size() + 8, ctx.engine->max_ctx())));
    configure_job(*job, q);
    ctx.engine->submit(job);
    const std::string id = rand_id("msg_");
    if (!q.stream) return respond_plain(*job, q, id, *ctx.chat, req, res);
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream",
                                     [job, q = q, id = id, fmt = ctx.chat](size_t, httplib::DataSink& sink) {
                                         stream_message(*job, q, id, *fmt, sink);
                                         return true;
                                     });
}

}  // namespace

void register_anthropic_routes(ServerContext& ctx) {
    ctx.svr.Post("/v1/messages/count_tokens", [&ctx](const httplib::Request& req, httplib::Response& res) {
        try {
            const Request q = from_anthropic(json::parse(req.body), ctx.opt.think_default);
            res.set_content(json({{"input_tokens", prompt_of(ctx, q).size()}}).dump(), "application/json");
        } catch (const std::exception& e) {
            send_error(res, 400, anthropic_error("invalid_request_error", e.what()));
        }
    });
    ctx.svr.Post("/v1/messages",
                 [&ctx](const httplib::Request& req, httplib::Response& res) { handle_messages(ctx, req, res); });
}

}  // namespace tona::server
