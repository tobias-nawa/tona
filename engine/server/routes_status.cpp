// Routes that carry no generation: gate, /status, /health, /v1/models, error handling and the request log
#include <cstdio>
#include <cstdlib>

#include "model/progress.h"
#include "server/http_util.h"
#include "server/request_guard.h"
#include "server/server_context.h"
#include "server/system_status.h"

namespace tona::server {

namespace {

// Constant-time comparison of the presented key with the configured one
bool authorized(const ServerContext& ctx, const httplib::Request& req) {
    const std::string& api_key = ctx.opt.api_key;
    if (api_key.empty() || !req.path.starts_with("/v1/")) return true;
    std::string got = req.get_header_value("x-api-key");
    if (got.empty()) {
        const std::string a = req.get_header_value("Authorization");
        if (a.starts_with("Bearer ")) got = a.substr(7);
    }
    if (got.size() != api_key.size()) return false;
    unsigned char d = 0;
    for (size_t i = 0; i < got.size(); i++) d |= (unsigned char)(got[i] ^ api_key[i]);
    return d == 0;
}

json status_body(ServerContext& ctx) {
    json ph = json::array();
    for (const auto& p : tona::progress::phases())
        ph.push_back({{"name", p.name},
                      {"done", p.done},
                      {"total", p.total},
                      {"t0", p.t0},
                      {"t1", p.t1 < 0 ? json(nullptr) : json(p.t1)}});
    json out = {
        {"ready", ctx.ready.load()}, {"uptime", tona::progress::now()}, {"phases", ph}, {"system", system_status()}};
    {
        std::lock_guard lk(ctx.load_mu);
        out["config"] = ctx.config;
        if (!ctx.load_error.empty()) out["error"] = ctx.load_error;
    }
    if (ctx.ready) out.update(ctx.engine->status());
    return out;
}

// Host (only on a loopback bind: there every client names 127.0.0.1, localhost or [::1]) and Origin, before the key:
// a page the user visits must neither drive nor, after DNS rebinding, read the server (server/request_guard.h)
bool browser_guard(const ServerContext& ctx, const httplib::Request& req, httplib::Response& res) {
    static const std::vector<std::string> extra = parse_origin_list(std::getenv("TONA_ALLOWED_ORIGINS"));
    const bool host_ok = !is_loopback_bind(ctx.opt.host) || host_allowed(req.get_header_value("Host"), ctx.opt.port);
    const bool origin_ok = origin_allowed(req.get_header_value("Origin"), ctx.opt.port, extra);
    if (host_ok && origin_ok) return true;
    fprintf(stderr, "[%s] refused %s %s: %s not allowed (TONA_ALLOWED_ORIGINS adds origins)\n", hms().c_str(),
            req.method.c_str(), req.path.c_str(), host_ok ? "Origin" : "Host");
    res.status = 403;
    res.set_content(
        json({{"type", "error"},
              {"error", {{"type", "permission_error"}, {"message", "request from a foreign site refused"}}}})
            .dump(),
        "application/json");
    return false;
}

}  // namespace

// /status and /health stay open to local clients (no content, only numbers and project names)
void register_gate(ServerContext& ctx) {
    ctx.svr.set_pre_routing_handler([&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!browser_guard(ctx, req, res)) return httplib::Server::HandlerResponse::Handled;
        if (!authorized(ctx, req)) {
            res.status = 401;
            res.set_content(json({{"type", "error"},
                                  {"error", {{"type", "authentication_error"}, {"message", "invalid x-api-key"}}}})
                                .dump(),
                            "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (ctx.ready || req.path == "/status") return httplib::Server::HandlerResponse::Unhandled;
        res.status = 503;
        res.set_header("Retry-After", "5");
        res.set_content(
            json({{"status", "loading"},
                  {"error", {{"type", "overloaded_error"}, {"message", "tona-server is still loading the model"}}}})
                .dump(),
            "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });
}

void register_status_routes(ServerContext& ctx) {
    ctx.svr.Get("/status", [&ctx](const httplib::Request&, httplib::Response& res) {
        res.set_content(status_body(ctx).dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });
    // n_ctx: context size for clients (just claude derives CLAUDE_CODE_MAX_CONTEXT_TOKENS from it)
    ctx.svr.Get("/health", [&ctx](const httplib::Request&, httplib::Response& res) {
        res.set_content(json({{"status", "ok"}, {"n_ctx", ctx.engine->max_ctx()}}).dump(), "application/json");
    });
    ctx.svr.Get("/v1/models", [&ctx](const httplib::Request&, httplib::Response& res) {
        // Both shapes in one: OpenAI (object) and Anthropic (type, display_name)
        const json mdl = {{"id", model_id()},
                          {"object", "model"},
                          {"type", "model"},
                          {"owned_by", "local"},
                          {"display_name", model_display_name()},
                          {"created_at", "2026-10-23T00:00:00Z"},
                          {"meta", {{"n_ctx", ctx.engine->max_ctx()}}}};
        res.set_content(json({{"object", "list"}, {"data", {mdl}}, {"has_more", false}}).dump(), "application/json");
    });
    // An exception in a handler goes to the log and to the client as an error in the API's format, instead of an empty
    // 500
    ctx.svr.set_exception_handler([](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
        std::string what;
        try {
            std::rethrow_exception(std::move(ep));
        } catch (const std::exception& e) {
            what = e.what();
        } catch (...) {
            what = "unknown";
        }
        fprintf(stderr, "%s Exception in %s: %s\n", hms().c_str(), req.path.c_str(), what.c_str());
        res.status = 500;
        res.set_content(anthropic_error("api_error", what).dump(), "application/json");
    });
    ctx.svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        if (req.path != "/health" && req.path != "/status")
            fprintf(stderr, "%s %s %s → %d\n", hms().c_str(), req.method.c_str(), req.path.c_str(), res.status);
    });
}

}  // namespace tona::server
