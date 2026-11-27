// HTTP server with the Anthropic (/v1/messages) and OpenAI API (/v1/chat/completions), streaming over SSE.
// One engine thread works through the queue (engine.h), the HTTP threads read the output.
// Usage: tona-server MODEL.gguf (--dflash DRAFTER.gguf | --mtp MTP.gguf) [--host 127.0.0.1] [--port 8662] [--ctx 16384]
//                   [--thinking on|off] [--profile EXPERTS.bin] [--api-key-file FILE] [--backend NAME|auto]
// HTTP listens while the model loads: GET /status (JSON, for tona) reports the startup phases and then live
// numbers; /v1/* answers 503 until loading is done. Sampling: temperature, top_p, top_k, seed of the request; without a
// temperature Qwen's recommendation (request_options.cpp).
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#include "backend/builtin.h"
#include "backend/env.h"
#include "backend/registry.h"
#include "backend/runtime_env.h"
#include "model/model_registry.h"
#include "model/progress.h"
#include "server/drafter_factory.h"
#include "server/drafter_options.h"
#include "server/http_util.h"
#include "server/server_context.h"

using tona::server::ServerContext;
using tona::server::ServerOptions;

namespace {

// 0: ok, otherwise the exit code
int read_api_key(const std::string& path, std::string& key) {
    std::ifstream f(path);
    std::getline(f, key);
    while (!key.empty() && std::isspace((unsigned char)key.back())) key.pop_back();
    if (key.size() < 16) {
        fprintf(stderr, "Key in %s is missing or shorter than 16 characters\n", path.c_str());
        return 2;
    }
    return 0;
}

template <class T>
bool parse_number(const std::string& text, T& out) {
    const auto r = std::from_chars(text.data(), text.data() + text.size(), out);
    return r.ec == std::errc() && r.ptr == text.data() + text.size();
}

int bad_value(const std::string& option, const std::string& value) {
    fprintf(stderr, "%s: \"%s\" is not a number\n", option.c_str(), value.c_str());
    return 2;
}

// Options that take their value as it is; false if a is none of them
bool text_option(const std::string& a, const std::string& v, ServerOptions& opt) {
    std::string* to = a == "--host"      ? &opt.host
                      : a == "--dflash"  ? &opt.dflash_path
                      : a == "--mtp"     ? &opt.mtp_path
                      : a == "--backend" ? &opt.backend
                                         : nullptr;
    if (to) *to = v;
    return to != nullptr;
}

// Parses the options after the model path into opt; returns the exit code (0: continue)
int parse_options(int argc, char** argv, ServerOptions& opt) {
    const char* env_backend = tona::env::text("TONA_BACKEND");
    opt.backend = env_backend ? env_backend : "auto";
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (text_option(a, v, opt)) continue;
        if (a == "--port") {
            if (!parse_number(v, opt.port)) return bad_value(a, v);
        } else if (a == "--ctx") {
            if (!parse_number(v, opt.ctx)) return bad_value(a, v);
        } else if (a == "--thinking")
            opt.think_default = v == "on";
        else if (a == "--profile")
            setenv("TONA_EXPERT_PROFILE", v.c_str(), 1);
        else if (a == "--api-key-file") {
            if (const int rc = read_api_key(v, opt.api_key)) return rc;
        } else {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    return 0;
}

// Loads the model, drafter, backend, generator and engine, then publishes the numbers for /status
void load_all(ServerContext& c, const char* model_path) {
    const ServerOptions& opt = c.opt;
    const auto t0 = std::chrono::steady_clock::now();
    tona::progress::begin("Open model");
    c.model = tona::load_model(model_path);
    // The model declares which drafter files it takes (ModelBase::drafters); make_drafters builds them after this check
    const std::string drafter_err =
        tona::server::drafter_error(c.model->drafters(), std::string(tona::arch_name(c.model->arch())),
                                    !opt.dflash_path.empty(), !opt.mtp_path.empty());
    if (!drafter_err.empty()) throw std::runtime_error(drafter_err);
    tona::server::set_model_identity(tona::server::identity_of(
        c.model->gguf().has("general.name") ? c.model->gguf().get_string("general.name") : std::string()));
    c.chat = &tona::server::chat_format(c.model->chat_family());
    c.tok = std::make_unique<tona::Tokenizer>(c.model->gguf());
    tona::server::Drafters dr = tona::server::make_drafters(*c.model, opt.dflash_path, opt.mtp_path);
    c.dg = std::move(dr.dflash_file), c.dfl = std::move(dr.dflash), c.mtp = std::move(dr.mtp);
    tona::backend::CreateOptions bo;
    bo.max_ctx = opt.ctx;
    bo.feat_layers = c.dfl ? c.dfl->target_layers() : std::vector<int64_t>{};
    bo.drafter = c.dg.get();
    bo.mtp = c.mtp.get();
    c.mg = tona::backend::registry::create(opt.backend, *c.model, bo);
    c.gen = std::make_unique<tona::server::Generator>(*c.model, *c.mg, c.dfl.get(), *c.tok, *c.chat);
    c.engine = std::make_unique<tona::server::Engine>(*c.gen, *c.tok);
    tona::progress::finish();
    const auto& info = c.mg->info();
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    {
        std::lock_guard lk(c.load_mu);
        c.config["kv_gb"] = info.kv_bytes / 1e9;
        c.config["kv_in"] = info.kv_host ? "RAM" : "VRAM";
        c.config["vram_gb"] = info.vram_bytes / 1e9;
        c.config["pinned_gb"] = info.pinned_bytes / 1e9;
        c.config["slots"] = info.expert_slots;
        c.config["load_s"] = load_s;
    }
    c.ready = true;
    fprintf(stderr,
            "loaded in %.1f s: context %lld (KV int8 %.2f GB in %s), VRAM %.2f GB, %d expert slots per layer, "
            "%.1f GB pinned\n",
            load_s, (long long)opt.ctx, info.kv_bytes / 1e9, info.kv_host ? "RAM" : "VRAM", info.vram_bytes / 1e9,
            info.expert_slots, info.pinned_bytes / 1e9);
}

}  // namespace

static int run(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--list-backends") {
        std::fputs(tona::backend::registry::support_matrix().c_str(), stdout);
        return 0;
    }
    if (argc < 2) {
        fprintf(stderr,
                "Tona. Usage: tona-server MODEL.gguf [--dflash DRAFTER.gguf | --mtp MTP.gguf] [--host H] [--port P] "
                "[--ctx N] [--thinking on|off] [--profile FILE] [--api-key-file FILE] [--backend NAME|auto]\n"
                "       tona-server --list-backends\n");
        return 2;
    }
    ServerContext ctx;
    if (const int rc = parse_options(argc, argv, ctx.opt)) return rc;
    const ServerOptions& opt = ctx.opt;
    // HTTP first, loading follows on the main thread; until then only /status answers (and /health with 503)
    ctx.config = {{"model", std::filesystem::path(argv[1]).filename().string()},
                  {"drafter", !opt.dflash_path.empty() ? "dflash"
                              : !opt.mtp_path.empty()  ? "mtp"
                                                       : "none"},
                  {"host", opt.host},
                  {"port", opt.port},
                  {"n_ctx", opt.ctx},
                  {"pid", (int64_t)getpid()}};
    tona::server::register_gate(ctx);
    tona::server::register_status_routes(ctx);
    tona::server::register_anthropic_routes(ctx);
    tona::server::register_openai_routes(ctx);
    if (!ctx.svr.bind_to_port(opt.host, opt.port)) {
        fprintf(stderr, "Error: cannot listen on %s:%d\n", opt.host.c_str(), opt.port);
        return 1;
    }
    std::thread http([&] { ctx.svr.listen_after_bind(); });
    fprintf(
        stderr,
        "Tona (tona-server) on http://%s:%d (Anthropic: /v1/messages, OpenAI: /v1/chat/completions), %s, loading ...\n",
        opt.host.c_str(), opt.port, opt.api_key.empty() ? "without key" : "with key");
    if (opt.api_key.empty() && opt.host != "127.0.0.1" && opt.host != "localhost")
        fprintf(stderr, "Warning: %s is reachable from outside and has no key (--api-key-file)\n", opt.host.c_str());
    try {
        load_all(ctx, argv[1]);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        {
            std::lock_guard lk(ctx.load_mu);
            ctx.load_error = e.what();
        }
        // Give tona time to pick up the error
        std::this_thread::sleep_for(std::chrono::seconds(tona::env::integer("TONA_ERROR_LINGER_S", 3)));
        ctx.svr.stop();
        http.join();
        return 1;
    }
    http.join();
    return 0;
}

int main(int argc, char** argv) {
    tona::runtime_env::set_defaults(argv);
    try {
        tona::backend::register_builtin();
        return run(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "Error: unknown exception\n");
    }
    return 1;
}
