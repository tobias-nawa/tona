// State shared by the HTTP routes and the startup code
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cpu/dflash.h"
#include "model/model_base.h"
#include "model/qwen4exp.h"
#include "model/tokenizer.h"
#include "server/api.h"
#include "server/chat_format.h"
#include "server/engine.h"
#include "server/generator.h"
#include "third_party/httplib.h"

namespace tona::server {

struct ServerOptions {
    std::string host = "127.0.0.1", dflash_path, mtp_path, api_key;
    int port = 8662;
    int64_t ctx = 16384;
    bool think_default = false;
    std::string backend;  // --backend, else TONA_BACKEND, else "auto"
};

struct ServerContext {
    ServerOptions opt;
    httplib::Server svr;
    std::atomic<bool> ready{false};
    std::mutex load_mu;  // guards load_error and config
    std::string load_error;
    json config;
    // Declaration order is destruction order backwards: the engine goes first
    std::unique_ptr<tona::ModelBase> model;  // any architecture; the Coder-only parts take model_as<Qwen4ExpModel>
    const ChatFormat* chat = nullptr;        // format of the model's family, set at load
    std::unique_ptr<tona::Tokenizer> tok;
    std::unique_ptr<tona::Gguf> dg;
    std::unique_ptr<tona::ref::DFlash> dfl;
    std::unique_ptr<tona::MtpHead> mtp;
    std::unique_ptr<tona::backend::Backend> mg;  // reference backend gpu::Mega
    std::unique_ptr<Generator> gen;
    std::unique_ptr<Engine> engine;
};

std::vector<int32_t> prompt_of(const ServerContext& ctx, const Request& q);
void send_error(httplib::Response& res, int status, const json& body);
// tona_warm field of a warm-up answer: prompt tokens the stored state covers, the prompt, the last stored position, and
// whether a real request interrupted it
json warm_json(const Job& j);

// /v1/* needs the API key (if set); until the model is loaded only /status answers
void register_gate(ServerContext& ctx);
// /status, /health, /v1/models, exception handler and request log
void register_status_routes(ServerContext& ctx);
void register_anthropic_routes(ServerContext& ctx);
void register_openai_routes(ServerContext& ctx);

}  // namespace tona::server
