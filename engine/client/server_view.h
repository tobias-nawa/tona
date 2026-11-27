// What the server reports, as typed values: GET /status parsed once, so the screens and the commands without a screen
// read the same fields. Pure: no terminal, no network.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "third_party/nlohmann/json.hpp"

namespace tona::client {

using nlohmann::json;

struct Conversation {
    std::string agent, project, role;  // agent: "claude", "opencode" ...; role: "main", "auxiliary" or "subagent"
    std::string state;                 // "decode" (writes), "prefill" (reads in) or "waiting"
    double tok_s = 0;                  // decode rate over the whole request
    double prefill_fraction = 0, prefill_tok_s = 0;
};

struct RecentRequest {
    int64_t time = 0;
    std::string agent, project;
    double out_tokens = 0, tok_s = 0;
};

struct LoadPhase {
    std::string name;
    double fraction = 0;  // 0..1, 1 when finished
    bool done = false;
};

// All sizes in bytes
struct Memory {
    double vram_used = 0, vram_total = 0;
    double ram_total = 0, ram_available = 0;
    double ram_tona = 0;   // pinned experts and a KV cache in RAM, from the server's config
    bool unified = false;  // CPU and GPU share the RAM (Apple silicon): no graphics memory of its own
    double tona_gpu = 0;   // what the model holds for the GPU: weights and buffers (config.vram_gb)
};

struct ServerView {
    bool up = false;         // /status answered
    bool ready = false;      // model loaded
    std::string error;       // why not reachable, or the load error
    std::string model_file;  // file name of the loaded model (config.model)
    std::string drafter;
    double uptime = 0, n_ctx = 0;
    std::vector<LoadPhase> phases;
    std::vector<Conversation> running;  // running requests, then the waiting ones
    int slots = 0;                      // conversations that fit at once
    int queue = 0;
    std::vector<RecentRequest> recent;  // newest last
    Memory mem;

    // Loading: server up, model not ready, no error. Progress is the mean over the phases.
    bool loading() const { return up && !ready && error.empty(); }
    double load_fraction() const;
    // Running conversations (decode or prefill), not the waiting ones
    int conversation_count() const;
    // Sum of the decode rates of all running requests
    double total_tok_s() const;
};

ServerView parse_status(const json& status, bool up, const std::string& error);

// "Qwen3.8-...-00001-of-00002.gguf" -> "Qwen3.8-..." (part number and extension removed)
std::string model_stem(const std::string& file);

}  // namespace tona::client
