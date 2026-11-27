#include "client/server_view.h"

#include <algorithm>

#include "client/term.h"

namespace tona::client {

namespace {

using top::jnum;
using top::jstr;

Conversation parse_conversation(const json& a, const std::string& state) {
    Conversation c;
    const json origin = a.contains("origin") ? a["origin"] : json::object();
    c.agent = jstr(origin, "agent"), c.project = jstr(origin, "project"), c.role = jstr(origin, "role");
    c.state = state;
    c.tok_s = jnum(a, "tok_s");
    const double done = jnum(a, "prefill_done"), total = jnum(a, "prefill_total"), elapsed = jnum(a, "elapsed");
    c.prefill_fraction = total > 0 ? std::min(1.0, done / total) : 0;
    c.prefill_tok_s = elapsed > 0 ? done / elapsed : 0;
    return c;
}

void parse_requests(const json& engine, ServerView& v) {
    // engine.active lists every running request; an older server only has the engine itself
    std::vector<json> active;
    if (engine.contains("active") && engine["active"].is_array())
        active.assign(engine["active"].begin(), engine["active"].end());
    else if (jstr(engine, "state", "idle") != "idle")
        active.push_back(engine);
    for (const json& a : active) {
        const std::string state = jstr(a, "state");
        if (state == "decode" || state == "prefill") v.running.push_back(parse_conversation(a, state));
    }
    v.queue = (int)jnum(engine, "queue");
    v.slots = (int)jnum(engine, "slots");
    if (engine.contains("waiting") && engine["waiting"].is_array())
        for (const json& w : engine["waiting"]) {
            Conversation c;
            c.agent = jstr(w, "agent"), c.project = jstr(w, "project"), c.role = jstr(w, "role"), c.state = "waiting";
            v.running.push_back(c);
        }
}

void parse_memory(const json& status, ServerView& v) {
    const json sys = status.contains("system") ? status["system"] : json::object();
    const json gpu = sys.contains("gpu") ? sys["gpu"] : json::object();
    const json ram = sys.contains("ram") ? sys["ram"] : json::object();
    v.mem.vram_used = jnum(gpu, "vram_used"), v.mem.vram_total = jnum(gpu, "vram_total");
    v.mem.unified = gpu.value("unified", false);
    v.mem.ram_total = jnum(ram, "total"), v.mem.ram_available = jnum(ram, "available");
    const json cfg = status.contains("config") ? status["config"] : json::object();
    // Pinned experts, plus the KV cache when it lives in RAM (server_main.cpp: config.kv_in)
    double tona = jnum(cfg, "pinned_gb");
    if (jstr(cfg, "kv_in") == "RAM") tona += jnum(cfg, "kv_gb");
    v.mem.ram_tona = tona * 1e9;
    v.mem.tona_gpu = jnum(cfg, "vram_gb") * 1e9;
}

void parse_phases(const json& status, ServerView& v) {
    if (!status.contains("phases") || !status["phases"].is_array()) return;
    for (const json& p : status["phases"]) {
        LoadPhase ph;
        ph.name = jstr(p, "name");
        ph.done = p.contains("t1") && !p["t1"].is_null();
        const double total = jnum(p, "total");
        ph.fraction = ph.done ? 1.0 : total > 0 ? std::min(1.0, jnum(p, "done") / total) : 0;
        v.phases.push_back(ph);
    }
}

}  // namespace

double ServerView::load_fraction() const {
    if (ready) return 1;
    if (phases.empty()) return 0;
    double sum = 0;
    for (const LoadPhase& p : phases) sum += p.fraction;
    return sum / (double)phases.size();
}

int ServerView::conversation_count() const {
    return (int)std::count_if(running.begin(), running.end(),
                              [](const Conversation& c) { return c.state != "waiting"; });
}

double ServerView::total_tok_s() const {
    double sum = 0;
    for (const Conversation& c : running)
        if (c.state == "decode") sum += c.tok_s;
    return sum;
}

std::string model_stem(const std::string& file) {
    std::string m = file;
    if (const size_t k = m.find("-00001-of-"); k != std::string::npos) m.resize(k);
    if (m.ends_with(".gguf")) m.resize(m.size() - 5);
    return m;
}

ServerView parse_status(const json& status, bool up, const std::string& error) {
    ServerView v;
    v.up = up, v.error = error;
    if (!up || !status.is_object()) return v;
    v.ready = status.value("ready", false);
    v.uptime = jnum(status, "uptime");
    if (status.contains("error")) v.error = jstr(status, "error");
    const json cfg = status.contains("config") ? status["config"] : json::object();
    v.model_file = jstr(cfg, "model"), v.drafter = jstr(cfg, "drafter"), v.n_ctx = jnum(cfg, "n_ctx");
    parse_phases(status, v);
    parse_memory(status, v);
    if (status.contains("engine") && status["engine"].is_object()) parse_requests(status["engine"], v);
    if (status.contains("recent") && status["recent"].is_array())
        for (const json& r : status["recent"]) {
            RecentRequest x;
            x.time = (int64_t)jnum(r, "time"), x.agent = jstr(r, "agent"), x.project = jstr(r, "project");
            x.out_tokens = jnum(r, "out"), x.tok_s = jnum(r, "tok_s");
            v.recent.push_back(x);
        }
    return v;
}

}  // namespace tona::client
