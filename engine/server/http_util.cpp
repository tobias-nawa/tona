#include "server/http_util.h"

#include <chrono>
#include <ctime>
#include <format>
#include <mutex>
#include <random>

namespace tona::server {

namespace {
std::mutex g_identity_mu;
// Function-local: built on first use, so a failed allocation is an exception in a caller instead of a crash at startup
ModelIdentity& identity() {
    static ModelIdentity id = identity_of(kCoderGgufName);
    return id;
}
}  // namespace

namespace {
bool is_alnum(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

// Lowercase letters, digits and dots; every other run of characters becomes one '-', none at either end
std::string slug_of(const std::string& name) {
    std::string id;
    for (const char c : name) {
        if (is_alnum(c) || c == '.') {
            id += static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
        } else if (!id.empty() && id.back() != '-') {
            id += '-';
        }
    }
    while (!id.empty() && id.back() == '-') id.pop_back();
    return id;
}
}  // namespace

ModelIdentity identity_of(const std::string& gguf_name) {
    const std::string id = gguf_name == kCoderGgufName ? std::string() : slug_of(gguf_name);
    if (id.empty()) return {kModelName, "Qwen3.8 Flash Next Coder (local)"};
    return {id, gguf_name + " (local)"};
}

void set_model_identity(ModelIdentity id) {
    std::lock_guard lk(g_identity_mu);
    identity() = std::move(id);
}

std::string model_id() {
    std::lock_guard lk(g_identity_mu);
    return identity().id;
}

std::string model_display_name() {
    std::lock_guard lk(g_identity_mu);
    return identity().display_name;
}

std::string rand_id(const char* prefix) {
    static std::mt19937_64 rng(std::random_device{}());
    static std::mutex mu;
    std::lock_guard lk(mu);
    return std::format("{}{:016x}{:08x}", prefix, rng(), (uint32_t)rng());
}

std::string hms() {
    const std::time_t t = std::time(nullptr);
    char b[16];
    std::strftime(b, sizeof b, "%H:%M:%S", std::localtime(&t));
    return b;
}

std::string sse(const std::string& event, const json& data) {
    return (event.empty() ? "" : "event: " + event + "\n") +
           "data: " + data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

json anthropic_error(const std::string& type, const std::string& msg) {
    return {{"type", "error"}, {"error", {{"type", type}, {"message", msg}}}};
}

void drain(Job& j, const std::function<bool(const std::string&)>& on_piece, const std::function<bool()>& alive) {
    auto last = std::chrono::steady_clock::now();
    while (true) {
        std::string s;
        bool fin;
        {
            std::unique_lock lk(j.mu);
            j.cv.wait_for(lk, std::chrono::milliseconds(500), [&] { return j.done || !j.pieces.empty(); });
            if (!j.pieces.empty()) {
                s = std::move(j.pieces.front());
                j.pieces.pop_front();
            }
            fin = j.done && j.pieces.empty();
        }
        if (!s.empty() && !on_piece(s)) {
            j.cancel = true;
            return;
        }
        if (fin && s.empty()) return;
        if (const auto now = std::chrono::steady_clock::now(); !fin && now - last >= std::chrono::milliseconds(500)) {
            last = now;
            if (!alive()) {
                j.cancel = true;
                return;
            }
        }
    }
}

bool wait_started(Job& j, const std::function<bool()>& ping) {
    std::unique_lock lk(j.mu);
    while (!j.cv.wait_for(lk, std::chrono::seconds(5), [&] { return j.started; })) {
        lk.unlock();
        const bool ok = ping();
        lk.lock();
        if (!ok) {
            j.cancel = true;
            return false;
        }
    }
    return true;
}

}  // namespace tona::server
