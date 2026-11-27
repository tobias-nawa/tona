#include "client/agent_server.h"

#include <chrono>
#include <ostream>
#include <thread>

#include "core/sys.h"
#include "third_party/httplib.h"
#include "third_party/nlohmann/json.hpp"

namespace tona::client {

namespace {

using nlohmann::json;

// True and the body for a 2xx answer
bool get(int port, const char* path, const httplib::Headers& headers, int timeout_s, std::string& body) {
    httplib::Client c("http://127.0.0.1:" + std::to_string(port));
    c.set_connection_timeout(timeout_s, 0);
    c.set_read_timeout(timeout_s, 0);
    const auto res = c.Get(path, headers);
    if (!res || res->status < 200 || res->status > 299) return false;
    body = res->body;
    return true;
}

json get_json(int port, const char* path, const httplib::Headers& headers = {}, int timeout_s = 1) {
    std::string body;
    if (!get(port, path, headers, timeout_s, body)) return json();
    return json::parse(body, nullptr, false);
}

std::string loading_line(const json& status) {
    std::string name = "\xe2\x80\xa6";  // …
    std::string percent;
    if (status.contains("phases") && status["phases"].is_array() && !status["phases"].empty()) {
        const json& p = status["phases"].back();
        if (p.is_object() && p.value("name", json()).is_string()) name = p["name"].get<std::string>();
        const double total = p.is_object() ? p.value("total", 0.0) : 0.0;
        if (total > 0) percent = " " + std::to_string((int)(100 * p.value("done", 0.0) / total)) + " %";
    }
    return "  loading: " + name + percent;
}

// The journal of the service says why it did not come up
void print_journal(std::ostream& err) {
    err << "\r\033[Ktona-server did not come up. Log:\n";
    err << install::run("journalctl --user -u tona-server -n 15 -o cat --no-pager 2>&1").out;
}

}  // namespace

bool server_answers(int port) {
    std::string body;
    return get(port, "/status", {}, 1, body);
}

std::string server_engine_state(int port) {
    const json status = get_json(port, "/status");
    if (!status.is_object() || !status.contains("engine") || !status["engine"].is_object()) return {};
    const json& state = status["engine"].value("state", json());
    return state.is_string() ? state.get<std::string>() : std::string();
}

int64_t server_context_size(int port) {
    const json health = get_json(port, "/health", {}, 2);
    return health.is_object() && health.value("n_ctx", json()).is_number() ? health["n_ctx"].get<int64_t>() : 0;
}

void server_model_names(int port, const std::string& key, std::string& id, std::string& name) {
    const json models = get_json(port, "/v1/models", {{"Authorization", "Bearer " + key}}, 2);
    if (!models.is_object() || !models.value("data", json()).is_array() || models["data"].empty()) return;
    const json& first = models["data"][0];
    if (!first.is_object() || !first.value("id", json()).is_string() || first["id"].get<std::string>().empty()) return;
    id = first["id"].get<std::string>();
    const json& display = first.value("display_name", json());
    name = display.is_string() && !display.get<std::string>().empty() ? display.get<std::string>() : id;
}

bool wait_until_ready(int port, int wait_s, std::ostream& out, std::ostream& err) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const json status = get_json(port, "/status");
        if (status.is_object()) {
            if (status.value("ready", false)) return out << "\r\033[K" << std::flush, true;
            const json& e = status.value("error", json());
            if (e.is_string() && !e.get<std::string>().empty()) {
                err << "\r\033[KError while loading: " << e.get<std::string>() << "\n";
                return false;
            }
            out << "\r\033[K" << loading_line(status) << std::flush;
        } else {
            const bool failed = install::run("systemctl --user -q is-failed tona-server 2>/dev/null").status == 0;
            const auto waited = std::chrono::steady_clock::now() - t0;
            if (failed || waited > std::chrono::seconds(wait_s)) return print_journal(err), false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

}  // namespace tona::client
