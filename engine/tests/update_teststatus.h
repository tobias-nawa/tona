// A stand-in for tona-server's /status endpoint on 127.0.0.1, for the switch and updater tests
#pragma once

#include <atomic>
#include <string>
#include <thread>

#include "tests/update_testkit.h"

namespace tona::test {

// Stands in for tona-server's /status; the restart hook decides how it behaves after each "restart"
class FakeStatus {
public:
    enum class Mode { Idle, Busy, Loading, Error };

    FakeStatus() {
        svr_.Get("/status", [this](const httplib::Request&, httplib::Response& res) {
            res.set_content(body().dump(), "application/json");
        });
        port_ = svr_.bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { svr_.listen_after_bind(); });
        svr_.wait_until_ready();
    }
    ~FakeStatus() {
        svr_.stop();
        thread_.join();
    }
    FakeStatus(const FakeStatus&) = delete;
    FakeStatus& operator=(const FakeStatus&) = delete;

    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
    void set(Mode m) { mode_ = m; }

private:
    json body() const {
        switch (mode_.load()) {
            case Mode::Idle:
                return {{"ready", true}, {"engine", {{"state", "idle"}, {"queue", 0}, {"active", json::array()}}}};
            case Mode::Busy:
                return {{"ready", true}, {"engine", {{"state", "decode"}, {"queue", 0}, {"active", json::array({1})}}}};
            case Mode::Loading:
                return {{"ready", false}};
            case Mode::Error:
                return {{"ready", true}, {"error", "load failed"}};
        }
        return {};
    }

    httplib::Server svr_;
    std::thread thread_;
    int port_ = 0;
    std::atomic<Mode> mode_{Mode::Idle};
};

}  // namespace tona::test
