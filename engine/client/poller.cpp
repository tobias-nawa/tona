// Data source of tona: polls GET /status of the server and derives the decode-rate history.
#include <algorithm>
#include <chrono>
#include <deque>
#include <format>
#include <map>
#include <set>
#include <thread>

#include "client/term.h"
#include "third_party/httplib.h"

namespace tona::top {

namespace {

using Clock = std::chrono::steady_clock;
using Parts = std::vector<std::pair<int, float>>;

constexpr int kHistoryPolls = 600;  // 2.5 min at four polls per second
constexpr double kRateWindowSeconds = 1.0;

class StatusPoller {
public:
    StatusPoller(Shared& sh, const PollerConfig& cfg) : sh_(sh), cfg_(cfg), client_(cfg.url) {
        client_.set_connection_timeout(0, 300000);
        client_.set_read_timeout(2, 0);
    }

    void run() {
        const auto t0 = Clock::now();
        auto next_log = Clock::now();
        bool start_checked = !cfg_.autostart;
        while (!*cfg_.quit) {
            const double t = cfg_.now ? cfg_.now() : std::chrono::duration<double>(Clock::now() - t0).count();
            const bool up = poll_once(t);
            if (!start_checked) {
                start_checked = true;
                if (!up && cfg_.local) cfg_.start();
            }
            if (*cfg_.show_log && Clock::now() >= next_log) {
                next_log = Clock::now() + std::chrono::seconds(1);
                refresh_log();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }

private:
    // One request; returns whether the server answered.
    bool poll_once(double t) {
        auto res = client_.Get("/status");
        std::lock_guard lk(sh_.mu);
        store_status(res);
        sh_.rate.push_back(sample_rates(t));
        // Know the projects from the recent requests too, so the view selection offers them
        if (sh_.up && sh_.status.contains("recent") && sh_.status["recent"].is_array())
            for (const json& x : sh_.status["recent"]) sh_.project(jstr(x, "project"));
        if (sh_.rate.size() > kHistoryPolls) sh_.rate.pop_front();
        return sh_.up;
    }

    void store_status(const httplib::Result& res) {
        sh_.up = res && res->status == 200;
        if (sh_.up) {
            sh_.error.clear();  // the reason of an earlier failure is not the state of this answer
            try {
                sh_.status = json::parse(res->body);
            } catch (const std::exception& e) {
                sh_.up = false, sh_.error = e.what();
            }
        } else {
            sh_.error = res ? std::format("HTTP {}", res->status) : httplib::to_string(res.error());
        }
    }

    // Running requests: engine.active (several conversations); older servers only have engine.
    static std::vector<json> running_requests(const json& engine) {
        std::vector<json> act;
        if (engine.contains("active") && engine["active"].is_array())
            act.assign(engine["active"].begin(), engine["active"].end());
        else
            act.push_back(engine);
        return act;
    }

    // Decode tok/s of one request over the last second
    double decode_rate(int64_t id, double t, double out_tokens) {
        auto& win = windows_[id];
        if (!win.empty() && out_tokens < win.back().second) win.clear();  // new request
        win.emplace_back(t, out_tokens);
        while (win.size() > 1 && t - win.front().first > kRateWindowSeconds) win.pop_front();
        return win.size() > 1 ? (out_tokens - win.front().second) / (t - win.front().first) : 0;
    }

    static void add_rate(Parts& sample, int project, double rate) {
        auto it = std::find_if(sample.begin(), sample.end(), [&](const auto& x) { return x.first == project; });
        if (it != sample.end())
            it->second += (float)rate;
        else
            sample.emplace_back(project, (float)rate);
    }

    Parts sample_rates(double t) {
        Parts sample;
        std::set<int64_t> seen;
        if (sh_.up) {
            for (const json& a : running_requests(sh_.status["engine"])) {
                if (jstr(a, "state") != "decode") continue;
                const int64_t id = (int64_t)jnum(a, "id");
                const int project = sh_.project(a.contains("origin") ? jstr(a["origin"], "project") : "");
                seen.insert(id);
                const double rate = decode_rate(id, t, jnum(a, "out_tokens"));
                if (rate > 0) add_rate(sample, project, rate);
            }
        }
        std::erase_if(windows_, [&](const auto& w) { return !seen.contains(w.first); });
        return sample;
    }

    void refresh_log() {
        std::string all;
        tona::top::run("journalctl --user -u tona-server -n 60 -o cat --no-pager", &all);
        std::vector<std::string> lines;
        for (size_t a = 0, b; a < all.size(); a = b + 1) {
            b = all.find('\n', a);
            if (b == std::string::npos) b = all.size();
            if (b > a) lines.push_back(all.substr(a, b - a));
        }
        std::lock_guard lk(sh_.mu);
        sh_.log = std::move(lines);
    }

    Shared& sh_;
    const PollerConfig& cfg_;
    httplib::Client client_;
    // per running request (id): (time, decode tokens) of the last second
    std::map<int64_t, std::deque<std::pair<double, double>>> windows_;
};

}  // namespace

void poll_status(Shared& sh, const PollerConfig& cfg) { StatusPoller(sh, cfg).run(); }

}  // namespace tona::top
