#include "model/progress.h"

#include <atomic>
#include <chrono>
#include <mutex>

namespace tona::progress {
namespace {
std::mutex mu;
std::vector<Phase> list;
bool done_all = false;
std::atomic<int64_t> pf_done{0}, pf_total{0};
const auto t_start = std::chrono::steady_clock::now();
}  // namespace

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); }

void begin(const std::string& name, double total) {
    std::lock_guard lk(mu);
    const double t = now();
    if (!list.empty() && list.back().t1 < 0) list.back().t1 = t;
    list.push_back({name, 0, total, t, -1});
}

void add(double n) {
    std::lock_guard lk(mu);
    if (!list.empty()) list.back().done += n;
}

void set_total(double total) {
    std::lock_guard lk(mu);
    if (!list.empty()) list.back().total = total;
}

void finish() {
    std::lock_guard lk(mu);
    if (!list.empty() && list.back().t1 < 0) list.back().t1 = now();
    done_all = true;
}

std::vector<Phase> phases() {
    std::lock_guard lk(mu);
    return list;
}

bool ready() {
    std::lock_guard lk(mu);
    return done_all;
}

void prefill(int64_t done, int64_t total) {
    pf_total.store(total, std::memory_order_relaxed);
    pf_done.store(done, std::memory_order_relaxed);
}

void prefill_done(int64_t done) { pf_done.store(done, std::memory_order_relaxed); }

void prefill_state(int64_t& done, int64_t& total) {
    done = pf_done.load(std::memory_order_relaxed);
    total = pf_total.load(std::memory_order_relaxed);
}

}  // namespace tona::progress
