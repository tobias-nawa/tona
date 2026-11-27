#include "server/hold_controller.h"

#include "backend/env.h"

namespace tona::server {

namespace {
double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
}  // namespace

std::unique_ptr<HoldController> HoldController::from_env(backend::Backend& b) {
    const int64_t mode = tona::env::integer("TONA_CLOCK_HOLD", 0);
    if (mode != 1 && mode != 2) return nullptr;
    // 20 s of hold for each ramp saved (about 540 J for 0.2 s of prefill): the exchange rate of the rule, see
    // hold_policy.h
    return std::make_unique<HoldController>(b, tona::env::number("TONA_CLOCK_HOLD_S_PER_SAVE", 20));
}

HoldController::HoldController(backend::Backend& b, double seconds_per_save)
    : be_(b), per_save_(seconds_per_save), th_([this] { loop(); }) {}

HoldController::~HoldController() {
    {
        std::lock_guard lk(mu_);
        quit_ = true;
        stop_locked();
    }
    cv_.notify_all();
    th_.join();
}

void HoldController::stop_locked() {
    if (!holding_) return;
    holding_ = false;
    tot_.held_s += since(hold_start_);
    be_.clock_hold(false);
}

void HoldController::arrival(bool warm) {
    std::lock_guard lk(mu_);
    if (idle_ && !warm) window_.add(since(idle_since_));
    if (holding_ && !warm) tot_.covered++;
    idle_ = false;
    stop_locked();
    cv_.notify_all();
}

void HoldController::idle() {
    {
        std::lock_guard lk(mu_);
        idle_ = true;
        idle_since_ = Clock::now();
        seq_++;
    }
    cv_.notify_all();
}

HoldController::Totals HoldController::totals() const {
    std::lock_guard lk(mu_);
    Totals t = tot_;
    if (holding_) t.held_s += since(hold_start_);
    return t;
}

// Thread: for each idle period one hold of the length the rule gives, ended early by the next arrival
void HoldController::loop() {
    std::unique_lock lk(mu_);
    uint64_t handled = 0;
    while (true) {
        cv_.wait(lk, [&] { return quit_ || seq_ != handled; });
        if (quit_) return;
        handled = seq_;
        const auto start = idle_since_;
        const auto over = [&] { return quit_ || !idle_ || seq_ != handled; };
        if (!idle_) continue;
        const double t = hold_seconds(window_.gaps(), per_save_);
        tot_.last_hold_s = t;
        if (t <= 0) continue;  // nothing to go on, or no pause worth covering: the next request needs no stop
        holding_ = true;
        hold_start_ = Clock::now();
        tot_.holds++;
        be_.clock_hold(true);
        cv_.wait_until(lk, start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(t)), over);
        stop_locked();  // the hold ran out; an arrival has stopped it already
    }
}

}  // namespace tona::server
