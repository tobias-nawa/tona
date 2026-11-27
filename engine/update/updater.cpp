#include "update/updater.h"

#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <cstring>
#include <filesystem>

#include "update/activate.h"
#include "update/config.h"
#include "update/fsutil.h"
#include "update/keys.h"
#include "update/script_service.h"
#include "update/state.h"

namespace fs = std::filesystem;

namespace tona::update {

namespace {

constexpr auto kCheckInterval = std::chrono::hours(24);

// Path of this program with its links resolved (through `current` into versions/<v>/bin)
std::string self_exe() {
    std::error_code ec;
#if defined(__APPLE__)
    uint32_t n = 0;
    _NSGetExecutablePath(nullptr, &n);
    std::string p(n, '\0');
    if (_NSGetExecutablePath(p.data(), &n) != 0) return {};
    p.resize(std::strlen(p.c_str()));
    const fs::path real = fs::canonical(p, ec);
#else
    const fs::path real = fs::read_symlink("/proc/self/exe", ec);
#endif
    return ec ? std::string() : real.string();
}

using Phase = UpdateView::Phase;

}  // namespace

Updater::Updater(UpdateEnv env, SwitchConfig sw, bool release_install) : env_(std::move(env)), switch_(std::move(sw)) {
    view_.enabled = release_install;
    if (const StateResult st = load_state(env_.layout); st.ok) view_.checks_on = st.state.checks_enabled;
    env_.fetch.cancel = &cancel_;
    switch_.cancel = &cancel_;
    switch_.say = [this](const std::string& text) { update_view([&](UpdateView& v) { v.message = text; }); };
}

Updater::~Updater() {
    cancel_ = true;
    if (worker_.joinable()) worker_.join();
}

UpdateView Updater::view() const {
    std::lock_guard lk(mu_);
    return view_;
}

void Updater::update_view(const std::function<void(UpdateView&)>& fn) {
    std::lock_guard lk(mu_);
    fn(view_);
}

bool Updater::runs_from_release(const Layout& layout, const std::string& exe) {
    const fs::path versions = layout.versions();
    std::error_code ec;
    const fs::path canonical_root = fs::weakly_canonical(versions, ec);
    if (ec) return false;
    const std::string prefix = canonical_root.string() + "/";
    return exe.starts_with(prefix);
}

void Updater::launch(std::function<void()> job) {
    bool expected = false;
    if (!busy_.compare_exchange_strong(expected, true)) return;
    if (worker_.joinable()) worker_.join();
    worker_ = std::thread([this, job = std::move(job)] {
        try {
            job();
        } catch (const std::exception& e) {
            update_view([&](UpdateView& v) {
                v.phase = Phase::Idle;
                v.failed = true;
                v.message = std::string("update failed: ") + e.what();
            });
        }
        busy_ = false;
    });
}

void Updater::tick() {
    const UpdateView v = view();
    if (!v.enabled || !v.checks_on || busy_) return;
    const auto now = std::chrono::steady_clock::now();
    if (checked_once_ && now - last_check_ < kCheckInterval) return;
    checked_once_ = true;
    last_check_ = now;
    launch([this] { run_check(); });
}

void Updater::check_now() {
    launch([this] { run_check(); });
}

void Updater::set_checks(bool on) {
    StateResult st = load_state(env_.layout);
    if (!st.ok) {
        update_view([&](UpdateView& v) {
            v.failed = true;
            v.message = st.error;
        });
        return;
    }
    st.state.checks_enabled = on;
    std::string err;
    try {
        ensure_private_root(env_.layout);
        err = save_state(env_.layout, st.state);
    } catch (const std::exception& e) {
        err = e.what();
    }
    update_view([&](UpdateView& v) {
        v.failed = !err.empty();
        if (err.empty()) v.checks_on = on;
        v.message = err.empty() ? (on ? "Update checks are on" : "Update checks are off") : err;
        if (!on) v.available = false;
    });
}

void Updater::run_check() {
    update_view([](UpdateView& v) { v.phase = Phase::Checking; });
    const CheckOutcome out = check_for_update(env_);
    std::lock_guard lk(mu_);
    view_.phase = Phase::Idle;
    view_.failed = out.kind == CheckOutcome::Kind::Failed;
    view_.available = out.kind == CheckOutcome::Kind::Available;
    if (view_.failed) view_.message = "update check: " + out.error;
    if (out.kind == CheckOutcome::Kind::UpToDate) view_.message = "Up to date";
    if (view_.available) {
        available_ = out.release;
        view_.version = available_.manifest.version;
        view_.changelog = available_.manifest.changelog;
        view_.security = available_.manifest.security;
        view_.download_bytes = available_.manifest.total_bytes();
        view_.message = "Update " + view_.version + " available";
    }
}

void Updater::install() {
    if (!view().available) return;
    launch([this] { run_install(); });
}

void Updater::run_install() {
    update_view([](UpdateView& v) {
        v.phase = Phase::Downloading;
        v.done_bytes = 0;
        v.failed = false;
    });
    InstallHooks hooks;
    hooks.cancel = &cancel_;
    hooks.progress = [this](uint64_t done, uint64_t total) {
        update_view([&](UpdateView& v) {
            v.done_bytes = done;
            v.download_bytes = total;
        });
    };
    const InstallOutcome in = install_release(env_, available_, hooks);
    if (!in.ok) {
        update_view([&](UpdateView& v) {
            v.phase = Phase::Idle;
            v.failed = true;
            v.message = "install failed, nothing changed: " + in.error;
        });
        return;
    }
    update_view([](UpdateView& v) {
        v.phase = Phase::Switching;
        v.available = false;
    });
    const SwitchOutcome sw = switch_to_current(env_.layout, switch_);
    update_view([&](UpdateView& v) {
        v.phase = Phase::Idle;
        v.failed = sw.kind != SwitchOutcome::Kind::Switched;
        v.message = sw.message;
    });
}

void Updater::roll_back() {
    launch([this] { run_roll_back(); });
}

void Updater::run_roll_back() {
    update_view([](UpdateView& v) { v.phase = Phase::Switching; });
    const SwitchOutcome sw = roll_back_to_previous(env_.layout, switch_);
    update_view([&](UpdateView& v) {
        v.phase = Phase::Idle;
        v.failed = sw.kind != SwitchOutcome::Kind::Switched;
        v.message = sw.message;
    });
}

std::unique_ptr<Updater> make_default_updater(const std::string& status_url, std::string& note) {
    UpdateEnv env;
    env.layout.root = install_root();
    env.keys = release_keys();
    std::string why;
    auto base = resolve_base(why);
    if (!base) {
        note = why;
        return nullptr;
    }
    env.base = *base;
    env.fetch.allow_http_loopback = base->allow_http_loopback;
    SwitchConfig sw;
    sw.status_url = status_url;
#if defined(__APPLE__)
    const ScriptService service = release_script_service(env.layout, status_url);
    sw.restart = [service](std::string& why) { return restart_script_service(service, why); };
    sw.alive = [service] { return script_service_alive(service); };
#else
    sw.alive = [] { return systemd_unit_active(); };
#endif
    const std::string exe = self_exe();
    const bool release = !exe.empty() && !env.layout.root.empty() && Updater::runs_from_release(env.layout, exe);
    return std::make_unique<Updater>(std::move(env), std::move(sw), release);
}

}  // namespace tona::update
