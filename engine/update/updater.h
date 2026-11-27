// The updater as tona uses it: one background worker at a time, a snapshot for drawing, nothing blocking in the
// calling (UI) thread.
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "update/install.h"
#include "update/switch.h"

namespace tona::update {

struct UpdateView {
    enum class Phase { Idle, Checking, Downloading, Switching };
    bool enabled = false;    // this tona belongs to a release installation
    bool checks_on = false;  // user turned update checks on
    Phase phase = Phase::Idle;
    bool available = false;  // a newer release is known
    bool security = false;
    std::string version;  // of the available release
    std::string changelog;
    uint64_t download_bytes = 0;  // total
    uint64_t done_bytes = 0;
    std::string message;  // last result or error, shown in the footer
    bool failed = false;
};

class Updater {
public:
    Updater(UpdateEnv env, SwitchConfig sw, bool release_install);
    ~Updater();
    Updater(const Updater&) = delete;
    Updater& operator=(const Updater&) = delete;

    UpdateView view() const;

    // Called often from the UI loop: starts the first check at start-up and then one per 24 hours, if checks are on
    void tick();
    // Turns update checks on or off (persisted in state.json); the UI asks for confirmation before calling it
    void set_checks(bool on);
    // One check now, whatever the switch says (`tona update`); the result is in view() when busy() is false again
    void check_now();
    bool busy() const { return busy_; }  // a check, an install or a roll back runs
    void install();                      // the available release; ignored if none or busy
    void roll_back();                    // back to the previous version

    // True when `exe` (usually /proc/self/exe) is inside <root>/versions/
    static bool runs_from_release(const Layout& layout, const std::string& exe);

private:
    void launch(std::function<void()> job);
    void run_check();
    void run_install();
    void run_roll_back();
    void update_view(const std::function<void(UpdateView&)>& fn);

    UpdateEnv env_;
    SwitchConfig switch_;
    mutable std::mutex mu_;
    UpdateView view_;
    Release available_;
    std::thread worker_;
    std::atomic<bool> busy_{false};
    std::atomic<bool> cancel_{false};
    bool checked_once_ = false;
    std::chrono::steady_clock::time_point last_check_{};
};

// The updater of this process: layout and base from the environment, keys of the release build; restart through
// systemctl on Linux, through `<root>/current/bin/tona start --foreground` and the pid file on macOS
// (script_service.h). `status_url` is the local server whose restarts it manages. `note` receives a problem with the
// configuration.
std::unique_ptr<Updater> make_default_updater(const std::string& status_url, std::string& note);

}  // namespace tona::update
