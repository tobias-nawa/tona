// The control centre as a program: the terminal loop that joins the feed, the state, the keys and the views, and runs
// the actions the keys ask for (in background threads: nothing here blocks the screen).
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "client/control.h"
#include "client/env.h"
#include "client/feed.h"
#include "client/ui.h"
#include "update/updater.h"

namespace tona::client {

struct CenterOptions {
    std::string url;         // server to watch (default: the port of the settings)
    bool autostart = false;  // start the server if none answers (just tona / tona without a running server)
    std::string tab;         // first page: "start", "models", "agents", "settings"
};

// What is left to do after the screen is gone: run a program in place of this one (an agent or the installer)
struct ExecRequest {
    std::vector<std::string> argv;
    std::string chdir;
    std::vector<std::pair<std::string, std::string>> env;
};

class Center {
public:
    Center(const Env& env, CenterOptions opts);
    // With another way to start and stop the server (the tests pass a fake)
    Center(Env env, CenterOptions opts, std::unique_ptr<Control> control);
    ~Center();
    Center(const Center&) = delete;
    Center& operator=(const Center&) = delete;

    // Runs until the user closes it; returns the exit code. A requested program is started after the terminal is back.
    int run();

private:
    friend struct CenterProbe;  // tests/test_client_center.cpp: drives one action without a terminal
    void tick();
    void draw();
    void read_keys();
    void apply(const Action& action);
    void apply_page(const Action& action);  // the actions that are not about the model or the server
    void post(std::function<void(UiState&)> change);
    void background(const std::string& what, std::function<std::string()> work, bool restarts = false);
    void drain_posted();
    void refresh_slow();
    void refresh_update();
    void refresh_loaded();

    // actions (app_actions.cpp)
    void do_switch(const std::string& id);
    void do_download(const std::string& id);
    void do_start(bool free_port);
    void do_update_key(const std::string& key);
    void do_launch_agent();
    void do_recommendation();
    void do_save();
    void do_quit(bool stop_server, const std::string& remember);
    void do_project_changed();

    Env env_;
    CenterOptions opts_;
    UiState ui_;
    top::Shared sh_;
    std::atomic<bool> quit_{false};
    std::atomic<bool> show_log_{false};
    std::unique_ptr<Control> control_;
    std::vector<std::thread> workers_;
    std::mutex post_mu_;
    std::vector<std::function<void(UiState&)>> posted_;
    std::atomic<bool> busy_{false};
    std::atomic<bool> start_pending_{false};
    std::atomic<int> server_down_{0};
    std::thread slow_;
    ExecRequest exec_;
    bool local_ = true;
    bool stop_on_exit_ = false;
    top::KeyConfirm confirm_;
    std::unique_ptr<update::Updater> updater_;  // after sh_ and quit_: its threads never outlive them
};

}  // namespace tona::client
