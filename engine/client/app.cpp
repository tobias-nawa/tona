#include "client/app.h"

#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>

#include "client/doctor.h"
#include "client/format.h"
#include "client/history.h"
#include "client/screen.h"
#include "client/text.h"
#include "client/view.h"
#include "core/hw.h"

namespace fs = std::filesystem;

namespace tona::client {

namespace {

constexpr int kFrameMs = 250;  // the poller asks four times a second; drawing faster shows nothing new
constexpr auto kSlowEvery = std::chrono::seconds(5);

int port_of(const std::string& url) {
    const size_t colon = url.rfind(':');
    return colon == std::string::npos ? 0 : to_int(url.substr(colon + 1));
}

}  // namespace

Center::Center(const Env& env, CenterOptions opts) : Center(env, std::move(opts), system_control(env)) {}

Center::Center(Env env, CenterOptions opts, std::unique_ptr<Control> control)
    : env_(std::move(env)), opts_(std::move(opts)), control_(std::move(control)) {
    if (opts_.url.empty()) opts_.url = env_.url();
    if (const int port = port_of(opts_.url); port > 0) env_.port = port;
    local_ = opts_.url.contains("127.0.0.1") || opts_.url.contains("localhost");
    std::string note;
    if (local_) updater_ = update::make_default_updater(opts_.url, note);
    ui_.config = load_config(env_);
    ui_.launches = load_launches(env_.state_dir);
    std::error_code ec;
    ui_.agents.project = fs::current_path(ec).string();
    ui_.languages = detect_languages(ui_.agents.project);
    const std::string effort = ui_.config.s.effort;
    ui_.agents.effort = effort == "low" ? 0 : effort == "high" ? 2 : 1;
    if (opts_.tab == "models") ui_.tab = Tab::Models;
    if (opts_.tab == "agents") ui_.tab = Tab::Agents;
    if (opts_.tab == "settings") ui_.tab = Tab::Settings;
}

Center::~Center() {
    quit_ = true;
    if (slow_.joinable()) slow_.join();
    for (std::thread& t : workers_) t.join();
}

void Center::post(std::function<void(UiState&)> change) {
    std::lock_guard lk(post_mu_);
    posted_.push_back(std::move(change));
}

void Center::drain_posted() {
    std::vector<std::function<void(UiState&)>> now;
    {
        std::lock_guard lk(post_mu_);
        now.swap(posted_);
    }
    for (auto& change : now) change(ui_);
}

void Center::background(const std::string& what, std::function<std::string()> work, bool restarts) {
    if (busy_.exchange(true)) {
        ui_.message = tr("msg_busy");
        return;
    }
    ui_.job = what + " …";
    ui_.restarting = restarts;
    workers_.emplace_back([this, what = what, work = std::move(work)] {  // non-const copies: a const one throws on move
        const std::string err = work();
        post([what = what, err = err](UiState& ui) {
            ui.job.clear(), ui.restarting = false;
            if (!err.empty()) ui.message = what + ": " + err;
        });
        busy_ = false;
    });
}

void Center::refresh_slow() {
    const install::Hardware hw = install::detect(env_.target().dir());
    double vram = 0;
    for (const install::Gpu& g : hw.gpus)
        if (!g.integrated) vram = std::max(vram, (double)g.vram_total / 1e9);
    post([ram = (double)hw.ram_total / 1e9, vram](UiState& ui) { ui.ram_total_gb = ram, ui.vram_total_gb = vram; });
    while (!quit_) {
        const std::vector<install::ModelInfo> catalog = installed_catalog(env_);
        const bool down = server_down_ > 0;
        std::vector<Check> checks;
        int free_port = 0;
        if (down) {
            checks = run_checks(gather_facts(env_, load_settings(env_)));
            for (int p = env_.port + 1; p < env_.port + 20 && !free_port; p++)
                if (probe_port(p) == PortState::Free) free_port = p;
        }
        post([catalog, checks, free_port](UiState& ui) {
            ui.catalog = catalog, ui.offer = build_offer();
            if (!checks.empty()) ui.checks = checks, ui.free_port = free_port;
        });
        for (auto t = std::chrono::steady_clock::now(); !quit_ && std::chrono::steady_clock::now() - t < kSlowEvery;)
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void Center::refresh_update() {
    UpdateInfo& u = ui_.update;
    u = {};
    if (!updater_) return;
    const update::UpdateView v = updater_->view();
    u.switching = v.phase == update::UpdateView::Phase::Switching;
    u.enabled = v.enabled, u.checks_on = v.checks_on, u.available = v.available, u.security = v.security;
    u.checking = v.phase == update::UpdateView::Phase::Checking;
    u.downloading = v.phase == update::UpdateView::Phase::Downloading;
    u.fraction = v.download_bytes ? (double)v.done_bytes / (double)v.download_bytes : 0;
    u.version = v.version, u.message = v.message, u.failed = v.failed;
    u.changelog = v.changelog, u.download_mb = (double)v.download_bytes / 1e6;
}

void Center::refresh_loaded() {
    ui_.loaded_id = model_id_of_file(ui_.catalog, ui_.server.model_file);
    server_down_ = ui_.server.up ? 0 : 1;
    if (ui_.switching_to.empty() || !ui_.server.ready) return;
    if (ui_.loaded_id == ui_.switching_to) ui_.switching_from.clear(), ui_.switching_to.clear();
}

void Center::tick() {
    winsize ws{};
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
    ui_.width = ws.ws_col ? ws.ws_col : 80, ui_.height = ws.ws_row ? ws.ws_row : 24;
    ui_.now = (long long)std::time(nullptr);
    fill_from_feed(ui_, sh_);
    drain_posted();
    refresh_update();
    refresh_loaded();
    ui_.show_log = show_log_;
}

void Center::draw() {
    const std::string frame = frame_text(render(ui_), !no_color_requested());
    (void)!write(STDOUT_FILENO, frame.data(), frame.size());
}

void Center::read_keys() {
    pollfd pfd{STDIN_FILENO, POLLIN, 0};
    if (::poll(&pfd, 1, kFrameMs) <= 0) return;
    char buf[64];
    const ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
    if (n <= 0) return;
    for (const Key& key : parse_keys(std::string(buf, (size_t)n))) {
        // On the too-small page only q works: nothing else is visible
        if (ui_.width < kMinWidth || ui_.height < kMinHeight) {
            if (key.code == KeyCode::Char && key.ch == 'q') apply({ActionKind::Quit, ""});
            continue;
        }
        ui_.message.clear();
        apply(handle_key(ui_, key));
    }
}

void Center::apply(const Action& a) {
    switch (a.kind) {
        case ActionKind::None:
            break;
        case ActionKind::Quit:
            do_quit(false, a.arg);
            break;
        case ActionKind::QuitAndStop:
            do_quit(true, a.arg);
            break;
        case ActionKind::SwitchModel:
            do_switch(a.arg);
            break;
        case ActionKind::Download:
            do_download(a.arg);
            break;
        case ActionKind::DownloadRecommended:
            if (!ui_.offer.cards.empty()) do_download(ui_.offer.cards.front().id);
            break;
        case ActionKind::Remove:
            background(tr("job_remove"), [this, id = a.arg] { return control_->remove(id); });
            break;
        case ActionKind::StartServer:
            do_start(false);
            break;
        case ActionKind::StartOnFreePort:
            do_start(true);
            break;
        case ActionKind::RestartServer:
            background(tr("job_restart"), [this] { return control_->restart(); }, true);
            break;
        case ActionKind::StopServer:
            background(tr("job_stop"), [this] { return control_->stop(); });
            break;
        default:
            apply_page(a);
            break;
    }
}

void Center::apply_page(const Action& a) {
    switch (a.kind) {
        case ActionKind::ToggleLog:
            show_log_ = !show_log_;
            break;
        case ActionKind::UpdateKey:
            do_update_key(a.arg);
            break;
        case ActionKind::SaveConfig:
            do_save();
            break;
        case ActionKind::ApplyRecommendation:
            do_recommendation();
            break;
        case ActionKind::LaunchAgent:
            do_launch_agent();
            break;
        case ActionKind::RunInstaller:
            exec_.argv = {env_.launcher(), "setup"};
            quit_ = true;
            break;
        case ActionKind::ProjectChanged:
            do_project_changed();
            break;
        default:
            break;
    }
}

int Center::run() {
    const top::PollerConfig cfg{opts_.url, opts_.autostart, local_,
                                &quit_,    &show_log_,      [this] { start_pending_ = true; }};
    std::thread poller([&] { top::poll_status(sh_, cfg); });
    slow_ = std::thread([this] { refresh_slow(); });
    top::term_setup();
    while (!quit_) {
        tick();
        draw();
        if (updater_) updater_->tick();
        if (start_pending_.exchange(false)) do_start(false);  // set by the poller when --start found no server
        read_keys();
    }
    top::term_restore();
    poller.join();
    if (stop_on_exit_) {
        std::puts(tr("job_stop").c_str());
        if (const std::string err = control_->stop(); !err.empty()) std::fprintf(stderr, "%s\n", err.c_str());
    }
    if (!exec_.argv.empty()) {
        for (const auto& [k, v] : exec_.env) ::setenv(k.c_str(), v.c_str(), 1);
        if (!exec_.chdir.empty() && ::chdir(exec_.chdir.c_str()) != 0) return 1;
        std::vector<char*> argv(exec_.argv.size());
        std::transform(exec_.argv.begin(), exec_.argv.end(), argv.begin(), [](std::string& a) { return a.data(); });
        argv.push_back(nullptr);
        ::execvp(argv[0], argv.data());
        std::perror(exec_.argv[0].c_str());
        return 127;
    }
    return 0;
}

}  // namespace tona::client
