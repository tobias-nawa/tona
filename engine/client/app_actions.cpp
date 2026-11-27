// What the centre does when a key asks for something: each action runs in a background thread when it can take long, so
// the screen keeps drawing
#include <filesystem>

#include "client/app.h"
#include "client/history.h"
#include "client/text.h"
#include "core/catalog.h"
#include "core/hw.h"
#include "core/module.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::client {

void Center::do_quit(bool stop_server, const std::string& remember) {
    if (!remember.empty()) {
        set_value(ui_.config, "close", stop_server ? "stop" : "keep");
        if (const std::string err = save_config(env_, ui_.config); !err.empty())
            std::fprintf(stderr, "%s\n", err.c_str());
    }
    stop_on_exit_ = stop_server;
    quit_ = true;
}

void Center::do_switch(const std::string& id) {
    const bool cancel = id == ui_.switching_from;  // Esc while loading: the old model comes back
    if (!cancel && id == ui_.loaded_id) {
        ui_.message = tr("msg_already_loaded");
        return;
    }
    if (busy_) {  // another job runs: start nothing and leave the state as it is
        ui_.message = tr("msg_busy");
        return;
    }
    if (cancel)
        ui_.switching_from.clear(), ui_.switching_to.clear();
    else
        ui_.switching_from = ui_.loaded_id, ui_.switching_to = id;
    background(
        tr("job_switch"),
        [this, id = id, catalog = ui_.catalog] {
            const std::string err = switch_model(env_, *control_, catalog, id);
            if (!err.empty()) post([](UiState& ui) { ui.switching_from.clear(), ui.switching_to.clear(); });
            return err;
        },
        true);
}

void Center::do_download(const std::string& id) {
    const std::string drafter = install::drafter_entry_id(ui_.config.s.drafter, id);
    background(tr("job_download"), [this, id = id, drafter = drafter] {
        return control_->download(
            id, drafter, [this](const std::string& line) { post([line = line](UiState& ui) { ui.job = line; }); });
    });
}

void Center::do_start(bool free_port) {
    if (!free_port) {
        background(tr("job_start"), [this] { return control_->start(); });
        return;
    }
    const int port = ui_.free_port ? ui_.free_port : env_.port + 1;
    set_value(ui_.config, "port", std::to_string(port));
    if (const std::string err = save_config(env_, ui_.config); !err.empty()) {
        ui_.message = err;
        return;
    }
    // The agents read the port from the settings; this centre watches the old one, so it starts again on the new port
    env_.port = port;
    background(tr("job_start"), [this] {
        std::string err = control_->start();
        if (!err.empty()) return err;
        post([this](UiState&) {
            exec_.argv = {install::self_exe(), "--url", env_.url()};
            quit_ = true;
        });
        return std::string();
    });
}

void Center::do_update_key(const std::string& key) {
    ui_.message = top::update_key(updater_.get(), key == "r" ? 'r' : key[0], confirm_);
}

void Center::do_launch_agent() {
    const AgentForm& f = ui_.agents;
    const std::string dir = expand_home(f.project);
    std::error_code ec;
    if (!dir.empty() && !fs::is_directory(dir, ec)) {
        ui_.message = fill(tr("msg_no_folder"), dir);
        return;
    }
    LaunchEntry entry{agent_name(f.agent), f.project, loaded_name(ui_), ui_.now};
    if (const std::string err = record_launch(env_.state_dir, entry); !err.empty())
        std::fprintf(stderr, "%s\n", err.c_str());
    exec_.argv = {env_.launcher(), agent_name(f.agent)};
    exec_.chdir = dir;
    if (!f.all_tools) exec_.env.emplace_back("TONA_TOOLS", "lean");
    exec_.env.emplace_back("TONA_EFFORT", f.effort == 0 ? "low" : f.effort == 2 ? "high" : "medium");
    quit_ = true;
}

void Center::do_recommendation() {
    const install::Hardware h = install::detect(env_.target().dir());
    const install::Module* m = install::find_module(h);
    if (!m) {
        ui_.message = tr("msg_no_recommendation");
        return;
    }
    const install::Settings rec = m->recommend(h);
    install::Settings& s = ui_.config.s;
    s.ram_gb = rec.ram_gb, s.vram_gb = rec.vram_gb, s.seqs = rec.seqs, s.ctx = rec.ctx, s.ctx_auto = rec.ctx_auto,
    s.oea = rec.oea;
    ui_.settings.dirty = true;
    ui_.message = tr("msg_recommendation_set");
}

void Center::do_save() {
    const std::string err = save_config(env_, ui_.config);
    ui_.message = err.empty() ? tr("msg_saved") : err;
    if (err.empty()) ui_.settings.dirty = false;
}

void Center::do_project_changed() { ui_.languages = detect_languages(expand_home(ui_.agents.project)); }

}  // namespace tona::client
