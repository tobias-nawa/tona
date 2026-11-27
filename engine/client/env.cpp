#include "client/env.h"

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>

#include "backend/env.h"
#include "client/format.h"
#include "core/config.h"
#include "core/release_steps.h"
#include "core/sys.h"
#include "core/unit.h"
#include "update/config.h"
#include "update/updater.h"

namespace fs = std::filesystem;

namespace tona::client {

namespace {

// The repository above the program: a justfile and engine/CMakeLists.txt. Both are required so that a stray justfile in
// a parent directory of an installation is never taken for a checkout, and a release installation never looks for one:
// a justfile higher up in the file system would get `just` to run code that is not ours.
std::string find_repo(const std::string& exe_path) {
    const fs::path exe(exe_path);
    update::Layout layout;
    layout.root = update::install_root();
    if (!layout.root.empty() && update::Updater::runs_from_release(layout, exe_path)) return {};
    fs::path dir = exe.parent_path();
    for (int up = 0; !dir.empty() && up < 8; ++up, dir = dir.parent_path()) {
        std::error_code ec;
        if (fs::exists(dir / "justfile", ec) && fs::exists(dir / "engine" / "CMakeLists.txt", ec)) return dir.string();
        if (dir == dir.root_path()) break;
    }
    return {};
}

int port_from(const std::string& conf) {
    if (const char* p = tona::env::text("TONA_PORT")) return to_int(p);
    const install::Values v = install::read_values(conf);
    if (const auto it = v.find("TONA_PORT"); it != v.end() && to_int(it->second) > 0) return to_int(it->second);
    return 8662;
}

}  // namespace

Env Env::for_checkout(const std::string& repo) {
    Env e;
    e.repo = repo;
    e.conf = install::conf_path(repo);
    e.models_dir = repo + "/work/models";
    e.state_dir = repo + "/work/tona";
    e.port = port_from(e.conf);
    return e;
}

Env Env::for_release(const std::string& root) {
    Env e;
    e.release = true;
#ifdef __APPLE__
    e.macos = true;
#endif
    e.root = root;
    e.conf = install::home() + "/.config/tona/tona.conf";
    e.models_dir = root + "/models";
    e.state_dir = update::state_dir();
    e.port = port_from(e.conf);
    return e;
}

Env Env::detect() {
    const std::string exe = install::self_exe();
    if (const std::string repo = exe.empty() ? "" : find_repo(exe); !repo.empty()) return for_checkout(repo);
    const char* dir = std::getenv("TONA_INSTALL_DIR");
    return for_release(dir && *dir ? dir : update::install_root());
}

std::string Env::url() const { return std::format("http://127.0.0.1:{}", port); }

install::Target Env::target() const {
    if (release) return install::release_target(root);
    install::Target t;
    t.repo = repo;
    return t;
}

std::string Env::launcher() const { return release ? root + "/current/bin/tona" : repo + "/work/build/engine/tona"; }

std::string Env::server_log() const { return state_dir + "/server.log"; }
std::string Env::server_pid() const { return state_dir + "/server.pid"; }

namespace {

// macOS has no systemd: the server runs as a background process of the user, started as the unit starts it
// (`tona start --foreground`). That process writes its own pid file, so this start and the LaunchAgent give the same
// state for `tona stop`.
std::string macos_start(const Env& e) {
    return std::format(
        "mkdir -p {0} && if [ -f {1} ] && kill -0 \"$(cat {1})\" 2>/dev/null; then echo 'already running'; "
        "else TONA_INSTALL_DIR={2} nohup {3} start --foreground >>{4} 2>&1 </dev/null & fi",
        install::quote(e.state_dir), install::quote(e.server_pid()), install::quote(e.root),
        install::quote(e.launcher()), install::quote(e.server_log()));
}

}  // namespace

std::string Env::start_command() const {
    if (const char* c = tona::env::text("TONA_START_CMD")) return c;
    if (release && macos) return macos_start(*this);
    if (release) return "systemctl --user reset-failed tona-server 2>/dev/null; systemctl --user start tona-server";
    return std::format("just -f {} port={} start", install::quote(repo + "/justfile"), port);
}

std::string Env::restart_command() const {
    return "systemctl --user reset-failed tona-server 2>/dev/null; systemctl --user restart tona-server";
}

std::string Env::stop_command() const {
    // Only a process that is still a tona-server (or the tona that is about to become one) is ended, and the command
    // returns when it is gone (at most 30 s, then SIGKILL): a pid file left over from before a reboot may name another
    // program by now. Nothing of ours running is not an error: it says "not running" and exits 0.
    if (release && macos)
        return std::format(
            "pid=$(cat {0} 2>/dev/null); name=$(ps -p \"$pid\" -o comm= 2>/dev/null | sed 's#.*/##'); "
            "if [ -n \"$pid\" ] && {{ [ \"$name\" = tona-server ] || [ \"$name\" = tona ]; }}; then kill \"$pid\" "
            "&& {{ i=0; while kill -0 \"$pid\" 2>/dev/null && [ $i -lt 300 ]; do sleep 0.1; i=$((i+1)); done; "
            "kill -9 \"$pid\" 2>/dev/null; rm -f {0}; }}; else rm -f {0}; echo 'not running'; fi",
            install::quote(server_pid()));
    if (release) return "systemctl --user stop tona-server";
    return std::format("just -f {} port={} stop", install::quote(repo + "/justfile"), port);
}

std::string Env::refresh_unit() const {
    if (!release || macos) return {};
    const std::string dir = install::home() + "/.config/systemd/user";
    const fs::path unit = fs::path(dir) / (std::string(install::kUnitName) + ".service");
    std::error_code ec;
    if (!fs::exists(unit, ec)) return {};
    std::ifstream in(unit);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text == install::render_unit(root) || text.contains("start --foreground")) return {};
    if (const std::string err = install::write_unit_file(root, dir); !err.empty()) return err;
    install::run("systemctl --user daemon-reload");
    return {};
}

install::Settings load_settings(const Env& env) {
    install::Settings defaults;
    defaults.port = env.port;
    return install::load(env.conf, defaults);
}

std::string save_settings(const Env& env, const install::Settings& s) {
    if (!env.release) return install::save(s, env.repo) ? "" : env.conf + ": cannot write";
    // The release conf keeps keys that are not Settings (the client's own); only Settings' keys are replaced. The unit
    // and the LaunchAgent hold no settings, so nothing else is written.
    return install::set_values(env.conf, install::parse_values(install::render_release_conf(s)));
}

}  // namespace tona::client
