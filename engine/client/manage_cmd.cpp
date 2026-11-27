#include "client/manage_cmd.h"

#include <filesystem>
#include <ostream>

#include "core/autostart.h"
#include "core/module.h"
#include "core/shell_path.h"
#include "core/sys.h"
#include "core/unit.h"
#include "update/manifest.h"

namespace fs = std::filesystem;

namespace tona::client {

namespace {

// Never remove something that is not an installation of ours: a wrong TONA_INSTALL_DIR (the home folder, a project)
// must not turn `tona uninstall` into a delete of the user's files
bool looks_like_installation(const fs::path& root) {
    std::error_code ec;
    return !root.empty() && root.has_relative_path() && root != fs::path(install::home()) &&
           fs::is_directory(root / "versions", ec);
}

// The link of the command `tona` that setup made, only if it points into this installation
void remove_command_link(const fs::path& root, std::ostream& out) {
    const fs::path link = fs::path(install::home()) / ".local/bin/tona";
    std::error_code ec;
    if (!fs::is_symlink(link, ec)) return;
    const fs::path target = fs::read_symlink(link, ec);
    if (ec || !target.string().starts_with(root.string() + "/")) return;
    fs::remove(link, ec);
    if (!ec) out << "removed " << link.string() << "\n";
}

void remove_unit(bool macos, std::ostream& out) {
    if (macos) return;  // the LaunchAgent went with set_autostart(off)
    const fs::path unit =
        fs::path(install::home()) / ".config/systemd/user" / (std::string(install::kUnitName) + ".service");
    std::error_code ec;
    if (!fs::exists(unit, ec)) return;
    fs::remove(unit, ec);
    install::run("systemctl --user daemon-reload 2>/dev/null");
    out << "removed " << unit.string() << "\n";
}

// Everything in the install directory except the models; with `models` also those and then the directory itself
std::string remove_installation(const fs::path& root, bool models) {
    std::error_code ec;
    for (const fs::directory_entry& e : fs::directory_iterator(root, ec)) {
        if (e.path().filename() == "models" && !models) continue;
        fs::remove_all(e.path(), ec);
        if (ec) return e.path().string() + ": " + ec.message();
    }
    if (models) fs::remove(root, ec);  // only when empty
    return {};
}

void remove_settings(const Env& env, std::ostream& out) {
    std::error_code ec;
    const fs::path dir = fs::path(env.conf).parent_path();  // ~/.config/tona: tona.conf and the API key
    for (const fs::path& p : {dir, fs::path(env.state_dir)})
        if (fs::exists(p, ec)) fs::remove_all(p, ec), out << "removed " << p.string() << "\n";
}

bool take_flag(std::vector<std::string>& args, const std::string& flag) {
    for (auto it = args.begin(); it != args.end(); ++it)
        if (*it == flag) return args.erase(it), true;
    return false;
}

}  // namespace

int run_uninstall(const Env& env, Control& control, const std::vector<std::string>& args_in, Io& io) {
    std::vector<std::string> args = args_in;
    const bool yes = take_flag(args, "--yes"), with_models = take_flag(args, "--models");
    if (!args.empty()) return *io.err << "usage: tona uninstall [--yes] [--models]\n", kExitUsage;
    const fs::path root = env.root;
    if (!env.release) return *io.err << "tona: uninstall is for a release installation\n", kExitFailed;
    if (!looks_like_installation(root))
        return *io.err << "tona: " << env.root << " is not a Tona installation (no versions/ folder)\n", kExitFailed;
    const std::string question = "Stop the server and remove Tona (" + env.root +
                                 ", the service, the settings and the API key in " +
                                 fs::path(env.conf).parent_path().string() + ")? The models stay.";
    if (!yes) {
        if (!io.ask) return *io.err << "tona: " << question << " Run it with --yes to answer yes.\n", kExitNeedsConfirm;
        if (!io.ask(question)) return kExitOk;
    }
    bool models = with_models;
    std::error_code ec;
    if (!models && io.ask && fs::exists(fs::path(env.models_dir), ec))
        models = io.ask("Also remove the models (" + env.models_dir + ")?");
    if (const std::string err = control.stop(); !err.empty())
        *io.out << "server: " << err << "\n";  // nothing to stop is fine
    const std::string off = install::set_autostart(env.macos, env.root, false, false, [&](const std::string&) {});
    if (!off.empty()) *io.out << "autostart: " << off << "\n";
    remove_unit(env.macos, *io.out);
    remove_command_link(root, *io.out);
    install::remove_from_path(install::current_path_env(), [&](const std::string& l) { *io.out << l << "\n"; });
    remove_settings(env, *io.out);
    if (const std::string err = remove_installation(root, models); !err.empty())
        return *io.err << "tona: " << err << "\n", kExitFailed;
    *io.out << "removed " << env.root << (models ? " with the models" : ", the models stay in " + env.models_dir)
            << "\n";
    return kExitOk;
}

std::string version_of_exe(const std::string& exe) {
    const std::string marker = "/versions/";
    const size_t at = exe.rfind(marker);
    if (at == std::string::npos) return "dev";
    const size_t start = at + marker.size(), end = exe.find('/', start);
    return end == std::string::npos ? "dev" : exe.substr(start, end - start);
}

std::string version_line(const std::string& exe) {
#ifdef __APPLE__
    const std::string backend = install::metal_apple_module().name();
#else
    const std::string backend = install::hip_gfx1200_module().name();
#endif
    return "tona " + version_of_exe(exe) + " " + std::string(update::kPlatform) + " " + backend + "\n";
}

}  // namespace tona::client
