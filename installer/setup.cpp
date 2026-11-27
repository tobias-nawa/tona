// `tona setup`: guided setup of Tona. Detects the hardware, asks what the user wants to do, recommends one model per
// task with its measured speed, checks packages and fetches models (SHA-256). In a checkout (developers) it then
// builds, measures a trial run and, on request, sets up the service and the command tona. Without a checkout (users)
// it fetches and verifies the signed release, installs it into ~/.local/share/tona and sets up the systemd user unit
// that starts it. Every choice of the pages also works on the command line (headless.cpp).
#include "setup.h"

#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "core/catalog.h"
#include "core/cli_args.h"
#include "core/exit_codes.h"
#include "core/sys.h"
#include "headless.h"

namespace fs = std::filesystem;

int run_tui(const tona::install::Options& o, const tona::install::Target& target);     // tui.cpp
int dump_pages(const tona::install::Options& o, const tona::install::Target& target);  // tui_dump.cpp

namespace tona::install {

namespace {

// The repository: the first folder with toolchain/models.toml above the binary or the current folder.
std::string find_repo() {
    for (const fs::path& start : {fs::path(self_exe()).parent_path(), fs::current_path()})
        for (fs::path p = start; !p.empty() && p != p.root_path(); p = p.parent_path())
            if (fs::exists(p / "toolchain/models.toml")) return p.string();
    return {};
}

// Checkout (developer) or release installation: a checkout is used only if one is found and --release is not given
Target make_target(const Options& o) {
    if (o.release || o.repo.empty()) {
        Target t = release_target(o.install_dir);
        if (!o.models_dir.empty()) t.models_dir = fs::absolute(o.models_dir).string();
        if (!o.cache_dir.empty()) t.cache_dir = fs::absolute(o.cache_dir).string();
        t.root = fs::absolute(t.root).string();
        if (o.models_dir.empty()) t.models_dir = t.root + "/models";
        if (o.cache_dir.empty()) t.cache_dir = t.root + "/prompt-cache";
        return t;
    }
    Target t;
    t.repo = fs::canonical(o.repo).string();
    return t;
}

bool has_terminal() { return isatty(STDIN_FILENO) && isatty(STDOUT_FILENO); }

// The folders the options name must exist; returns an exit code, 0 when they do
int check_paths(Options& o) {
    if (o.repo.empty() && !o.release) o.repo = find_repo();
    if (!o.repo.empty() && !fs::exists(fs::path(o.repo) / "toolchain/models.toml")) {
        std::fprintf(stderr, "Not an engine repository (no toolchain/models.toml): %s\n", o.repo.c_str());
        return kExitUsage;
    }
    if (o.model_from.empty()) return kExitOk;
    if (!fs::is_directory(o.model_from)) {
        std::fprintf(stderr, "--model-from: not a directory: %s\n", o.model_from.c_str());
        return kExitUsage;
    }
    o.model_from = fs::canonical(o.model_from).string();  // steps run in the repo, so the path must be absolute
    return kExitOk;
}

}  // namespace

int run_setup(const std::vector<std::string>& args) {
    std::vector<const char*> argv = {"tona setup"};
    for (const std::string& a : args) argv.push_back(a.c_str());
    Options o;
    std::string error;
    if (!parse_args(static_cast<int>(argv.size()), argv.data(), o, error)) {
        std::fprintf(stderr, "tona setup: %s\n\n%s", error.c_str(), kUsage);
        return kExitUsage;
    }
    if (o.help || o.help_dev) return std::fputs(o.help_dev ? kUsageDev : kUsage, stdout), kExitOk;
    if (const int bad = check_paths(o)) return bad;
    const Target t = make_target(o);
    if (o.dump_pages) return dump_pages(o, t);
    // Plans change nothing: --print, --json and --dry-run need no question and no confirmation
    if (o.print || o.json || o.dry) return run_headless(o, t, true);
    if (o.yes || !o.conf.empty()) return run_headless(o, t, false);
    if (!has_terminal()) {
        std::fprintf(stderr,
                     "no terminal: use tona setup --yes (with --task or --models), --dry-run or --config FILE\n");
        return kExitUsage;
    }
    return run_tui(o, t);
}

}  // namespace tona::install
