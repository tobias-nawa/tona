// Installation steps. In a checkout almost every step is a shell command in the repository, so --dry-run shows exactly
// what will happen and each step can be repeated by hand. The commands use the engine's scripts (scripts/, tools/,
// just); the installer reimplements none of them. It writes only its own files directly (work/tona.conf, the link for
// the command tona). A release installation has no checkout: its steps run inside the installer (release_steps.h).
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/hw.h"
#include "core/plan.h"

namespace tona::install {

using LineFn = std::function<void(const std::string&)>;

struct Step {
    std::string title;                         // for humans
    std::string cmd;                           // shell command run in the repo (a description if action is set)
    bool done = false;                         // already done (skipped)
    std::function<int(const LineFn&)> action;  // runs inside the installer instead of cmd
};

// Where an installation goes. Checkout (developers): the engine repository, everything below work/. Release (users): an
// install root laid out as in docs/maintain/release.md, with the models, the expert profile and the prompt cache below
// it.
struct Target {
    bool release = false;
    bool macos = false;                 // release on macOS: no systemd unit (docs/use/macos.md)
    std::string repo;                   // checkout: the engine repository (steps run in it)
    std::string root;                   // release: install root, default ~/.local/share/tona
    std::string models_dir, cache_dir;  // release: default <root>/models and <root>/prompt-cache
    std::string unit_dir, bin_dir;      // release: ~/.config/systemd/user and ~/.local/bin
    // Base for hardware detection (free space, drive): the repository or the install root
    const std::string& dir() const { return release ? root : repo; }
};

// Release target with every default filled in; an empty `root` means the default install root
Target release_target(const std::string& root);

std::vector<Step> plan_steps(const Hardware& h, const Settings& s, const Target& t);
// Runs one step; on_line gets the output line by line. Returns the command's exit status.
int run_step(const Step& st, const std::string& repo, const LineFn& on_line);

}  // namespace tona::install
