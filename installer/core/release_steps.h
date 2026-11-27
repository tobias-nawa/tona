// The steps of a release installation: no checkout, no just, no toolchain. The signed release comes through the same
// code tona updates with (engine/update/: signature, hashes, layout, `current`); the models come from the signed
// catalog (catalog.h); the systemd user unit starts <root>/current/bin/tona-server (unit.h).
#pragma once

#include <sys/types.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "core/model_fetch.h"
#include "core/shell_path.h"
#include "core/steps.h"
#include "core/unit.h"
#include "update/release.h"

namespace tona::install {

// What the steps talk to. Tests replace it; a real run takes the defaults.
struct ReleaseSources {
    std::optional<update::UpdateEnv> update;  // empty: TONA_UPDATE_BASE or GitHub, the keys of this build
    ModelSource models;
};

std::vector<Step> plan_release_steps(const Hardware& h, const Settings& s, const Target& t, const ReleaseSources& src);

// The unit for these settings: paths below the target, key in ~/.config/tona/api-key
UnitSpec unit_spec(const Settings& s, const Target& t);

// Writes <unit_dir>/tona-server.service (atomic); returns an error text, empty on success. Does not call systemctl.
std::string write_unit_file(const std::string& root, const std::string& unit_dir);

// Settings the command tona reads in a release installation (~/.config/tona/tona.conf): port, effort, key file
std::string render_release_conf(const Settings& s);

// Links <bin_dir>/<name> -> target for each pair; a file that is not a link is left alone. One line per link to `out`.
struct CommandLink {
    std::string name, target;
};
void link_commands(const std::string& bin_dir, const std::vector<CommandLink>& links, const LineFn& out);

// The steps after the installation: put the command's folder on PATH (only when the user said yes and it is missing)
// and, in a release, the update checks. Not part of plan_steps: the guided setup asks on its Finish screen, the
// command line takes --modify-path and --updates.
std::vector<Step> finish_steps(const Settings& s, const Target& t, const PathEnv& env);

// Lines to tell the user how to add bin_dir to PATH by hand (bash, zsh, fish); empty when it is on PATH. Printed when
// the user turned the profile edit off (shell_path.h).
std::vector<std::string> path_hint(const std::string& bin_dir, const std::string& path_env);

// Writes `text` to `path` through a temporary file and a rename, so a reader never sees half of it; returns an error
// text, empty on success
std::string write_atomic(const std::filesystem::path& path, const std::string& text, mode_t mode);

// Creates the key file (mode 0600, 32 random bytes as hex) if it does not exist or is empty
std::string ensure_api_key(const std::string& path);

}  // namespace tona::install
