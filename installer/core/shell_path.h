// Puts the folder of the command tona (~/.local/bin) on PATH: one marked block in the profile of each shell the user
// has. The block is added once, found again by its markers, and removed by `tona uninstall`. Nothing else in a profile
// is read or changed.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "core/steps.h"

namespace tona::install {

// The environment the choice of files depends on, so tests can set it without touching the process environment
struct PathEnv {
    std::string home, shell, zdotdir, xdg_config, path;  // $HOME, $SHELL, $ZDOTDIR, $XDG_CONFIG_HOME, $PATH
    bool macos = false;                                  // Terminal starts login shells: bash reads ~/.bash_profile
};
PathEnv current_path_env();

// One profile that gets the block
struct ProfileFile {
    std::string shell;  // zsh, bash, fish or sh (every other POSIX shell)
    std::filesystem::path file;
};

// The profiles to cover: the user's login shell ($SHELL) and every other shell of zsh, bash and fish that has its file
// or folder already
std::vector<ProfileFile> profile_files(const PathEnv& env);

bool on_path(const std::string& path_env, const std::string& dir);

// Adds the block to every file of profile_files that lacks it, one line per file to `out`. A file that cannot be
// written is skipped and returned, so the caller can show the hint line for its shell.
std::vector<ProfileFile> add_to_path(const PathEnv& env, const std::string& dir, const LineFn& out);

// Removes the block from every profile it can be in (also those of shells that are no longer the user's); returns the
// number of files changed
int remove_from_path(const PathEnv& env, const LineFn& out);

// The line that adds `dir` to PATH by hand for one shell (bash, zsh, fish, sh)
std::string hint_line(const std::string& shell, const std::string& dir, const std::string& home);

// The header and the lines for bash, zsh and fish; folders below `home` are written with ~ or $HOME
std::vector<std::string> hint_lines(const std::string& dir, const std::string& home);

// What the last page tells: empty when `dir` is on PATH, one line "added ... open a new terminal" when the block is in
// the profile of the user's shell, else the lines to add it by hand
std::vector<std::string> path_advice(const PathEnv& env, const std::string& dir, bool modify_path);

// The plan step "Command tona: add <dir> to PATH in <files>"; skipped when `dir` is on PATH already
Step path_step(const std::string& dir, const PathEnv& env);

}  // namespace tona::install
