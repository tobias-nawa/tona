// Where this installation lives: a checkout (the program built in work/build/engine/, a justfile and engine/ above it,
// settings in work/tona.conf, server through `just`) or a release installation (<root>/versions/<v>/bin/, settings in
// ~/.config/tona/tona.conf, server is the systemd user unit that runs `tona start --foreground`). This is the one place
// of the client that knows the difference.
#pragma once

#include <string>

#include "core/plan.h"
#include "core/steps.h"

namespace tona::client {

struct Env {
    bool release = false;
    bool macos =
        false;         // release on macOS: no systemd; the server is a background process of `tona start --foreground`
    std::string repo;  // checkout
    std::string root;  // release install root
    std::string conf;  // tona.conf
    std::string models_dir;
    std::string state_dir;  // launch history and the client's own preferences
    int port = 8662;

    std::string url() const;  // http://127.0.0.1:<port>
    install::Target target() const;
    // The program the Agents page runs for an agent: this program, in a release by the link that follows updates
    std::string launcher() const;
    std::string server_log() const;  // macOS: where the background server writes
    std::string server_pid() const;  // macOS: pid of the server that `tona start` started
    // Shell commands that start and stop the server (TONA_START_CMD overrides the start)
    std::string start_command() const;
    std::string restart_command() const;  // release unit; the start limit is reset first
    std::string stop_command() const;
    // Linux release: a unit written by an older tona holds the settings itself; it is written again in the form that
    // runs `tona start --foreground`, so that `tona use` and the settings take effect. Returns an error text, empty if
    // nothing was needed or it worked.
    std::string refresh_unit() const;

    // From the place of the running program and the settings file
    static Env detect();
    static Env for_checkout(const std::string& repo);
    static Env for_release(const std::string& root);
};

// Settings of the installation as the installer reads them (one parser: installer/core/config.h)
install::Settings load_settings(const Env& env);
// Writes all of Settings (a checkout keeps work/tona.conf in the installer's format, a release
// ~/.config/tona/tona.conf; the server reads it at its next start). Returns an error text, empty on success.
std::string save_settings(const Env& env, const install::Settings& s);

}  // namespace tona::client
