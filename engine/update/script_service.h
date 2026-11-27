// The server as a background process of the user, started as `tona start` starts it (`<root>/current/bin/tona start
// --foreground`): the restart of the updater on macOS, which has no systemd. It uses the same files as `tona start` and
// `tona stop` (client/env.cpp): the pid in <state>/server.pid, the output appended to <state>/server.log.
#pragma once

#include <string>

#include "update/layout.h"

namespace tona::update {

struct ScriptService {
    std::string program;      // <root>/current/bin/tona; it is run with `start --foreground`
    std::string install_dir;  // <root>, passed as TONA_INSTALL_DIR
    std::string pid_file;     // <state>/server.pid
    std::string log_file;     // <state>/server.log
    std::string status_url;   // the server's address; nothing may answer there between stop and start
    int stop_seconds = 30;    // wait for the old server to exit after SIGTERM, then SIGKILL
};

// The service of a release installation: <root>/current/bin/tona and the state directory of tona
ScriptService release_script_service(const Layout& layout, const std::string& status_url);

// Ends the server named by the pid file (only if that process is still a tona-server or the tona that becomes one: a
// pid file from before a reboot may name another program by now), waits until it is gone and nothing answers at
// status_url, then starts the program in the background and records the new pid. False with a reason, for example when
// a server that `tona start` did not start holds the address.
bool restart_script_service(const ScriptService& service, std::string& why);

// The server that the pid file names is still running
bool script_service_alive(const ScriptService& service);

// Short name of a running process ("" if there is none)
std::string process_name(int pid);
// True while the process exists and is not a zombie
bool process_running(int pid);

}  // namespace tona::update
