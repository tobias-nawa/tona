#include "core/autostart.h"

#include <filesystem>

#include "core/release_steps.h"
#include "core/sys.h"
#include "core/unit.h"
#include "update/config.h"

namespace fs = std::filesystem;

namespace tona::install {

namespace {

std::string xml(const std::string& s) {
    std::string o;
    for (const char c : s) {
        if (c == '&')
            o += "&amp;";
        else if (c == '<')
            o += "&lt;";
        else if (c == '>')
            o += "&gt;";
        else
            o += c;
    }
    return o;
}

std::string launchd_domain() { return "gui/$(id -u)"; }

// bootout of a job that is not loaded fails; that is the state we want, so its status does not count
std::string unload_agent(const LineFn& out) {
    run("launchctl bootout " + launchd_domain() + "/" + kLaunchAgentLabel + " 2>/dev/null", out);
    return "";
}

std::string set_launch_agent(const std::string& root, bool on, bool now, const LineFn& out) {
    const std::string path = launch_agent_path();
    if (!on) {
        // Removing the file is enough for the next login. No bootout and no `launchctl disable`: bootout ends the
        // running server (launchctl(1)), disable would make a later bootstrap fail until `enable`. A loaded job
        // stays loaded; `tona stop` ends the server through its pid file.
        if (!fs::exists(path)) return "";
        std::error_code ec;
        fs::remove(path, ec);
        return ec ? path + ": " + ec.message() : "";
    }
    // launchd opens the log before the server runs and may not create the folder (not yet checked on a Mac)
    std::error_code mk;
    fs::create_directories(update::state_dir(), mk);
    if (std::string err = write_atomic(path, render_launch_agent(root), 0644); !err.empty()) return err;
    out("written: " + path);
    if (!now) return "";  // launchd loads every agent of ~/Library/LaunchAgents at the next login
    unload_agent(out);    // a loaded older copy would keep its old arguments
    const RunResult r = run("launchctl bootstrap " + launchd_domain() + " " + quote(path), out);
    return r.status == 0 ? "" : "launchctl bootstrap failed (status " + std::to_string(r.status) + ")";
}

}  // namespace

std::string launch_agent_path() { return home() + "/Library/LaunchAgents/" + kLaunchAgentLabel + ".plist"; }

std::string render_launch_agent(const std::string& root) {
    const std::string r = xml(root);
    const std::string log = xml(update::state_dir() + "/server.log");  // the file `tona start` writes too
    // `tona start --foreground` becomes the server (exec), so the job is the server and launchd needs no more
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
           "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
           "<plist version=\"1.0\">\n<dict>\n"
           "  <key>Label</key><string>" +
           std::string(kLaunchAgentLabel) +
           "</string>\n"
           "  <key>ProgramArguments</key><array><string>" +
           r +
           "/current/bin/tona</string><string>start</string><string>--foreground</string>"
           "</array>\n"
           "  <key>EnvironmentVariables</key><dict><key>TONA_INSTALL_DIR</key><string>" +
           r +
           "</string></dict>\n"
           "  <key>RunAtLoad</key><true/>\n"
           "  <key>StandardOutPath</key><string>" +
           log +
           "</string>\n"
           "  <key>StandardErrorPath</key><string>" +
           log +
           "</string>\n"
           "</dict>\n</plist>\n";
}

std::string set_autostart(bool macos, const std::string& root, bool on, bool now, const LineFn& out) {
    if (macos) return set_launch_agent(root, on, now, out);
    const std::string verb = on ? (now ? "enable --now " : "enable ") : "disable ";
    const RunResult r = run("systemctl --user " + verb + kUnitName, out);
    return r.status == 0 ? "" : "systemctl --user " + verb + kUnitName + " failed";
}

bool autostart_on(bool macos) {
    if (macos) return fs::exists(launch_agent_path());
    return run(std::string("systemctl --user -q is-enabled ") + kUnitName + " 2>/dev/null").status == 0;
}

}  // namespace tona::install
