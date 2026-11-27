#include "client/agent_launch.h"

#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <ostream>

#include "backend/env.h"
#include "client/agent_server.h"
#include "client/format.h"
#include "core/config.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::client {

namespace {

// The working directory as the user spelled it (symbolic links not resolved), as a shell shows it in $PWD. The
// project name comes from this path.
std::string logical_dir(const std::string& arg) {
    fs::path p(arg);
    if (p.is_relative()) {
        const char* pwd = std::getenv("PWD");
        std::error_code ec;
        const fs::path base = pwd && *pwd && fs::equivalent(pwd, ".", ec) ? fs::path(pwd) : fs::current_path(ec);
        p = base / p;
    }
    p = p.lexically_normal();
    return p.has_filename() || p == p.root_path() ? p.string() : p.parent_path().string();
}

std::string project_of(const std::string& dir) {
    // ${PWD##*/}: everything after the last slash, so the root directory has no name
    return dir.substr(dir.rfind('/') + 1);
}

std::string conf_value(const install::Values& v, const char* key, const std::string& fallback) {
    const auto it = v.find(key);
    return it != v.end() && !it->second.empty() ? it->second : fallback;
}

std::string env_or(const char* name, const std::string& fallback) {
    const char* v = tona::env::text(name);
    return v ? v : fallback;
}

// The key of the key file, or "local" when there is none (the server then does not ask for one)
std::string read_key(const Env& env) {
    const install::Values conf = install::read_values(env.conf);
    const std::string file =
        env_or("TONA_API_KEY_FILE", conf_value(conf, "TONA_API_KEY_FILE", install::home() + "/.config/tona/api-key"));
    std::string key = install::read_file(file);
    if (key.empty()) return "local";
    while (!key.empty() && key.back() == '\n') key.pop_back();  // as a shell strips them from $(<file)
    return key;
}

// Starts the server if none answers. Returns false when it could not; `started` says whether this call started it.
bool ensure_server(const Env& env, Control& control, LaunchIo& io, bool& started) {
    if (server_answers(env.port)) return true;
    *io.out << "tona-server is not running, starting it ...\n";
    if (const std::string err = control.start(); !err.empty()) return *io.err << "tona: " << err << "\n", false;
    started = true;
    return true;
}

// The question on exit: only for a server this call started, and only when no conversation runs
void offer_stop(const Env& env, Control& control, LaunchIo& io) {
    if (server_engine_state(env.port) != "idle") return;
    const std::string a = io.ask ? io.ask("Stop tona-server (frees RAM and GPU)? [Y/n] ") : std::string();
    if (a == "n" || a == "N" || a == "no") {
        *io.out << "keeps running, stop it later with: tona stop\n";
        return;
    }
    if (const std::string err = control.stop(); !err.empty())
        *io.err << "tona: " << err << "\n";
    else
        *io.out << "stopped\n";
}

}  // namespace

AgentSettings agent_settings(const Env& env) {
    const install::Values conf = install::read_values(env.conf);
    AgentSettings s;
    s.tools = env_or("TONA_TOOLS", "");
    s.compact = env_or("TONA_COMPACT", s.compact);
    s.mode = env_or("TONA_MODE", s.mode);
    s.effort = env_or("TONA_EFFORT", conf_value(conf, "TONA_EFFORT", s.effort));
    s.pi_dir = env_or("PI_CODING_AGENT_DIR", install::home() + "/.pi/agent");
    return s;
}

int spawn_agent(const AgentCommand& cmd, const std::string& cwd, std::ostream& err) {
    std::vector<char*> argv{const_cast<char*>(cmd.program.c_str())};
    for (const std::string& a : cmd.args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    struct sigaction ignore{}, old_int{}, old_quit{};
    ignore.sa_handler = SIG_IGN;
    sigaction(SIGINT, &ignore, &old_int);
    sigaction(SIGQUIT, &ignore, &old_quit);
    const pid_t pid = ::fork();
    if (pid == 0) {
        sigaction(SIGINT, &old_int, nullptr);  // the agent gets the signals as if tona were not there
        sigaction(SIGQUIT, &old_quit, nullptr);
        if (!cwd.empty()) ::setenv("PWD", cwd.c_str(), 1);
        for (const auto& [k, v] : cmd.env) ::setenv(k.c_str(), v.c_str(), 1);
        ::execvp(argv[0], argv.data());
        std::perror(argv[0]);
        ::_exit(127);
    }
    int status = 0;
    if (pid > 0) {
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
    }
    sigaction(SIGINT, &old_int, nullptr);
    sigaction(SIGQUIT, &old_quit, nullptr);
    if (pid < 0) return err << "tona: cannot start " << cmd.program << "\n", 1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

int run_agent(const Env& env, Control& control, const std::vector<std::string>& args, LaunchIo& io,
              const LaunchOptions& options) {
    const std::string agent = args.empty() ? "" : args[0];
    std::vector<std::string> rest(args.size() > 1 ? args.begin() + 1 : args.end(), args.end());
    if (!is_agent(agent) || !install::which(agent)) return *io.err << "tona: " << agent << " is not installed\n", 1;
    // The first argument may be the project folder (the control centre starts agents this way); the agent runs in it
    // and X-Tona-Project names it. Other arguments go to the agent as they are.
    std::error_code ec;
    std::string cwd;
    if (!rest.empty() && !rest[0].empty() && fs::is_directory(rest[0], ec)) {
        cwd = logical_dir(rest[0]);
        if (::chdir(rest[0].c_str()) != 0) return *io.err << "tona: cannot enter " << rest[0] << "\n", 1;
        rest.erase(rest.begin());
    } else {
        cwd = logical_dir(".");
    }
    bool started = false;
    if (!ensure_server(env, control, io, started)) return 1;
    if (!wait_until_ready(env.port, options.wait_s, *io.out, *io.err)) return 1;

    AgentContext c;
    c.port = env.port;
    c.key = read_key(env);
    c.n_ctx = server_context_size(env.port);
    server_model_names(env.port, c.key, c.model_id, c.model_name);
    c.project = project_of(cwd);
    c.state_dir = env.state_dir;
    c.hint_file = env.state_dir + "/agent-hint.md";
    c.hint_text = hint_text(env.port);
    c.settings = agent_settings(env);
    if (const std::string err = write_file_atomic(c.hint_file, c.hint_text + "\n"); !err.empty())
        return *io.err << "tona: " << err << "\n", 1;
    AgentCommand cmd;
    if (const std::string err = agent_command(agent, c, rest, cmd); !err.empty())
        return *io.err << "tona: " << err << "\n", 1;

    const int rc = spawn_agent(cmd, cwd, *io.err);
    if (started) offer_stop(env, control, io);
    return rc;
}

}  // namespace tona::client
