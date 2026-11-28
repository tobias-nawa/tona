// tona: the one program of the user. It opens the control centre, runs the guided setup, answers the commands without a
// screen, starts the coding agents against the server and runs the server in the background (docs/use/client.md):
//   tona [top] [--url URL] [--tab start|models|agents|settings]
//       the centre with four pages: Start, Models, Agents, Settings; starts the server if none runs (a terminal is
//       needed; without one: status). When nothing is installed yet: the guided setup.
//   tona setup ...                              the guided setup and its options (installer/)
//   tona status|models|use|config|doctor ...    the same logic without a screen, plain text or --json
//   tona claude|opencode|junie|pi [folder] [agent options]
//   tona start|stop|restart|update|autostart|uninstall|--version
// Variables: TONA_PORT, TONA_START_CMD, TONA_TOOLS, TONA_EFFORT and more; see docs/use/environment.md.
#include <unistd.h>

#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>

#include "client/agent_launch.h"
#include "client/app.h"
#include "client/commands.h"
#include "client/control.h"
#include "client/foreground.h"
#include "client/manage_cmd.h"
#include "client/service_cmd.h"
#include "client/update_cmd.h"
#include "core/sys.h"
#include "setup.h"

namespace {

using namespace tona::client;

constexpr const char* kUsage =
    "Usage: tona [top] [--url http://127.0.0.1:8662] [--tab start|models|agents|settings]\n"
    "       tona setup | status | models | use | config | doctor ... (tona help)\n"
    "       tona claude|opencode|junie|pi [folder] [agent options]\n"
    "       tona start | stop | restart | update | autostart on|off|status | uninstall | --version\n";

bool has_terminal() { return isatty(STDIN_FILENO) && isatty(STDOUT_FILENO); }

bool ask_yes_no(const std::string& question) {
    std::cout << question << " [y/N] " << std::flush;
    std::string answer;
    std::getline(std::cin, answer);
    return answer == "y" || answer == "Y" || answer == "yes";
}

int run_screenless(const std::vector<std::string>& args, const Env& env) {
    std::unique_ptr<Control> control = system_control(env);
    Services svc = system_services(env, *control);
    Io io;
    io.out = &std::cout, io.err = &std::cerr;
    if (has_terminal()) io.ask = ask_yes_no;
    return run_command(args, env, svc, io);
}

// The control centre. `tona` and `tona top` start the server when none answers; the bare options (--url, --tab) as the
// screenshots and `just tona` give them do so only with --start.
int run_center(const std::vector<std::string>& args, const Env& env, bool autostart) {
    CenterOptions opts;
    opts.url = env.url();
    opts.autostart = autostart;
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--url" && i + 1 < args.size())
            opts.url = args[++i];
        else if (args[i] == "--tab" && i + 1 < args.size())
            opts.tab = args[++i];
        else if (args[i] == "--start")
            opts.autostart = true;
        else
            return std::fputs(kUsage, stderr), 2;
    }
    if (!has_terminal()) return run_screenless({"status"}, env);  // no screen: the text
    return Center(env, opts).run();
}

int run_agent_command(const std::vector<std::string>& args, const Env& env) {
    std::unique_ptr<Control> control = system_control(env);
    LaunchIo io;
    io.out = &std::cout, io.err = &std::cerr;
    io.ask = [](const std::string& prompt) {
        // As the shell's `read -p`: the prompt only on a terminal; the end of input counts as no answer
        if (isatty(STDIN_FILENO)) std::cerr << prompt << std::flush;
        std::string answer;
        std::getline(std::cin, answer);
        return answer;
    };
    return run_agent(env, *control, args, io);
}

// Nothing installed yet: no checkout, and no version in the install folder
bool not_installed(const Env& env) {
    std::error_code ec;
    return env.release && !std::filesystem::exists(std::filesystem::path(env.root) / "current", ec);
}

int run_lifecycle(const std::string& word, const std::vector<std::string>& args, const Env& env) {
    std::unique_ptr<Control> control = system_control(env);
    Io io;
    io.out = &std::cout, io.err = &std::cerr;
    if (has_terminal()) io.ask = ask_yes_no;
    const std::vector<std::string> rest(args.begin() + 1, args.end());
    if (word == "update") return run_update_command(env, rest, io);
    if (word == "uninstall") return run_uninstall(env, *control, rest, io);
    if (word == "start" && !rest.empty() && rest[0] == "--foreground") return run_foreground(env, std::cerr);
    if (word == "autostart") return run_autostart_command(env, rest.empty() ? "" : rest[0], std::cout, std::cerr);
    return run_service_command(env, word, std::cout);
}

bool is_lifecycle(const std::string& word) {
    return is_service_command(word) || word == "autostart" || word == "update" || word == "uninstall";
}

int run_app(int argc, char** argv) {
    const Env env = Env::detect();
    const std::vector<std::string> args(argv + 1, argv + argc);
    const std::string word = args.empty() ? "" : args[0];
    if (word == "--version") return std::fputs(version_line(tona::install::self_exe()).c_str(), stdout), 0;
    if (word == "setup") return tona::install::run_setup(std::vector<std::string>(args.begin() + 1, args.end()));
    if (is_command(word)) return run_screenless(args, env);
    if (word.empty() || word == "top") {
        if (not_installed(env))
            return has_terminal() ? tona::install::run_setup({})
                                  : (std::cerr << "tona: Tona is not installed yet. Run tona setup.\n", 1);
        return run_center(args.empty() ? args : std::vector<std::string>(args.begin() + 1, args.end()), env, true);
    }
    if (word.starts_with("--")) return run_center(args, env, false);
    if (!is_agent(word) && !is_lifecycle(word)) {
        std::cerr << "tona: unknown command '" << word
                  << "'. Commands: tona | tona setup | tona status | tona models | "
                  << "tona use | tona config | tona doctor | tona claude | tona opencode | tona junie | tona pi | tona "
                  << "start | tona stop | tona restart | tona update | tona autostart | tona uninstall | tona "
                  << "--version\n";
        return 2;
    }
    if (const std::string err = layout_error(env); !err.empty()) return std::cerr << "tona: " << err << "\n", 1;
    if (is_lifecycle(word)) return run_lifecycle(word, args, env);
    return run_agent_command(args, env);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run_app(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
