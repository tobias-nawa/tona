#include "core/cli_args.h"

#include <cstdlib>
#include <utility>
#include <vector>

namespace tona::install {

const char* const kUsage =
    "usage: tona setup [OPTIONS]\n"
    "\n"
    "Without options, runs the guided setup in the terminal (every page pre-filled, Enter alone gives a good setup).\n"
    "The same choices work without a screen:\n"
    "\n"
    "  --task LIST        tasks, comma separated: coding, knowledge (others are listed as coming and cannot be named)\n"
    "  --lang LIST        programming languages (with coding): python, typescript, javascript, rust, csharp, go, ...\n"
    "  --models LIST      models by name or id: coder, ornith, bolt; --yes downloads exactly these\n"
    "  --priority P       who has priority in a shortage: work, balanced (default), tona\n"
    "  --use U            desktop (the PC is used for other work, default) or server (only for Tona)\n"
    "  --json             print the plan as JSON on stdout and change nothing\n"
    "  --dry-run          print the plan and the steps, change nothing\n"
    "  --yes              no questions: do the plan (what is named; without a name the coding recommendation, or an\n"
    "                     existing work/tona.conf)\n"
    "  --print            detect and recommend only, change nothing\n"
    "  --config FILE      no questions: take the settings from FILE (format of work/tona.conf)\n"
    "  --port N           the port of the server (default 8662; the setup proposes the next free one if it is taken)\n"
    "  --updates U        release: check for a new release once a day: on (default) or off\n"
    "  --modify-path      add ~/.local/bin to PATH in your shell profiles (one marked block); the guided setup asks\n"
    "  --no-modify-path   never touch the shell profiles (without a screen this is the default)\n"
    "  --autostart        release: start the server at every login (default: off; later: tona autostart on|off)\n"
    "  --install-dir DIR  release: install root (default $TONA_INSTALL_DIR or ~/.local/share/tona)\n"
    "  --models-dir DIR   release: where the models go (default <install-dir>/models)\n"
    "  --cache-dir DIR    release: prompt checkpoints (default <install-dir>/prompt-cache)\n"
    "  -h, --help         this text (--help-dev: the options for developers)\n"
    "\n"
    "Without a terminal the output is plain text. The guided setup respects NO_COLOR.\n"
    "\n"
    "Exit codes:\n"
    "  0  done (or, with --dry-run/--json/--print, the plan can run)\n"
    "  1  a step failed, or errors in the plan have to be fixed first\n"
    "  2  usage: unknown option or name, or no terminal and nothing to do (add --yes or --dry-run)\n"
    "  3  no backend for this hardware\n"
    "  4  not enough free space for what is named\n"
    "  5  a download failed (the next run resumes it)\n"
    "  6  a named task is not available yet\n"
    "  7  a required program is missing\n";

const char* const kUsageDev =
    "options for developers, in addition to `tona setup --help`:\n"
    "\n"
    "  --repo DIR         engine repository (default: the one above the program, or the current one);\n"
    "                     without one, tona setup installs the signed release\n"
    "  --release          install the signed release even inside a checkout\n"
    "  --model-from DIR   verify and link an existing model download instead of fetching it again\n"
    "  --dump-pages       print every page and state of the guided setup as text at 80 x 24 (documentation)\n";

namespace {

bool is_one_of(const std::string& v, const std::vector<const char*>& allowed) {
    for (const char* a : allowed)
        if (v == a) return true;
    return false;
}

// Options with a value, and flags
std::string* value_slot(Options& o, const std::string& a) {
    const std::pair<const char*, std::string*> values[] = {
        {"--config", &o.conf},           {"--repo", &o.repo},
        {"--model-from", &o.model_from}, {"--install-dir", &o.install_dir},
        {"--models-dir", &o.models_dir}, {"--cache-dir", &o.cache_dir},
        {"--priority", &o.priority},     {"--use", &o.use},
        {"--updates", &o.updates},       {"--port", &o.port}};
    for (const auto& [name, slot] : values)
        if (a == name) return slot;
    return nullptr;
}

std::vector<std::string>* list_slot(Options& o, const std::string& a) {
    if (a == "--task") return &o.named.tasks;
    if (a == "--lang") return &o.named.languages;
    if (a == "--models") return &o.named.models;
    return nullptr;
}

bool* flag_slot(Options& o, const std::string& a) {
    const std::pair<const char*, bool*> flags[] = {{"--release", &o.release},
                                                   {"--autostart", &o.autostart},
                                                   {"--print", &o.print},
                                                   {"--yes", &o.yes},
                                                   {"--dry-run", &o.dry},
                                                   {"--json", &o.json},
                                                   {"--help", &o.help},
                                                   {"-h", &o.help},
                                                   {"--help-dev", &o.help_dev},
                                                   {"--dump-pages", &o.dump_pages},
                                                   {"--no-modify-path", &o.no_modify_path},
                                                   {"--modify-path", &o.modify_path}};
    for (const auto& [name, slot] : flags)
        if (a == name) return slot;
    return nullptr;
}

bool check_values(const Options& o, std::string& error) {
    if (!o.priority.empty() && !is_one_of(o.priority, {"work", "balanced", "tona"}))
        return error = "--priority: work, balanced or tona", false;
    if (!o.updates.empty() && !is_one_of(o.updates, {"on", "off"})) return error = "--updates: on or off", false;
    if (!o.port.empty()) {
        char* end = nullptr;
        const long n = std::strtol(o.port.c_str(), &end, 10);
        if (*end || n < 1024 || n > 65535) return error = "--port: a number from 1024 to 65535", false;
    }
    if (!o.use.empty() && !is_one_of(o.use, {"desktop", "server"})) return error = "--use: desktop or server", false;
    return true;
}

}  // namespace

bool parse_args(int argc, const char* const* argv, Options& o, std::string& error) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        std::string* value = value_slot(o, a);
        std::vector<std::string>* list = list_slot(o, a);
        if (value || list) {
            const std::string v = i + 1 < argc ? argv[++i] : "";
            if (v.empty()) return error = a + " needs a value", false;
            if (list)
                for (const std::string& part : split_list(v)) list->push_back(part);
            else
                *value = v;
        } else if (bool* flag = flag_slot(o, a)) {
            *flag = true;
        } else {
            return error = "unknown option: " + a, false;
        }
    }
    return check_values(o, error);
}

}  // namespace tona::install
