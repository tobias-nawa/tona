#include "client/commands.h"

#include <algorithm>
#include <ostream>

#include "client/commands_internal.h"
#include "client/term.h"
#include "client/text.h"
#include "client/ui.h"
#include "third_party/httplib.h"

namespace tona::client {

using nlohmann::json;

ServerView fetch_server_view(const std::string& url) {
    httplib::Client c(url);
    c.set_connection_timeout(1, 0);
    c.set_read_timeout(2, 0);
    auto res = c.Get("/status");
    if (!res) return parse_status({}, false, httplib::to_string(res.error()));
    if (res->status != 200) return parse_status({}, false, "HTTP " + std::to_string(res->status));
    const json status = json::parse(res->body, nullptr, false);
    if (status.is_discarded()) return parse_status({}, false, "answer is not JSON");
    return parse_status(status, true, "");
}

bool is_command(const std::string& word) {
    return word == "status" || word == "models" || word == "use" || word == "config" || word == "doctor" ||
           word == "help" || word == "--help" || word == "-h";
}

std::string usage_text() {
    return "usage: tona [command]\n"
           "\n"
           "  tona                         the control centre (without a terminal: tona status); the guided setup\n"
           "                               when nothing is installed yet\n"
           "  tona setup                   the guided setup again: models, tasks, room (tona setup --help)\n"
           "  tona start | stop | restart  the server on, off, stop then start\n"
           "  tona status [--json]         server, model, conversations, memory; exit 3 when the server is down\n"
           "  tona models [--task ID] [--json]\n"
           "                               the models: installed, to download, coming; a task without a model\n"
           "                               is listed as coming with its note (exit 0, nothing is hidden)\n"
           "  tona use <model> [--yes] [--no-restart]\n"
           "                               switch the model and restart the server; --yes cuts running conversations\n"
           "  tona config get|set <key> [value] [--yes] [--no-restart] | tona config list\n"
           "                               setting agents or context restarts the server like use\n"
           "  tona doctor [--json]         checks with the fix for each line; exit 5 when one is not ok\n"
           "  tona claude|opencode|junie|pi [dir]   start an agent against the server (project folder optional)\n"
           "  tona update [--yes]          check for a new release, show version and size, ask, install, restart;\n"
           "                               goes back by itself if the new version does not start (--rollback: on "
           "purpose)\n"
           "  tona autostart on|off|status the server starts at login\n"
           "  tona uninstall [--yes] [--models]\n"
           "                               stop, remove the service and the install directory; the models stay\n"
           "                               unless you say so\n"
           "  tona --version               version, platform, backend\n"
           "\n"
           "exit codes: 0 ok, 1 failed, 2 usage, 3 server down (status), 4 a question and no terminal to ask on (use, "
           "update, uninstall), 5 doctor warnings\n";
}

namespace {

bool has_flag(std::vector<std::string>& args, const std::string& flag) {
    const auto it = std::find(args.begin(), args.end(), flag);
    if (it == args.end()) return false;
    args.erase(it);
    return true;
}

// Value of `--name X`; removes both; empty if absent
std::string take_option(std::vector<std::string>& args, const std::string& name) {
    const auto it = std::find(args.begin(), args.end(), name);
    if (it == args.end() || it + 1 == args.end()) return {};
    const std::string value = *(it + 1);
    args.erase(it, it + 2);
    return value;
}

json conversation_json(const Conversation& c) {
    return {{"agent", c.agent}, {"project", c.project}, {"role", c.role}, {"state", c.state}, {"tok_s", c.tok_s}};
}

std::string state_word(const ServerView& v) {
    return !v.up ? "down" : v.loading() ? "loading" : !v.error.empty() ? "failed" : "running";
}

json status_json(const Env& env, const ServerView& v, const std::string& model_id) {
    json conversations = json::array();
    for (const Conversation& c : v.running) conversations.push_back(conversation_json(c));
    return {{"server", {{"state", state_word(v)}, {"url", env.url()}, {"port", env.port}, {"error", v.error}}},
            {"model", {{"id", model_id}, {"file", v.model_file}}},
            {"conversations", conversations},
            {"decode_tok_s", v.total_tok_s()},
            {"load_fraction", v.load_fraction()},
            {"memory_bytes",
             {{"vram_used", v.mem.vram_used},
              {"vram_total", v.mem.vram_total},
              {"ram_total", v.mem.ram_total},
              {"ram_available", v.mem.ram_available},
              {"ram_tona", v.mem.ram_tona},
              {"unified", v.mem.unified},
              {"tona_gpu", v.mem.tona_gpu}}}};
}

// Unified memory: Tona's share of the RAM. A graphics memory the server could not read (total -1, no sysfs) gets no
// line: "-0.0 of -0 GB" said nothing
std::string memory_line(const Memory& m) {
    if (m.unified) return fill(tr("cmd_memory_unified"), fdec(m.tona_gpu / 1e9), fdec(m.ram_total / 1e9, 0)) + "\n";
    if (m.vram_total <= 0) return "";
    return fill(tr("cmd_memory"), fdec(m.vram_used / 1e9), fdec(m.vram_total / 1e9, 0), fnum(m.ram_tona / 1e9)) + "\n";
}

int cmd_status(std::vector<std::string> args, const Env& env, Services& svc, Io& io) {
    const bool as_json = has_flag(args, "--json");
    if (!args.empty()) return *io.err << usage_text(), kExitUsage;
    const ServerView v = svc.status();
    const std::string id = model_id_of_file(svc.catalog(), v.model_file);
    if (as_json) {
        *io.out << status_json(env, v, id).dump(2) << "\n";
    } else {
        *io.out << tr("cmd_server") << ": " << tr("cmd_state_" + state_word(v)) << " (" << env.url() << ")\n";
        if (!v.error.empty()) *io.out << tr("cmd_error") << ": " << v.error << "\n";
        if (v.up && !v.model_file.empty()) *io.out << tr("cmd_model") << ": " << model_stem(v.model_file) << "\n";
        if (v.loading()) *io.out << tr("cmd_loading") << ": " << fnum(100 * v.load_fraction()) << " %\n";
        if (v.ready) {
            *io.out << tr("cmd_conversations") << ": " << v.conversation_count() << " (" << tr("conv_writes") << " "
                    << fnum(v.total_tok_s()) << " " << tr("unit_tok_s") << ")\n";
            *io.out << memory_line(v.mem);
        }
    }
    return v.up ? kExitOk : kExitServerDown;
}

std::string model_state(const ModelRow& r) {
    if (r.section == ModelRow::Section::Coming) return "coming";
    if (r.loaded) return "loaded";
    return r.section == ModelRow::Section::Installed ? "installed" : "to_download";
}

// The task filter of `tona models --task ID`; false (and a message on err) for an id the offer does not know
bool select_task(UiState& s, const std::string& task, std::ostream& err) {
    const auto it =
        std::find_if(s.offer.tasks.begin(), s.offer.tasks.end(), [&](const Task& t) { return t.id == task; });
    if (it != s.offer.tasks.end()) return s.filter.task = (size_t)(it - s.offer.tasks.begin()) + 1, true;
    std::string all;
    for (const Task& t : s.offer.tasks) all += (all.empty() ? "" : ", ") + t.id;
    err << "tona models: unknown task '" << task << "' (known: " << all << ")\n";
    return false;
}

json model_json(const ModelRow& r) {
    const ModelCard& c = *r.card;
    return {{"id", c.id},
            {"name", c.name},
            {"status", c.coming ? "coming" : "available"},
            {"coming_note", c.coming ? c.note : std::string()},
            {"tasks", c.tasks},
            {"state", model_state(r)},
            {"size_gb", c.size_gb},
            {"decode_tok_s", c.decode_tok_s},
            {"decode_tok_s_high", c.decode_tok_s_hi},  // > 0: decode_tok_s .. this is an estimate for this machine
            {"speed_context_tokens", c.speed_context},
            {"speed_basis", basis_word(c.speed_basis)},
            {"strong", c.strong}};
}

void print_model_row(const ModelRow& r, std::ostream& out) {
    const ModelCard& c = *r.card;
    out << (c.id.empty() ? c.name : c.id) << "\t" << tr("row_" + model_state(r)) << "\t"
        << (c.size_gb > 0 ? fnum(c.size_gb) + " GB" : "") << (c.coming ? c.note : std::string())
        << (c.decode_tok_s_hi > 0 ? "\t" + fnum(c.decode_tok_s) + "-" + fnum(c.decode_tok_s_hi) + " tok/s" : "")
        << "\n";
}

int cmd_models(std::vector<std::string> args, Services& svc, Io& io) {
    const bool as_json = has_flag(args, "--json");
    const std::string task = take_option(args, "--task");
    if (!args.empty()) return *io.err << usage_text(), kExitUsage;
    UiState s;
    s.offer = svc.offer(), s.catalog = svc.catalog();
    s.filter.fits_only = false;  // the list shows everything; the screen filters by fit
    s.loaded_id = model_id_of_file(s.catalog, svc.status().model_file);
    if (!task.empty() && !select_task(s, task, *io.err)) return kExitUsage;
    json out = json::array();
    for (const ModelRow& r : model_rows(s)) {
        out.push_back(model_json(r));
        if (!as_json) print_model_row(r, *io.out);
    }
    if (as_json) *io.out << out.dump(2) << "\n";
    return kExitOk;
}

}  // namespace

int run_command(const std::vector<std::string>& args, const Env& env, Services& svc, Io& io) {
    if (args.empty() || args[0] == "help" || args[0] == "--help" || args[0] == "-h")
        return *io.out << usage_text(), args.empty() ? kExitUsage : kExitOk;
    const std::string& word = args[0];
    const std::vector<std::string> rest(args.begin() + 1, args.end());
    if (word == "status") return cmd_status(rest, env, svc, io);
    if (word == "models") return cmd_models(rest, svc, io);
    if (word == "use") return run_use(rest, env, svc, io);
    if (word == "config") return run_config(rest, env, svc, io);
    if (word == "doctor") return run_doctor(rest, env, svc, io);
    *io.err << "tona: unknown command '" << word << "'\n" << usage_text();
    return kExitUsage;
}

}  // namespace tona::client
