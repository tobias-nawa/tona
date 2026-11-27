// tona use, tona config, tona doctor and the real services
#include <algorithm>
#include <cctype>
#include <ostream>

#include "client/commands.h"
#include "client/commands_internal.h"
#include "client/format.h"
#include "client/text.h"
#include "client/ui.h"

namespace tona::client {

using nlohmann::json;

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

bool take_flag(std::vector<std::string>& args, const std::string& flag) {
    const auto it = std::find(args.begin(), args.end(), flag);
    if (it == args.end()) return false;
    args.erase(it);
    return true;
}

// Conversations that a restart cuts, as one line each
std::string conversation_list(const ServerView& v) {
    std::string out;
    for (const Conversation& c : v.running)
        if (c.state != "waiting")
            out += "  " + (c.agent.empty() ? "–" : c.agent) + "  " + project_label(c.project) + "\n";
    return out;
}

// True when the restart may go on: nothing runs, --yes, or the user said yes
bool may_restart(const ServerView& v, bool yes, Io& io) {
    if (v.conversation_count() == 0 || yes) return true;
    const std::string question = fill(tr("use_cuts"), fnum(v.conversation_count())) + "\n" + conversation_list(v);
    if (!io.ask) {
        *io.err << question << tr("use_need_yes") << "\n";
        return false;
    }
    return io.ask(question + tr("use_ask"));
}

}  // namespace

std::string resolve_model(const Offer& offer, const std::string& typed) {
    const std::string want = lower(typed);
    for (const ModelCard& c : offer.cards)
        if (!c.coming && (lower(c.id) == want || lower(c.name) == want)) return c.id;
    return {};
}

int run_use(std::vector<std::string> args, const Env& env, Services& svc, Io& io) {
    const bool yes = take_flag(args, "--yes"), no_restart = take_flag(args, "--no-restart");
    if (args.size() != 1) return *io.err << usage_text(), kExitUsage;
    const std::string id = resolve_model(svc.offer(), args[0]);
    if (id.empty()) return *io.err << fill(tr("use_unknown"), args[0]) << "\n", kExitUsage;
    const ServerView v = svc.status();
    const bool restart = v.up && !no_restart;
    if (restart && !may_restart(v, yes, io)) return *io.err << tr("use_cancelled") << "\n", kExitNeedsConfirm;
    const std::string err =
        restart ? switch_model(env, *svc.control, svc.catalog(), id) : prepare_switch(env, svc.catalog(), id);
    if (!err.empty()) return *io.err << "tona use: " << err << "\n", kExitFailed;
    *io.out << fill(tr(restart ? "use_done_restart" : "use_done_saved"), id) << "\n";
    return kExitOk;
}

namespace {

// Agents and context change what the server starts with: setting them restarts it, as `tona use` does
bool restarts_server(const std::string& key) { return key == "agents" || key == "context"; }

int set_config(std::vector<std::string> args, const Env& env, Services& svc, Io& io) {
    const bool yes = take_flag(args, "--yes"), no_restart = take_flag(args, "--no-restart");
    if (args.size() != 3) return *io.err << usage_text(), kExitUsage;
    Config c = load_config(env);
    if (svc.resolve) c.resolve = svc.resolve;
    if (const std::string err = set_value(c, args[1], args[2]); !err.empty())
        return *io.err << "tona config: " << err << "\n", kExitUsage;
    const ServerView v = restarts_server(args[1]) && !no_restart ? svc.status() : ServerView{};
    if (v.up && !may_restart(v, yes, io)) return *io.err << tr("use_cancelled") << "\n", kExitNeedsConfirm;
    if (const std::string err = save_config(env, c); !err.empty())
        return *io.err << "tona config: " << err << "\n", kExitFailed;
    if (!v.up)
        return *io.out << args[1] << "=" << get_value(c, args[1]) << "  (" << tr("config_applies") << ")\n", kExitOk;
    if (const std::string err = svc.control->restart(); !err.empty())
        return *io.err << "tona config: " << err << "\n", kExitFailed;
    return *io.out << fill(tr("config_done_restart"), args[1], get_value(c, args[1])) << "\n", kExitOk;
}

}  // namespace

int run_config(std::vector<std::string> args, const Env& env, Services& svc, Io& io) {
    if (!args.empty() && args[0] == "set") return set_config(std::move(args), env, svc, io);
    Config c = load_config(env);
    if (args.size() == 1 && args[0] == "list") {
        for (const SettingSpec& spec : setting_specs()) *io.out << spec.key << "=" << get_value(c, spec.key) << "\n";
        return kExitOk;
    }
    if (args.size() == 2 && args[0] == "get") {
        if (!find_setting(args[1])) return *io.err << "tona config: unknown key '" << args[1] << "'\n", kExitUsage;
        return *io.out << get_value(c, args[1]) << "\n", kExitOk;
    }
    return *io.err << usage_text(), kExitUsage;
}

int run_doctor(std::vector<std::string> args, const Env& env, Services& svc, Io& io) {
    const bool as_json = take_flag(args, "--json");
    if (!args.empty()) return *io.err << usage_text(), kExitUsage;
    const std::vector<Check> checks = svc.doctor(load_settings(env));
    json out = json::array();
    bool warn = false;
    for (const Check& c : checks) {
        warn |= !c.ok;
        out.push_back({{"check", c.name}, {"ok", c.ok}, {"detail", c.detail}, {"fix", c.fix}});
        if (!as_json)
            *io.out << (c.ok ? "ok    " : "warn  ") << c.name << "  " << c.detail << (c.ok ? "" : "  -> " + c.fix)
                    << "\n";
    }
    if (as_json) *io.out << out.dump(2) << "\n";
    return warn ? kExitWarnings : kExitOk;
}

Services system_services(const Env& env, Control& control) {
    Services svc;
    svc.control = &control;
    svc.status = [env] { return fetch_server_view(env.url()); };
    svc.catalog = [env] { return installed_catalog(env); };
    svc.offer = [] { return build_offer(); };
    svc.doctor = [env](const install::Settings& s) { return run_checks(gather_facts(env, s)); };
    return svc;
}

}  // namespace tona::client
