#include "client/settings.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <optional>

#include "client/format.h"
#include "core/context_plan.h"
#include "core/hw.h"
#include "core/module.h"

namespace tona::client {

namespace {

// A function, not a variable: building the vector can throw, which a static initialiser cannot handle
const std::vector<std::string>& on_off_choices() {
    static const std::vector<std::string> choices = {"off", "on"};
    return choices;
}

std::vector<SettingSpec> make_specs() {
    using K = Kind;
    return {
        {"use", "group_memory", K::Choice, {"desktop", "server"}, 0, 0, 0, "", "sethelp_use", "TONA_USE"},
        {"priority",
         "group_memory",
         K::Choice,
         {"work", "balanced", "tona"},
         0,
         0,
         0,
         "",
         "sethelp_priority",
         "TONA_PRIORITY"},
        {"ram_gb", "group_memory", K::Number, {}, 8, 512, 2, "GB", "sethelp_ram", ""},
        {"vram_gb", "group_memory", K::Number, {}, 4, 128, 0.5, "GB", "sethelp_vram", ""},
        {"agents", "group_chats", K::Number, {}, 1, install::kMaxAgents, 1, "", "sethelp_agents", ""},
        {"context", "group_chats", K::Choice, {"auto", "64K", "128K", "256K"}, 0, 0, 0, "", "sethelp_context", ""},
        {"oea", "group_chats", K::Toggle, on_off_choices(), 0, 0, 0, "", "sethelp_oea", ""},
        {"api_key", "group_access", K::Toggle, on_off_choices(), 0, 0, 0, "", "sethelp_api_key", ""},
        {"port", "group_access", K::Number, {}, 1024, 65535, 1, "", "sethelp_port", ""},
        {"close", "group_start", K::Choice, {"ask", "keep", "stop"}, 0, 0, 0, "", "sethelp_close", "TONA_CLOSE"},
        {"clock_hold", "group_power", K::Toggle, on_off_choices(), 0, 0, 0, "", "sethelp_clock_hold", ""},
        {"effort", "group_expert", K::Choice, {"low", "medium", "high"}, 0, 0, 0, "", "sethelp_effort", ""},
        {"drafter", "group_expert", K::Choice, {"mtp", "dflash", "none"}, 0, 0, 0, "", "sethelp_drafter", ""},
        {"prompt_cache_gb", "group_expert", K::Number, {}, 0, 256, 4, "GB", "sethelp_cache", ""},
    };
}

std::string number_text(double v) { return std::format("{:g}", v); }

const std::string& extra_or(const Config& c, const std::string& conf_key, const std::string& fallback) {
    const auto it = c.extras.find(conf_key);
    return it != c.extras.end() && !it->second.empty() ? it->second : fallback;
}

}  // namespace

const std::vector<SettingSpec>& setting_specs() {
    static const std::vector<SettingSpec> specs = make_specs();
    return specs;
}

const SettingSpec* find_setting(const std::string& key) {
    for (const SettingSpec& s : setting_specs())
        if (s.key == key) return &s;
    return nullptr;
}

std::function<void(install::Settings&)> local_resolver(const Env& env) {
    // Hardware detection reads the system: once per resolver, and only when a setting changes
    auto module = std::make_shared<std::optional<const install::Module*>>();
    return [env, module](install::Settings& s) {
        if (!*module) *module = install::find_module(install::detect(env.target().dir()));
        install::resolve_context(**module, s);
    };
}

Config load_config(const Env& env) {
    Config c;
    c.resolve = local_resolver(env);
    c.s = load_settings(env);
    c.extras = install::read_values(env.conf);
    return c;
}

std::string save_config(const Env& env, const Config& c) {
    if (const std::string err = save_settings(env, c.s); !err.empty()) return err;
    install::Values mine;
    for (const SettingSpec& spec : setting_specs())
        if (!spec.conf_key.empty() && c.extras.contains(spec.conf_key))
            mine[spec.conf_key] = c.extras.at(spec.conf_key);
    return mine.empty() ? "" : install::set_values(env.conf, mine);
}

namespace {

std::string on_off(bool b) { return b ? "on" : "off"; }

}  // namespace

std::string get_value(const Config& c, const std::string& key) {
    const install::Settings& s = c.s;
    const std::map<std::string, std::function<std::string()>> getters = {
        {"ram_gb", [&] { return number_text(s.ram_gb); }},
        {"vram_gb", [&] { return number_text(s.vram_gb); }},
        {"agents", [&] { return std::to_string(s.seqs); }},
        {"context", [&] { return s.ctx_auto ? std::string("auto") : install::context_text(s.ctx); }},
        {"oea", [&] { return on_off(s.oea); }},
        {"api_key", [&] { return on_off(s.api_key); }},
        {"port", [&] { return std::to_string(s.port); }},
        {"clock_hold", [&] { return on_off(s.clock_hold); }},
        {"effort", [&] { return s.effort; }},
        {"drafter", [&] { return s.drafter; }},
        {"prompt_cache_gb", [&] { return number_text(s.prompt_cache_gb); }},
        {"use", [&] { return extra_or(c, "TONA_USE", "desktop"); }},
        {"priority", [&] { return extra_or(c, "TONA_PRIORITY", "balanced"); }},
        {"close", [&] { return extra_or(c, "TONA_CLOSE", "ask"); }},
    };
    const auto it = getters.find(key);
    return it == getters.end() ? std::string() : it->second();
}

namespace {

std::string parse_number(const SettingSpec& spec, const std::string& value, double& out) {
    char* end = nullptr;
    out = std::strtod(value.c_str(), &end);
    if (value.empty() || end != value.c_str() + value.size()) return spec.key + ": not a number: " + value;
    if (out < spec.min || out > spec.max)
        return std::format("{}: {} is outside {:g} to {:g}", spec.key, value, spec.min, spec.max);
    return {};
}

void store_number(install::Settings& s, const std::string& key, double v) {
    if (key == "ram_gb") s.ram_gb = v;
    if (key == "vram_gb") s.vram_gb = v;
    if (key == "agents") s.seqs = (int)v;
    if (key == "port") s.port = (int)v;
    if (key == "prompt_cache_gb") s.prompt_cache_gb = v;
}

void store_text(Config& c, const SettingSpec& spec, const std::string& v) {
    const std::string& key = spec.key;
    if (!spec.conf_key.empty()) c.extras[spec.conf_key] = v;
    if (key == "context") {
        c.s.ctx_auto = v == "auto";
        if (!c.s.ctx_auto) c.s.ctx = install::parse_context(v);
    }
    if (key == "oea") c.s.oea = v == "on";
    if (key == "api_key") c.s.api_key = v == "on";
    if (key == "clock_hold") c.s.clock_hold = v == "on";
    if (key == "effort") c.s.effort = v;
    if (key == "drafter") c.s.drafter = v;
}

// The context of "auto" follows what it is made of
void resolve_after(Config& c, const std::string& key) {
    if (c.resolve && (key == "agents" || key == "context" || key == "ram_gb")) c.resolve(c.s);
}

}  // namespace

std::string set_value(Config& c, const std::string& key, const std::string& value) {
    const SettingSpec* spec = find_setting(key);
    if (!spec) return "unknown setting: " + key;
    if (spec->kind == Kind::Number) {
        double v = 0;
        if (const std::string err = parse_number(*spec, value, v); !err.empty()) return err;
        store_number(c.s, key, v);
        resolve_after(c, key);
        return {};
    }
    if (std::find(spec->choices.begin(), spec->choices.end(), value) == spec->choices.end()) {
        std::string all;
        for (const std::string& ch : spec->choices) all += (all.empty() ? "" : "|") + ch;
        return std::format("{}: {} is not one of {}", key, value, all);
    }
    store_text(c, *spec, value);
    resolve_after(c, key);
    return {};
}

std::string step_value(const Config& c, const SettingSpec& spec, int direction) {
    const std::string now = get_value(c, spec.key);
    if (spec.kind == Kind::Number) {
        const double v = std::clamp(to_double(now) + direction * spec.step, spec.min, spec.max);
        return number_text(v);
    }
    const auto it = std::find(spec.choices.begin(), spec.choices.end(), now);
    const long n = (long)spec.choices.size(), at = it == spec.choices.end() ? 0 : it - spec.choices.begin();
    if (spec.kind == Kind::Toggle) return spec.choices[(at + 1) % n];
    return spec.choices[(size_t)std::clamp(at + direction, 0L, n - 1)];
}

}  // namespace tona::client
