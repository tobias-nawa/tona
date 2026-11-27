#include "core/cli_plan.h"

#include <format>

#include "core/catalog.h"
#include "core/describe.h"
#include "core/exit_codes.h"
#include "core/report.h"
#include "core/room.h"
#include "core/speed.h"
#include "nlohmann/json.hpp"

namespace tona::install {

namespace {

const char* kRoleStart = "starts with the server";
const char* kRoleExtra = "additional";

bool has_error_note(const std::vector<Note>& notes) {
    for (const Note& n : notes)
        if (n.warn) return true;
    return false;
}

// The model and its drafter are all there
bool model_there(const std::string& models_dir, const std::string& id) {
    return model_split(id, models_dir).download == 0;
}

std::string model_text(const PlanDoc& p, const std::string& id) {
    const offer::ModelCard* c = offer::model_card(id);
    if (!c) return "  " + id + "\n";
    const bool first = id == p.settings.model;
    std::string t = std::format("  {} ({}){}\n", c->name, id, first ? std::string("  [") + kRoleStart + "]" : "");
    t += std::format("    size {}{}\n", size_text(static_cast<double>(model_bytes(id))),
                     model_there(p.models_dir, id) ? ", already here, verified on install" : "");
    const Estimate e = p.module ? p.module->estimate(p.hw, p.settings, id) : Estimate{};
    t += std::format("    {}{}, {}\n", speed_plain(e),
                     e.width > 0 ? std::format(" (estimate, +-{:.0f} %)", e.width * 100) : "", verdict_line(*c));
    if (e.per_agent.hi > 0)
        t += std::format("    under your settings ({} x {}): {} (estimate, +-{:.0f} %)\n", e.agents,
                         context_label(e.load_context), load_speed_text(e), e.width_load * 100);
    return t;
}

// low and high in tok/s, never one exact number; "unknown" where the module has no estimate
nlohmann::json speed_json(const Estimate& e) {
    using nlohmann::json;
    if (e.decode.hi <= 0) return "unknown";
    json j = {{"decode", {{"low", e.decode.lo}, {"high", e.decode.hi}, {"context_tokens", e.context_tokens}}},
              {"width", e.width},
              {"anchor", "measured on " + e.anchor},
              {"unknown_inputs", e.unknown}};
    j["settings"] = {{"agents", e.agents},
                     {"context_tokens", e.load_context},
                     {"per_agent", {{"low", e.per_agent.lo}, {"high", e.per_agent.hi}}},
                     {"together", {{"low", e.together.lo}, {"high", e.together.hi}}},
                     {"estimate", true}};
    if (e.prefill.hi > 0) j["prefill"] = {{"low", e.prefill.lo}, {"high", e.prefill.hi}};
    return j;
}

}  // namespace

PlanDoc make_plan(const Hardware& h, const Module* m, const Selection& sel, const Settings& s, bool release,
                  const std::string& models_dir) {
    PlanDoc p;
    p.hw = h, p.module = m, p.match = match_hw_class(h), p.selection = sel, p.settings = s;
    p.models_dir = models_dir;
    p.free_bytes = static_cast<uint64_t>(h.disk_free);
    const LoadSplit load = load_split(s, models_dir);
    p.to_load = load.download, p.here = load.here;
    if (m) p.notes = m->check(h, s);
    p.packages = check_packages(h, m ? m->packages(h) : std::vector<Package>{}, release);
    return p;
}

int plan_exit_code(const PlanDoc& p) {
    if (!p.module) return kExitNoHardware;
    if (!packages_ok(p.packages)) return kExitMissingTools;
    if (p.to_load + kBufferBytes > p.free_bytes) return kExitNoSpace;
    return has_error_note(p.notes) ? kExitFailed : kExitOk;
}

std::string plan_text(const PlanDoc& p) {
    const Settings& s = p.settings;
    std::string t = hardware_text(p.hw, p.module);
    if (const offer::HwClass* c = offer::find_hw_class(p.match.cls))
        t += "Class:   " + class_text(*c, p.match.exact, p.hw) + "\n";
    t += "Tasks:   " + join_list(p.selection.tasks) + "\n";
    if (!p.selection.languages.empty()) t += "Langs:   " + join_list(p.selection.languages) + "\n";
    t += "Models:\n";
    for (const std::string& id : models_of(s)) t += model_text(p, id);
    const std::string dl = load_text({p.to_load, p.here});
    const size_t paren = dl.find(" (");
    t += std::format("Download: {} still to load{}, {} at an assumed {:.0f} Mbit/s; {} free, {} kept as buffer\n",
                     dl.substr(0, paren), paren == std::string::npos ? "" : dl.substr(paren),
                     duration_text(seconds_to_load(p.to_load)), kAssumedMbit,
                     size_text(static_cast<double>(p.free_bytes)), size_text(static_cast<double>(kBufferBytes)));
    if (p.module) {
        const RoomPlan r = room_plan(p.hw, s.use == "server", priority_from_id(s.priority), s.vram_gb);
        t += std::format(
            "Room:    Tona up to {:g} of {:g} GB RAM, own programs {:g} GB, system {:g} GB; use {}, priority {}\n",
            s.ram_gb, r.ram_total, r.programs, r.system, s.use, s.priority);
    }
    t += "\n" + settings_text(s);
    if (p.module) {
        t += "Estimate:\n";
        for (const std::string& id : models_of(s)) t += "  " + estimate_text(p.module->estimate(p.hw, s, id));
    }
    for (const Note& n : p.notes) t += std::string(n.warn ? "ERROR: " : "Note: ") + n.text + "\n";
    t += "\nPackages:\n" + packages_text(p.hw, p.packages);
    if (!p.steps.empty()) t += "\nSteps:\n" + steps_text(p.steps);
    return t;
}

std::string plan_json(const PlanDoc& p) {
    using nlohmann::json;
    const Settings& s = p.settings;
    json j;
    j["schema"] = 1;
    j["hardware"] = {{"class", p.match.cls},
                     {"class_exact", p.match.exact},
                     {"ram_bytes", p.hw.ram_total},
                     {"disk_free_bytes", p.hw.disk_free},
                     {"backend", p.module ? json(p.module->name()) : json()}};
    j["tasks"] = p.selection.tasks, j["languages"] = p.selection.languages;
    json models = json::array();
    for (const std::string& id : models_of(s)) {
        json m = {{"id", id},
                  {"role", id == s.model ? "primary" : kRoleExtra},
                  {"size_bytes", model_bytes(id)},
                  {"present", model_there(p.models_dir, id)}};
        m["speed"] = speed_json(p.module ? p.module->estimate(p.hw, s, id) : Estimate{});
        models.push_back(m);
    }
    j["models"] = models;
    j["drafter"] = s.drafter == "none" ? json() : json(drafter_entry_id(s.drafter, s.model));
    j["download"] = {{"bytes", p.to_load},
                     {"here_bytes", p.here},
                     {"seconds", seconds_to_load(p.to_load)},
                     {"assumed_mbit_per_s", kAssumedMbit}};
    j["space"] = {{"free_bytes", p.free_bytes},
                  {"buffer_bytes", kBufferBytes},
                  {"fits", p.to_load + kBufferBytes <= p.free_bytes}};
    j["room"] = {{"ram_limit_gb", s.ram_gb}, {"vram_limit_gb", s.vram_gb}, {"use", s.use}, {"priority", s.priority}};
    json notes = json::array();
    for (const Note& n : p.notes) notes.push_back({{"error", n.warn}, {"text", n.text}});
    j["notes"] = notes;
    j["packages_ok"] = packages_ok(p.packages);
    json steps = json::array();
    for (const Step& st : p.steps) steps.push_back({{"title", st.title}, {"command", st.cmd}, {"done", st.done}});
    j["steps"] = steps;
    j["exit_code"] = plan_exit_code(p);
    return j.dump(2) + "\n";
}

}  // namespace tona::install
