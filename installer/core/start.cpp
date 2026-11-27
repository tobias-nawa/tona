#include "core/start.h"

#include <filesystem>
#include <utility>

#include "core/config.h"
#include "core/context_plan.h"
#include "core/room.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::install {

std::string models_dir_of(const Target& t) {
    return t.release ? t.models_dir : (fs::path(t.repo) / "work/models").string();
}

namespace {

Settings with_conf(const Settings& rec, const Target& t, const Options& o, bool& had_conf) {
    had_conf = false;
    if (!o.conf.empty()) return had_conf = true, load(o.conf, rec);
    if (!t.release && fs::exists(conf_path(t.repo))) return had_conf = true, load(conf_path(t.repo), rec);
    return rec;
}

bool choose(const Options& o, const Start& st, bool interactive, Selection& sel, FlowError& err) {
    // A platform whose model is not in the offer data (the Mac, today) keeps what its module recommends
    if (!o.names_something() && resolve_model(st.recommended.model).empty()) return sel = {}, true;
    if (o.names_something()) return selection_from_names(o.named, st.match.cls, sel, err);
    if (st.had_conf || !interactive) {
        // headless on a fresh machine: the model the settings start with is the coding recommendation
        sel = selection_from_settings(st.settings, st.match.cls);
        return true;
    }
    sel = default_selection(st.match.cls);
    return true;
}

}  // namespace

bool make_start(const Options& o, const Target& t, bool interactive, Start& out, FlowError& err) {
    Start st;
    st.h = detect(t.dir());
    // once per run (0.3 s): the Mac's unified memory comes from its chip, not from a measurement
    if (st.h.os != "macos") st.h.ram_gbs = measure_ram_gbs(st.h.cores);
    st.h.disk_speed = t.release ? measure_disk(t.dir(), largest_model_file(t.models_dir)) : measure_disk(t.repo);
    st.module = find_module(st.h);
    st.match = match_hw_class(st.h);
    st.recommended = st.module ? st.module->recommend(st.h) : Settings{};
    st.settings = with_conf(st.recommended, t, o, st.had_conf);
    if (!choose(o, st, interactive, st.selection, err)) return false;
    st.settings = apply_selection(st.settings, st.selection);
    if (!st.had_conf || !o.priority.empty() || !o.use.empty()) {
        const bool server = (o.use.empty() ? st.settings.use : o.use) == "server";
        const Priority p = priority_from_id(o.priority.empty() ? st.settings.priority : o.priority);
        apply_room(st.settings, room_plan(st.h, server, p, st.settings.vram_gb), server, p);
    }
    resolve_context(st.module, st.settings);
    st.settings.model_from = o.model_from;
    if (!o.port.empty()) st.settings.port = std::stoi(o.port);
    if (o.autostart) st.settings.service = true;
    // Profiles are written only with a yes: a tick on the Finish screen (preselected) or --modify-path
    st.settings.modify_path = !o.no_modify_path && (interactive || o.modify_path);
    if (!o.updates.empty()) st.settings.update_checks = o.updates == "on";
    out = std::move(st);
    return true;
}

}  // namespace tona::install
