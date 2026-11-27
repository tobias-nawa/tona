#include "core/flow.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <utility>

#include "core/exit_codes.h"

namespace tona::install {

namespace {

constexpr double GIB = 1073741824.0;

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// The models a task offers on a class, recommendation first
std::vector<std::string> offered(const std::string& task, const std::string& cls) {
    std::vector<std::string> out;
    if (const offer::Offer* o = offer::offer_for(task, cls)) {
        out.push_back(o->recommended);
        if (!o->alternative.empty()) out.push_back(o->alternative);
    }
    return out;
}

// Order of the offer data: tasks as listed, a recommendation before its alternative; chosen ids only
void normalize(Selection& sel, const std::string& cls) {
    std::vector<std::string> tasks, models;
    for (const offer::Task& t : offer::tasks()) {
        if (!contains(sel.tasks, t.id)) continue;
        tasks.push_back(t.id);
        for (const std::string& m : offered(t.id, cls))
            if (contains(sel.models, m) && !contains(models, m)) models.push_back(m);
    }
    std::vector<std::string> langs;
    for (const offer::Language& l : offer::languages())
        if (contains(sel.languages, l.id)) langs.push_back(l.id);
    sel.tasks = tasks, sel.models = models, sel.languages = langs;
}

void flip(std::vector<std::string>& v, const std::string& id) {
    const auto it = std::find(v.begin(), v.end(), id);
    if (it == v.end())
        v.push_back(id);
    else
        v.erase(it);
}

bool offered_by_chosen_task(const Selection& sel, const std::string& model, const std::string& cls) {
    return std::any_of(sel.tasks.begin(), sel.tasks.end(),
                       [&](const std::string& t) { return contains(offered(t, cls), model); });
}

// "pc-<n>gb-vram-<m>gb-ram" of the offer data
bool parse_class(const std::string& id, int& vram_gb, int& ram_gb) {
    const std::string head = "pc-", mid = "gb-vram-", tail = "gb-ram";
    char* end = nullptr;
    if (!id.starts_with(head)) return false;
    vram_gb = static_cast<int>(std::strtol(id.c_str() + head.size(), &end, 10));
    if (!std::string(end).starts_with(mid)) return false;
    ram_gb = static_cast<int>(std::strtol(end + mid.size(), &end, 10));
    return std::string(end) == tail;
}

// "mac-<n>gb" of the offer data: unified memory, so the class is the memory size
bool parse_mac_class(const std::string& id, int& ram_gb) {
    const std::string head = "mac-", tail = "gb";
    char* end = nullptr;
    if (!id.starts_with(head)) return false;
    ram_gb = static_cast<int>(std::strtol(id.c_str() + head.size(), &end, 10));
    return std::string(end) == tail;
}

bool is_mac(const Hardware& h) {
    return h.os == "macos" ||
           std::any_of(h.gpus.begin(), h.gpus.end(), [](const Gpu& g) { return g.vendor == "Apple"; });
}

// The largest Mac class this memory reaches (the chip sets the speed, not the class: the estimate reads it). Below the
// smallest class: that class, not exact.
HwMatch match_mac_class(const Hardware& h) {
    const double ram = h.ram_total / GIB;
    HwMatch best, smallest;
    int best_ram = 0, smallest_ram = 0;
    for (const offer::HwClass& c : offer::hw_classes()) {
        int r = 0;
        if (!parse_mac_class(c.id, r)) continue;
        if (smallest.cls.empty() || r < smallest_ram) smallest = {c.id, false}, smallest_ram = r;
        if (r <= ram * 1.1 && r >= best_ram) best = {c.id, true}, best_ram = r;
    }
    return best.exact ? best : smallest;
}

const offer::Offer* coding_offer(const std::string& cls) { return offer::offer_for("coding", cls); }

offer::Level level_of(const std::string& model, const std::string& language) {
    const offer::ModelCard* c = offer::model_card(model);
    const offer::LanguageVerdict* v = c ? c->verdict(language) : nullptr;
    return v ? v->level : offer::Level::Open;
}

HwMatch match_pc_class(const Hardware& h) {
    double vram = 0;
    for (const Gpu& g : h.gpus)
        if (!g.integrated && g.vendor != "Apple") vram = std::max(vram, g.vram_total / GIB);
    const double ram = h.ram_total / GIB;
    HwMatch best;
    int best_vram = 0, best_ram = 0;
    for (const offer::HwClass& c : offer::hw_classes()) {
        int v = 0, r = 0;
        if (!parse_class(c.id, v, r)) continue;
        // the system reports a bit less than the installed memory (reserved ranges): 10 % slack
        if (v > vram * 1.1 || r > ram * 1.1 || v < best_vram || (v == best_vram && r < best_ram)) continue;
        best = {c.id, true}, best_vram = v, best_ram = r;
    }
    if (best.exact) return best;
    for (const offer::HwClass& c : offer::hw_classes())
        if (c.measured) return {c.id, false};
    return {offer::hw_classes().empty() ? "" : offer::hw_classes().front().id, false};
}

}  // namespace

bool contains(const std::vector<std::string>& v, const std::string& id) {
    return std::find(v.begin(), v.end(), id) != v.end();
}

HwMatch match_hw_class(const Hardware& h) {
    if (is_mac(h)) {
        if (const HwMatch m = match_mac_class(h); !m.cls.empty()) return m;
    }
    return match_pc_class(h);
}

Selection default_selection(const std::string& cls) {
    Selection sel;
    for (const char* t : {"coding", "knowledge"})
        if (offer::offer_for(t, cls)) toggle_task(sel, t, cls);
    return sel;
}

bool toggle_task(Selection& sel, const std::string& task, const std::string& cls) {
    const offer::Task* t = offer::find_task(task);
    if (!t || !t->available) return false;
    const bool on = contains(sel.tasks, task);
    flip(sel.tasks, task);
    if (!on) {
        if (const offer::Offer* o = offer::offer_for(task, cls); o && (!contains(sel.models, o->recommended)))
            sel.models.push_back(o->recommended);
    } else {
        std::vector<std::string> keep;
        for (const std::string& m : sel.models)
            if (offered_by_chosen_task(sel, m, cls)) keep.push_back(m);
        sel.models = std::move(keep);
    }
    normalize(sel, cls);
    return true;
}

void toggle_language(Selection& sel, const std::string& language) {
    flip(sel.languages, language);
    std::vector<std::string> langs;
    for (const offer::Language& l : offer::languages())
        if (contains(sel.languages, l.id)) langs.push_back(l.id);
    sel.languages = std::move(langs);
}

bool toggle_model(Selection& sel, const std::string& model, const std::string& cls) {
    if (!offered_by_chosen_task(sel, model, cls)) return false;
    flip(sel.models, model);
    normalize(sel, cls);
    return true;
}

std::vector<TaskModels> task_models(const Selection& sel, const std::string& cls) {
    std::vector<TaskModels> out;
    for (const std::string& id : sel.tasks) {
        const offer::Offer* o = offer::offer_for(id, cls);
        if (!o) continue;
        out.push_back({offer::find_task(id), o, offer::model_card(o->recommended),
                       o->alternative.empty() ? nullptr : offer::model_card(o->alternative)});
    }
    return out;
}

bool needs_languages(const Selection& sel) { return contains(sel.tasks, "coding"); }

std::vector<Page> page_order(const Selection& sel) {
    std::vector<Page> p = {Page::Machine, Page::Tasks};
    if (needs_languages(sel)) p.push_back(Page::Languages);
    for (Page x : {Page::Models, Page::Room, Page::Go}) p.push_back(x);
    return p;
}

Page next_page(Page from, const Selection& sel) {
    const std::vector<Page> p = page_order(sel);
    const auto it = std::find(p.begin(), p.end(), from);
    return it == p.end() || it + 1 == p.end() ? from : *(it + 1);
}

Page prev_page(Page from, const Selection& sel) {
    const std::vector<Page> p = page_order(sel);
    const auto it = std::find(p.begin(), p.end(), from);
    return it == p.end() || it == p.begin() ? from : *(it - 1);
}

std::string primary_model(const Selection& sel) { return sel.models.empty() ? "" : sel.models.front(); }

std::string default_drafter(const std::string& model) { return model == "coder-iq1_m" ? "mtp" : "none"; }

std::vector<LanguageProposal> language_proposals(const Selection& sel, const std::string& cls) {
    std::vector<LanguageProposal> out;
    const offer::Offer* o = coding_offer(cls);
    if (!o) return out;
    for (const std::string& lang : sel.languages) {
        LanguageProposal p{lang, o->recommended, level_of(o->recommended, lang), offer::Basis::None};
        if (!o->alternative.empty() && p.level != offer::Level::Strong &&
            level_of(o->alternative, lang) == offer::Level::Strong)
            p.model = o->alternative, p.level = offer::Level::Strong;
        if (const offer::ModelCard* c = offer::model_card(p.model))
            if (const offer::LanguageVerdict* v = c->verdict(lang)) p.basis = v->basis;
        out.push_back(p);
    }
    return out;
}

std::vector<std::string> proposed_models(const Selection& sel, const std::string& cls) {
    std::vector<std::string> out;
    for (const LanguageProposal& p : language_proposals(sel, cls))
        if (!contains(out, p.model) && offered_by_chosen_task(sel, p.model, cls)) out.push_back(p.model);
    return out;
}

std::vector<std::string> models_to_consider(const Selection& sel, const std::string& cls) {
    std::vector<std::string> out;
    if (!needs_languages(sel)) return out;
    for (const std::string& lang : sel.languages) {
        const bool covered = std::any_of(sel.models.begin(), sel.models.end(), [&](const std::string& m) {
            return level_of(m, lang) == offer::Level::Strong;
        });
        if (covered) continue;
        for (const std::string& m : offered("coding", cls))
            if (!contains(sel.models, m) && !contains(out, m) && level_of(m, lang) == offer::Level::Strong)
                out.push_back(m);
    }
    return out;
}

std::string resolve_model(const std::string& name) {
    const std::string n = lower(name);
    if (n.empty()) return "";
    for (const offer::ModelCard& c : offer::model_cards())
        if (lower(c.id) == n || lower(c.name) == n) return c.id;
    for (const offer::ModelCard& c : offer::model_cards())
        if (lower(c.id).starts_with(n)) return c.id;
    return "";
}

Settings apply_selection(Settings s, const Selection& sel) {
    s.languages = sel.languages;
    if (sel.models.empty()) return s;  // a platform without offer data keeps what its module recommends
    s.models = sel.models, s.model = primary_model(sel);
    s.drafter = default_drafter(s.model);
    return s;
}

std::vector<std::string> split_list(const std::string& csv) {
    std::vector<std::string> out;
    size_t from = 0;
    while (from <= csv.size()) {
        const size_t to = csv.find(',', from);
        const std::string part = csv.substr(from, to == std::string::npos ? std::string::npos : to - from);
        if (!part.empty()) out.push_back(part);
        if (to == std::string::npos) break;
        from = to + 1;
    }
    return out;
}

std::string join_list(const std::vector<std::string>& v) {
    std::string o;
    for (const std::string& x : v) o += (o.empty() ? "" : ",") + x;
    return o;
}

namespace {

bool fail(FlowError& err, int code, std::string text) {
    err = {code, std::move(text)};
    return false;
}

bool add_languages(const NamedChoice& in, Selection& out, FlowError& err) {
    for (const std::string& name : in.languages) {
        const auto it = std::find_if(offer::languages().begin(), offer::languages().end(), [&](const auto& l) {
            return lower(l.id) == lower(name) || lower(l.label) == lower(name);
        });
        if (it == offer::languages().end()) return fail(err, kExitUsage, "unknown language: " + name);
        if (!contains(out.languages, it->id)) out.languages.push_back(it->id);
    }
    return true;
}

bool add_tasks(const NamedChoice& in, const std::string& cls, Selection& out, FlowError& err) {
    for (const std::string& name : in.tasks) {
        const offer::Task* t = offer::find_task(lower(name));
        if (!t) return fail(err, kExitUsage, "unknown task: " + name);
        if (!t->available)
            return fail(err, kExitTaskComing, "task " + t->id + " is not available yet: " + t->coming_note);
        toggle_task(out, t->id, cls);
    }
    return true;
}

// The models decide; each task that offers one of them is chosen with it
bool add_models(const NamedChoice& in, const std::string& cls, Selection& out, FlowError& err) {
    for (const std::string& name : in.models) {
        const std::string id = resolve_model(name);
        if (id.empty()) return fail(err, kExitUsage, "unknown model: " + name);
        bool known = false;
        for (const offer::Task& t : offer::tasks())
            if (t.available && contains(offered(t.id, cls), id)) {
                known = true;
                if (!contains(out.tasks, t.id)) out.tasks.push_back(t.id);
                if (!contains(out.models, id)) out.models.push_back(id);
            }
        if (!known) return fail(err, kExitUsage, "model " + id + " is not offered for this hardware class");
    }
    return true;
}

}  // namespace

bool selection_from_names(const NamedChoice& in, const std::string& cls, Selection& out, FlowError& err) {
    out = {};
    if (in.tasks.empty() && in.models.empty() && in.languages.empty()) {
        out = default_selection(cls);
        return true;
    }
    if (!in.models.empty()) {
        if (!add_models(in, cls, out, err)) return false;
    } else if (!in.tasks.empty()) {
        if (!add_tasks(in, cls, out, err)) return false;
    } else {
        out = default_selection(cls);
    }
    if (!in.models.empty() && !in.tasks.empty()) {  // tasks named next to models are checked, not added alone
        Selection probe;
        if (!add_tasks(in, cls, probe, err)) return false;
    }
    if (!add_languages(in, out, err)) return false;
    normalize(out, cls);
    return true;
}

Selection selection_from_settings(const Settings& s, const std::string& cls) {
    Selection sel;
    FlowError err;
    NamedChoice named;
    named.models = models_of(s);
    named.languages = s.languages;
    if (selection_from_names(named, cls, sel, err) && !sel.models.empty()) return sel;
    return default_selection(cls);
}

}  // namespace tona::install
