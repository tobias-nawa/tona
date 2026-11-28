// The first four pages: machine, tasks, languages, models, and the details of a model.
#include <algorithm>
#include <cmath>
#include <format>

#include "core/context_plan.h"
#include "core/describe.h"
#include "core/sys.h"
#include "tui.h"
#include "tui_widgets.h"

using namespace ftxui;

namespace tona::install {

namespace {

constexpr double GIB = 1073741824.0;

// "~/.local/share/tona" for a path below the home folder
std::string short_path(const std::string& p) {
    const std::string h = home();
    return !h.empty() && p.starts_with(h) ? "~" + p.substr(h.size()) : p;
}

Element machine_row(const std::string& label, const std::string& value, const std::string& mark, bool ok) {
    return hbox({ui::plain(ui::pad(label, 16) + ui::pad(value, 46)), ok ? ui::good(" " + mark) : ui::bad(" " + mark)});
}

const Gpu* shown_gpu(const Hardware& h) {
    for (const Gpu& g : h.gpus)
        if (!g.integrated) return &g;
    return h.gpus.empty() ? nullptr : &h.gpus.front();
}

std::string gpu_text(const Gpu* g) {
    if (!g) return "none found";
    const std::string name = g->name.empty() ? g->arch : g->name;
    // Apple names the chip "Apple M1 Max" already
    const std::string who = name.starts_with(g->vendor) ? name : g->vendor + " " + name;
    return std::format("{} · {:.0f} GB", who, std::round(static_cast<double>(g->vram_total) / GIB));
}

std::string model_names_of(const std::string& cls) {
    std::string out;
    for (const offer::Task& t : offer::tasks()) {
        const offer::Offer* o = offer::offer_for(t.id, cls);
        if (!o) continue;
        for (const std::string& id : {o->recommended, o->alternative})
            if (const offer::ModelCard* c = id.empty() ? nullptr : offer::model_card(id))
                out += (out.empty() ? "" : ", ") + c->name;
    }
    return out;
}

Element basis_element_for(const std::string& word, offer::Basis b) {
    return b == offer::Basis::Measured      ? ui::good(word)
           : b == offer::Basis::Provisional ? ui::warn(word)
                                            : ui::plain(word);
}

// The models a user can switch on the model page, in the order of the page
std::vector<std::string> model_rows(const Selection& sel, const std::string& cls) {
    std::vector<std::string> rows;
    for (const TaskModels& tm : task_models(sel, cls)) {
        rows.push_back(tm.recommended->id);
        if (tm.alternative) rows.push_back(tm.alternative->id);
    }
    return rows;
}

std::string chosen_language_labels(const Selection& sel) {
    std::string out;
    for (const std::string& id : sel.languages)
        for (const offer::Language& l : offer::languages())
            if (l.id == id) out += (out.empty() ? "" : ", ") + l.label;
    return out;
}

void push_labeled_verdict(Elements& parts, const std::string& text, bool is_good) {
    const size_t colon = text.find(':');
    const auto accent = [is_good](const std::string& s) { return is_good ? ui::good(s) : ui::warn(s); };
    if (colon == std::string::npos) {
        parts.push_back(accent(text));
        return;
    }
    parts.push_back(accent(text.substr(0, colon + 1)));
    parts.push_back(ui::dim(text.substr(colon + 1)));
}

Element verdict_row(const VerdictParts& v) {
    Elements parts = {ui::plain("      ")};
    if (!v.strong.empty()) push_labeled_verdict(parts, v.strong, true);
    if (!v.strong.empty() && !v.weak.empty()) parts.push_back(ui::dim(" · "));
    if (!v.weak.empty()) push_labeled_verdict(parts, v.weak, false);
    return hbox(std::move(parts));
}

}  // namespace

Elements Wizard::machine_rows() {
    const Hardware& h = st_.h;
    const Gpu* g = shown_gpu(h);
    const bool gpu_ok = st_.module != nullptr;
    Elements l;
    l.push_back(machine_row("Graphics card", gpu_text(g), gpu_ok ? "ok" : "not supported", gpu_ok));
    l.push_back(machine_row("Processor", std::format("{} · {} {}", h.cpu, h.cores, "cores"), "ok", true));
    l.push_back(machine_row("Memory", std::format("{:.0f} GB", nominal_ram_gb(h)), "ok", true));
    l.push_back(machine_row("Free disk",
                            size_text(static_cast<double>(h.disk_free)) + " " + "under " + short_path(target_.dir()),
                            "ok", h.disk_free > 0));
    if (g && g->vram_used > 300'000'000)
        l.push_back(machine_row(
            "Screen",
            std::format("{} {:.1f} GB, {}", "uses", static_cast<double>(g->vram_used) / 1e9, "Tona leaves it room"),
            "ok", true));
    return l;
}

Element Wizard::machine_body() {
    Elements l = {ui::title("Your machine"), ui::blank()};
    for (Element& r : machine_rows()) l.push_back(std::move(r));
    const Hardware& h = st_.h;
    l.push_back(ui::blank());
    if (const offer::HwClass* c = offer::find_hw_class(st_.match.cls)) {
        l.push_back(
            hbox({ui::plain(ui::pad("Your class", 16)), ui::title(class_title(*c, st_.match.exact, h.os == "macos"))}));
        l.push_back(ui::plain(ui::pad("Models", 16) + model_names_of(st_.match.cls)));
    }
    for (const Note& n : st_.module ? st_.module->check(h, s()) : std::vector<Note>{})
        l.push_back(n.warn ? ui::bad(ui::fit("! " + n.text, 78)) : ui::warn(ui::fit("· " + n.text, 78)));
    return vbox(l);
}

Element Wizard::tasks_body() {
    Elements l = {ui::spread("What do you want to do?", "several possible"), ui::blank()};
    const std::vector<int> rows = task_rows();
    for (size_t i = 0; i < offer::tasks().size(); i++) {
        const offer::Task& t = offer::tasks()[i];
        const auto at = std::find(rows.begin(), rows.end(), static_cast<int>(i));
        const bool here = at != rows.end() && static_cast<int>(at - rows.begin()) == cursor_;
        const std::string head = ui::cursor(here) +
                                 (t.available ? ui::box(contains(st_.selection.tasks, t.id)) : "[–]") + " " +
                                 ui::pad(t.label, 24);
        if (t.available && at != rows.end()) {
            Element row = here ? ui::plain(head + t.help) : hbox({ui::plain(head), ui::dim(t.help)});
            l.push_back(ui::highlight(std::move(row), here));
        } else if (t.available) {
            l.push_back(
                ui::dim("[–] " + ui::pad(t.label, 24) + "not on " + this_machine(st_.h.os == "macos") + " yet"));
        } else {
            l.push_back(ui::dim(head + ui::fit(t.coming_note, 79 - 31)));
        }
    }
    if (!message_.empty()) l.push_back(ui::warn(message_));
    return vbox(l);
}

bool Wizard::key_tasks(const Event& e) {
    const std::vector<int> rows = task_rows();
    if (ui::is_down(e) && cursor_ + 1 < static_cast<int>(rows.size())) return ++cursor_, true;
    if (ui::is_up(e) && cursor_ > 0) return --cursor_, true;
    if (ui::is_space(e) && cursor_ < static_cast<int>(rows.size())) {
        toggle_task(st_.selection, offer::tasks()[rows[cursor_]].id, st_.match.cls);
        s() = apply_selection(s(), st_.selection);
        message_.clear();
        return true;
    }
    if (e == Event::Return) return forward(), true;
    if (ui::is_back(e)) return back(), true;
    return true;
}

Element Wizard::languages_body() const {
    Elements l = {ui::title("Which languages do you program in?"), ui::blank()};
    const auto& langs = offer::languages();
    for (size_t row = 0; row * 4 < langs.size(); row++) {
        Elements cols;
        for (size_t col = 0; col < 4 && row * 4 + col < langs.size(); col++) {
            const size_t i = row * 4 + col;
            const bool here = static_cast<int>(i) == cursor_;
            const std::string cell = ui::cursor(here) + ui::box(contains(st_.selection.languages, langs[i].id)) + " " +
                                     ui::pad(langs[i].label, 12);
            cols.push_back(ui::highlight_cell(cell, here));
        }
        l.push_back(hbox(std::move(cols)));
    }
    const std::vector<LanguageProposal> props = language_proposals(st_.selection, st_.match.cls);
    if (!props.empty()) {
        l.push_back(ui::blank());
        l.push_back(ui::title("Proposal for your choice"));
    }
    for (size_t i = 0; i < props.size() && i < 7; i++) {
        const offer::ModelCard* c = offer::model_card(props[i].model);
        std::string label;
        for (const offer::Language& lg : langs)
            if (lg.id == props[i].language) label = lg.label;
        const std::string word = level_text(props[i].level, props[i].basis);
        // weak is never the green of strong: the basis colours only a strong verdict
        const Element verdict = props[i].level == offer::Level::Open   ? ui::dim(word)
                                : props[i].level == offer::Level::Weak ? ui::warn(word)
                                                                       : basis_element_for(word, props[i].basis);
        l.push_back(hbox({ui::plain(ui::pad(label, 14) + "→  " + ui::pad(c ? c->name : props[i].model, 9)), verdict}));
    }
    if (props.size() > 7) l.push_back(ui::dim(std::format("… +{}", props.size() - 7)));
    return vbox(l);
}

bool Wizard::key_languages(const Event& e) {
    const int n = static_cast<int>(offer::languages().size());
    if (ui::is_right(e))
        cursor_ = std::min(cursor_ + 1, n - 1);
    else if (ui::is_left(e))
        cursor_ = std::max(cursor_ - 1, 0);
    else if (ui::is_down(e))
        cursor_ = std::min(cursor_ + 4, n - 1);
    else if (ui::is_up(e))
        cursor_ = std::max(cursor_ - 4, 0);
    else if (ui::is_space(e))
        toggle_language(st_.selection, offer::languages()[cursor_].id), s() = apply_selection(s(), st_.selection);
    else if (e == Event::Return)
        forward();
    else if (ui::is_back(e))
        back();
    return true;
}

// Per task a group: the rows in fixed columns (name, speed, size, status), then what the model is strong and weak in
Elements Wizard::model_list(const std::string& dir) const {
    // the column names above the rows, in the columns of the rows (cursor and box take 6)
    Elements l = {ui::dim(ui::pad("", 6) + ui::pad("Model", 22) + ui::pad("Speed per agent", 17) + ui::pad("Size", 10) +
                          "Status")};
    int row = 0;
    for (const TaskModels& tm : task_models(st_.selection, st_.match.cls)) {
        const std::string langs = tm.task->id == "coding" ? chosen_language_labels(st_.selection) : "";
        if (l.size() > 1) l.push_back(ui::blank());
        l.push_back(langs.empty() ? ui::title(tm.task->label)
                                  : hbox({ui::title(tm.task->label), ui::dim(" · " + langs)}));
        for (const offer::ModelCard* c : {tm.recommended, tm.alternative}) {
            if (!c) continue;
            const bool here = row == cursor_;
            const bool present = model_split(c->id, dir).download == 0;
            const std::string name = c->name + (c == tm.recommended ? "" : " (alternative)");
            const std::string prefix = ui::cursor(here) + ui::box(contains(st_.selection.models, c->id)) + " " +
                                       ui::pad(name, 22) + ui::pad(speed_text(model_estimate(c->id)), 17) +
                                       ui::pad(size_text(static_cast<double>(model_bytes(c->id))), 10);
            const std::string status = present ? "already here" : "download";
            Element line = here ? ui::plain(prefix + status)
                                : hbox({ui::plain(prefix), present ? ui::good(status) : ui::plain(status)});
            l.push_back(ui::highlight(std::move(line), here));
            l.push_back(verdict_row(verdict_parts(*c)));
            row++;
        }
    }
    return l;
}

// The sums below a rule, then the lines about models that are off or worth adding
Elements Wizard::model_totals(const std::string& dir) const {
    const LoadSplit load = load_split(apply_selection(s(), st_.selection), dir);
    std::string rule;
    for (int i = 0; i < 78; i++) rule += "─";
    Elements l = {
        ui::dim(rule),
        hbox({ui::dim(ui::pad("To download", 14)), ui::title(load_text(load))}),
        hbox({ui::dim(ui::pad("Free", 14)), ui::plain(ui::pad(size_text(static_cast<double>(st_.h.disk_free)), 10)),
              ui::dim("in " + short_path(target_.dir()))}),
        hbox({ui::dim(ui::pad("Time", 14)), ui::plain(duration_text(seconds_to_load(load.download))),
              ui::dim(std::format("  (assumed {:.0f} Mbit/s)", kAssumedMbit))}),
    };
    for (const std::string& id : unfit_)
        if (const offer::ModelCard* c = offer::model_card(id); c && !contains(st_.selection.models, id))
            l.push_back(ui::warn(ui::fit(std::format("{} ({}) is off: not enough free disk", c->name,
                                                     size_text(static_cast<double>(model_bytes(c->id)))),
                                         78)));
    for (const std::string& id : models_to_consider(st_.selection, st_.match.cls))
        if (const offer::ModelCard* c = offer::model_card(id); c && !contains(unfit_, id))
            l.push_back(ui::warn(
                ui::fit(std::format("{} {}: {} ({})", c->name, "is stronger in your languages",
                                    "add it with the space bar", size_text(static_cast<double>(model_bytes(c->id)))),
                        78)));
    return l;
}

Element Wizard::models_body() {
    const std::string dir = models_dir_of(target_);
    // the preselection follows the languages of the page before
    Elements l = {st_.selection.languages.empty() ? ui::title("Your models")
                                                  : ui::spread("Your models", "preselected from your languages")};
    for (Element& e : model_list(dir)) l.push_back(std::move(e));
    for (Element& e : model_totals(dir)) l.push_back(std::move(e));
    if (!message_.empty()) l.push_back(ui::warn(message_));
    l.push_back(ui::dim(speed_footer(st_.h.os == "macos")));
    return vbox(l);
}

bool Wizard::key_models(const Event& e) {
    const std::vector<std::string> rows = model_rows(st_.selection, st_.match.cls);
    const int n = static_cast<int>(rows.size());
    if (ui::is_down(e) && cursor_ + 1 < n) return ++cursor_, true;
    if (ui::is_up(e) && cursor_ > 0) return --cursor_, true;
    if (ui::is_space(e) && cursor_ < n) {
        toggle_model(st_.selection, rows[cursor_], st_.match.cls);
        s() = apply_selection(s(), st_.selection);
        resolve_context(st_.module, s());  // the model sets how much RAM its experts need
        message_.clear();
        return true;
    }
    if (e == Event::Character('i') && cursor_ < n) return view_ = View::Details, true;
    if (e == Event::Return) return forward(), true;
    if (ui::is_back(e)) return back(), true;
    return true;
}

Estimate Wizard::model_estimate(const std::string& id) const {
    return st_.module ? st_.module->estimate(st_.h, s(), id) : Estimate{};
}

Element Wizard::details_body() const {
    const std::vector<std::string> rows = model_rows(st_.selection, st_.match.cls);
    const offer::ModelCard* c = cursor_ < static_cast<int>(rows.size()) ? offer::model_card(rows[cursor_]) : nullptr;
    if (!c) return ui::plain("");
    Elements l = {ui::title(std::format("{} · {}", c->name, size_text(static_cast<double>(model_bytes(c->id)))))};
    l.push_back(ui::blank());
    for (const std::string& line : speed_detail_lines(model_estimate(c->id)))
        for (Element& w : ui::wrap(line)) l.push_back(std::move(w));
    l.push_back(ui::blank());
    for (const std::string& line : detail_lines(*c))
        for (Element& w : ui::wrap(line)) l.push_back(std::move(w));
    return vbox(l);
}

}  // namespace tona::install
