// Wizard core: the frame around the pages, navigation and the keys that mean the same everywhere. The pages are in
// tui_pages.cpp, tui_room.cpp and tui_states.cpp, the installation thread in tui_install.cpp.
#include "tui.h"

#include <algorithm>
#include <cstdio>
#include <format>
#include <utility>

#include "core/pkgs.h"
#include "ftxui/screen/terminal.hpp"
#include "tui_widgets.h"

using namespace ftxui;

namespace tona::install {

Wizard::Wizard(Start start, Options opts, Target target)
    : target_(std::move(target)), opts_(std::move(opts)), st_(std::move(start)) {
    pk_ = check_packages(st_.h, st_.module ? st_.module->packages(st_.h) : std::vector<Package>{}, target_.release);
    server_only_ = s().use == "server";
    priority_ = priority_from_id(s().priority);
    if (!st_.module) view_ = View::NoGpu;
}

Wizard::~Wizard() {
    if (worker_.joinable()) worker_.join();
}

std::string Wizard::page_name(Page p) const {
    switch (p) {
        case Page::Machine:
            return "Machine";
        case Page::Tasks:
            return "Tasks";
        case Page::Languages:
            return "Languages";
        case Page::Models:
            return "Models";
        case Page::Room:
            return "Room";
        case Page::Go:
            return "Go";
    }
    return "";
}

Element Wizard::crumbs() {
    Elements parts;
    for (Page p : page_order(st_.selection)) {
        if (!parts.empty()) parts.push_back(ui::dim(" › "));
        parts.push_back(p == page_ ? ui::title(page_name(p)) | inverted : ui::dim(page_name(p)));
    }
    return hbox(parts);
}

Element Wizard::frame(const Element& content, const std::string& keys) {
    // while the steps run and after them the pages are behind us: no step count of the pages, no crumbs
    if (installing_ && page_ == Page::Go && view_ == View::Pages) {
        const auto [head, tail] = run_header();
        return vbox({ui::spread(head, tail), separator(), content | size(HEIGHT, EQUAL, ui::kBody + 1), separator(),
                     ui::dim(keys)});
    }
    const std::vector<Page> order = page_order(st_.selection);
    const auto at = std::find(order.begin(), order.end(), page_);
    const std::string step = std::format("{}/{}", at - order.begin() + 1, order.size());
    return vbox({ui::spread("Set up Tona", "tona setup  " + step), crumbs(), separator(),
                 content | size(HEIGHT, EQUAL, ui::kBody), separator(), ui::dim(keys)});
}

Element Wizard::too_small(int w, int h) {
    return vbox({ui::bad("The terminal is too small."), ui::plain("Tona needs at least 80 × 24 characters."),
                 ui::plain(std::format("{} {} × {}.", "Right now it is", w, h)),
                 ui::plain("Make the window larger and it goes on at once."), ui::dim("q  close")});
}

Element Wizard::body() {
    switch (view_) {
        case View::Details:
            return details_body();
        case View::Advanced:
            return advanced_body();
        case View::Help:
            return help_body();
        case View::NoGpu:
            return no_gpu_body();
        case View::NoSpace:
            return no_space_body();
        case View::Failed:
            return failed_body();
        case View::Pages:
            break;
    }
    switch (page_) {
        case Page::Machine:
            return machine_body();
        case Page::Tasks:
            return tasks_body();
        case Page::Languages:
            return languages_body();
        case Page::Models:
            return models_body();
        case Page::Room:
            return room_body();
        case Page::Go:
            return go_body();
    }
    return ui::blank();
}

namespace {

// The key line of every page and state: one text per line
enum KeyLine {
    kHelp,
    kDetails,
    kAdvanced,
    kNoGpu,
    kNoSpace,
    kFailed,
    kMachine,
    kTasks,
    kLanguages,
    kModels,
    kRoom,
    kGoReady,
    kGoRunning,
    kGoDone,
    kFinish
};
constexpr struct {
    const char* en;
} kKeyTexts[] = {
    {"Esc back"},
    {"Esc back  ? help"},
    {"↑↓ row  ←→ choice  type a number  Esc back  ? help"},
    {"↑↓ choose  Enter go on  b report  ? help  q quit"},
    {"↑↓ choose  Enter take  p check again  Esc back  ? help"},
    {"↑↓ choose  Enter run  l log  ? help"},
    {"Enter next  ? help  q quit"},
    {"↑↓ choose  Space on/off  Enter next  Esc back  ? help"},
    {"←↑↓→ choose  Space on/off  Enter next  Esc back  ? help"},
    {"↑↓ choose  Space on/off  i details  Enter next  Esc back  ? help"},
    {"↑↓ row  ←→ choice  e more  Enter next  Esc back  ? help"},
    {"Enter start  e more  Esc back  ? help"},
    {"l log  Ctrl+C ends after the step; a download resumes where it left off"},
    {"Enter finish  l log"},
    {"↑↓ choose  Space on/off  Enter apply"},
};

}  // namespace

int Wizard::page_keys() const {
    switch (page_) {
        case Page::Machine:
            return kMachine;
        case Page::Tasks:
            return kTasks;
        case Page::Languages:
            return kLanguages;
        case Page::Models:
            return kModels;
        case Page::Room:
            return kRoom;
        case Page::Go:
            break;
    }
    return !installing_ ? kGoReady : finishing_now() ? kFinish : finished_ ? kGoDone : kGoRunning;
}

std::string Wizard::keys_line() {
    int k = kMachine;
    switch (view_) {
        case View::Help:
            k = kHelp;
            break;
        case View::Details:
            k = kDetails;
            break;
        case View::Advanced:
            k = kAdvanced;
            break;
        case View::NoGpu:
            k = kNoGpu;
            break;
        case View::NoSpace:
            k = kNoSpace;
            break;
        case View::Failed:
            k = kFailed;
            break;
        case View::Pages:
            k = page_keys();
            break;
    }
    return kKeyTexts[k].en;
}

// The installation thread cannot touch the view; the screen follows its flag here
void Wizard::sync_view() {
    if (fetch_failed_ && finished_ && page_ == Page::Go && view_ == View::Pages) view_ = View::Failed, cursor_ = 0;
}

Element Wizard::render() {
    sync_view();
    const Dimensions d = Terminal::Size();
    if (d.dimx < ui::kWidth || d.dimy < ui::kHeight) return too_small(d.dimx, d.dimy);
    return frame(body(), keys_line());
}

std::vector<int> Wizard::task_rows() const {
    std::vector<int> rows;
    for (size_t i = 0; i < offer::tasks().size(); i++)
        if (offer::tasks()[i].available && offer::offer_for(offer::tasks()[i].id, st_.match.cls))
            rows.push_back(static_cast<int>(i));
    return rows;
}

void Wizard::enter_page(Page p) {
    page_ = p, cursor_ = 0, message_.clear(), view_ = View::Pages;
    if (p == Page::Room) room_changed();
    if (p == Page::Go) plan_ = plan_steps(st_.h, s(), target_);
}

// Per language once: a later visit to the language page keeps what the user switched off on the model page
void Wizard::adopt_language_models() {
    Selection fresh = st_.selection;
    fresh.languages.clear();
    for (const std::string& l : st_.selection.languages)
        if (!contains(seen_languages_, l)) fresh.languages.push_back(l);
    seen_languages_ = st_.selection.languages;
    const std::string dir = models_dir_of(target_);
    for (const std::string& id : proposed_models(fresh, st_.match.cls)) {
        if (contains(st_.selection.models, id)) continue;
        Selection trial = st_.selection;
        toggle_model(trial, id, st_.match.cls);
        if (fits(apply_selection(s(), trial), dir, static_cast<uint64_t>(st_.h.disk_free)))
            st_.selection = trial;
        else if (!contains(unfit_, id))
            unfit_.push_back(id);
    }
    s() = apply_selection(s(), st_.selection);
}

void Wizard::forward() {
    if (page_ == Page::Tasks && st_.selection.tasks.empty()) {
        message_ = "Choose at least one task.";
        return;
    }
    if (page_ == Page::Models) {
        if (st_.selection.models.empty()) {
            message_ = "Choose at least one model.";
            return;
        }
        s() = apply_selection(s(), st_.selection);
        check_space();
        if (view_ == View::NoSpace) return;
    }
    if (page_ == Page::Languages) adopt_language_models();
    if (page_ != Page::Go) enter_page(next_page(page_, st_.selection));
}

void Wizard::back() {
    if (installing_ && !finished_) return;
    if (page_ != Page::Machine) enter_page(prev_page(page_, st_.selection));
}

// Keys with the same meaning on every page and state
bool Wizard::key_common(const Event& e) {
    if (e == Event::Character('?') && view_ != View::Help) {
        before_help_ = view_;
        view_ = View::Help;
        return true;
    }
    if (view_ == View::Help) {
        if (e == Event::Escape || e == Event::Character('?') || e == Event::Backspace) view_ = before_help_;
        return true;
    }
    if (e == Event::Character('q')) {
        if (installing_ && !finished_) {
            message_ =
                "Still running. Ctrl+C ends after the current step; what is loaded stays and the next run goes on.";
            return true;
        }
        screen_.Exit();
        return true;
    }
    return false;
}

// Keys of a state that covers the pages
bool Wizard::on_view_event(const Event& e) {
    switch (view_) {
        case View::Advanced:
            return key_advanced(e);
        case View::NoGpu:
            return key_no_gpu(e);
        case View::NoSpace:
            return key_no_space(e);
        case View::Failed:
            return key_failed(e);
        case View::Details:
            if (ui::is_back(e) || e == Event::Character('i')) view_ = View::Pages;
            return true;
        case View::Help:
        case View::Pages:
            break;
    }
    return true;
}

bool Wizard::on_event(const Event& e) {
    sync_view();
    const Dimensions d = Terminal::Size();
    if (d.dimx < ui::kWidth || d.dimy < ui::kHeight) return e == Event::Character('q') ? (screen_.Exit(), true) : true;
    if (editing_) return key_advanced(e);
    if (key_common(e)) return true;
    if (view_ != View::Pages) return on_view_event(e);
    switch (page_) {
        case Page::Machine:
            if (e == Event::Return) forward();
            return true;
        case Page::Tasks:
            return key_tasks(e);
        case Page::Languages:
            return key_languages(e);
        case Page::Models:
            return key_models(e);
        case Page::Room:
            return key_room(e);
        case Page::Go:
            return key_go(e);
    }
    return false;
}

int Wizard::run() {
    auto ui_component =
        Renderer([this] { return render(); }) | CatchEvent([this](const Event& e) { return on_event(e); });
    screen_.Loop(ui_component);
    if (worker_.joinable()) worker_.join();
    return result_;
}

}  // namespace tona::install

int run_tui(const tona::install::Options& o, const tona::install::Target& target) {
    using namespace tona::install;
    Start st;
    FlowError err;
    if (!make_start(o, target, true, st, err)) {
        std::fprintf(stderr, "tona setup: %s\n", err.text.c_str());
        return err.code;
    }
    Wizard w(std::move(st), o, target);
    return w.run();
}
