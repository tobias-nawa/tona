// The last page and the states of the design: no suitable graphics card, too little space, download failed, help.
#include <format>

#include "core/describe.h"
#include "tui.h"
#include "tui_widgets.h"

using namespace ftxui;

namespace tona::install {

namespace {

const char* priority_word(Priority p) {
    return p == Priority::Work ? "work first" : p == Priority::Tona ? "Tona first" : "balanced";
}

}  // namespace

Element plan_line(const std::string& prefix, const std::string& text) {
    const size_t at = text.find("(already here)");
    if (at == std::string::npos) return hbox({ui::dim(prefix), ui::plain(text)});
    return hbox({ui::dim(prefix), ui::plain(text.substr(0, at)), ui::good(text.substr(at))});
}

// The numbered steps of the plan; a step without text (the drafter, part of its model) has no row
Elements Wizard::plan_rows() const {
    Elements l;
    size_t number = 0;
    for (size_t i = 0; i < plan_.size(); i++) {
        const std::string label = step_label(plan_[i]);
        if (label.empty()) continue;
        const std::string text = label + (plan_[i].done && !label.contains("already here") ? " (done)" : "");
        number++;
        const std::vector<std::string> lines = ui::wrap_lines(text, 72);
        for (size_t k = 0; k < lines.size(); k++)
            l.push_back(plan_line(k == 0 ? std::format("{:>2}. ", number) : "    ", lines[k]));
    }
    return l;
}

Element Wizard::go_body() {
    if (installing_) return finishing() ? finish_body() : step_rows();
    const bool mac = st_.h.os == "macos";
    Elements l = {ui::title("Ready to set up")};
    for (Element& r : plan_rows()) l.push_back(std::move(r));
    // all steps stay; the blank line before the summary goes first when the page is full (the title and the rows of
    // settings)
    const bool gap = l.size() + 6 <= static_cast<size_t>(ui::kBody);
    if (gap) l.push_back(ui::blank());
    l.push_back(ui::title("What you get"));
    const auto kv = [](const char* k, const std::string& v) { return hbox({ui::dim(ui::pad(k, 12)), ui::plain(v)}); };
    l.push_back(
        kv("Server", std::format("port {} · {} · {}", s().port,
                                 s().network ? other_machines(mac) + " with a key" : this_machine(mac) + " only",
                                 s().service ? "starts at login" : "not at login")));
    l.push_back(kv("Memory", std::format("{} · Tona {:g} GB, video up to {:g} GB", priority_word(priority_), s().ram_gb,
                                         s().vram_gb)));
    l.push_back(kv("Agents", std::format("{} · {}", s().seqs,
                                         s().ctx_auto ? std::string("context auto")
                                                      : std::format("context {}K", s().ctx / 1024))));
    const uint64_t load = load_split(s(), models_dir_of(target_)).download;
    l.push_back(kv("Download",
                   load == 0 ? std::string("nothing, all models are here")
                             : size_text(static_cast<double>(load)) + " · " + duration_text(seconds_to_load(load))));
    if (!message_.empty()) l.push_back(ui::warn(message_));
    return vbox(l);
}

bool Wizard::key_go(const Event& e) {
    if (installing_) {
        if (finishing()) return key_finish(e);
        if (finished_ && result_ == 0 && e == Event::Return) screen_.Exit();
        if (e == Event::Character('l')) show_log_ = !show_log_;
        return true;
    }
    if (e == Event::Return) return start_install(), true;
    if (e == Event::Character('e')) return view_ = View::Advanced, adv_cursor_ = 0, true;
    if (ui::is_back(e)) return back(), true;
    return true;
}

Element Wizard::help_body() {
    const auto k = [](const char* key, const std::string& what) { return ui::plain(ui::pad(key, 14) + what); };
    return vbox({ui::title("Help"), ui::blank(), k("↑↓ j k", "move"), k("←→", "change a choice, move in a grid"),
                 k("Space", "switch on and off"), k("Enter", "the page's main action, shown in the footer"),
                 k("Esc ⌫", "back, leave a field"), k("e  i", "more options / details of the model"),
                 k("?  q", "this help / close"), ui::blank(), ui::title("Words"),
                 ui::plain("token          word piece, about ¾ of a word"),
                 ui::plain("decode         how fast the answer appears"),
                 ui::plain("prefill        Tona reads your text before it answers"),
                 ui::plain("speed range    estimate for this machine: docs/use/installer.md#speed-estimate"),
                 ui::plain("measured       tested in Tona; provisional: test running;"),
                 ui::plain("               vendor: number from the maker")});
}

Element Wizard::no_gpu_body() {
    Elements l = {ui::title("Your machine")};
    for (Element& r : machine_rows()) l.push_back(std::move(r));
    l.push_back(ui::blank());
    l.push_back(ui::bad("Tona has no backend for this graphics card yet."));
    for (Element& w : ui::wrap("Tona is built for AMD RDNA4 today. Other cards come as separate backends "
                               "(docs/backends/README.md); the status is at tobiasnawa.com/tona."))
        l.push_back(std::move(w));
    for (Element& r : Elements{ui::blank(), ui::blank(), ui::title("What you can do now"),
                               ui::dim(ui::cursor(false) + "Set up with the processor only: not in the installer yet"),
                               ui::highlight(ui::plain(ui::cursor(true) + "Quit, nothing is installed"), true),
                               ui::blank(), ui::dim("Tona changed nothing. Report for developers: key b")})
        l.push_back(std::move(r));
    if (report_)
        for (Element& w : ui::wrap(hardware_text(st_.h, nullptr))) l.push_back(std::move(w));
    return vbox(l);
}

bool Wizard::key_no_gpu(const Event& e) {
    if (e == Event::Return || ui::is_back(e)) return screen_.Exit(), true;
    if (e == Event::Character('b')) report_ = !report_;
    return true;
}

Element Wizard::no_space_body() {
    const uint64_t load = bytes_to_load(s(), models_dir_of(target_));
    Elements l = {ui::title("Your models"),
                  ui::bad("Not enough space:"),
                  ui::plain(std::format("{} {}, {} {}.", "to download", size_text(static_cast<double>(load)), "free",
                                        size_text(static_cast<double>(st_.h.disk_free)))),
                  ui::plain(std::format("{} {}.", "A buffer for conversations on the SSD comes on top:",
                                        size_text(static_cast<double>(kBufferBytes)))),
                  ui::blank(),
                  ui::title("What you can do")};
    for (size_t i = 0; i < space_opts_.size(); i++) {
        const bool here = static_cast<int>(i) == cursor_;
        l.push_back(ui::highlight(ui::plain(ui::cursor(here) + ui::fit(space_opts_[i].label, 76)), here));
        if (!space_opts_[i].note.empty()) l.push_back(ui::dim("    " + space_opts_[i].note));
    }
    if (space_opts_.empty()) l.push_back(ui::warn("None of it fits. Free some space, then p."));
    l.push_back(ui::blank());
    l.push_back(ui::dim("Nothing is downloaded before it fits."));
    return vbox(l);
}

void Wizard::check_space() {
    const std::string dir = models_dir_of(target_);
    if (target_.release && places_.empty()) places_ = find_places(dir);
    if (fits(s(), dir, static_cast<uint64_t>(st_.h.disk_free))) return;
    SpaceContext ctx;
    ctx.free_bytes = static_cast<uint64_t>(st_.h.disk_free), ctx.models_dir = dir, ctx.places = places_;
    space_opts_ = space_options(s(), st_.selection, st_.match.cls, ctx);
    view_ = View::NoSpace, cursor_ = 0;
}

void Wizard::recheck_space() {
    st_.h.disk_free = detect(target_.dir()).disk_free;
    view_ = View::Pages;
    check_space();
}

void Wizard::apply_space_option(size_t i) {
    const SpaceOption& o = space_opts_[i];
    st_.selection = o.selection;
    s() = apply_selection(s(), st_.selection);
    if (!o.place.empty()) {
        target_.models_dir = o.place + "/models";
        for (const Place& p : places_)
            if (o.place == p.path + "/tona") st_.h.disk_free = static_cast<int64_t>(p.free_bytes);
    }
    view_ = View::Pages, cursor_ = 0;
    check_space();
}

bool Wizard::key_no_space(const Event& e) {
    const int n = static_cast<int>(space_opts_.size());
    if (ui::is_down(e) && cursor_ + 1 < n) return ++cursor_, true;
    if (ui::is_up(e) && cursor_ > 0) return --cursor_, true;
    if (e == Event::Return && cursor_ < n) return apply_space_option(static_cast<size_t>(cursor_)), true;
    if (e == Event::Character('p')) return recheck_space(), true;
    if (ui::is_back(e)) return view_ = View::Pages, true;
    return true;
}

Element Wizard::failed_body() {
    std::string why;
    {
        std::lock_guard lk(log_mu_);
        for (auto it = log_.rbegin(); it != log_.rend() && why.empty(); ++it)
            if (it->starts_with("Failed:") || it->contains("interrupted")) why = *it;
    }
    Elements l = {ui::title("Here we go"),
                  ui::bad("Download failed"),
                  ui::plain(ui::fit(why.empty() ? std::string("The connection broke.") : why, 78)),
                  ui::dim(ui::fit("What is there stays and is verified.", 79)),
                  ui::blank(),
                  step_rows(),
                  ui::blank(),
                  ui::title("What you can do")};
    const char* const en[] = {"Resume", "Quit; the next tona setup goes on"};
    for (int i = 0; i < 2; i++) l.push_back(ui::highlight(ui::plain(ui::cursor(i == cursor_) + en[i]), i == cursor_));
    l.push_back(ui::dim("Same from a shell: tona setup --models <name> --yes"));
    return vbox(l);
}

bool Wizard::key_failed(const Event& e) {
    if (ui::is_down(e) && cursor_ < 1) return ++cursor_, true;
    if (ui::is_up(e) && cursor_ > 0) return --cursor_, true;
    if (e == Event::Return) {
        if (cursor_ == 1) return screen_.Exit(), true;
        if (worker_.joinable()) worker_.join();
        fetch_failed_ = false, view_ = View::Pages;
        return start_install(), true;
    }
    return true;
}

}  // namespace tona::install
