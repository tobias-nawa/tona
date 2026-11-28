// Every page and state of the guided setup as text at 80 x 24 (tona setup --dump-pages): for the documentation and
// for a look at the pages without a terminal. The states that need a special machine are staged on the real one.
#include <cstdio>

#include "tui.h"
#include "tui_widgets.h"

namespace tona::install {

std::string Wizard::dump_all() {
    std::string out;
    const auto shot = [&](const std::string& name) {
        out += "=== " + name + "\n" + ui::render_text(frame(body(), keys_element())) + "\n";
    };
    for (Page p : {Page::Machine, Page::Tasks}) enter_page(p), shot(page_name(p));
    for (const char* l : {"python", "typescript", "rust"}) toggle_language(st_.selection, l);
    s() = apply_selection(s(), st_.selection);
    enter_page(Page::Languages), shot(page_name(Page::Languages));
    adopt_language_models();  // what Enter on the language page does
    enter_page(Page::Models), shot(page_name(Page::Models));
    view_ = View::Details, shot(page_name(Page::Models) + " (i)");
    enter_page(Page::Room), shot(page_name(Page::Room));
    view_ = View::Advanced, shot(page_name(Page::Room) + " (e)");
    enter_page(Page::Go), shot(page_name(Page::Go));
    // the second model is downloading: the release and the first model are done, the rest waits
    constexpr size_t kRunning = 2;
    state_.assign(plan_.size(), StepState::Waiting);
    for (size_t i = 0; i < plan_.size(); i++)
        if (i < kRunning) state_[i] = StepState::Done;
    state_[kRunning] = StepState::Running;
    installing_ = true, has_progress_ = true, progress_ = {Phase::Download, 12'300'000'000, 21'700'000'000};
    dl_total_ = 106'300'000'000, dl_base_ = 62'500'000'000;
    for (int t = 0; t <= 4; t++) {
        meters_[0].add(t, static_cast<uint64_t>(t) * 45'200'000);  // 45.2 MB/s
        total_meter_.add(t, dl_base_ + 12'300'000'000 - static_cast<uint64_t>(4 - t) * 45'200'000);
    }
    shot(page_name(Page::Go) + " (running)");
    state_.back() = StepState::Failed, fetch_failed_ = true, has_progress_ = false;
    installing_ = false;
    enter_page(Page::Models);
    const int64_t free_real = st_.h.disk_free;
    st_.h.disk_free = 58'000'000'000;
    view_ = View::Pages, check_space(), shot("not enough space");
    st_.h.disk_free = free_real;
    view_ = View::Failed, cursor_ = 0, installing_ = true, finished_ = true, shot("download failed");
    installing_ = false;
    view_ = View::Help, shot("help");
    out += "=== too small (52 x 12)\n" + ui::render_text(too_small(52, 12), 52, 12);
    Gpu nvidia;
    nvidia.vendor = "NVIDIA", nvidia.name = "GeForce RTX 3060", nvidia.vram_total = 12'884'901'888;
    st_.h.gpus = {nvidia}, st_.module = nullptr, view_ = View::NoGpu;
    shot("no suitable graphics card");
    return out;
}

}  // namespace tona::install

int dump_pages(const tona::install::Options& o, const tona::install::Target& target) {
    using namespace tona::install;
    Start st;
    FlowError err;
    if (!make_start(o, target, true, st, err)) {
        std::fprintf(stderr, "tona setup: %s\n", err.text.c_str());
        return err.code;
    }
    // Staged: the pages as a fresh machine shows them (no configuration), and the graphics card without the load of
    // whatever runs on it right now, so the numbers are those of the reference PC
    st.selection = default_selection(st.match.cls);
    for (Gpu& g : st.h.gpus)
        if (g.vram_used > 4'000'000'000) g.vram_used = 1'600'000'000;
    if (st.module) st.recommended = st.module->recommend(st.h);
    st.settings = apply_selection(st.recommended, st.selection);
    // as a user sees it: the signed release, nothing downloaded yet
    Wizard w(std::move(st), o, release_target(""));
    std::fputs(w.dump_all().c_str(), stdout);
    return 0;
}
