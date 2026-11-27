// The Finish screen between the installation and "What next": the tasks that change things outside Tona's own folders
// (the shell profiles) or switch a background job (the update checks). Each one is ticked by the user; nothing here
// runs before Enter.
#include <format>
#include <vector>

#include "core/release_steps.h"
#include "core/shell_path.h"
#include "core/sys.h"
#include "tui.h"
#include "tui_widgets.h"

using namespace ftxui;

namespace tona::install {

namespace {

std::string tilde_of(const std::string& path, const std::string& home) {
    return !home.empty() && path.starts_with(home + "/") ? "~" + path.substr(home.size()) : path;
}

}  // namespace

bool Wizard::finishing() {
    if (!finishing_now()) return false;
    build_finish();
    if (finish_.empty()) finish_applied_ = true;  // nothing to ask: straight to What next
    return !finish_.empty();
}

void Wizard::build_finish() {
    if (finish_built_) return;
    finish_built_ = true, cursor_ = 0;
    const std::string bin = target_.bin_dir.empty() ? path_env_.home + "/.local/bin" : target_.bin_dir;
    if (s().shell && !on_path(path_env_.path, bin)) {
        std::string files;
        for (const ProfileFile& p : profile_files(path_env_))
            files += (files.empty() ? "" : ", ") + tilde_of(p.file.string(), path_env_.home);
        finish_.push_back({"Add " + tilde_of(bin, path_env_.home) + " to PATH in " + files,
                           "adds a marked block; tona uninstall removes it", s().modify_path, true});
    }
    if (target_.release)
        finish_.push_back({"Check for updates once a day",
                           "only checks and tells you; nothing installs without your ok", s().update_checks, false});
}

Element Wizard::finish_body() {
    Elements l = {ui::title("Last two things"), ui::blank()};
    for (size_t i = 0; i < finish_.size(); i++) {
        const std::vector<std::string> lines = ui::wrap_lines(finish_[i].label, 68);
        for (size_t k = 0; k < lines.size(); k++)
            l.push_back(
                ui::plain(k == 0 ? ui::cursor(static_cast<int>(i) == cursor_) + ui::box(finish_[i].on) + " " + lines[k]
                                 : "      " + lines[k]));
        l.push_back(ui::dim("      " + finish_[i].note));
    }
    if (!finish_errors_.empty()) l.push_back(ui::blank());
    for (size_t i = finish_errors_.size() > 6 ? finish_errors_.size() - 6 : 0; i < finish_errors_.size(); i++)
        for (const std::string& w : ui::wrap_lines(finish_errors_[i], 76)) l.push_back(ui::bad("  " + w));
    if (!message_.empty()) l.push_back(ui::warn(message_));
    return vbox(l);
}

bool Wizard::key_finish(const Event& e) {
    const int n = static_cast<int>(finish_.size());
    if (ui::is_down(e) && cursor_ + 1 < n) return ++cursor_, true;
    if (ui::is_up(e) && cursor_ > 0) return --cursor_, true;
    if (ui::is_space(e) && cursor_ < n) return finish_[cursor_].on = !finish_[cursor_].on, true;
    if (e == Event::Return) return apply_finish(), true;
    return true;
}

void Wizard::apply_finish() {
    for (const FinishRow& r : finish_) (r.path ? s().modify_path : s().update_checks) = r.on;
    finish_errors_.clear();
    for (const Step& st : finish_steps(s(), target_, path_env_)) {
        std::vector<std::string> said;
        if (!st.action || st.action([&said](const std::string& line) { said.push_back(line); }) == 0) continue;
        finish_errors_.push_back("Failed: " + st.title);
        for (const std::string& line : said)
            for (size_t from = 0; from <= line.size();) {
                const size_t nl = line.find('\n', from);
                finish_errors_.push_back(line.substr(from, nl == std::string::npos ? std::string::npos : nl - from));
                if (nl == std::string::npos) break;
                from = nl + 1;
            }
    }
    // A failed task keeps the page: its cause and the fix are on screen, and Enter tries again
    if (finish_errors_.empty()) finish_applied_ = true, cursor_ = 0;
}

void Wizard::stage_finished() {
    plan_ = plan_steps(st_.h, s(), target_);
    state_.assign(plan_.size(), StepState::Done);
    result_ = 0, finished_ = true, installing_ = true, page_ = Page::Go, view_ = View::Pages;
}

void Wizard::stage_failed(const std::string& message) {
    stage_finished();
    state_.back() = StepState::Failed, result_ = 1;
    step_out_.clear();
    log_line(message);
}

}  // namespace tona::install
