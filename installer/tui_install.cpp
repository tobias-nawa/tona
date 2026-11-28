// The installation thread of the wizard and the rows that show its progress.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <format>

#include "core/describe.h"
#include "core/release_steps.h"
#include "core/shell_path.h"
#include "core/sys.h"
#include "tui.h"
#include "tui_widgets.h"

using namespace ftxui;

namespace tona::install {

namespace {

// Columns of "What next": the command, then what it does
constexpr int kCommandWidth = 24;
constexpr const char* kAgentsTitle = "Code with an agent against Tona";

// The agents `tona <agent>` starts (engine/client/agent.cpp): program name and product name
struct AgentName {
    const char* program;
    const char* product;
};
// Screen updates per second at most: a fast line reports far more often, and every update redraws the page
constexpr auto kProgressInterval = std::chrono::milliseconds(250);

}  // namespace

const AgentName kAgents[] = {{"claude", "Claude Code"}, {"opencode", "OpenCode"}, {"junie", "Junie"}, {"pi", "Pi"}};

std::vector<std::string> installed_agents() {
    std::vector<std::string> out;
    for (const AgentName& a : kAgents)
        if (which(a.program)) out.emplace_back(a.program);
    return out;
}

std::vector<std::string> what_next_lines(const std::vector<std::string>& path_hint,
                                         const std::vector<std::string>& installed) {
    std::vector<std::string> out;
    const auto row = [&](const std::string& cmd, const std::string& what) {
        out.emplace_back(ui::pad(cmd, kCommandWidth) + what);
    };
    row("tona", "open the control centre");
    row("tona status", "see the state, also over SSH");
    out.emplace_back(kAgentsTitle);
    bool any = false;
    for (const AgentName& a : kAgents) {
        const bool have = std::find(installed.begin(), installed.end(), a.program) != installed.end();
        any = any || have;
        row(std::string("tona ") + a.program + " ~/project", std::string(a.product) + (have ? "" : " (not installed)"));
    }
    if (!any) out.emplace_back("No coding agent found. Install one first: docs/use/client.md#coding-agents");
    if (!path_hint.empty()) out.emplace_back();
    out.insert(out.end(), path_hint.begin(), path_hint.end());
    return out;
}

std::string Wizard::step_label(const Step& st) const { return step_text(st, models_dir_of(target_)); }

void Wizard::log_line(const std::string& line) {
    {
        std::lock_guard lk(log_mu_);
        // A message with the command on its own line is two entries: the log view cuts at 79 columns
        for (size_t from = 0; from <= line.size();) {
            const size_t nl = line.find('\n', from);
            const std::string part = line.substr(from, nl == std::string::npos ? std::string::npos : nl - from);
            log_.push_back(part), step_out_.push_back(part);
            if (log_.size() > 400) log_.pop_front();
            if (nl == std::string::npos) break;
            from = nl + 1;
        }
    }
    screen_.PostEvent(Event::Custom);
}

void Wizard::on_progress(const Progress& p) {
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard lk(log_mu_);
        if (has_progress_ && p.done < p.total && now - last_progress_ < kProgressInterval) return;
        last_progress_ = now;
        has_progress_ = true, progress_ = p;
        const double t = std::chrono::duration<double>(now - started_).count();
        meters_[static_cast<int>(p.phase)].add(t, p.done);
        if (p.phase == Phase::Download) dl_step_total_ = p.total, total_meter_.add(t, dl_base_ + p.done);
    }
    screen_.PostEvent(Event::Custom);
}

void Wizard::reset_progress() {
    std::lock_guard lk(log_mu_);
    has_progress_ = false;
    step_out_.clear();
    for (RateMeter& m : meters_) m.reset();
}

void Wizard::run_install_steps() {
    set_progress_sink([this](const Progress& p) { on_progress(p); });
    for (size_t i = 0; i < plan_.size() && result_ == 0; i++) {
        if (plan_[i].done) {
            state_[i] = StepState::Done;
            continue;
        }
        reset_progress();
        state_[i] = StepState::Running;
        screen_.PostEvent(Event::Custom);
        result_ = run_step(plan_[i], target_.dir(), [this](const std::string& l) { log_line(l); });
        state_[i] = result_ == 0 ? StepState::Done : StepState::Failed;
        {
            std::lock_guard lk(log_mu_);
            dl_base_ += dl_step_total_, dl_step_total_ = 0;
        }
        if (result_ != 0) {
            log_line(std::format("Step failed (status {}): {}", result_.load(), plan_[i].cmd));
            fetch_failed_ = plan_[i].title.starts_with("Fetch and verify");
        }
    }
    finished_ = true;
    screen_.PostEvent(Event::Custom);
}

void Wizard::start_install() {
    plan_ = plan_steps(st_.h, s(), target_);
    state_.assign(plan_.size(), StepState::Waiting);
    result_ = 0, finished_ = false, installing_ = true;
    dl_total_ = load_split(s(), models_dir_of(target_)).download, dl_base_ = dl_step_total_ = 0;
    total_meter_.reset();
    worker_ = std::thread([this] { run_install_steps(); });
}

// The lines under a step: its progress while it runs, its whole error when it failed. Called with log_mu_ held.
Elements Wizard::step_detail(StepState st) {
    constexpr int kMark = 3;
    Elements l;
    if (st == StepState::Running && has_progress_) {
        const int pct = percent_of(progress_);
        const char* what = progress_.phase == Phase::Download ? "downloading" : "verifying SHA-256";
        l.push_back(hbox({ui::dim(ui::pad("", kMark) + ui::pad(what, 18)), ui::progress_bar(pct, 30),
                          ui::plain(std::format(" {} %", pct))}));
        l.push_back(ui::dim(ui::pad("", kMark) + progress_text(progress_, meters_[static_cast<int>(progress_.phase)])));
    } else if (st == StepState::Failed) {
        // The whole error, wrapped: the cause and the command to fix it must not be cut at the screen edge
        for (size_t k = step_out_.size() > 4 ? step_out_.size() - 4 : 0; k < step_out_.size(); k++)
            for (const std::string& w : ui::wrap_lines(step_out_[k], 72)) l.push_back(ui::bad(ui::pad("", kMark) + w));
    } else if (st == StepState::Running && !log_.empty()) {
        l.push_back(ui::dim(ui::pad("", kMark) + ui::fit(log_.back(), 66)));
    }
    return l;
}

Elements Wizard::step_row(size_t i) {
    Elements l;
    if (step_label(plan_[i]).empty()) return l;  // the drafter is part of its model
    const StepState st = state_[i];
    // a character as well as a colour: NO_COLOR keeps the state readable
    const char* mark = st == StepState::Done      ? "✓"
                       : st == StepState::Running ? "▸"
                       : st == StepState::Failed  ? "✗"
                                                  : "·";
    constexpr int kMark = 3;
    const std::vector<std::string> text = ui::wrap_lines(step_label(plan_[i]), 72);
    for (size_t k = 0; k < text.size(); k++) {
        const std::string prefix = ui::pad(k == 0 ? mark : "", kMark);
        if (st == StepState::Done)
            l.push_back(hbox({ui::good(prefix), ui::plain(text[k])}));
        else if (st == StepState::Running)
            l.push_back(ui::title(prefix + text[k]));
        else if (st == StepState::Failed)
            l.push_back(ui::bad(prefix + text[k]));
        else
            l.push_back(ui::dim(prefix + text[k]));
    }
    for (Element& e : step_detail(st)) l.push_back(std::move(e));
    return l;
}

// A rule and one bar over everything to download (not the hash passes): percent, bytes, time left. Nothing to
// download, nothing to show. Called with log_mu_ held.
Element Wizard::total_rows() {
    if (dl_total_ == 0) return ui::blank();
    const bool fetching = has_progress_ && progress_.phase == Phase::Download;
    const uint64_t done = std::min(dl_total_, dl_base_ + (fetching ? progress_.done : dl_step_total_));
    const int pct = static_cast<int>(done * 100 / dl_total_);
    std::string tail = std::format(" {} %  {} of {}", pct, size_text(double(done)), size_text(double(dl_total_)));
    const double left = fetching ? total_meter_.remaining(dl_total_) : -1;
    if (left >= 0) tail += "  " + duration_text(left) + " left";
    std::string rule;
    for (int i = 0; i < 78; i++) rule += "─";
    return vbox({ui::dim(rule), hbox({ui::plain(ui::pad("Total", 8)), ui::progress_bar(pct, 24), ui::plain(tail)})});
}

// What follows when the steps are over: the next commands, or why it stopped
Elements Wizard::finish_rows() {
    Elements l;
    if (!finished_) return l;
    if (result_ != 0) {
        l.push_back(ui::bad("Stopped, see the log (l). After fixing it, run tona setup again."));
        return l;
    }
    l.push_back(ui::blank());
    l.push_back(ui::title("What next"));
    const std::string bin = target_.bin_dir.empty() ? home() + "/.local/bin" : target_.bin_dir;
    const std::vector<std::string> hint =
        s().shell ? path_advice(path_env_, bin, s().modify_path) : std::vector<std::string>{};
    for (const std::string& line : what_next_lines(hint, installed_agents())) {
        if (line == kAgentsTitle || line.contains("(not installed)") || line.starts_with("No coding agent"))
            l.push_back(ui::dim(line));
        else if (line.starts_with("✓"))
            l.push_back(hbox({ui::good("✓"), ui::plain(line.substr(3))}));
        else if (line.starts_with("tona") && line.size() > kCommandWidth)
            l.push_back(hbox({ui::title(line.substr(0, kCommandWidth)), ui::plain(line.substr(kCommandWidth))}));
        else
            l.push_back(ui::plain(line));
    }
    return l;
}

// "Setting up Tona" with the number of the running step, then "Tona is installed" and "Tona is ready"
std::pair<std::string, std::string> Wizard::run_header() const {
    size_t shown = 0, done = 0;
    for (size_t i = 0; i < plan_.size(); i++) {
        if (step_label(plan_[i]).empty()) continue;
        shown++;
        if (state_[i] == StepState::Done) done++;
    }
    if (finishing_now()) return {"Tona is installed", std::format("all {} steps done", shown)};
    if (finished_ && result_ == 0) return {"Tona is ready", ""};
    return {"Setting up Tona", std::format("step {} of {}", std::min(done + 1, shown), shown)};
}

Element Wizard::step_rows() {
    std::lock_guard lk(log_mu_);
    Elements l;
    // after a good run the steps give way to the last page: it needs the room
    const bool done = finished_ && result_ == 0;
    for (size_t i = 0; i < (show_log_ || done ? 0 : plan_.size()); i++)
        for (Element& r : step_row(i)) l.push_back(std::move(r));
    if (!done && !show_log_ && installing_) l.push_back(total_rows());
    if (show_log_)
        for (size_t i = log_.size() > 16 ? log_.size() - 16 : 0; i < log_.size(); i++)
            l.push_back(ui::dim(ui::fit(log_[i], 79)));
    for (Element& r : finish_rows()) l.push_back(std::move(r));
    if (!message_.empty()) l.push_back(ui::warn(message_));
    return vbox(l);
}

}  // namespace tona::install
