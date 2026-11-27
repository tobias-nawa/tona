// Terminal wizard of the installer (FTXUI), after the design of the terminal programs: six pages (machine, tasks,
// languages, models, room, go), pre-filled so that Enter alone gives a good setup, 80 x 24, the same keys on every
// page. The advanced settings (drafter, context, conversations, key, service, command) sit behind the key `e`, also
// pre-filled. What the pages decide is in core/flow.h; this file only shows it and takes keys.
#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/cli_args.h"
#include "core/flow.h"
#include "core/progress.h"
#include "core/report.h"
#include "core/room.h"
#include "core/shell_path.h"
#include "core/space.h"
#include "core/start.h"
#include "core/steps.h"
#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "tui_widgets.h"

namespace tona::install {

// The lines of the final page "What next" in fixed columns: one row per agent (a missing one says "(not installed)"),
// then the PATH hint (empty when the command is on PATH)
std::vector<std::string> what_next_lines(const std::vector<std::string>& path_hint,
                                         const std::vector<std::string>& installed);
// The agent programs (claude, opencode, junie, pi) found on PATH
std::vector<std::string> installed_agents();

// What the screen shows besides the pages
enum class View { Pages, Details, Advanced, Help, NoGpu, NoSpace, Failed };

// State of one step while the installation runs
enum class StepState { Waiting, Running, Done, Failed };

class Wizard {
public:
    Wizard(Start start, Options opts, Target target);
    ~Wizard();
    Wizard(const Wizard&) = delete;
    Wizard& operator=(const Wizard&) = delete;

    // Runs until the user quits; returns the exit status of the installation (0 if none ran).
    int run();

    // For the tests: the machine's PATH and profiles, the state after the steps, a key, the screen as text
    void set_path_env(PathEnv e) { path_env_ = std::move(e); }
    void stage_finished();
    void stage_failed(const std::string& message);  // the last step failed and said `message`
    bool press(const ftxui::Event& e) { return on_event(e); }
    std::string screen_text() { return ui::render_text(frame(body(), keys_line())); }

    // Every page and state as text at 80 x 24, for the documentation and for a look without a terminal. Uses the
    // machine as it is; the states that need a special machine are staged.
    std::string dump_all();

private:
    // rendering (tui.cpp)
    ftxui::Element render();
    ftxui::Element frame(const ftxui::Element& body, const std::string& keys);
    ftxui::Element crumbs();
    ftxui::Element too_small(int w, int h);
    std::string page_name(Page p) const;
    ftxui::Element body();
    std::string keys_line();
    int page_keys() const;

    // pages (tui_pages.cpp)
    ftxui::Element machine_body();
    ftxui::Elements machine_rows();
    ftxui::Element tasks_body();
    ftxui::Element languages_body() const;
    ftxui::Elements plan_rows() const;
    ftxui::Element models_body();
    ftxui::Elements model_list(const std::string& dir) const;
    ftxui::Elements model_totals(const std::string& dir) const;
    ftxui::Element details_body() const;
    // speed of a model on this machine under the current settings; hi == 0 without a backend module
    Estimate model_estimate(const std::string& id) const;
    bool key_tasks(const ftxui::Event& e);
    bool key_languages(const ftxui::Event& e);
    bool key_models(const ftxui::Event& e);

    // room and advanced settings (tui_room.cpp)
    ftxui::Element room_body();
    ftxui::Element advanced_body();
    bool key_room(const ftxui::Event& e);
    bool key_advanced(const ftxui::Event& e);
    bool key_advanced_row(const ftxui::Event& e);
    void room_changed();
    void change_room_row(int dir);
    std::string speed_line() const;
    int advanced_rows() const;

    // states and the last page (tui_states.cpp)
    ftxui::Element go_body();
    ftxui::Element help_body();
    ftxui::Element no_gpu_body();
    ftxui::Element no_space_body();
    ftxui::Element failed_body();
    bool key_go(const ftxui::Event& e);
    bool key_no_space(const ftxui::Event& e);
    bool key_failed(const ftxui::Event& e);
    bool key_no_gpu(const ftxui::Event& e);
    void sync_view();
    void adopt_language_models();  // the model page starts with what the language page proposed
    void check_space();
    void recheck_space();
    void apply_space_option(size_t i);

    // navigation and events (tui.cpp)
    bool on_event(const ftxui::Event& e);
    bool key_common(const ftxui::Event& e);
    bool on_view_event(const ftxui::Event& e);
    void enter_page(Page p);
    void forward();
    void back();
    std::vector<int> task_rows() const;  // indexes of the tasks that can be switched

    // installation (tui_install.cpp)
    void start_install();
    void run_install_steps();
    void log_line(const std::string& line);  // one entry per line of `line`
    void on_progress(const Progress& p);
    void reset_progress();
    std::string step_label(const Step& st) const;
    ftxui::Element step_rows();
    ftxui::Elements step_row(size_t i);
    ftxui::Elements step_detail(StepState st);
    ftxui::Elements finish_rows();
    ftxui::Element total_rows();                             // rule and the bar over all downloads, while the steps run
    std::pair<std::string, std::string> run_header() const;  // the header line while the steps run and after them

    // the Finish screen (tui_finish.cpp): the tasks after the steps, each with the user's tick
    bool finishing();  // the steps are done and the screen has not been applied yet
    bool finishing_now() const { return installing_ && finished_ && result_ == 0 && !finish_applied_; }
    void build_finish();
    ftxui::Element finish_body();
    bool key_finish(const ftxui::Event& e);
    void apply_finish();

    Target target_;
    Options opts_;
    Start st_;
    Settings& s() { return st_.settings; }
    const Settings& s() const { return st_.settings; }
    std::vector<Package> pk_;

    ftxui::ScreenInteractive screen_ = ftxui::ScreenInteractive::Fullscreen();
    Page page_ = Page::Machine;
    View view_ = View::Pages;
    View before_help_ = View::Pages;
    int cursor_ = 0;       // row on the current page or state
    int adv_cursor_ = 0;   // row in the advanced settings
    bool report_ = false;  // the hardware report on the page without a graphics card
    bool show_log_ = false;
    std::atomic<bool> fetch_failed_ = false;  // the installation stopped at a download
    bool server_only_ = false;
    Priority priority_ = Priority::Balanced;
    bool editing_ = false;  // a number is being typed in the advanced settings
    std::string edit_;
    std::vector<SpaceOption> space_opts_;
    std::vector<Place> places_;
    std::vector<std::string> seen_languages_;  // languages whose proposal is already on the model page
    std::vector<std::string> unfit_;           // proposed models that stay off because the drive is too small
    std::string message_;                      // one line below the page: why Enter did nothing

    struct FinishRow {
        std::string label, note;
        bool on = true;
        bool path = false;  // the PATH row, else the update checks
    };
    std::vector<FinishRow> finish_;
    bool finish_built_ = false, finish_applied_ = false;
    std::vector<std::string> finish_errors_;  // what a failed Finish task said; the page stays until Enter succeeds
    PathEnv path_env_ = current_path_env();

    std::vector<Step> plan_;  // the steps, computed when the last page is entered
    std::vector<StepState> state_;
    bool installing_ = false;
    std::atomic<bool> finished_ = false;
    std::atomic<int> result_ = 0;
    std::mutex log_mu_;
    std::deque<std::string> log_;
    std::vector<std::string> step_out_;  // what the running step said (guarded by log_mu_): shown if it fails
    std::string current_;
    bool has_progress_ = false;  // a step reports numbers (guarded by log_mu_, like the rest below)
    Progress progress_;
    RateMeter meters_[2];  // one per phase: the download rate survives a hash pass in between
    // The bar over all downloads: bytes to fetch in all, bytes of the finished steps, the total of the running step
    uint64_t dl_total_ = 0, dl_base_ = 0, dl_step_total_ = 0;
    RateMeter total_meter_;
    std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now(), last_progress_ = started_;
    std::thread worker_;
};

}  // namespace tona::install
