// State of the control centre (tona) and what its keys do. No terminal in here: `handle_key` takes a key and the state
// and returns what the program has to do (an Action); the views (view_*.cpp) turn the state into lines of text. Both
// are tested without a terminal.
#pragma once

#include <string>
#include <vector>

#include "client/doctor.h"
#include "client/languages.h"
#include "client/offer.h"
#include "client/server_view.h"
#include "client/settings.h"
#include "core/plan.h"

namespace tona::client {

enum class Tab { Start, Models, Agents, Settings };

enum class KeyCode { None, Char, Up, Down, Left, Right, Enter, Esc, Backspace, Tab, BackTab };
struct Key {
    KeyCode code = KeyCode::None;
    char ch = 0;  // KeyCode::Char: one byte (UTF-8 text arrives byte by byte and is put together by the text field)
    bool operator==(const Key&) const = default;
};
// Bytes read from the terminal to keys (escape sequences for arrows, \r and \n as Enter, DEL and ^H as Backspace)
std::vector<Key> parse_keys(const std::string& bytes);

enum class Modal { None, Help, Close, Confirm };

// What a Confirm dialog asks for
enum class Ask { None, SwitchModel, Restart, Update, Remove, StopServer };

enum class ActionKind {
    None,
    Quit,         // leave the centre, the server keeps running
    QuitAndStop,  // leave and stop the server
    SwitchModel,  // arg: model id (also "back to the previous one" while loading)
    Download,     // arg: model id
    Remove,       // arg: model id
    StartServer,
    StartOnFreePort,
    RestartServer,
    ToggleLog,
    UpdateKey,  // arg: "u", "c" or "r" (rollback), as the updater's keys
    SaveConfig,
    ApplyRecommendation,
    LaunchAgent,   // starts the agent chosen on the Agents page (the program exits into it)
    RunInstaller,  // tona setup (empty page: "choose a model")
    DownloadRecommended,
    ProjectChanged,  // the project folder changed: look at its languages
    StopServer
};

struct Action {
    ActionKind kind = ActionKind::None;
    std::string arg;
};

// One row of the model list
struct ModelRow {
    enum class Section { Installed, ToDownload, Coming } section = Section::Installed;
    const ModelCard* card = nullptr;
    bool loaded = false;  // the server runs it
};

struct ModelFilter {
    size_t task = 0;      // index into the tasks, 0 = every task
    size_t language = 0;  // index into the languages of the cards, 0 = any
    bool fits_only = true;
};

struct AgentForm {
    enum Field { Agent, Project, Model, Tools, Effort, FieldCount };
    int field = Agent;
    int agent = 0;        // 0 Claude Code, 1 OpenCode, 2 Junie
    std::string project;  // the project field takes text while it has the focus; digits and q do not switch tabs then
    bool all_tools = true;
    int effort = 1;  // 0 short (low), 1 medium, 2 long (high)
};

struct SettingsForm {
    size_t group = 0;
    size_t row = 0;
    bool dirty = false;  // changed and not written yet
};

// What the updater reports, in the terms the Start page needs (engine/update/updater.h has the full view)
struct UpdateInfo {
    bool enabled = false;  // a release installation; a checkout updates with git
    bool checks_on = false;
    bool checking = false, downloading = false, available = false, security = false, failed = false;
    bool switching = false;  // the new version is being started: the server is down on purpose
    double fraction = 0;     // download
    std::string version, message;
    std::string changelog;  // text of the release notes, shown in the dialog before installing
    double download_mb = 0;
};

struct LaunchEntry {
    std::string agent, project, model;
    long long time = 0;
};

struct UiState {
    Tab tab = Tab::Start;
    Modal modal = Modal::None;
    Ask ask = Ask::None;
    std::string ask_arg;  // model id of a switch or removal
    int modal_cursor = 0;
    bool remember_choice = false;  // Close dialog: do not ask again

    // Data, replaced by the program on every frame
    ServerView server;
    std::vector<double> rates;          // decode rate per poll, newest last
    std::vector<std::string> projects;  // project names seen, for the per-project view of Start
    int project_view = 0;               // 0 all, else projects[project_view - 1]
    std::vector<std::string> log;
    bool show_log = false;
    UpdateInfo update;
    std::string message;  // the line above the keys
    Config config;
    Offer offer;
    std::vector<install::ModelInfo> catalog;
    std::vector<LanguageShare> languages;  // of the project of the Agents page
    std::vector<LaunchEntry> launches;
    std::string loaded_id;       // catalog id of the model the server runs ("" if unknown)
    std::vector<Check> checks;   // what tona doctor found, for the "server not running" page
    int free_port = 0;           // a port nobody holds, for "start on a free port"
    std::string job;             // a long action runs ("download ...")
    bool restarting = false;     // the job restarts the server (restart, switching the model): it is down on purpose
    std::string switching_from;  // model id before a switch that is still loading ("" when none)
    std::string switching_to;
    double ram_total_gb = 0, vram_total_gb = 0;
    std::string launch_command;  // shown on the Agents page; set by the reducer
    int width = 80, height = 24;
    long long now = 0;  // epoch seconds of this frame (the views never read the clock)

    // Cursors and forms
    int start_cursor = 0;
    int models_cursor = 0;
    ModelFilter filter;
    AgentForm agents;
    SettingsForm settings;
};

// The rows of the Models page for the current filter, in section order
std::vector<ModelRow> model_rows(const UiState& s);
// The card of the cursor row, or nullptr
const ModelRow* selected_model(const UiState& s, const std::vector<ModelRow>& rows);
// Names of the tasks and languages the filter cycles through
std::vector<std::string> filter_languages(const UiState& s);
// A better installed model for the languages of the project than the loaded one, or nullptr (suggestion only)
const ModelCard* suggest_model(const UiState& s);
// True when the model is installed (verified) according to the catalog
bool is_installed(const UiState& s, const ModelCard& c);
// Name of the loaded model for the status line ("Coder"), or the file stem
std::string loaded_name(const UiState& s);
// The conversations that would be cut by a restart or a switch
std::vector<Conversation> running_conversations(const UiState& s);

// The command `tona <agent> <dir>` that Enter on the Agents page runs, and its environment words (TONA_TOOLS, ...)
std::string agent_name(int agent);
std::string agent_command(const UiState& s);

// Applies a key; returns what the program has to do
Action handle_key(UiState& s, Key key);

// True while a text field has the focus: the keys 1-4, q and ? are text there, not commands
bool in_text_field(const UiState& s);
// No model on the disk: the first-start page
bool nothing_installed(const UiState& s);

// A planned restart runs (restart, switching the model, the update): a server that does not answer is going to
inline bool restarting(const UiState& s) { return s.restarting || s.update.switching; }
// Groups of the Settings page (text keys, in order) and the specs of one group
std::vector<std::string> setting_groups();
std::vector<const SettingSpec*> group_rows(size_t group);
// Entries of the Close dialog, in the order shown: 0 keep the server, 1 stop it too, 2 cancel
std::vector<int> close_options(const UiState& s);

}  // namespace tona::client
