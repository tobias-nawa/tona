// The pages of the centre as text: 80 x 24 exactly, the words of the design, NO_COLOR, the states, the
// feed from a recorded-shape /status document. TONA_DUMP_SCREENS=1 prints every page (for the review of the design).
#include <cstdlib>
#include <iostream>
#include <regex>
#include <set>

#include "client/feed.h"
#include "client/screen.h"
#include "client/term.h"
#include "tests/client_testkit.h"

using namespace tona::client;
using namespace tona::test;
namespace top = tona::top;

namespace {

std::string g_data;

// The text as a person sees it: every escape sequence removed
std::string plain(const Lines& lines) {
    static const std::regex escape("\x1b\\[[0-9;]*m");
    std::string s;
    for (const std::string& l : lines) s += std::regex_replace(l, escape, "") + "\n";
    return s;
}

void dump(const char* title, const Lines& lines) {
    if (std::getenv("TONA_DUMP_SCREENS")) std::cout << "=== " << title << " ===\n" << plain(lines) << "\n";
}

// Every line is exactly as wide as the terminal and there are as many lines as it has rows
void expect_fits(const Lines& lines, int w, int h) {
    EXPECT((int)lines.size() == h);
    for (const std::string& l : lines) EXPECT(top::width(l) == w);
}

// A text key that shows up raw means a missing translation (tr returns the key for an unknown key)
void expect_no_raw_keys(const std::string& text) {
    std::set<std::string> prefixes;
    for (size_t i = 0; i < text_count(); i++) {
        const std::string key = text_key(i);
        prefixes.insert(key.substr(0, key.find('_')) + "_");
        EXPECT(!has(text, " " + key + " ") && !has(text, " " + key + "\n"));
    }
    static const std::regex word("\\b[a-z]+_[a-z0-9_]+\\b");
    for (std::sregex_iterator it(text.begin(), text.end(), word), end; it != end; ++it) {
        const std::string w = it->str();
        EXPECT(!prefixes.contains(w.substr(0, w.find('_') + 1)) || w == "ram_gb" || w == "vram_gb");
    }
}

UiState state(const char* status) { return test_state(status, g_data); }

// Apple silicon: no graphics memory row, one shared row with the model's GPU share as Tona's; an unreadable graphics
// memory elsewhere: no row rather than "-0.0 of -0 GB"
void test_start_page_memory_without_vram() {
    UiState s = state("status-decode.json");
    s.server.mem = {.vram_used = -1,
                    .vram_total = -1,
                    .ram_total = 34.4e9,
                    .ram_available = 9e9,
                    .ram_tona = 0,
                    .unified = true,
                    .tona_gpu = 21e9};
    std::string t = plain(render(s));
    EXPECT(!has(t, "Graphics memory") && has(t, "Shared") && has(t, "Tona 21 · others 4 · free 9 GB"));
    s.server.mem.unified = false;
    t = plain(render(s));
    EXPECT(!has(t, "Graphics memory") && !has(t, "-0") && has(t, "Main memory"));
}

void test_start_page() {
    UiState s = state("status-decode.json");
    const Lines lines = render(s);
    expect_fits(lines, 80, 24);
    const std::string t = plain(lines);
    EXPECT(has(t, "Server running · model Coder · 2 conversations · priority balanced"));
    EXPECT(has(t, "claude") && has(t, "writes") && has(t, "96 tokens/s"));
    EXPECT(has(t, "reads in") && has(t, "42 % · 2,897 tokens/s"));  // decode and prefill rates are named
    EXPECT(has(t, "docs") && has(t, "waits"));                      // the waiting conversation
    EXPECT(has(t, "Graphics memory") && has(t, "15.1 of 17 GB"));
    EXPECT(has(t, "Tona 35 · others 14 · free 15 GB"));  // the numbers of the design
    EXPECT(has(t, "2 of 6 possible at once"));
    EXPECT(has(t, "1,840 tokens written"));
    EXPECT(has(t, "█") && has(t, "▒") && has(t, "░"));  // bars carry characters, not only colour
    s.update.enabled = true, s.update.available = true, s.update.version = "0.3.1";
    const std::string page = plain(render(s));
    EXPECT(has(page, "Update 0.3.1 ready, signed. Key u, asks first."));  // one plain line, no hints block
    EXPECT(!has(page, "Hints") && !has(page, "You are working"));
    dump("start", lines);
}

void test_start_variants() {
    UiState s = state("status-loading.json");
    const std::string loading = plain(render(s));
    EXPECT(has(loading, "Loading") && has(loading, "Fill the expert slots") && has(loading, "60 %"));
    EXPECT(has(loading, "Server loading 80 %"));
    s = state("status-idle.json");
    EXPECT(has(plain(render(s)), "Nothing is running."));
    s = state("status-decode.json");
    s.project_view = 1;  // only the project "tona"
    const std::string one = plain(render(s));
    EXPECT(has(one, "Now · tona") && !has(one, "opencode") && has(one, "claude"));
    s.project_view = 0, s.show_log = true, s.log = {"line one of the log", "line two"};
    const std::string log = plain(render(s));
    EXPECT(has(log, "line two") && has(log, "Log (journalctl"));
    expect_fits(render(s), 80, 24);
}

// The Models page lists a task without a model under "Coming to Tona" and keeps it when that task is the filter
void test_coming_task_rows() {
    UiState s = state("status-decode.json");
    s.offer = build_offer(nullptr);
    s.tab = Tab::Models;
    const auto images =
        std::find_if(s.offer.tasks.begin(), s.offer.tasks.end(), [](const Task& t) { return t.id == "images"; });
    EXPECT(images != s.offer.tasks.end());
    s.filter.task = (size_t)(images - s.offer.tasks.begin()) + 1;
    s.models_cursor = 0;
    const std::string t = plain(render(s));
    EXPECT(has(t, "Coming to Tona") && has(t, "Generate images") && has(t, "coming"));
    EXPECT(!has(t, "no model built in"));  // the same long note on every row said nothing
}

void test_models_page() {
    UiState s = state("status-decode.json");
    s.tab = Tab::Models;
    s.models_cursor = 1;  // Ornith
    Lines lines = render(s);
    expect_fits(lines, 80, 24);
    const std::string t = plain(lines);
    EXPECT(has(t, "Installed") && has(t, "To download") && has(t, "Coming to Tona"));
    EXPECT(has(t, "Coder") && has(t, "running now") && !has(t, "measured"));  // section headers say the state
    EXPECT(has(t, "Qwen3.8 27B"));                                            // the section header says "coming"
    EXPECT(has(t, "Good for") && has(t, "Rust, Go") && has(t, "writes ≈ 120.0 tokens/s at 16K context"));
    EXPECT(has(t, "licence text not checked yet"));
    EXPECT(has(t, "Enter switches to Ornith. Tona asks first, 2 conversations are running."));
    s.switching_from = "coder-iq1_m", s.switching_to = "ornith-1.5-35b-a3b";
    s.server = parse_status(nlohmann::json::parse(read_text(g_data + "/status-loading.json")), true, "");
    EXPECT(has(plain(render(s)), "Ornith is loading, 80 %") && has(plain(render(s)), "Esc cancels, Coder stays"));
    dump("models", lines);
}

void test_agents_and_settings_pages() {
    UiState s = state("status-decode.json");
    s.tab = Tab::Agents;
    Lines lines = render(s);
    expect_fits(lines, 80, 24);
    std::string t = plain(lines);
    EXPECT(has(t, "(•) Claude Code") && has(t, "found: Rust 81 %, Shell 12 %"));
    EXPECT(has(t, "Suggestion: Ornith suits Rust better.") && has(t, "key w"));
    EXPECT(has(t, "Command: tona claude ~/projekte/sensor-daemon"));
    EXPECT(!has(t, "Tona never switches") && !has(t, "Enter starts."));
    EXPECT(has(t, "A switch would cut the 2 running conversations."));
    EXPECT(has(t, "Started recently") && has(t, "17 min ago"));
    s.agents.field = AgentForm::Project;
    expect_fits(render(s), 80, 24);
    s.agents.project.clear();
    expect_fits(render(s), 80, 24);
    dump("agents", lines);
    s.tab = Tab::Settings;
    s.config.s.ram_gb = 40;
    lines = render(s);
    expect_fits(lines, 80, 24);
    t = plain(lines);
    EXPECT(has(t, "Memory, priority") && has(t, "(•) I work on it") && has(t, "( ) server"));
    EXPECT(has(t, "When it is tight") && has(t, "◀ 40 GB ▶"));
    EXPECT(has(t, "Applies after a restart. Now: key r, asks first."));
    EXPECT(has(t, "tona config set use desktop"));
    s.settings.dirty = true;
    EXPECT(has(plain(render(s)), "not saved (key s)"));
    dump("settings", lines);
}

void test_states() {
    UiState s = state("status-decode.json");
    s.server = parse_status({}, false, "connection refused");
    s.checks = {{"unit", false, "failed at 08:14", "systemctl --user reset-failed tona-server"},
                {"port", false, "8662 busy", "tona config set port 8663"},
                {"models", true, "coder-iq1_m ok", ""}};
    s.free_port = 8663;
    Lines lines = render(s);
    expect_fits(lines, 80, 24);
    std::string t = plain(lines);
    EXPECT(has(t, "Server not running · no model loaded"));
    EXPECT(has(t, "The server cannot be reached (connection refused).") && has(t, "What Tona found out"));
    EXPECT(has(t, "warn") && has(t, "→ tona config set port 8663") && has(t, "Start on a free port (8663)"));
    EXPECT(has(t, "tona doctor"));
    dump("server-down", lines);
    // A planned restart (restart, switching the model, the update) says so instead of "cannot be reached"
    for (const bool update : {false, true}) {
        UiState r = s;
        r.restarting = !update, r.update.switching = update;
        t = plain(render(r));
        EXPECT(has(t, "Server restarting") && has(t, "Restarting the server."));
        EXPECT(!has(t, "cannot be reached") && !has(t, "not running") && !has(t, "Load failed"));
    }
    s = state("status-decode.json");
    for (in::ModelInfo& m : s.catalog) m.present = false;  // nothing installed
    s.loaded_id.clear();
    s.server = parse_status({}, false, "");
    lines = render(s);
    expect_fits(lines, 80, 24);
    t = plain(lines);
    EXPECT(has(t, "Welcome to Tona.") && has(t, "Choose a model") && has(t, "Download the recommendation"));
    EXPECT(has(t, "Tip: Ornith (21 GB) is quicker to get ready"));
    dump("empty", lines);
    s = state("status-decode.json");
    s.modal = Modal::Help;
    lines = render(s);
    expect_fits(lines, 80, 24);
    EXPECT(has(plain(lines), "Everywhere") && has(plain(lines), "Words") &&
           has(plain(lines), "q asks whether the server stops too"));
    dump("help", lines);
}

void test_dialogs() {
    UiState s = state("status-decode.json");
    s.tab = Tab::Models, s.modal = Modal::Confirm, s.ask = Ask::SwitchModel, s.ask_arg = "ornith-1.5-35b-a3b";
    Lines lines = render(s);
    expect_fits(lines, 80, 24);
    std::string t = plain(lines);
    EXPECT(has(t, "Switch to Ornith?") && has(t, "These conversations are running and are cut:"));
    EXPECT(has(t, "claude") && has(t, "opencode") && has(t, "Esc cancel"));
    dump("switch", lines);
    s.server.running.clear();
    EXPECT(has(plain(render(s)), "No conversation is running."));
    s.modal = Modal::Close, s.ask = Ask::None, s.modal_cursor = 0;
    s = state("status-decode.json");
    s.modal = Modal::Close;
    lines = render(s);
    expect_fits(lines, 80, 24);
    t = plain(lines);
    EXPECT(has(t, "Close the centre") && has(t, "Close the centre only, the server keeps running"));
    EXPECT(has(t, "2 running conversations are cut") && has(t, "35 GB main memory"));
    EXPECT(has(t, "[ ] Remember my choice and do not ask again"));
    dump("close", lines);
    s = state("status-decode.json");
    s.modal = Modal::Confirm, s.ask = Ask::Update, s.update.version = "0.4.0", s.update.download_mb = 41.5;
    s.update.changelog = "first\nsecond\x1b[2Jevil";
    t = plain(render(s));
    EXPECT(has(t, "Install update 0.4.0?") && has(t, "41.5 MB, signed.") && has(t, "first") && has(t, "second"));
    EXPECT(!has(t, "\x1b[2J"));  // release notes never reach the terminal with an escape character
}

// The update dialog with release notes in the form CHANGELOG.md gives them (Markdown, long lines, blank lines)
void test_update_dialog_release_notes() {
    UiState s = state("status-decode.json");
    s.modal = Modal::Confirm, s.ask = Ask::Update, s.update.version = "0.4.0", s.update.download_mb = 41.5;
    s.update.changelog = read_text(g_data + "/release-notes.md");
    const Lines lines = render(s);
    expect_fits(lines, 80, 24);
    const std::string t = plain(lines);
    EXPECT(has(t, "### Added") && has(t, "`tona`, the control centre"));
    // The long first item is wrapped, not cut at the border: its last words are on screen
    EXPECT(has(t, "per task and downloads it."));
    EXPECT(has(t, "..."));  // the rows beyond the eighth are marked as left out
    EXPECT(!has(t, "Fifth change"));
    s.update.changelog = "Short note.";
    EXPECT(!has(plain(render(s)), "..."));
    dump("update-notes", lines);
}

void test_too_small_and_sizes() {
    UiState s = state("status-decode.json");
    s.width = 52, s.height = 12;
    Lines lines = render(s);
    expect_fits(lines, 52, 12);
    EXPECT(has(plain(lines), "The terminal is too small.") && has(plain(lines), "at least 80 × 24") &&
           has(plain(lines), "It is 52 × 12 now."));
    s.width = 80, s.height = 23;
    EXPECT(has(plain(render(s)), "too small"));
    s.width = 120, s.height = 40;  // larger terminals are filled, not left ragged
    expect_fits(render(s), 120, 40);
    dump("too-small", lines);
}

void test_no_color() {
    UiState s = state("status-decode.json");
    const Lines start = render(s);
    expect_fits(start, 80, 24);
    std::string t = plain(start);
    EXPECT(has(t, "Server running · model Coder · 2 conversations · priority balanced"));
    EXPECT(has(t, "2,897 tokens/s") && has(t, "15.1 of 17 GB"));
    EXPECT(has(t, "Tona 35 · others 14 · free 15 GB"));
    for (Tab tab : {Tab::Start, Tab::Models, Tab::Agents, Tab::Settings}) {
        s.tab = tab;
        for (Modal modal : {Modal::None, Modal::Help, Modal::Close}) {
            s.modal = modal;
            const Lines lines = render(s);
            expect_fits(lines, 80, 24);
            expect_no_raw_keys(plain(lines));
        }
    }
    s.modal = Modal::None, s.tab = Tab::Models;
    t = plain(render(s));
    EXPECT(!has(t, "measured"));  // the usual basis is not repeated on every row
    // NO_COLOR: no colour sequence is left, bold and inverse stay, and the content is the same
    const std::string coloured = frame_text(start, true), bare = frame_text(start, false);
    EXPECT(has(coloured, "\x1b[3") && !has(bare, "\x1b[3") && !has(bare, "\x1b[9"));
    EXPECT(has(bare, "\x1b[1m") && has(bare, "█") && bare.contains("\x1b[0m"));
    EXPECT(strip_color("\x1b[1;31mx\x1b[0m") == "\x1b[1mx\x1b[0m");
    EXPECT(strip_color("\x1b[38;5;196mx\x1b[0m") == "x\x1b[0m");
    setenv("NO_COLOR", "1", 1);
    EXPECT(no_color_requested());
    unsetenv("NO_COLOR");
    EXPECT(!no_color_requested());
}

void test_feed() {
    top::Shared sh;
    sh.up = true;
    sh.status = nlohmann::json::parse(read_text(g_data + "/status-decode.json"));
    const int tona = sh.project("tona"), shop = sh.project("shop");
    sh.rate.push_back({{tona, 90.f}, {shop, 10.f}});
    sh.rate.push_back({{tona, 96.f}});
    UiState ui;
    fill_from_feed(ui, sh);
    EXPECT(ui.server.up && ui.server.ready && ui.server.running.size() == 3 && ui.projects.size() == 2);
    EXPECT(ui.rates.size() == 2 && ui.rates[0] == 100 && ui.rates[1] == 96);
    ui.project_view = 2;  // shop only
    fill_from_feed(ui, sh);
    EXPECT(ui.rates[0] == 10 && ui.rates[1] == 0);
    sh.up = false, sh.error = "Connection refused";
    fill_from_feed(ui, sh);
    EXPECT(!ui.server.up && ui.server.error == "Connection refused");
}

void test_server_view() {
    const ServerView v = parse_status(nlohmann::json::parse(read_text(g_data + "/status-decode.json")), true, "");
    EXPECT(v.ready && v.conversation_count() == 2 && v.running.size() == 3 && v.slots == 6 && v.queue == 1);
    EXPECT(v.running[0].agent == "claude" && v.running[0].state == "decode" && v.running[0].tok_s == 96);
    EXPECT(v.running[1].state == "prefill" && v.running[1].prefill_fraction > 0.41 &&
           v.running[1].prefill_fraction < 0.43);
    EXPECT(v.running[2].state == "waiting" && v.running[2].agent == "junie");
    EXPECT(v.total_tok_s() == 96);
    EXPECT(v.mem.ram_tona == 35e9 && v.recent.size() == 2 && v.recent[1].out_tokens == 1840);
    EXPECT(model_stem(v.model_file) == "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M");
}

void test_server_view_states() {
    const ServerView loading =
        parse_status(nlohmann::json::parse(read_text(g_data + "/status-loading.json")), true, "");
    EXPECT(loading.loading() && loading.load_fraction() > 0.79 && loading.load_fraction() < 0.81);
    const ServerView down = parse_status({}, false, "refused");
    EXPECT(!down.up && down.error == "refused" && down.running.empty());
}

void test_old_server_view() {
    // An older server has no engine.active: the engine itself is the one running request
    nlohmann::json old = nlohmann::json::parse(read_text(g_data + "/status-decode.json"));
    old["engine"].erase("active");
    old["engine"]["state"] = "decode";
    EXPECT(parse_status(old, true, "").conversation_count() == 1);
}

}  // namespace

int main(int argc, char** argv) {
    return run("client_views", [argc, argv] {
        g_data = argc > 1 ? argv[1] : ".";  // inside run: the copy may throw, and run reports it
        test_coming_task_rows();
        test_server_view();
        test_server_view_states();
        test_old_server_view();
        test_feed();
        test_start_page();
        test_start_page_memory_without_vram();
        test_start_variants();
        test_models_page();
        test_agents_and_settings_pages();
        test_states();
        test_dialogs();
        test_update_dialog_release_notes();
        test_too_small_and_sizes();
        test_no_color();
    });
}
