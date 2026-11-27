// The state of the centre without a terminal: keys, dialogs, filters, the settings table, languages of a folder, the
// launch history
#include <filesystem>
#include <fstream>

#include "client/format.h"
#include "client/history.h"
#include "client/languages.h"
#include "client/settings.h"
#include "core/hw.h"
#include "offer/offer.h"
#include "tests/client_testkit.h"

using namespace tona::client;
using namespace tona::test;
namespace fs = std::filesystem;

namespace {

std::string g_data;  // tests/data/client, the first argument

Key ch(char c) { return {KeyCode::Char, c}; }
Key code(KeyCode c) { return {c, 0}; }

UiState running_state() { return test_state("status-decode.json", g_data); }

void test_parse_keys() {
    const std::vector<Key> k = parse_keys("a\x1b[A\x1b[B\x1b[C\x1b[D\x1b[Z\r\x7f\t\x1b");
    EXPECT(k.size() == 10);
    EXPECT(k[0] == ch('a') && k[1] == code(KeyCode::Up) && k[2] == code(KeyCode::Down));
    EXPECT(k[3] == code(KeyCode::Right) && k[4] == code(KeyCode::Left) && k[5] == code(KeyCode::BackTab));
    EXPECT(k[6] == code(KeyCode::Enter) && k[7] == code(KeyCode::Backspace) && k[8] == code(KeyCode::Tab));
    EXPECT(k[9] == code(KeyCode::Esc));
    EXPECT(parse_keys("\x1bOA")[0] == code(KeyCode::Up));  // application mode arrows
}

void test_tabs_and_help() {
    UiState s = running_state();
    handle_key(s, ch('3'));
    EXPECT(s.tab == Tab::Agents);
    handle_key(s, code(KeyCode::Esc));  // Esc on another page goes back to Start
    EXPECT(s.tab == Tab::Start);
    handle_key(s, ch('?'));
    EXPECT(s.modal == Modal::Help);
    handle_key(s, ch('2'));  // nothing but Esc leaves help
    EXPECT(s.modal == Modal::Help && s.tab == Tab::Start);
    handle_key(s, code(KeyCode::Esc));
    EXPECT(s.modal == Modal::None);
    handle_key(s, ch('w'));  // w goes to the model list from every page
    EXPECT(s.tab == Tab::Models);
}

void test_close_dialog() {
    UiState s = running_state();
    EXPECT(handle_key(s, ch('q')).kind == ActionKind::None && s.modal == Modal::Close);
    // Conversations run: "keep the server" first, stopping it is the second entry
    EXPECT(close_options(s) == std::vector<int>({0, 1, 2}));
    EXPECT(handle_key(s, code(KeyCode::Enter)).kind == ActionKind::Quit);
    s.modal = Modal::Close, s.modal_cursor = 0;
    EXPECT(handle_key(s, ch('s')).kind == ActionKind::QuitAndStop);
    s.modal = Modal::Close;
    EXPECT(handle_key(s, code(KeyCode::Esc)).kind == ActionKind::None && s.modal == Modal::None);
    s.modal = Modal::Close, s.modal_cursor = 0;
    handle_key(s, ch(' '));
    EXPECT(s.remember_choice);
    EXPECT(handle_key(s, code(KeyCode::Enter)).arg == "remember");
    // Nobody is talking to the server: stopping it comes first (design: "Läuft kein Gespräch ...")
    s.server.running.clear();
    EXPECT(close_options(s) == std::vector<int>({1, 0, 2}));
    // A remembered choice closes without asking; a server that is down is simply left
    s.config.extras["TONA_CLOSE"] = "stop";
    EXPECT(handle_key(s, ch('q')).kind == ActionKind::QuitAndStop && s.modal == Modal::None);
    s.server.up = false;
    EXPECT(handle_key(s, ch('q')).kind == ActionKind::Quit);
}

void test_models_page_keys() {
    UiState s = running_state();
    s.tab = Tab::Models;
    const std::vector<ModelRow> rows = model_rows(s);
    // Installed (loaded first), to download, coming; "fits this PC" hides nothing here (64 GB RAM)
    EXPECT(rows.size() == 4);
    EXPECT(rows[0].card->name == "Coder" && rows[0].loaded && rows[1].card->name == "Ornith");
    EXPECT(rows[2].section == ModelRow::Section::ToDownload && rows[3].section == ModelRow::Section::Coming);
    handle_key(s, code(KeyCode::Down));
    const Action asked = handle_key(s, code(KeyCode::Enter));  // Enter on Ornith opens the dialog, nothing happens yet
    EXPECT(asked.kind == ActionKind::None && s.modal == Modal::Confirm && s.ask == Ask::SwitchModel);
    EXPECT(s.ask_arg == "ornith-1.5-35b-a3b");
    const Action go = handle_key(s, code(KeyCode::Enter));
    EXPECT(go.kind == ActionKind::SwitchModel && go.arg == "ornith-1.5-35b-a3b" && s.modal == Modal::None);
    handle_key(s, code(KeyCode::Down));
    EXPECT(handle_key(s, ch('d')).arg == "nemotron-bolt-74b-a9b-24gb");  // d downloads the model to fetch
    handle_key(s, code(KeyCode::Down));
    handle_key(s, code(KeyCode::Enter));  // coming: says so, asks nothing
    EXPECT(s.modal == Modal::None && !s.message.empty());
    s.models_cursor = 0;
    EXPECT(handle_key(s, ch('x')).kind == ActionKind::None && s.modal == Modal::None);  // the loaded model stays
    s.models_cursor = 1;
    handle_key(s, ch('x'));
    EXPECT(s.modal == Modal::Confirm && s.ask == Ask::Remove);
}

void test_models_filters() {
    UiState s = running_state();
    s.tab = Tab::Models;
    handle_key(s, ch('f'));             // task: coding
    EXPECT(model_rows(s).size() == 3);  // Bolt is a knowledge model
    handle_key(s, ch('f'));
    EXPECT(model_rows(s).size() == 1);  // knowledge: Bolt only (the coming card lists coding)
    handle_key(s, ch('f'));
    EXPECT(s.filter.task == 0);
    handle_key(s, ch('t'));  // language: the first of the sorted list, C#
    // The coming row stays: a task without a model is never hidden by the language filter
    EXPECT(model_rows(s).size() == 2 && model_rows(s)[0].card->name == "Coder" && model_rows(s)[1].card->coming);
    s.filter.language = 0;
    s.ram_total_gb = 32;  // Coder (55 GB) and Bolt (41 GB) do not fit this PC: hidden; "m" shows everything again
    EXPECT(model_rows(s).size() == 2 && model_rows(s)[0].card->name == "Ornith");  // and the coming card
    handle_key(s, ch('m'));
    EXPECT(model_rows(s).size() == 4);
}

// A Mac (unified memory, 58 GB Coder, 32 GB RAM): the loaded Coder is listed under Installed with its range; the PC
// rule of thumb (RAM plus graphics memory minus 8 GB) would hide it
void test_models_on_a_mac() {
    tona::install::Hardware h;
    h.os = "macos", h.cpu = "Apple M1 Max", h.cpu_vendor = "Apple", h.ram_total = 32LL << 30;
    tona::install::Gpu g;
    g.vendor = "Apple", g.arch = "apple-m1-max", g.vram_total = h.ram_total / 100 * 78;
    h.gpus = {g};
    UiState s = running_state();
    s.tab = Tab::Models;
    s.offer = build_offer(&h);
    s.ram_total_gb = 34.4, s.vram_total_gb = 26.8;
    const std::vector<ModelRow> rows = model_rows(s);
    EXPECT(!rows.empty() && rows[0].card->id == "coder-iq1_m" && rows[0].section == ModelRow::Section::Installed);
    EXPECT(!rows.empty() && rows[0].loaded && rows[0].card->decode_tok_s_hi > rows[0].card->decode_tok_s);
}

void test_agents_page() {
    UiState s = running_state();
    handle_key(s, ch('3'));
    EXPECT(agent_command(s) == "tona claude ~/projekte/sensor-daemon");
    handle_key(s, code(KeyCode::Right));
    EXPECT(s.agents.agent == 1 && agent_command(s).starts_with("tona opencode"));
    handle_key(s, code(KeyCode::Tab));  // Tab moves between the fields: now the project text
    EXPECT(s.agents.field == AgentForm::Project && in_text_field(s));
    handle_key(s, ch('q'));  // a text field takes the key: nothing closes, nothing switches
    handle_key(s, ch('2'));
    EXPECT(s.modal == Modal::None && s.tab == Tab::Agents &&
           s.agents.project.ends_with("daemon"
                                      "q2"));
    handle_key(s, code(KeyCode::Backspace));
    handle_key(s, code(KeyCode::Backspace));
    EXPECT(s.agents.project.ends_with("daemon"));
    EXPECT(handle_key(s, code(KeyCode::Enter)).kind == ActionKind::ProjectChanged);  // Enter leaves the field
    EXPECT(!in_text_field(s));
    s.agents.field = AgentForm::Tools;
    handle_key(s, ch(' '));
    s.agents.field = AgentForm::Effort;
    handle_key(s, code(KeyCode::Right));
    EXPECT(agent_command(s) == "TONA_TOOLS=lean TONA_EFFORT=high tona opencode ~/projekte/sensor-daemon");
    EXPECT(handle_key(s, code(KeyCode::Enter)).kind == ActionKind::LaunchAgent);
}

void test_suggestion_never_switches() {
    UiState s = running_state();
    // Rust project, Coder loaded, Ornith installed and strong in Rust: a suggestion
    const ModelCard* c = suggest_model(s);
    EXPECT(c && c->name == "Ornith");
    s.tab = Tab::Agents;
    EXPECT(handle_key(s, ch('w')).kind == ActionKind::None && s.modal == Modal::Confirm);  // w asks, never acts
    s.modal = Modal::None;
    s.languages = {{"Python", 1.0}};  // the loaded model is strong in it: nothing to suggest
    EXPECT(suggest_model(s) == nullptr);
    s.languages = {{"Rust", 1.0}};
    s.catalog[1].present = false;  // Ornith is not installed: nothing to suggest either
    EXPECT(suggest_model(s) == nullptr);
}

void test_settings_page() {
    UiState s = running_state();
    s.config.s.ram_gb = 40;
    handle_key(s, ch('4'));
    EXPECT(setting_groups().size() == 6 && group_rows(0).size() == 4);
    handle_key(s, code(KeyCode::Right));  // first row "use": desktop -> server
    EXPECT(get_value(s.config, "use") == "server" && s.settings.dirty);
    handle_key(s, code(KeyCode::Down));
    handle_key(s, code(KeyCode::Down));
    handle_key(s, code(KeyCode::Right));  // ram_gb + 2
    EXPECT(get_value(s.config, "ram_gb") == "42");
    EXPECT(handle_key(s, ch('s')).kind == ActionKind::SaveConfig);
    handle_key(s, code(KeyCode::Tab));
    EXPECT(s.settings.group == 1 && s.settings.row == 0);
    EXPECT(handle_key(s, ch('r')).kind == ActionKind::None && s.ask == Ask::Restart);
}

void test_settings_table_validates() {
    Config c;
    EXPECT(set_value(c, "port", "8700").empty() && c.s.port == 8700);
    EXPECT(!set_value(c, "port", "80").empty());  // below 1024
    EXPECT(!set_value(c, "port", "abc").empty());
    EXPECT(set_value(c, "priority", "tona").empty() && c.extras["TONA_PRIORITY"] == "tona");
    EXPECT(!set_value(c, "priority", "maybe").empty());  // not one of the choices
    EXPECT(!set_value(c, "model", "x").empty());         // models change with `tona use`
    EXPECT(set_value(c, "oea", "off").empty() && !c.s.oea);
    const SettingSpec& ram = *find_setting("ram_gb");
    EXPECT(step_value(c, ram, 1) == "8" && step_value(c, ram, -1) == "8");  // unset (0) plus a step: the minimum is 8
    c.s.ram_gb = 40;
    EXPECT(step_value(c, ram, 1) == "42" && step_value(c, ram, -1) == "38");
}

void write_file(const fs::path& p, size_t bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << std::string(bytes, 'x');
}

void test_language_detection() {
    TempDir dir("client-lang");
    write_file(dir.path() / "src/main.rs", 8100);
    write_file(dir.path() / "src/lib.rs", 100);
    write_file(dir.path() / "run.sh", 1200);
    write_file(dir.path() / "README.md", 5000);             // no source file: ignored
    write_file(dir.path() / "target/debug/big.rs", 50000);  // build output: skipped
    write_file(dir.path() / ".git/hooks/x.py", 50000);      // hidden folder: skipped
    write_file(dir.path() / "node_modules/a/b.js", 50000);
    const std::vector<LanguageShare> l = detect_languages(dir.path().string());
    EXPECT(l.size() == 2 && l[0].language == "Rust" && l[1].language == "Shell/Bash");
    EXPECT(l[0].share > 0.87 && l[0].share < 0.88);
    EXPECT(languages_text(l) == "Rust 87 %, Shell/Bash 13 %");
    EXPECT(detect_languages((dir.path() / "missing").string()).empty());
    EXPECT(language_of("a.rs") == "Rust" && language_of("Makefile").empty() && language_of("x.CS").empty());
    setenv("HOME", "/home/x", 1);
    EXPECT(expand_home("~/p") == "/home/x/p" && expand_home("/a") == "/a" && expand_home("~u/p") == "~u/p");
}

void test_launch_history() {
    TempDir dir("client-history");
    EXPECT(load_launches(dir.path().string()).empty());  // no file: empty, no error
    for (int i = 0; i < 12; i++)
        EXPECT(record_launch(dir.path().string(), {"claude", "p" + std::to_string(i), "Coder", 100 + i}).empty());
    const std::vector<LaunchEntry> l = load_launches(dir.path().string());
    EXPECT(l.size() == kMaxLaunches && l[0].project == "p11" && l[0].time == 111);  // newest first, capped
    std::ofstream(dir.path() / "launches.json") << "{broken";
    EXPECT(load_launches(dir.path().string()).empty());  // a damaged file is an empty history
}

}  // namespace

// The offer comes from the shared data: the measured rates, the verdicts per language, "open" where nothing is known
static void test_number_parsing() {
    EXPECT(to_int("8662") == 8662 && to_int("") == 0 && to_int("12x", 7) == 7 && to_int("-3") == -3);
    EXPECT(to_double("14.5") == 14.5 && to_double("abc", 2.5) == 2.5 && to_double("", 1) == 1);
}

// every card shows the rate of its own model's card, at the context it names (never another model's)
static void test_card_rates_are_the_models_own(const Offer& o) {
    for (const tona::offer::ModelCard& m : tona::offer::model_cards()) {
        const ModelCard* c = o.find(m.id);
        const tona::offer::Rate* r = m.rate_at(16384);
        EXPECT(c && r && c->decode_tok_s == r->decode_tps.value_or(0) && c->speed_context == r->context_tokens);
    }
}

static void test_offer_languages_and_tasks(const Offer& o);

// A Mac lists the models of its class with the range of the Mac estimate, never the PC rates (same class choice as
// `tona setup`)
static void test_offer_on_a_mac() {
    tona::install::Hardware h;
    h.os = "macos", h.cpu = "Apple M1 Max", h.cpu_vendor = "Apple", h.ram_total = 32LL << 30;
    tona::install::Gpu g;
    g.vendor = "Apple", g.arch = "apple-m1-max", g.vram_total = h.ram_total / 100 * 78;
    h.gpus = {g};
    const Offer o = build_offer(&h);
    const ModelCard* coder = o.find("coder-iq1_m");
    EXPECT(coder && !o.find("ornith-1.5-35b-a3b") && !o.find("nemotron-bolt-74b-a9b-24gb"));
    EXPECT(coder && coder->decode_tok_s > 0 && coder->decode_tok_s < 17.9 && coder->decode_tok_s_hi > 17.9);
    EXPECT(coder && coder->speed_basis == Basis::Provisional && coder->decode_tok_s_hi < 67.8);
    EXPECT(build_offer(nullptr).cards.size() > o.cards.size());
}

static void test_offer_from_data() {
    const Offer o = build_offer(nullptr);
    const ModelCard* coder = o.find("coder-iq1_m");
    const ModelCard* ornith = o.find("ornith-1.5-35b-a3b");
    EXPECT(coder && ornith && coder->name == "Coder" && ornith->name == "Ornith");
    if (!coder || !ornith) return;  // EXPECT records and goes on; the checks below dereference both
    EXPECT(coder->decode_tok_s == 67.8 && coder->speed_context == 16000 && coder->speed_basis == Basis::Measured);
    test_card_rates_are_the_models_own(o);
    EXPECT(card_strong_in(*coder, "Python") && !card_strong_in(*coder, "Rust"));
    EXPECT(!coder->weak_note.empty() && !coder->open.empty());  // languages without a measurement stay open
    EXPECT(coder->size_gb > 40 && !coder->license.empty());
    EXPECT(basis_word(Basis::Measured) == "measured" && basis_word(Basis::None) == "open");
    test_offer_languages_and_tasks(o);
}

static void test_offer_languages_and_tasks(const Offer& o) {
    EXPECT(build_offer(nullptr).cards.size() == o.cards.size() && !build_offer(nullptr).tasks.empty());
    bool coming = false;
    for (const Task& task : o.tasks) coming = coming || task.coming;
    EXPECT(coming);  // tasks without a model are marked, not hidden
}

int main(int argc, char** argv) {
    g_data = argc > 1 ? argv[1] : ".";
    return run("client_state", [] {
        test_parse_keys();
        test_tabs_and_help();
        test_close_dialog();
        test_models_page_keys();
        test_models_filters();
        test_agents_page();
        test_suggestion_never_switches();
        test_settings_page();
        test_settings_table_validates();
        test_language_detection();
        test_launch_history();
        test_number_parsing();
        test_offer_from_data();
        test_offer_on_a_mac();
        test_models_on_a_mac();
        test_offer_on_a_mac();
    });
}
