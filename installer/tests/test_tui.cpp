// Snapshots of the pages that show speed and steps: the model page, its details and the Go page, on
// an invented reference PC (the dump of the Wizard at 80 x 24). Usage: test_tui
#include <filesystem>
#include <regex>
#include <string>

#include "core/flow.h"
#include "core/module.h"
#include "core/report.h"
#include "tests/test_support.h"
#include "tui.h"

namespace in = tona::install;
namespace fs = std::filesystem;
using namespace tona::test;

namespace {

in::Hardware ref_pc() {
    in::Hardware h;
    h.os = "linux", h.distro = "fedora", h.cpu = "AMD Ryzen 9 9950X", h.cpu_vendor = "AuthenticAMD", h.cpu_family = 26;
    h.cores = 16, h.ram_total = static_cast<int64_t>(64e9), h.ram_avail = static_cast<int64_t>(50e9), h.ram_gbs = 72;
    h.kfd_access = true, h.disk_free = static_cast<int64_t>(500e9);
    in::Gpu g;
    g.pci_vendor = 0x1002, g.pci_device = 0x7590, g.vendor = "AMD", g.arch = "gfx1200";
    g.vram_total = static_cast<int64_t>(17.1e9), g.vram_used = static_cast<int64_t>(1.6e9);
    h.gpus = {g};
    return h;
}

// The text of the pages of the dump, one string per page name ("=== Models" up to the next "===")
std::string page(const std::string& dump, const std::string& name) {
    const size_t a = dump.find("=== " + name + "\n");
    if (a == std::string::npos) return {};
    const size_t b = dump.find("\n=== ", a + 1);
    return dump.substr(a, b == std::string::npos ? std::string::npos : b - a);
}

std::string dump_of(const fs::path& root, double disk_free = 500e9) {
    in::Start st;
    st.h = ref_pc();
    st.h.disk_free = static_cast<int64_t>(disk_free);
    st.module = in::find_module(st.h);
    st.match = in::match_hw_class(st.h);
    st.recommended = st.module->recommend(st.h);
    st.selection = in::default_selection(st.match.cls);
    st.settings = in::apply_selection(st.recommended, st.selection);
    in::Wizard w(std::move(st), in::Options{}, in::release_target(root.string()));
    return w.dump_all();
}

size_t count(const std::string& text, const std::regex& re) {
    return static_cast<size_t>(
        std::distance(std::sregex_iterator(text.begin(), text.end(), re), std::sregex_iterator()));
}

// Rows of "N. text" of the Go page: the number of the steps listed
size_t numbered_rows(const std::string& go) { return count(go, std::regex("\n\\s*[0-9]+\\. ")); }

bool has_range(const std::string& text, const std::string& unit) {
    return count(text, std::regex("[0-9]+–[0-9]+ " + unit)) > 0;
}

// rows: a range per model, no exact number, the one footer line
void test_models_page(const std::string& models, const std::string& footer, const std::string& unit) {
    EXPECT(count(models, std::regex("[0-9]+–[0-9]+ " + unit)) >= 3);
    EXPECT(!std::regex_search(models, std::regex("≈ [0-9]+ " + unit)));
    EXPECT(!std::regex_search(models, std::regex("[0-9]\\.[0-9] " + unit)));
    EXPECT(!models.contains("measured"));
    EXPECT(models.contains(footer));
}

// details: both rates as ranges, the inputs and the anchor line
void test_details_page(const std::string& details, const std::string& unit) {
    EXPECT(count(details, std::regex("[0-9]+–[0-9]+ " + unit)) == 2 && has_range(details, unit));
    EXPECT(details.contains("Anchor: measured on RX 9060 XT 16 GB"));
    EXPECT(details.contains("Inputs: "));
    EXPECT(details.contains("±15 %"));
}

// Go: every step listed, none cut off, the models by their names
void test_go_page(const std::string& go) {
    EXPECT(numbered_rows(go) >= 8 && !go.contains("…"));
    EXPECT(go.contains("Download and verify: Coder · 62.5 GB"));
    EXPECT(!go.contains("coder-iq1_m") && !go.contains("Fetch and verify") && !go.contains("Expert profile coder"));
    EXPECT(go.contains("What you get") && go.contains("Server      port 8662 · this PC only · not at login"));
    EXPECT(go.contains("Agents      2 · context auto"));
    EXPECT(std::regex_search(go, std::regex("Download    [0-9]+\\.[0-9] GB · ≈ ")));
    EXPECT(!go.contains("[Y/n]") && !go.contains("checked once a day"));
    EXPECT(!go.contains("Check for updates") && !go.contains("to PATH in"));  // asked on the Finish screen
    EXPECT(go.contains("Ready to set up") && go.contains("Memory      balanced · Tona"));
}

// Running: the header has no page count, finished steps a check mark, the running one its bar, then the total bar
void test_running_page(const std::string& run) {
    EXPECT(run.contains("Setting up Tona") && run.contains("step 3 of ") && !run.contains("tona setup"));
    EXPECT(!run.contains("Machine ›"));
    EXPECT(run.contains("✓  Download, verify and install the signed release"));
    EXPECT(run.contains("▸  Download and verify: Ornith") && run.contains("·  Download and verify: Bolt"));
    EXPECT(std::regex_search(run, std::regex("downloading +[█▒░]+ [0-9]+ %")));
    EXPECT(std::regex_search(run, std::regex("\nTotal +[█▒░]+ 70 %  74\\.8 GB of 106\\.3 GB  .* left\n")));
    EXPECT(run.contains("Ctrl+C ends after the step; a download resumes where it left off"));
}

// Room: the two fields with their default, the speed range of the model for them, nothing about them in "advanced"
void test_room_page(const std::string& room, const std::string& advanced) {
    EXPECT(room.contains("Parallel agents") && room.contains("(•) 2"));
    EXPECT(room.contains("Context per agent") && room.contains("(•) auto") && room.contains("256K"));
    // one line per chosen model: the sizes are planned per model; rate of each agent and of all together
    EXPECT(room.contains("Decode on this PC, estimated"));
    EXPECT(std::regex_search(room, std::regex("Coder +2 x 128K +[0-9]+–[0-9]+ tok/s each, [0-9]+–[0-9]+ together")));
    EXPECT(std::regex_search(room, std::regex("Bolt +2 x [0-9]+K +[0-9]+–[0-9]+ tok/s each")) &&
           !room.contains("not pinned"));  // auto: 38 GB, experts pinned
    EXPECT(room.contains("Port") && !room.contains("check for a new release"));
    EXPECT(!advanced.contains("Context per") && !advanced.contains("Simultaneous"));
}

// The model page follows the language page (Python, TypeScript, Rust: Ornith is strong in Rust), and the columns,
// the sums and the rule are in place
void test_follows_languages(const std::string& models) {
    EXPECT(models.contains("[x] Coder") && models.contains("[x] Ornith (alternative)") && models.contains("[x] Bolt"));
    EXPECT(!models.contains("is stronger in your languages") && !models.contains("is off"));
    EXPECT(models.contains("strong: JavaScript, Rust"));
    EXPECT(std::regex_search(models, std::regex("Coder +[0-9]+–[0-9]+ tok/s +[0-9]+\\.[0-9] GB +download")));
    EXPECT(models.contains("To download   ") && models.contains("Free          ") && models.contains("Time          "));
    EXPECT(models.contains("\n\nKnowledge and research\n"));
    EXPECT(models.contains("Your models") && models.contains("preselected from your languages"));
    EXPECT(std::regex_search(models, std::regex("\n +Model +Speed per agent +Size +Status\n")));
}

// A drive that holds Coder and Bolt but not Ornith too: Ornith stays off and the page says why in one line
void test_proposal_too_big() {
    const std::string models = page(dump_of(fs::temp_directory_path() / "tona-test-tui", 100e9), "Models");
    EXPECT(models.contains("[ ] Ornith (alternative)") && models.contains("[x] Coder"));
    EXPECT(models.contains("Ornith (21.7 GB) is off: not enough free disk"));
    EXPECT(!models.contains("add it with the space bar"));
}

void test_pages(const char* footer, const char* unit) {
    const std::string dump = dump_of(fs::temp_directory_path() / "tona-test-tui");
    test_models_page(page(dump, "Models"), footer, unit);
    test_follows_languages(page(dump, "Models"));
    test_proposal_too_big();
    test_details_page(page(dump, "Models (i)"), unit);
    test_go_page(page(dump, "Go"));
    test_room_page(page(dump, "Room"), page(dump, "Room (e)"));
    test_running_page(page(dump, "Go (running)"));
}

// A Mac: the chip name is not shown twice, and the plan text names the real folder and no "?" for the drive
void test_mac_machine() {
    in::Hardware h = ref_pc();
    h.os = "macos", h.distro_name = "macOS 15", h.cpu = "Apple M1 Max", h.cpu_vendor = "Apple";
    in::Gpu g;
    g.vendor = "Apple", g.name = "Apple M1 Max", g.arch = "apple-m1-max", g.vram_total = static_cast<int64_t>(50e9);
    h.gpus = {g};
    h.disk_path = "/Users/x/.local/share/tona";
    in::Start st;
    st.h = h;
    st.module = in::find_module(h);
    st.match = in::match_hw_class(h);
    st.selection = in::default_selection(st.match.cls);
    in::Wizard w(std::move(st), in::Options{}, in::release_target("/Users/x/.local/share/tona"));
    const std::string dump = w.dump_all();
    const std::string machine = page(dump, "Machine");
    EXPECT(machine.contains("Apple M1 Max") && !machine.contains("Apple Apple"));
    EXPECT(machine.contains("Your class") && machine.contains("Mac, 64 GB") && !machine.contains("none of Tona's"));
    EXPECT(!dump.contains("this PC") && !dump.contains("other PCs"));
    EXPECT(!dump.contains("Ornith") && !dump.contains("Bolt"));  // models the Metal backend cannot run are not offered
    const std::string room = page(dump, "Room");
    EXPECT(room.contains("Decode on this Mac") && room.contains("I work on this Mac"));
    // the Mac runs one agent at 32K: both are marked, nothing else is offered
    EXPECT(room.contains("Parallel agents     (•) 1\n") || room.contains("(•) 1 "));
    EXPECT(room.contains("(•) 32K") && !room.contains("256K") && !room.contains("auto"));
    EXPECT(room.contains("(coming)") && std::regex_search(room, std::regex("Coder +1 x 32K")));
    const std::string text = in::hardware_text(h, nullptr);
    EXPECT(text.contains("free under /Users/x/.local/share/tona") && !text.contains("(?"));
}

// A size that is no radio (a hand-written configuration) is listed and marked, never an empty choice
void test_room_marks_the_current_context() {
    in::Start st;
    st.h = ref_pc();
    st.module = in::find_module(st.h);
    st.match = in::match_hw_class(st.h);
    st.recommended = st.module->recommend(st.h);
    st.selection = in::default_selection(st.match.cls);
    st.settings = in::apply_selection(st.recommended, st.selection);
    st.settings.ctx_auto = false, st.settings.ctx = 32768;
    in::Wizard w(std::move(st), in::Options{}, in::release_target("/Users/x/.local/share/tona"));
    const std::string room = page(w.dump_all(), "Room");
    EXPECT(room.contains("(•) 32K") && room.contains("( ) auto") && room.contains("256K"));
    EXPECT(count(room, std::regex("Context per agent[^\n]*\\(•\\)")) == 1);
}

}  // namespace

int main() {
    return run("tui", [] {
        test_mac_machine();
        test_room_marks_the_current_context();
        test_pages("Decode at 16K context, estimated for this PC from GPU, RAM and CPU.", "tok/s");
        return 0;
    });
}
