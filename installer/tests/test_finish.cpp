// The Finish screen after the installation: the rows that show, Space, Enter, and what lands in the profile and on
// "What next". HOME and every folder are below a temporary folder. Usage: test_finish
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "core/module.h"
#include "core/shell_path.h"
#include "tests/test_support.h"
#include "tests/update_testkit.h"
#include "tui.h"

namespace in = tona::install;
namespace fs = std::filesystem;
using namespace tona::test;
using ftxui::Event;

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

// A wizard whose installation just finished, on a PC whose HOME is `home`
std::unique_ptr<in::Wizard> finished(const fs::path& home, bool release, const std::string& path_env) {
    in::Start st;
    st.h = ref_pc();
    st.module = in::find_module(st.h);
    st.match = in::match_hw_class(st.h);
    st.recommended = st.module->recommend(st.h);
    st.selection = in::default_selection(st.match.cls);
    st.settings = in::apply_selection(st.recommended, st.selection);
    in::Target t = release ? in::release_target((home / "root").string()) : in::Target{};
    t.repo = (home / "repo").string();
    t.bin_dir = (home / ".local/bin").string();
    auto w = std::make_unique<in::Wizard>(std::move(st), in::Options{}, t);
    in::PathEnv env;
    env.home = home.string(), env.shell = "/bin/zsh", env.path = path_env;
    w->set_path_env(env);
    w->stage_finished();
    return w;
}

std::string slurp(const fs::path& p) { return read_file(p.string()); }

void test_rows_toggle_and_apply() {
    TempDir dir("finish-apply");
    const fs::path& home = dir.path();
    auto w = finished(home, true, "/usr/bin");
    std::string screen = w->screen_text();
    EXPECT(screen.starts_with("Tona is installed") && screen.contains("all ") && screen.contains(" steps done"));
    EXPECT(screen.contains("Last two things") && screen.contains("[x] Add ~/.local/bin to PATH in ~/.zshrc"));
    EXPECT(screen.contains("adds a marked block; tona uninstall removes it"));
    EXPECT(screen.contains("[x] Check for updates once a day") && screen.contains("nothing installs without your ok"));
    EXPECT(!fs::exists(home / ".zshrc"));  // nothing is written before Enter
    w->press(Event::Character(' '));       // switch the PATH row off, then on again
    EXPECT(w->screen_text().contains("[ ] Add ~/.local/bin"));
    w->press(Event::Character(' '));
    EXPECT(w->screen_text().contains("[x] Add ~/.local/bin"));
    w->press(Event::ArrowDown);
    w->press(Event::Character(' '));  // the update checks off
    EXPECT(w->screen_text().contains("[ ] Check for updates"));
    w->press(Event::Return);
    EXPECT(slurp(home / ".zshrc").contains("# >>> tona >>>"));
    screen = w->screen_text();
    EXPECT(screen.starts_with("Tona is ready") && !screen.contains("tona setup"));
    EXPECT(screen.contains("What next") && screen.contains("✓ added to PATH in ~/.zshrc, open a new terminal"));
    EXPECT(slurp(home / "root/state.json").contains("\"checks\":false"));
}

void test_unticked_path_writes_nothing_and_shows_the_hint() {
    TempDir dir("finish-no");
    // a long home (macOS temp folders are): the lines still fit the 80 columns
    const fs::path home = dir.path() / std::string(70, 'h');
    fs::create_directories(home);
    auto w = finished(home, true, "/usr/bin");
    w->press(Event::Character(' '));
    w->press(Event::Return);
    EXPECT(!fs::exists(home / ".zshrc"));
    const std::string screen = w->screen_text();
    EXPECT(screen.contains("is not in your PATH. Add it:") && screen.contains("fish_add_path ~/.local/bin"));
}

// The error of a root open to others: the chmod command is on the screen in full width, however long the path is
void test_failed_step_shows_the_whole_error() {
    TempDir dir("finish-failed");
    const fs::path home = dir.path() / std::string(70, 'h');
    fs::create_directories(home);
    auto w = finished(home, true, "/usr/bin");
    const std::string root = (home / "root").string();
    w->stage_failed("Failed: install root is open to others; run:\n  chmod 700 " + root);
    const std::string screen = w->screen_text();
    EXPECT(screen.contains("install root is open to others; run:"));
    EXPECT(screen.contains("chmod 700"));
    EXPECT(!screen.contains("…"));  // nothing cut
    size_t from = 0;
    while (from < screen.size()) {  // no line wider than the 80 columns (one column per code point here)
        const size_t nl = screen.find('\n', from);
        const size_t end = nl == std::string::npos ? screen.size() : nl;
        size_t cols = 0;
        for (size_t i = from; i < end; i++) cols += (static_cast<unsigned char>(screen[i]) & 0xC0) != 0x80;
        EXPECT(cols <= 80);
        from = end + 1;
    }
}

// A Finish task that fails says why on the Finish page and keeps it; after the fix, Enter goes on to What next
void test_failed_finish_task_shows_its_cause() {
    TempDir dir("finish-task-failed");
    const fs::path& home = dir.path();
    auto w = finished(home, true, "/usr/bin");
    fs::create_directories(home / "root");
    fs::permissions(home / "root", fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec |
                                       fs::perms::others_read | fs::perms::others_exec);
    w->press(Event::Character(' '));  // PATH off, the update checks stay on
    w->press(Event::Return);
    std::string screen = w->screen_text();
    EXPECT(screen.contains("Last two things") && screen.contains("Failed: ") && screen.contains("chmod 700"));
    EXPECT(!screen.contains("What next"));
    fs::permissions(home / "root", fs::perms::owner_all);
    w->press(Event::Return);
    screen = w->screen_text();
    EXPECT(screen.contains("What next") && !screen.contains("Failed: "));
}

void test_hidden_rows() {
    TempDir dir("finish-hidden");
    const fs::path& home = dir.path();
    // on PATH already: only the update row
    auto on = finished(home, true, "/usr/bin:" + (home / ".local/bin").string());
    std::string screen = on->screen_text();
    EXPECT(screen.contains("Last two things") && !screen.contains("to PATH in") &&
           screen.contains("Check for updates"));
    // not a release: only the PATH row
    auto checkout = finished(home, false, "/usr/bin");
    screen = checkout->screen_text();
    EXPECT(screen.contains("to PATH in") && !screen.contains("Check for updates"));
    // neither row: no Finish screen, What next at once
    auto none = finished(home, false, "/usr/bin:" + (home / ".local/bin").string());
    screen = none->screen_text();
    EXPECT(!screen.contains("Last two things") && screen.contains("What next"));
}

}  // namespace

int main() {
    test_failed_finish_task_shows_its_cause();
    return run("finish", [] {
        test_rows_toggle_and_apply();
        test_unticked_path_writes_nothing_and_shows_the_hint();
        test_failed_step_shows_the_whole_error();
        test_hidden_rows();
        return 0;
    });
}
