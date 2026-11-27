// PATH in the shell profiles: which files, one block, no second block, symlinked and read-only files, ZDOTDIR, and the
// removal of exactly the block. HOME and every file are below a temporary folder. Usage: test_shell_path
#include <sys/stat.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/shell_path.h"
#include "tests/test_support.h"
#include "tests/update_testkit.h"

namespace fs = std::filesystem;
namespace in = tona::install;
using namespace tona::test;

namespace {

std::string slurp(const fs::path& p) { return read_file(p.string()); }

void put(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
}

in::PathEnv env_for(const fs::path& home, const std::string& shell) {
    in::PathEnv e;
    e.home = home.string();
    e.shell = shell;
    e.path = "/usr/bin:/bin";
    return e;
}

const in::LineFn kQuiet = [](const std::string&) {};

int count(const std::string& text, const std::string& what) {
    int n = 0;
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) n++;
    return n;
}

void test_shells_and_files() {
    TempDir dir("shell-path-files");
    const fs::path home = dir.path() / "home";
    put(home / ".zshrc", "# mine\nalias l='ls'\n");
    put(home / ".bashrc", "export A=1");  // no final newline
    put(home / ".config/fish/config.fish", "# fish\n");
    const in::PathEnv env = env_for(home, "/usr/bin/zsh");
    EXPECT(in::profile_files(env).size() == 3);
    EXPECT(in::add_to_path(env, home.string() + "/.local/bin", kQuiet).empty());
    const std::string zsh = slurp(home / ".zshrc");
    EXPECT(zsh.starts_with("# mine\nalias l='ls'\n\n# >>> tona >>>\n") && zsh.ends_with("# <<< tona <<<\n"));
    EXPECT(zsh.contains("export PATH=\"$HOME/.local/bin:$PATH\""));
    EXPECT(slurp(home / ".bashrc").starts_with("export A=1\n\n# >>> tona >>>\n"));
    EXPECT(slurp(home / ".config/fish/conf.d/tona.fish").contains("fish_add_path --global \"$HOME/.local/bin\""));
    EXPECT(slurp(home / ".config/fish/config.fish") == "# fish\n");  // the user's fish file is not touched
    EXPECT(!fs::exists(home / ".profile"));
    // a second run adds nothing
    std::vector<std::string> lines;
    in::add_to_path(env, home.string() + "/.local/bin", [&](const std::string& l) { lines.push_back(l); });
    EXPECT(count(slurp(home / ".zshrc"), "# >>> tona >>>") == 1 && lines.size() == 3 && lines[0].contains("already"));
}

void test_login_shell_creates_its_file_and_others_wait() {
    TempDir dir("shell-path-create");
    const fs::path home = dir.path() / "home";
    fs::create_directories(home);
    const std::vector<in::ProfileFile> zsh = in::profile_files(env_for(home, "/bin/zsh"));
    EXPECT(zsh.size() == 1 && zsh[0].file == home / ".zshrc");
    in::PathEnv mac = env_for(home, "/bin/bash");
    mac.macos = true;
    EXPECT(in::profile_files(mac)[0].file == home / ".bash_profile");
    EXPECT(in::profile_files(env_for(home, "/bin/bash"))[0].file == home / ".bashrc");
    const std::vector<in::ProfileFile> fish = in::profile_files(env_for(home, "/usr/bin/fish"));
    EXPECT(fish.size() == 1 && fish[0].file == home / ".config/fish/conf.d/tona.fish");
    const std::vector<in::ProfileFile> other = in::profile_files(env_for(home, "/bin/dash"));
    EXPECT(other.size() == 1 && other[0].file == home / ".profile");
    in::add_to_path(env_for(home, "/bin/zsh"), home.string() + "/.local/bin", kQuiet);
    EXPECT(slurp(home / ".zshrc").starts_with("# >>> tona >>>\n"));  // a new file starts with the block
}

void test_zdotdir_and_xdg() {
    TempDir dir("shell-path-zdotdir");
    const fs::path home = dir.path() / "home";
    in::PathEnv env = env_for(home, "/bin/zsh");
    env.zdotdir = (dir.path() / "zdot").string();
    in::add_to_path(env, home.string() + "/.local/bin", kQuiet);
    EXPECT(fs::exists(dir.path() / "zdot/.zshrc") && !fs::exists(home / ".zshrc"));
    in::PathEnv fish = env_for(home, "/usr/bin/fish");
    fish.xdg_config = (dir.path() / "xdg").string();
    in::add_to_path(fish, home.string() + "/.local/bin", kQuiet);
    EXPECT(fs::exists(dir.path() / "xdg/fish/conf.d/tona.fish"));
}

void test_symlink_and_mode() {
    TempDir dir("shell-path-link");
    const fs::path home = dir.path() / "home";
    put(dir.path() / "dotfiles/zshrc", "# managed\n");
    ::chmod((dir.path() / "dotfiles/zshrc").c_str(), 0600);
    fs::create_directories(home);
    fs::create_symlink(dir.path() / "dotfiles/zshrc", home / ".zshrc");
    const in::PathEnv env = env_for(home, "/bin/zsh");
    EXPECT(in::add_to_path(env, home.string() + "/.local/bin", kQuiet).empty());
    EXPECT(fs::is_symlink(home / ".zshrc"));  // the link stays, its target changed
    EXPECT(slurp(dir.path() / "dotfiles/zshrc").contains("# >>> tona >>>"));
    struct stat st{};
    ::stat((dir.path() / "dotfiles/zshrc").c_str(), &st);
    EXPECT((st.st_mode & 07777) == 0600);
    in::remove_from_path(env, kQuiet);
    EXPECT(fs::is_symlink(home / ".zshrc") && slurp(dir.path() / "dotfiles/zshrc") == "# managed\n");
}

void test_unwritable_file_falls_back_to_the_hint() {
    TempDir dir("shell-path-ro");
    const fs::path home = dir.path() / "home";
    put(home / ".zshrc", "# locked\n");
    put(home / ".bashrc", "# open\n");
    ::chmod((home / ".zshrc").c_str(), 0444);
    const in::PathEnv env = env_for(home, "/bin/zsh");
    const std::vector<in::ProfileFile> failed = in::add_to_path(env, home.string() + "/.local/bin", kQuiet);
    EXPECT(failed.size() == 1 && failed[0].shell == "zsh");
    EXPECT(slurp(home / ".zshrc") == "# locked\n" && slurp(home / ".bashrc").contains("# >>> tona >>>"));
    // the last page shows the lines for the shell the user is in, since its file has no block
    const std::vector<std::string> advice = in::path_advice(env, home.string() + "/.local/bin", true);
    EXPECT(advice.size() == 4 && advice[2].contains("zsh") && advice[2].contains(">> ~/.zshrc"));
}

void test_advice() {
    TempDir dir("shell-path-advice");
    const fs::path home = dir.path() / "home";
    const std::string bin = home.string() + "/.local/bin";
    in::PathEnv env = env_for(home, "/bin/zsh");
    EXPECT(in::path_advice(env, bin, true).size() == 4);  // no block yet
    in::add_to_path(env, bin, kQuiet);
    const std::vector<std::string> a = in::path_advice(env, bin, true);
    EXPECT(a.size() == 1 && a[0] == "✓ added to PATH in ~/.zshrc, open a new terminal");
    EXPECT(in::path_advice(env, bin, false).size() == 4);  // the user said no: the hint
    env.path = "/usr/bin:" + bin;
    EXPECT(in::path_advice(env, bin, true).empty());
    EXPECT(in::path_step(bin, env).title.contains("PATH in ~/.zshrc"));
}

void test_remove_takes_exactly_the_block() {
    TempDir dir("shell-path-remove");
    const fs::path home = dir.path() / "home";
    const std::string zshrc = "# top\nexport X=1\n\n# end\n";
    put(home / ".zshrc", zshrc);
    put(home / ".bashrc", "# >>> other >>>\nkeep\n# <<< other <<<\n");
    put(home / ".config/fish/conf.d/other.fish", "# not ours\n");
    in::PathEnv env = env_for(home, "/bin/zsh");
    const std::string bin = home.string() + "/.local/bin";
    in::add_to_path(env, bin, kQuiet);
    put(home / ".profile", "# hand-written\n");  // no block: untouched
    EXPECT(slurp(home / ".zshrc") != zshrc);
    EXPECT(in::remove_from_path(env, kQuiet) == 3);  // .zshrc, .bashrc and the fish file
    EXPECT(slurp(home / ".zshrc") == zshrc);         // byte for byte as before
    EXPECT(slurp(home / ".bashrc") == "# >>> other >>>\nkeep\n# <<< other <<<\n");
    EXPECT(!fs::exists(home / ".config/fish/conf.d/tona.fish") && fs::exists(home / ".config/fish/conf.d/other.fish"));
    EXPECT(slurp(home / ".profile") == "# hand-written\n");
    EXPECT(in::remove_from_path(env, kQuiet) == 0);  // nothing left to remove
}

}  // namespace

int main() {
    return run("shell_path", [] {
        test_shells_and_files();
        test_login_shell_creates_its_file_and_others_wait();
        test_zdotdir_and_xdg();
        test_symlink_and_mode();
        test_unwritable_file_falls_back_to_the_hint();
        test_advice();
        test_remove_takes_exactly_the_block();
        return 0;
    });
}
