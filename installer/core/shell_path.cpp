#include "core/shell_path.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <format>
#include <fstream>
#include <sstream>

#include "core/release_steps.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::install {

namespace {

constexpr const char* kBegin = "# >>> tona >>>";
constexpr const char* kEnd = "# <<< tona <<<";

std::string env_or_empty(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

std::string tilde(const std::string& path, const std::string& home) {
    return !home.empty() && path.starts_with(home + "/") ? "~" + path.substr(home.size()) : path;
}

// "$HOME/.local/bin" below the home folder, so the profile still works if the folder moves
std::string dir_expr(const std::string& dir, const std::string& home) {
    return !home.empty() && dir.starts_with(home + "/") ? "$HOME" + dir.substr(home.size()) : dir;
}

std::string read_text(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

fs::path fish_file(const PathEnv& env) {
    const fs::path config = env.xdg_config.empty() ? fs::path(env.home) / ".config" : fs::path(env.xdg_config);
    return config / "fish/conf.d/tona.fish";
}

std::string block_for(const ProfileFile& p, const std::string& dir, const std::string& home) {
    const std::string d = dir_expr(dir, home);
    std::string body;
    if (p.shell == "fish")
        body = "fish_add_path --global \"" + d + "\"\n";
    else
        body = std::format(R"(case ":$PATH:" in *":{0}:"*) ;; *) export PATH="{0}:$PATH" ;; esac)", d) + "\n";
    return std::string(kBegin) + "\n# added by tona setup, removed by tona uninstall\n" + body + kEnd + "\n";
}

// The file a symlinked profile points to: dotfile managers link them, and a rename over the link would replace it
fs::path resolve(const fs::path& p, std::error_code& ec) {
    const bool link = fs::is_symlink(fs::symlink_status(p, ec));
    ec.clear();  // a file that does not exist yet is no error
    if (!link) return p;
    return fs::canonical(p, ec);
}

mode_t mode_of(const fs::path& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 ? (st.st_mode & 07777) : 0644;
}

// Writes `text` over the file `p` (or what it links to), keeping its mode; "" on success, else why not
std::string write_profile(const fs::path& p, const std::string& text) {
    std::error_code ec;
    const fs::path target = resolve(p, ec);
    if (ec) return p.string() + ": link does not resolve";
    const bool exists = fs::exists(target, ec);
    if (exists && ::access(target.c_str(), W_OK) != 0) return target.string() + ": not writable";
    return write_atomic(target, text, exists ? mode_of(target) : 0644);
}

// Text without the marked block and the blank line before it; unchanged when there is none
std::string without_block(const std::string& text) {
    const size_t b = text.find(kBegin);
    if (b == std::string::npos) return text;
    size_t e = text.find(kEnd, b);
    if (e == std::string::npos) return text;
    e += std::string(kEnd).size();
    if (e < text.size() && text[e] == '\n') e++;
    size_t start = b;
    if (start > 0 && text[start - 1] == '\n' && (start == 1 || text[start - 2] == '\n')) start--;
    return text.substr(0, start) + text.substr(e);
}

bool has_block(const fs::path& file) {
    std::error_code ec;
    return fs::exists(file, ec) && read_text(file).contains(kBegin);
}

// The shell of $SHELL as zsh, bash, fish or sh (another POSIX shell, or none known)
std::string shell_name(const std::string& shell) {
    const std::string base = fs::path(shell).filename().string();
    return base == "zsh" || base == "bash" || base == "fish" ? base : "sh";
}

}  // namespace

PathEnv current_path_env() {
    PathEnv e;
    e.home = home();
    e.shell = env_or_empty("SHELL");
    e.zdotdir = env_or_empty("ZDOTDIR");
    e.xdg_config = env_or_empty("XDG_CONFIG_HOME");
    e.path = env_or_empty("PATH");
#ifdef __APPLE__
    e.macos = true;
#endif
    return e;
}

std::vector<ProfileFile> profile_files(const PathEnv& env) {
    std::error_code ec;
    const std::string mine = shell_name(env.shell);
    const fs::path home = env.home;
    const fs::path zsh = (env.zdotdir.empty() ? home : fs::path(env.zdotdir)) / ".zshrc";
    const fs::path bash = home / (env.macos ? ".bash_profile" : ".bashrc");
    const fs::path fish = fish_file(env);
    std::vector<ProfileFile> out;
    // The login shell always gets its file (created if needed); another shell only where the user already has one
    if (mine == "zsh" || fs::exists(zsh, ec)) out.push_back({"zsh", zsh});
    if (mine == "bash" || fs::exists(bash, ec)) out.push_back({"bash", bash});
    if (mine == "fish" || fs::exists(fish.parent_path().parent_path(), ec)) out.push_back({"fish", fish});
    if (mine == "sh") out.push_back({"sh", home / ".profile"});
    return out;
}

bool on_path(const std::string& path_env, const std::string& dir) {
    std::string rest = path_env;
    while (!rest.empty()) {
        const size_t colon = rest.find(':');
        if (fs::path(rest.substr(0, colon)) == fs::path(dir)) return true;
        rest = colon == std::string::npos ? "" : rest.substr(colon + 1);
    }
    return false;
}

std::vector<ProfileFile> add_to_path(const PathEnv& env, const std::string& dir, const LineFn& out) {
    std::vector<ProfileFile> failed;
    for (const ProfileFile& p : profile_files(env)) {
        const std::string shown = tilde(p.file.string(), env.home);
        std::error_code ec;
        const std::string old = fs::exists(p.file, ec) ? read_text(p.file) : "";
        if (old.contains(kBegin)) {
            out(shown + ": already set up");
            continue;
        }
        const std::string sep = old.empty() ? "" : (old.ends_with('\n') ? "\n" : "\n\n");
        std::string text = old;
        text += sep;
        text += block_for(p, dir, env.home);
        const std::string err = write_profile(p.file, text);
        if (err.empty()) {
            out(std::format("added {} to PATH in {}", dir, shown));
        } else {
            out(std::format("Skipped {}: {}", shown, err));
            failed.push_back(p);
        }
    }
    return failed;
}

int remove_from_path(const PathEnv& env, const LineFn& out) {
    const fs::path home = env.home;
    std::vector<fs::path> files = {home / ".zshrc", home / ".bashrc", home / ".bash_profile", home / ".profile",
                                   fish_file(env)};
    if (!env.zdotdir.empty()) files.push_back(fs::path(env.zdotdir) / ".zshrc");
    int changed = 0;
    for (const fs::path& f : files) {
        if (!has_block(f)) continue;
        const std::string text = without_block(read_text(f));
        std::error_code ec;
        // the fish file is ours alone: nothing else in it, so it goes
        const bool ours = f == fish_file(env) && text.empty();
        const std::string err = ours ? (fs::remove(f, ec), ec ? ec.message() : "") : write_profile(f, text);
        out(err.empty() ? "removed the PATH block from " + tilde(f.string(), env.home)
                        : "Skipped " + tilde(f.string(), env.home) + ": " + err);
        if (err.empty()) changed++;
    }
    return changed;
}

std::string hint_line(const std::string& shell, const std::string& dir, const std::string& home) {
    if (shell == "fish") return "  fish: fish_add_path " + tilde(dir, home);
    const std::string d = dir_expr(dir, home);
    const std::string file = shell == "bash" ? "~/.bashrc" : shell == "zsh" ? "~/.zshrc" : "~/.profile";
    return "  " + shell + (shell == "sh" ? ":   " : ":  ") + "echo 'export PATH=\"" + d + ":$PATH\"' >> " + file;
}

namespace {

// "~/.zshrc, ~/.bashrc": the profiles that hold the block
std::string files_with_block(const PathEnv& env) {
    std::string out;
    for (const ProfileFile& p : profile_files(env))
        if (has_block(p.file)) out += (out.empty() ? "" : ", ") + tilde(p.file.string(), env.home);
    return out;
}

}  // namespace

// A home folder shortens the lines: a long absolute path would be cut at 80 columns on the last page
std::vector<std::string> hint_lines(const std::string& dir, const std::string& home) {
    return {tilde(dir, home) + " is not in your PATH. Add it:", hint_line("bash", dir, home),
            hint_line("zsh", dir, home), hint_line("fish", dir, home)};
}

std::vector<std::string> path_advice(const PathEnv& env, const std::string& dir, bool modify_path) {
    if (on_path(env.path, dir)) return {};
    if (modify_path) {
        const std::string mine = shell_name(env.shell);
        for (const ProfileFile& p : profile_files(env))
            if (p.shell == mine && has_block(p.file))
                return {"✓ added to PATH in " + files_with_block(env) + ", open a new terminal"};
    }
    return hint_lines(dir, env.home);
}

Step path_step(const std::string& dir, const PathEnv& env) {
    std::string files;
    for (const ProfileFile& p : profile_files(env))
        files += (files.empty() ? "" : ", ") + tilde(p.file.string(), env.home);
    return {"Command tona: add " + tilde(dir, env.home) + " to PATH in " + files,
            "(one block between '" + std::string(kBegin) + "' and '" + kEnd + "' in each file)", on_path(env.path, dir),
            [dir = dir, env = env](const LineFn& out) {
                for (const ProfileFile& p : add_to_path(env, dir, out)) out(hint_line(p.shell, dir, env.home));
                return 0;  // a profile that cannot be written is no failure of the installation
            }};
}

}  // namespace tona::install
