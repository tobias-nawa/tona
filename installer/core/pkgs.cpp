#include "core/pkgs.h"

#include <format>
#include <regex>

#include "core/sys.h"

namespace tona::install {

namespace {

// first version number in the output of "prog --version"
std::string version_of(const std::string& prog) {
    const RunResult r = run(quote(prog) + " --version");
    std::smatch m;
    static const std::regex re(R"((\d+)\.(\d+)(\.\d+)?)");
    return std::regex_search(r.out, m, re) ? m.str(0) : "";
}

bool at_least(const std::string& v, int major, int minor) {
    std::smatch m;
    static const std::regex re(R"((\d+)\.(\d+))");
    if (!std::regex_search(v, m, re)) return false;
    const int a = std::stoi(m.str(1)), b = std::stoi(m.str(2));
    return a > major || (a == major && b >= minor);
}

std::vector<Package> common() {
    // C++23 with <format>: GCC 13 or newer; CMake 3.28 or newer (engine/CMakeLists.txt)
    auto same = [](const char* id, const char* purpose) {
        return Package{.id = id, .purpose = purpose, .dnf = id, .apt = id, .pacman = id, .brew = id};
    };
    return {
        {.id = "g++",
         .purpose = "C++ compiler (engine and installer)",
         .dnf = "gcc-c++",
         .apt = "g++",
         .pacman = "gcc"},
        same("cmake", "build"),
        {.id = "ninja",
         .purpose = "build",
         .dnf = "ninja-build",
         .apt = "ninja-build",
         .pacman = "ninja",
         .brew = "ninja"},
        same("git", "sources and toolchain"),
        {.id = "python3",
         .purpose = "fetch the toolchain, download models (3.11 or newer)",
         .dnf = "python3",
         .apt = "python3",
         .pacman = "python",
         .brew = "python"},
        same("curl", "the trial run and downloads (just, scripts)"),
        same("jq", "the trial run reads the server status"),
        same("just", "engine commands (just start, just top …)"),
        {.id = "claude",
         .purpose = "Claude Code, with the engine through the command tona claude",
         .required = false,
         .hint = "curl -fsSL https://claude.ai/install.sh | bash"},
    };
}

std::string version_problem(const Package& p) {
    if (p.id == "g++" && !at_least(p.version, 13, 0)) return "GCC " + p.version + ", needs 13 or newer";
    if (p.id == "cmake" && !at_least(p.version, 3, 28)) return "CMake " + p.version + ", needs 3.28 or newer";
    if (p.id == "python3" && !at_least(p.version, 3, 11)) return "Python " + p.version + ", needs 3.11 or newer";
    return {};
}

}  // namespace

std::vector<Package> check_packages(const Hardware&, std::vector<Package> extra, bool release) {
    std::vector<Package> all = common();
    if (release) std::erase_if(all, [](const Package& p) { return p.required; });
    for (Package& p : extra) all.push_back(std::move(p));
    for (Package& p : all) {
        if (p.id.empty() || p.id.front() == '/') continue;  // the module checks files itself
        p.found = which(p.id).has_value();
        if (!p.found) continue;
        p.version = version_of(p.id);
        p.problem = version_problem(p);
    }
    return all;
}

std::string package_manager(const Hardware& h) {
    const std::string ids = " " + h.distro + " " + h.distro_like + " ";
    auto has = [&](const char* s) { return ids.contains(std::string(" ") + s + " "); };
    if (h.os == "macos") return "brew";
    if (has("fedora") || has("rhel") || has("centos")) return "dnf";
    if (has("debian") || has("ubuntu")) return "apt";
    if (has("arch")) return "pacman";
    return {};
}

std::string install_command(const Hardware& h, const std::vector<Package>& pkgs) {
    const std::string pm = package_manager(h);
    std::string names;
    for (const Package& p : pkgs) {
        if (p.found && p.problem.empty()) continue;
        const std::string& n = pm == "dnf" ? p.dnf : pm == "apt" ? p.apt : pm == "pacman" ? p.pacman : p.brew;
        if (!n.empty()) names += " " + n;
    }
    if (names.empty() || pm.empty()) return {};
    if (pm == "dnf") return "sudo dnf install" + names;
    if (pm == "apt") return "sudo apt install" + names;
    if (pm == "pacman") return "sudo pacman -S --needed" + names;
    return "brew install" + names;
}

}  // namespace tona::install
