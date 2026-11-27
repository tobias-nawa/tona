#include "client/foreground.h"

#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <sstream>
#include <vector>

#include "core/release_steps.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::client {

ForegroundPlan plan_foreground(const Env& env) {
    ForegroundPlan p;
    if (!env.release) {
        p.error = "start --foreground is for a release installation; a checkout starts with just start";
        return p;
    }
    const install::Settings s = load_settings(env);
    const install::UnitSpec spec = install::unit_spec(s, env.target());
    if (spec.model_file.empty() || !fs::exists(spec.model_file)) {
        p.error = (spec.model_file.empty() ? "unknown model " + s.model : "model file missing: " + spec.model_file) +
                  "; run tona setup";
        return p;
    }
    p.launch = install::server_launch(spec, env.macos);
    p.ram_gb = spec.ram_gb;
    return p;
}

std::string own_unit_name(const std::string& cgroup_text) {
    std::istringstream in(cgroup_text);
    std::string line, last;
    while (std::getline(in, line))
        if (!line.empty()) last = line;
    const size_t slash = last.rfind('/');
    const std::string leaf = slash == std::string::npos ? last : last.substr(slash + 1);
    return leaf.ends_with(".service") ? leaf : std::string();
}

namespace {

// MemoryMax follows the RAM limit of tona.conf, which the unit does not know. A warning, not a stop: the server keeps
// to TONA_RAM_GB itself, the cgroup limit is the second line of defence.
void limit_memory(double ram_gb, std::ostream& err) {
    std::ifstream in("/proc/self/cgroup");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string unit = own_unit_name(ss.str());
    if (unit.empty() || ram_gb <= 0) return;
    const install::RunResult r =
        install::run("systemctl --user set-property --runtime " + install::quote(unit) +
                     " MemoryMax=" + std::to_string(install::memory_max_bytes(ram_gb)) + " MemorySwapMax=0 2>&1");
    if (r.status != 0)
        err << "tona: could not set the memory limit of " << unit << ": " << install::trim(r.out) << "\n";
}

// macOS: the pid file that `tona stop` reads; the server keeps this pid through the exec
void write_pid_file(const Env& env) {
    std::error_code ec;
    fs::create_directories(env.state_dir, ec);
    std::ofstream(env.server_pid()) << ::getpid() << "\n";
}

}  // namespace

int run_foreground(const Env& env, std::ostream& err) {
    const ForegroundPlan p = plan_foreground(env);
    if (!p.error.empty()) return err << "tona: " << p.error << "\n", 1;
    if (env.macos)
        write_pid_file(env);
    else
        limit_memory(p.ram_gb, err);
    for (const auto& [name, value] : p.launch.env) ::setenv(name.c_str(), value.c_str(), 1);
    if (::chdir(p.launch.workdir.c_str()) != 0) return err << "tona: cannot enter " << p.launch.workdir << "\n", 1;
    std::vector<char*> argv;
    argv.reserve(p.launch.argv.size() + 1);
    for (const std::string& a : p.launch.argv) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    ::execv(argv[0], argv.data());
    err << "tona: cannot start " << p.launch.argv[0] << ": " << std::strerror(errno) << "\n";
    return 1;
}

}  // namespace tona::client
