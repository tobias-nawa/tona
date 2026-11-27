#include "client/doctor.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <filesystem>
#include <format>

#include "client/control.h"
#include "client/text.h"
#include "core/catalog.h"
#include "core/hw.h"
#include "core/sys.h"
#include "third_party/httplib.h"

namespace fs = std::filesystem;

namespace tona::client {

namespace {

bool present(const std::vector<install::ModelInfo>& cat, const std::string& id) {
    for (const install::ModelInfo& m : cat)
        if (m.id == id) return m.present;
    return id.empty();
}

std::string unit_state() {
    if (!install::which("systemctl")) return {};
    const install::RunResult r = install::run("systemctl --user is-active tona-server 2>/dev/null");
    return install::trim(r.out);
}

}  // namespace

Facts gather_facts(const Env& env, const install::Settings& s) {
    Facts f;
    f.release = env.release;
    f.server_binary = env.release ? env.root + "/current/bin/tona-server" : env.repo + "/work/build/engine/tona-server";
    f.server_binary_exists = fs::exists(f.server_binary);
    f.toolchain_needed = !env.release;
    f.toolchain_path = env.repo + "/work/rocm-10.1/bin/hipconfig";
    f.toolchain_exists = fs::exists(f.toolchain_path);
    const std::vector<install::ModelInfo> cat = installed_catalog(env);
    f.model_id = s.model, f.drafter_id = install::drafter_entry_id(s.drafter, s.model);
    f.model_present = present(cat, f.model_id), f.drafter_present = present(cat, f.drafter_id);
    f.profile_needed = fs::exists(fs::path(env.release ? env.root : env.repo) /
                                  (env.release ? "profiles" : "toolchain/profiles") / (s.model + ".bin"));
    f.profile_path =
        env.release ? env.root + "/profiles/" + s.model + ".bin" : env.repo + "/work/profile/" + s.model + ".bin";
    f.profile_exists = fs::exists(f.profile_path);
    f.port = s.port, f.port_state = probe_port(s.port);
    f.unit_state = unit_state();
    const install::Hardware h = install::detect(env.target().dir());
    f.ram_total_gb = (double)h.ram_total / 1e9;
    for (const install::Gpu& g : h.gpus)
        if (!g.integrated) f.vram_total_gb = std::max(f.vram_total_gb, (double)g.vram_total / 1e9);
    f.ram_limit_gb = s.ram_gb, f.vram_limit_gb = s.vram_gb;
    f.key_needed = s.api_key;
    f.key_exists = fs::exists(install::home() + "/.config/tona/api-key");
    return f;
}

namespace {

Check check(const char* name, bool ok, std::string detail, std::string fix = "") {
    return {name, ok, std::move(detail), ok ? "" : std::move(fix)};
}

std::vector<Check> file_checks(const Facts& f) {
    std::vector<Check> out;
    out.push_back(check("binaries", f.server_binary_exists, f.server_binary, f.release ? "tona setup" : "just build"));
    if (f.toolchain_needed) out.push_back(check("toolchain", f.toolchain_exists, f.toolchain_path, "scripts/setup.sh"));
    std::string models = f.model_id + (f.model_present ? " ok" : " missing");
    if (!f.drafter_id.empty()) models += ", " + f.drafter_id + (f.drafter_present ? " ok" : " missing");
    out.push_back(check("models", f.model_present && f.drafter_present, models,
                        "tona models (then d on the model), or tona setup --models " + f.model_id));
    if (f.profile_needed) out.push_back(check("profile", f.profile_exists, f.profile_path, "tona setup"));
    out.push_back(check("key", !f.key_needed || f.key_exists, f.key_needed ? "~/.config/tona/api-key" : "no key set",
                        "just start creates it, or tona setup"));
    return out;
}

std::vector<Check> state_checks(const Facts& f) {
    std::vector<Check> out;
    const char* port_word = f.port_state == PortState::Free   ? "free"
                            : f.port_state == PortState::Ours ? "ours"
                                                              : "busy";
    out.push_back(check("port", f.port_state != PortState::Busy, std::format("{} {}", f.port, port_word),
                        std::format("another program holds port {}: tona config set port {}", f.port, f.port + 1)));
    out.push_back(check("unit", f.unit_state != "failed", f.unit_state.empty() ? "no systemd" : f.unit_state,
                        "systemctl --user reset-failed tona-server; journalctl --user -u tona-server -n 30"));
    const bool ram_ok = f.ram_limit_gb <= 0 || f.ram_total_gb <= 0 || f.ram_limit_gb <= f.ram_total_gb - 4;
    out.push_back(check("ram", ram_ok, std::format("limit {:g} of {:.0f} GB", f.ram_limit_gb, f.ram_total_gb),
                        std::format("leave 4 GB for the system: tona config set ram_gb {:.0f}", f.ram_total_gb - 4)));
    const bool vram_ok = f.vram_limit_gb <= 0 || f.vram_total_gb <= 0 || f.vram_limit_gb <= f.vram_total_gb - 1;
    out.push_back(
        check("vram", vram_ok, std::format("limit {:g} of {:.1f} GB", f.vram_limit_gb, f.vram_total_gb),
              std::format("leave 1 GB for the desktop: tona config set vram_gb {:.0f}", f.vram_total_gb - 1)));
    return out;
}

}  // namespace

std::vector<Check> run_checks(const Facts& f) {
    std::vector<Check> out = file_checks(f);
    for (Check& c : state_checks(f)) out.push_back(std::move(c));
    return out;
}

}  // namespace tona::client
