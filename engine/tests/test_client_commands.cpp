// The commands without a screen: output, JSON, exit codes, the confirmation of `tona use`, the settings file
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>

#include "client/commands.h"
#include "core/config.h"
#include "core/context_plan.h"
#include "core/module.h"
#include "tests/client_testkit.h"

using namespace tona::client;
using namespace tona::test;
namespace fs = std::filesystem;

namespace {

std::string g_data;

struct FakeControl : Control {
    std::vector<std::string> calls;
    std::string start() override { return calls.emplace_back("start"), ""; }
    std::string stop() override { return calls.emplace_back("stop"), ""; }
    std::string restart() override { return calls.emplace_back("restart"), ""; }
    std::string download(const std::string& id, const std::string& drafter, const LineFn&) override {
        return calls.push_back("download " + id + " " + drafter), "";
    }
    std::string remove(const std::string& id) override { return calls.push_back("remove " + id), ""; }
};

struct Rig {
    TempDir dir;
    Env env;
    FakeControl control;
    Services svc;
    std::ostringstream out, err;
    Io io;
    bool server_up = true;
    std::string answer;  // "y" or "n"; empty: no terminal to ask on
    std::string question;
    std::function<void(nlohmann::json&)> edit_status = [](nlohmann::json&) {};  // a variant of the fixture

    explicit Rig(const std::string& name) : dir(name) {
        fs::create_directories(dir.path() / "work");
        env = Env::for_checkout(dir.path().string());
        svc.control = &control;
        svc.status = [this] {
            const std::string file = server_up ? "status-decode.json" : "";
            if (!server_up) return parse_status({}, false, "connection refused");
            nlohmann::json j = nlohmann::json::parse(read_text(g_data + "/" + file));
            edit_status(j);
            return parse_status(j, true, "");
        };
        svc.catalog = [] { return test_catalog(); };
        svc.offer = [] { return test_offer(); };
        svc.doctor = [](const tona::install::Settings&) {
            return std::vector<Check>{{"binaries", true, "work/build/engine/tona-server", ""},
                                      {"port", false, "8662 busy", "tona config set port 8663"}};
        };
        io.out = &out, io.err = &err;
    }
    void ask_with(const std::string& a) {
        answer = a;
        io.ask = [this](const std::string& q) { return question = q, answer == "y"; };
    }
    int run(const std::vector<std::string>& args) {
        out.str(""), err.str("");
        return run_command(args, env, svc, io);
    }
};

// What a server on Apple silicon reports (system_status.cpp): no sysfs, so the graphics fields are -1, and one memory
void test_status_unified_memory() {
    Rig r("cmd-status-mac");
    r.edit_status = [](nlohmann::json& j) {
        j["system"]["gpu"] = {{"vram_used", -1}, {"vram_total", -1}, {"unified", true}};
        j["system"]["ram"] = {{"total", 34.36e9}, {"available", 20e9}};
        j["config"]["vram_gb"] = 21.97;
    };
    EXPECT(r.run({"status"}) == kExitOk);
    EXPECT(has(r.out.str(), "Memory shared by CPU and GPU: Tona 22.0 of 34 GB"));
    EXPECT(!has(r.out.str(), "Graphics memory"));
    EXPECT(r.run({"status", "--json"}) == kExitOk);
    const nlohmann::json j = nlohmann::json::parse(r.out.str());
    EXPECT(j["memory_bytes"]["unified"] == true && j["memory_bytes"]["tona_gpu"] == 21.97e9);
    // A server that could read no graphics memory and is not on unified memory: no line rather than "-0.0 of -0 GB"
    r.edit_status = [](nlohmann::json& j) { j["system"]["gpu"] = {{"vram_used", -1}, {"vram_total", -1}}; };
    EXPECT(r.run({"status"}) == kExitOk && !has(r.out.str(), "memory") && !has(r.out.str(), "-0"));
}

void test_status() {
    Rig r("cmd-status");
    EXPECT(r.run({"status"}) == kExitOk);
    EXPECT(has(r.out.str(), "Server: running (http://127.0.0.1:8662)") &&
           has(r.out.str(), "Model: Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M"));
    EXPECT(has(r.out.str(), "Conversations: 2 (writes 96 tokens/s)"));
    EXPECT(has(r.out.str(), "Graphics memory 15.1 of 17 GB, Tona in main memory 35 GB"));
    EXPECT(r.run({"status", "--json"}) == kExitOk);
    const nlohmann::json j = nlohmann::json::parse(r.out.str());
    EXPECT(j["server"]["state"] == "running" && j["model"]["id"] == "coder-iq1_m" && j["conversations"].size() == 3);
    EXPECT(j["decode_tok_s"] == 96 && j["memory_bytes"]["ram_tona"] == 35e9);
    r.server_up = false;
    EXPECT(r.run({"status"}) == kExitServerDown && has(r.out.str(), "not running") &&
           has(r.out.str(), "connection refused"));
    EXPECT(r.run({"status", "--json"}) == kExitServerDown &&
           nlohmann::json::parse(r.out.str())["server"]["state"] == "down");
    EXPECT(r.run({"status", "--bogus"}) == kExitUsage);
}

void test_models() {
    Rig r("cmd-models");
    EXPECT(r.run({"models"}) == kExitOk);
    EXPECT(has(r.out.str(), "coder-iq1_m\tloaded\t55 GB") && has(r.out.str(), "ornith-1.5-35b-a3b\tinstalled\t21 GB"));
    EXPECT(has(r.out.str(), "nemotron-bolt-74b-a9b-24gb\tto download") && has(r.out.str(), "Qwen3.8 27B\tcoming"));
    EXPECT(r.run({"models", "--task", "knowledge", "--json"}) == kExitOk);
    const nlohmann::json j = nlohmann::json::parse(r.out.str());
    EXPECT(j.size() == 1 && j[0]["id"] == "nemotron-bolt-74b-a9b-24gb" && j[0]["state"] == "to_download");
    EXPECT(r.run({"models", "--task", "painting"}) == kExitUsage && has(r.err.str(), "known: coding, knowledge"));
}

// A task without a model is announced with its note, never an empty list (the real offer data)
void test_coming_tasks() {
    {
        Rig r("cmd-coming");
        r.svc.offer = [] { return build_offer(nullptr); };
        EXPECT(r.run({"models", "--task", "images"}) == kExitOk);
        EXPECT(has(r.out.str(), "coming"));
        EXPECT(r.run({"models", "--task", "images", "--json"}) == kExitOk);
        const nlohmann::json j = nlohmann::json::parse(r.out.str());
        EXPECT(j.size() == 1 && j[0]["status"] == "coming" && j[0]["state"] == "coming" &&
               j[0]["tasks"][0] == "images");
        EXPECT(!j[0]["coming_note"].get<std::string>().empty());
        EXPECT(has(j[0]["coming_note"].get<std::string>(), "coming"));
    }
}

void test_use() {
    Rig r("cmd-use");
    // Two conversations run and nobody can be asked: exit 4, nothing written, nothing restarted
    EXPECT(r.run({"use", "ornith"}) == kExitNeedsConfirm);
    EXPECT(has(r.err.str(), "2 conversations are running") && has(r.err.str(), "claude") &&
           has(r.err.str(), "Use --yes"));
    EXPECT(r.control.calls.empty() && !fs::exists(r.env.conf));
    // A terminal that is asked and says no
    r.ask_with("n");
    EXPECT(r.run({"use", "ornith"}) == kExitNeedsConfirm && r.control.calls.empty());
    EXPECT(has(r.question, "Restart anyway?"));
    // --yes writes the model into the settings and restarts
    EXPECT(r.run({"use", "Ornith", "--yes"}) == kExitOk);
    EXPECT(r.control.calls == std::vector<std::string>({"restart"}));
    EXPECT(has(r.out.str(), "Switched to ornith-1.5-35b-a3b, the server restarts."));
    EXPECT(tona::install::read_values(r.env.conf).contains("TONA_MODEL"));
    // A model that is not on the disk is refused before anything is written or restarted
    r.control.calls.clear();
    EXPECT(r.run({"use", "nemotron-bolt-74b-a9b-24gb", "--yes"}) == kExitFailed && has(r.err.str(), "not installed"));
    EXPECT(r.control.calls.empty());
    EXPECT(r.run({"use", "gpt-9", "--yes"}) == kExitUsage && has(r.err.str(), "tona models"));
    EXPECT(r.run({"use", "qwen3.8 27b", "--yes"}) == kExitUsage);  // a model that is coming cannot be used
    // No conversations, no question
    r.svc.status = [] {
        return parse_status(nlohmann::json::parse(read_text(g_data + "/status-idle.json")), true, "");
    };
    r.io.ask = nullptr;
    EXPECT(r.run({"use", "coder-iq1_m"}) == kExitOk && r.control.calls == std::vector<std::string>({"restart"}));
    // The server is down: the setting is written, nothing is started
    r.control.calls.clear(), r.server_up = false;
    r.svc.status = [] { return parse_status({}, false, "refused"); };
    EXPECT(r.run({"use", "ornith"}) == kExitOk && r.control.calls.empty() && has(r.out.str(), "next server start"));
    EXPECT(r.run({"use", "ornith", "--no-restart"}) == kExitOk);
}

// The PC's module is injected: the host's own (the Metal module on a Mac) cuts agents and context to its limits
void use_pc_module(Rig& r) {
    r.svc.resolve = [](tona::install::Settings& s) {
        tona::install::resolve_context(&tona::install::hip_gfx1200_module(), s);
    };
}

void test_config_keeps_other_lines() {
    Rig r("cmd-config");
    use_pc_module(r);
    fs::create_directories(fs::path(r.env.conf).parent_path());
    std::ofstream(r.env.conf) << "TONA_CTX=65536\nSOMETHING_ELSE='keep me'\n";
    r.svc.status = [] { return parse_status({}, false, "refused"); };  // down: no restart, only the file
    EXPECT(r.run({"config", "set", "context", "64K"}) == kExitOk);
    EXPECT(r.run({"config", "get", "context"}) == kExitOk && r.out.str() == "64K\n");
    EXPECT(r.run({"config", "set", "priority", "work"}) == kExitOk);
    EXPECT(r.run({"config", "set", "context", "128K"}) == kExitOk);
    EXPECT(r.run({"config", "get", "priority"}) == kExitOk && r.out.str() == "work\n");
    const std::string text = read_text(r.env.conf);
    EXPECT(has(text, "SOMETHING_ELSE='keep me'") && has(text, "TONA_PRIORITY='work'"));
    EXPECT(has(text, "TONA_CTX='131072'") && !has(text, "TONA_CTX=65536"));
    EXPECT(r.run({"config", "set", "ram_gb", "7"}) == kExitUsage);
    EXPECT(r.run({"config", "set", "agents", "5"}) == kExitUsage &&
           r.run({"config", "set", "agents", "0"}) == kExitUsage);
    EXPECT(r.run({"config", "set", "context", "1M"}) == kExitUsage &&
           has(r.err.str(), "auto|64K|128K|256K"));  // below the table's minimum
    EXPECT(r.run({"config", "set", "nope", "1"}) == kExitUsage && r.run({"config", "get", "nope"}) == kExitUsage);
    EXPECT(r.run({"config", "list"}) == kExitOk && has(r.out.str(), "priority=work") &&
           has(r.out.str(), "context=128K") && has(r.out.str(), "agents=2"));
    // The installer's own save keeps the client's keys
    tona::install::Settings s = tona::install::load(r.env.conf, {});
    s.seqs = 3;
    EXPECT(tona::install::save(s, r.env.repo));
    EXPECT(has(read_text(r.env.conf), "TONA_PRIORITY='work'") &&
           has(read_text(r.env.conf), "SOMETHING_ELSE='keep me'"));
    EXPECT(r.run({"config"}) == kExitUsage);
}

void test_config_restart(Rig& r);

// On a Mac module the limits apply: one agent at 32K, whatever is asked
void test_config_mac_limits() {
    Rig r("cmd-config-mac");
    r.svc.resolve = [](tona::install::Settings& s) {
        tona::install::resolve_context(&tona::install::metal_apple_module(), s);
    };
    r.svc.status = [] { return parse_status({}, false, "refused"); };
    EXPECT(r.run({"config", "set", "agents", "3"}) == kExitOk &&
           r.run({"config", "set", "context", "128K"}) == kExitOk);
    const tona::install::Settings s = tona::install::load(r.env.conf, {});
    EXPECT(s.seqs == 1 && s.ctx == 32768 && !s.ctx_auto);
}

// agents and context: "auto" is the default and follows agents; a size is exact; a running server restarts like `use`
void test_config_agents_and_context() {
    Rig r("cmd-agents");
    r.svc.resolve = [](tona::install::Settings& s) {
        tona::install::resolve_context(&tona::install::hip_gfx1200_module(), s);
    };
    r.svc.status = [] { return parse_status({}, false, "refused"); };
    EXPECT(r.run({"config", "get", "context"}) == kExitOk && r.out.str() == "auto\n");
    EXPECT(r.run({"config", "get", "agents"}) == kExitOk && r.out.str() == "2\n");
    EXPECT(r.run({"config", "set", "ram_gb", "38"}) == kExitOk);
    EXPECT(r.run({"config", "set", "agents", "2"}) == kExitOk);
    tona::install::Settings s = tona::install::load(r.env.conf, {});
    EXPECT(s.ctx_auto && s.ctx == 131072 && s.seqs == 2);  // 256K for two would leave the experts unpinned at 38 GB
    EXPECT(r.run({"config", "set", "agents", "1"}) == kExitOk);
    s = tona::install::load(r.env.conf, {});
    EXPECT(s.ctx_auto && s.ctx == 262144 && s.seqs == 1);
    EXPECT(has(read_text(r.env.conf), "TONA_CTX_AUTO='1'"));
    EXPECT(r.run({"config", "set", "context", "64K"}) == kExitOk && r.run({"config", "set", "agents", "4"}) == kExitOk);
    s = tona::install::load(r.env.conf, {});
    EXPECT(!s.ctx_auto && s.ctx == 65536 && s.seqs == 4);  // a size stays whatever the agents are
    EXPECT(r.run({"config", "set", "context", "auto"}) == kExitOk);
    s = tona::install::load(r.env.conf, {});
    EXPECT(s.ctx_auto && s.ctx == 65536);  // 38 GB, 4 agents: 64K fits besides the experts
    EXPECT(r.control.calls.empty());
    test_config_restart(r);
}

// A running server restarts like `tona use`: with conversations it asks first
void test_config_restart(Rig& r) {
    r.server_up = true;
    r.svc.status = [] {
        return parse_status(nlohmann::json::parse(read_text(g_data + "/status-decode.json")), true, "");
    };
    EXPECT(r.run({"config", "set", "agents", "1"}) == kExitNeedsConfirm && r.control.calls.empty());
    EXPECT(r.run({"config", "set", "agents", "1", "--yes"}) == kExitOk && r.control.calls.size() == 1 &&
           has(r.out.str(), "agents=1, the server restarts"));
    EXPECT(r.run({"config", "set", "context", "128K", "--no-restart"}) == kExitOk && r.control.calls.size() == 1);
    EXPECT(r.run({"config", "set", "port", "9001"}) == kExitOk &&
           r.control.calls.size() == 1);  // other keys: no restart
}

void test_doctor_and_dispatch() {
    Rig r("cmd-doctor");
    EXPECT(r.run({"doctor"}) == kExitWarnings);
    EXPECT(has(r.out.str(), "ok    binaries") &&
           has(r.out.str(), "warn  port  8662 busy  -> tona config set port 8663"));
    EXPECT(r.run({"doctor", "--json"}) == kExitWarnings && nlohmann::json::parse(r.out.str()).size() == 2);
    r.svc.doctor = [](const tona::install::Settings&) { return std::vector<Check>{{"binaries", true, "x", ""}}; };
    EXPECT(r.run({"doctor"}) == kExitOk);
    EXPECT(r.run({"help"}) == kExitOk && has(r.out.str(), "exit codes: 0 ok"));
    EXPECT(r.run({}) == kExitUsage);
    EXPECT(r.run({"frobnicate"}) == kExitUsage && has(r.err.str(), "unknown command 'frobnicate'"));
    EXPECT(is_command("status") && is_command("doctor") && !is_command("claude") && !is_command("--url"));
}

void test_doctor_checks_pure() {
    Facts f;
    f.server_binary = "/x/tona-server", f.server_binary_exists = true, f.toolchain_needed = true,
    f.toolchain_exists = true;
    f.model_id = "coder-iq1_m", f.drafter_id = "mtp-q8_0", f.model_present = true, f.drafter_present = true;
    f.port = 8662, f.port_state = PortState::Ours, f.unit_state = "active", f.ram_total_gb = 64, f.vram_total_gb = 16;
    f.ram_limit_gb = 58, f.vram_limit_gb = 14.5;
    for (const Check& c : run_checks(f)) EXPECT(c.ok);
    f.drafter_present = false, f.port_state = PortState::Busy, f.unit_state = "failed", f.ram_limit_gb = 62;
    f.server_binary_exists = false;
    int bad = 0;
    for (const Check& c : run_checks(f)) {
        bad += !c.ok;
        EXPECT(c.ok == c.fix.empty());  // every warning names its fix, an ok line names none
    }
    EXPECT(bad == 5);  // binaries, models, port, unit, ram
}

}  // namespace

int main(int argc, char** argv) {
    g_data = argc > 1 ? argv[1] : ".";
    return run("client_commands", [] {
        test_status();
        test_status_unified_memory();
        test_models();
        test_coming_tasks();
        test_use();
        test_config_keeps_other_lines();
        test_config_agents_and_context();
        test_config_mac_limits();
        test_doctor_and_dispatch();
        test_doctor_checks_pure();
    });
}
