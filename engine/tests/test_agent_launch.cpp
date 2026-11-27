// The launcher with a fake agent on PATH that writes down its arguments, working directory and environment, against a
// stub server on port 8676 (the test port of this work; the server on 8662 is never touched). The terminal of the exit
// question is a function.
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>

#include "client/agent_launch.h"
#include "tests/client_testkit.h"
#include "third_party/httplib.h"

using namespace tona::client;
using namespace tona::test;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr int kPort = 8676;

// Answers /status, /health and /v1/models like the real server; `models` empty: that route is missing (404)
class StubServer {
public:
    // `statuses`: the answers of /status in turn, the last one stays; empty: a server that is up and idle
    explicit StubServer(std::string models = "", std::vector<std::string> statuses = {})
        : models_(std::move(models)), statuses_(std::move(statuses)) {
        if (statuses_.empty()) statuses_.emplace_back(R"({"ready": true, "engine": {"state": "idle"}})");
        svr_.Get("/status", [this](const httplib::Request&, httplib::Response& res) {
            const size_t i = std::min<size_t>(polls_++, statuses_.size() - 1);
            res.set_content(statuses_[i], "application/json");
        });
        svr_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"n_ctx": 262144})", "application/json");
        });
        svr_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
            asked_with_ = req.get_header_value("Authorization");
            if (models_.empty()) return res.status = 404, void();
            res.set_content(models_, "application/json");
        });
        thread_ = std::thread([this] { svr_.listen("127.0.0.1", kPort); });
        svr_.wait_until_ready();
    }
    ~StubServer() {
        svr_.stop();
        if (thread_.joinable()) thread_.join();
    }
    StubServer(const StubServer&) = delete;
    StubServer& operator=(const StubServer&) = delete;
    std::string asked_with_;

private:
    std::string models_;
    std::vector<std::string> statuses_;
    std::atomic<size_t> polls_{0};
    httplib::Server svr_;
    std::thread thread_;
};

// Starts and stops the stub like the unit would; counts the calls
struct FakeControl : Control {
    std::unique_ptr<StubServer>* server = nullptr;
    std::string models;
    std::vector<std::string> statuses;
    std::string stop_error;
    int starts = 0, stops = 0;
    std::string start() override { return starts++, server->reset(new StubServer(models, statuses)), ""; }
    std::string stop() override { return stops++, server->reset(), stop_error; }
    std::string restart() override { return ""; }
    std::string download(const std::string&, const std::string&, const LineFn&) override { return ""; }
    std::string remove(const std::string&) override { return ""; }
};

std::vector<std::string> split_nul(const std::string& s) {
    std::vector<std::string> parts;
    size_t from = 0;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] != '\0') continue;
        parts.push_back(s.substr(from, i - from));
        from = i + 1;
    }
    return parts;
}

// One launch in a directory of its own: HOME, PATH with the fake agents, a checkout with work/, a key file
struct Rig {
    TempDir dir;
    fs::path home, bin, repo, project;
    Env env;
    std::unique_ptr<StubServer> server;
    FakeControl control;
    std::ostringstream out, err;
    LaunchIo io;
    std::string question, answer;
    std::string old_path;
    bool had_path = false;
    fs::path old_cwd;  // a path already: converting a string in the destructor could throw

    explicit Rig(const std::string& name) : dir(name) {
        home = dir.path() / "home", bin = dir.path() / "fakebin", repo = dir.path() / "repo";
        project = dir.path() / "my project";  // a folder with a space: the arguments must arrive unsplit
        for (const fs::path& p : {home / ".config/tona", bin, repo / "work", project}) fs::create_directories(p);
        std::ofstream(home / ".config/tona/api-key") << "secretkey\n";
        for (const char* agent : {"claude", "opencode", "junie", "pi"}) fake_agent(agent);
        const char* path = std::getenv("PATH");  // may be unset in a bare environment
        had_path = path != nullptr;
        old_path = had_path ? path : "";
        old_cwd = fs::current_path();
        ::setenv("HOME", home.c_str(), 1);
        ::setenv("PATH", (bin.string() + ":" + old_path).c_str(), 1);
        for (const char* v : {"TONA_TOOLS", "TONA_EFFORT", "TONA_MODE", "TONA_COMPACT", "TONA_PORT",
                              "TONA_API_KEY_FILE", "PI_CODING_AGENT_DIR"})
            ::unsetenv(v);
        env = Env::for_checkout(repo.string());
        env.port = kPort;
        control.server = &server;
        io.out = &out, io.err = &err;
        io.ask = [this](const std::string& q) { return question = q, answer; };
    }
    ~Rig() {
        if (had_path)
            ::setenv("PATH", old_path.c_str(), 1);
        else
            ::unsetenv("PATH");
        std::error_code ec;
        fs::current_path(old_cwd, ec);
    }
    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;

    void fake_agent(const std::string& name) const {
        const fs::path script = bin / name;
        std::ofstream(script) << "#!/bin/sh\nd=" << dir.path() / "out" << "\nmkdir -p \"$d\"\n"
                              << R"(printf '%s\0' "$@" > "$d/)" << name << ".args\"\n"
                              << "pwd -P > \"$d/" << name << ".cwd\"\necho \"$PWD\" > \"$d/" << name << ".pwd\"\n"
                              << "env -0 > \"$d/" << name << ".env\"\nexit \"${FAKE_EXIT:-0}\"\n";
        fs::permissions(script, fs::perms::owner_all);
    }
    std::vector<std::string> args_of(const std::string& agent) const {
        return split_nul(read_text((dir.path() / "out" / (agent + ".args")).string()));
    }
    std::string env_of(const std::string& agent, const std::string& name) const {
        for (const std::string& kv : split_nul(read_text((dir.path() / "out" / (agent + ".env")).string())))
            if (kv.starts_with(name + "=")) return kv.substr(name.size() + 1);
        return "<unset>";
    }
    std::string file_of(const std::string& agent, const char* ext) const {
        std::string t = read_text((dir.path() / "out" / (agent + "." + ext)).string());
        while (!t.empty() && t.back() == '\n') t.pop_back();
        return t;
    }
    bool ran(const std::string& agent) const { return fs::exists(dir.path() / "out" / (agent + ".args")); }
    int run(const std::vector<std::string>& args, const LaunchOptions& options = {}) {
        return run_agent(env, control, args, io, options);
    }
};

void test_claude_in_a_folder_with_spaces() {
    Rig rig("launch-claude");
    StubServer stub;  // a server already runs: it is not started, and nothing is asked on exit
    EXPECT(rig.run({"claude", rig.project.string(), "-p", "hi there", "--flag=a b"}) == 0);
    const std::vector<std::string> args = rig.args_of("claude");
    EXPECT(args.size() >= 8 && args[0] == "--permission-mode" && args[1] == "acceptEdits");
    EXPECT(std::find(args.begin(), args.end(), "--effort") != args.end());
    EXPECT(args[args.size() - 3] == "-p" && args[args.size() - 2] == "hi there" && args.back() == "--flag=a b");
    EXPECT(rig.file_of("claude", "cwd") == rig.project.string());
    EXPECT(rig.file_of("claude", "pwd") == rig.project.string());
    EXPECT(rig.env_of("claude", "ANTHROPIC_BASE_URL") == "http://127.0.0.1:8676");
    EXPECT(rig.env_of("claude", "ANTHROPIC_AUTH_TOKEN") == "secretkey");  // the newline of the key file is gone
    EXPECT(rig.env_of("claude", "ANTHROPIC_CUSTOM_HEADERS") == "X-Tona-Project: my project");
    EXPECT(rig.env_of("claude", "CLAUDE_CODE_MAX_CONTEXT_TOKENS") == "262144");
    EXPECT(rig.control.starts == 0 && rig.question.empty());
    EXPECT(fs::exists(rig.repo / "work/tona/agent-hint.md"));
}

void test_no_folder_runs_in_the_current_directory() {
    Rig rig("launch-nofolder");
    StubServer stub;
    fs::current_path(rig.project);
    EXPECT(rig.run({"claude", "-r"}) == 0);
    EXPECT(rig.file_of("claude", "cwd") == rig.project.string());
    EXPECT(rig.env_of("claude", "ANTHROPIC_CUSTOM_HEADERS") == "X-Tona-Project: my project");
    EXPECT(rig.args_of("claude").back() == "-r");  // not a folder: an option of the agent
    EXPECT(rig.run({"claude", "no-such-folder", "-p"}) == 0);
    EXPECT(rig.args_of("claude")[rig.args_of("claude").size() - 2] == "no-such-folder");  // goes to the agent as it is
}

void test_the_agent_exit_code_is_the_result() {
    Rig rig("launch-exit");
    StubServer stub;
    ::setenv("FAKE_EXIT", "7", 1);
    EXPECT(rig.run({"claude"}) == 7);
    ::unsetenv("FAKE_EXIT");
}

void test_model_and_settings_come_from_the_server_and_the_conf() {
    Rig rig("launch-model");
    StubServer stub(R"json({"data": [{"id": "other-model", "display_name": "Other (local)"}]})json");
    std::ofstream(rig.repo / "work/tona.conf") << "TONA_EFFORT='high'\n";
    ::setenv("TONA_COMPACT", "200k", 1);
    EXPECT(rig.run({"claude"}) == 0);
    ::unsetenv("TONA_COMPACT");
    EXPECT(rig.env_of("claude", "ANTHROPIC_MODEL") == "other-model");
    EXPECT(rig.env_of("claude", "CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "200k");
    const std::vector<std::string> args = rig.args_of("claude");
    EXPECT(std::find(args.begin(), args.end(), "high") != args.end());
    EXPECT(stub.asked_with_ == "Bearer secretkey");
}

void test_the_coder_is_the_fallback_without_the_models_route() {
    Rig rig("launch-fallback");
    StubServer stub;
    EXPECT(rig.run({"claude"}) == 0);
    EXPECT(rig.env_of("claude", "ANTHROPIC_MODEL") == "qwen3.8-flash-next-coder");
}

void test_key_file_from_the_environment_and_no_key() {
    Rig rig("launch-key");
    StubServer stub;
    ::setenv("TONA_API_KEY_FILE", (rig.dir.path() / "other-key").c_str(), 1);
    EXPECT(rig.run({"claude"}) == 0 && rig.env_of("claude", "ANTHROPIC_AUTH_TOKEN") == "local");  // no such file
    std::ofstream(rig.dir.path() / "other-key") << "abc";
    EXPECT(rig.run({"claude"}) == 0 && rig.env_of("claude", "ANTHROPIC_AUTH_TOKEN") == "abc");
    ::unsetenv("TONA_API_KEY_FILE");
}

void test_the_other_agents_start_the_same_way() {
    Rig rig("launch-others");
    StubServer stub;
    EXPECT(rig.run({"opencode", rig.project.string(), "run", "task"}) == 0);
    EXPECT(rig.args_of("opencode") == std::vector<std::string>({"run", "task"}));
    EXPECT(rig.file_of("opencode", "cwd") == rig.project.string() &&
           rig.env_of("opencode", "TONA_PROJECT") == "my project");
    EXPECT(json::parse(rig.env_of("opencode", "OPENCODE_CONFIG_CONTENT"))["provider"]["tona"]["models"].size() == 1);
    EXPECT(rig.run({"junie", rig.project.string()}) == 0);
    EXPECT(rig.args_of("junie")[0] == "--model-location" && fs::exists(rig.repo / "work/tona/junie-models/tona.json"));
    ::setenv("PI_CODING_AGENT_DIR", (rig.dir.path() / "pi").c_str(), 1);
    EXPECT(rig.run({"pi", rig.project.string()}) == 0);
    EXPECT(rig.args_of("pi")[0] == "--model" && fs::exists(rig.dir.path() / "pi/models.json"));
    ::unsetenv("PI_CODING_AGENT_DIR");
}

void test_exit_question_stops_the_server_it_started() {
    Rig rig("launch-ask-yes");
    rig.control.models = "";
    EXPECT(rig.run({"claude"}) == 0);
    EXPECT(rig.control.starts == 1 && has(rig.out.str(), "tona-server is not running, starting it"));
    EXPECT(has(rig.question, "Stop tona-server (frees RAM and GPU)? [Y/n]"));
    EXPECT(rig.control.stops == 1 && has(rig.out.str(), "stopped"));  // an empty answer is yes

    Rig again("launch-ask-n");
    again.answer = "n";
    EXPECT(again.run({"claude"}) == 0);
    EXPECT(again.control.starts == 1 && again.control.stops == 0);
    EXPECT(has(again.out.str(), "keeps running, stop it later with: tona stop"));
    again.control.server->reset();
}

void test_loading_progress_is_shown_until_the_model_is_ready() {
    Rig rig("launch-loading");
    // The first answer is the check whether a server runs; then two loading states, then ready
    const std::string loading =
        R"json({"ready": false, "phases": [{"name": "weights", "done": 50, "total": 100}]})json";
    const std::string none = R"json({"ready": false, "phases": []})json";
    StubServer stub("", {loading, loading, none, R"json({"ready": true, "engine": {"state": "idle"}})json"});
    EXPECT(rig.run({"claude"}) == 0);
    EXPECT(has(rig.out.str(), "loading: weights 50 %") && has(rig.out.str(), "loading: \xe2\x80\xa6"));
    EXPECT(rig.control.starts == 0 && rig.ran("claude"));
}

void test_error_while_loading_stops_the_launch() {
    Rig rig("launch-load-error");
    StubServer stub("", {R"json({"ready": false})json", R"json({"ready": false, "error": "no model file"})json"});
    EXPECT(rig.run({"claude"}) == 1);
    EXPECT(has(rig.err.str(), "Error while loading: no model file") && !rig.ran("claude"));
}

void test_a_failing_stop_is_reported_and_a_broken_agent_gives_127() {
    Rig rig("launch-stop-fails");
    rig.control.stop_error = "stop failed (1): unit not found";
    EXPECT(rig.run({"claude"}) == 0 && has(rig.err.str(), "tona: stop failed (1): unit not found"));
    EXPECT(!has(rig.out.str(), "stopped"));
    rig.control.server->reset();

    StubServer stub;
    std::ofstream(rig.bin / "claude") << "#!/nonexistent/interpreter\n";  // found on PATH, but cannot be executed
    ::setenv("PATH", rig.bin.c_str(), 1);  // no other claude further down the path that execvp could fall back to
    EXPECT(rig.run({"claude"}) == 127);
}

void test_no_terminal_means_stop() {
    Rig rig("launch-noterminal");
    rig.io.ask = nullptr;  // no terminal to ask on
    EXPECT(rig.run({"claude"}) == 0 && rig.control.stops == 1);
}

void test_agent_not_installed_and_server_that_never_comes_up() {
    Rig rig("launch-errors");
    fs::remove(rig.bin / "pi");
    ::setenv("PATH", rig.bin.c_str(), 1);  // nothing else: pi is not found
    EXPECT(rig.run({"pi"}) == 1 && has(rig.err.str(), "tona: pi is not installed") && rig.control.starts == 0);

    // The start worked as far as the command goes, but no server answers (the unit failed or is slow)
    struct SilentControl : FakeControl {
        std::string start() override { return starts++, ""; }
    } silent;
    ::setenv("PATH", (rig.bin.string() + ":/usr/bin:/bin").c_str(), 1);
    EXPECT(run_agent(rig.env, silent, {"claude"}, rig.io, LaunchOptions{1}) == 1);
    EXPECT(silent.starts == 1 && !rig.ran("claude") && has(rig.err.str(), "did not come up"));

    struct FailingControl : FakeControl {
        std::string start() override { return "start failed (1): no unit"; }
    } failing;
    EXPECT(run_agent(rig.env, failing, {"claude"}, rig.io) == 1 && has(rig.err.str(), "start failed (1): no unit"));
}

}  // namespace

int main() {
    return run("agent_launch", [] {
        test_claude_in_a_folder_with_spaces();
        test_no_folder_runs_in_the_current_directory();
        test_the_agent_exit_code_is_the_result();
        test_model_and_settings_come_from_the_server_and_the_conf();
        test_the_coder_is_the_fallback_without_the_models_route();
        test_key_file_from_the_environment_and_no_key();
        test_the_other_agents_start_the_same_way();
        test_exit_question_stops_the_server_it_started();
        test_no_terminal_means_stop();
        test_loading_progress_is_shown_until_the_model_is_ready();
        test_error_while_loading_stops_the_launch();
        test_a_failing_stop_is_reported_and_a_broken_agent_gives_127();
        test_agent_not_installed_and_server_that_never_comes_up();
    });
}
