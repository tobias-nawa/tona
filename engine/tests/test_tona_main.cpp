// The program itself, as a user runs it: the built `tona` copied into a release layout in a temporary directory,
// started with its own HOME, PATH and install folder, against a stub server on port 8677 (the test port of this work;
// the server on 8662 is never touched). stdin and stdout are no terminal, so the control centre falls back to its text.
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>

#include <cstdlib>
#include <iterator>
#include <sstream>
#include <thread>

#include "core/catalog.h"
#include "core/release_steps.h"
#include "tests/client_testkit.h"
#include "third_party/httplib.h"
#include "update/manifest.h"

using namespace tona::test;
namespace fs = std::filesystem;

namespace {

constexpr int kPort = 8677;
std::string g_tona;  // the built program, first argument of this test

class StubServer {
public:
    StubServer() {
        svr_.Get("/status", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"ready": true, "engine": {"state": "idle"}})", "application/json");
        });
        svr_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"n_ctx": 131072})", "application/json");
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

private:
    httplib::Server svr_;
    std::thread thread_;
};

struct Result {
    int code = -1;
    std::string out, err;
};

struct Rig {
    TempDir dir;
    fs::path root, program, bin;

    explicit Rig(const std::string& name) : dir(name) {
        root = dir.path() / "share/tona", bin = dir.path() / "fakebin";
        const fs::path versioned = root / "versions/0.0.1/bin";
        fs::create_directories(versioned), fs::create_directories(bin), fs::create_directories(dir.path() / "home");
        fs::copy_file(g_tona, versioned / "tona");
        program = versioned / "tona";
        fs::create_directory_symlink("versions/0.0.1", root / "current");
        for (const char* name_ : {"claude", "systemctl", "launchctl"}) {
            std::ofstream(bin / name_) << "#!/bin/sh\necho \"" << name_ << " $*\" >> " << dir.path() / "calls.log"
                                       << "\npwd -P >> " << dir.path() / "calls.log" << "\nexit 0\n";
            fs::permissions(bin / name_, fs::perms::owner_all);
        }
    }
    std::string calls() const { return read_text((dir.path() / "calls.log").string()); }

    Result run(const std::string& args, const fs::path& exe = {}, const fs::path& install = {}) const {
        const fs::path out = dir.path() / "stdout", err = dir.path() / "stderr";
        // posix_spawn, not system(): no shell parses the arguments, and the child gets exactly this environment
        std::vector<std::string> argv = {"env",
                                         "-i",
                                         "HOME=" + (dir.path() / "home").string(),
                                         "PATH=" + bin.string() + ":/usr/bin:/bin",
                                         "TONA_INSTALL_DIR=" + (install.empty() ? root : install).string(),
                                         "XDG_STATE_HOME=" + (dir.path() / "state").string(),
                                         "TONA_PORT=" + std::to_string(kPort),
                                         (exe.empty() ? program : exe).string()};
        std::istringstream words(args);
        argv.insert(argv.end(), std::istream_iterator<std::string>(words), std::istream_iterator<std::string>());
        std::vector<char*> raw;
        raw.reserve(argv.size() + 1);
        for (std::string& a : argv) raw.push_back(a.data());
        raw.push_back(nullptr);
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
        posix_spawn_file_actions_addopen(&actions, 1, out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        posix_spawn_file_actions_addopen(&actions, 2, err.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        char* no_env[] = {nullptr};
        pid_t pid = 0;
        int status = -1;
        if (posix_spawn(&pid, "/usr/bin/env", &actions, nullptr, raw.data(), no_env) == 0) waitpid(pid, &status, 0);
        posix_spawn_file_actions_destroy(&actions);
        return {WIFEXITED(status) ? WEXITSTATUS(status) : -1, read_text(out.string()), read_text(err.string())};
    }
};

// What `tona start --foreground` needs to get as far as the server: tona.conf, the model file the settings name and a
// stand-in tona-server whose script is `body`. Returns the model file.
fs::path install_server(const Rig& rig, const std::string& body, in::Settings settings = {}) {
    settings.model = "coder-iq1_m", settings.drafter = "none", settings.port = kPort, settings.api_key = false;
    const in::CatalogEntry* model = in::find_entry(settings.model);
    if (!model) return {};
    const fs::path file = in::model_dir((rig.root / "models").string(), *model) / model->files[0].name();
    fs::create_directories(file.parent_path());
    std::ofstream(file) << "x";
    fs::create_directories(rig.dir.path() / "home/.config/tona");
    std::ofstream(rig.dir.path() / "home/.config/tona/tona.conf") << in::render_release_conf(settings);
    const fs::path server = rig.root / "versions/0.0.1/bin/tona-server";
    std::ofstream(server) << "#!/bin/sh\n" << body;
    fs::permissions(server, fs::perms::owner_all);
    return file;
}

void test_usage_and_unknown_commands() {
    Rig rig("main-usage");
    const Result unknown = rig.run("frobnicate");
    EXPECT(unknown.code == 2 && has(unknown.err, "tona: unknown command 'frobnicate'"));
    const Result bad = rig.run("--bogus");
    EXPECT(bad.code == 2 && has(bad.err, "Usage: tona"));
    const Result help = rig.run("help");
    EXPECT(help.code == 0 && has(help.out, "tona setup") && has(help.out, "tona uninstall"));
    for (const char* gone : {"install-service", "start-server", "--foreground"}) EXPECT(!has(help.out, gone));
}

void test_without_a_terminal_the_centre_prints_the_status() {
    Rig rig("main-status");
    for (const char* args : {"", "top --tab models", "--url http://127.0.0.1:8677 --tab start --start", "status"}) {
        const Result down = rig.run(args);
        EXPECT(down.code == 3);  // the server does not answer: status exits 3
        EXPECT(has(down.out, "http://127.0.0.1:8677"));
    }
    StubServer stub;
    const Result up = rig.run("status");
    EXPECT(up.code == 0 && has(up.out, "8677"));
}

void test_an_agent_and_the_service_commands() {
    Rig rig("main-agent");
    StubServer stub;
    fs::create_directories(rig.dir.path() / "my project");
    const Result run = rig.run("claude '" + (rig.dir.path() / "my project").string() + "' -p hi");
    EXPECT(run.code == 0);
    EXPECT(has(rig.calls(), "claude --permission-mode acceptEdits"));
    EXPECT(has(rig.calls(), (rig.dir.path() / "my project").string()));  // the folder it ran in
    EXPECT(fs::exists(rig.dir.path() / "state/tona/agent-hint.md"));  // the release keeps its state in the user's home

#ifdef __APPLE__
    // macOS has no systemd: `tona start` runs the real `tona start --foreground` in the background, which writes
    // server.pid before it becomes the server (here a stand-in that ends at once)
    install_server(rig, "exit 0\n");
    const fs::path pid_file = rig.dir.path() / "state/tona/server.pid";
    const Result start = rig.run("start");
    EXPECT(start.code == 0 && has(start.out, "started on port 8677") && fs::exists(pid_file));
    const Result restart = rig.run("restart");  // the pid names nothing that runs: "not running", then a start
    EXPECT(restart.code == 0 && has(restart.out, "started on port 8677") && fs::exists(pid_file));
    EXPECT(!has(rig.calls(), "systemctl"));
#else
    const Result start = rig.run("start");
    EXPECT(start.code == 0 && has(start.out, "started on port 8677"));
    EXPECT(has(rig.calls(), "systemctl --user start tona-server"));
    const Result restart = rig.run("restart");
    EXPECT(restart.code == 0 && has(rig.calls(), "systemctl --user restart tona-server"));
#endif
    const Result gone = rig.run("install-service");  // a `just` recipe of a checkout now
    EXPECT(gone.code == 2 && has(gone.err, "unknown command 'install-service'"));
}

void test_version_and_nothing_installed() {
    Rig rig("main-version");
    const Result v = rig.run("--version");
    EXPECT(v.code == 0);
    EXPECT(v.out.starts_with("tona 0.0.1 ") && has(v.out, std::string(tona::update::kPlatform)));
#ifdef __APPLE__
    const std::string backend = "metal";  // the name of the backend module
#else
    const std::string backend = "hip-gfx1200";
#endif
    EXPECT(v.out == "tona 0.0.1 " + std::string(tona::update::kPlatform) + " " + backend + "\n");
    // A tona that is not in an installation and finds none: no centre without a terminal, a pointer to setup
    Rig empty("main-empty");
    fs::remove(empty.root / "current");
    const Result r = empty.run("");
    EXPECT(r.code == 1 && has(r.err, "tona setup"));
    EXPECT(empty.run("start").code == 1);
}

// `tona start --foreground`: the settings of tona.conf become the arguments and the environment of the server, which
// replaces the process (a stand-in that writes them down)
void test_start_foreground_runs_the_server_with_the_settings() {
    Rig rig("main-foreground");
    in::Settings settings;
    settings.ram_gb = 40, settings.vram_gb = 14, settings.ctx = 65536;
    const fs::path record = rig.dir.path() / "server.txt";
    const fs::path file = install_server(
        rig,
        "echo \"$@\" > " + record.string() + "\necho \"ram=$TONA_RAM_GB cwd=$(pwd -P)\" >> " + record.string() + "\n",
        settings);
    EXPECT(!file.empty());
    if (file.empty()) return;
    const Result r = rig.run("start --foreground");
    EXPECT(r.code == 0);
    const std::string seen = read_text(record.string());
    EXPECT(has(seen, file.string() + " --port " + std::to_string(kPort) + " --ctx 65536"));
    EXPECT(!has(seen, "--api-key-file") && !has(seen, "--mtp"));
    EXPECT(has(seen, "ram=40 cwd=" + fs::canonical(rig.root).string()));
    // Without the model file nothing starts, and the message says what to do
    fs::remove(file);
    const Result missing = rig.run("start --foreground");
    EXPECT(missing.code == 1 && has(missing.err, "model file missing") && has(missing.err, "tona setup"));
}

// A fake HOME and install folder: nothing outside them is touched, the models stay unless asked for
void test_uninstall_removes_the_installation_only() {
    Rig rig("main-uninstall");
    fs::create_directories(rig.root / "models/coder");
    std::ofstream(rig.root / "models/coder/m.gguf") << "model";
    const fs::path home = rig.dir.path() / "home";
    fs::create_directories(home / ".local/bin"), fs::create_directories(home / ".config/tona");
    fs::create_directories(home / ".config/systemd/user"), fs::create_directories(home / "Library/LaunchAgents");
    std::ofstream(home / "Library/LaunchAgents/com.tobiasnawa.tona.plist") << "<plist/>\n";
    fs::create_symlink(rig.root / "current/bin/tona", home / ".local/bin/tona");
    std::ofstream(home / ".config/tona/tona.conf") << "TONA_PORT=1\n";
    std::ofstream(home / ".config/systemd/user/tona-server.service") << "[Unit]\n";
    std::ofstream(home / "mine.txt") << "keep";
    // the PATH block of setup goes with the installation, the rest of the profile stays
    std::ofstream(home / ".zshrc") << "# mine\n\n# >>> tona >>>\nexport PATH=x\n# <<< tona <<<\n";
    const Result no_terminal = rig.run("uninstall");
    EXPECT(no_terminal.code == 4 && fs::exists(rig.root / "versions"));
    EXPECT(has(no_terminal.err, "settings and the API key"));  // asks (nobody can answer) and names what goes
    const Result r = rig.run("uninstall --yes");
    EXPECT(r.code == 0 && has(r.out, "the models stay"));
    EXPECT(!fs::exists(rig.root / "versions") && !fs::exists(rig.root / "current"));
    EXPECT(fs::exists(rig.root / "models/coder/m.gguf") && fs::exists(home / "mine.txt"));
    EXPECT(!fs::is_symlink(home / ".local/bin/tona") && !fs::exists(home / ".config/tona"));
    EXPECT(std::string(std::istreambuf_iterator<char>(std::ifstream(home / ".zshrc").rdbuf()), {}) == "# mine\n" &&
           has(r.out, "removed the PATH block from"));
#ifdef __APPLE__
    // The LaunchAgent is unloaded and removed; there is no unit and no systemctl
    EXPECT(!fs::exists(home / "Library/LaunchAgents/com.tobiasnawa.tona.plist"));
    EXPECT(has(rig.calls(), "launchctl bootout") && !has(rig.calls(), "systemctl"));
    EXPECT(fs::exists(home / ".config/systemd/user/tona-server.service"));  // not ours on a Mac: left alone
#else
    EXPECT(!fs::exists(home / ".config/systemd/user/tona-server.service"));
    EXPECT(has(rig.calls(), "systemctl --user stop tona-server") && has(rig.calls(), "systemctl --user disable"));
    EXPECT(fs::exists(home / "Library/LaunchAgents/com.tobiasnawa.tona.plist"));  // not ours on Linux: left alone
#endif
}

void test_uninstall_with_models() {
    Rig rig("main-uninstall-models");
    fs::create_directories(rig.root / "models/coder");
    std::ofstream(rig.root / "models/coder/m.gguf") << "model";
    const Result r = rig.run("uninstall --yes --models");
    EXPECT(r.code == 0 && has(r.out, "with the models") && !fs::exists(rig.root));
}

void test_a_copy_outside_any_installation_says_so() {
    Rig rig("main-lone");
    fs::create_directories(rig.dir.path() / "lone");
    fs::copy_file(g_tona, rig.dir.path() / "lone/tona");
    const Result r = rig.run("start", rig.dir.path() / "lone/tona", rig.dir.path() / "empty");
    EXPECT(r.code == 1 && has(r.err, "neither a checkout nor a release installation"));
    EXPECT(rig.calls().empty());  // nothing was started
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return std::fputs("usage: test_tona_main <path of tona>\n", stderr), 2;
    g_tona = argv[1];
    return run("tona_main", [] {
        test_usage_and_unknown_commands();
        test_without_a_terminal_the_centre_prints_the_status();
        test_an_agent_and_the_service_commands();
        test_a_copy_outside_any_installation_says_so();
        test_version_and_nothing_installed();
        test_start_foreground_runs_the_server_with_the_settings();
        test_uninstall_removes_the_installation_only();
        test_uninstall_with_models();
    });
}
