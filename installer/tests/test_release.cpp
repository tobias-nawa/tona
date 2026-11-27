// Release installation without a checkout: the systemd unit, the embedded catalog, model download with resume and
// checks, and the first installation of a signed release against a local server. Nothing here touches the real
// ~/.config or calls systemctl: HOME and every directory are below a temporary folder.
#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/catalog.h"
#include "core/config.h"
#include "core/release_steps.h"
#include "tests/test_support.h"
#include "tests/update_testkit.h"
#include "update/activate.h"
#include "update/fsutil.h"
#include "update/state.h"

namespace fs = std::filesystem;
namespace in = tona::install;
using namespace tona::test;

namespace {

std::string slurp(const fs::path& p) { return tona::test::read_file(p.string()); }

mode_t mode_of(const fs::path& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 ? st.st_mode & 07777 : 0;
}

in::UnitSpec reference_spec() {
    in::UnitSpec u;
    u.root = "/home/u/.local/share/tona";
    u.model_file = u.root + "/models/coder-iq1_m/m-00001-of-00002.gguf";
    u.drafter_flag = "--mtp";
    u.drafter_file = u.root + "/models/mtp-q8_0/mtp.gguf";
    u.profile = u.root + "/profiles/coder-iq1_m.bin";
    u.prompt_cache = u.root + "/prompt-cache";
    u.api_key_file = "/home/u/.config/tona/api-key";
    u.ram_gb = 60;
    u.vram_gb = 14.9;
    return u;
}

bool has_line(const std::string& text, const std::string& line) { return ("\n" + text).contains("\n" + line + "\n"); }

// The unit holds one line that matters and no setting: the settings are read from tona.conf by `tona start`
void test_unit_text() {
    const std::string t = in::render_unit(reference_spec().root);
    EXPECT(has_line(t, "ExecStart=/home/u/.local/share/tona/current/bin/tona start --foreground"));
    EXPECT(has_line(t, "Environment=TONA_INSTALL_DIR=/home/u/.local/share/tona"));
    EXPECT(has_line(t, "MemorySwapMax=0"));
    EXPECT(has_line(t, "Restart=on-failure"));
    EXPECT(has_line(t, "RestartSec=30"));
    EXPECT(has_line(t, "StartLimitBurst=3"));
    EXPECT(has_line(t, "StartLimitIntervalSec=600"));
    EXPECT(has_line(t, "WantedBy=default.target"));
    // No model, port, limit or tuning in the unit: a change of settings never writes it again
    for (const char* setting : {"--port", "--ctx", "MemoryMax", "TONA_RAM_GB", "TONA_VRAM_GB", "TONA_OEA_K0", "models/",
                                "profiles/", "api-key", "bin/tona-server"})
        EXPECT(!t.contains(setting));
    EXPECT(t.find("ExecStart=") == t.rfind("ExecStart="));
    EXPECT(!t.contains("versions/"));  // current/: the updater only moves the link
}

// What `tona start --foreground` gives the server: the old ExecStart and Environment lines, now in one place
void test_server_launch() {
    const in::ServerLaunch l = in::server_launch(reference_spec(), false);
    EXPECT(l.workdir == "/home/u/.local/share/tona");
    EXPECT((l.argv == std::vector<std::string>{"/home/u/.local/share/tona/current/bin/tona-server",
                                               "/home/u/.local/share/tona/models/coder-iq1_m/m-00001-of-00002.gguf",
                                               "--mtp", "/home/u/.local/share/tona/models/mtp-q8_0/mtp.gguf", "--port",
                                               "8662", "--ctx", "262144", "--api-key-file",
                                               "/home/u/.config/tona/api-key"}));
    const auto value = [&l](const std::string& name) {
        for (const auto& [n, v] : l.env)
            if (n == name) return v;
        return std::string("<unset>");
    };
    EXPECT(value("TONA_EXPERT_PROFILE") == "/home/u/.local/share/tona/profiles/coder-iq1_m.bin");
    EXPECT(value("TONA_RAM_GB") == "60" && value("TONA_VRAM_GB") == "14.9" && value("TONA_OEA_K0") == "7");
    EXPECT(value("ROCR_VISIBLE_DEVICES") == "0" && value("OMP_NUM_THREADS") == "<unset>");
    EXPECT(value("TONA_CLOCK_HOLD") == "<unset>" && value("TONA_ROLE_EFFORT") == "<unset>");  // off: not set
    in::UnitSpec bare = reference_spec();
    bare.drafter_flag.clear(), bare.api_key_file.clear(), bare.oea = false, bare.role_effort = "subagent=low";
    bare.port = 9000, bare.clock_hold = true, bare.host = "0.0.0.0";
    const in::ServerLaunch b = in::server_launch(bare, false);
    EXPECT(std::find(b.argv.begin(), b.argv.end(), "--mtp") == b.argv.end());
    EXPECT(std::find(b.argv.begin(), b.argv.end(), "--api-key-file") == b.argv.end());
    EXPECT(std::find(b.argv.begin(), b.argv.end(), "9000") != b.argv.end());
    EXPECT(std::find(b.argv.begin(), b.argv.end(), "0.0.0.0") != b.argv.end());
    // macOS: the thread count of `just start`, no ROCm variable
    const in::ServerLaunch mac = in::server_launch(reference_spec(), true);
    bool omp = false, rocr = false;
    for (const auto& [n, v] : mac.env)
        omp = omp || (n == "OMP_NUM_THREADS" && v == "6"), rocr = rocr || n == "ROCR_VISIBLE_DEVICES";
    EXPECT(omp && !rocr);
}

// The agents and the context the user chose reach the server (via `tona start --foreground`) as TONA_SEQS and --ctx;
// "auto" is written to the conf with its result
bool launches_with(const in::Settings& s, const std::string& seqs, const std::string& ctx) {
    const in::ServerLaunch l =
        in::server_launch(in::unit_spec(s, in::release_target("/home/u/.local/share/tona")), false);
    bool env = false, arg = false;
    for (const auto& [n, v] : l.env) env = env || (n == "TONA_SEQS" && v == seqs);
    for (size_t i = 0; i + 1 < l.argv.size(); i++) arg = arg || (l.argv[i] == "--ctx" && l.argv[i + 1] == ctx);
    return env && arg;
}

void test_unit_agents_and_context() {
    in::Settings s;
    s.seqs = 1, s.ctx = 131072, s.ctx_auto = false;
    EXPECT(launches_with(s, "1", "131072"));
    s.seqs = 4, s.ctx = 65536, s.ctx_auto = true;
    EXPECT(launches_with(s, "4", "65536"));
    EXPECT(in::render_release_conf(s).contains("TONA_CTX_AUTO=1\nTONA_SEQS=4"));
    s.ctx_auto = false;
    EXPECT(in::render_release_conf(s).contains("TONA_CTX_AUTO=0"));
}

void test_unit_quoting() {
    EXPECT(in::unit_quote("/plain/path") == "/plain/path");
    EXPECT(in::unit_quote("/with space/x") == "\"/with space/x\"");
    EXPECT(in::unit_quote("100%") == "\"100%%\"");
    EXPECT(in::unit_quote("$HOME") == "\"$$HOME\"");
    EXPECT(in::unit_quote("a\"b") == "\"a\\\"b\"");
    const std::string t = in::render_unit("/home/u/my dir");
    EXPECT(has_line(t, "ExecStart=\"/home/u/my dir/current/bin/tona\" start --foreground"));
    EXPECT(has_line(t, "Environment=\"TONA_INSTALL_DIR=/home/u/my dir\""));
}

void test_unit_file_and_key(const fs::path& dir) {
    const fs::path units = dir / "config/systemd/user";
    EXPECT(in::write_unit_file(reference_spec().root, units.string()).empty());
    EXPECT(slurp(units / "tona-server.service") == in::render_unit(reference_spec().root));
    EXPECT(!fs::exists(units / "tona-server.service.new"));
    const fs::path key = dir / "config/tona/api-key";
    EXPECT(in::ensure_api_key(key.string()).empty());
    const std::string first = slurp(key);
    EXPECT(first.size() == 64 && mode_of(key) == 0600);
    EXPECT(in::ensure_api_key(key.string()).empty() && slurp(key) == first);  // an existing key is kept
}

void test_embedded_catalog() {
    const in::CatalogEntry* m = in::find_entry("coder-iq1_m");
    EXPECT(m && m->kind == "model" && m->files.size() == 2);
    EXPECT(m && m->files[0].sha256.size() == 64 && m->files[0].size > 1000000000ULL);
    EXPECT(m && in::model_url({}, *m, m->files[0]).starts_with("https://huggingface.co/") &&
           in::model_url({}, *m, m->files[0]).contains("/resolve/" + m->revision + "/"));
    EXPECT(in::find_entry("mtp-q8_0") && in::find_entry("dflash-q8_0") && !in::find_entry("nothing"));
    EXPECT(in::drafter_entry_id("mtp", "coder-iq1_m") == "mtp-q8_0" &&
           in::drafter_entry_id("none", "coder-iq1_m").empty());
    const in::ProfileBlob* p = in::find_profile("coder-iq1_m");
    EXPECT(p && p->bytes.size() > 1000 && p->bytes[0] == 'I');  // header "IEXP" (engine/gpu/mega_init.cpp)
}

in::CatalogEntry small_entry(const std::string& body) {
    tona::update::Sha256 h;
    h.update(body.data(), body.size());
    const auto d = h.finish();
    in::CatalogEntry e;
    e.id = "tiny", e.kind = "model", e.repo = "org/tiny", e.revision = std::string(40, 'a');
    e.files = {{"sub/tiny.gguf", body.size(), tona::update::to_hex(d.data(), d.size()), "", "", ""}};
    return e;
}

// the revision of small_entry: a full commit hash, so the entry counts as pinned
std::string tiny_url() { return "/org/tiny/resolve/" + std::string(40, 'a') + "/sub/tiny.gguf"; }

in::ModelSource loopback_source(const TestServer& s) {
    in::ModelSource src;
    src.base = "http://127.0.0.1:" + std::to_string(s.port());
    src.fetch.allow_http_loopback = true;
    src.fetch.connect_seconds = 3;
    src.fetch.read_seconds = 5;
    src.attempts = 2;
    return src;
}

void test_model_download(const fs::path& dir) {
    std::string body(300000, 'x');
    for (size_t i = 0; i < body.size(); i++) body[i] = static_cast<char>(i * 7 % 251);
    const in::CatalogEntry e = small_entry(body);
    TestServer srv;
    srv.set(tiny_url(), Route(body));
    const std::string models = (dir / "models").string();
    EXPECT(!in::model_present(models, e));
    // A partial file from an interrupted run is continued, not restarted
    fs::create_directories(dir / "models/tiny");
    std::ofstream(dir / "models/tiny/tiny.gguf.part", std::ios::binary) << body.substr(0, 100000);
    std::vector<std::string> lines;
    const auto out = [&](const std::string& l) { lines.push_back(l); };
    const std::string err = in::fetch_model(e, models, loopback_source(srv), "", out);
    EXPECT(err.empty());
    if (!err.empty()) std::fprintf(stderr, "%s\n", err.c_str());
    EXPECT(slurp(dir / "models/tiny/tiny.gguf") == body);
    EXPECT(!fs::exists(dir / "models/tiny/tiny.gguf.part") && in::model_present(models, e));
    EXPECT(srv.last_headers()["Range"] == "bytes=100000-");
    // A present, marked file is not fetched again
    const int hits = srv.hits(tiny_url());
    EXPECT(in::fetch_model(e, models, loopback_source(srv), "", out).empty());
    EXPECT(srv.hits(tiny_url()) == hits);
}

void test_model_rejects_wrong_bytes(const fs::path& dir) {
    std::string body(5000, 'a');
    in::CatalogEntry e = small_entry(body);
    body[100] = 'b';  // the server delivers other bytes than the signed catalog promises
    TestServer srv;
    srv.set(tiny_url(), Route(body));
    const std::string models = (dir / "bad").string();
    EXPECT(!in::fetch_model(e, models, loopback_source(srv), "", [](const std::string&) {}).empty());
    EXPECT(!fs::exists(fs::path(models) / "tiny/tiny.gguf") && !fs::exists(fs::path(models) / "tiny/tiny.gguf.part"));
    EXPECT(!in::model_present(models, e));
}

void test_model_link(const fs::path& dir) {
    const std::string body(4096, 'z');
    const in::CatalogEntry e = small_entry(body);
    fs::create_directories(dir / "existing");
    std::ofstream(dir / "existing/tiny.gguf", std::ios::binary) << body;
    TestServer srv;  // knows nothing: a download would fail
    const std::string models = (dir / "linked").string();
    EXPECT(in::fetch_model(e, models, loopback_source(srv), (dir / "existing").string(), [](const std::string&) {
           }).empty());
    EXPECT(fs::is_symlink(fs::path(models) / "tiny/tiny.gguf") && in::model_present(models, e));
    // A file with other content is refused
    std::ofstream(dir / "existing/tiny.gguf", std::ios::binary) << std::string(4096, 'y');
    EXPECT(!in::fetch_model(e, (dir / "linked2").string(), loopback_source(srv), (dir / "existing").string(),
                            [](const std::string&) {})
                .empty());
}

// Runs the first step of a release plan: signature, hashes, layout
void test_first_install(const fs::path& dir) {
    TestServer srv;
    const FakeRelease r = make_release("0.3.0", 3);
    srv.serve_latest(r);
    in::ReleaseSources src;
    src.update = make_env(srv, dir);
    in::Target t = in::release_target(src.update->layout.root.string());
    t.unit_dir = (dir / "units").string(), t.bin_dir = (dir / "bin").string();
    in::Hardware hw;
    in::Settings s;
    const std::vector<in::Step> steps = in::plan_release_steps(hw, s, t, src);
    EXPECT(!steps.empty() && !steps[0].done);
    std::vector<std::string> lines;
    EXPECT(steps[0].action([&](const std::string& l) { lines.push_back(l); }) == 0);
    EXPECT(tona::update::linked_version(src.update->layout, "current") == "0.3.0");
    EXPECT(fs::exists(t.root + "/current/bin/tona-server"));
    EXPECT(in::plan_release_steps(hw, s, t, src)[0].done);  // a second run skips it

    // The command links step
    const auto at =
        std::find_if(steps.begin(), steps.end(), [](const in::Step& x) { return x.title.starts_with("Command "); });
    EXPECT(at != steps.end());
    const in::Step& links = *at;
    links.action([](const std::string&) {});
    EXPECT(fs::read_symlink(dir / "bin/tona") == t.root + "/current/bin/tona");  // the one command
    EXPECT(!fs::exists(dir / "bin/tona-install"));
}

// A release whose signature is wrong installs nothing
void test_unsigned_release_installs_nothing(const fs::path& dir) {
    TestServer srv;
    FakeRelease r = make_release("0.3.0", 3);
    r.manifest_text += ' ';  // the signature no longer fits
    srv.serve_latest(r);
    in::ReleaseSources src;
    src.update = make_env(srv, dir / "unsigned");
    const in::Target t = in::release_target(src.update->layout.root.string());
    std::vector<std::string> lines;
    const std::vector<in::Step> steps = in::plan_release_steps({}, {}, t, src);
    EXPECT(steps[0].action([&](const std::string& l) { lines.push_back(l); }) != 0);
    EXPECT(!fs::exists(t.root + "/current") && !fs::exists(t.root + "/versions"));
    EXPECT(!lines.empty() && lines.back().find("Nothing installed") == 0);
}

void test_command_links_and_path_hint(const fs::path& dir) {
    const std::string bin = (dir / "cmd/bin").string();
    fs::create_directories(bin);
    std::ofstream(bin + "/other") << "mine";  // a real file is never replaced
    std::vector<std::string> lines;
    in::link_commands(bin, {{"tona", "/x/current/bin/tona"}, {"other", "/x/current/bin/other"}},
                      [&](const std::string& l) { lines.push_back(l); });
    EXPECT(fs::is_symlink(bin + "/tona") && fs::read_symlink(bin + "/tona") == "/x/current/bin/tona");
    EXPECT(!fs::is_symlink(bin + "/other") && slurp(bin + "/other") == "mine" && lines.size() == 2);
    EXPECT(in::path_hint(bin, "/usr/bin:" + bin + ":/bin").empty());
    const std::vector<std::string> hint = in::path_hint(bin, "/usr/bin:/bin");
    EXPECT(hint.size() == 4 && hint[1].contains("bash") && hint[2].contains("zsh") &&
           hint[3].contains("fish_add_path"));
    // below $HOME the lines name $HOME, not the user's path
    const char* home_env = std::getenv("HOME");  // set for the test (ctest); never null in practice
    const std::string home = home_env ? home_env : "";
    const std::string home_bin = home + "/.local/bin";
    EXPECT(in::path_hint(home_bin, "/usr/bin")[1].contains("$HOME/.local/bin:$PATH"));
    in::Settings s;
    s.port = 9000, s.effort = "high";
    const std::string conf = in::render_release_conf(s);
    EXPECT(conf.contains("TONA_PORT=9000\n") && conf.contains("TONA_EFFORT=high\n"));
    EXPECT(conf.contains("TONA_API_KEY_FILE=" + home + "/.config/tona/api-key"));
}

void test_conf_round_trip() {
    in::Settings s;
    // All of Settings is in the file: `tona start` starts the server from it
    s.model = "ornith-1.5-35b-a3b", s.drafter = "mtp", s.ram_gb = 48, s.vram_gb = 14.5, s.seqs = 3, s.ctx = 131072;
    s.clock_hold = true, s.oea = false, s.role_effort = "aux=low";
    const std::string full = in::render_release_conf(s);
    const in::Values v = in::parse_values(full);
    EXPECT(v.at("TONA_MODEL_ID") == "ornith-1.5-35b-a3b" && v.at("TONA_DRAFTER") == "mtp" &&
           v.at("TONA_RAM_GB") == "48");
    EXPECT(v.at("TONA_VRAM_GB") == "14.5" && v.at("TONA_SEQS") == "3" && v.at("TONA_CTX") == "131072");
    EXPECT(v.at("TONA_CLOCK_HOLD") == "1" && v.at("TONA_OEA_K0") == "0" && v.at("TONA_ROLE_EFFORT") == "aux=low");
    TempDir scratch("conf-roundtrip");
    const std::string file = (scratch.path() / "tona.conf").string();
    std::ofstream(file) << full;
    const in::Settings back = in::load(file, in::Settings{});
    EXPECT(back.model == s.model && back.drafter == "mtp" && back.ram_gb == 48 && back.seqs == 3 && back.ctx == 131072);
    EXPECT(back.clock_hold && !back.oea && back.role_effort == "aux=low");
}

void test_plan_order_and_paths(const fs::path& dir) {
    in::ReleaseSources src;
    in::Target t = in::release_target((dir / "root").string());
    t.unit_dir = (dir / "u").string();
    t.macos = false;  // the Linux plan with its systemd unit, also on a Mac host; test_macos_plan has the other
    in::Settings s;
    const std::vector<in::Step> st = in::plan_release_steps({}, s, t, src);
    EXPECT(st.size() >= 6);
    EXPECT(st[0].title.contains("signed release"));
    EXPECT(st[1].title.contains("coder-iq1_m") && st[2].title.contains("mtp-q8_0"));
    const in::UnitSpec u = in::unit_spec(s, t);
    EXPECT(u.model_file.starts_with(t.root + "/models/coder-iq1_m/") && u.model_file.ends_with("00001-of-00002.gguf"));
    EXPECT(u.drafter_flag == "--mtp" && u.profile == t.root + "/profiles/coder-iq1_m.bin");
    // The unit is always written; enabling it (autostart) is the user's choice and off by default
    const auto unit_step = [&](const in::Settings& cfg) {
        for (const in::Step& step : in::plan_release_steps({}, cfg, t, src))
            if (step.title.contains("service")) return step;
        return in::Step{};
    };
    EXPECT(!s.service && unit_step(s).title.contains("no autostart"));
    EXPECT(!unit_step(s).cmd.contains("enable"));
    in::Settings on = s;
    on.service = true;
    EXPECT(unit_step(on).cmd.contains("enable --now"));
    s.drafter = "none";
    EXPECT(in::unit_spec(s, t).drafter_flag.empty() && in::plan_release_steps({}, s, t, src).size() == st.size() - 1);
}

// The tick on the Finish screen: state.json gets the answer, the highest accepted seq stays
void test_update_checks_step(const fs::path& dir) {
    in::ReleaseSources src;
    in::Target t = in::release_target((dir / "updates-root").string());
    const tona::update::Layout layout{t.root};
    const auto run_step = [&](bool on) {
        in::Settings s;
        s.update_checks = on;
        for (const in::Step& step : in::finish_steps(s, t, in::PathEnv{}))
            if (step.title.starts_with("Update checks")) {
                EXPECT(step.title.contains(on ? "on, once a day" : "off"));
                return step.action([](const std::string&) {});
            }
        return 99;
    };
    tona::update::State seeded;
    seeded.seq = 7;
    tona::update::ensure_private_root(layout);
    EXPECT(tona::update::save_state(layout, seeded).empty());
    EXPECT(run_step(true) == 0);
    tona::update::StateResult st = tona::update::load_state(layout);
    EXPECT(st.ok && st.state.checks_enabled && st.state.seq == 7);
    EXPECT(run_step(false) == 0);
    st = tona::update::load_state(layout);
    EXPECT(st.ok && !st.state.checks_enabled && st.state.seq == 7);
}

// macOS: a LaunchAgent step instead of the systemd unit, nothing that calls systemctl and no start script
void test_macos_plan(const fs::path& dir) {
    in::ReleaseSources src;
    in::Target t = in::release_target((dir / "mac-root").string());
    t.macos = true;
    const std::vector<in::Step> steps = in::plan_release_steps({}, in::Settings{}, t, src);
    bool login = false, links = false;
    for (const in::Step& step : steps) {
        EXPECT(!step.cmd.contains("systemctl") && !step.title.contains("start script"));
        login = login || step.title.contains("autostart");
        links = links || step.title.starts_with("Command ");  // the PATH hint is on the last page
    }
    EXPECT(login && links);
    for (const in::Step& step : steps)
        if (step.title.contains("autostart")) EXPECT(step.action([](const std::string&) {}) == 0);
    EXPECT(!fs::exists(fs::path(t.root) / "start-server.sh"));
}

}  // namespace

int main() {
    return run("release", [] {
        TempDir dir("release");
        ::setenv("HOME", (dir.path() / "home").c_str(), 1);
        test_unit_text();
        test_server_launch();
        test_unit_quoting();
        test_unit_agents_and_context();
        test_unit_file_and_key(dir.path());
        test_embedded_catalog();
        test_model_download(dir.path());
        test_model_rejects_wrong_bytes(dir.path());
        test_model_link(dir.path());
        test_first_install(dir.path());
        test_unsigned_release_installs_nothing(dir.path());
        test_command_links_and_path_hint(dir.path());
        test_conf_round_trip();
        test_plan_order_and_paths(dir.path());
        test_macos_plan(dir.path());
        test_update_checks_step(dir.path());
    });
}
