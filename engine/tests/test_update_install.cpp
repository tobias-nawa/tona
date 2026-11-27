// Installing a release: the resulting layout, and that every failure leaves the installed versions and the state alone.
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>

#include <filesystem>
#include <string>

#include "tests/test_support.h"
#include "tests/update_testkit.h"
#include "update/activate.h"
#include "update/fsutil.h"
#include "update/install.h"
#include "update/state.h"

using namespace tona::test;
namespace fs = std::filesystem;
namespace {

struct Setup {
    TempDir dir;
    TestServer server;
    up::UpdateEnv env;

    explicit Setup(const std::string& name) : dir(name), env(make_env(server, dir.path())) {}

    up::InstallOutcome install(const FakeRelease& r) {
        server.serve_latest(r);
        const auto check = up::check_for_update(env);
        if (check.kind != up::CheckOutcome::Kind::Available) {
            up::InstallOutcome failed;
            failed.error = "check: " + check.error;
            return failed;
        }
        return up::install_release(env, check.release, {});
    }

    fs::path root() const { return env.layout.root; }
    std::string current() const { return up::linked_version(env.layout, "current"); }
    std::string previous() const { return up::linked_version(env.layout, "previous"); }
    uint64_t seq() const { return up::load_state(env.layout).state.seq; }
    bool has_version(const std::string& v) const { return fs::exists(env.layout.version(v)); }
    size_t staging_entries() const {
        return fs::exists(env.layout.staging()) ? std::distance(fs::directory_iterator(env.layout.staging()), {}) : 0;
    }
};

mode_t mode_of(const fs::path& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 ? st.st_mode & 07777 : 0;
}

void test_fresh_install() {
    Setup s("install-fresh");
    const FakeRelease r = make_release("0.3.0", 3);
    const auto out = s.install(r);
    EXPECT(out.ok);
    if (!out.ok) std::fprintf(stderr, "%s\n", out.error.c_str());
    EXPECT(s.current() == "0.3.0");
    EXPECT(s.previous().empty());
    EXPECT(s.seq() == 3);
    const fs::path v = s.env.layout.version("0.3.0");
    EXPECT(read_file((v / "bin/tona-server").string()) == r.assets.at("tona-server"));
    EXPECT(mode_of(v / "bin/tona-server") == 0755);
    EXPECT(mode_of(s.root()) == 0700);
    EXPECT(read_file((v / "manifest.json").string()) == r.manifest_text);
    EXPECT(read_file((v / "manifest.json.minisig").string()) == r.signature);
    // The runtime is shared and reachable through lib
    const fs::path rt = s.env.layout.runtime("888888888888");
    EXPECT(mode_of(rt / "libhip.so.7") == 0644);
    EXPECT(read_file((v / "lib/libhip.so.7").string()) == r.assets.at("rocm-888888888888-libhip.so.7"));
    EXPECT(up::read_link((v / "lib").string()) == "../../runtime/rocm-888888888888");
    EXPECT(up::read_link(s.env.layout.current().string()) == "versions/0.3.0");
    EXPECT(s.staging_entries() == 0);
    EXPECT(up::load_state(s.env.layout).state.checks_enabled == false);
}

void test_second_release_reuses_runtime() {
    Setup s("install-second");
    EXPECT(s.install(make_release("0.3.0", 3)).ok);
    const std::string lib_asset = "/releases/download/v0.4.0/rocm-888888888888-libhip.so.7";
    const FakeRelease r2 = make_release("0.4.0", 4);
    const auto o2 = s.install(r2);
    if (!o2.ok) std::fprintf(stderr, "ERR: %s\n", o2.error.c_str());
    EXPECT(o2.ok);
    EXPECT(s.server.hits(lib_asset) == 0);  // same ROCm lock: not downloaded again
    EXPECT(s.current() == "0.4.0");
    EXPECT(s.previous() == "0.3.0");
    EXPECT(s.seq() == 4);
    // A new lock hash gets its own runtime directory, and the unused one is pruned later
    EXPECT(s.install(make_release("0.5.0", 5, std::string(64, '7'))).ok);
    EXPECT(fs::exists(s.env.layout.runtime("777777777777")));
    EXPECT(s.previous() == "0.4.0");
}

void test_changed_runtime_is_refused() {
    Setup s("install-runtime");
    EXPECT(s.install(make_release("0.3.0", 3)).ok);
    {
        std::ofstream(s.env.layout.runtime("888888888888") / "libhip.so.7", std::ios::app) << "tampered";
    }
    const auto out = s.install(make_release("0.4.0", 4));
    EXPECT(!out.ok);
    EXPECT(out.error.contains("installed runtime does not match"));
    EXPECT(s.current() == "0.3.0");
    EXPECT(!s.has_version("0.4.0"));
}

void test_prune_keeps_three() {
    Setup s("install-prune");
    for (int i = 3; i <= 7; i++) EXPECT(s.install(make_release("0." + std::to_string(i) + ".0", i)).ok);
    EXPECT(s.current() == "0.7.0");
    EXPECT(s.previous() == "0.6.0");
    EXPECT(s.has_version("0.5.0"));
    EXPECT(!s.has_version("0.4.0"));
    EXPECT(!s.has_version("0.3.0"));
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(s.env.layout.versions())) n += e.is_directory();
    EXPECT(n == 3);
    EXPECT(up::version_less("0.9.0", "0.10.0"));
    EXPECT(!up::version_less("0.10.0", "0.9.0"));
    EXPECT(!up::version_less("a", "0.10.0"));
}

// A failing install changes nothing visible
void expect_untouched(Setup& s, const up::InstallOutcome& out, const std::string& why) {
    EXPECT(!out.ok);
    EXPECT(out.error.contains(why));
    EXPECT(s.current() == "0.3.0");
    EXPECT(s.seq() == 3);
    EXPECT(!s.has_version("0.4.0"));
    EXPECT(!fs::exists(s.env.layout.runtime("444444444444")));
    EXPECT(s.staging_entries() == 0);
}

void test_failed_downloads_change_nothing() {
    Setup s("install-fail");
    EXPECT(s.install(make_release("0.3.0", 3)).ok);
    const std::string lock = std::string(64, '4');
    const auto path = [](const std::string& asset) { return "/releases/download/v0.4.0/" + asset; };
    const std::string server_asset = "tona-server", lib_asset = "rocm-444444444444-libhip.so.7";

    FakeRelease wrong_hash = make_release("0.4.0", 4, lock);
    std::string flipped = wrong_hash.assets.at(server_asset);
    flipped[0] ^= 1;
    s.server.serve_latest(wrong_hash);
    s.server.set(path(server_asset), Route(flipped));
    auto check = up::check_for_update(s.env);
    expect_untouched(s, up::install_release(s.env, check.release, {}), "hash");

    FakeRelease too_big = make_release("0.4.0", 4, lock);
    s.server.serve_latest(too_big);
    s.server.set(path(lib_asset), Route(too_big.assets.at(lib_asset) + "extra"));
    check = up::check_for_update(s.env);
    expect_untouched(s, up::install_release(s.env, check.release, {}), lib_asset);

    FakeRelease cut = make_release("0.4.0", 4, lock);
    s.server.serve_latest(cut);
    Route half(cut.assets.at(server_asset));
    half.cut_after = 20;
    s.server.set(path(server_asset), half);
    check = up::check_for_update(s.env);
    expect_untouched(s, up::install_release(s.env, check.release, {}), server_asset);

    FakeRelease missing = make_release("0.4.0", 4, lock);
    s.server.serve_latest(missing);
    s.server.set(path(server_asset), [] {
        Route r;
        r.status = 404;
        return r;
    }());
    check = up::check_for_update(s.env);
    expect_untouched(s, up::install_release(s.env, check.release, {}), "404");
}

void test_check_rejections() {
    Setup s("install-check");
    EXPECT(s.install(make_release("0.3.0", 3)).ok);
    const auto kind = [&](const FakeRelease& r) {
        s.server.serve_latest(r);
        return up::check_for_update(s.env);
    };
    EXPECT(kind(make_release("0.3.0", 3)).kind == up::CheckOutcome::Kind::UpToDate);
    const auto old = kind(make_release("0.2.0", 2));
    EXPECT(old.kind == up::CheckOutcome::Kind::Failed);
    EXPECT(old.error.contains("older"));
    FakeRelease signed_by_other = make_release("0.4.0", 4);
    signed_by_other.signature = other_key().sign_file(signed_by_other.manifest_text, comment_for(4, "0.4.0"));
    EXPECT(kind(signed_by_other).error.contains("signature check failed"));
    FakeRelease tampered = make_release("0.4.0", 4);
    tampered.manifest_text[tampered.manifest_text.find("Test")] = 'X';
    EXPECT(kind(tampered).kind == up::CheckOutcome::Kind::Failed);
    FakeRelease expired = make_release("0.4.0", 4);
    expired.manifest["expires"] = "2026-11-01T00:00:00Z";
    expired.sign(main_key());
    EXPECT(kind(expired).error.contains("expired"));
    FakeRelease other_platform = make_release("0.4.0", 4);
    s.env.platform = "linux-x86_64-gfx1100";  // a platform the manifest has no files for
    EXPECT(kind(other_platform).error.contains("bin/tona-server is missing"));
    s.env.platform = std::string(up::kPlatform);
    FakeRelease bad_comment = make_release("0.4.0", 4);
    bad_comment.signature = main_key().sign_file(bad_comment.manifest_text, comment_for(5, "0.4.0"));
    EXPECT(kind(bad_comment).error.contains("trusted comment"));
    s.env.keys.clear();  // a build without trusted keys accepts nothing
    EXPECT(kind(make_release("0.4.0", 4)).kind == up::CheckOutcome::Kind::Failed);
    EXPECT(s.current() == "0.3.0");
}

// The Mac release has no runtime: no runtime directory, no lib link, and the files of the other platform stay unfetched
void test_release_without_runtime() {
    Setup s("install-no-runtime");
    const auto out = s.install(make_release_without_runtime("0.3.0", 3));
    EXPECT(out.ok);
    if (!out.ok) std::fprintf(stderr, "%s\n", out.error.c_str());
    EXPECT(s.current() == "0.3.0");
    const fs::path v = s.env.layout.version("0.3.0");
    EXPECT(fs::exists(v / "bin" / "tona-server"));
    EXPECT(!fs::exists(v / "bin" / "other-platform-tona-server"));
    EXPECT(!fs::is_symlink(v / "lib"));
    EXPECT(!fs::exists(s.env.layout.runtime("888888888888")));
    EXPECT(s.install(make_release_without_runtime("0.4.0", 4)).ok);
    EXPECT(s.current() == "0.4.0");
}

void test_install_verifies_again() {
    Setup s("install-reverify");
    const FakeRelease r = make_release("0.3.0", 3);
    s.server.serve_latest(r);
    auto check = up::check_for_update(s.env);
    check.release.manifest_text[check.release.manifest_text.find("Test")] = 'X';
    const auto out = up::install_release(s.env, check.release, {});
    EXPECT(!out.ok);
    EXPECT(out.error.contains("signature"));
    EXPECT(!fs::exists(s.env.layout.versions() / "0.3.0"));
}

void test_root_permissions_and_state() {
    Setup s("install-root");
    fs::create_directories(s.root());
    ::chmod(s.root().c_str(), 0755);
    auto out = s.install(make_release("0.3.0", 3));
    EXPECT(!out.ok);
    EXPECT(out.error.contains("chmod 700 " + s.root().string()));  // the message names the fix
    ::chmod(s.root().c_str(), 0700);
    // A damaged state file must not reset the rollback counter
    EXPECT(up::save_state(s.env.layout, {5, true}).empty());
    EXPECT(!s.install(make_release("0.3.0", 3)).ok);  // seq 3 < 5
    {
        std::ofstream(s.env.layout.state()) << R"({"seq": 0, "checks": false, "x": 1})";
    }
    EXPECT(!up::load_state(s.env.layout).ok);
    EXPECT(!s.install(make_release("0.9.0", 9)).ok);
    fs::remove(s.env.layout.state());
    EXPECT(s.install(make_release("0.9.0", 9)).ok);
    const auto st = up::load_state(s.env.layout);
    EXPECT(st.ok && st.state.seq == 9);
}

void test_already_installed_only_switches() {
    Setup s("install-again");
    EXPECT(s.install(make_release("0.3.0", 3)).ok);
    EXPECT(s.install(make_release("0.4.0", 4)).ok);
    EXPECT(up::swap_current_previous(s.env.layout));
    EXPECT(s.current() == "0.3.0");
    const FakeRelease again = make_release("0.4.0", 4);
    s.server.serve_latest(again);
    const int hits = s.server.hits("/releases/download/v0.4.0/tona-server");
    const auto check = up::check_for_update(s.env);
    EXPECT(check.kind == up::CheckOutcome::Kind::UpToDate);  // seq equals the state
    const auto out = up::install_release(s.env, check.release, {});
    EXPECT(out.ok && out.already_installed);
    EXPECT(s.current() == "0.4.0");
    EXPECT(s.server.hits("/releases/download/v0.4.0/tona-server") == hits);
}

void test_lock_and_progress() {
    Setup s("install-lock");
    const FakeRelease r = make_release("0.3.0", 3);
    s.server.serve_latest(r);
    const auto check = up::check_for_update(s.env);
    uint64_t last = 0, total = 0;
    up::InstallHooks hooks;
    hooks.progress = [&](uint64_t done, uint64_t all) {
        last = done;
        total = all;
    };
    {
        fs::create_directories(s.root());
        ::chmod(s.root().c_str(), 0700);
        const int fd = ::open((s.root() / ".lock").c_str(), O_RDWR | O_CREAT, 0600);
        EXPECT(::flock(fd, LOCK_EX | LOCK_NB) == 0);
        const auto busy = up::install_release(s.env, check.release, hooks);
        EXPECT(!busy.ok);
        EXPECT(busy.error.contains("another update"));
        ::close(fd);
    }
    EXPECT(up::install_release(s.env, check.release, hooks).ok);
    EXPECT(total == check.release.manifest.total_bytes());
    EXPECT(last == total);
    std::atomic<bool> stop{true};
    up::InstallHooks cancelled;
    cancelled.cancel = &stop;
    s.env.fetch.cancel = &stop;
    const FakeRelease next = make_release("0.4.0", 4);
    s.server.serve_latest(next);
    s.env.fetch.cancel = nullptr;
    const auto c2 = up::check_for_update(s.env);
    s.env.fetch.cancel = &stop;
    EXPECT(!up::install_release(s.env, c2.release, cancelled).ok);
    EXPECT(s.current() == "0.3.0");
}

void test_fs_primitives() {
    TempDir dir("install-fs");
    const up::Dir d = up::Dir::open(dir.path().string());
    // A symlink where a file should be created is refused, and nothing is written through it
    fs::create_symlink(dir.path() / "target", dir.path() / "link");
    EXPECT(throws([&] { up::FileWriter w(d, "link"); }));
    EXPECT(!fs::exists(dir.path() / "target"));
    {
        up::FileWriter w(d, "f");
        EXPECT(w.write("abc", 3));
        w.finish(0640);
    }
    EXPECT(throws([&] { up::FileWriter w(d, "f"); }));  // exists: O_EXCL
    EXPECT(mode_of(dir.path() / "f") == 0640);
    up::Sha256 h;
    h.update("abc", 3);
    const auto digest = h.finish();
    const std::string hex = up::to_hex(digest.data(), digest.size());
    EXPECT(up::verify_file(d, "f", 3, hex).empty());
    EXPECT(!up::verify_file(d, "f", 4, hex).empty());
    EXPECT(!up::verify_file(d, "f", 3, std::string(64, '0')).empty());
    EXPECT(!up::verify_file(d, "link", 0, hex).empty());
    EXPECT(!up::verify_file(d, "nothing", 3, hex).empty());
    fs::create_directory(dir.path() / "sub");
    EXPECT(!up::verify_file(d, "sub", 3, hex).empty());
    EXPECT(throws([&] { up::Dir::open((dir.path() / "link").string()); }));
    EXPECT(throws([&] { d.make_subdir("sub"); }));
    up::set_symlink_atomic("x", (dir.path() / "l2").string());
    up::set_symlink_atomic("y", (dir.path() / "l2").string());
    EXPECT(up::read_link((dir.path() / "l2").string()) == "y");
    EXPECT(up::read_link((dir.path() / "none").string()).empty());
    EXPECT(throws([&] { up::rename_noreplace((dir.path() / "f").string(), (dir.path() / "sub").string()); }));
}

}  // namespace

int main() {
    return run("update_install", [] {
        test_fresh_install();
        test_second_release_reuses_runtime();
        test_changed_runtime_is_refused();
        test_prune_keeps_three();
        test_failed_downloads_change_nothing();
        test_check_rejections();
        test_release_without_runtime();
        test_install_verifies_again();
        test_root_permissions_and_state();
        test_already_installed_only_switches();
        test_lock_and_progress();
        test_fs_primitives();
    });
}
