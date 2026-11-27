#include "update/install.h"

#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <optional>

#include "update/activate.h"
#include "update/crypto.h"
#include "update/fsutil.h"
#include "update/state.h"

namespace fs = std::filesystem;

namespace tona::update {

namespace {

// Holds an exclusive lock on the install root for the whole run: two tona instances never install at once
class RootLock {
public:
    explicit RootLock(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd_ < 0 || ::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            if (fd_ >= 0) ::close(fd_);
            throw FsError("another update is running");
        }
    }
    ~RootLock() { ::close(fd_); }
    RootLock(const RootLock&) = delete;
    RootLock& operator=(const RootLock&) = delete;

private:
    int fd_ = -1;
};

void clear_staging(const Layout& layout) {
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(layout.staging(), ec)) fs::remove_all(e.path(), ec);
}

// Removes a staging directory when the run ends, whether it succeeded or not
struct StageCleanup {
    fs::path path;
    ~StageCleanup() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

class Installer {
public:
    Installer(const UpdateEnv& env, const Release& rel, const InstallHooks& hooks)
        : env_(env), rel_(rel), m_(rel.manifest), hooks_(hooks) {}

    InstallOutcome run() {
        InstallOutcome out;
        refuse_root();
        ensure_private_root(env_.layout);
        RootLock lock((env_.layout.lock()).string());
        const StateResult st = load_state(env_.layout);
        if (!st.ok) throw FsError(st.error);
        // Verify again under the lock, against the seq as it is now
        const VerifyOutcome v = verify_release(env_, rel_.manifest_text, rel_.signature_text, st.state.seq);
        if (!v.ok) throw FsError(v.error);
        if (fs::exists(env_.layout.version(m_.version))) {
            out.already_installed = true;
        } else {
            stage_and_commit();
        }
        activate_version(env_.layout, m_.version);
        State next = st.state;
        next.seq = std::max(next.seq, m_.seq);
        if (std::string err = save_state(env_.layout, next); !err.empty()) throw FsError(err);
        prune(env_.layout);
        out.ok = true;
        return out;
    }

private:
    static void refuse_root() {
        if (::geteuid() == 0) throw FsError("refusing to install as root");
    }

    // A platform without runtime files (macOS) has no shared runtime directory and no lib link
    bool runtime_present() const { return !m_.has_runtime() || fs::exists(env_.layout.runtime(m_.lock_id())); }

    uint64_t bytes_to_download() const {
        uint64_t n = 0;
        for (const ManifestFile& f : m_.files)
            if (f.area == "bin" || !runtime_present()) n += f.size;
        return n;
    }

    void download(const ManifestFile& f, const Dir& target) {
        FileWriter w(target, f.name);
        const FetchRequest req{env_.base.asset_url(m_.version, f.asset), f.size, f.size, 3600};
        const uint64_t before = done_;
        const FetchResult r = fetch(
            env_.fetch, req, [&](const char* d, size_t n) { return w.write(d, n); },
            [&](uint64_t got) {
                done_ = before + got;
                if (hooks_.progress) hooks_.progress(done_, total_);
            });
        if (!r.ok) throw FsError(f.asset + ": " + r.error);
        if (r.sha256 != f.sha256) throw FsError(f.asset + ": hash does not match the manifest");
        w.finish(f.mode);
        if (std::string err = verify_file(target, f.name, f.size, f.sha256); !err.empty()) throw FsError(err);
    }

    // The shared runtime exists already (same ROCm lock): check it instead of downloading it again
    void check_existing_runtime() const {
        if (!m_.has_runtime()) return;
        const Dir rt = Dir::open(env_.layout.runtime(m_.lock_id()).string());
        for (const ManifestFile& f : m_.files) {
            if (f.area != "runtime") continue;
            if (std::string err = verify_file(rt, f.name, f.size, f.sha256); !err.empty())
                throw FsError("installed runtime does not match this release: " + err);
        }
    }

    void write_text(const Dir& dir, const std::string& name, const std::string& text) const {
        FileWriter w(dir, name);
        if (!w.write(text.data(), text.size())) throw FsError("cannot write " + name);
        w.finish(0644);
    }

    void stage_and_commit() {
        total_ = bytes_to_download();
        uint8_t rnd[8];
        random_bytes(rnd, sizeof rnd);
        const std::string stage_name = to_hex(rnd, sizeof rnd);
        clear_staging(env_.layout);
        const Dir staging = Dir::open(env_.layout.staging().string());
        const Dir stage = staging.make_subdir(stage_name);
        const fs::path stage_path = env_.layout.staging() / stage_name;
        const StageCleanup cleanup{stage_path};
        const Dir vdir = stage.make_subdir("v");
        const Dir bin = vdir.make_subdir("bin");
        const bool has_runtime = m_.has_runtime();
        const bool need_runtime = has_runtime && !runtime_present();
        std::optional<Dir> rt;
        if (need_runtime) rt = stage.make_subdir("rt");
        for (const ManifestFile& f : m_.files) {
            if (f.area == "bin") {
                download(f, bin);
            } else if (need_runtime) {
                download(f, *rt);
            }
        }
        if (!need_runtime) check_existing_runtime();
        write_text(vdir, "manifest.json", rel_.manifest_text);
        write_text(vdir, "manifest.json.minisig", rel_.signature_text);
        const std::string link_target = "../../runtime/rocm-" + m_.lock_id();
        if (has_runtime && ::symlinkat(link_target.c_str(), vdir.fd(), "lib") != 0)
            throw FsError("cannot create lib link");
        if (need_runtime) rename_noreplace((stage_path / "rt").string(), env_.layout.runtime(m_.lock_id()).string());
        rename_noreplace((stage_path / "v").string(), env_.layout.version(m_.version).string());
    }

    const UpdateEnv& env_;
    const Release& rel_;
    const Manifest& m_;
    const InstallHooks& hooks_;
    uint64_t done_ = 0;
    uint64_t total_ = 0;
};

}  // namespace

InstallOutcome install_release(const UpdateEnv& env, const Release& release, const InstallHooks& hooks) {
    try {
        return Installer(env, release, hooks).run();
    } catch (const std::exception& e) {
        InstallOutcome out;
        out.error = e.what();
        return out;
    }
}

}  // namespace tona::update
