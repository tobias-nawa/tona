#include "server/checkpoint_file.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <format>
#include <fstream>

namespace tona::server {

namespace {

// A new file that is ours alone: any old file or link under the name is removed first, and the open refuses to
// create it through a link (O_EXCL, O_NOFOLLOW on POSIX; CREATE_NEW on Windows), so a link planted in a writable
// directory cannot make us truncate or overwrite another file. Written through the handle of that one open.
class NewFile {
public:
    explicit NewFile(const std::string& path) {
#ifdef _WIN32
        DeleteFileA(path.c_str());
        h_ = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        ok_ = h_ != INVALID_HANDLE_VALUE;
#else
        ::unlink(path.c_str());
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        ok_ = fd_ >= 0;
#endif
    }
    ~NewFile() {
#ifdef _WIN32
        if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
#else
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    NewFile(const NewFile&) = delete;
    NewFile& operator=(const NewFile&) = delete;

    void put(const void* p, size_t bytes) {
        const char* c = static_cast<const char*>(p);
        while (ok_ && bytes > 0) {
#ifdef _WIN32
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(bytes, 0x7fffffffu));
            DWORD n = 0;
            if (!WriteFile(h_, c, chunk, &n, nullptr) || n == 0)
                ok_ = false;
            else
                c += n, bytes -= n;
#else
            const ssize_t n = ::write(fd_, c, bytes);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0)
                ok_ = false;
            else
                c += n, bytes -= (size_t)n;
#endif
        }
    }
    bool ok() const { return ok_; }

private:
#ifdef _WIN32
    HANDLE h_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    bool ok_ = false;
};

bool head_matches(const DiskHead& hd, uint64_t fingerprint) {
    return hd.magic == DISK_MAGIC && hd.version == DISK_VERSION && hd.fingerprint == fingerprint;
}

}  // namespace

uint64_t fnv(uint64_t h, const void* p, size_t n) {
    Hasher x(h);
    x.bytes(p, n);
    return x.digest();
}

std::string checkpoint_path(const std::string& dir, uint64_t fingerprint, const std::vector<int32_t>& tokens) {
    const uint64_t th = fnv(FNV_BASIS, tokens.data(), tokens.size() * sizeof(int32_t));
    return std::format("{}/{:016x}-{:016x}.bin", dir, fingerprint, th);
}

bool write_checkpoint(const std::string& path, uint64_t fingerprint, const std::vector<float>& logits,
                      const backend::HostSnap& h) {
    const DiskHead hd{DISK_MAGIC, DISK_VERSION,  fingerprint,    h.tokens.size(), h.pos,         h.commit,
                      h.parity,   logits.size(), h.state.size(), h.kv.size(),     h.pool.size(), h.dring.size()};
    // 0600: the content is a function of the prompt text
    NewFile out(path);
    out.put(&hd, sizeof hd);
    out.put(h.tokens.data(), h.tokens.size() * sizeof(int32_t));
    out.put(logits.data(), logits.size() * sizeof(float));
    for (const auto* v : {&h.state, &h.kv, &h.pool, &h.dring}) out.put(v->data(), v->size());
    return out.ok();
}

bool secure_checkpoint_dir(const std::string& dir, bool created, std::string& why) {
    namespace fs = std::filesystem;
#ifdef _WIN32
    (void)created;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return why = "cannot read the directory", false;
    return true;
#else
    struct stat st{};
    if (::stat(dir.c_str(), &st) != 0) return why = "cannot read the directory", false;
    std::error_code ec;
    bool only_ours = true;
    for (const auto& f : fs::directory_iterator(dir, ec))
        only_ours = only_ours && (f.path().extension() == ".bin" || f.path().extension() == ".tmp");
    if (st.st_uid == ::geteuid() && (created || only_ours)) {
        fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);
        if (::stat(dir.c_str(), &st) != 0) return why = "cannot read the directory", false;
    }
    if (st.st_mode & (S_IWGRP | S_IWOTH))
        return why = "writable by others and cannot be restricted to the owner (foreign owner or foreign files in it)",
               false;
    return true;
#endif
}

bool read_checkpoint_tokens(const std::string& path, uint64_t fingerprint, std::vector<int32_t>& tokens,
                            bool& has_logits) {
    std::ifstream in(path, std::ios::binary);
    DiskHead hd{};
    if (!in.read(reinterpret_cast<char*>(&hd), sizeof hd) || !head_matches(hd, fingerprint)) return false;
    // A damaged head must not make us allocate for tokens the file cannot hold
    std::error_code ec;
    if (hd.ntok > (std::filesystem::file_size(path, ec) - sizeof hd) / sizeof(int32_t) || ec) return false;
    tokens.assign(hd.ntok, 0);
    has_logits = hd.n_logits != 0;
    return (bool)in.read(reinterpret_cast<char*>(tokens.data()), hd.ntok * sizeof(int32_t));
}

bool read_checkpoint(const std::string& path, const CheckpointExpect& want, std::vector<float>& logits,
                     backend::HostSnap& h) {
    std::ifstream in(path, std::ios::binary);
    DiskHead hd{};
    if (!in.read(reinterpret_cast<char*>(&hd), sizeof hd) || !head_matches(hd, want.fingerprint) ||
        hd.ntok != want.tokens.size() || (hd.n_logits != want.vocab && hd.n_logits != 0) ||
        hd.n_state != want.state_bytes)
        return false;
    std::error_code ec;
    const uint64_t size = sizeof hd + hd.ntok * sizeof(int32_t) + hd.n_logits * sizeof(float) + hd.n_state + hd.n_kv +
                          hd.n_pool + hd.n_dring;
    if (std::filesystem::file_size(path, ec) != size || ec) return false;
    h.tokens.resize(hd.ntok), logits.resize(hd.n_logits);
    h.state.resize(hd.n_state), h.kv.resize(hd.n_kv), h.pool.resize(hd.n_pool), h.dring.resize(hd.n_dring);
    h.pos = hd.pos, h.commit = hd.commit, h.parity = hd.parity;
    in.read(reinterpret_cast<char*>(h.tokens.data()), hd.ntok * sizeof(int32_t));
    in.read(reinterpret_cast<char*>(logits.data()), hd.n_logits * sizeof(float));
    for (auto* v : {&h.state, &h.kv, &h.pool, &h.dring}) in.read(reinterpret_cast<char*>(v->data()), v->size());
    if (in && h.tokens == want.tokens) return true;
    h = backend::HostSnap{};  // the vectors are not zero-filled: a short read must not leave stale bytes behind
    return false;
}

}  // namespace tona::server
