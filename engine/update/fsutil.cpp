#include "update/fsutil.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "update/crypto.h"

namespace fs = std::filesystem;

namespace tona::update {

namespace {

[[noreturn]] void fail(const std::string& what) { throw FsError(what + ": " + std::strerror(errno)); }

}  // namespace

Dir::~Dir() {
    if (fd_ >= 0) ::close(fd_);
}

int Dir::release() {
    const int fd = fd_;
    fd_ = -1;
    return fd;
}

Dir& Dir::operator=(Dir&& o) noexcept {
    if (this != &o) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = o.release();
    }
    return *this;
}

Dir Dir::open(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) fail("cannot open directory " + path);
    return Dir(fd);
}

Dir Dir::open_subdir(const std::string& name) const {
    const int fd = ::openat(fd_, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) fail("cannot open directory " + name);
    return Dir(fd);
}

Dir Dir::make_subdir(const std::string& name) const {
    if (::mkdirat(fd_, name.c_str(), 0700) != 0) fail("cannot create directory " + name);
    return open_subdir(name);
}

FileWriter::FileWriter(const Dir& dir, const std::string& name) {
    fd_ = ::openat(dir.fd(), name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd_ < 0) fail("cannot create " + name);
}

FileWriter::~FileWriter() {
    if (fd_ >= 0) ::close(fd_);
}

bool FileWriter::write(const char* data, size_t n) const {
    while (n > 0) {
        const ssize_t w = ::write(fd_, data, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        data += w;
        n -= static_cast<size_t>(w);
    }
    return true;
}

void FileWriter::finish(int mode) {
    if (::fchmod(fd_, static_cast<mode_t>(mode)) != 0) fail("fchmod");
    if (::fsync(fd_) != 0) fail("fsync");
    const int fd = fd_;
    fd_ = -1;
    if (::close(fd) != 0) fail("close");
}

std::string verify_file(const Dir& dir, const std::string& name, uint64_t size, const std::string& sha256_hex) {
    const int fd = ::openat(dir.fd(), name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return name + ": cannot be opened";
    struct stat st{};
    std::string error;
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || static_cast<uint64_t>(st.st_size) != size) {
        error = name + ": not a regular file of the expected size";
    } else {
        Sha256 hash;
        std::vector<char> buf(1 << 20);
        for (ssize_t r; (r = ::read(fd, buf.data(), buf.size())) != 0;) {
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) {
                error = name + ": read error";
                break;
            }
            hash.update(buf.data(), static_cast<size_t>(r));
        }
        const auto digest = hash.finish();
        if (error.empty() && to_hex(digest.data(), digest.size()) != sha256_hex) error = name + ": hash mismatch";
    }
    ::close(fd);
    return error;
}

void set_symlink_atomic(const std::string& target, const std::string& link_path) {
    uint8_t rnd[6];
    random_bytes(rnd, sizeof rnd);
    const std::string tmp = link_path + ".tmp-" + to_hex(rnd, sizeof rnd);
    if (::symlink(target.c_str(), tmp.c_str()) != 0) fail("symlink " + tmp);
    if (::rename(tmp.c_str(), link_path.c_str()) != 0) {
        const int saved = errno;
        ::unlink(tmp.c_str());
        errno = saved;
        fail("rename " + tmp);
    }
}

std::string read_link(const std::string& link_path) {
    char buf[4096];
    const ssize_t n = ::readlink(link_path.c_str(), buf, sizeof buf);
    return n > 0 && static_cast<size_t>(n) < sizeof buf ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

void fsync_dir(const Dir& dir) {
    if (::fsync(dir.fd()) != 0) fail("fsync directory");
}

void rename_noreplace(const std::string& from, const std::string& to) {
#if defined(__APPLE__)
    // renamex_np with RENAME_EXCL: the same atomic no-replace guarantee as renameat2(RENAME_NOREPLACE), EEXIST when the
    // target exists (rename(2) man page, macOS)
    if (::renamex_np(from.c_str(), to.c_str(), RENAME_EXCL) != 0) fail("rename " + from + " to " + to);
#else
    if (::renameat2(AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), RENAME_NOREPLACE) != 0)
        fail("rename " + from + " to " + to);
#endif
}

// The install root belongs to the user alone: created 0700, and refused if somebody else could write to it
void ensure_private_root(const Layout& layout) {
    std::error_code ec;
    if (!fs::exists(layout.root, ec)) {
        fs::create_directories(layout.root, ec);
        if (::chmod(layout.root.c_str(), 0700) != 0) throw FsError("cannot restrict " + layout.root.string());
    }
    struct stat st{};
    if (::lstat(layout.root.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
        throw FsError("install root is not a directory");
    if (st.st_uid != ::geteuid())
        throw FsError("install root " + layout.root.string() + " belongs to another user; choose another folder");
    if ((st.st_mode & 077) != 0)
        // The command on a line of its own: a screen cuts long lines, and a long path must not hide it
        throw FsError("install root is open to others; run:\n  chmod 700 " + layout.root.string());
    for (const fs::path& sub : {layout.versions(), layout.runtime_parent(), layout.staging()}) {
        if (::mkdir(sub.c_str(), 0700) != 0 && errno != EEXIST) throw FsError("cannot create " + sub.string());
    }
}

}  // namespace tona::update
